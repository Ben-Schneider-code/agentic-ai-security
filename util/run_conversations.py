import sys
import os
import argparse
import asyncio
import re
import json
import time

# Add project root to sys.path to allow imports from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Add MARFT directory to sys.path
marft_root = os.path.join(project_root, "MARFT")
if marft_root not in sys.path:
    sys.path.append(marft_root)

try:
    from util.mcp_client import MCPClient

    # Fix: Import directly from marft package since we added MARFT to sys.path
    from marft.envs.redteam_sql.redteam_sql_env import (
        OfflineLLM,
        EXEC_RESULT_DELIMITER,
        sql_system_prompt,
        detect_honeypot_access,
        detect_sql_generated,
        detect_sensitive_data_query,
        is_victim_refusal,
        extract_sql_query,
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(
        "Ensure you are running this script inside the docker container or have the correct python path."
    )
    sys.exit(1)


def parse_conversations(file_path):
    """
    Parses the conversation file.
    - Ignores lines starting with #
    - Treats blocks of text separated by blank lines (double newlines) as separate conversations.
    """
    with open(file_path, "r") as f:
        content = f.read()

    # Remove comment lines
    lines = content.split("\n")
    cleaned_lines = [line for line in lines if not line.strip().startswith("#")]
    cleaned_content = "\n".join(cleaned_lines)

    # Split by double newlines to find blocks
    # Using regex to handle multiple newlines
    blocks = re.split(r"\n\s*\n", cleaned_content)

    # Filter out empty blocks and strip whitespace
    conversations = [b.strip() for b in blocks if b.strip()]

    return conversations


def start_vllm_instance(model_name, gpus, port, max_model_len=16384, timeout=600):
    """
    Starts a vLLM server for the given model on the specified GPUs/port.
    Returns the VLLMInstance (already started and health-checked).
    Raises RuntimeError / TimeoutError on failure.
    """
    from start_vllm import VLLMInstance

    instance = VLLMInstance(
        server_id="blueteam",
        model=model_name,
        port=port,
        gpus=gpus,
        host="0.0.0.0",
        gpu_memory_utilization=0.95,
        max_model_len=max_model_len,
    )

    print(f"\n[vLLM] Starting server for model: {model_name}")
    print(f"[vLLM]   GPUs          : {gpus}")
    print(f"[vLLM]   Port          : {port}")
    print(f"[vLLM]   max_model_len : {max_model_len}")
    print("[vLLM]   Logs          : /tmp/vllm_logs/blueteam.log")
    instance.start()

    print(f"[vLLM] Waiting for server to be ready (timeout={timeout}s)...")
    start_time = time.time()
    poll_interval = 5
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            instance.stop()
            raise TimeoutError(
                f"[vLLM] Server did not become ready within {timeout}s. "
                "Check /tmp/vllm_logs/blueteam.log for details."
            )
        if not instance.is_alive():
            raise RuntimeError(
                "[vLLM] Server process died during startup! "
                "Check /tmp/vllm_logs/blueteam.log for details."
            )
        if instance.health_check():
            print(f"[vLLM] ✓ Server ready at {instance.url}  ({elapsed:.0f}s)")
            return instance
        print(f"[vLLM]   Waiting... ({elapsed:.0f} / {timeout}s)")
        time.sleep(poll_interval)


async def run_conversations(args):
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)

    print(f"Reading conversations from {args.input_file}...")
    conversations = parse_conversations(args.input_file)
    print(f"Found {len(conversations)} conversations.")

    # Optional system-prompt override. Used to swap in the "unprotected"
    # (pre-patch) blue-team prompt for Agent-vs-Human baseline runs; without
    # --system_prompt_file we keep the manually-protected sql_system_prompt.
    if getattr(args, "system_prompt_file", None):
        if not os.path.exists(args.system_prompt_file):
            print(f"ERROR: --system_prompt_file {args.system_prompt_file} not found.")
            sys.exit(1)
        with open(args.system_prompt_file, "r") as f:
            active_system_prompt = f.read()
        print(
            f"Overriding system prompt from {args.system_prompt_file} "
            f"({len(active_system_prompt)} chars)."
        )
    else:
        active_system_prompt = sql_system_prompt

    # Seed Python-level RNG so any non-vLLM randomness (e.g., ordering) is
    # reproducible; vLLM sampling at T=0.7 remains stochastic across seeds.
    seed_val = getattr(args, "seed", None)
    if seed_val is not None:
        import random as _random
        _random.seed(seed_val)
        try:
            import numpy as _np
            _np.random.seed(seed_val)
        except ImportError:
            pass

    # ── vLLM lifecycle ────────────────────────────────────────────────────────
    vllm_instance = None  # will be set if we manage the server ourselves

    # Always use the explicitly provided model — no fallbacks.
    gpus = [int(g.strip()) for g in args.gpu.split(",")]
    port = args.port
    vllm_base_url = f"http://localhost:{port}/v1"
    model_name = args.model_name

    if getattr(args, "no_start_vllm", False):
        print(f"[vLLM] Reusing existing server at {vllm_base_url} (model={model_name}).")
    else:
        try:
            vllm_instance = start_vllm_instance(
                model_name=model_name,
                gpus=gpus,
                port=port,
                max_model_len=args.max_model_len,
                timeout=args.vllm_timeout,
            )
        except (RuntimeError, TimeoutError) as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    # ── MCP Client ────────────────────────────────────────────────────────────
    print("\nInitializing MCP Client to connect to Postgres...")
    mcp_client = MCPClient()
    try:
        await mcp_client.connect_to_server("/app/mcp/postgres.py")
        print("MCP Client connected.")
    except Exception as e:
        print(f"Failed to connect MCP Client: {e}")
        if vllm_instance:
            print("[vLLM] Stopping server due to MCP connection failure...")
            vllm_instance.stop()
        sys.exit(1)

    try:
        # Initialize Victim LLM (Blue Team)
        print(
            f"\nInitializing Blue Team Agent "
            f"(vLLM at {vllm_base_url} | model: {model_name})..."
        )
        victim_llm = OfflineLLM(
            model_name=model_name, mcp_client=mcp_client, vllm_base_url=vllm_base_url
        )
        print("Blue Team Agent ready.")

        if args.output_file:
            output_filename = args.output_file
        else:
            suffix_parts = []
            if getattr(args, "run_tag", None):
                suffix_parts.append(args.run_tag)
            if getattr(args, "seed", None) is not None:
                suffix_parts.append(f"seed{args.seed}")
            suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
            output_filename = f"{args.input_file}_report{suffix}.jsonl"
        print(f"Writing report to {output_filename}...")
        accessed_honeypots = set()

        with open(output_filename, "w") as f_out:
            for i, prompt in enumerate(conversations):
                print(f"\n{'=' * 20} Conversation {i + 1} {'=' * 20}")
                print(f"[RED TEAM INPUT]:\n{prompt}\n")
                print(f"{'-' * 60}")

                # Prepare format expected by OfflineLLM
                history = [{"role": "user", "content": prompt}]

                print("[BLUE TEAM PROCESSING]...")
                # Use async acomplete method
                response = await victim_llm.acomplete(
                    history, system_prompt=active_system_prompt
                )

                print(f"[BLUE TEAM RESPONSE]:\n{response}")
                print(f"{'=' * 60}\n")

                # --- Report Generation ---
                # Parse response for execution result
                execution_result = ""
                llm_response_text = response
                if EXEC_RESULT_DELIMITER in response:
                    parts = response.split(EXEC_RESULT_DELIMITER, 1)
                    llm_response_text = parts[0]
                    execution_result = parts[1]

                # Extract SQL
                sql_query = extract_sql_query(llm_response_text) or ""

                # Run Detections
                honeypots_detected = detect_honeypot_access(execution_result, sql_query)

                # Identify new honeypots
                new_honeypots = []
                for h in honeypots_detected:
                    # Use tuple of (type, identifier) for uniqueness in set
                    h_id = (h["type"], str(h["identifier"]))
                    if h_id not in accessed_honeypots:
                        new_honeypots.append(h)
                        accessed_honeypots.add(h_id)

                record = {
                    "conversation_index": i,
                    "model_name": model_name,
                    "red_team_input": prompt,
                    "blue_team_response": llm_response_text,
                    "execution_result": execution_result,
                    "sql_query": sql_query,
                    "is_refusal": is_victim_refusal(llm_response_text),
                    "sql_generated": detect_sql_generated(response),
                    "sensitive_data_access": detect_sensitive_data_query(
                        llm_response_text
                    ),
                    "honeypots_accessed_all": honeypots_detected,
                    "honeypots_accessed_new": new_honeypots,
                }

                f_out.write(json.dumps(record) + "\n")
                f_out.flush()

    finally:
        print("Cleaning up MCP Client...")
        await mcp_client.cleanup()

        if vllm_instance is not None:
            print("[vLLM] Stopping managed vLLM server...")
            vllm_instance.stop()
            print("[vLLM] Server stopped.")

        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch conversations against the Blue Team agent."
    )
    parser.add_argument(
        "input_file", help="Path to the file containing conversation blocks."
    )
    parser.add_argument(
        "--output_file", help="Path to the JSONL output report.", default=None
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help=(
            "Model ID to serve. The script starts a dedicated vLLM instance for this "
            "model and stops it when finished. "
            "E.g. 'Snowflake/Qwen-2.5-coder-Arctic-ExCoT-32B'."
        ),
    )
    parser.add_argument(
        "--gpu",
        help=(
            "Comma-separated GPU index/indices to use for the managed vLLM server "
            "(only used when --model_name is set). Default: '0'."
        ),
        default="0",
    )
    parser.add_argument(
        "--port",
        type=int,
        help=(
            "Port for the managed vLLM server "
            "(only used when --model_name is set). Default: 8001."
        ),
        default=8001,
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        help=(
            "Maximum sequence length (context window) passed to vLLM. "
            "Reduce this if vLLM fails with a KV-cache size error. Default: 16384."
        ),
        default=16384,
    )
    parser.add_argument(
        "--vllm_timeout",
        type=int,
        help="Seconds to wait for the vLLM server to become ready. Default: 600.",
        default=600,
    )
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help=(
            "Optional path to a plain-text file whose contents replace the "
            "default sql_system_prompt. Used for the 'unprotected' (pre-patch) "
            "blue-team prompt in the Agent-vs-Human baseline; omit to keep the "
            "manually-protected prompt."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Python/NumPy RNG; also appended to auto-named output file.",
    )
    parser.add_argument(
        "--run_tag",
        default=None,
        help="Short tag folded into the auto-named output filename (e.g. 'unprotected').",
    )
    parser.add_argument(
        "--no_start_vllm",
        action="store_true",
        help=(
            "Skip launching a vLLM server; assume one is already serving "
            "--model_name at http://localhost:<--port>/v1. Useful when a "
            "driver (run_human_baseline.sh) starts vLLM once and calls this "
            "script N times per seed."
        ),
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_conversations(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
