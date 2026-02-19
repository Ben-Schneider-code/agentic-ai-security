import sys
import os
import argparse
import asyncio
import re
import json

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


async def run_conversations(args):
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)

    print(f"Reading conversations from {args.input_file}...")
    conversations = parse_conversations(args.input_file)
    print(f"Found {len(conversations)} conversations.")

    # Initialize MCP Client
    print("\nInitializing MCP Client to connect to Postgres...")
    mcp_client = MCPClient()
    try:
        await mcp_client.connect_to_server("/app/mcp/postgres.py")
        print("MCP Client connected.")
    except Exception as e:
        print(f"Failed to connect MCP Client: {e}")
        sys.exit(1)

    # Determine model and URL
    # Default values requested by user
    vllm_base_url = "http://localhost:8001/v1"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        # Try to find the student model in the registry
        from start_vllm import read_registry

        print("Checking vLLM registry for student model...")
        registry = read_registry()

        target_entry = None
        # Priority: 'student' -> 'policy' -> search by model name
        if "student" in registry:
            target_entry = registry["student"]
            print("Using 'student' model from registry.")
        elif "policy" in registry:
            target_entry = registry["policy"]
            print("Using 'policy' model from registry as student.")
        else:
            # Fallback: search for the specific model name
            for key, entry in registry.items():
                if entry.get("model") == "meta-llama/Llama-3.1-8B-Instruct":
                    target_entry = entry
                    print(f"Found requested model under key '{key}' in registry.")
                    break

        if target_entry:
            # Registry URLs typically don't include /v1
            base_url = target_entry["url"]
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            vllm_base_url = base_url
            model_name = target_entry["model"]

    except Exception as e:
        print(f"Registry lookup not fully successful (using defaults): {e}")

    try:
        # Initialize Victim LLM (Blue Team)
        # This connects to vLLM (default port 8000) and uses mcp_client for SQL execution
        print(
            f"Initializing Blue Team Agent (connecting to vLLM at {vllm_base_url} for model {model_name})..."
        )
        victim_llm = OfflineLLM(
            model_name=model_name, mcp_client=mcp_client, vllm_base_url=vllm_base_url
        )
        print("Blue Team Agent ready.")

        output_filename = (
            args.output_file if args.output_file else f"{args.input_file}_report.jsonl"
        )
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
                    history, system_prompt=sql_system_prompt
                )

                print(f"[BLUE TEAM RESPONSE]:\n{response}")
                print(f"{'=' * 60}\n")

                # --- Report Generation ---
                # Parse response for execution result
                execution_result = ""
                llm_response_text = response
                if "\nExecution Result: " in response:
                    parts = response.rsplit("\nExecution Result: ", 1)
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
    args = parser.parse_args()

    try:
        asyncio.run(run_conversations(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
