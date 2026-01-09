import subprocess
import time
import signal
import sys
import argparse
import os
from typing import List, Dict, Optional


class VLLMServerManager:
    def __init__(self, max_retries: int = 3, retry_delay: int = 30):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.server_configs: Dict[str, dict] = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_counts: Dict[str, int] = {}
        self.used_gpus = set()

    def start_server(
        self,
        server_id: str,
        model: str,
        port: int,
        host: str = "0.0.0.0",
        gpu_memory_utilization: float = 0.95,
        gpu_id: int = None,
        tensor_parallel_size: int = 1,
    ):
        """Start a vLLM server with the specified model and port"""
        # Check if the model needs a fallback chat template
        template_file = None
        try:
            from transformers import AutoTokenizer, AutoConfig

            # Load tokenizer and config
            # We trust remote code for accuracy in detection
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

            if not tokenizer.chat_template:
                print(
                    f"Model {model} has no default chat template. Attempting to detect suitable fallback..."
                )

                try:
                    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
                    model_type = getattr(config, "model_type", "").lower()
                    vocab_size = getattr(config, "vocab_size", 0)

                    print(
                        f"Detected model type: {model_type}, vocab size: {vocab_size}"
                    )

                    if model_type == "llama":
                        # Llama 3 usually has vocab size ~128k (128256), Llama 2 is 32k
                        if vocab_size >= 128000:
                            template_file = "/app/util/llama3.jinja"
                            print("Selected Llama 3 template.")
                        else:
                            template_file = "/app/util/llama2.jinja"
                            print("Selected Llama 2 template.")

                    elif model_type in ["qwen", "qwen2"]:
                        template_file = "/app/util/chatml.jinja"
                        print("Selected ChatML template (Qwen/Yi style).")

                    elif model_type == "mistral":
                        template_file = "/app/util/llama2.jinja"
                        print("Selected Llama 2/Mistral template.")

                    else:
                        template_file = "/app/util/chatml.jinja"
                        print("Unknown model type, defaulting to ChatML template.")

                except Exception as e:
                    print(
                        f"Error detecting model configuration: {e}. Defaulting to ChatML."
                    )
                    template_file = "/app/util/chatml.jinja"

        except Exception as e:
            print(
                f"Warning: Could not check tokenizer for chat template ({e}). Assuming default is fine or vLLM will handle."
            )

        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(port),
            "--host",
            host,
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--trust-remote-code",
            "--disable-log-requests",
            "--enforce-eager",  # Disables CUDA graph capture (faster startup)
            # "--max-model-len",
            # "8192",  # Limit context to save memory/startup time
        ]

        if template_file:
            print(f"Applying fallback chat template: {template_file}")
            cmd.extend(["--chat-template", template_file])

        # Set up environment for GPU selection
        env = os.environ.copy()

        if "HF_TOKEN" in os.environ:
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]

        if "HF_TOKEN" in os.environ:
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]

        # Check available GPU count
        import torch

        available_gpus = torch.cuda.device_count()

        if gpu_id is not None:
            # Specific GPU requested
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(
                f"Starting vLLM server on port {port} with model {model} (GPU {gpu_id})"
            )

        elif tensor_parallel_size > 1:
            # TP > 1: Let vLLM use multiple GPUs.
            # We don't restrict CUDA_VISIBLE_DEVICES unless we want to partition (e.g. use GPUs 0,1 for one model and 2,3 for another).
            # For simplicity here, if TP > 1, we assume we use the first N GPUs or all visible ones.
            # If we are in a container with --gpus '"device=1,2"', these will be seen as 0 and 1 inside.
            print(
                f"Starting vLLM server on port {port} with model {model} (TP={tensor_parallel_size}, using {available_gpus} available GPUs)"
            )
            if available_gpus < tensor_parallel_size:
                print(
                    f"WARNING: TP={tensor_parallel_size} requested but only {available_gpus} GPUs visible!"
                )

        else:
            # TP = 1 and no specific GPU requested: Auto-select best single GPU
            # Auto-select gpu with the most free memory
            try:
                import gpustat

                stats = gpustat.GPUStatCollection.new_query()
                # Filter out used GPUs
                candidates = [g for g in stats if g.index not in self.used_gpus]
                if not candidates:
                    # Fallback if all strictly tracked are used (or none found), try just picking next index
                    candidates = stats

                best_gpu = max(candidates, key=lambda g: g.memory_free).index
                self.used_gpus.add(best_gpu)
                env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                print(
                    f"Starting vLLM server on port {port} with model {model} (auto GPU selection: {best_gpu})"
                )
            except Exception as e:
                print(f"Warning: GPU auto-selection failed ({e}), defaulting to GPU 0")
                env["CUDA_VISIBLE_DEVICES"] = "0"

        # Set cache directory to avoid conflicts
        cache_dir = os.path.expanduser(f"~/.cache/vllm/{server_id}")
        os.makedirs(cache_dir, exist_ok=True)
        env["VLLM_CACHE_ROOT"] = cache_dir

        print(f"Command: {' '.join(cmd)}")

        # Store server configuration for retries
        server_config = {
            "model": model,
            "port": port,
            "host": host,
            "gpu_memory_utilization": gpu_memory_utilization,
            "gpu_id": gpu_id,
            "tensor_parallel_size": tensor_parallel_size,
            "cmd": cmd,
            "env": env,
        }

        self.server_configs[server_id] = server_config
        self.retry_counts[server_id] = 0

        try:
            process = subprocess.Popen(cmd, env=env, universal_newlines=True)
            self.processes[server_id] = process
            print(f"Started process PID: {process.pid}\n")
            return process
        except Exception as e:
            print(f"Error starting server on port {port}: {e}")
            return None

    def wait_for_servers(self, timeout: int = 600):
        # Give servers time to initialize
        time.sleep(10)

        # Check if processes are still running
        for server_id, process in self.processes.items():
            poll_result = process.poll()
            if poll_result is not None:
                print(f"!!! Server {server_id} exited with code {poll_result}")
                # Try to get any error output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stderr:
                        print(f"Server {server_id} stderr: {stderr}")
                except:
                    pass

    def restart_server(self, server_id: str) -> bool:
        """Restart a specific server"""
        config = self.server_configs[server_id]
        retry_count = self.retry_counts[server_id]

        if retry_count >= self.max_retries:
            print(
                f"!!! Server {server_id} has exceeded max retries ({self.max_retries})"
            )
            return False

        self.retry_counts[server_id] += 1

        print(
            f"\n=> Attempting to restart Server {server_id} (attempt {retry_count + 1}/{self.max_retries})"
        )
        print(f"Waiting {self.retry_delay} seconds before restart...")
        time.sleep(self.retry_delay)

        # Clean up the old process if it exists
        if server_id in self.processes and self.processes[server_id]:
            try:
                self.processes[server_id].terminate()
                self.processes[server_id].wait(timeout=5)
            except:
                pass

        # Start the server again
        process = subprocess.Popen(
            config["cmd"], env=config["env"], universal_newlines=True
        )

        self.processes[server_id] = process

        print(f"Server {server_id} restarted with PID: {process.pid}")
        return True

    def check_and_restart_failed_servers(self) -> bool:
        """Check all servers and restart any that have failed"""
        any_restarted = False

        for server_id, process in self.processes.items():
            if process and process.poll() is not None:
                exit_code = process.poll()
                print(f"!!! Server {server_id} has stopped (exit code: {exit_code})")

                if self.restart_server(server_id):
                    any_restarted = True
                    # Give the server some time to start
                    time.sleep(5)
                else:
                    print(
                        f"!!! Server {server_id} will not be restarted (max retries exceeded)"
                    )

        return any_restarted

    def stop_servers(self):
        """Stop all running servers"""
        print("Stopping all servers...")
        for process in self.processes.values():
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.processes.clear()

    def signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("\nReceived interrupt signal. Shutting down servers...")
        self.stop_servers()
        sys.exit(0)

    def print_server_status(self):
        print("\nCurrent server statuses:")
        for server_id, process in self.processes.items():
            status = "Running" if process and process.poll() is None else "Stopped"
            config = self.server_configs[server_id]
            print(
                f"Server {server_id}: http://{config['host']}:{config['port']} (Model: {config['model']}) - {status}"
            )

    def monitor_servers(self, check_interval: int = 30):
        """Monitor servers and restart if needed"""
        print("\n" + "=" * 50)
        self.print_server_status()
        print("=" * 50)
        print("\nPress Ctrl+C to stop all servers")

        try:
            while True:
                time.sleep(check_interval)
                if self.check_and_restart_failed_servers():
                    print("!!! Some servers were restarted")
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_servers()


def setup_model_server(
    server_id: str,
    model: str,
    port: int,
    host: str = "0.0.0.0",
    gpu_memory_utilization: float = 0.95,
    gpu_id: Optional[int] = None,
    tensor_parallel_size: int = 1,
    manager: Optional[VLLMServerManager] = None,
) -> VLLMServerManager:
    """
    Set up a vLLM model server with the specified configuration.

    Args:
        server_id: Unique identifier for this server
        model: HuggingFace model name/path
        port: Port to run the server on
        host: Host address (default: 0.0.0.0)
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        gpu_id: Specific GPU ID to use (None for auto-selection)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        manager: Existing VLLMServerManager to use (creates new if None)

    Returns:
        VLLMServerManager instance with the server started
    """
    if manager is None:
        manager = VLLMServerManager()

    manager.start_server(
        server_id=server_id,
        model=model,
        port=port,
        host=host,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_id=gpu_id,
        tensor_parallel_size=tensor_parallel_size,
    )

    return manager


def main():
    """Main entry point - currently only runs DB agent model"""
    manager = VLLMServerManager()

    # Signal handler for shutdown
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)

    # ------------------------------------------------------------------
    # Model Selection Logic
    # ------------------------------------------------------------------
    import json
    import constants

    # 1. Determine Runtime Model
    runtime_model, source = constants.get_runtime_model_id()

    # Extract baked ID for logging if possible (reverse look up not strictly needed but good for logs)
    # The constants function returns the path as 'runtime_model' if it's baked in.
    # We can rely on the 'source' description to tell us what happened.

    # 2. Echo Status
    print("\n" + "=" * 50)
    print("MODEL CONFIGURATION LOG")
    print(f"  * Selected Model:  {runtime_model}")
    print(f"  * Selection Source: {source}")
    print("=" * 50 + "\n")

    # Setup only DB agent model for now
    # Determine TP size based on available hardware and model needs
    import torch

    available_gpus = torch.cuda.device_count()

    # Logic: If we have multiple GPUs, use them (TP=N).
    # Especially important for 32B models which might not fit on one.
    tp_size = available_gpus if available_gpus > 0 else 1

    # Setup only DB agent model for now
    setup_model_server(
        server_id="db_model",
        model=runtime_model,
        port=8000,
        manager=manager,
        # Remove hardcoded gpu_id=1 so it can use all if TP > 1
        gpu_id=None,
        tensor_parallel_size=tp_size,
    )

    # Optionally setup red team model too:
    # setup_model_server(
    #     server_id="red_model",
    #     model="meta-llama/Meta-Llama-3-8B-Instruct",
    #     port=8001,
    #     manager=manager
    # )

    manager.wait_for_servers()
    manager.check_and_restart_failed_servers()

    # Monitor servers
    manager.monitor_servers(check_interval=30)

    return 0


if __name__ == "__main__":
    exit(main())
