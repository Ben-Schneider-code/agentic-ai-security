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
        gpu_memory_utilization: float = 0.2,
        gpu_id: int = None,
        tensor_parallel_size: int = 1,
    ):
        """Start a vLLM server with the specified model and port"""
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
        ]

        # Set up environment for GPU selection
        env = os.environ.copy()

        if "HF_TOKEN" in os.environ:
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]

        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(
                f"Starting vLLM server on port {port} with model {model} (GPU {gpu_id})"
            )
        else:
            # Auto-select gpu with the most free memory
            import gpustat

            stats = gpustat.GPUStatCollection.new_query()
            best_gpu = max(
                stats,
                key=lambda g: g.memory_free if g.index not in self.used_gpus else 0,
            ).index
            self.used_gpus.add(best_gpu)
            print(
                f"Starting vLLM server on port {port} with model {model} (auto GPU selection: {best_gpu})"
            )
            env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

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

    def wait_for_servers(self, timeout: int = 60):
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
    gpu_memory_utilization: float = 0.45,
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

    # 1. Inspect Baked-in Model
    baked_info_path = constants.BAKED_MODEL_INFO_PATH
    baked_model_id = "None"
    baked_model_path = None

    if os.path.exists(baked_info_path):
        try:
            with open(baked_info_path, "r") as f:
                info = json.load(f)
                baked_model_id = info.get("model_id", "Unknown")
                baked_model_path = info.get("local_dir")
        except Exception as e:
            print(f"Warning: Failed to read baked model info: {e}")

    # 2. Determine Runtime Model
    # Default fallback
    runtime_model = constants.DEFAULT_MODEL_ID
    source = "Hardcoded Default"

    # Check if user wants to override via env var
    env_model_id = os.environ.get("RUNTIME_MODEL_ID") or os.environ.get("MODEL_ID")

    if env_model_id:
        runtime_model = env_model_id
        source = "Environment Variable (RUNTIME_MODEL_ID/MODEL_ID)"
    elif baked_model_path and os.path.exists(baked_model_path):
        runtime_model = baked_model_path
        source = f"Baked-in Weights ({baked_model_id})"

    # 3. Echo Status
    print("\n" + "=" * 50)
    print("MODEL CONFIGURATION LOG")
    print(f"  * Baked-in Model:  {baked_model_id}")
    print(f"  * Selected Model:  {runtime_model}")
    print(f"  * Selection Source: {source}")
    print("=" * 50 + "\n")

    # Setup only DB agent model for now
    setup_model_server(
        server_id="db_model", model=runtime_model, port=8000, manager=manager, gpu_id=1
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
