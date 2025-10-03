import subprocess
import time
import signal
import sys
import argparse
import os
from typing import List, Dict

class VLLMServerManager:
    def __init__(self, max_retries: int = 3, retry_delay: int = 30):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.server_configs: Dict[str, dict] = {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_counts: Dict[str, int] = {}
        self.used_gpus = set()

    def start_server(self, server_id: str, model: str, port: int, host: str = "0.0.0.0", gpu_memory_utilization: float = 0.2, gpu_id: int = None, tensor_parallel_size: int = 1):
        """Start a vLLM server with the specified model and port"""
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--host", host,
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--trust-remote-code"
        ]
        
        # Set up environment for GPU selection
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Starting vLLM server on port {port} with model {model} (GPU {gpu_id})")
        else:
            # Auto-select gpu with the most free memory
            import gpustat
            stats = gpustat.GPUStatCollection.new_query()
            best_gpu = max(stats, key=lambda g: g.memory_free if g.index not in self.used_gpus else 0).index
            self.used_gpus.add(best_gpu)
            print(f"Starting vLLM server on port {port} with model {model} (auto GPU selection: {best_gpu})")
            env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

        # Set cache directory to avoid conflicts
        cache_dir = os.path.expanduser(f"~/.cache/vllm/{server_id}")
        os.makedirs(cache_dir, exist_ok=True)
        env["VLLM_CACHE_ROOT"] = cache_dir

        print(f"Command: {' '.join(cmd)}")
        
        # Store server configuration for retries
        server_config = {
            'model': model,
            'port': port,
            'host': host,
            'gpu_memory_utilization': gpu_memory_utilization,
            'gpu_id': gpu_id,
            'tensor_parallel_size': tensor_parallel_size,
            'cmd': cmd,
            'env': env
        }
        
        self.server_configs[server_id] = server_config
        self.retry_counts[server_id] = 0
        
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                universal_newlines=True
            )
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
            print(f"!!! Server {server_id} has exceeded max retries ({self.max_retries})")
            return False

        self.retry_counts[server_id] += 1

        print(f"\n=> Attempting to restart Server {server_id} (attempt {retry_count + 1}/{self.max_retries})")
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
            config['cmd'],
            env=config['env'],
            universal_newlines=True
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
                    print(f"!!! Server {server_id} will not be restarted (max retries exceeded)")
        
        return any_restarted
    
    def stop_servers(self):
        """Stop all running servers"""
        print("Stopping all servers...")
        for process in self.processes:
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
            print(f"Server {server_id}: http://{config['host']}:{config['port']} (Model: {config['model']}) - {status}")

def main():
    model_db = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
    model_red = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4"
    
    port_db_model = 8000
    port_red_model = 8001

    host = "0.0.0.0"

    gpu_memory_utilization = 0.2

    gpu_id_db = None
    gpu_id_red = None

    tensor_parallel_size_db_model = 1
    tensor_parallel_size_red_model = 1

    manager = VLLMServerManager()
    
    # Signal handler for shutdown
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    # Start both servers
    server_db_model = manager.start_server(
        "db_model",
        model_db, 
        port_db_model, 
        host, 
        gpu_memory_utilization, 
        gpu_id_db,
        tensor_parallel_size_db_model,
    )
    server_red_model = manager.start_server(
        "red_model",
        model_red, 
        port_red_model, 
        host, 
        gpu_memory_utilization, 
        gpu_id_red,
        tensor_parallel_size_red_model,
    )
    
    manager.wait_for_servers()
    manager.check_and_restart_failed_servers()
    
    print("\n" + "="*50)
    manager.print_server_status()
    print("="*50)
    print("\nPress Ctrl+C to stop all servers")
    
    try:
        # Keep the script running and monitor servers
        check_interval = 30
        while True:
            time.sleep(check_interval)
            if manager.check_and_restart_failed_servers():
                print("!!! Some servers were restarted")
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_servers()
    
    return 0

if __name__ == "__main__":
    exit(main())