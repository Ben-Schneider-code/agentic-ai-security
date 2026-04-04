#!/usr/bin/env python3
"""
Config-driven multi-model vLLM server launcher.

Usage:
    python start_vllm.py --config experiments/my_experiment.json
    python start_vllm.py --config experiments/my_experiment.json --wait-only  # just block until ctrl+c

Config format (JSON):
{
    "servers": [
        {
            "id": "policy",
            "model": "meta-llama/Llama-3.1-70B-Instruct",
            "gpus": [0, 1, 2, 3],
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.92
        },
        {
            "id": "reward",
            "model": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
            "gpus": [4],
            "gpu_memory_utilization": 0.90
        },
        {
            "id": "reference",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "gpus": [5],
            "max_model_len": 4096
        }
    ],
    "base_port": 8100,
    "host": "0.0.0.0"
}

Outputs a registry file at $VLLM_REGISTRY (default: /tmp/vllm_registry.json):
{
    "policy":    {"url": "http://0.0.0.0:8100", "model": "meta-llama/Llama-3.1-70B-Instruct", "gpus": [0,1,2,3]},
    "reward":    {"url": "http://0.0.0.0:8101", "model": "Skywork/...", "gpus": [4]},
    "reference": {"url": "http://0.0.0.0:8102", "model": "meta-llama/Llama-3.1-8B-Instruct", "gpus": [5]}
}
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────────────
# Chat template detection (carried over from your team's script)
# ──────────────────────────────────────────────
TEMPLATE_DIR = os.environ.get("CHAT_TEMPLATE_DIR", "/app/util")


def detect_chat_template(model: str) -> Optional[str]:
    """Return path to a fallback Jinja chat template if the model lacks one."""
    try:
        from transformers import AutoTokenizer, AutoConfig

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if tokenizer.chat_template:
            return None  # model already has one

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        model_type = getattr(config, "model_type", "").lower()
        vocab_size = getattr(config, "vocab_size", 0)

        templates = {
            ("llama", True): f"{TEMPLATE_DIR}/llama3.jinja",  # vocab >= 128k
            ("llama", False): f"{TEMPLATE_DIR}/llama2.jinja",
            "mistral": f"{TEMPLATE_DIR}/llama2.jinja",
            "qwen": f"{TEMPLATE_DIR}/chatml.jinja",
            "qwen2": f"{TEMPLATE_DIR}/chatml.jinja",
        }

        if model_type == "llama":
            key = ("llama", vocab_size >= 128000)
        else:
            key = model_type

        template = templates.get(key, f"{TEMPLATE_DIR}/chatml.jinja")
        if os.path.exists(template):
            print(f"  [chat-template] {model_type} → {template}")
            return template
        else:
            print(f"  [chat-template] Template file not found: {template}, skipping")
            return None

    except Exception as e:
        print(f"  [chat-template] Detection failed ({e}), skipping")
        return None


# ──────────────────────────────────────────────
# Server lifecycle
# ──────────────────────────────────────────────
class VLLMInstance:
    def __init__(
        self,
        server_id: str,
        model: str,
        port: int,
        gpus: List[int],
        host: str = "0.0.0.0",
        gpu_memory_utilization: float = 0.95,
        max_model_len: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
    ):
        self.server_id = server_id
        self.model = model
        self.port = port
        self.gpus = gpus
        self.host = host
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.extra_args = extra_args or []
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None
        self.log_file_path = None
        self.tp_size = len(gpus)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_cmd(self) -> List[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tp_size),
            "--trust-remote-code",
            "--disable-log-requests",
            "--enforce-eager",
        ]
        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.tp_size > 1:
            cmd.append("--disable-custom-all-reduce")

        # Detect and apply chat template if needed
        template = detect_chat_template(self.model)
        if template:
            cmd.extend(["--chat-template", template])

        cmd.extend(self.extra_args)
        return cmd

    def build_env(self) -> dict:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus)
        if self.tp_size > 1:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # Isolate cache per instance
        cache_dir = os.path.expanduser(f"~/.cache/vllm/{self.server_id}")
        os.makedirs(cache_dir, exist_ok=True)
        env["VLLM_CACHE_ROOT"] = cache_dir
        return env

    def start(self):
        cmd = self.build_cmd()
        env = self.build_env()
        cuda_vis = env.get("CUDA_VISIBLE_DEVICES", "?")
        print(
            f"[{self.server_id}] Starting on port {self.port}, "
            f"GPUs={self.gpus} (CUDA_VISIBLE_DEVICES={cuda_vis}), "
            f"TP={self.tp_size}"
        )
        print(f"[{self.server_id}] {' '.join(cmd)}")

        # Ensure log dir exists
        log_dir = Path("/tmp/vllm_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = log_dir / f"{self.server_id}.log"
        print(f"[{self.server_id}] Logging output to {self.log_file_path}")
        self.log_file = open(self.log_file_path, "w")

        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout file
            universal_newlines=True,
        )
        print(f"[{self.server_id}] PID={self.process.pid}")

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def health_check(self) -> bool:
        """Check if the /v1/models endpoint is ready (same check as OfflineLLM)."""
        try:
            req = urllib.request.urlopen(f"{self.url}/v1/models", timeout=5)
            if req.status == 200:
                data = json.loads(req.read())
                # Verify models endpoint returns valid data
                return "data" in data and len(data["data"]) > 0
            return False
        except Exception:
            return False

    def stop(self):
        if self.process:
            print(f"[{self.server_id}] Terminating PID={self.process.pid} (SIGTERM, 60s timeout)")
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # SIGKILL on a process with active CUDA contexts bricks the GPU
                # (100% utilization, 0% RAM, no process visible, requires server restart).
                # Send SIGTERM again and give it more time before resorting to SIGKILL.
                print(f"[{self.server_id}] Still alive after 60s, sending SIGTERM again...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(
                        f"[{self.server_id}] WARNING: Sending SIGKILL — this may brick "
                        f"GPU(s) {self.gpus}. Run 'nvidia-smi --gpu-reset -i {self.gpus[0]}' "
                        f"if the GPU becomes unresponsive."
                    )
                    self.process.kill()
                    self.process.wait()
            self.process = None


class ServerFleet:
    def __init__(self, registry_path: Optional[str] = None):
        self.instances: Dict[str, VLLMInstance] = {}
        self._placeholder_entries: Dict[str, dict] = {}
        self.registry_path = registry_path or os.environ.get(
            "VLLM_REGISTRY", "/tmp/vllm_registry.json"
        )

    def add(self, instance: VLLMInstance):
        self.instances[instance.server_id] = instance

    def validate_gpu_assignments(self):
        """Check for GPU conflicts BEFORE starting anything."""
        all_gpus = []
        for inst in self.instances.values():
            for g in inst.gpus:
                if g in all_gpus:
                    raise ValueError(
                        f"GPU {g} is assigned to multiple servers! "
                        f"Check your config — server '{inst.server_id}' conflicts."
                    )
                all_gpus.append(g)
        print(
            f"GPU assignments validated: {dict((i.server_id, i.gpus) for i in self.instances.values())}"
        )

    def validate_ports(self):
        """Check for port conflicts."""
        ports = {}
        for inst in self.instances.values():
            if inst.port in ports:
                raise ValueError(
                    f"Port {inst.port} assigned to both '{ports[inst.port]}' and '{inst.server_id}'!"
                )
            ports[inst.port] = inst.server_id
        print(
            f"Port assignments validated: {dict((i.server_id, i.port) for i in self.instances.values())}"
        )

    def start_all(self):
        self.validate_gpu_assignments()
        self.validate_ports()
        print(f"\nStarting {len(self.instances)} vLLM server(s)...\n")
        for inst in self.instances.values():
            inst.start()
            time.sleep(2)  # small stagger to avoid I/O contention on model download

    def wait_until_ready(self, timeout: int = 600, poll_interval: int = 5):
        """Block until all servers respond to /health or timeout."""
        print(f"\nWaiting for all servers to be ready (timeout={timeout}s)...")
        ready = {sid: False for sid in self.instances}
        start_time = time.time()

        while not all(ready.values()):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                not_ready = [sid for sid, ok in ready.items() if not ok]
                raise TimeoutError(
                    f"Servers not ready after {timeout}s: {not_ready}. "
                    f"Check logs — a model may be too large for assigned GPUs."
                )

            for sid, inst in self.instances.items():
                if ready[sid]:
                    continue
                if not inst.is_alive():
                    # Dump logs to help debug
                    print(f"\n========= CRASH LOGS: {sid} =========\n")
                    try:
                        if inst.log_file_path is None:
                            print("(no log file available)")
                            raise FileNotFoundError("log_file_path is None")
                        with open(inst.log_file_path, "r") as f:
                            print(f.read())
                    except Exception as e:
                        print(f"Could not read log file: {e}")
                    print("\n=====================================\n")

                    raise RuntimeError(
                        f"Server '{sid}' process died during startup! "
                        f"Model: {inst.model}, GPUs: {inst.gpus}"
                    )
                if inst.health_check():
                    ready[sid] = True
                    print(f"  ✓ [{sid}] ready at {inst.url}  ({elapsed:.0f}s)")

            if not all(ready.values()):
                time.sleep(poll_interval)

        print(f"\nAll {len(self.instances)} servers ready.\n")

    def write_registry(self):
        """Write a JSON file that the training script reads to discover endpoints."""
        registry = {}
        for sid, inst in self.instances.items():
            # Use 127.0.0.1 for client connections when server binds to 0.0.0.0
            client_host = "127.0.0.1" if inst.host == "0.0.0.0" else inst.host
            registry[sid] = {
                "url": f"http://{client_host}:{inst.port}",
                "model": inst.model,
                "gpus": inst.gpus,
                "port": inst.port,
            }
        # Merge in placeholder entries (e.g. blueteam student alias)
        for sid, entry in getattr(self, "_placeholder_entries", {}).items():
            if sid not in registry:
                registry[sid] = entry
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"Registry written to {self.registry_path}")
        print(json.dumps(registry, indent=2))

    def stop_all(self):
        print("\nStopping all servers...")
        for inst in self.instances.values():
            inst.stop()
        # Clean up registry
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)
        print("All servers stopped.")

    def monitor(self, check_interval: int = 30, max_restarts: int = 3):
        """Block forever, restarting crashed servers."""
        restart_counts = {sid: 0 for sid in self.instances}
        print("\nMonitoring servers (Ctrl+C to stop)...\n")
        try:
            while True:
                time.sleep(check_interval)
                for sid, inst in self.instances.items():
                    if not inst.is_alive():
                        if restart_counts[sid] >= max_restarts:
                            print(
                                f"[{sid}] FATAL: exceeded {max_restarts} restarts, giving up"
                            )
                            continue
                        restart_counts[sid] += 1
                        print(
                            f"[{sid}] Crashed — restarting ({restart_counts[sid]}/{max_restarts})..."
                        )
                        inst.start()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()


# ──────────────────────────────────────────────
# Registry helpers (importable by other scripts)
# ──────────────────────────────────────────────
DEFAULT_REGISTRY_PATH = os.environ.get("VLLM_REGISTRY", "/tmp/vllm_registry.json")


def read_registry(registry_path: Optional[str] = None) -> dict:
    """Read the vLLM server registry written by start_vllm.py.

    Returns a dict like:
        {"server_id": {"url": "http://...", "model": "...", "gpus": [...], "port": N}, ...}
    """
    path = registry_path or DEFAULT_REGISTRY_PATH
    with open(path) as f:
        return json.load(f)


def wait_for_registry(
    registry_path: Optional[str] = None, timeout: int = 660, poll_interval: int = 5
) -> dict:
    """Block until the registry file exists and is non-empty, then read it."""
    path = registry_path or DEFAULT_REGISTRY_PATH
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path):
            try:
                reg = read_registry(path)
                if reg:
                    return reg
            except (json.JSONDecodeError, IOError):
                pass
        time.sleep(poll_interval)
    raise TimeoutError(f"Registry not found at {path} after {timeout}s")


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def fleet_from_config(config: dict) -> ServerFleet:
    base_port = config.get("base_port", 8100)
    host = config.get("host", "0.0.0.0")
    registry_path = config.get("registry_path", None)

    fleet = ServerFleet(registry_path=registry_path)

    for i, srv in enumerate(config["servers"]):
        # Skip placeholder entries (not real servers to launch)
        if srv.get("_skip") or "_note" in srv:
            continue
        port = srv.get("port", base_port + i)  # auto-increment if not specified
        inst = VLLMInstance(
            server_id=srv["id"],
            model=srv["model"],
            port=port,
            gpus=srv["gpus"],
            host=host,
            gpu_memory_utilization=srv.get("gpu_memory_utilization", 0.95),
            max_model_len=srv.get("max_model_len"),
            extra_args=srv.get("extra_args", []),
        )
        fleet.add(inst)

    # Inject placeholder entries (with _note) directly into the registry
    # so that scripts reading the registry can resolve all server IDs.
    fleet._placeholder_entries = {
        srv["id"]: {
            "url": f"http://127.0.0.1:{srv['port']}",
            "model": srv["model"],
            "gpus": srv["gpus"],
            "port": srv["port"],
        }
        for srv in config["servers"]
        if "_note" in srv
    }

    return fleet


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server fleet")
    parser.add_argument(
        "--config", required=True, help="Path to experiment JSON config"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Startup timeout in seconds"
    )
    parser.add_argument(
        "--wait-only",
        action="store_true",
        help="If set, just block (don't monitor/restart)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    fleet = fleet_from_config(config)

    # Graceful shutdown on signals
    def _shutdown(signum, frame):
        fleet.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    fleet.start_all()
    fleet.wait_until_ready(timeout=args.timeout)
    fleet.write_registry()

    if args.wait_only:
        print("Servers running. Ctrl+C to stop.")
        signal.pause()
    else:
        fleet.monitor()


if __name__ == "__main__":
    main()
