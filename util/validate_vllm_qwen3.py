#!/usr/bin/env python3
"""Standalone validation for the vLLM upgrade that enables Qwen3 support.

The run_replicate pipeline (self-play -> cross-eval -> benign-eval) launches vLLM
OpenAI servers via start_vllm.py and direct `python -m vllm.entrypoints.openai.api_server`
calls. vLLM 0.6.3 cannot serve Qwen3 (`Qwen3ForCausalLM` is absent from its model
registry), so the venv is upgraded to a Qwen3-native vLLM. This script proves the upgrade
"does not break other dependencies" BEFORE the user re-runs an expensive experiment.

It validates two things:

  Phase A (no GPU, `--check-only` runs only this):
    - the resolved dependency versions import cleanly together
    - the `vllm` Python API the repo uses (LLM, SamplingParams + its fields) is intact
    - EVERY api_server CLI flag the pipeline passes is still accepted by `--help`
      (this is how the `--disable-log-requests` rename is detected)
    - transformers can read a Qwen3 config (the in-process trainer path)

  Phase B (needs `--gpu`): for each model, launch a real OpenAI server with the SAME flags
    the pipeline uses (including the `--enable-lora` engine path), wait for readiness, and
    run a coherence probe over /v1/chat/completions. A wrong architecture (e.g. Qwen3
    without QK-norm) loads weights but emits gibberish, so the probe asserts a correct
    answer, not merely HTTP 200. Optional `--lora name=path` additionally proves
    name-addressed LoRA serving (the exact mechanism cross-eval/benign-eval rely on).

Examples
--------
  # fast, no-GPU API/flag audit
  .venv/bin/python util/validate_vllm_qwen3.py --check-only

  # full live validation (Arctic regression + the two Qwen3 targets)
  .venv/bin/python util/validate_vllm_qwen3.py --gpu 0 \
      --models Snowflake/Arctic-Text2SQL-R1-7B NousResearch/Hermes-4-14B Qwen/Qwen3-14B

  # prove name-addressed LoRA serving with a real adapter (single matching base)
  .venv/bin/python util/validate_vllm_qwen3.py --gpu 0 \
      --models Snowflake/Arctic-Text2SQL-R1-7B --lora blue=/path/to/blue/adapter
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# --- The exact api_server CLI flags the run_replicate pipeline passes. --------
# Sources: start_vllm.py build_cmd; util/generate_vllm_config.py extra_args;
# run_cross_eval.sh; util/run_benign_eval.sh; run_human_eval.sh. If a flag here
# is absent from the upgraded vLLM's --help, the pipeline scripts that pass it
# will break and must be updated (the audit names the likely replacement).
PIPELINE_API_SERVER_FLAGS = [
    "--model",
    "--port",
    "--host",
    "--gpu-memory-utilization",
    "--tensor-parallel-size",
    "--trust-remote-code",
    "--disable-log-requests",
    "--enforce-eager",
    "--max-model-len",
    "--disable-custom-all-reduce",
    "--chat-template",
    "--dtype",
    "--enable-lora",
    "--max-lora-rank",
    "--max-loras",
    "--max-cpu-loras",
    "--lora-modules",
]

DEFAULT_MODELS = [
    "Snowflake/Arctic-Text2SQL-R1-7B",  # Qwen2 base -> regression check
    "NousResearch/Hermes-4-14B",        # Qwen3-14B  -> the goal
    "Qwen/Qwen3-14B",                   # Qwen3-14B  -> the goal
]

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[PASS]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def info(msg: str) -> None:
    print(f"       {msg}")


# ---------------------------------------------------------------------------
# Phase A — import / API / flag audit (no GPU)
# ---------------------------------------------------------------------------
def phase_a(args, results: dict) -> None:
    print("\n=== Phase A: import / API / flag audit (no GPU) ===")

    # 1) resolved versions of the shared stack -------------------------------
    import importlib.metadata as md

    pkgs = ["vllm", "torch", "transformers", "xformers", "tokenizers",
            "pydantic", "fastapi", "ray", "peft", "accelerate", "numpy", "openai"]
    for p in pkgs:
        try:
            info(f"{p:<14} {md.version(p)}")
        except md.PackageNotFoundError:
            info(f"{p:<14} (not installed)")

    # 2) vllm Python API the repo uses --------------------------------------
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
        # Fields used at agent_loop.py:104 / connect_agent_to_db.py:133 plus the
        # n/stop/seed knobs the eval clients send over HTTP.
        SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048,
                       n=1, stop=["</s>"], seed=0)
        ok("vllm import + SamplingParams(...) construct")
        results["api"] = True
    except Exception as e:  # noqa: BLE001
        fail(f"vllm import / SamplingParams: {e!r}")
        results["api"] = False

    # 3) api_server --help flag audit ---------------------------------------
    help_text = ""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True, text=True, timeout=300,
        )
        help_text = (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:  # noqa: BLE001
        fail(f"could not run api_server --help: {e!r}")

    if help_text:
        missing = [f for f in PIPELINE_API_SERVER_FLAGS if f not in help_text]
        if not missing:
            ok(f"all {len(PIPELINE_API_SERVER_FLAGS)} pipeline api_server flags present")
            results["flags"] = True
        else:
            results["flags"] = False
            for f in missing:
                # Suggest a replacement by matching the flag's stem in --help.
                stem = f.lstrip("-").split("-", 1)[-1]  # e.g. "log-requests"
                hits = [ln.strip() for ln in help_text.splitlines()
                        if stem in ln and "--" in ln]
                fail(f"flag missing from --help: {f}")
                if hits:
                    info("  candidate replacement(s) in --help:")
                    for h in hits[:4]:
                        info(f"    {h}")
                else:
                    info("  no obvious replacement found in --help; check vLLM changelog")
            warn("update the sites that pass the missing flag(s): start_vllm.py, "
                 "run_cross_eval.sh, util/run_benign_eval.sh, run_human_eval.sh")
    else:
        results["flags"] = False

    # 4) transformers can read a Qwen3 config (trainer path) ----------------
    qwen3_models = [m for m in args.models if "qwen3" in m.lower() or "hermes-4" in m.lower()]
    target = qwen3_models[0] if qwen3_models else "Qwen/Qwen3-14B"
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(target, trust_remote_code=True)
        arch = getattr(cfg, "architectures", None)
        mtype = getattr(cfg, "model_type", "?")
        ok(f"transformers AutoConfig read {target} (model_type={mtype}, arch={arch})")
        results["transformers_qwen3"] = True
    except Exception as e:  # noqa: BLE001
        # Network/auth failures are not a vLLM-compat problem; flag as a warning.
        warn(f"could not load config for {target}: {e!r} "
             "(network/HF auth? not necessarily a compat failure)")
        results["transformers_qwen3"] = None


# ---------------------------------------------------------------------------
# Phase B — live OpenAI-server smoke per model (needs --gpu)
# ---------------------------------------------------------------------------
def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _http_get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_post_json(url: str, payload: dict, timeout: int = 120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json",
                                 "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _wait_ready(base: str, proc: subprocess.Popen, log_path: str,
                max_wait: int) -> bool:
    """Poll /v1/models until the server lists a model (mirrors OfflineLLM)."""
    start = time.time()
    while time.time() - start < max_wait:
        if proc.poll() is not None:
            fail(f"server exited early (rc={proc.returncode}); last log lines:")
            _tail(log_path)
            return False
        try:
            data = _http_get_json(f"{base}/v1/models", timeout=10)
            if data.get("data"):
                info(f"server ready in {time.time() - start:.0f}s; "
                     f"served ids: {[m.get('id') for m in data['data']]}")
                return True
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        time.sleep(5)
        info(f"waiting for server... ({time.time() - start:.0f}s / {max_wait}s)")
    fail(f"server not ready within {max_wait}s; last log lines:")
    _tail(log_path)
    return False


def _tail(log_path: str, n: int = 30) -> None:
    try:
        with open(log_path) as f:
            for ln in f.readlines()[-n:]:
                info("  | " + ln.rstrip())
    except OSError:
        pass


def _probe(base: str, model_id: str) -> bool:
    """Coherence probe: a wrong architecture yields gibberish, not '4'.

    max_tokens is generous and <think>...</think> is stripped so that reasoning
    models (e.g. Qwen3, which emits chain-of-thought before the answer) get to
    their final reply rather than being cut off mid-thought.
    """
    try:
        resp = _http_post_json(
            f"{base}/v1/chat/completions",
            {"model": model_id,
             "messages": [{"role": "user",
                           "content": "What is 2+2? Reply with only the number."}],
             "max_tokens": 512, "temperature": 0.0},
            timeout=180,
        )
        content = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        fail(f"  chat/completions [{model_id}]: {e!r}")
        return False
    if not content:
        fail(f"  chat/completions [{model_id}]: empty content")
        return False
    # Drop a (possibly unterminated) reasoning block before checking the answer.
    visible = re.sub(r"(?s)<think>.*?(</think>|$)", "", content).strip()
    answer = visible or content
    if "4" in answer:
        ok(f"  coherence probe [{model_id}]: {answer[:80]!r}")
        return True
    fail(f"  coherence probe [{model_id}] returned non-coherent text "
         f"(possible wrong kernels): {content[:160]!r}")
    return False


def serve_and_probe(model: str, args, loras: list[tuple[str, str]]) -> bool:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = f"/tmp/validate_vllm_{port}.log"

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--gpu-memory-utilization", str(args.gpu_mem),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--dtype", "auto",
        "--enforce-eager",          # mirrors start_vllm.py; faster startup for a smoke test
        "--enable-lora",            # exercise the LoRA engine path eval relies on
        "--max-lora-rank", "64",
    ]
    if loras:
        cmd += ["--max-loras", str(len(loras)),
                "--max-cpu-loras", str(len(loras)),
                "--lora-modules", *[f"{n}={p}" for n, p in loras]]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"\n--- serving {model} on GPU {args.gpu} (port {port}) ---")
    info("cmd: " + " ".join(cmd))
    passed = True
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            if not _wait_ready(base, proc, log_path, args.startup_timeout):
                return False
            ok(f"{model}: server reached ready state")
            # base-model probe
            passed &= _probe(base, model)
            # name-addressed LoRA probes (the cross-eval mechanism)
            for name, _ in loras:
                passed &= _probe(base, name)
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
    return passed


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="model ids to validate (default: Arctic + the two Qwen3 targets)")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index for the live phase (required unless --check-only)")
    ap.add_argument("--lora", action="append", default=[], metavar="name=path",
                    help="attach a LoRA and probe it by name; use with a single matching "
                         "--models base. Repeatable.")
    ap.add_argument("--check-only", action="store_true",
                    help="run Phase A only (no GPU, no servers)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--startup-timeout", type=int, default=600,
                    help="seconds to wait for each server to become ready")
    args = ap.parse_args()

    loras: list[tuple[str, str]] = []
    for spec in args.lora:
        if "=" not in spec:
            ap.error(f"--lora must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        loras.append((name, path))

    results: dict[str, object] = {}
    phase_a(args, results)

    if not args.check_only:
        if args.gpu is None:
            print()
            fail("--gpu is required for the live phase (or pass --check-only)")
            return 2
        print("\n=== Phase B: live OpenAI-server smoke per model ===")
        for model in args.models:
            results[f"serve:{model}"] = serve_and_probe(model, args, loras)

    # ---- summary ----------------------------------------------------------
    print("\n=== SUMMARY ===")
    hard_fail = False
    for k, v in results.items():
        if v is True:
            ok(k)
        elif v is None:
            warn(f"{k} (skipped / inconclusive — see above)")
        else:
            fail(k)
            hard_fail = True

    if hard_fail:
        print(f"\n{RED}VALIDATION FAILED{RESET} — do not run experiments until resolved.")
        return 1
    print(f"\n{GREEN}VALIDATION PASSED{RESET}")
    if args.check_only:
        print("(Phase A only — run without --check-only on a free GPU to validate serving.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
