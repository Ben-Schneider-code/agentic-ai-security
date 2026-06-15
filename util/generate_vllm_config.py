#!/usr/bin/env python3
"""Generate dynamic vLLM configs for actor models (student/opponent).

Blueteam mode supports a *pool* of opponent LoRAs (the red checkpoint registry)
so blue trains against the entire history of red iterations within a self-play
cell. Adapters are pre-loaded via --lora-modules at server start (NEVER via the
runtime /v1/load_lora_adapter endpoint; vLLM <0.6 leaks GPU mem on dynamic load).

Redteam mode runs a single student vLLM with optional student/opponent LoRAs.
"""

import argparse
import json
import os

MAX_LORA_RANK = 64
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GPU_MEMORY_UTIL = 0.90

# Bound on simultaneously-loaded adapters in vLLM. Each rank-64 LoRA on a 7B
# model is ~80–120 MB BF16 in GPU mem; --max-cpu-loras can exceed this since
# CPU RAM swaps in/out. Iteration count must satisfy K <= max_cpu_loras.
DEFAULT_MAX_LORAS = 8
DEFAULT_MAX_CPU_LORAS = 16


def _adapter_base(path: str) -> str | None:
    """Read a LoRA's adapter_config.json base_model_name_or_path (or None)."""
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            return json.load(f).get("base_model_name_or_path")
    except Exception:  # noqa: BLE001
        return None


def _assert_adapter_base(name: str, path: str, server_base: str) -> None:
    """Fail-fast if a LoRA's base model is incompatible with the server it will
    be loaded onto. vLLM requires every adapter on a server to share that
    server's base model — a mismatch (e.g. a red non-SQL adapter on a blue
    text2sql server) crashes deep in vLLM with an opaque shape error, so we
    catch it here. Compared by basename to be robust to HF-id vs local-path."""
    adapter_base = _adapter_base(path)
    if adapter_base is None:
        return  # no adapter_config to check — let vLLM be the backstop
    if os.path.basename(adapter_base.rstrip("/")) != os.path.basename(server_base.rstrip("/")):
        raise SystemExit(
            f"[generate_vllm_config] LoRA '{name}' was trained on base "
            f"{adapter_base!r}, but its server hosts {server_base!r}. A vLLM "
            f"server's LoRA pool must all share that server's base model. For a "
            f"heterogeneous red/blue run, pass the OPPONENT's base as --model."
        )


def _load_red_pool(pool_path: str | None) -> list[dict]:
    """Read the red LoRA registry. Returns [{name, path}, ...]."""
    if not pool_path:
        return []
    if not os.path.exists(pool_path):
        print(f"[generate_vllm_config] WARNING: pool path {pool_path} does not exist")
        return []
    with open(pool_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):
        return data
    print(f"[generate_vllm_config] WARNING: pool {pool_path} has unexpected shape")
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        required=True,
        choices=["redteam", "blueteam"],
        help="The agent being trained",
    )
    parser.add_argument(
        "--opponent-lora",
        help=(
            "Path to a single opponent LoRA. For blueteam this is treated as a "
            "fallback pool of size 1 if --opponent-lora-pool is not given."
        ),
    )
    parser.add_argument(
        "--opponent-lora-pool",
        help=(
            "Path to red_lora_registry.json (blueteam only). When provided, blue's "
            "vLLM pre-loads every adapter listed and the env samples one per episode."
        ),
    )
    parser.add_argument(
        "--student-lora",
        help="Path to student trained LoRA (useful for hosting/evaluation)",
    )
    parser.add_argument(
        "--out-config", required=True, help="Path to write the generated JSON config"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model for the ACTOR vLLM server (the OPPONENT being served: "
        "the victim in red training, the red pool in blue training).",
    )
    parser.add_argument(
        "--student-base-model",
        default=None,
        help="Base model of the STUDENT being trained. Defaults to --model "
        "(homogeneous). When it differs from --model (heterogeneous red/blue), "
        "the student LoRA is NOT co-loaded on the opponent's actor server — the "
        "student generates in-process during training, so the adapter is unused "
        "there and would be invalid on the opponent's base.",
    )
    parser.add_argument(
        "--actor-gpu",
        type=int,
        required=True,
        help="Physical GPU index for actor vLLM servers (required — no default; "
        "a wrong index would launch vLLM on another user's GPU on a shared host).",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=DEFAULT_MAX_LORAS,
        help="vLLM --max-loras (concurrent on-GPU adapters)",
    )
    parser.add_argument(
        "--max-cpu-loras",
        type=int,
        default=DEFAULT_MAX_CPU_LORAS,
        help="vLLM --max-cpu-loras (CPU-resident adapter pool)",
    )
    parser.add_argument(
        "--registry-path",
        required=True,
        help="Path the launched fleet writes its actor registry JSON to "
        "(per-run, under the run's runtime dir — never a shared /tmp path).",
    )
    parser.add_argument(
        "--actor-port",
        type=int,
        required=True,
        help="TCP port for the actor vLLM server (dynamically allocated per run).",
    )
    args = parser.parse_args()

    if args.target == "blueteam" and not (args.opponent_lora or args.opponent_lora_pool):
        parser.error(
            "--opponent-lora or --opponent-lora-pool is required when target is blueteam"
        )

    config = {
        "_comment": f"Dynamic vLLM config generated for '{args.target}' training",
        "base_port": args.actor_port,
        "host": "0.0.0.0",
        "registry_path": args.registry_path,
        "servers": [],
    }

    # Student base may differ from the actor (opponent) base in a heterogeneous
    # red/blue run. args.model is ALWAYS the opponent/actor base.
    student_base = args.student_base_model or args.model
    heterogeneous = (
        os.path.basename(student_base.rstrip("/"))
        != os.path.basename(args.model.rstrip("/"))
    )

    if args.target == "redteam":
        # Redteam training: the actor server hosts the VICTIM (blue) base; the
        # red student generates in-process, so its LoRA is only co-loaded here in
        # the homogeneous case (it would be invalid on a different victim base).
        student_extra_args = ["--dtype", "auto"]
        lora_modules: list[str] = []
        if args.student_lora and not heterogeneous:
            lora_modules.append(f"student={args.student_lora}")
        elif args.student_lora and heterogeneous:
            print(
                f"[generate_vllm_config] heterogeneous bases (student "
                f"{student_base!r} != victim {args.model!r}); NOT co-loading the "
                f"student LoRA on the victim server (student runs in-process)."
            )
        if args.opponent_lora:
            _assert_adapter_base("opponent_lora", args.opponent_lora, args.model)
            lora_modules.append(f"opponent_lora={args.opponent_lora}")

        if lora_modules:
            student_extra_args += [
                "--enable-lora",
                "--max-lora-rank", str(MAX_LORA_RANK),
                "--max-loras", str(max(args.max_loras, len(lora_modules))),
                "--max-cpu-loras", str(max(args.max_cpu_loras, len(lora_modules))),
                "--lora-modules", *lora_modules,
            ]
        config["servers"].append(
            {
                "id": "student",
                "model": args.model,
                "gpus": [args.actor_gpu],
                "port": args.actor_port,
                "max_model_len": DEFAULT_MAX_MODEL_LEN,
                "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTIL,
                "extra_args": student_extra_args,
            }
        )
    else:
        # Blueteam training: redteam vLLM holds the full red checkpoint pool
        # so the env can swap adapters per episode.
        red_pool = _load_red_pool(args.opponent_lora_pool)
        # Fallback: if no pool but a single --opponent-lora was passed, build a
        # one-entry pool keyed as `redteam` (legacy compatibility).
        if not red_pool and args.opponent_lora:
            red_pool = [{"name": "redteam", "path": args.opponent_lora}]
        if not red_pool:
            raise SystemExit(
                "[generate_vllm_config] No red adapters to load — pool empty and "
                "no --opponent-lora given."
            )

        # Validate pool: every adapter must have a name and existing path.
        validated: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        for entry in red_pool:
            name = entry.get("name")
            path = entry.get("path")
            if not name or not path:
                print(f"[generate_vllm_config] WARNING: malformed pool entry: {entry!r}")
                continue
            if name in seen_names:
                print(f"[generate_vllm_config] WARNING: duplicate adapter name '{name}' — skipping")
                continue
            if not os.path.exists(path):
                print(f"[generate_vllm_config] WARNING: adapter path missing: {path} (name={name})")
                continue
            # vLLM requires every adapter on this server to share its base model
            # (args.model = the RED/opponent base in the blue phase). Fail-fast on
            # a heterogeneous mismatch rather than crashing deep inside vLLM.
            _assert_adapter_base(name, path, args.model)
            validated.append((name, path))
            seen_names.add(name)

        if not validated:
            raise SystemExit(
                "[generate_vllm_config] All adapters in pool are invalid — aborting."
            )

        # Auto-scale the LoRA caps to the pool size.
        max_loras = max(args.max_loras, len(validated))
        max_cpu_loras = max(args.max_cpu_loras, len(validated))

        lora_modules = [f"{name}={path}" for name, path in validated]

        config["_pool_size"] = len(validated)
        config["_pool_names"] = [name for name, _ in validated]

        config["servers"].append(
            {
                "id": "redteam",
                "model": args.model,
                "gpus": [args.actor_gpu],
                "port": args.actor_port,
                "max_model_len": DEFAULT_MAX_MODEL_LEN,
                "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTIL,
                "extra_args": [
                    "--dtype", "auto",
                    "--enable-lora",
                    "--max-lora-rank", str(MAX_LORA_RANK),
                    "--max-loras", str(max_loras),
                    "--max-cpu-loras", str(max_cpu_loras),
                    "--lora-modules", *lora_modules,
                ],
            }
        )
        # Placeholder student entry so registry-readers don't fail. Not launched
        # (start_vllm skips _note entries); model reflects the student base.
        config["servers"].append(
            {
                "id": "student",
                "model": student_base,
                "gpus": [args.actor_gpu],
                "port": args.actor_port,
                "max_model_len": DEFAULT_MAX_MODEL_LEN,
                "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTIL,
                "_note": "placeholder — blueteam env uses REDTEAM_VLLM_URL, not STUDENT_VLLM_URL",
            }
        )

    out_dir = os.path.dirname(args.out_config)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_config, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Generated vLLM config for {args.target} actors at {args.out_config}")


if __name__ == "__main__":
    main()
