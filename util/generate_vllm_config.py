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
        help="Base model for actors",
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

    if args.target == "redteam":
        # Redteam training: single student vLLM with student + opponent LoRAs.
        student_extra_args = ["--dtype", "auto"]
        lora_modules: list[str] = []
        if args.student_lora:
            lora_modules.append(f"student={args.student_lora}")
        if args.opponent_lora:
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
        # Placeholder student entry so registry-readers don't fail.
        config["servers"].append(
            {
                "id": "student",
                "model": args.model,
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
