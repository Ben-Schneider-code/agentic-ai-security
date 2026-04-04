#!/usr/bin/env python3
"""
Generate dynamic vLLM configs for actor models (student/opponent)
depending on the training target.
"""

import argparse
import json
import os

MAX_LORA_RANK = 64
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY_UTIL = 0.90


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        required=True,
        choices=["redteam", "blueteam"],
        help="The agent being trained",
    )
    parser.add_argument(
        "--opponent-lora", help="Path to opponent LoRA (required if target is blueteam)"
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
        default=1,
        help="GPU index for actor vLLM servers (default: 1)",
    )
    args = parser.parse_args()

    if args.target == "blueteam" and not args.opponent_lora:
        parser.error("--opponent-lora is required when target is blueteam")

    config = {
        "_comment": f"Dynamic vLLM config generated for '{args.target}' training",
        "base_port": 8001,
        "host": "0.0.0.0",
        "registry_path": "/tmp/vllm_actor_registry.json",
        "servers": [],
    }

    if args.target == "redteam":
        # Redteam training: student vLLM on actor_gpu
        student_extra_args = ["--dtype", "auto"]
        lora_modules = []
        if args.student_lora:
            lora_modules.append(f"student={args.student_lora}")
        if args.opponent_lora:
            lora_modules.append(f"opponent_lora={args.opponent_lora}")

        if lora_modules:
            student_extra_args.append("--enable-lora")
            student_extra_args.append("--max-lora-rank")
            student_extra_args.append(str(MAX_LORA_RANK))
            student_extra_args.append("--lora-modules")
            student_extra_args.extend(lora_modules)
        config["servers"].append(
            {
                "id": "student",
                "model": args.model,
                "gpus": [args.actor_gpu],
                "port": 8001,
                "max_model_len": DEFAULT_MAX_MODEL_LEN,
                "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTIL,
                "extra_args": student_extra_args,
            }
        )
    else:
        # Blueteam training: NO student vLLM — the blueteam env calls redteam vLLM
        # directly, so the student vLLM would just waste a GPU.
        config["servers"].append(
            {
                "id": "redteam",
                "model": args.model,
                "gpus": [args.actor_gpu],
                "port": 8002,
                "max_model_len": DEFAULT_MAX_MODEL_LEN,
                "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTIL,
                "extra_args": [
                    "--dtype",
                    "auto",
                    "--enable-lora",
                    "--max-lora-rank",
                    str(MAX_LORA_RANK),
                    "--lora-modules",
                    f"redteam={args.opponent_lora}",
                ],
            }
        )
        # Placeholder student entry so run_training.sh registry reads don't fail.
        # The blueteam env ignores STUDENT_VLLM_URL and uses REDTEAM_VLLM_URL.
        config["servers"].append(
            {
                "id": "student",
                "model": args.model,
                "gpus": [args.actor_gpu],
                "port": 8002,
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
