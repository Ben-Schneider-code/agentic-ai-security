#!/usr/bin/env python3
"""
Generate dynamic vLLM configs for actor models (student/opponent)
depending on the training target.
"""

import argparse
import json
import os


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

    student_extra_args = ["--dtype", "auto"]
    if args.student_lora:
        student_extra_args.extend([
            "--enable-lora",
            "--lora-modules", f"student={args.student_lora}"
        ])

    student_server = {
        "id": "student",
        "model": args.model,
        "gpus": [2],
        "port": 8001,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "extra_args": student_extra_args,
    }

    config["servers"].append(student_server)

    if args.target == "blueteam":
        opponent_server = {
            "id": "redteam",
            "model": args.model,
            "gpus": [3],
            "port": 8002,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.90,
            "extra_args": [
                "--dtype",
                "auto",
                "--enable-lora",
                "--lora-modules",
                f"redteam={args.opponent_lora}",
            ],
        }
        config["servers"].append(opponent_server)

    out_dir = os.path.dirname(args.out_config)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_config, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Generated vLLM config for {args.target} actors at {args.out_config}")


if __name__ == "__main__":
    main()
