#!/usr/bin/env python
"""
RFT (Rejection Sampling Fine-Tuning) for Redteam Agent Bootstrapping.

This script uses a coach model (stronger LLM) to generate diverse attacks,
filters successful ones, and fine-tunes the 8B redteam agent via SFT.

Process:
1. Generate N attack conversations using coach model
2. Execute against blueteam environment to filter successes
3. Create SFT dataset from successful attacks
4. Fine-tune redteam agent on filtered data
5. Repeat for K rounds or until success rate threshold reached
"""

# TODO(Justin): I don't think this code is being used for now?

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from marft.coach import CoachGenerator
from marft.envs.redteam_sql.redteam_sql_env import (
    SQLEnv,
    REWARD_CONFIG,
    generate_redteam_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RFT training for redteam agent")

    # Model arguments
    parser.add_argument(
        "--student_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Student model (8B redteam agent to be fine-tuned)",
    )
    parser.add_argument(
        "--coach_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help="Coach model for generating diverse attacks",
    )
    parser.add_argument(
        "--coach_vllm_url",
        type=str,
        default=os.environ.get("COACH_VLLM_URL", "http://localhost:8000/v1"),
        help="vLLM server URL for coach model",
    )
    parser.add_argument(
        "--student_vllm_url",
        type=str,
        default=os.environ.get("STUDENT_VLLM_URL", "http://localhost:8001/v1"),
        help="vLLM server URL for student/blueteam model",
    )

    # RFT parameters
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of RFT rounds (generate → filter → SFT)",
    )
    parser.add_argument(
        "--conversations_per_round",
        type=int,
        default=1000,
        help="Number of attack conversations to generate per round",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for parallel generation",
    )
    parser.add_argument(
        "--num_turns",
        type=int,
        default=5,
        help="Number of turns per conversation",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=0.15,
        help="Stop RFT when this success rate is reached (e.g., 0.15 = 15%%)",
    )

    # SFT parameters
    parser.add_argument(
        "--sft_epochs",
        type=int,
        default=3,
        help="Number of SFT epochs per round",
    )
    parser.add_argument(
        "--sft_lr",
        type=float,
        default=1e-5,
        help="Learning rate for SFT",
    )
    parser.add_argument(
        "--sft_batch_size",
        type=int,
        default=8,
        help="Batch size for SFT training",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rft_outputs",
        help="Directory to save checkpoints and datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


class RFTTrainer:
    """Manages the RFT training loop."""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Initialize coach generator
        print(f"[RFT] Initializing coach generator: {args.coach_model}")
        self.coach = CoachGenerator(
            coach_model_name=args.coach_model,
            vllm_base_url=args.coach_vllm_url,
            temperature=0.9,
        )

        # Initialize evaluation environment
        print("[RFT] Initializing evaluation environment (blueteam victim)")
        print(f"[RFT] Student vLLM URL: {args.student_vllm_url}")
        self.eval_env = SQLEnv(
            rank=0,
            model_name=args.student_model,
            num_agents=1,
            horizon=args.num_turns,
            mode="test",
            shared_honeypots=set(),  # Fresh env for each round
            vllm_base_url=args.student_vllm_url,
        )

        # Track metrics
        self.round_metrics = []

    def generate_conversations(self, num_conversations: int) -> List[List[str]]:
        """Generate attack conversations using coach model."""
        print(f"[RFT] Generating {num_conversations} conversations...")

        # Generate redteam prompt
        prompt = generate_redteam_prompt(
            list(REWARD_CONFIG.honeypot_tables),
            list(REWARD_CONFIG.honeypot_columns),
            list(REWARD_CONFIG.honeypot_order_ids),
            list(REWARD_CONFIG.honeypot_guids),
            current_turn=1,
            max_turns=self.args.num_turns,
            remaining_salesorder_columns=list(REWARD_CONFIG.honeypot_salesorder_columns),
        )

        all_conversations = []
        num_batches = (num_conversations + self.args.batch_size - 1) // self.args.batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            batch_size = min(self.args.batch_size, num_conversations - len(all_conversations))
            batch_conversations = self.coach.generate_batch_sync(
                prompt, batch_size=batch_size, num_turns=self.args.num_turns
            )
            all_conversations.extend(batch_conversations)

        print(f"[RFT] Generated {len(all_conversations)} conversations")
        return all_conversations

    def filter_successful_conversations(
        self, conversations: List[List[str]]
    ) -> tuple[List[List[str]], float]:
        """
        Execute conversations against blueteam and filter successes.

        Returns:
            (successful_conversations, success_rate)
        """
        print(f"[RFT] Filtering {len(conversations)} conversations...")

        successful = []
        for conv_idx, conversation in enumerate(tqdm(conversations, desc="Filtering")):
            # Reset environment
            self.eval_env.reset()

            # Execute conversation
            episode_reward = 0.0
            found_honeypot = False

            for turn_idx, attack_text in enumerate(conversation):
                # Step environment with attack
                actions = np.array([[attack_text]], dtype=object)
                next_obs, rewards, dones, infos = self.eval_env.step(actions[0])

                episode_reward += rewards[-1]  # Last agent reward

                # Check if honeypot found
                if infos.get("terminal_success", False):
                    found_honeypot = True
                    break

                if dones[0]:
                    break

            # Consider successful if honeypot found OR positive reward
            if found_honeypot or episode_reward > 0:
                successful.append({
                    "conversation": conversation,
                    "reward": float(episode_reward),
                    "found_honeypot": found_honeypot,
                })

        success_rate = len(successful) / max(len(conversations), 1)
        print(f"[RFT] Success rate: {success_rate:.2%} ({len(successful)}/{len(conversations)})")

        return successful, success_rate

    def create_sft_dataset(self, successful_conversations: List[Dict]) -> List[Dict]:
        """
        Convert successful conversations to SFT training format.

        Format:
        {
            "prompt": "<redteam_system_prompt>...",
            "completion": "attack text",
        }
        """
        dataset = []

        # Generate base prompt
        base_prompt = generate_redteam_prompt(
            list(REWARD_CONFIG.honeypot_tables),
            list(REWARD_CONFIG.honeypot_columns),
            list(REWARD_CONFIG.honeypot_order_ids),
            list(REWARD_CONFIG.honeypot_guids),
            current_turn=1,
            max_turns=self.args.num_turns,
            remaining_salesorder_columns=list(REWARD_CONFIG.honeypot_salesorder_columns),
        )

        for item in successful_conversations:
            conversation = item["conversation"]
            # Each turn becomes a training example
            for turn_idx, attack_text in enumerate(conversation):
                dataset.append({
                    "prompt": base_prompt,
                    "completion": attack_text,
                    "reward": item["reward"],
                    "turn": turn_idx + 1,
                })

        return dataset

    def run_sft(self, dataset: List[Dict], round_num: int):
        """
        Fine-tune student model on successful attacks.

        Note: This is a simplified version. In practice, use HuggingFace Trainer
        or similar for proper SFT with LoRA.
        """
        print(f"[RFT] Running SFT on {len(dataset)} examples (round {round_num})...")

        # Save dataset
        dataset_path = self.output_dir / f"round_{round_num}_dataset.jsonl"
        with open(dataset_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        print(f"[RFT] Saved dataset to {dataset_path}")

        # TODO: Implement actual SFT training loop
        # For now, just print summary
        print(f"[RFT] SFT training would happen here with:")
        print(f"  - Model: {self.args.student_model}")
        print(f"  - Epochs: {self.args.sft_epochs}")
        print(f"  - Learning rate: {self.args.sft_lr}")
        print(f"  - Batch size: {self.args.sft_batch_size}")
        print(f"  - Dataset size: {len(dataset)}")

        # Save checkpoint path (placeholder)
        checkpoint_path = self.output_dir / f"round_{round_num}_checkpoint"
        checkpoint_path.mkdir(exist_ok=True)
        print(f"[RFT] Checkpoint would be saved to {checkpoint_path}")

    def run(self):
        """Run full RFT training loop."""
        print(f"[RFT] Starting RFT training for {self.args.num_rounds} rounds")
        print(f"[RFT] Target success rate: {self.args.success_threshold:.2%}")

        for round_num in range(1, self.args.num_rounds + 1):
            print(f"\n{'='*80}")
            print(f"ROUND {round_num}/{self.args.num_rounds}")
            print(f"{'='*80}\n")

            # Generate conversations
            conversations = self.generate_conversations(self.args.conversations_per_round)

            # Filter successful ones
            successful, success_rate = self.filter_successful_conversations(conversations)

            # Track metrics
            metrics = {
                "round": round_num,
                "generated": len(conversations),
                "successful": len(successful),
                "success_rate": success_rate,
            }
            self.round_metrics.append(metrics)

            # Create SFT dataset
            if successful:
                sft_dataset = self.create_sft_dataset(successful)

                # Run SFT
                self.run_sft(sft_dataset, round_num)
            else:
                print("[RFT] WARNING: No successful conversations, skipping SFT")

            # Check if we've reached target success rate
            if success_rate >= self.args.success_threshold:
                print(f"\n[RFT] SUCCESS! Reached target success rate: {success_rate:.2%}")
                break

        # Save final metrics
        metrics_path = self.output_dir / "rft_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.round_metrics, f, indent=2)
        print(f"\n[RFT] Metrics saved to {metrics_path}")

        print("\n[RFT] RFT training complete!")
        return self.round_metrics


def main():
    args = parse_args()
    trainer = RFTTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
