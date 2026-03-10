#!/usr/bin/env python3
"""
Plot self-play relative strength metrics across iterations.

Usage:
    python util/plot_selfplay_results.py <selfplay_dir>

Example:
    python util/plot_selfplay_results.py results-20231024-1234-abcde
"""

# TODO: This can be improved eventually

import os
import sys
import json
import matplotlib.pyplot as plt


def parse_redteam_log(filepath):
    """Returns Attack Success Rate (ASR) for an iteration."""
    total_episodes = 0
    successes = 0
    opponent_model = None
    opponent_path = None

    with open(filepath, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                total_episodes += 1
                if data.get("outcome_tier") == "honeypot_access":
                    successes += 1

                if opponent_model is None:
                    opponent_model = data.get("opponent_model")
                if opponent_path is None:
                    opponent_path = data.get("opponent_lora_path")
            except:
                continue

    asr = (successes / total_episodes * 100) if total_episodes > 0 else 0
    return asr, total_episodes, opponent_model, opponent_path


def parse_blueteam_log(filepath):
    """Returns True Positive Rate (Benign Allowed) and True Negative Rate (Attack Refused)."""
    benign_total = 0
    true_positives = 0

    attack_total = 0
    true_negatives = 0

    opponent_model = None
    opponent_path = None

    with open(filepath, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                turn_type = data.get("turn_type")
                outcome = data.get("outcome_tier")

                if turn_type == "benign":
                    benign_total += 1
                    if outcome == "true_positive":
                        true_positives += 1
                elif turn_type == "attack":
                    attack_total += 1
                    if outcome == "true_negative":
                        true_negatives += 1

                if opponent_model is None:
                    opponent_model = data.get("opponent_model")
                if opponent_path is None:
                    opponent_path = data.get("opponent_lora_path")
            except:
                continue

    tpr = (true_positives / benign_total * 100) if benign_total > 0 else 0
    tnr = (true_negatives / attack_total * 100) if attack_total > 0 else 0

    return tpr, tnr, opponent_model, opponent_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python util/plot_selfplay_results.py <selfplay_dir>")
        sys.exit(1)

    selfplay_dir = sys.argv[1]
    if not os.path.isdir(selfplay_dir):
        print(f"Error: Directory not found: {selfplay_dir}")
        sys.exit(1)

    iteration_dirs = []
    for item in os.listdir(selfplay_dir):
        if item.startswith("iter_") and os.path.isdir(os.path.join(selfplay_dir, item)):
            iteration_dirs.append(item)

    # Sort logically (iter_1, iter_2, iter_10, etc)
    iteration_dirs.sort(
        key=lambda x: int(x.split("_")[1]) if len(x.split("_")) > 1 else 0
    )

    if not iteration_dirs:
        print(f"No iteration directories (iter_X) found in {selfplay_dir}")
        sys.exit(1)

    iterations = []
    redteam_asr = []
    blueteam_tpr = []
    blueteam_tnr = []

    print(f"Analyzing Self-Play Run: {selfplay_dir}")
    print("-" * 80)
    print(
        f"{'Iter':<6} | {'Red ASR':<10} | {'Red Opponent':<20} | {'Blue TNR':<10} | {'Blue TPR':<10} | {'Blue Opponent':<20}"
    )
    print("-" * 80)

    for iter_dir in iteration_dirs:
        iter_path = os.path.join(selfplay_dir, iter_dir)
        iter_num = int(iter_dir.split("_")[1])

        red_log = os.path.join(iter_path, "redteam", "debug_logs", "reward_debug.jsonl")
        blue_log = os.path.join(
            iter_path, "blueteam", "debug_logs", "reward_debug.jsonl"
        )

        asr = 0
        r_opp_model = "None"
        r_opp_path = "None"
        if os.path.exists(red_log):
            asr, _, r_opp_model, r_opp_path = parse_redteam_log(red_log)
        elif os.path.exists(os.path.join(iter_path, "redteam", "reward_debug.jsonl")):
            asr, _, r_opp_model, r_opp_path = parse_redteam_log(
                os.path.join(iter_path, "redteam", "reward_debug.jsonl")
            )

        tpr = 0
        tnr = 0
        b_opp_model = "None"
        b_opp_path = "None"
        if os.path.exists(blue_log):
            tpr, tnr, b_opp_model, b_opp_path = parse_blueteam_log(blue_log)
        elif os.path.exists(os.path.join(iter_path, "blueteam", "reward_debug.jsonl")):
            tpr, tnr, b_opp_model, b_opp_path = parse_blueteam_log(
                os.path.join(iter_path, "blueteam", "reward_debug.jsonl")
            )

        iterations.append(iter_num)
        redteam_asr.append(asr)
        blueteam_tpr.append(tpr)
        blueteam_tnr.append(tnr)

        r_opp_display = r_opp_model if r_opp_model else "Base"
        b_opp_display = b_opp_model if b_opp_model else "Base"

        print(
            f"{iter_num:<6} | {asr:>8.1f}% | {r_opp_display:<20} | {tnr:>8.1f}% | {tpr:>8.1f}% | {b_opp_display:<20}"
        )

    print("-" * 80)

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(
        iterations,
        redteam_asr,
        marker="o",
        color="#e74c3c",
        linewidth=2,
        label="Red Team ASR (Attack Success)",
    )
    plt.plot(
        iterations,
        blueteam_tnr,
        marker="s",
        color="#3498db",
        linewidth=2,
        label="Blue Team TNR (Attacks Prevented)",
    )
    plt.plot(
        iterations,
        blueteam_tpr,
        marker="^",
        color="#2ecc71",
        linewidth=2,
        linestyle="--",
        label="Blue Team TPR (Benign Allowed)",
    )

    plt.title("Self-Play Relative Strength Across Iterations", fontsize=14)
    plt.xlabel("Self-Play Iteration", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.xticks(iterations)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center right")

    out_file = os.path.join(selfplay_dir, "selfplay_relative_strength.png")
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"\nSaved plot to: {out_file}")


if __name__ == "__main__":
    main()
