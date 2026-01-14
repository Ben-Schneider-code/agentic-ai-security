#!/usr/bin/env python3
"""
Plot training results using detailed reward debug logs or tensorboard logs.

Usage:
    python util/plot_training_results.py <run_dir>

Example:
    python util/plot_training_results.py results/.../run_1

This script generates a graph with:
1. Total reward per episode
2. Cumulative outcomes by tier (Permission Violation, Honeypot, etc.)
3. Special markers for high-quality attacks (Permission Violation + Fluent)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_debug_logs(run_dir: str):
    """
    Parse reward_debug.jsonl to extract detailed episode data.

    Returns:
        tuple: (episodes, rewards, metrics_dict, fluent_violations)
    """
    # Try finding the debug log in likely locations
    candidates = [
        os.path.join(run_dir, "debug_logs", "reward_debug.jsonl"),
        os.path.join(run_dir, "reward_debug.jsonl"),
    ]

    log_file = None
    for c in candidates:
        if os.path.exists(c):
            log_file = c
            break

    if not log_file:
        print(
            "No reward_debug.jsonl found. Falling back to simple log parsing is not supported for these new metrics."
        )
        return None

    print(f"Parsing detailed logs from: {log_file}")

    episodes = []
    rewards = []

    # Cumulative counters
    counts = defaultdict(list)
    categories = [
        "permission_violation",
        "honeypot_access",
        "sensitive_query",
        "sql_generated",
        "sql_error",
        "no_sql",
    ]

    # Running totals
    totals = defaultdict(int)

    # Special marker list: (episode_num, reward)
    fluent_violations = []

    with open(log_file, "r") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)

                # Determine episode number (1-indexed)
                ep_num = i + 1
                if "step_count" in data:
                    # In single-step episodes, line count is episode count
                    pass

                episodes.append(ep_num)
                rewards.append(data.get("final_reward", 0.0))

                tier = data.get("outcome_tier", "no_sql")
                breakdown = data.get("reward_breakdown", {})

                # Determine specific category
                category = "no_sql"

                if tier == "permission_violation":
                    category = "permission_violation"
                    # Check for fluency
                    fluency = data.get("fluency_details")
                    if fluency:
                        # "Non-gibberish" = Fluent
                        # Check keys from FluencyJudge.get_fluency_info
                        is_fluent = fluency.get("is_fluent", False)
                        # Also double check heuristic just in case
                        is_gibberish = fluency.get("heuristic_gibberish", False)

                        if is_fluent and not is_gibberish:
                            fluent_violations.append(ep_num)

                elif tier == "honeypot_access":
                    category = "honeypot_access"
                elif tier == "sensitive_query":
                    category = "sensitive_query"
                elif tier == "sql_generated":
                    category = "sql_generated"
                elif tier == "no_sql":
                    if "sql_error" in breakdown:
                        category = "sql_error"
                    else:
                        category = "no_sql"

                totals[category] += 1

                # Update history for all categories
                for cat in categories:
                    counts[cat].append(totals[cat])

            except json.JSONDecodeError:
                continue

    return episodes, rewards, counts, fluent_violations


def plot_training_results(run_dir: str) -> None:
    # Validate directory
    if not os.path.exists(run_dir):
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    # Try parsing detailed debug logs first
    data = parse_debug_logs(run_dir)

    if data:
        episodes, rewards, counts, fluent_violations = data
        cumulative_rewards = np.cumsum(rewards)
        using_debug_logs = True
    else:
        # Fallback to Tensorboard (Legacy)
        # Note: This won't support the requested "SQL Error" vs "No SQL" distinction
        # or the green dots, but kept for backward compatibility if needed.
        print("Falling back to Tensorboard logs...")
        event_files = [
            f for f in os.listdir(run_dir) if f.startswith("events.out.tfevents")
        ]
        if not event_files:
            print("No tensorboard events found either. Exiting.")
            sys.exit(1)

        event_file = os.path.join(run_dir, sorted(event_files)[-1])
        ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
        ea.Reload()

        if "episodic_return" not in ea.Tags()["scalars"]:
            print("No episodic_return tag found.")
            sys.exit(1)

        events = ea.Scalars("episodic_return")
        episodes = list(range(1, len(events) + 1))
        rewards = [e.value for e in events]
        cumulative_rewards = np.cumsum(rewards)

        # We can't plot the detailed categories without the debug log
        counts = {}
        fluent_violations = []
        using_debug_logs = False

    # === Plotting ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # --- Plot 1: Rewards ---
    ax1.plot(
        episodes,
        rewards,
        color="#3498db",
        alpha=0.6,
        label="Reward per Episode",
        linewidth=1,
    )
    ax1.plot(
        episodes,
        cumulative_rewards,
        color="#27ae60",
        alpha=0.9,
        label="Cumulative Reward",
        linewidth=2,
    )

    # Plot "Permission Violation + Fluent" as green dots on data points, not just x-axis
    if fluent_violations:
        # Find rewards for these episodes (ep_num is 1-indexed)
        # indices = [ep - 1 for ep in fluent_violations if ep <= len(rewards)]
        # y_vals = [rewards[i] for i in indices]
        # ax1.scatter(fluent_violations, y_vals, color="#00ff00", edgecolor="black", s=50, zorder=5, label="Vital Success (Fluent Violation)")

        # User asked for "green dot on the x-axis"
        # We'll put them at the bottom of the graph
        min_y = min(rewards) if rewards else 0
        y_pos = [min_y] * len(fluent_violations)
        ax1.scatter(
            fluent_violations,
            y_pos,
            color="#2ecc71",
            marker="o",
            s=60,
            label="Fluent Perm. Violation",
            zorder=10,
            edgecolor="black",
        )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards and Successes")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Cumulative Outcomes ---
    if using_debug_logs:
        colors = {
            "permission_violation": "#e74c3c",  # Red
            "honeypot_access": "#9b59b6",  # Purple
            "sensitive_query": "#f39c12",  # Orange
            "sql_generated": "#3498db",  # Blue
            "sql_error": "#95a5a6",  # Gray
            "no_sql": "#bdc3c7",  # Light Gray
        }

        labels = {
            "permission_violation": "Permission Violation",
            "honeypot_access": "Honeypot Access",
            "sensitive_query": "Sensitive Query",
            "sql_generated": "Valid SQL Generated",
            "sql_error": "SQL Error",
            "no_sql": "No SQL / Refusal",
        }

        for cat, data_points in counts.items():
            if data_points and data_points[-1] > 0:
                ax2.plot(
                    episodes,
                    data_points,
                    color=colors.get(cat, "black"),
                    label=labels.get(cat, cat),
                    linewidth=2,
                )
    else:
        ax2.text(
            0.5,
            0.5,
            "Detailed outcome metrics require reward_debug.jsonl",
            ha="center",
            va="center",
        )

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Count")
    ax2.set_title("Cumulative Outcomes by Category")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(run_dir, "training_results_detailed.png")
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Print Summary
    print("\n=== Summary Statistics ===")
    print(f"Total Episodes: {len(episodes)}")
    if using_debug_logs:
        for cat, data_points in counts.items():
            final_count = data_points[-1]
            pct = (final_count / len(episodes)) * 100
            print(f"{labels.get(cat, cat):<25}: {final_count:>4} ({pct:.1f}%)")
        print(f"Fluent Perm. Violations  : {len(fluent_violations):>4}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    plot_training_results(sys.argv[1])


if __name__ == "__main__":
    main()
