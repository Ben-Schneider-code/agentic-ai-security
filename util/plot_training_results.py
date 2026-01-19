#!/usr/bin/env python3
"""
Plot training results using detailed reward debug logs or tensorboard logs.

Usage:
    python util/plot_training_results.py <run_dir>

Example:
    python util/plot_training_results.py results/.../run_1

This script generates a graph with:
1. Total reward per episode + Cumulative reward
2. Cumulative outcomes by tier (Permission Violation, Honeypot, etc.)
3. Gibberish rate over time (KEY WARNING SIGN for mode collapse)
4. Perplexity and Fluency Penalty trends
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Import evaluation metric functions
try:
    from calc_bleu import bleu_score

    HAVE_BLEU = True
except ImportError:
    try:
        from util.calc_bleu import bleu_score

        HAVE_BLEU = True
    except ImportError:
        HAVE_BLEU = False

try:
    from bert_score import score as bert_score_fn

    HAVE_BERTSCORE = True
except ImportError:
    HAVE_BERTSCORE = False


def parse_debug_logs(run_dir: str):
    """
    Parse reward_debug.jsonl to extract detailed episode data.

    Returns:
        tuple: (episodes, rewards, metrics_dict, fluent_violations, diagnostic_data)
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

    # Diagnostic data for new plots
    diagnostic_data = {
        "is_gibberish": [],  # Boolean per episode
        "perplexity": [],  # Perplexity value per episode
        "fluency_penalty": [],  # Fluency penalty per episode
        "is_fluent": [],  # Boolean per episode
        "is_successful_attack": [],  # Boolean per episode
        "got_fluency_bonus": [],  # Boolean per episode
        "base_reward": [],  # Base reward before penalties
        # New: reward decay and honeypot tracking
        "episode_num": [],  # Episode number from log
        "decay_factor": [],  # Decay factor applied (None if no decay)
        "new_honeypot_accessed": [],  # Honeypot ID accessed (None if none)
        # Text fields for evaluation metrics
        "red_team_input": [],  # Red team prompt
        "ground_truth": [],  # Ground truth SQL
        "victim_response": [],  # Victim LLM response
    }

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
                fluency = data.get("fluency_details", {})

                # Extract diagnostic metrics
                is_gibberish = (
                    fluency.get("heuristic_gibberish", False) if fluency else False
                )
                perplexity = fluency.get("perplexity", 0.0) if fluency else 0.0
                fluency_penalty = fluency.get("penalty", 0.0) if fluency else 0.0

                # New metrics added in the latest update
                is_fluent = data.get(
                    "is_fluent", not is_gibberish and perplexity <= 500
                )
                is_successful = data.get(
                    "is_successful_attack",
                    tier in ["permission_violation", "honeypot_access"],
                )
                got_bonus = data.get("got_fluency_bonus", False)
                base_reward = data.get("base_reward", data.get("final_reward", 0.0))

                diagnostic_data["is_gibberish"].append(is_gibberish)
                diagnostic_data["perplexity"].append(perplexity)
                diagnostic_data["fluency_penalty"].append(fluency_penalty)
                diagnostic_data["is_fluent"].append(is_fluent)
                diagnostic_data["is_successful_attack"].append(is_successful)
                diagnostic_data["got_fluency_bonus"].append(got_bonus)
                diagnostic_data["base_reward"].append(base_reward)

                # New: decay and honeypot tracking
                diagnostic_data["episode_num"].append(data.get("episode", ep_num))
                diagnostic_data["decay_factor"].append(data.get("decay_factor"))
                diagnostic_data["new_honeypot_accessed"].append(
                    data.get("new_honeypot_accessed")
                )

                # Text fields for evaluation metrics
                diagnostic_data["red_team_input"].append(data.get("red_team_input", ""))
                diagnostic_data["ground_truth"].append(data.get("ground_truth", ""))
                diagnostic_data["victim_response"].append(
                    data.get("victim_full_response", "")
                )

                # Determine specific category
                category = "no_sql"

                if tier == "permission_violation":
                    category = "permission_violation"
                    # Check for fluency
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

    return episodes, rewards, counts, fluent_violations, diagnostic_data


def compute_rolling_average(data, window=50):
    """Compute rolling average with given window size."""
    if len(data) < window:
        window = max(1, len(data) // 2)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def compute_eval_metrics(diagnostic_data: dict, sample_size: int = 100) -> dict:
    """
    Compute evaluation metrics (BLEU, BERTScore) from logged text data.

    Uses sampling to avoid excessive computation time.
    Compares red_team_input against ground_truth SQL.

    Args:
        diagnostic_data: Dict containing red_team_input, ground_truth, etc.
        sample_size: Number of episodes to sample for metric computation

    Returns:
        Dict with computed metrics
    """
    results = {
        "bleu_scores": [],
        "bert_f1_scores": [],
        "computed_count": 0,
    }

    red_team_inputs = diagnostic_data.get("red_team_input", [])
    ground_truths = diagnostic_data.get("ground_truth", [])

    if not red_team_inputs or not ground_truths:
        print("[EVAL] No text data available for evaluation metrics")
        return results

    # Sample indices for computation
    total = len(red_team_inputs)
    if total <= sample_size:
        indices = list(range(total))
    else:
        # Sample evenly across the run
        indices = list(np.linspace(0, total - 1, sample_size, dtype=int))

    # Compute BLEU scores
    if HAVE_BLEU:
        print(f"[EVAL] Computing BLEU scores for {len(indices)} samples...")
        for idx in indices:
            candidate = red_team_inputs[idx] or ""
            reference = ground_truths[idx] or ""
            if candidate and reference:
                try:
                    score = bleu_score(candidate, reference)
                    results["bleu_scores"].append(score.get("bleu", 0.0))
                except Exception:
                    results["bleu_scores"].append(0.0)
            else:
                results["bleu_scores"].append(0.0)

    # Compute BERTScore (batch for efficiency)
    if HAVE_BERTSCORE:
        candidates = [red_team_inputs[i] or "" for i in indices]
        references = [ground_truths[i] or "" for i in indices]

        # Filter out empty pairs
        valid_pairs = [(c, r) for c, r in zip(candidates, references) if c and r]

        if valid_pairs:
            print(f"[EVAL] Computing BERTScore for {len(valid_pairs)} valid pairs...")
            try:
                cands, refs = zip(*valid_pairs)
                P, R, F1 = bert_score_fn(
                    list(cands), list(refs), lang="en", verbose=False
                )
                results["bert_f1_scores"] = F1.tolist()
            except Exception as e:
                print(f"[EVAL] BERTScore computation failed: {e}")

    results["computed_count"] = len(indices)
    return results


def plot_training_results(run_dir: str) -> None:
    # Validate directory
    if not os.path.exists(run_dir):
        print(f"Error: Directory not found: {run_dir}")
        sys.exit(1)

    # Try parsing detailed debug logs first
    data = parse_debug_logs(run_dir)

    if data:
        episodes, rewards, counts, fluent_violations, diagnostic_data = data
        cumulative_rewards = np.cumsum(rewards)
        using_debug_logs = True
    else:
        # Fallback to Tensorboard (Legacy)
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
        diagnostic_data = None
        using_debug_logs = False

    # === Plotting ===
    # TODO: Review for later tweaking (01/14/26) - Added 4-panel diagnostic visualization
    if using_debug_logs and diagnostic_data:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        ax1, ax2, ax3, ax4 = axes.flatten()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        ax3, ax4 = None, None

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

    # Plot "Permission Violation + Fluent" as green dots
    if fluent_violations:
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

    # --- Plot 3: Gibberish Rate (KEY WARNING SIGN) ---
    if ax3 is not None and diagnostic_data:
        # Compute rolling gibberish rate
        gibberish_array = np.array(diagnostic_data["is_gibberish"], dtype=float)
        window = 50
        rolling_gibberish_rate = compute_rolling_average(gibberish_array, window) * 100

        # X-axis for rolling average (centered)
        rolling_x = np.arange(window // 2, len(episodes) - window // 2 + 1)
        if len(rolling_x) > len(rolling_gibberish_rate):
            rolling_x = rolling_x[: len(rolling_gibberish_rate)]
        elif len(rolling_gibberish_rate) > len(rolling_x):
            rolling_gibberish_rate = rolling_gibberish_rate[: len(rolling_x)]

        ax3.plot(
            rolling_x,
            rolling_gibberish_rate,
            color="#e74c3c",
            linewidth=2,
            label=f"Gibberish Rate ({window}-ep rolling avg)",
        )

        # Add danger zone shading
        ax3.axhspan(50, 100, alpha=0.2, color="red", label="Danger Zone")
        ax3.axhspan(25, 50, alpha=0.1, color="orange", label="Warning Zone")

        # Also plot fluent rate
        fluent_array = np.array(diagnostic_data["is_fluent"], dtype=float)
        rolling_fluent_rate = compute_rolling_average(fluent_array, window) * 100
        ax3.plot(
            rolling_x,
            rolling_fluent_rate[: len(rolling_x)],
            color="#27ae60",
            linewidth=2,
            linestyle="--",
            label=f"Fluent Rate ({window}-ep rolling avg)",
        )

        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Percentage (%)")
        ax3.set_title("⚠️ Gibberish vs Fluent Rate (Mode Collapse Indicator)")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

    # --- Plot 4: Perplexity and Fluency Penalty ---
    if ax4 is not None and diagnostic_data:
        perplexity = np.array(diagnostic_data["perplexity"])
        fluency_penalty = np.array(diagnostic_data["fluency_penalty"])

        # Clip perplexity for visualization (cap at 2000 for readability)
        perplexity_clipped = np.clip(perplexity, 0, 2000)

        window = 50
        rolling_ppl = compute_rolling_average(perplexity_clipped, window)
        rolling_penalty = compute_rolling_average(fluency_penalty, window)

        rolling_x = np.arange(window // 2, len(episodes) - window // 2 + 1)
        if len(rolling_x) > len(rolling_ppl):
            rolling_x = rolling_x[: len(rolling_ppl)]

        # Perplexity on primary y-axis
        ax4.plot(
            rolling_x,
            rolling_ppl[: len(rolling_x)],
            color="#9b59b6",
            linewidth=2,
            label=f"Perplexity ({window}-ep avg, capped at 2000)",
        )
        ax4.axhline(
            y=500,
            color="#9b59b6",
            linestyle=":",
            alpha=0.7,
            label="PPL Threshold (500)",
        )
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Perplexity", color="#9b59b6")
        ax4.tick_params(axis="y", labelcolor="#9b59b6")

        # Fluency penalty on secondary y-axis
        ax4_twin = ax4.twinx()
        ax4_twin.plot(
            rolling_x,
            rolling_penalty[: len(rolling_x)],
            color="#e74c3c",
            linewidth=2,
            linestyle="--",
            label=f"Avg Fluency Penalty ({window}-ep)",
        )
        ax4_twin.set_ylabel("Fluency Penalty", color="#e74c3c")
        ax4_twin.tick_params(axis="y", labelcolor="#e74c3c")

        ax4.set_title("Perplexity and Fluency Penalty Trends")
        ax4.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    output_path = os.path.join(run_dir, "training_results_detailed.png")
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Print Summary
    print("\n=== Summary Statistics ===")
    print(f"Total Episodes: {len(episodes)}")

    if using_debug_logs:
        labels = {
            "permission_violation": "Permission Violation",
            "honeypot_access": "Honeypot Access",
            "sensitive_query": "Sensitive Query",
            "sql_generated": "Valid SQL Generated",
            "sql_error": "SQL Error",
            "no_sql": "No SQL / Refusal",
        }
        for cat, data_points in counts.items():
            if data_points:
                final_count = data_points[-1]
                pct = (final_count / len(episodes)) * 100
                print(f"{labels.get(cat, cat):<25}: {final_count:>4} ({pct:.1f}%)")
        print(f"{'Fluent Perm. Violations':<25}: {len(fluent_violations):>4}")

        if diagnostic_data:
            # Diagnostic summary
            total_gibberish = sum(diagnostic_data["is_gibberish"])
            total_fluent_bonus = sum(diagnostic_data["got_fluency_bonus"])
            avg_ppl = np.mean([p for p in diagnostic_data["perplexity"] if p > 0])
            avg_penalty = np.mean(diagnostic_data["fluency_penalty"])

            print("\n=== Diagnostic Metrics ===")
            print(
                f"{'Total Gibberish Episodes':<25}: {total_gibberish:>4} ({100 * total_gibberish / len(episodes):.1f}%)"
            )
            print(f"{'Got Fluency Bonus':<25}: {total_fluent_bonus:>4}")
            print(f"{'Average Perplexity':<25}: {avg_ppl:.1f}")
            print(f"{'Average Fluency Penalty':<25}: {avg_penalty:.2f}")

            # Early vs Late comparison (warning sign detection)
            mid = len(episodes) // 2
            if mid > 0:
                early_gibberish = sum(diagnostic_data["is_gibberish"][:mid]) / mid
                late_gibberish = sum(diagnostic_data["is_gibberish"][mid:]) / (
                    len(episodes) - mid
                )

                print("\n=== Trend Analysis ===")
                print(f"Early Gibberish Rate (0-{mid}): {100 * early_gibberish:.1f}%")
                print(f"Late Gibberish Rate ({mid}+):   {100 * late_gibberish:.1f}%")

                if late_gibberish > early_gibberish * 2 and late_gibberish > 0.5:
                    print(
                        "⚠️  WARNING: Gibberish rate increased significantly - possible mode collapse!"
                    )
                elif late_gibberish < early_gibberish * 0.5:
                    print(
                        "✅ Good: Gibberish rate decreased - model is learning fluent attacks"
                    )

            # Decay and Honeypot Summary
            decay_factors = [
                d for d in diagnostic_data["decay_factor"] if d is not None
            ]
            honeypots_accessed = [
                h for h in diagnostic_data["new_honeypot_accessed"] if h is not None
            ]

            if decay_factors or honeypots_accessed:
                print("\n=== Reward Decay & Honeypot Tracking ===")

                if decay_factors:
                    avg_decay = np.mean(decay_factors)
                    min_decay = min(decay_factors)
                    print(f"{'Avg Decay Factor':<25}: {avg_decay:.4f}")
                    print(f"{'Min Decay Factor':<25}: {min_decay:.4f}")

                if honeypots_accessed:
                    unique_honeypots = set(honeypots_accessed)
                    print(f"{'Honeypots Accessed':<25}: {len(honeypots_accessed):>4}")
                    print(f"{'Unique Honeypots':<25}: {len(unique_honeypots):>4}")
                    for hp in sorted(unique_honeypots):
                        hp_count = honeypots_accessed.count(hp)
                        print(f"  - {hp}: {hp_count} time(s)")

            # Evaluation Metrics (BLEU, BERTScore)
            print("\n=== Evaluation Metrics ===")
            if not HAVE_BLEU and not HAVE_BERTSCORE:
                print(
                    "(Install bert-score and ensure calc_bleu.py is available for metrics)"
                )
            else:
                eval_results = compute_eval_metrics(diagnostic_data, sample_size=100)

                if eval_results["bleu_scores"]:
                    avg_bleu = np.mean(eval_results["bleu_scores"])
                    max_bleu = max(eval_results["bleu_scores"])
                    print(f"{'Avg BLEU Score':<25}: {avg_bleu:.4f}")
                    print(f"{'Max BLEU Score':<25}: {max_bleu:.4f}")

                if eval_results["bert_f1_scores"]:
                    avg_bert = np.mean(eval_results["bert_f1_scores"])
                    max_bert = max(eval_results["bert_f1_scores"])
                    print(f"{'Avg BERTScore F1':<25}: {avg_bert:.4f}")
                    print(f"{'Max BERTScore F1':<25}: {max_bert:.4f}")

                if eval_results["computed_count"] > 0:
                    print(
                        f"{'Samples Evaluated':<25}: {eval_results['computed_count']:>4}"
                    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    plot_training_results(sys.argv[1])


if __name__ == "__main__":
    main()
