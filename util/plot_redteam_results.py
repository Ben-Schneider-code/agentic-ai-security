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
import math
from collections import defaultdict, Counter
import yaml


def compute_tfidf_cosine_similarity(texts):
    if len(texts) < 2:
        return 0.0

    def tokenize(text):
        if not text:
            return []
        return text.lower().split()

    tokenized_texts = [tokenize(t) for t in texts]
    df = Counter()
    for tokens in tokenized_texts:
        df.update(set(tokens))

    N = len(texts)
    vectors = []
    for tokens in tokenized_texts:
        counter = Counter(tokens)
        vec = {}
        for token, count in counter.items():
            tf = count / len(tokens) if tokens else 0
            idf = math.log(N / (1 + df[token])) if df[token] > 0 else 0
            vec[token] = tf * idf
        vectors.append(vec)

    similarities = []
    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = vectors[i], vectors[j]
            dot_product = sum(v1.get(k, 0) * v2.get(k, 0) for k in v1)
            mag1 = math.sqrt(sum(v**2 for v in v1.values()))
            mag2 = math.sqrt(sum(v**2 for v in v2.values()))
            if mag1 > 0 and mag2 > 0:
                sim = dot_product / (mag1 * mag2)
            else:
                sim = 0.0
            similarities.append(sim)

    if not similarities:
        return 0.0
    return sum(similarities) / len(similarities)


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
        "repeated_honeypot",
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
        # New: degenerate input detection
        "is_degenerate": [],  # Boolean per episode
        "degenerate_penalty": [],  # Degenerate penalty per episode
        "degenerate_reason": [],  # Reason for degenerate detection
        "in_warmup_period": [],  # Whether episode was in warmup
        # New: reward decay and honeypot tracking
        "episode_num": [],  # Episode number from log
        "decay_factor": [],  # Decay factor applied (None if no decay)
        "new_honeypots_accessed": [],  # List of Honeypot IDs accessed
        "attempted_honeypots": [],  # NEW: Attempted but failed execution
        # Text fields for evaluation metrics
        "red_team_input": [],  # Red team prompt
        "ground_truth": [],  # Ground truth SQL
        "victim_response": [],  # Victim LLM response
        "execution_result": [],
        "step_count": [],
        # New: Token tracking
        "red_team_tokens": [],
        "victim_tokens": [],
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

                # New: degenerate input detection
                is_degenerate = data.get("is_degenerate", False)
                degenerate_penalty = data.get("degenerate_penalty", 0.0)
                degenerate_reason = data.get("degenerate_reason", None)
                in_warmup = data.get("in_warmup_period", True)
                diagnostic_data["is_degenerate"].append(is_degenerate)
                diagnostic_data["degenerate_penalty"].append(degenerate_penalty)
                diagnostic_data["degenerate_reason"].append(degenerate_reason)
                diagnostic_data["in_warmup_period"].append(in_warmup)

                # New: decay and honeypot tracking
                diagnostic_data["episode_num"].append(data.get("episode", ep_num))
                diagnostic_data["decay_factor"].append(data.get("decay_factor"))

                # Check for new list format first
                new_hps = data.get("new_honeypots_accessed")
                if new_hps is None:
                    # Fallback to old format
                    old_hp = data.get("new_honeypot_accessed")
                    if old_hp:
                        new_hps = [old_hp]
                    else:
                        new_hps = []

                diagnostic_data["new_honeypots_accessed"].append(new_hps)

                # NEW: Track attempted but failed honeypots
                attempted_hps = breakdown.get("attempted_but_failed_honeypots", [])
                diagnostic_data["attempted_honeypots"].append(attempted_hps)

                # Text fields for evaluation metrics
                diagnostic_data["red_team_input"].append(data.get("red_team_input", ""))
                diagnostic_data["ground_truth"].append(data.get("ground_truth", ""))
                diagnostic_data["victim_response"].append(
                    data.get("victim_full_response", "")
                )
                diagnostic_data["execution_result"].append(
                    data.get("execution_result", "")
                )
                diagnostic_data["step_count"].append(data.get("step_count", 1))
                # Estimate token counts (approx words * 1.3)
                rt_input = data.get("red_team_input", "")
                v_response = data.get("victim_full_response", "")
                rt_tokens = len(rt_input.split()) * 1.3 if rt_input else 0
                v_tokens = len(v_response.split()) * 1.3 if v_response else 0
                diagnostic_data["red_team_tokens"].append(rt_tokens)
                diagnostic_data["victim_tokens"].append(v_tokens)

                # Determine specific category
                category = "no_sql"

                if tier == "permission_violation":
                    category = "permission_violation"
                    # Check for fluency
                    if is_fluent and not is_gibberish:
                        fluent_violations.append(ep_num)

                elif tier == "honeypot_access":
                    category = "honeypot_access"
                elif tier == "repeated_honeypot":
                    category = "repeated_honeypot"
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

    unique_episodes = sorted(list(set(diagnostic_data["episode_num"])))
    diagnostic_data["ttr_per_episode"] = []
    diagnostic_data["variance_per_episode"] = []
    diagnostic_data["turns_to_success_per_episode"] = []

    ep_grouped = defaultdict(list)
    for idx, ep in enumerate(diagnostic_data["episode_num"]):
        ep_grouped[ep].append(idx)

    for ep in unique_episodes:
        indices = ep_grouped[ep]
        tool_calls = sum(
            1 for idx in indices if diagnostic_data["execution_result"][idx]
        )
        ttr = (tool_calls / len(indices)) * 100 if indices else 0.0
        diagnostic_data["ttr_per_episode"].append(ttr)

        prompts = [diagnostic_data["red_team_input"][idx] for idx in indices]
        sim = compute_tfidf_cosine_similarity(prompts)
        diagnostic_data["variance_per_episode"].append(sim)

        turns_to_success = None
        for idx in indices:
            hps = diagnostic_data["new_honeypots_accessed"][idx]
            if hps and any(hps):
                turns_to_success = diagnostic_data["step_count"][idx]
                break
        diagnostic_data["turns_to_success_per_episode"].append(turns_to_success)

    diagnostic_data["unique_episodes"] = unique_episodes

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

    # Read exit reason if available
    exit_reason = None
    exit_reason_path = os.path.join(run_dir, "exit_reason.txt")
    if os.path.exists(exit_reason_path):
        with open(exit_reason_path, "r") as f:
            exit_reason = f.read().strip()
        print(f"Exit reason: {exit_reason}")

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
    if using_debug_logs and diagnostic_data:
        # Increased height for 6 rows
        fig, axes = plt.subplots(6, 2, figsize=(16, 38))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes.flatten()

        ax_rewards = ax1
        ax_cum_counts = ax2
        ax_cum_pct = ax3
        ax_gibberish = ax4
        ax_ppl = ax5
        ax_honeypot_cumulative = ax6
        ax_honeypot_bar = ax7
        ax_honeypot_timeline = ax8
        ax_tokens = ax9
        ax_efficiency = ax10
        ax_ttr = ax11
        ax_hdr_turns = ax12
    else:
        # Fallback layout
        fig, axes = plt.subplots(6, 2, figsize=(16, 38))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes.flatten()
        ax_rewards = ax1
        ax_cum_counts = ax2
        ax_cum_pct, ax_gibberish, ax_ppl = None, None, None
        ax_honeypot_cumulative, ax_honeypot_bar, ax_honeypot_timeline = None, None, None
        ax_tokens, ax_efficiency = None, None
        ax_ttr, ax_hdr_turns = None, None
        for i in range(2, 12):
            fig.delaxes(axes[i])

    # --- Plot 1: Rewards ---
    ax_rewards.plot(
        episodes,
        rewards,
        color="#3498db",
        alpha=0.6,
        label="Reward per Episode",
        linewidth=1,
    )
    ax_rewards.plot(
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
        ax_rewards.scatter(
            fluent_violations,
            y_pos,
            color="#2ecc71",
            marker="o",
            s=60,
            label="Fluent Perm. Violation",
            zorder=10,
            edgecolor="black",
        )

    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Reward")
    ax_rewards.set_title("Training Rewards and Successes")
    ax_rewards.legend(loc="upper left")
    ax_rewards.grid(True, alpha=0.3)

    # Define colors and labels for outcome plots
    colors = {
        "permission_violation": "#e74c3c",  # Red
        "honeypot_access": "#9b59b6",  # Purple
        "repeated_honeypot": "#f1c40f",  # Yellow/Gold
        "sensitive_query": "#f39c12",  # Orange
        "sql_generated": "#3498db",  # Blue
        "sql_error": "#95a5a6",  # Gray
        "no_sql": "#bdc3c7",  # Light Gray
    }

    labels = {
        "permission_violation": "Permission Violation",
        "honeypot_access": "Honeypot Access",
        "repeated_honeypot": "Repeated Honeypot",
        "sensitive_query": "Sensitive Query",
        "sql_generated": "Valid SQL Generated",
        "sql_error": "SQL Error",
        "no_sql": "No SQL / Refusal",
    }

    # Priority order for stacking (bottom to top)
    # Put most important/rare 'successes' at the bottom or top?
    # Usually "No SQL" is the bulk base, so putting it at bottom might be good,
    # but often we want the 'interesting' stuff to be visible.
    # Let's align with the order used in 'categories' list for consistency
    categories_order = [
        "no_sql",
        "sql_error",
        "sql_generated",
        "sensitive_query",
        "repeated_honeypot",
        "honeypot_access",
        "permission_violation",
    ]

    # --- Plot 2: Cumulative Outcomes (Counts) ---
    if using_debug_logs:
        for cat in categories_order:  # Use consistent order
            # Check if cat is in counts (it should be)
            if cat in counts:
                data_points = counts[cat]
                if data_points and data_points[-1] > 0:
                    ax_cum_counts.plot(
                        episodes,
                        data_points,
                        color=colors.get(cat, "black"),
                        label=labels.get(cat, cat),
                        linewidth=2,
                    )
    else:
        ax_cum_counts.text(
            0.5,
            0.5,
            "Detailed outcome metrics require reward_debug.jsonl",
            ha="center",
            va="center",
        )

    ax_cum_counts.set_xlabel("Episode")
    ax_cum_counts.set_ylabel("Cumulative Count")
    ax_cum_counts.set_title("Cumulative Outcomes by Category")
    ax_cum_counts.legend(loc="upper left")
    ax_cum_counts.grid(True, alpha=0.3)

    # --- Plot 3: Cumulative Outcomes (Percentage Stackplot) ---
    if using_debug_logs and ax_cum_pct:
        stack_y = []
        stack_labels_list = []
        stack_colors_list = []

        # Ensure we have data for all episodes
        # counts[cat] is cumulative count.
        # But we need to ensure we iterate in a stable order.
        # Use categories_order

        for cat in categories_order:
            if cat in counts:
                data = np.array(counts[cat])
                # Safety check on length
                if len(data) == len(episodes):
                    # Calculate percentage: count / episode_index
                    # episodes are 1-based, so dividing by np.array(episodes) works
                    pcts = (data / np.array(episodes)) * 100

                    stack_y.append(pcts)
                    stack_labels_list.append(labels.get(cat, cat))
                    stack_colors_list.append(colors.get(cat, "gray"))

        if stack_y:
            try:
                ax_cum_pct.stackplot(
                    episodes,
                    *stack_y,
                    labels=stack_labels_list,
                    colors=stack_colors_list,
                    alpha=0.8,
                )
            except Exception as e:
                print(f"Error creating stackplot: {e}")

            ax_cum_pct.set_xlabel("Episode")
            ax_cum_pct.set_ylabel("Percentage (%)")
            ax_cum_pct.set_title("Cumulative Outcome Distribution (%)")
            # Move legend outside or make it small, as stackplots can get crowded
            ax_cum_pct.legend(loc="lower left", fontsize="x-small")
            ax_cum_pct.set_ylim(0, 100)
            ax_cum_pct.grid(True, alpha=0.3)
            # Add margins to eliminate whitespace on sides
            ax_cum_pct.set_xlim(min(episodes), max(episodes))

    # --- Plot 4: Gibberish & Degenerate Rate (KEY WARNING SIGN) ---
    if ax_gibberish is not None and diagnostic_data:
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

        ax_gibberish.plot(
            rolling_x,
            rolling_gibberish_rate,
            color="#e74c3c",
            linewidth=2,
            label=f"Gibberish Rate ({window}-ep)",
        )

        # Add danger zone shading
        ax_gibberish.axhspan(50, 100, alpha=0.2, color="red", label="Danger Zone")
        ax_gibberish.axhspan(25, 50, alpha=0.1, color="orange", label="Warning Zone")

        # Also plot fluent rate
        fluent_array = np.array(diagnostic_data["is_fluent"], dtype=float)
        rolling_fluent_rate = compute_rolling_average(fluent_array, window) * 100
        ax_gibberish.plot(
            rolling_x,
            rolling_fluent_rate[: len(rolling_x)],
            color="#27ae60",
            linewidth=2,
            linestyle="--",
            label=f"Fluent Rate ({window}-ep)",
        )

        # NEW: Plot degenerate input rate
        degenerate_array = np.array(diagnostic_data["is_degenerate"], dtype=float)
        rolling_degenerate_rate = (
            compute_rolling_average(degenerate_array, window) * 100
        )
        ax_gibberish.plot(
            rolling_x,
            rolling_degenerate_rate[: len(rolling_x)],
            color="#9b59b6",
            linewidth=2,
            linestyle=":",
            label=f"Degenerate Rate ({window}-ep)",
        )

        ax_gibberish.set_xlabel("Episode")
        ax_gibberish.set_ylabel("Percentage (%)")
        ax_gibberish.set_title("Input Quality (Gibberish/Degenerate = Bad)")
        ax_gibberish.legend(loc="upper right", fontsize="small")
        ax_gibberish.grid(True, alpha=0.3)
        ax_gibberish.set_ylim(0, 100)

    # --- Plot 5: Perplexity and Fluency Penalty ---
    if ax_ppl is not None and diagnostic_data:
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
        ax_ppl.plot(
            rolling_x,
            rolling_ppl[: len(rolling_x)],
            color="#9b59b6",
            linewidth=2,
            label=f"Perplexity ({window}-ep avg, capped at 2000)",
        )
        ax_ppl.axhline(
            y=500,
            color="#9b59b6",
            linestyle=":",
            alpha=0.7,
            label="PPL Threshold (500)",
        )
        ax_ppl.set_xlabel("Episode")
        ax_ppl.set_ylabel("Perplexity", color="#9b59b6")
        ax_ppl.tick_params(axis="y", labelcolor="#9b59b6")

        # Fluency penalty on secondary y-axis
        ax_ppl_twin = ax_ppl.twinx()
        ax_ppl_twin.plot(
            rolling_x,
            rolling_penalty[: len(rolling_x)],
            color="#e74c3c",
            linewidth=2,
            linestyle="--",
            label=f"Avg Fluency Penalty ({window}-ep)",
        )
        ax_ppl_twin.set_ylabel("Fluency Penalty", color="#e74c3c")
        ax_ppl_twin.tick_params(axis="y", labelcolor="#e74c3c")

        ax_ppl.set_title("Perplexity and Fluency Penalty Trends")
        ax_ppl.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax_ppl.get_legend_handles_labels()
        lines2, labels2 = ax_ppl_twin.get_legend_handles_labels()
        ax_ppl.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # --- Plot 6: Cumulative Honeypots Captured ---
    if ax_honeypot_cumulative is not None and diagnostic_data:
        honeypots_list = diagnostic_data.get("new_honeypots_accessed", [])

        # Try to load total honeypots from saved reward_config.yaml first
        TOTAL_HONEYPOTS = None
        reward_config_path = os.path.join(run_dir, "reward_config.yaml")
        if os.path.exists(reward_config_path):
            try:
                with open(reward_config_path, "r") as f:
                    config = yaml.unsafe_load(f)
                    if "total_honeypots" in config:
                        TOTAL_HONEYPOTS = config["total_honeypots"]
                        print(
                            f"Loaded TOTAL_HONEYPOTS={TOTAL_HONEYPOTS} from reward_config.yaml"
                        )
            except Exception as e:
                print(f"Failed to load reward_config.yaml: {e}")

        if TOTAL_HONEYPOTS is None:
            # Use the shared get_total_honeypots() function for consistency with training runner
            try:
                from MARFT.marft.envs.redteam_sql.redteam_sql_env import (
                    get_total_honeypots,
                )

                TOTAL_HONEYPOTS = get_total_honeypots()
            except ImportError:
                # Fallback: use unique honeypots discovered as the known total
                # This gives a "coverage of discovered" rather than "coverage of all"
                unique_honeypots = set()
                for hps in honeypots_list:
                    if hps:
                        for hp in hps:
                            if hp:
                                unique_honeypots.add(hp)

                TOTAL_HONEYPOTS = max(
                    len(unique_honeypots), 1
                )  # Avoid division by zero

        # Build cumulative tracking data
        cumulative_total_accesses = []
        cumulative_unique_honeypots = []
        seen_honeypots = set()
        total_accesses = 0

        for hps in honeypots_list:
            if hps:
                for hp in hps:
                    if hp is not None:
                        total_accesses += 1
                        seen_honeypots.add(hp)
            cumulative_total_accesses.append(total_accesses)
            cumulative_unique_honeypots.append(len(seen_honeypots))

        if any(cumulative_total_accesses):
            ax_honeypot_cumulative.plot(
                episodes,
                cumulative_total_accesses,
                color="#9b59b6",
                linewidth=2,
                label="Total Honeypot Accesses",
            )
            ax_honeypot_cumulative.plot(
                episodes,
                cumulative_unique_honeypots,
                color="#e74c3c",
                linewidth=2,
                linestyle="--",
                label="Unique Honeypots Discovered",
            )

            # Add milestone markers for each new unique honeypot
            discovery_episodes = []
            discovery_honeypots = []
            seen_for_markers = set()
            for i, hps in enumerate(honeypots_list):
                if hps:
                    for hp in hps:
                        if hp is not None and hp not in seen_for_markers:
                            seen_for_markers.add(hp)
                            discovery_episodes.append(i + 1)
                            discovery_honeypots.append(len(seen_for_markers))

            if discovery_episodes:
                ax_honeypot_cumulative.scatter(
                    discovery_episodes,
                    discovery_honeypots,
                    color="#2ecc71",
                    marker="*",
                    s=100,
                    zorder=10,
                    edgecolor="black",
                    label="New Honeypot Discovered",
                )

            ax_honeypot_cumulative.set_xlabel("Episode")
            ax_honeypot_cumulative.set_ylabel("Count")
            ax_honeypot_cumulative.set_title("Cumulative Honeypots Captured")
            ax_honeypot_cumulative.legend(loc="upper left")
            ax_honeypot_cumulative.grid(True, alpha=0.3)

            # Add secondary y-axis showing percentage of total honeypots
            ax_honeypot_pct = ax_honeypot_cumulative.twinx()
            cumulative_pct = [
                (count / TOTAL_HONEYPOTS) * 100 for count in cumulative_unique_honeypots
            ]
            ax_honeypot_pct.plot(
                episodes,
                cumulative_pct,
                color="#27ae60",
                linewidth=2,
                linestyle=":",
                alpha=0.7,
                label="Coverage %",
            )
            ax_honeypot_pct.set_ylabel("Coverage (%)", color="#27ae60")
            ax_honeypot_pct.tick_params(axis="y", labelcolor="#27ae60")
            ax_honeypot_pct.set_ylim(0, 100)

            # Add horizontal line at 100% coverage
            ax_honeypot_pct.axhline(y=100, color="#27ae60", linestyle="--", alpha=0.3)

            # Add prominent coverage percentage annotation
            final_unique = (
                cumulative_unique_honeypots[-1] if cumulative_unique_honeypots else 0
            )
            final_pct = (final_unique / TOTAL_HONEYPOTS) * 100
            ax_honeypot_cumulative.text(
                0.98,
                0.85,
                f"Coverage: {final_pct:.1f}%\n({final_unique}/{TOTAL_HONEYPOTS} honeypots)",
                transform=ax_honeypot_cumulative.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round",
                    facecolor="lightgreen",
                    alpha=0.8,
                    edgecolor="green",
                ),
            )
        else:
            ax_honeypot_cumulative.text(
                0.5,
                0.5,
                "No honeypots captured yet",
                ha="center",
                va="center",
                fontsize=14,
                color="gray",
            )
            ax_honeypot_cumulative.set_title("Cumulative Honeypots Captured")
            ax_honeypot_cumulative.axis("off")

    # --- Plot 7: Honeypot Access Frequency Bar Chart ---
    if ax_honeypot_bar is not None and diagnostic_data:
        honeypots_list = diagnostic_data.get("new_honeypots_accessed", [])

        # Flatten list
        all_hps = []
        for hps in honeypots_list:
            if hps:
                for hp in hps:
                    if hp:
                        all_hps.append(hp)

        if all_hps:
            from collections import Counter

            hp_counts = Counter(all_hps)

            # Sort by count
            sorted_hps = sorted(hp_counts.items(), key=lambda x: x[1], reverse=True)
            top_hps = sorted_hps[:15]  # Top 15

            names = [x[0] for x in top_hps]
            access_counts = [x[1] for x in top_hps]

            y_pos = np.arange(len(names))

            ax_honeypot_bar.barh(y_pos, access_counts, align="center", color="#3498db")
            ax_honeypot_bar.set_yticks(y_pos)
            ax_honeypot_bar.set_yticklabels(names)
            ax_honeypot_bar.invert_yaxis()  # labels read top-to-bottom
            ax_honeypot_bar.set_xlabel("Access Count")
            ax_honeypot_bar.set_title("Top Accessed Honeypots")
        else:
            ax_honeypot_bar.text(
                0.5,
                0.5,
                "No honeypots accessed",
                ha="center",
                va="center",
                color="gray",
            )
            ax_honeypot_bar.axis("off")

    # --- Plot 8: Honeypot Discovery Timeline ---
    if ax_honeypot_timeline is not None and diagnostic_data:
        # Same logic as scatter plot in Plot 6 but focused on timeline
        cumulative_unique_honeypots = []
        cumulative_unique_attempts = []
        seen_honeypots = set()
        seen_attempts = set()

        honeypots_list = diagnostic_data.get("new_honeypots_accessed", [])
        attempted_list = diagnostic_data.get("attempted_honeypots", [])

        for i in range(len(episodes)):
            hps = honeypots_list[i] if i < len(honeypots_list) else []
            if hps:
                for hp in hps:
                    if hp:
                        seen_honeypots.add(hp)
            cumulative_unique_honeypots.append(len(seen_honeypots))

            att = attempted_list[i] if i < len(attempted_list) else []
            if att:
                for a in att:
                    if a:
                        seen_attempts.add(a)

            # Total unique honeypots targeted (whether successful or not)
            targeted = seen_attempts.union(seen_honeypots)
            cumulative_unique_attempts.append(len(targeted))

        ax_honeypot_timeline.plot(
            episodes,
            cumulative_unique_attempts,
            color="#95a5a6",
            linewidth=2,
            linestyle="--",
            label="Targeted (Attempted or Accessed)",
        )

        ax_honeypot_timeline.plot(
            episodes,
            cumulative_unique_honeypots,
            color="#2ecc71",
            linewidth=2,
            label="Successfully Accessed",
        )

        ax_honeypot_timeline.set_xlabel("Episode")
        ax_honeypot_timeline.set_ylabel("Unique Honeypots")
        ax_honeypot_timeline.set_title("Honeypots Targeted vs Actual Extraction")
        ax_honeypot_timeline.legend(loc="upper left")
        ax_honeypot_timeline.grid(True, alpha=0.3)

    # --- Plot 9: Token Usage Trends ---
    if ax_tokens is not None and diagnostic_data:
        rt_tokens = np.array(diagnostic_data.get("red_team_tokens", []))
        v_tokens = np.array(diagnostic_data.get("victim_tokens", []))

        window = 50
        rolling_rt = compute_rolling_average(rt_tokens, window)
        rolling_v = compute_rolling_average(v_tokens, window)

        # X-axis alignment
        rolling_x = np.arange(window // 2, len(episodes) - window // 2 + 1)
        if len(rolling_x) > len(rolling_rt):
            rolling_x = rolling_x[: len(rolling_rt)]

        ax_tokens.plot(
            rolling_x,
            rolling_rt[: len(rolling_x)],
            color="#c0392b",
            label=f"Red Team Tokens ({window}-ep avg)",
        )
        ax_tokens.plot(
            rolling_x,
            rolling_v[: len(rolling_x)],
            color="#2980b9",
            label=f"Victim Tokens ({window}-ep avg)",
        )

        ax_tokens.set_xlabel("Episode")
        ax_tokens.set_ylabel("Estimated Tokens")
        ax_tokens.set_title("Token Usage Trends (Efficiency)")
        ax_tokens.legend()
        ax_tokens.grid(True, alpha=0.3)

    # --- Plot 10: Efficiency (Reward per Token) ---
    if ax_efficiency is not None and diagnostic_data:
        rt_tokens = np.array(diagnostic_data.get("red_team_tokens", []))
        rewards_arr = np.array(rewards)

        # Avoid division by zero
        safe_tokens = np.where(rt_tokens > 0, rt_tokens, 1)
        efficiency = rewards_arr / safe_tokens

        # Clip specifically for visualization - remove extreme outliers
        efficiency = np.clip(efficiency, -1.0, 2.0)

        window = 100
        rolling_eff = compute_rolling_average(efficiency, window)

        rolling_x = np.arange(window // 2, len(episodes) - window // 2 + 1)
        if len(rolling_x) > len(rolling_eff):
            rolling_x = rolling_x[: len(rolling_eff)]

        ax_efficiency.plot(
            rolling_x,
            rolling_eff[: len(rolling_x)],
            color="#27ae60",
            linewidth=2,
            label="Efficiency (Reward/Token)",
        )

        # Add a baseline at 0
        ax_efficiency.axhline(y=0, color="black", linestyle="-", alpha=0.2)

        ax_efficiency.set_xlabel("Episode")
        ax_efficiency.set_ylabel("Reward per Token")
        ax_efficiency.set_title(f"Red Team Efficiency ({window}-ep avg)")
        ax_efficiency.legend()
        ax_efficiency.grid(True, alpha=0.3)

    # === Plot 11: Tool-Trigger Rate (TTR) and Embedding Variance ===
    if ax_ttr is not None and diagnostic_data and "ttr_per_episode" in diagnostic_data:
        unique_eps = diagnostic_data["unique_episodes"]
        ttr_arr = np.array(diagnostic_data["ttr_per_episode"])
        var_arr = np.array(diagnostic_data["variance_per_episode"])

        window = 50
        if len(unique_eps) < window:
            window = max(1, len(unique_eps) // 2)

        rolling_ttr = compute_rolling_average(ttr_arr, window)
        rolling_var = compute_rolling_average(var_arr, window)

        rolling_eps = np.arange(window // 2, len(unique_eps) - window // 2 + 1)
        min_len = min(len(rolling_eps), len(rolling_ttr))
        rolling_eps = rolling_eps[:min_len]
        rolling_ttr = rolling_ttr[:min_len]
        rolling_var = rolling_var[:min_len]

        ax_ttr.plot(
            rolling_eps,
            rolling_ttr,
            color="#e67e22",
            linewidth=2,
            label=f"TTR % ({window}-ep avg)",
        )
        ax_ttr.set_xlabel("Episode")
        ax_ttr.set_ylabel("Tool-Trigger Rate (%)", color="#e67e22")
        ax_ttr.tick_params(axis="y", labelcolor="#e67e22")
        ax_ttr.set_title("TTR & Intra-Episode Prompt Similarity")
        ax_ttr.grid(True, alpha=0.3)
        ax_ttr.set_ylim(0, 100)

        ax_var_twin = ax_ttr.twinx()
        ax_var_twin.plot(
            rolling_eps,
            rolling_var,
            color="#8e44ad",
            linewidth=2,
            linestyle="--",
            label=f"Prompt Similarity ({window}-ep avg)",
        )
        ax_var_twin.set_ylabel("Cosine Similarity", color="#8e44ad")
        ax_var_twin.tick_params(axis="y", labelcolor="#8e44ad")
        ax_var_twin.set_ylim(0, 1.0)
        ax_var_twin.axhline(
            y=0.9, color="red", linestyle=":", alpha=0.5, label="High Repetition (>0.9)"
        )

        lines1, labels1 = ax_ttr.get_legend_handles_labels()
        lines2, labels2 = ax_var_twin.get_legend_handles_labels()
        ax_ttr.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # === Plot 12: Honeypot Discovery Rate (Turns-to-Success) ===
    if (
        ax_hdr_turns is not None
        and diagnostic_data
        and "turns_to_success_per_episode" in diagnostic_data
    ):
        unique_eps = np.array(diagnostic_data["unique_episodes"])
        turns_arr = np.array(diagnostic_data["turns_to_success_per_episode"])

        valid_idx = [i for i, v in enumerate(turns_arr) if v is not None]
        valid_eps = unique_eps[valid_idx]
        valid_turns = turns_arr[valid_idx].astype(float)

        if len(valid_turns) > 0:
            window = 10
            if len(valid_turns) > window:
                rolling_turns = compute_rolling_average(valid_turns, window)
                rolling_eps_turns = valid_eps[
                    window // 2 : len(rolling_turns) + window // 2
                ]
                min_len = min(len(rolling_eps_turns), len(rolling_turns))

                ax_hdr_turns.scatter(
                    valid_eps,
                    valid_turns,
                    alpha=0.3,
                    color="#2ecc71",
                    s=10,
                    label="Turns to Hit Honeypot",
                )
                ax_hdr_turns.plot(
                    rolling_eps_turns[:min_len],
                    rolling_turns[:min_len],
                    color="#27ae60",
                    linewidth=3,
                    label=f"Avg Turns ({window}-success avg)",
                )

                ax_hdr_turns.set_xlabel("Episode")
                ax_hdr_turns.set_ylabel("Turn Number")
                ax_hdr_turns.set_title("Honeypot Discovery Rate (Turns-to-Success)")
                ax_hdr_turns.legend()
                ax_hdr_turns.grid(True, alpha=0.3)
            else:
                ax_hdr_turns.scatter(
                    valid_eps, valid_turns, alpha=0.8, color="#2ecc71", s=20
                )
                ax_hdr_turns.set_title(
                    "Honeypot Discovery Rate (Not enough successes for avg)"
                )
                ax_hdr_turns.set_xlabel("Episode")
                ax_hdr_turns.set_ylabel("Turn Number")
        else:
            ax_hdr_turns.text(
                0.5,
                0.5,
                "No honeypots accessed to calculate turn rate",
                ha="center",
                va="center",
            )
            ax_hdr_turns.set_title("Honeypot Discovery Rate")
            ax_hdr_turns.axis("off")

    if exit_reason:
        fig.suptitle(f"Exit Reason: {exit_reason}", fontsize=13, fontweight="bold", color="#c0392b", y=1.002)

    plt.tight_layout()
    output_path = os.path.join(run_dir, "training_results_detailed.png")
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Print Summary
    print("\n=== Summary Statistics ===")
    if exit_reason:
        print(f"Exit Reason: {exit_reason}")
    print(f"Total Episodes: {len(episodes)}")

    if using_debug_logs and "red_team_tokens" in diagnostic_data:
        avg_rt = np.mean(diagnostic_data["red_team_tokens"])
        avg_v = np.mean(diagnostic_data["victim_tokens"])
        print(f"Avg Red Team Tokens: {avg_rt:.1f}")
        print(f"Avg Victim Tokens:   {avg_v:.1f}")

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
            total_degenerate = sum(diagnostic_data["is_degenerate"])
            total_fluent_bonus = sum(diagnostic_data["got_fluency_bonus"])
            avg_ppl = np.mean([p for p in diagnostic_data["perplexity"] if p > 0])
            avg_penalty = np.mean(diagnostic_data["fluency_penalty"])
            avg_degen_penalty = np.mean(diagnostic_data["degenerate_penalty"])

            print("\n=== Diagnostic Metrics ===")
            print(
                f"{'Total Gibberish Episodes':<25}: {total_gibberish:>4} ({100 * total_gibberish / len(episodes):.1f}%)"
            )
            print(
                f"{'Total Degenerate Inputs':<25}: {total_degenerate:>4} ({100 * total_degenerate / len(episodes):.1f}%)"
            )
            print(f"{'Got Fluency Bonus':<25}: {total_fluent_bonus:>4}")
            print(f"{'Average Perplexity':<25}: {avg_ppl:.1f}")
            print(f"{'Average Fluency Penalty':<25}: {avg_penalty:.2f}")
            print(f"{'Average Degenerate Penalty':<25}: {avg_degen_penalty:.2f}")

            # Degenerate reason breakdown
            degenerate_reasons = [r for r in diagnostic_data["degenerate_reason"] if r]
            if degenerate_reasons:
                reason_counts = {}
                for reason in degenerate_reasons:
                    # Extract the main reason (before any parentheses)
                    main_reason = reason.split(" (")[0] if " (" in reason else reason
                    reason_counts[main_reason] = reason_counts.get(main_reason, 0) + 1
                print("\n=== Degenerate Input Reasons ===")
                for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                    print(f"  {reason}: {count}")

            # Early vs Late comparison (warning sign detection)
            mid = len(episodes) // 2
            if mid > 0:
                early_gibberish = sum(diagnostic_data["is_gibberish"][:mid]) / mid
                late_gibberish = sum(diagnostic_data["is_gibberish"][mid:]) / (
                    len(episodes) - mid
                )
                early_degenerate = sum(diagnostic_data["is_degenerate"][:mid]) / mid
                late_degenerate = sum(diagnostic_data["is_degenerate"][mid:]) / (
                    len(episodes) - mid
                )

                print("\n=== Trend Analysis ===")
                print(f"Early Gibberish Rate (0-{mid}): {100 * early_gibberish:.1f}%")
                print(f"Late Gibberish Rate ({mid}+):   {100 * late_gibberish:.1f}%")
                print(f"Early Degenerate Rate (0-{mid}): {100 * early_degenerate:.1f}%")
                print(f"Late Degenerate Rate ({mid}+):   {100 * late_degenerate:.1f}%")

                if late_gibberish > early_gibberish * 2 and late_gibberish > 0.5:
                    print(
                        "⚠️  WARNING: Gibberish rate increased significantly - possible mode collapse!"
                    )
                elif late_gibberish < early_gibberish * 0.5:
                    print(
                        "✅ Good: Gibberish rate decreased - model is learning fluent attacks"
                    )

                if late_degenerate > early_degenerate * 2 and late_degenerate > 0.3:
                    print(
                        "⚠️  WARNING: Degenerate input rate increased - model may be stuck!"
                    )
                elif late_degenerate < early_degenerate * 0.5:
                    print(
                        "✅ Good: Degenerate rate decreased - model is producing meaningful inputs"
                    )

            # Decay and Honeypot Summary
            decay_factors = [
                d for d in diagnostic_data["decay_factor"] if d is not None
            ]
            raw_hps = diagnostic_data["new_honeypots_accessed"]
            honeypots_accessed = []
            for hps in raw_hps:
                if hps:
                    for hp in hps:
                        if hp is not None:
                            honeypots_accessed.append(hp)

            if (
                decay_factors
                or honeypots_accessed
                or any(diagnostic_data.get("attempted_honeypots", []))
            ):
                print("\n=== Reward Decay & Honeypot Tracking ===")

                if decay_factors:
                    avg_decay = np.mean(decay_factors)
                    min_decay = min(decay_factors)
                    print(f"{'Avg Decay Factor':<25}: {avg_decay:.4f}")
                    print(f"{'Min Decay Factor':<25}: {min_decay:.4f}")

                attempts_flattened = []
                for att in diagnostic_data.get("attempted_honeypots", []):
                    if att:
                        attempts_flattened.extend(att)

                if attempts_flattened:
                    unique_attempts = set(attempts_flattened)
                    print(
                        f"{'Failed HP Attempts':<25}: {len(attempts_flattened):>4} ({len(unique_attempts)} unique)"
                    )

                if honeypots_accessed:
                    unique_honeypots = set(honeypots_accessed)
                    print(
                        f"{'Honeypots Successfully Accessed':<25}: {len(honeypots_accessed):>4}"
                    )
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
