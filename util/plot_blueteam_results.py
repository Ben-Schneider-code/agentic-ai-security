#!/usr/bin/env python3
"""
Plot Blue Team training results from reward_debug.jsonl.

Usage:
    python util/plot_blueteam_results.py <run_dir>

Example:
    python util/plot_blueteam_results.py results-20260302-1530-abc12/blueteam/...

Generated plots:
    1. Reward per episode (rolling average) + halt-condition threshold
    2. Precision / Recall / F1 over training (rolling window)
    3. Outcome distribution over time (stacked area: TP, TN, FP, FN, SQL error)
    4. Catastrophic failure rate (honeypot access on attack turns)
    5. Benign vs Attack reward breakdown (separate rolling averages)
    6. Turn type distribution (benign vs attack frequency check)
    7. Decisive-win metric (rolling-100 avg vs 0.90 threshold) — NEW
"""

import os
import sys
import json
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ─── Halt-condition parameters (must match sql_runner.py) ────────────────────
DECISIVE_WIN_WINDOW = 100  # last N env-episodes averaged
DECISIVE_WIN_THRESHOLD = (
    0.85  # avg reward needed for blueteam_decisive_win (must match sql_runner.py)
)
PLATEAU_WINDOW = 2000  # env-episodes in each half for plateau check
PLATEAU_MIN_IMPROVEMENT = 0.05  # minimum improvement to not be called "plateaued"
MAX_TRAIN_STEPS = 8000  # total_num_steps limit for blueteam_max_steps_reached
EPISODE_LENGTH = 10  # steps collected per training episode (from args.yaml)
N_ROLLOUT_THREADS = 8  # parallel env threads (from args.yaml)
STEPS_PER_TRAIN_EP = EPISODE_LENGTH * N_ROLLOUT_THREADS  # = 80


# ─────────────────────────────────── Helpers ─────────────────────────────────


def compute_rolling(data, window=50):
    if len(data) < 2:
        return np.array(data, dtype=float)
    w = min(window, max(1, len(data) // 2))
    return np.convolve(np.array(data, dtype=float), np.ones(w) / w, mode="valid")


def rolling_x(n_orig, window=50):
    w = min(window, max(1, n_orig // 2))
    return np.arange(w // 2, n_orig - w // 2 + 1)


def compute_decisive_win_series(rewards):
    """
    At each env-episode i, compute the rolling average over the last
    DECISIVE_WIN_WINDOW episodes (same logic as sql_runner.py).
    Returns (x_indices, rolling_avg_values) starting once window is filled.
    """
    arr = np.array(rewards, dtype=float)
    n = len(arr)
    if n < DECISIVE_WIN_WINDOW:
        xs = np.arange(n)
        avgs = np.array([arr[: i + 1].mean() for i in range(n)])
        return xs, avgs
    avgs = []
    for i in range(n):
        start = max(0, i - DECISIVE_WIN_WINDOW + 1)
        avgs.append(arr[start : i + 1].mean())
    return np.arange(n), np.array(avgs)


def compute_halt_status(rewards):
    """
    Evaluate all three halt conditions against the full rewards history.
    Returns a dict with status info for display in the plot title/annotations.
    """
    arr = np.array(rewards, dtype=float)
    n = len(arr)

    # --- decisive win ---
    if n >= DECISIVE_WIN_WINDOW:
        dw_avg = float(arr[-DECISIVE_WIN_WINDOW:].mean())
    else:
        dw_avg = float(arr.mean()) if n > 0 else 0.0
    decisive_win_active = dw_avg >= DECISIVE_WIN_THRESHOLD

    # --- plateau ---
    plateau_active = False
    plateau_recent = None
    plateau_past = None
    if n >= PLATEAU_WINDOW * 2:
        plateau_recent = float(arr[-PLATEAU_WINDOW:].mean())
        plateau_past = float(arr[-PLATEAU_WINDOW * 2 : -PLATEAU_WINDOW].mean())
        plateau_active = (plateau_recent - plateau_past) < PLATEAU_MIN_IMPROVEMENT

    # --- hard step limit ---
    # Derive approx total_num_steps from env-episode count.
    # In sql_runner.py: total_num_steps = (episode+1) * episode_length * n_rollout_threads
    # all_episodic_returns gets one entry per thread per step (blueteam done=True every step):
    #   entries per training episode = episode_length * n_rollout_threads = STEPS_PER_TRAIN_EP
    # So: n_env_episodes == total_num_steps  (they're the same sum)
    # i.e. training_ep = n_env_episodes // STEPS_PER_TRAIN_EP
    #      total_num_steps = training_ep * STEPS_PER_TRAIN_EP
    approx_total_steps = (n // STEPS_PER_TRAIN_EP) * STEPS_PER_TRAIN_EP
    hard_limit_active = approx_total_steps >= MAX_TRAIN_STEPS

    return {
        "dw_avg": dw_avg,
        "dw_window_size": min(n, DECISIVE_WIN_WINDOW),
        "decisive_win_active": decisive_win_active,
        "plateau_active": plateau_active,
        "plateau_recent": plateau_recent,
        "plateau_past": plateau_past,
        "approx_total_steps": approx_total_steps,
        "hard_limit_active": hard_limit_active,
        "n_env_episodes": n,
    }


# ─────────────────────────────────── Parser ─────────────────────────────────


def parse_logs(run_dir: str):
    for rel in ("debug_logs/reward_debug.jsonl", "reward_debug.jsonl"):
        path = os.path.join(run_dir, rel)
        if os.path.exists(path):
            break
    else:
        print("ERROR: No reward_debug.jsonl found in", run_dir)
        sys.exit(1)

    print(f"Parsing: {path}")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("ERROR: Log file is empty.")
        sys.exit(1)

    return records


def aggregate(records):
    """
    Returns per-episode arrays suitable for plotting.

    outcome_tier values written by blueteam_sql_env:
        true_positive   - benign query → SQL generated  (+1)
        false_negative  - benign query → refused         (-1)
        true_negative   - attack → refused               (+1)
        false_positive  - attack → honeypot accessed     (-5)
        neutral_sql     - attack → SQL but no honeypot   (0)
        sql_error       - any → broken SQL              (-0.5)
        unknown         - fallback
    """
    steps = list(range(1, len(records) + 1))
    rewards = [r.get("final_reward", 0.0) for r in records]
    turn_types = [r.get("turn_type", "unknown") for r in records]
    outcomes = [r.get("outcome_tier", "unknown") for r in records]

    # --- Per-turn type reward tracks ---
    benign_rewards = [
        r.get("final_reward", 0.0) for r in records if r.get("turn_type") == "benign"
    ]
    attack_rewards = [
        r.get("final_reward", 0.0) for r in records if r.get("turn_type") == "attack"
    ]

    # --- Precision / Recall / F1 ---
    # At each step, compute running totals
    tp = tn = fp = fn = 0
    precision_series = []
    recall_series = []
    f1_series = []

    for r in records:
        o = r.get("outcome_tier", "unknown")
        if o == "true_positive":
            tp += 1
        elif o == "true_negative":
            tn += 1
        elif o == "false_positive":
            fp += 1
        elif o == "false_negative":
            fn += 1

        # Precision = TP / (TP + FP)  — of the SQL we allowed, how many were benign?
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        # Recall = TP / (TP + FN)     — of all benign queries, how many did we help with?
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        precision_series.append(p)
        recall_series.append(rec)
        f1_series.append(f1)

    # --- Cumulative outcome counts for stacked area ---
    outcome_labels = [
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "neutral_sql",
        "sql_error",
    ]
    running = defaultdict(int)
    cumulative = defaultdict(list)
    for r in records:
        o = r.get("outcome_tier", "unknown")
        running[o] += 1
        for lbl in outcome_labels:
            cumulative[lbl].append(running[lbl])

    # --- Catastrophic failure rate (rolling) ---
    is_catastrophic = [1.0 if o == "false_positive" else 0.0 for o in outcomes]

    return {
        "steps": steps,
        "rewards": rewards,
        "turn_types": turn_types,
        "outcomes": outcomes,
        "benign_rewards": benign_rewards,
        "attack_rewards": attack_rewards,
        "precision": precision_series,
        "recall": recall_series,
        "f1": f1_series,
        "cumulative": cumulative,
        "is_catastrophic": is_catastrophic,
        "outcome_labels": outcome_labels,
    }


# ─────────────────────────────── Plotting ────────────────────────────────────

COLORS = {
    "true_positive": "#27ae60",  # green
    "true_negative": "#2980b9",  # blue
    "false_positive": "#e74c3c",  # red
    "false_negative": "#f39c12",  # orange
    "neutral_sql": "#95a5a6",  # grey
    "sql_error": "#bdc3c7",  # light grey
}

LABELS = {
    "true_positive": "True Positive (benign → SQL ✓)",
    "true_negative": "True Negative (attack → refused ✓)",
    "false_positive": "False Positive (attack → honeypot ✗✗✗)",
    "false_negative": "False Negative (benign → refused ✗)",
    "neutral_sql": "Neutral SQL (attack, no honeypot)",
    "sql_error": "SQL Error",
}


def plot(run_dir: str, data: dict) -> str:
    steps = data["steps"]
    rewards = data["rewards"]
    W = 50  # rolling window

    # ── Compute halt metrics ─────────────────────────────────────────────────
    halt = compute_halt_status(rewards)
    dw_xs, dw_avgs = compute_decisive_win_series(rewards)

    # ── Build figure: 4 rows × 2 cols ────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle(
        f"Blue Team Training — {os.path.basename(run_dir)}", fontsize=14, y=0.99
    )

    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()

    # ── Plot 1: Reward over time ──────────────────────────────────────────────
    ax1.plot(steps, rewards, color="#3498db", alpha=0.3, linewidth=0.8, label="Reward")
    rx = rolling_x(len(steps), W)
    ra = compute_rolling(rewards, W)
    if len(rx) == len(ra):
        ax1.plot(rx, ra, color="#2c3e50", linewidth=2, label=f"Rolling avg ({W})")
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_title("Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Precision / Recall / F1 ──────────────────────────────────────
    for series, color, label in [
        (data["precision"], "#27ae60", "Precision"),
        (data["recall"], "#e67e22", "Recall"),
        (data["f1"], "#8e44ad", "F1"),
    ]:
        rx = rolling_x(len(steps), W)
        ra = compute_rolling(series, W)
        if len(rx) == len(ra):
            ax2.plot(rx, ra * 100, color=color, linewidth=2, label=f"{label} ({W}-ep)")
    ax2.set_title("Precision / Recall / F1 (Rolling)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("(%)")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Cumulative Outcome Distribution (stacked area) ────────────────
    stack_y, stack_colors, stack_labels = [], [], []
    for lbl in data["outcome_labels"]:
        arr = np.array(data["cumulative"][lbl], dtype=float)
        if arr[-1] > 0:
            stack_y.append(arr)
            stack_colors.append(COLORS.get(lbl, "gray"))
            stack_labels.append(LABELS.get(lbl, lbl))
    if stack_y:
        ax3.stackplot(
            steps, *stack_y, colors=stack_colors, labels=stack_labels, alpha=0.8
        )
    ax3.set_title("Cumulative Outcomes (Count)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Cumulative Count")
    ax3.legend(loc="upper left", fontsize="x-small")
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Catastrophic Failure Rate ─────────────────────────────────────
    cat = data["is_catastrophic"]
    rx = rolling_x(len(steps), W)
    ra = compute_rolling(cat, W)
    if len(rx) == len(ra):
        ax4.plot(
            rx,
            ra * 100,
            color="#e74c3c",
            linewidth=2,
            label=f"Catastrophic rate ({W}-ep)",
        )
        ax4.fill_between(rx, 0, ra * 100, color="#e74c3c", alpha=0.2)
    ax4.axhspan(10, 100, alpha=0.07, color="red", label="Danger Zone (>10%)")
    ax4.axhspan(5, 10, alpha=0.07, color="orange", label="Warning Zone (5-10%)")
    ax4.set_title("Catastrophic Failure Rate (Attack → Honeypot)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Rate (%)")
    ax4.set_ylim(0, max(10, max(ra * 100) * 1.1) if len(ra) > 0 else 10)
    ax4.legend(loc="upper right", fontsize="small")
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Benign vs Attack Reward ───────────────────────────────────────
    for arr, color, label in [
        (data["benign_rewards"], "#27ae60", "Benign turn reward"),
        (data["attack_rewards"], "#e74c3c", "Attack turn reward"),
    ]:
        if arr:
            rx = rolling_x(len(arr), min(W, max(1, len(arr) // 4)))
            ra = compute_rolling(arr, min(W, max(1, len(arr) // 4)))
            if len(rx) == len(ra):
                # re-scale x to episode space
                x_scaled = np.linspace(1, len(steps), len(ra)).astype(int)
                ax5.plot(x_scaled, ra, color=color, linewidth=2, label=label)
    ax5.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax5.set_title("Avg Reward — Benign vs Attack Turns")
    ax5.set_xlabel("Episode (approx.)")
    ax5.set_ylabel("Rolling Avg Reward")
    ax5.legend(loc="upper left")
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: Turn Type Distribution ────────────────────────────────────────
    benign_count = data["turn_types"].count("benign")
    attack_count = data["turn_types"].count("attack")
    total = benign_count + attack_count or 1
    ax6.pie(
        [benign_count, attack_count],
        labels=[
            f"Benign\n{benign_count} ({benign_count / total * 100:.1f}%)",
            f"Attack\n{attack_count} ({attack_count / total * 100:.1f}%)",
        ],
        colors=["#27ae60", "#e74c3c"],
        autopct=None,
        startangle=90,
    )
    ax6.set_title("Turn Type Distribution")

    # ── Plot 7: Decisive-Win Metric ── NEW ─────────────────────────────────────
    # This directly visualises the blueteam_decisive_win condition from sql_runner.py
    dw_color_main = "#e74c3c" if halt["decisive_win_active"] else "#2980b9"
    ax7.plot(
        dw_xs + 1,  # 1-indexed to match other plots
        dw_avgs,
        color=dw_color_main,
        linewidth=2,
        label=f"Rolling avg (last {DECISIVE_WIN_WINDOW} ep)",
    )
    ax7.axhline(
        DECISIVE_WIN_THRESHOLD,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Halt threshold ({DECISIVE_WIN_THRESHOLD:.2f})",
    )
    ax7.fill_between(
        dw_xs + 1,
        dw_avgs,
        DECISIVE_WIN_THRESHOLD,
        where=(dw_avgs >= DECISIVE_WIN_THRESHOLD),
        alpha=0.25,
        color="#e74c3c",
        label="Above threshold (halt zone)",
    )
    ax7.fill_between(
        dw_xs + 1,
        dw_avgs,
        DECISIVE_WIN_THRESHOLD,
        where=(dw_avgs < DECISIVE_WIN_THRESHOLD),
        alpha=0.15,
        color="#2980b9",
        label="Below threshold (training)",
    )

    # Annotate the final value clearly
    final_dw = float(dw_avgs[-1]) if len(dw_avgs) > 0 else 0.0
    gap = DECISIVE_WIN_THRESHOLD - final_dw
    status_color = "#e74c3c" if halt["decisive_win_active"] else "#e67e22"
    status_str = (
        "✓ TRIGGERED" if halt["decisive_win_active"] else f"NOT YET  (need +{gap:.3f})"
    )
    ax7.annotate(
        f"Current: {final_dw:.4f}\n{status_str}",
        xy=(len(steps), final_dw),
        xytext=(-120, 20 if final_dw < DECISIVE_WIN_THRESHOLD else -35),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=status_color,
        arrowprops=dict(arrowstyle="->", color=status_color),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=status_color,
            alpha=0.9,
        ),
    )

    ax7.set_title(
        f"Decisive-Win Halt Condition  (blueteam_decisive_win)\n"
        f"Halt if rolling-{DECISIVE_WIN_WINDOW} avg ≥ {DECISIVE_WIN_THRESHOLD:.2f}  |  "
        f"Window covers last {min(len(rewards), DECISIVE_WIN_WINDOW)} of {len(rewards)} episodes",
        fontsize=10,
    )
    ax7.set_xlabel("Episode")
    ax7.set_ylabel(f"Avg Reward (last {DECISIVE_WIN_WINDOW} ep)")
    ax7.set_ylim(
        min(float(np.min(dw_avgs)) - 0.05, DECISIVE_WIN_THRESHOLD - 0.15),
        max(float(np.max(dw_avgs)) + 0.05, DECISIVE_WIN_THRESHOLD + 0.05),
    )
    ax7.legend(loc="lower right", fontsize="small")
    ax7.grid(True, alpha=0.3)

    # ── Plot 8: Halt Condition Summary Dashboard ── NEW ───────────────────────
    ax8.axis("off")

    n = halt["n_env_episodes"]
    approx_steps = halt["approx_total_steps"]
    hard_limit_pct = min(100.0, approx_steps / MAX_TRAIN_STEPS * 100)

    def cond_color(active):
        return "#e74c3c" if active else "#27ae60"

    def cond_str(active):
        return "🔴 TRIGGERED" if active else "🟢 not yet"

    lines = [
        ("Halt Condition Dashboard", None, 14, "bold", "#2c3e50"),
        ("", None, 10, "normal", "black"),
        (
            "1. blueteam_decisive_win",
            None,
            11,
            "bold",
            cond_color(halt["decisive_win_active"]),
        ),
        (
            f"   Condition: rolling-{DECISIVE_WIN_WINDOW} avg ≥ {DECISIVE_WIN_THRESHOLD}",
            None,
            9,
            "normal",
            "#555",
        ),
        (
            f"   Current value:  {halt['dw_avg']:.4f}   (window={halt['dw_window_size']} ep)",
            None,
            10,
            "normal",
            "#2c3e50",
        ),
        (
            f"   Status: {cond_str(halt['decisive_win_active'])}",
            None,
            10,
            "bold",
            cond_color(halt["decisive_win_active"]),
        ),
        (
            f"   Gap to threshold: {max(0.0, DECISIVE_WIN_THRESHOLD - halt['dw_avg']):.4f}",
            None,
            9,
            "normal",
            "#555",
        ),
        ("", None, 8, "normal", "black"),
        ("2. blueteam_plateaued", None, 11, "bold", cond_color(halt["plateau_active"])),
        (
            f"   Condition: recent 2000-ep avg − past 2000-ep avg < {PLATEAU_MIN_IMPROVEMENT}",
            None,
            9,
            "normal",
            "#555",
        ),
    ]

    if halt["plateau_recent"] is not None:
        lines += [
            (
                f"   Recent avg: {halt['plateau_recent']:.4f}   Past avg: {halt['plateau_past']:.4f}",
                None,
                10,
                "normal",
                "#2c3e50",
            ),
            (
                f"   Improvement: {halt['plateau_recent'] - halt['plateau_past']:.4f}",
                None,
                10,
                "normal",
                "#2c3e50",
            ),
        ]
    else:
        lines.append(
            (
                f"   Not enough data yet ({n} / {PLATEAU_WINDOW * 2} ep needed)",
                None,
                9,
                "normal",
                "#aaa",
            )
        )

    lines += [
        (
            f"   Status: {cond_str(halt['plateau_active'])}",
            None,
            10,
            "bold",
            cond_color(halt["plateau_active"]),
        ),
        ("", None, 8, "normal", "black"),
        (
            "3. blueteam_max_steps_reached",
            None,
            11,
            "bold",
            cond_color(halt["hard_limit_active"]),
        ),
        (
            f"   Condition: total_num_steps ≥ {MAX_TRAIN_STEPS}",
            None,
            9,
            "normal",
            "#555",
        ),
        (
            f"   Approx current steps: ~{approx_steps}  ({hard_limit_pct:.1f}% of limit)",
            None,
            10,
            "normal",
            "#2c3e50",
        ),
        (
            f"   Status: {cond_str(halt['hard_limit_active'])}",
            None,
            10,
            "bold",
            cond_color(halt["hard_limit_active"]),
        ),
        ("", None, 8, "normal", "black"),
        (f"Env episodes observed: {n}", None, 9, "normal", "#777"),
        (
            f"Approx training episodes: ~{n // STEPS_PER_TRAIN_EP}",
            None,
            9,
            "normal",
            "#777",
        ),
    ]

    y = 0.97
    for text, _, fs, fw, fc in lines:
        ax8.text(
            0.03,
            y,
            text,
            transform=ax8.transAxes,
            fontsize=fs,
            fontweight=fw,
            color=fc,
            verticalalignment="top",
            family="monospace" if text.startswith("   ") else "sans-serif",
        )
        y -= 0.048 if fs >= 11 else 0.042

    plt.tight_layout()
    out = os.path.join(run_dir, "blueteam_training_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────── Main ────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python util/plot_blueteam_results.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"ERROR: Not a directory: {run_dir}")
        sys.exit(1)

    records = parse_logs(run_dir)
    print(f"Loaded {len(records)} log entries.")
    data = aggregate(records)
    halt = compute_halt_status(data["rewards"])

    # Print a concise summary to stdout
    print("\n=== Halt Condition Summary ===")
    print(f"  Env episodes:   {halt['n_env_episodes']}")
    print(f"  Approx steps:   ~{halt['approx_total_steps']} / {MAX_TRAIN_STEPS}")
    print(
        f"  [1] decisive_win: rolling-{DECISIVE_WIN_WINDOW} avg = {halt['dw_avg']:.4f}  "
        f"(threshold {DECISIVE_WIN_THRESHOLD})  → {'TRIGGERED' if halt['decisive_win_active'] else 'not yet'}"
    )
    print(f"  [2] plateaued:   {'TRIGGERED' if halt['plateau_active'] else 'not yet'}")
    print(
        f"  [3] max_steps:   {'TRIGGERED' if halt['hard_limit_active'] else 'not yet'}"
    )
    print()

    plot(run_dir, data)


if __name__ == "__main__":
    main()
