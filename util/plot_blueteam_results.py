#!/usr/bin/env python3
"""
Plot Blue Team training results from reward_debug.jsonl.

Usage:
    python util/plot_blueteam_results.py <run_dir>

Example:
    python util/plot_blueteam_results.py results-20260302-1530-abc12/blueteam/...

Generated plots (5 rows × 2 cols):
    (0,0) Reward per episode (rolling average) + halt-condition threshold
    (0,1) Precision / Recall / F1 over training (rolling window)
    (1,0) Benign Turn Outcome Rates (TP/FN/sql_error on benign turns — utility check)
    (1,1) Attack Turn Outcome Rates (TN/FP/neutral_sql on attack turns — defense check)
    (2,0) Catastrophic failure rate (honeypot access on attack turns)
    (2,1) Benign vs Attack reward breakdown (separate rolling averages)
    (3,0) Refusal Rate by Turn Type (attack vs benign — detects degenerate refuser)
    (3,1) Utility vs Security Composite Score (vs naive-refuser baseline)
    (4,0) Decisive-win metric (rolling-100 avg vs 0.90 threshold)
    (4,1) Halt condition dashboard
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

    # --- Catastrophic failure rate (rolling) ---
    is_catastrophic = [1.0 if o == "false_positive" else 0.0 for o in outcomes]

    # --- Per-turn-type binary outcome arrays ---
    # Benign turns
    benign_is_tp = []
    benign_is_fn = []
    benign_is_sqlerr = []
    benign_is_refusal = []
    benign_episode_indices = []

    # Attack turns
    attack_is_tn = []
    attack_is_fp = []
    attack_is_neutral = []
    attack_is_sqlerr = []
    attack_is_refusal = []
    attack_episode_indices = []

    for i, r in enumerate(records):
        tt = r.get("turn_type", "unknown")
        o = r.get("outcome_tier", "unknown")
        ep_idx = i + 1  # 1-indexed global episode

        if tt == "benign":
            benign_is_tp.append(1.0 if o == "true_positive" else 0.0)
            benign_is_fn.append(1.0 if o == "false_negative" else 0.0)
            benign_is_sqlerr.append(1.0 if o == "sql_error" else 0.0)
            benign_is_refusal.append(1.0 if o == "false_negative" else 0.0)
            benign_episode_indices.append(ep_idx)
        elif tt == "attack":
            attack_is_tn.append(1.0 if o == "true_negative" else 0.0)
            attack_is_fp.append(1.0 if o == "false_positive" else 0.0)
            attack_is_neutral.append(1.0 if o == "neutral_sql" else 0.0)
            attack_is_sqlerr.append(1.0 if o == "sql_error" else 0.0)
            attack_is_refusal.append(1.0 if o == "true_negative" else 0.0)
            attack_episode_indices.append(ep_idx)

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
        "is_catastrophic": is_catastrophic,
        # Benign-turn arrays
        "benign_is_tp": benign_is_tp,
        "benign_is_fn": benign_is_fn,
        "benign_is_sqlerr": benign_is_sqlerr,
        "benign_is_refusal": benign_is_refusal,
        "benign_episode_indices": benign_episode_indices,
        # Attack-turn arrays
        "attack_is_tn": attack_is_tn,
        "attack_is_fp": attack_is_fp,
        "attack_is_neutral": attack_is_neutral,
        "attack_is_sqlerr": attack_is_sqlerr,
        "attack_is_refusal": attack_is_refusal,
        "attack_episode_indices": attack_episode_indices,
    }


# ─────────────────────────────── Plotting ────────────────────────────────────


def plot(run_dir: str, data: dict) -> str:
    steps = data["steps"]
    rewards = data["rewards"]
    W = 50  # rolling window

    # ── Compute halt metrics ─────────────────────────────────────────────────
    halt = compute_halt_status(rewards)
    dw_xs, dw_avgs = compute_decisive_win_series(rewards)

    # ── Build figure: 5 rows × 2 cols ────────────────────────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(16, 30))
    fig.suptitle(
        f"Blue Team Training — {os.path.basename(run_dir)}", fontsize=14, y=0.99
    )

    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = axes.flatten()

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

    # ── Plot 3: Benign Turn Outcome Rates — "is utility maintained?" ──────────
    benign_eps = np.array(data["benign_episode_indices"])
    for arr, color, label, lw in [
        (data["benign_is_tp"], "#27ae60", "TP rate — utility (benign→SQL ✓)", 2.5),
        (data["benign_is_fn"], "#f39c12", "FN rate — over-refusal (benign→refused ✗)", 1.5),
        (data["benign_is_sqlerr"], "#bdc3c7", "SQL error rate", 1.0),
    ]:
        if len(arr) >= 2:
            rx = rolling_x(len(arr), W)
            ra = compute_rolling(arr, W)
            if len(rx) == len(ra) and len(rx) > 0:
                global_x = benign_eps[rx]
                ax3.plot(global_x, ra * 100, color=color, linewidth=lw, label=label)
    ax3.axhspan(80, 100, alpha=0.08, color="#27ae60", label="Healthy utility zone (80–100%)")
    ax3.set_title("Benign Turn Outcome Rates — Is Utility Maintained?")
    ax3.set_xlabel("Episode (global)")
    ax3.set_ylabel("Rate (%)")
    ax3.set_ylim(-2, 105)
    ax3.legend(loc="lower left", fontsize="small")
    ax3.grid(True, alpha=0.3)

    # ── Plot 4: Attack Turn Outcome Rates — "how is it defending?" ────────────
    attack_eps = np.array(data["attack_episode_indices"])
    for arr, color, label, lw in [
        (data["attack_is_tn"], "#2980b9", "TN rate — defense (attack→refused ✓)", 2.5),
        (data["attack_is_fp"], "#e74c3c", "FP rate — catastrophic (attack→honeypot ✗)", 1.5),
        (data["attack_is_neutral"], "#95a5a6", "Neutral SQL rate (attack→SQL, no honeypot)", 1.0),
        (data["attack_is_sqlerr"], "#bdc3c7", "SQL error rate", 1.0),
    ]:
        if len(arr) >= 2:
            rx = rolling_x(len(arr), W)
            ra = compute_rolling(arr, W)
            if len(rx) == len(ra) and len(rx) > 0:
                global_x = attack_eps[rx]
                ax4.plot(global_x, ra * 100, color=color, linewidth=lw, label=label)
    ax4.set_title("Attack Turn Outcome Rates — How Is It Defending?")
    ax4.set_xlabel("Episode (global)")
    ax4.set_ylabel("Rate (%)")
    ax4.set_ylim(-2, 105)
    ax4.legend(loc="upper left", fontsize="small")
    ax4.grid(True, alpha=0.3)

    # ── Plot 5: Catastrophic Failure Rate ─────────────────────────────────────
    cat = data["is_catastrophic"]
    rx = rolling_x(len(steps), W)
    ra = compute_rolling(cat, W)
    if len(rx) == len(ra):
        ax5.plot(
            rx,
            ra * 100,
            color="#e74c3c",
            linewidth=2,
            label=f"Catastrophic rate ({W}-ep)",
        )
        ax5.fill_between(rx, 0, ra * 100, color="#e74c3c", alpha=0.2)
    ax5.axhspan(10, 100, alpha=0.07, color="red", label="Danger Zone (>10%)")
    ax5.axhspan(5, 10, alpha=0.07, color="orange", label="Warning Zone (5-10%)")
    ax5.set_title("Catastrophic Failure Rate (Attack → Honeypot)")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Rate (%)")
    ax5.set_ylim(0, max(10, max(ra * 100) * 1.1) if len(ra) > 0 else 10)
    ax5.legend(loc="upper right", fontsize="small")
    ax5.grid(True, alpha=0.3)

    # ── Plot 6: Benign vs Attack Reward ───────────────────────────────────────
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
                ax6.plot(x_scaled, ra, color=color, linewidth=2, label=label)
    ax6.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax6.set_title("Avg Reward — Benign vs Attack Turns")
    ax6.set_xlabel("Episode (approx.)")
    ax6.set_ylabel("Rolling Avg Reward")
    ax6.legend(loc="upper left")
    ax6.grid(True, alpha=0.3)

    # ── Plot 7: Refusal Rate by Turn Type — "refuse everything" detector ──────
    attack_refusal_global_x = None
    attack_refusal_ra = None
    benign_refusal_global_x = None
    benign_refusal_ra = None

    if len(data["attack_is_refusal"]) >= 2:
        rx = rolling_x(len(data["attack_is_refusal"]), W)
        ra = compute_rolling(data["attack_is_refusal"], W)
        if len(rx) == len(ra) and len(rx) > 0:
            attack_refusal_global_x = attack_eps[rx]
            attack_refusal_ra = ra * 100
            ax7.plot(
                attack_refusal_global_x,
                attack_refusal_ra,
                color="#2980b9",
                linewidth=2,
                label="Attack refusal rate (high = good ✓)",
            )

    if len(data["benign_is_refusal"]) >= 2:
        rx = rolling_x(len(data["benign_is_refusal"]), W)
        ra = compute_rolling(data["benign_is_refusal"], W)
        if len(rx) == len(ra) and len(rx) > 0:
            benign_refusal_global_x = benign_eps[rx]
            benign_refusal_ra = ra * 100
            ax7.plot(
                benign_refusal_global_x,
                benign_refusal_ra,
                color="#f39c12",
                linewidth=2,
                label="Benign refusal rate (high = bad ✗)",
            )

    # Green fill where attack refusal > benign refusal (discrimination gap)
    if (
        attack_refusal_global_x is not None
        and benign_refusal_global_x is not None
        and len(attack_refusal_global_x) > 0
        and len(benign_refusal_global_x) > 0
    ):
        # Interpolate both series onto a common x grid
        x_min = max(attack_refusal_global_x[0], benign_refusal_global_x[0])
        x_max = min(attack_refusal_global_x[-1], benign_refusal_global_x[-1])
        if x_max > x_min:
            common_x = np.linspace(x_min, x_max, 200)
            attack_interp = np.interp(common_x, attack_refusal_global_x, attack_refusal_ra)
            benign_interp = np.interp(common_x, benign_refusal_global_x, benign_refusal_ra)
            ax7.fill_between(
                common_x,
                benign_interp,
                attack_interp,
                where=(attack_interp > benign_interp),
                alpha=0.2,
                color="#27ae60",
                label="Discrimination gap (attack > benign refusal)",
            )
            # Annotate final gap
            final_attack = float(np.interp(x_max, attack_refusal_global_x, attack_refusal_ra))
            final_benign = float(np.interp(x_max, benign_refusal_global_x, benign_refusal_ra))
            final_gap = final_attack - final_benign
            gap_color = "#27ae60" if final_gap > 0 else "#e74c3c"
            ax7.annotate(
                f"Final gap: {final_gap:+.1f}%",
                xy=(x_max, (final_attack + final_benign) / 2),
                xytext=(-100, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=gap_color,
                arrowprops=dict(arrowstyle="->", color=gap_color),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=gap_color, alpha=0.9),
            )

    ax7.set_title("Refusal Rate by Turn Type — Degenerate Refuser Detector\n(both lines rising = refusing everything)")
    ax7.set_xlabel("Episode (global)")
    ax7.set_ylabel("Refusal Rate (%)")
    ax7.set_ylim(-2, 105)
    ax7.legend(loc="upper left", fontsize="small")
    ax7.grid(True, alpha=0.3)

    # ── Plot 8: Utility vs Security Composite Score ────────────────────────────
    utility_series = None
    security_series = None
    utility_global_x = None
    security_global_x = None

    if len(data["benign_is_tp"]) >= 2:
        rx = rolling_x(len(data["benign_is_tp"]), W)
        ra = compute_rolling(data["benign_is_tp"], W)
        if len(rx) == len(ra) and len(rx) > 0:
            utility_global_x = benign_eps[rx]
            utility_series = ra * 100
            ax8.plot(
                utility_global_x,
                utility_series,
                color="#27ae60",
                linewidth=2,
                label="Utility score — TP rate on benign",
            )

    if len(data["attack_is_tn"]) >= 2:
        rx = rolling_x(len(data["attack_is_tn"]), W)
        ra = compute_rolling(data["attack_is_tn"], W)
        if len(rx) == len(ra) and len(rx) > 0:
            security_global_x = attack_eps[rx]
            security_series = ra * 100
            ax8.plot(
                security_global_x,
                security_series,
                color="#2980b9",
                linewidth=2,
                label="Security score — TN rate on attack",
            )

    # Composite score on common x grid
    if utility_global_x is not None and security_global_x is not None:
        x_min = max(utility_global_x[0], security_global_x[0])
        x_max = min(utility_global_x[-1], security_global_x[-1])
        if x_max > x_min:
            common_x = np.linspace(x_min, x_max, 300)
            u_interp = np.interp(common_x, utility_global_x, utility_series)
            s_interp = np.interp(common_x, security_global_x, security_series)
            composite = (u_interp + s_interp) / 2
            ax8.plot(
                common_x,
                composite,
                color="#8e44ad",
                linewidth=2.5,
                linestyle="--",
                label="Composite = (Utility + Security) / 2",
            )

    # Naive refuser baseline: refuses everything → 0% utility, 100% security → composite=50%
    ax8.axhline(
        50,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Naive refuser baseline (50% composite)",
    )
    ax8.set_title("Utility vs Security Composite Score\n(above dashed line = better than blind refusal)")
    ax8.set_xlabel("Episode (global)")
    ax8.set_ylabel("Score (%)")
    ax8.set_ylim(-2, 105)
    ax8.legend(loc="lower right", fontsize="small")
    ax8.grid(True, alpha=0.3)

    # ── Plot 9: Decisive-Win Metric ────────────────────────────────────────────
    dw_color_main = "#e74c3c" if halt["decisive_win_active"] else "#2980b9"
    ax9.plot(
        dw_xs + 1,  # 1-indexed to match other plots
        dw_avgs,
        color=dw_color_main,
        linewidth=2,
        label=f"Rolling avg (last {DECISIVE_WIN_WINDOW} ep)",
    )
    ax9.axhline(
        DECISIVE_WIN_THRESHOLD,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Halt threshold ({DECISIVE_WIN_THRESHOLD:.2f})",
    )
    ax9.fill_between(
        dw_xs + 1,
        dw_avgs,
        DECISIVE_WIN_THRESHOLD,
        where=(dw_avgs >= DECISIVE_WIN_THRESHOLD),
        alpha=0.25,
        color="#e74c3c",
        label="Above threshold (halt zone)",
    )
    ax9.fill_between(
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
    ax9.annotate(
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

    ax9.set_title(
        f"Decisive-Win Halt Condition  (blueteam_decisive_win)\n"
        f"Halt if rolling-{DECISIVE_WIN_WINDOW} avg ≥ {DECISIVE_WIN_THRESHOLD:.2f}  |  "
        f"Window covers last {min(len(rewards), DECISIVE_WIN_WINDOW)} of {len(rewards)} episodes",
        fontsize=10,
    )
    ax9.set_xlabel("Episode")
    ax9.set_ylabel(f"Avg Reward (last {DECISIVE_WIN_WINDOW} ep)")
    ax9.set_ylim(
        min(float(np.min(dw_avgs)) - 0.05, DECISIVE_WIN_THRESHOLD - 0.15),
        max(float(np.max(dw_avgs)) + 0.05, DECISIVE_WIN_THRESHOLD + 0.05),
    )
    ax9.legend(loc="lower right", fontsize="small")
    ax9.grid(True, alpha=0.3)

    # ── Plot 10: Halt Condition Summary Dashboard ──────────────────────────────
    ax10.axis("off")

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
        ax10.text(
            0.03,
            y,
            text,
            transform=ax10.transAxes,
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
