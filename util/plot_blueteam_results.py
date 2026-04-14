#!/usr/bin/env python3
"""
Plot Blue Team training results from reward_debug.jsonl.

Usage:
    python util/plot_blueteam_results.py <run_dir>

Example:
    python util/plot_blueteam_results.py results-20260302-1530-abc12/blueteam/...

Generated plots (6 rows × 2 cols):
    (0,0) Reward per episode (rolling average) + halt-condition threshold
    (0,1) Precision / Recall / F1 over training (rolling window)
    (1,0) Benign Turn Outcome Rates (TP/FN/sql_error on benign turns — utility check)
    (1,1) Attack Turn Outcome Rates (TN/FP/neutral_sql on attack turns — defense check)
    (2,0) Catastrophic failure rate (honeypot access on attack turns)
    (2,1) Benign vs Attack reward breakdown (separate rolling averages)
    (3,0) Refusal Rate by Turn Type (attack vs benign — detects degenerate refuser)
    (3,1) Utility vs Security Composite Score (vs naive-refuser baseline)
    (4,0) Decisive-win metric (rolling avg vs configured threshold)
    (4,1) Halt condition dashboard
    (5,0) Train vs Eval Generalization Gap — rolling training TP rate (line) + eval TP
          rate per eval run (scatter). Large gap → memorization; small gap → generalization.
    (5,1) Eval TP Rate by Query Category — bar chart showing per-category TP rate across
          all eval runs, using BENIGN_EVAL_QUERIES category groupings.
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

# ─── Eval query → category mapping (mirrors BENIGN_EVAL_QUERIES in blueteam_sql_env.py) ──
def _build_eval_query_categories():
    """Build category mapping from actual BENIGN_EVAL_QUERIES.

    Dynamically imports BENIGN_EVAL_QUERIES to stay in sync — avoids the stale
    hardcoded dict that previously contained queries referencing honeypot columns.
    Multi-turn sequences are keyed by their first turn's text.
    """
    try:
        from marft.envs.blueteam_sql.blueteam_sql_env import (
            BENIGN_EVAL_QUERIES,
            BlueTeamSQLEnv,
        )
    except ImportError:
        return {}

    cats = {}
    # Plain eval categories by section (positional — matches comment blocks)
    _PLAIN_SECTIONS = [
        (0, 5, "Account Profile"),
        (5, 10, "Order History"),
        (10, 15, "Order Status"),
        (15, 20, "Line Items"),
        (20, 25, "Financials"),
        (25, 30, "Catalog"),
        (30, 35, "Product Attrs"),
        (35, 40, "Localization"),
        (40, 45, "Cross-Ref"),
        (45, 50, "Temporal"),
        (50, 55, "Edge Cases"),
    ]

    str_idx = 0
    for query in BENIGN_EVAL_QUERIES:
        if isinstance(query, list):
            # Multi-turn sequence: key by first turn
            cats[query[0]] = "Multi-turn"
        else:
            style = BlueTeamSQLEnv._classify_benign_style(query)
            if style == "adversarial":
                cats[query] = "Adversarial"
            else:
                # Find positional category
                assigned = False
                for start, end, cat_name in _PLAIN_SECTIONS:
                    if start <= str_idx < end:
                        cats[query] = cat_name
                        assigned = True
                        break
                if not assigned:
                    cats[query] = "Edge Cases"
                str_idx += 1
    return cats


_EVAL_QUERY_CATEGORIES = None  # Lazy-loaded on first use


def _get_eval_query_categories():
    global _EVAL_QUERY_CATEGORIES
    if _EVAL_QUERY_CATEGORIES is None:
        _EVAL_QUERY_CATEGORIES = _build_eval_query_categories()
    return _EVAL_QUERY_CATEGORIES

# ─── Halt-condition parameters (loaded from run directory at runtime) ─────────
# These module-level defaults are overwritten by load_run_config() in main().
import yaml as _yaml

DECISIVE_WIN_WINDOW = 100
DECISIVE_WIN_THRESHOLD = 0.75
PLATEAU_WINDOW = 2000
PLATEAU_MIN_IMPROVEMENT = 0.05
MAX_TRAIN_EPS = 100
EPISODE_LENGTH = 10
N_ROLLOUT_THREADS = 8
STEPS_PER_TRAIN_EP = EPISODE_LENGTH * N_ROLLOUT_THREADS  # = 80


def load_run_config(run_dir: str):
    """Load reward_config.yaml and args.yaml from a run directory.

    Overwrites module-level halt-condition and layout constants so that plots
    reflect the configuration that was actually used during training, not
    whatever the current codebase defaults to.
    """
    global DECISIVE_WIN_WINDOW, DECISIVE_WIN_THRESHOLD, PLATEAU_WINDOW
    global PLATEAU_MIN_IMPROVEMENT, MAX_TRAIN_EPS
    global EPISODE_LENGTH, N_ROLLOUT_THREADS, STEPS_PER_TRAIN_EP

    reward_cfg_path = os.path.join(run_dir, "reward_config.yaml")
    if os.path.exists(reward_cfg_path):
        with open(reward_cfg_path, "r") as f:
            rc = _yaml.safe_load(f) or {}
        DECISIVE_WIN_WINDOW = rc.get("decisive_win_window", DECISIVE_WIN_WINDOW)
        DECISIVE_WIN_THRESHOLD = rc.get("decisive_win_threshold", DECISIVE_WIN_THRESHOLD)
        PLATEAU_WINDOW = rc.get("plateau_window", PLATEAU_WINDOW)
        PLATEAU_MIN_IMPROVEMENT = rc.get("plateau_min_improvement", PLATEAU_MIN_IMPROVEMENT)
        MAX_TRAIN_EPS = rc.get("max_training_episodes", MAX_TRAIN_EPS)
        print(f"Loaded halt config from {reward_cfg_path}")
    else:
        print(f"WARNING: {reward_cfg_path} not found — using code defaults")

    args_path = os.path.join(run_dir, "args.yaml")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = _yaml.safe_load(f) or {}
        EPISODE_LENGTH = args.get("episode_length", EPISODE_LENGTH)
        N_ROLLOUT_THREADS = args.get("n_rollout_threads", N_ROLLOUT_THREADS)
        STEPS_PER_TRAIN_EP = EPISODE_LENGTH * N_ROLLOUT_THREADS
        print(f"Loaded args from {args_path}: episode_length={EPISODE_LENGTH}, n_rollout_threads={N_ROLLOUT_THREADS}")
    else:
        print(f"WARNING: {args_path} not found — using code defaults")
        STEPS_PER_TRAIN_EP = EPISODE_LENGTH * N_ROLLOUT_THREADS


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

    # --- hard episode limit ---
    # Derive approx training-episode count from env-episode count.
    # In sql_runner.py blueteam emits one episodic return per env step, so
    # entries per training episode = episode_length * n_rollout_threads.
    approx_train_eps = n // STEPS_PER_TRAIN_EP
    hard_limit_active = approx_train_eps >= MAX_TRAIN_EPS

    return {
        "dw_avg": dw_avg,
        "dw_window_size": min(n, DECISIVE_WIN_WINDOW),
        "decisive_win_active": decisive_win_active,
        "plateau_active": plateau_active,
        "plateau_recent": plateau_recent,
        "plateau_past": plateau_past,
        "approx_train_eps": approx_train_eps,
        "hard_limit_active": hard_limit_active,
        "n_env_episodes": n,
    }


# ─────────────────────────────────── Parser ─────────────────────────────────


def parse_logs(run_dir: str):
    """Parse reward_debug.jsonl and return (train_records, eval_records)."""
    for rel in ("debug_logs/reward_debug.jsonl", "reward_debug.jsonl"):
        path = os.path.join(run_dir, rel)
        if os.path.exists(path):
            break
    else:
        print("ERROR: No reward_debug.jsonl found in", run_dir)
        sys.exit(1)

    print(f"Parsing: {path}")

    train_records = []
    eval_records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("is_eval", False):
                eval_records.append(rec)
            else:
                train_records.append(rec)

    if not train_records and not eval_records:
        print("ERROR: Log file is empty.")
        sys.exit(1)

    return train_records, eval_records


def aggregate_eval(eval_records):
    """
    Aggregate eval records into structures needed for generalization plots.

    Returns a dict with:
      - benign_is_tp: list of 0/1 for each benign eval turn
      - eval_global_episodes: training episode tag (from "episode" field) per benign turn
      - eval_query_outcomes: list of (query_text, outcome_tier) for all eval benign turns
    """
    benign_is_tp = []
    eval_global_episodes = []
    eval_query_outcomes = []

    for r in eval_records:
        if r.get("turn_type") != "benign":
            continue
        outcome = r.get("outcome_tier", "unknown")
        ep = r.get("episode", 0)
        query = r.get("user_message", "")
        benign_is_tp.append(1.0 if outcome == "true_positive" else 0.0)
        eval_global_episodes.append(ep)
        eval_query_outcomes.append((query, outcome))

    return {
        "benign_is_tp": benign_is_tp,
        "eval_global_episodes": eval_global_episodes,
        "eval_query_outcomes": eval_query_outcomes,
    }


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

    # Benign per-style refusal arrays (for regression tracking)
    benign_style_refusal = {"plain": [], "adversarial": [], "multi_turn": []}
    benign_style_ep_indices = {"plain": [], "adversarial": [], "multi_turn": []}

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
            # Per-style tracking
            style = r.get("benign_style", "plain")
            if style in benign_style_refusal:
                benign_style_refusal[style].append(
                    1.0 if o == "false_negative" else 0.0
                )
                benign_style_ep_indices[style].append(ep_idx)
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
        # Benign per-style regression tracking
        "benign_style_refusal": benign_style_refusal,
        "benign_style_ep_indices": benign_style_ep_indices,
        # Attack-turn arrays
        "attack_is_tn": attack_is_tn,
        "attack_is_fp": attack_is_fp,
        "attack_is_neutral": attack_is_neutral,
        "attack_is_sqlerr": attack_is_sqlerr,
        "attack_is_refusal": attack_is_refusal,
        "attack_episode_indices": attack_episode_indices,
    }


# ─────────────────────────────── Plotting ────────────────────────────────────


def plot(run_dir: str, data: dict, eval_data: dict) -> str:
    steps = data["steps"]
    rewards = data["rewards"]
    W = 50  # rolling window

    # ── Compute halt metrics ─────────────────────────────────────────────────
    halt = compute_halt_status(rewards)
    dw_xs, dw_avgs = compute_decisive_win_series(rewards)

    # ── Build figure: 6 rows × 2 cols ────────────────────────────────────────
    fig, axes = plt.subplots(6, 2, figsize=(16, 36))
    fig.suptitle(
        f"Blue Team Training — {os.path.basename(run_dir)}", fontsize=14, y=0.99
    )

    (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12) = axes.flatten()

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

    # Per-style benign refusal breakdown (regression detector)
    _STYLE_COLORS = {"plain": "#95a5a6", "adversarial": "#e74c3c", "multi_turn": "#8e44ad"}
    _STYLE_LABELS = {"plain": "Plain benign FN", "adversarial": "Adversarial benign FN", "multi_turn": "Multi-turn benign FN"}
    for style in ("plain", "adversarial", "multi_turn"):
        style_ref = data.get("benign_style_refusal", {}).get(style, [])
        style_eps_arr = data.get("benign_style_ep_indices", {}).get(style, [])
        if len(style_ref) >= 2:
            style_W = min(W, max(1, len(style_ref) // 2))
            rx_s = rolling_x(len(style_ref), style_W)
            ra_s = compute_rolling(style_ref, style_W)
            if len(rx_s) == len(ra_s) and len(rx_s) > 0:
                ep_arr = np.array(style_eps_arr)
                ax7.plot(
                    ep_arr[rx_s],
                    ra_s * 100,
                    color=_STYLE_COLORS[style],
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.8,
                    label=_STYLE_LABELS[style],
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
    approx_train_eps = halt["approx_train_eps"]
    hard_limit_pct = min(100.0, approx_train_eps / MAX_TRAIN_EPS * 100)

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
            f"   Condition: recent {PLATEAU_WINDOW}-ep avg − past {PLATEAU_WINDOW}-ep avg < {PLATEAU_MIN_IMPROVEMENT}",
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
            "3. blueteam_max_episodes_reached",
            None,
            11,
            "bold",
            cond_color(halt["hard_limit_active"]),
        ),
        (
            f"   Condition: training_episode + 1 ≥ {MAX_TRAIN_EPS}",
            None,
            9,
            "normal",
            "#555",
        ),
        (
            f"   Approx training episodes: ~{approx_train_eps}  ({hard_limit_pct:.1f}% of limit)",
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

    # ── Plot 11: Train vs Eval Generalization Gap ──────────────────────────────
    # Rolling training TP rate (line) + per-eval-run TP rate (scatter)
    train_benign_eps = np.array(data["benign_episode_indices"])
    if len(data["benign_is_tp"]) >= 2:
        rx = rolling_x(len(data["benign_is_tp"]), W)
        ra = compute_rolling(data["benign_is_tp"], W)
        if len(rx) == len(ra) and len(rx) > 0:
            ax11.plot(
                train_benign_eps[rx],
                ra * 100,
                color="#27ae60",
                linewidth=2,
                label=f"Train TP rate ({W}-ep rolling)",
            )

    # Scatter eval TP rate per eval run — group by episode number
    if eval_data and eval_data.get("benign_is_tp"):
        eval_tp_raw = eval_data.get("benign_is_tp", [])
        eval_ep_idx_raw = eval_data.get("eval_global_episodes", [])

        # Group eval benign TP by their global training episode tag
        ep_groups = defaultdict(list)
        for ep_tag, is_tp in zip(eval_ep_idx_raw, eval_tp_raw):
            ep_groups[ep_tag].append(is_tp)

        if ep_groups:
            eval_x = sorted(ep_groups.keys())
            eval_y = [np.mean(ep_groups[ep]) * 100 for ep in eval_x]
            ax11.scatter(
                eval_x,
                eval_y,
                color="#e74c3c",
                s=60,
                zorder=5,
                label="Eval TP rate (held-out queries)",
            )
            # Annotate final gap
            if len(eval_x) > 0 and len(data["benign_is_tp"]) >= 2:
                final_eval_ep = eval_x[-1]
                final_eval_tp = eval_y[-1]
                # Interpolate training TP at same episode
                if len(train_benign_eps) > 0 and len(data["benign_is_tp"]) >= 2:
                    rx2 = rolling_x(len(data["benign_is_tp"]), W)
                    ra2 = compute_rolling(data["benign_is_tp"], W)
                    if len(rx2) == len(ra2) and len(rx2) > 0:
                        train_x_arr = train_benign_eps[rx2]
                        train_tp_interp = float(
                            np.interp(final_eval_ep, train_x_arr, ra2 * 100)
                        )
                        gap = train_tp_interp - final_eval_tp
                        gap_color = "#27ae60" if abs(gap) < 10 else "#e74c3c"
                        ax11.annotate(
                            f"Gap: {gap:+.1f}%",
                            xy=(final_eval_ep, final_eval_tp),
                            xytext=(20, 10),
                            textcoords="offset points",
                            fontsize=9,
                            fontweight="bold",
                            color=gap_color,
                            arrowprops=dict(arrowstyle="->", color=gap_color),
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor=gap_color,
                                alpha=0.9,
                            ),
                        )
    else:
        ax11.text(
            0.5,
            0.5,
            "No eval data found\n(run with --use_eval)",
            transform=ax11.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#aaa",
        )

    ax11.set_title(
        "Train vs Eval Generalization Gap\n"
        "(scatter = held-out BENIGN_EVAL_QUERIES; close to line = generalizing)"
    )
    ax11.set_xlabel("Episode (global)")
    ax11.set_ylabel("TP Rate (%)")
    ax11.set_ylim(-2, 105)
    ax11.legend(loc="lower right", fontsize="small")
    ax11.grid(True, alpha=0.3)

    # ── Plot 12: Eval TP Rate by Query Category ────────────────────────────────
    if eval_data and eval_data.get("eval_query_outcomes"):
        cat_tp = defaultdict(list)
        for query, outcome in eval_data["eval_query_outcomes"]:
            cat = _get_eval_query_categories().get(query, "Unknown")
            cat_tp[cat].append(1.0 if outcome == "true_positive" else 0.0)

        if cat_tp:
            sorted_cats = sorted(cat_tp.keys())
            cat_means = [np.mean(cat_tp[c]) * 100 for c in sorted_cats]
            bar_colors = ["#27ae60" if m >= 70 else "#e67e22" if m >= 40 else "#e74c3c" for m in cat_means]
            bars = ax12.bar(range(len(sorted_cats)), cat_means, color=bar_colors, edgecolor="white")
            ax12.set_xticks(range(len(sorted_cats)))
            ax12.set_xticklabels(sorted_cats, rotation=30, ha="right", fontsize=8)
            ax12.set_ylim(0, 105)
            ax12.axhline(80, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.7, label="80% target")
            for bar, val in zip(bars, cat_means):
                ax12.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax12.legend(fontsize="small")
        else:
            ax12.text(0.5, 0.5, "No eval category data", transform=ax12.transAxes, ha="center", va="center", color="#aaa")
    else:
        ax12.text(
            0.5,
            0.5,
            "No eval data found\n(run with --use_eval)",
            transform=ax12.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#aaa",
        )

    ax12.set_title("Eval TP Rate by Query Category\n(held-out BENIGN_EVAL_QUERIES only)")
    ax12.set_xlabel("Category")
    ax12.set_ylabel("TP Rate (%)")
    ax12.grid(True, alpha=0.3, axis="y")

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

    load_run_config(run_dir)

    train_records, eval_records = parse_logs(run_dir)
    print(f"Loaded {len(train_records)} train entries, {len(eval_records)} eval entries.")

    if not train_records:
        print("ERROR: No training records found.")
        sys.exit(1)

    data = aggregate(train_records)
    eval_data = aggregate_eval(eval_records) if eval_records else {}
    halt = compute_halt_status(data["rewards"])

    # Print a concise summary to stdout
    print("\n=== Halt Condition Summary ===")
    print(f"  Env episodes:   {halt['n_env_episodes']}")
    print(f"  Approx train eps: ~{halt['approx_train_eps']} / {MAX_TRAIN_EPS}")
    print(
        f"  [1] decisive_win: rolling-{DECISIVE_WIN_WINDOW} avg = {halt['dw_avg']:.4f}  "
        f"(threshold {DECISIVE_WIN_THRESHOLD})  → {'TRIGGERED' if halt['decisive_win_active'] else 'not yet'}"
    )
    print(f"  [2] plateaued:   {'TRIGGERED' if halt['plateau_active'] else 'not yet'}")
    print(
        f"  [3] max_episodes: {'TRIGGERED' if halt['hard_limit_active'] else 'not yet'}"
    )

    # Print train vs eval generalization summary
    if eval_data and eval_data.get("benign_is_tp"):
        eval_tp_mean = np.mean(eval_data["benign_is_tp"]) * 100
        train_tp = data.get("benign_is_tp", [])
        train_tp_mean = np.mean(train_tp[-500:]) * 100 if len(train_tp) >= 500 else (np.mean(train_tp) * 100 if train_tp else float("nan"))
        print(f"\n=== Generalization Summary ===")
        print(f"  Train TP rate (last 500 benign turns): {train_tp_mean:.1f}%")
        print(f"  Eval TP rate  (held-out queries):      {eval_tp_mean:.1f}%")
        print(f"  Gap (train - eval):                    {train_tp_mean - eval_tp_mean:+.1f}%")
        if abs(train_tp_mean - eval_tp_mean) < 10:
            print("  → Small gap: model appears to GENERALIZE beyond training queries.")
        else:
            print("  → Large gap: possible MEMORIZATION of training query set.")
    print()

    plot(run_dir, data, eval_data)


if __name__ == "__main__":
    main()
