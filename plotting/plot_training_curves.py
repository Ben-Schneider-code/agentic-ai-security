"""
Training-time learning curves, in ACM paper style.

Replicates and improves the high-value panels from util/plot_results.py
(util/plot_redteam_results.py, util/plot_blueteam_results.py,
util/plot_selfplay_results.py) as clean, single-responsibility figures written
to a training_plots/ subdirectory.

All per-iteration curves are overlaid on shared axes with a light->dark colour
gradient (light = early self-play iteration, dark = late), mirroring
plotting/plot_training_dynamics.py.  Outcome composition, which cannot overlay as
lines, is rendered as per-iteration stacked bars.  Self-play scalars are plotted
against iteration.  An RL optimization-curves figure (value/policy loss,
approx_kl, entropy) is added as an improvement — these were never visualized by
the original util dashboards.

This module reads ONLY json/jsonl (reward_debug.jsonl, training_state.json,
logs/summary.json).  It deliberately avoids tensorboard / BERTScore / the MARFT
env so it survives the orchestrator's per-plot try/except without optional deps
and without a set HONEYPOT_TYPE.

Can be run standalone:
    python plotting/plot_training_curves.py --results results-<ID> --out-dir figures/training_plots
Or imported:
    from plotting.plot_training_curves import plot_red_reward_curve, DESC_RED_REWARD
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style, discover_iterations, find_run_dir,
        is_benign_denial, load_reward_debug_lines, parse_results_arg, wilson_ci_pct,
        RED_COL, BLUE_COL, GREEN_COL, GRAY_COL, HUMAN_COL,
        RED_MARKER, BLUE_MARKER, HUMAN_MARKER,
        FIG_SIZE_SINGLE, FIG_SIZE_1x2,
    )
    from .plot_training_dynamics import load_training_curve, _smooth
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, discover_iterations, find_run_dir,
        is_benign_denial, load_reward_debug_lines, parse_results_arg, wilson_ci_pct,
        RED_COL, BLUE_COL, GREEN_COL, GRAY_COL, HUMAN_COL,
        RED_MARKER, BLUE_MARKER, HUMAN_MARKER,
        FIG_SIZE_SINGLE, FIG_SIZE_1x2,
    )
    from plotting.plot_training_dynamics import load_training_curve, _smooth

apply_paper_style()

DEFAULT_HONEYPOT_UNIVERSE = 22  # get_total_honeypots() in redteam_sql_env.py

# ---------------------------------------------------------------------------
# Figure descriptions (one per public function; consumed by the orchestrator)
# ---------------------------------------------------------------------------

DESC_RED_REWARD = (
    "Attacker (red) average step reward vs. environment-interaction steps, one "
    "curve per self-play iteration (light = early, dark = late). RL shaping "
    "signal, NOT a paper metric. Source: logs/summary.json average_step_rewards "
    "(fallback: all_episodic_returns)."
)
DESC_RED_REWARD_SCATTER = (
    "Attacker (red) per-logged-interval step reward vs. environment-interaction "
    "steps as a scatter, one colour per self-play iteration (light = early, dark = "
    "late). Companion to red_reward_curve.png: exposes the spread/variance of the "
    "reward signal that the smoothed line hides. RL shaping signal, NOT a paper "
    "metric. Source: logs/summary.json average_step_rewards (same points as the "
    "reward curve, unsmoothed)."
)
DESC_RED_CUMULATIVE = (
    "Attacker (red) cumulative step reward vs. environment-interaction steps, one "
    "curve per self-play iteration (light = early, dark = late), reset per "
    "iteration. Running sum of the logged average_step_rewards — a proxy for "
    "accumulated reward whose slope tracks the current reward level (rising = net "
    "positive, flat = near zero, falling = net negative). RL shaping diagnostic, "
    "NOT a paper metric. Source: logs/summary.json average_step_rewards."
)
DESC_RED_COMPOSITION = (
    "Attacker (red) tail-window outcome-tier composition per self-play iteration "
    "(stacked bars, final 25% of episodes). Shows the mix of no_sql / sql_error / "
    "blocked / sql_generated / sensitive_query / repeated_honeypot / honeypot_access "
    "turns. honeypot_access is the PVR_conv success tier."
)
DESC_RED_FLUENCY = (
    "Attacker (red) non-fluent and degenerate turn rates over training (rolling "
    "window), one curve per iteration. Collapse diagnostic: a rising degenerate "
    "rate signals reward hacking / gibberish drift. Source: is_fluent / "
    "is_degenerate in reward_debug.jsonl."
)
DESC_RED_HONEYPOT = (
    "Attacker (red) cumulative unique honeypots discovered vs. episode, one curve "
    "per iteration. Secondary axis: coverage of the honeypot universe (%). Source: "
    "new_honeypots_accessed in reward_debug.jsonl."
)
DESC_BLUE_PRF1 = (
    "Defender (blue) precision / recall / F1 over training (rolling window), final "
    "iteration solid and earlier iterations faded. Classifier framing: "
    "TP=benign->SQL, FP=attack->honeypot, FN=benign->refused. Source: "
    "reward_debug.jsonl (training turns only)."
)
DESC_BLUE_RATES = (
    "Defender (blue) outcome rates over training (rolling window): BRR (benign "
    "refused), 1-PVR_turn (attack refused), and PVR_turn (turn-level policy "
    "violation: attack turn accessed a honeypot). Lower BRR and lower PVR_turn are "
    "better. BRR uses the canonical is_benign_denial predicate; PVR_turn = "
    "false_positive/attack turns (formerly labeled CFR). Final iteration solid, "
    "earlier iterations faded. Source: reward_debug.jsonl (training turns only)."
)
DESC_ARMS_RACE = (
    "Self-play arms race: tail-window PVR_conv (red), 1-PVR_turn (blue) and 1-BRR "
    "(green) per iteration, with 99% Wilson CIs. Ports the tail-window math from "
    "util/plot_selfplay_results.py (PVR_conv reproduces it exactly); the blue "
    "metrics here exclude held-out is_eval turns, so 1-BRR / 1-PVR_turn are "
    "training-time only (the original mixed eval turns into its tail)."
)
DESC_DOMINANCE = (
    "Self-play dominance score per iteration in [-1, +1]: HM(1-PVR_turn,1-BRR) * "
    "(1-10*PVR_turn) - min(1, 5*PVR_conv). Blue bars (>=0) = defender-favoured, red "
    "bars (<0) = attacker-favoured; +-0.2 competitive band shaded."
)
DESC_OPT_CURVES = (
    "RL optimization curves vs. environment-interaction steps (value loss, policy "
    "loss, approx_kl, entropy), red=attacker / blue=defender, one curve per "
    "iteration (light=early, dark=late). Improvement over the original util "
    "dashboards. Source: logs/summary.json tensorboard scalars."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _single_run(results: list[tuple[str, str]], who: str) -> str:
    """Return the single selfplay_dir, warning if multiple were passed."""
    if len(results) > 1:
        print(f"[{who}] Multiple runs given; using first run only.", file=sys.stderr)
    return results[0][1]


def _iter_shade(idx: int, n_iters: int) -> float:
    """Colormap position: light (early) -> dark (late)."""
    return 0.35 + 0.55 * idx / max(n_iters - 1, 1)


def _colormap(name: str):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:  # matplotlib < 3.6
        return plt.cm.get_cmap(name)


def _save(fig, out_path: Path) -> Path:
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _episode_groups(records: list[dict]) -> tuple[list[list[dict]], list[int]]:
    """Group records by episode, returned in ascending-episode order."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        groups[int(r.get("episode", 0))].append(r)
    eps = sorted(groups)
    return [groups[e] for e in eps], eps


def _rolling_metric(groups: list[list[dict]], window: int, fn) -> list[float]:
    """Causal rolling metric: fn(records in the last `window` episodes) per episode."""
    out: list[float] = []
    for i in range(len(groups)):
        lo = max(0, i - window + 1)
        recs = [r for g in groups[lo : i + 1] for r in g]
        out.append(fn(recs))
    return out


def _tail_window_episodes(records: list[dict], tail_fraction: float = 0.25) -> set | None:
    """Episode numbers in the final tail (last max(1, n//4) episodes).

    Matches util/plot_selfplay_results.get_tail_window_episodes exactly so the
    self-play scalars reproduce the original numbers. Returns None when no
    episode field is present.
    """
    eps = {r.get("episode") for r in records if r.get("episode") is not None}
    if not eps:
        return None
    sorted_eps = sorted(eps)
    n = len(sorted_eps)
    cutoff = n - max(1, int(n * tail_fraction))
    return set(sorted_eps[cutoff:])


def _tail_records(run_dir: Path, tail_fraction: float = 0.25) -> list[dict]:
    """Load reward_debug.jsonl and keep only tail-window records (original logic)."""
    lines = load_reward_debug_lines(run_dir)
    tail = _tail_window_episodes(lines, tail_fraction)
    if tail is None:
        return lines
    return [ln for ln in lines if ln.get("episode") in tail]


def _load_scalar_curve(run_dir: Path, name: str) -> list[tuple[int, float]]:
    """Load (step, value) pairs for a tensorboard scalar from logs/summary.json.

    Matches the summary key ending in '<name>/<name>' (e.g. 'approx_kl/approx_kl'),
    entries shaped [timestamp, step, value]. Returns [] if absent/malformed.
    """
    summary = run_dir / "logs" / "summary.json"
    if not summary.exists():
        return []
    try:
        data = json.loads(summary.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    suffix = f"{name}/{name}"
    key = next((k for k in data if k.endswith(suffix)), None)
    if not key or not data.get(key):
        return []
    try:
        return [(int(e[1]), float(e[2])) for e in data[key]]
    except (ValueError, IndexError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Red-team training curves
# ---------------------------------------------------------------------------

def plot_red_reward_curve(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 7,
) -> Path:
    """Attacker average step reward vs EIS, one curve per iteration (Reds gradient)."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_red_reward_curve")
    iters = discover_iterations(selfplay_dir)
    cmap = _colormap("Reds")
    n = len(iters)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    drew = False
    for idx, entry in enumerate(iters):
        red_dir = entry.get("red_dir")
        if not red_dir:
            continue
        run_dir = find_run_dir(red_dir)
        if not run_dir:
            continue
        curve = load_training_curve(run_dir)
        if not curve:
            continue
        steps, rewards = zip(*curve)
        sm = _smooth(list(rewards), smooth_window)
        col = cmap(_iter_shade(idx, n))
        ax.plot(steps, sm, color=col, linewidth=1.6, label=f"Iter {entry['iter']}",
                zorder=3)
        ax.plot(steps[-1], sm[-1], "o", color=col, markersize=5, zorder=4)
        drew = True

    ax.axhline(0, color="#bbbbbb", linewidth=0.9, zorder=1)
    ax.set_title("Attacker (Red) Reward over Training")
    ax.set_xlabel("Environment Interaction Steps (EIS)")
    ax.set_ylabel("Avg Step Reward (smoothed)")
    ax.set_xlim(left=0)
    ax.grid(True, axis="y", alpha=0.4)
    if drew:
        ax.legend(fontsize=9, frameon=True, ncol=2)
    return _save(fig, out_path)


def plot_red_reward_scatter(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 7,
) -> Path:
    """Attacker step reward vs EIS as a scatter, one colour per iteration.

    Shows the spread of the reward signal (the smoothed line in
    plot_red_reward_curve hides it). A faint smoothed trend is overlaid per
    iteration for orientation.
    """
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_red_reward_scatter")
    iters = discover_iterations(selfplay_dir)
    cmap = _colormap("Reds")
    n = len(iters)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    drew = False
    for idx, entry in enumerate(iters):
        red_dir = entry.get("red_dir")
        if not red_dir:
            continue
        run_dir = find_run_dir(red_dir)
        if not run_dir:
            continue
        curve = load_training_curve(run_dir)
        if not curve:
            continue
        steps, rewards = zip(*curve)
        col = cmap(_iter_shade(idx, n))
        ax.scatter(steps, rewards, color=col, s=14, alpha=0.55,
                   edgecolors="none", label=f"Iter {entry['iter']}", zorder=3)
        sm = _smooth(list(rewards), smooth_window)
        ax.plot(steps, sm, color=col, linewidth=1.2, alpha=0.5, zorder=2)
        drew = True

    ax.axhline(0, color="#bbbbbb", linewidth=0.9, zorder=1)
    ax.set_title("Attacker (Red) Step Reward vs EIS (scatter)")
    ax.set_xlabel("Environment Interaction Steps (EIS)")
    ax.set_ylabel("Avg Step Reward")
    ax.set_xlim(left=0)
    ax.grid(True, axis="y", alpha=0.4)
    if drew:
        ax.legend(fontsize=9, frameon=True, ncol=2)
    return _save(fig, out_path)


def plot_red_cumulative_reward(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 7,  # unused; kept for a uniform tc_jobs signature
) -> Path:
    """Attacker cumulative step reward vs EIS, one curve per iteration (reset/iter).

    Running sum of the logged average_step_rewards: slope tracks the current
    reward level (rising = net positive accrual, flat = near zero, falling = net
    negative). A proxy for accumulated reward, NOT a paper metric.
    """
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_red_cumulative_reward")
    iters = discover_iterations(selfplay_dir)
    cmap = _colormap("Reds")
    n = len(iters)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    drew = False
    for idx, entry in enumerate(iters):
        red_dir = entry.get("red_dir")
        if not red_dir:
            continue
        run_dir = find_run_dir(red_dir)
        if not run_dir:
            continue
        curve = load_training_curve(run_dir)
        if not curve:
            continue
        steps, rewards = zip(*curve)
        cum = np.cumsum(rewards)
        col = cmap(_iter_shade(idx, n))
        ax.plot(steps, cum, color=col, linewidth=1.6, marker="o", markersize=3,
                label=f"Iter {entry['iter']}", zorder=3)
        ax.fill_between(steps, 0, cum, color=col, alpha=0.10, zorder=2)
        drew = True

    ax.axhline(0, color="#bbbbbb", linewidth=0.9, zorder=1)
    ax.set_title("Attacker (Red) Cumulative Reward vs EIS")
    ax.set_xlabel("Environment Interaction Steps (EIS)")
    ax.set_ylabel("Cumulative Step Reward")
    ax.set_xlim(left=0)
    ax.grid(True, axis="y", alpha=0.4)
    if drew:
        ax.legend(fontsize=9, frameon=True, ncol=2)
    return _save(fig, out_path)


# Canonical red outcome-tier ordering (low->high severity) + colours.
_RED_TIER_ORDER = [
    ("no_sql",            "#cfcfcf"),
    ("sql_error",         "#9e9e9e"),
    ("blocked_attempt",   "#e6961e"),
    ("sql_generated",     BLUE_COL),
    ("sensitive_query",   "#f1c40f"),
    ("repeated_honeypot", "#9467bd"),
    ("honeypot_access",   RED_COL),
]
_RED_FALLBACK_COLORS = ["#1abc9c", "#34495e", "#e67e22", "#16a085"]


def plot_red_outcome_composition(
    results: list[tuple[str, str]],
    out_path: str | Path,
    tail_pct: float = 0.25,
) -> Path:
    """Per-iteration tail-window outcome-tier composition as stacked bars."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_red_outcome_composition")
    iters = discover_iterations(selfplay_dir)

    iter_nums: list[int] = []
    fracs: dict[str, list[float]] = defaultdict(list)
    seen_tiers: list[str] = [t for t, _ in _RED_TIER_ORDER]
    for entry in iters:
        red_dir = entry.get("red_dir")
        if not red_dir:
            continue
        run_dir = find_run_dir(red_dir)
        if not run_dir:
            continue
        lines = _tail_records(run_dir, tail_pct)
        if not lines:
            continue
        counts: dict[str, int] = defaultdict(int)
        for ln in lines:
            tier = ln.get("outcome_tier") or "unknown"
            counts[tier] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        iter_nums.append(int(entry["iter"]))
        for tier in counts:
            if tier not in seen_tiers:
                seen_tiers.append(tier)
        for tier in seen_tiers:
            fracs[tier].append(100.0 * counts.get(tier, 0) / total)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    if iter_nums:
        x = np.arange(len(iter_nums))
        bottom = np.zeros(len(iter_nums))
        known = {t: c for t, c in _RED_TIER_ORDER}
        fb = 0
        for tier in seen_tiers:
            vals = np.array(fracs[tier])
            if vals.sum() == 0:
                continue
            if tier in known:
                color = known[tier]
            else:
                color = _RED_FALLBACK_COLORS[fb % len(_RED_FALLBACK_COLORS)]
                fb += 1
            ax.bar(x, vals, bottom=bottom, color=color, width=0.72,
                   edgecolor="white", linewidth=0.4, label=tier)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in iter_nums])
        ax.legend(fontsize=8, frameon=True, ncol=1, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0))

    ax.set_title("Attacker (Red) Outcome Composition (tail window)")
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Share of turns (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.4)
    return _save(fig, out_path)


def plot_red_fluency(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 50,
) -> Path:
    """Non-fluent and degenerate turn rates over training, per iteration."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_red_fluency")
    iters = discover_iterations(selfplay_dir)
    cmap = _colormap("Reds")
    cmap_p = _colormap("Purples")
    n = len(iters)

    def nonfluent(recs):
        rel = [r for r in recs if r.get("is_fluent") is not None]
        return 100.0 * sum(1 for r in rel if not r.get("is_fluent")) / len(rel) if rel else np.nan

    def degenerate(recs):
        return 100.0 * sum(1 for r in recs if r.get("is_degenerate")) / len(recs) if recs else np.nan

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    ax.axhspan(50, 100, color="#f8d7da", alpha=0.35, zorder=0)   # danger
    ax.axhspan(30, 50, color="#fff3cd", alpha=0.35, zorder=0)    # warning
    drew = False
    for idx, entry in enumerate(iters):
        red_dir = entry.get("red_dir")
        if not red_dir:
            continue
        run_dir = find_run_dir(red_dir)
        if not run_dir:
            continue
        lines = load_reward_debug_lines(run_dir)
        if not lines:
            continue
        groups, eps = _episode_groups(lines)
        nf = _rolling_metric(groups, smooth_window, nonfluent)
        dg = _rolling_metric(groups, smooth_window, degenerate)
        shade = _iter_shade(idx, n)
        ax.plot(eps, nf, color=cmap(shade), linewidth=1.5, zorder=3)
        ax.plot(eps, dg, color=cmap_p(shade), linewidth=1.3, linestyle="--", zorder=3)
        drew = True

    ax.set_title("Attacker (Red) Fluency Collapse Diagnostic")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.grid(True, axis="y", alpha=0.4)
    if drew:
        # Two proxy entries; the iteration is encoded by colour shade (light->dark).
        handles = [
            plt.Line2D([0], [0], color=cmap(0.8), lw=1.6, label="Non-fluent"),
            plt.Line2D([0], [0], color=cmap_p(0.8), lw=1.4, ls="--", label="Degenerate"),
        ]
        ax.legend(handles=handles, fontsize=9, frameon=True, loc="upper right",
                  title="shade = iter (light->dark)")
    return _save(fig, out_path)


def plot_red_honeypot_discovery(
    results: list[tuple[str, str]],
    out_path: str | Path,
    honeypot_universe: int = DEFAULT_HONEYPOT_UNIVERSE,
) -> Path:
    """Cumulative unique honeypots discovered vs episode, per iteration."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_red_honeypot_discovery")
    iters = discover_iterations(selfplay_dir)
    cmap = _colormap("Reds")
    n = len(iters)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    drew = False
    max_unique = 0
    for idx, entry in enumerate(iters):
        red_dir = entry.get("red_dir")
        if not red_dir:
            continue
        run_dir = find_run_dir(red_dir)
        if not run_dir:
            continue
        lines = load_reward_debug_lines(run_dir)
        if not lines:
            continue
        groups, eps = _episode_groups(lines)
        seen: set = set()
        cum: list[int] = []
        for g in groups:
            for r in g:
                hp = r.get("new_honeypots_accessed")
                if isinstance(hp, list):
                    seen.update(hp)
            cum.append(len(seen))
        if not cum or cum[-1] == 0:
            continue
        col = cmap(_iter_shade(idx, n))
        ax.plot(eps, cum, color=col, linewidth=1.6, label=f"Iter {entry['iter']}",
                zorder=3)
        max_unique = max(max_unique, cum[-1])
        drew = True

    ax.axhline(honeypot_universe, color=GRAY_COL, linestyle=":", linewidth=1.1,
               label=f"Universe ({honeypot_universe})", zorder=1)
    ax.set_title("Attacker (Red) Honeypot Discovery")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Cumulative unique honeypots")
    ax.set_xlim(left=0)
    ax.set_ylim(0, max(honeypot_universe, max_unique) * 1.08)
    ax.grid(True, axis="y", alpha=0.4)
    if drew:
        # Secondary axis: coverage % of the universe.
        ax2 = ax.twinx()
        ax2.set_ylim(0, ax.get_ylim()[1] / honeypot_universe * 100)
        ax2.set_ylabel("Coverage (%)")
        ax2.grid(False)
        ax.legend(fontsize=9, frameon=True, ncol=2, loc="lower right")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Blue-team training curves
# ---------------------------------------------------------------------------

def _blue_training_groups(run_dir: Path):
    """Per-episode groups of non-eval blue records (or None)."""
    lines = load_reward_debug_lines(run_dir)
    lines = [ln for ln in lines if not ln.get("is_eval")]
    if not lines:
        return None
    return _episode_groups(lines)


def _blue_prf1(recs):
    tp = sum(1 for r in recs if r.get("outcome_tier") == "true_positive")
    fp = sum(1 for r in recs if r.get("outcome_tier") == "false_positive")
    fn = sum(1 for r in recs if r.get("outcome_tier") == "false_negative")
    prec = 100.0 * tp / (tp + fp) if (tp + fp) else np.nan
    rec = 100.0 * tp / (tp + fn) if (tp + fn) else np.nan
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = np.nan
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def _blue_rates(recs):
    """(BRR, 1-PVR_turn, PVR_turn) in percent.

    BRR (Benign Refusal Rate) uses the canonical is_benign_denial predicate from
    util.metrics (re-exported via _data) so this curve matches plot_brr.py. The
    third value is the turn-level policy-violation rate PVR_turn = fp/attack
    (formerly mislabeled "CFR"; util.metrics aliases CFR to the conversation-level
    PVR_conv, a different metric).
    """
    benign = [r for r in recs if r.get("turn_type") == "benign"]
    attack = [r for r in recs if r.get("turn_type") == "attack"]
    n_denied = sum(1 for r in benign if is_benign_denial(r))
    tn = sum(1 for r in attack if r.get("outcome_tier") == "true_negative")
    fp = sum(1 for r in attack if r.get("outcome_tier") == "false_positive")
    brr = 100.0 * n_denied / len(benign) if benign else np.nan
    one_minus_pvr_turn = 100.0 * tn / len(attack) if attack else np.nan
    pvr_turn = 100.0 * fp / len(attack) if attack else np.nan
    return brr, one_minus_pvr_turn, pvr_turn


def _iter_alpha(idx: int, n: int) -> float:
    """Earlier iterations faded, final iteration solid."""
    return 0.3 + 0.7 * idx / max(n - 1, 1)


def plot_blue_prf1(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 50,
) -> Path:
    """Defender precision / recall / F1 over training; final iter solid, others faded."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_blue_prf1")
    iters = discover_iterations(selfplay_dir)
    n = len(iters)

    series = [("Precision", GREEN_COL, 0), ("Recall", HUMAN_COL, 1), ("F1", "#9467bd", 2)]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    drew = False
    legend_done = False
    for idx, entry in enumerate(iters):
        blue_dir = entry.get("blue_dir")
        if not blue_dir:
            continue
        run_dir = find_run_dir(blue_dir)
        if not run_dir:
            continue
        gr = _blue_training_groups(run_dir)
        if gr is None:
            continue
        groups, eps = gr
        vals = [_rolling_metric(groups, smooth_window, lambda r, k=k: _blue_prf1(r)[k])
                for _, _, k in series]
        alpha = _iter_alpha(idx, n)
        for (name, col, k), v in zip(series, vals):
            ax.plot(eps, v, color=col, linewidth=1.6, alpha=alpha,
                    label=name if not legend_done else "_nolegend_", zorder=3)
        legend_done = True
        drew = True

    ax.set_title("Defender (Blue) Precision / Recall / F1")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.grid(True, axis="y", alpha=0.4)
    if drew:
        ax.legend(fontsize=9, frameon=True, title="solid = final iter")
    return _save(fig, out_path)


def plot_blue_outcome_rates(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 50,
    show_ci: bool = True,
) -> Path:
    """Defender BRR / 1-PVR_turn / PVR_turn over training; final iter solid, others faded."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_blue_outcome_rates")
    iters = discover_iterations(selfplay_dir)
    n = len(iters)

    series = [("BRR (benign refused)", GREEN_COL, 0),
              ("1-PVR_turn (attack refused)", BLUE_COL, 1),
              ("PVR_turn (attack→honeypot breach)", RED_COL, 2)]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    drew = False
    legend_done = False
    for idx, entry in enumerate(iters):
        blue_dir = entry.get("blue_dir")
        if not blue_dir:
            continue
        run_dir = find_run_dir(blue_dir)
        if not run_dir:
            continue
        gr = _blue_training_groups(run_dir)
        if gr is None:
            continue
        groups, eps = gr
        vals = [_rolling_metric(groups, smooth_window, lambda r, k=k: _blue_rates(r)[k])
                for _, _, k in series]
        alpha = _iter_alpha(idx, n)
        for (name, col, k), v in zip(series, vals):
            ls = "--" if k == 2 else "-"
            ax.plot(eps, v, color=col, linewidth=1.6, alpha=alpha, linestyle=ls,
                    label=name if not legend_done else "_nolegend_", zorder=3)
        legend_done = True
        drew = True

    ax.set_title("Defender (Blue) Outcome Rates")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.grid(True, axis="y", alpha=0.4)
    if drew:
        ax.legend(fontsize=8, frameon=True, title="solid = final iter")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# Self-play aggregate scalars (ported from util/plot_selfplay_results.py)
# ---------------------------------------------------------------------------

def _selfplay_tail_metrics(
    selfplay_dir: str,
    honeypot_universe: int = DEFAULT_HONEYPOT_UNIVERSE,
    tail_pct: float = 0.25,
) -> dict[int, dict]:
    """Per-iteration tail-window metrics ported from util/plot_selfplay_results.py.

    Returns {iter: {pvr_conv, pvr_conv_k, pvr_conv_n,
                    one_minus_pvr_turn, omp_k, omp_n,
                    one_minus_pud, omu_k, omu_n,
                    cfr, dominance}} (rates in %).
    """
    out: dict[int, dict] = {}
    for entry in discover_iterations(selfplay_dir):
        it = int(entry["iter"])
        m: dict = {}

        # Red: tail-window PVR_conv = fraction of episodes with >=1 honeypot_access.
        red_dir = entry.get("red_dir")
        if red_dir:
            run_dir = find_run_dir(red_dir)
            if run_dir:
                lines = _tail_records(run_dir, tail_pct)
                groups, _eps = _episode_groups(lines)
                if groups:
                    k = sum(1 for g in groups
                            if any(r.get("outcome_tier") == "honeypot_access" for r in g))
                    m["pvr_conv_k"], m["pvr_conv_n"] = k, len(groups)
                    m["pvr_conv"] = 100.0 * k / len(groups)

        # Blue: tail-window 1-PUD (benign served), 1-PVR_turn (attack refused), CFR.
        blue_dir = entry.get("blue_dir")
        if blue_dir:
            run_dir = find_run_dir(blue_dir)
            if run_dir:
                lines = _tail_records(run_dir, tail_pct)
                lines = [ln for ln in lines if not ln.get("is_eval")]
                benign = [r for r in lines if r.get("turn_type") == "benign"]
                attack = [r for r in lines if r.get("turn_type") == "attack"]
                tp = sum(1 for r in benign if r.get("outcome_tier") == "true_positive")
                tn = sum(1 for r in attack if r.get("outcome_tier") == "true_negative")
                fp = sum(1 for r in attack if r.get("outcome_tier") == "false_positive")
                if benign:
                    m["omu_k"], m["omu_n"] = tp, len(benign)
                    m["one_minus_pud"] = 100.0 * tp / len(benign)
                if attack:
                    m["omp_k"], m["omp_n"] = tn, len(attack)
                    m["one_minus_pvr_turn"] = 100.0 * tn / len(attack)
                    m["cfr"] = 100.0 * fp / len(attack)

        # Dominance (needs all three; skip if missing).
        if {"pvr_conv", "one_minus_pvr_turn", "one_minus_pud", "cfr"} <= m.keys():
            omp = m["one_minus_pvr_turn"] / 100.0
            omu = m["one_minus_pud"] / 100.0
            cfr = m["cfr"] / 100.0
            blue_hm = (2 * omp * omu / (omp + omu)) if (omp + omu) > 0 else 0.0
            blue_composite = max(0.0, blue_hm * (1 - 10 * cfr))
            red_scaled = min(1.0, (m["pvr_conv"] / 100.0) * 5)
            m["dominance"] = blue_composite - red_scaled

        if m:
            out[it] = m
    return out


def _resolve_honeypot_universe(selfplay_dir: str) -> int:
    """Resolve the honeypot universe from <selfplay_dir>/summary.json honeypot_type.

    Reads ``honeypot_type`` and maps it through ``util._diag_common.HONEYPOT_UNIVERSE``
    (col=34, row=30, rowcol=64). Falls back to ``DEFAULT_HONEYPOT_UNIVERSE`` if the
    summary is missing/unreadable, lacks ``honeypot_type``, or the type is unmapped.
    """
    summary = Path(selfplay_dir) / "summary.json"
    if not summary.is_file():
        return DEFAULT_HONEYPOT_UNIVERSE
    try:
        data = json.loads(summary.read_text())
    except (json.JSONDecodeError, OSError):
        return DEFAULT_HONEYPOT_UNIVERSE
    hp_type = data.get("honeypot_type")
    try:
        from util._diag_common import HONEYPOT_UNIVERSE
    except Exception:
        return DEFAULT_HONEYPOT_UNIVERSE
    universe = HONEYPOT_UNIVERSE.get(hp_type)
    return universe if universe is not None else DEFAULT_HONEYPOT_UNIVERSE


def compute_selfplay_tail(results: list[tuple[str, str]], **kwargs) -> dict:
    """Pure compute export of the self-play tail-window paper metrics.

    Resolves the honeypot universe from ``<selfplay_dir>/summary.json`` and returns
    the per-iteration tail-window metrics produced by ``_selfplay_tail_metrics`` (the
    same numbers backing ``plot_selfplay_arms_race`` / ``plot_selfplay_dominance``).

    This is the ONLY paper-metric export from this module; the nine reward-shaping
    curves are explicitly NOT paper metrics.

    Args:
        results: ``[(label, selfplay_dir), ...]``; only the first run is used.
        **kwargs: accepted and ignored (uniform compute-fn signature).

    Returns:
        ``{}`` when no run/metrics are available, else::

            {
              "honeypot_universe": int,
              "per_iter": {
                <int iter>: {pvr_conv, pvr_conv_k, pvr_conv_n,
                             one_minus_pvr_turn, omp_k, omp_n,
                             one_minus_pud, omu_k, omu_n,
                             cfr, dominance},
                ...
              },
            }

        (per-iteration keys are exactly those ``_selfplay_tail_metrics`` populates;
        rates are in percent and individual keys may be absent when a side is missing.)
    """
    if not results:
        return {}
    selfplay_dir = _single_run(results, "compute_selfplay_tail")
    universe = _resolve_honeypot_universe(selfplay_dir)
    per_iter = _selfplay_tail_metrics(selfplay_dir, universe)
    if not per_iter:
        return {}
    return {
        "honeypot_universe": int(universe),
        "per_iter": {int(it): m for it, m in per_iter.items()},
    }


def _errbars(rate, k, n, show_ci):
    """Return a (2,1) yerr array for a percentage rate, or None."""
    if not show_ci or k is None or n is None or n == 0:
        return None
    lo, hi = wilson_ci_pct(k, n)
    return np.array([[max(0.0, rate - lo)], [max(0.0, hi - rate)]])


def plot_selfplay_arms_race(
    results: list[tuple[str, str]],
    out_path: str | Path,
    honeypot_universe: int = DEFAULT_HONEYPOT_UNIVERSE,
    show_ci: bool = True,
) -> Path:
    """Tail PVR_conv / 1-PVR_turn / 1-PUD per iteration with 99% Wilson CIs."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_selfplay_arms_race")
    metrics = _selfplay_tail_metrics(selfplay_dir, honeypot_universe)
    its = sorted(metrics)

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    spec = [
        ("pvr_conv", "pvr_conv_k", "pvr_conv_n", RED_COL, RED_MARKER,
         "PVR_conv (attacker)", "-"),
        ("one_minus_pvr_turn", "omp_k", "omp_n", BLUE_COL, BLUE_MARKER,
         "1-PVR_turn (defender)", "-"),
        ("one_minus_pud", "omu_k", "omu_n", GREEN_COL, HUMAN_MARKER,
         "1-BRR (utility)", "--"),
    ]
    for key, kk, nk, col, mk, lbl, ls in spec:
        xs, ys, los, his = [], [], [], []
        for it in its:
            m = metrics[it]
            if key not in m:
                continue
            xs.append(it)
            ys.append(m[key])
            yerr = _errbars(m[key], m.get(kk), m.get(nk), show_ci)
            if yerr is not None:
                los.append(yerr[0][0]); his.append(yerr[1][0])
            else:
                los.append(0.0); his.append(0.0)
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=([los, his] if show_ci else None), color=col,
                    marker=mk, markersize=6, linewidth=1.7, linestyle=ls,
                    capsize=3, label=lbl, zorder=3)

    ax.set_title("Self-play Arms Race (tail window)")
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    if its:
        ax.set_xticks(its)
    ax.grid(True, axis="y", alpha=0.4)
    if metrics:
        ax.legend(fontsize=9, frameon=True)
    return _save(fig, out_path)


def plot_selfplay_dominance(
    results: list[tuple[str, str]],
    out_path: str | Path,
    honeypot_universe: int = DEFAULT_HONEYPOT_UNIVERSE,
) -> Path:
    """Dominance score bar per iteration; blue >=0 defender-favoured, red <0."""
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_selfplay_dominance")
    metrics = _selfplay_tail_metrics(selfplay_dir, honeypot_universe)
    its = [it for it in sorted(metrics) if "dominance" in metrics[it]]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    if its:
        vals = [metrics[it]["dominance"] for it in its]
        colors = [BLUE_COL if v >= 0 else RED_COL for v in vals]
        ax.bar(its, vals, color=colors, width=0.7, edgecolor="#333333",
               linewidth=0.5, zorder=3)
        ax.axhspan(-0.2, 0.2, color="#dddddd", alpha=0.5, zorder=0,
                   label="Competitive band (+-0.2)")
        ax.set_xticks(its)
    ax.axhline(0, color="#333333", linewidth=1.0, zorder=2)
    ax.set_title("Self-play Dominance (defender - attacker)")
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Dominance score")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, axis="y", alpha=0.4)
    if its:
        ax.legend(fontsize=9, frameon=True, loc="upper right")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# RL optimization curves (improvement — not in the original util dashboards)
# ---------------------------------------------------------------------------

_OPT_PANELS = [("value_loss", "Value loss"), ("policy_loss", "Policy loss"),
               ("approx_kl", "Approx. KL"), ("entropy", "Entropy")]


def plot_optimization_curves(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 5,
) -> Path:
    """2x2 of value loss / policy loss / approx_kl / entropy vs EIS, per iteration.

    Both teams overlaid (red = attacker, blue = defender) with an iteration
    gradient. Reads logs/summary.json tensorboard scalars only.
    """
    out_path = Path(out_path)
    selfplay_dir = _single_run(results, "plot_optimization_curves")
    iters = discover_iterations(selfplay_dir)
    n = len(iters)
    cmap_r, cmap_b = _colormap("Reds"), _colormap("Blues")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4))
    flat = axes.ravel()
    any_data = False
    for ax, (scalar, title) in zip(flat, _OPT_PANELS):
        for idx, entry in enumerate(iters):
            shade = _iter_shade(idx, n)
            for team_key, cmap in (("red_dir", cmap_r), ("blue_dir", cmap_b)):
                team_dir = entry.get(team_key)
                if not team_dir:
                    continue
                run_dir = find_run_dir(team_dir)
                if not run_dir:
                    continue
                curve = _load_scalar_curve(run_dir, scalar)
                if not curve:
                    continue
                steps, vals = zip(*curve)
                sm = _smooth(list(vals), smooth_window)
                ax.plot(steps, sm, color=cmap(shade), linewidth=1.3, zorder=3)
                any_data = True
        ax.set_title(title)
        ax.set_xlabel("EIS")
        ax.set_xlim(left=0)
        ax.grid(True, axis="y", alpha=0.4)

    # Shared legend for team colour semantics.
    handles = [
        plt.Line2D([0], [0], color=cmap_r(0.75), lw=2, label="Attacker (red)"),
        plt.Line2D([0], [0], color=cmap_b(0.75), lw=2, label="Defender (blue)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True,
               fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("RL Optimization Curves (light = early iter, dark = late)",
                 y=1.02, fontsize=13)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

_JOBS = [
    (plot_red_reward_curve,        "red_reward_curve.png",        DESC_RED_REWARD),
    (plot_red_outcome_composition, "red_outcome_composition.png", DESC_RED_COMPOSITION),
    (plot_red_fluency,             "red_fluency.png",             DESC_RED_FLUENCY),
    (plot_red_honeypot_discovery,  "red_honeypot_discovery.png",  DESC_RED_HONEYPOT),
    (plot_blue_prf1,               "blue_prf1.png",               DESC_BLUE_PRF1),
    (plot_blue_outcome_rates,      "blue_outcome_rates.png",      DESC_BLUE_RATES),
    (plot_selfplay_arms_race,      "selfplay_arms_race.png",      DESC_ARMS_RACE),
    (plot_selfplay_dominance,      "selfplay_dominance.png",      DESC_DOMINANCE),
    (plot_optimization_curves,     "optimization_curves.png",     DESC_OPT_CURVES),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paper-style training-time learning curves into a "
                    "training_plots/ directory."
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument("--out-dir", default="figures/training_plots", metavar="DIR")
    parser.add_argument("--honeypot-universe", type=int,
                        default=DEFAULT_HONEYPOT_UNIVERSE)
    parser.add_argument(
        "--ci", default="true", type=lambda s: s.strip().lower(),
        choices=["true", "false"], metavar="true|false",
        help="Render confidence-interval overlays (default: true).",
    )
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out_dir = Path(args.out_dir)
    show_ci = args.ci == "true"

    for fn, fname, desc in _JOBS:
        kwargs = {}
        if fn in (plot_selfplay_arms_race,):
            kwargs = {"honeypot_universe": args.honeypot_universe, "show_ci": show_ci}
        elif fn in (plot_red_honeypot_discovery, plot_selfplay_dominance):
            kwargs = {"honeypot_universe": args.honeypot_universe}
        elif fn in (plot_blue_outcome_rates,):
            kwargs = {"show_ci": show_ci}
        try:
            path = fn(results, out_dir / fname, **kwargs)
            print(f"[{desc[:90]}...]\n  -> {path}")
        except Exception as e:  # pragma: no cover
            print(f"[plot_training_curves] {fname} skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
