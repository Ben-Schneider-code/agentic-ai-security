"""
Adversarial training dynamics: reward-vs-compute learning curves per self-play iteration.

Left — Attacker (red): average step reward rises during honeypot discovery then
    declines to negative; the no_new_honeypot_for_1000_steps exit fires at varying
    EIS across iterations, producing different curve lengths.

Right — Defender (blue): all curves run to the 8,000-EIS budget ceiling. The
    decisive-win threshold (0.75) is never crossed, demonstrating that adaptive
    defense cannot converge within the allotted training budget.

Color gradient: light = iteration 1, dark = final iteration.
Data source: logs/summary.json (average_step_rewards) in each team's run directory,
    falling back to all_episodic_returns from training_state.json if absent.

Can be run standalone:
    python plotting/plot_training_dynamics.py --results results-<ID>
Or imported:
    from plotting.plot_training_dynamics import plot_training_dynamics, DESCRIPTION
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style, discover_iterations, find_run_dir, load_training_state,
        parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_1x2, FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, discover_iterations, find_run_dir, load_training_state,
        parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_1x2, FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = (
    "Per-iteration RL training curves for attacker (red) and defender (blue). "
    "NOTE: Y-axis shows the RL training reward used as a shaping signal during training "
    "— NOT the paper metrics PVR_conv or BRR. Red reward reflects refusal/honeypot "
    "dynamics; blue reward reflects classification accuracy against the current opponent. "
    "These are incommensurable quantities on different scales. "
    "For compute vs PVR_conv/BRR, see plot_compute_efficiency.py instead. "
    "X-axis: EIS within each iteration. Left — Attacker: curves end early via "
    "convergence signals (variable lengths). Right — Defender: all curves reach "
    "the 8,000-EIS budget ceiling without crossing decisive-win threshold (0.75). "
    "Color: light = early iteration, dark = late iteration."
)

_BLUE_DECISIVE_WIN = 0.75  # blueteam_decisive_win_threshold from reward config (same scale as avg step reward)
_BLUE_BUDGET_EIS = 8000    # blueteam_max_training_steps from reward config
_RED_WARMUP_EIS = 1600     # 20 warmup_episodes × 10 episode_length × 8 n_rollout_threads

RED_TERMINATION_DESCRIPTION = (
    "Per-iteration red-team EIS at exit and stopping reason. Red exits via "
    "no_new_honeypot_for_1000_steps, the novelty-exhaustion signal: training halts "
    "when 1000 env steps pass without discovering a new honeypot. Cumulative budget "
    "is bounded by |Q^P_deny| (the finite denied-query universe), not by compute. "
    "Blue's 8,000-EIS ceiling is overlaid as a compute-asymmetry reference."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _summary_key(data: dict) -> str | None:
    """Return the key ending in 'average_step_rewards/average_step_rewards'."""
    suffix = "average_step_rewards/average_step_rewards"
    for k in data:
        if k.endswith(suffix):
            return k
    return None


def load_training_curve(run_dir: Path) -> list[tuple[int, float]]:
    """
    Load (step, avg_reward) pairs from logs/summary.json.

    Falls back to all_episodic_returns × 80-step intervals from training_state.json
    when summary.json is absent or malformed.
    """
    summary = run_dir / "logs" / "summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            key = _summary_key(data)
            if key and data[key]:
                return [(int(e[1]), float(e[2])) for e in data[key]]
        except (json.JSONDecodeError, KeyError, ValueError, IndexError):
            pass

    # Fallback: reconstruct at ~80-step resolution from episodic returns
    try:
        state = load_training_state(run_dir)
    except FileNotFoundError:
        return []
    returns = state.get("all_episodic_returns", [])
    return [(80 * (i + 1), float(r)) for i, r in enumerate(returns)]


def _smooth(values: list[float], window: int) -> np.ndarray:
    """Causal rolling mean (no look-ahead)."""
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = arr[lo : i + 1].mean()
    return out


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------

def plot_training_dynamics(
    results: list[tuple[str, str]],
    out_path: str | Path,
    smooth_window: int = 7,
) -> Path:
    """
    Plot per-iteration training reward curves (red and blue) vs EIS.

    Args:
        results:       [(label, selfplay_dir), ...]  — only the first entry is used.
        out_path:      Destination PNG path.
        smooth_window: Causal rolling-mean window applied to avg step reward.

    Returns the resolved Path that was written.
    """
    out_path = Path(out_path)

    if len(results) > 1:
        print("[plot_training_dynamics] Multiple runs given; using first run only.",
              file=sys.stderr)
    _label, selfplay_dir = results[0]

    iters_data = discover_iterations(selfplay_dir)
    if not iters_data:
        print(f"[plot_training_dynamics] No iterations in {selfplay_dir}.", file=sys.stderr)
        return out_path

    n_iters = len(iters_data)

    try:
        cmap_r = matplotlib.colormaps["Reds"]
        cmap_b = matplotlib.colormaps["Blues"]
    except AttributeError:
        cmap_r = plt.cm.get_cmap("Reds")
        cmap_b = plt.cm.get_cmap("Blues")

    fig, (ax_r, ax_b) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    for idx, entry in enumerate(iters_data):
        shade = 0.35 + 0.55 * idx / max(n_iters - 1, 1)
        iter_lbl = f"Iter {entry['iter']}"

        # ── Red team ──────────────────────────────────────────────────────────
        red_dir = entry.get("red_dir")
        if red_dir:
            run_dir = find_run_dir(red_dir)
            if run_dir:
                curve = load_training_curve(run_dir)
                if curve:
                    steps, rewards = zip(*curve)
                    sm = _smooth(list(rewards), smooth_window)
                    col = cmap_r(shade)
                    ax_r.plot(steps, sm, color=col, linewidth=1.5,
                              label=iter_lbl, zorder=3)
                    # Exit marker: circle at end of each curve
                    ax_r.plot(steps[-1], sm[-1], "o", color=col,
                              markersize=5, zorder=4)

        # ── Blue team ─────────────────────────────────────────────────────────
        blue_dir = entry.get("blue_dir")
        if blue_dir:
            run_dir = find_run_dir(blue_dir)
            if run_dir:
                curve = load_training_curve(run_dir)
                if curve:
                    steps, rewards = zip(*curve)
                    sm = _smooth(list(rewards), smooth_window)
                    col = cmap_b(shade)
                    ax_b.plot(steps, sm, color=col, linewidth=1.5,
                              label=iter_lbl, zorder=3)

    # ── Red panel ─────────────────────────────────────────────────────────────
    ax_r.axvspan(0, _RED_WARMUP_EIS, color="#f0f0f0", zorder=0,
                 label=f"Warmup ({_RED_WARMUP_EIS:,} EIS)")
    ax_r.axhline(0, color="#bbbbbb", linewidth=0.9, linestyle="-", zorder=1)
    ax_r.set_title("Attacker (Red) Training Curves")
    ax_r.set_xlabel("Environment Interaction Steps (EIS)")
    ax_r.set_ylabel("Avg Step Reward (smoothed)")
    ax_r.legend(fontsize=9, frameon=True, ncol=2)
    ax_r.grid(True, axis="y", alpha=0.4)
    ax_r.set_xlim(left=0)

    # ── Blue panel ────────────────────────────────────────────────────────────
    ax_b.axvline(_BLUE_BUDGET_EIS, color="#888888", linestyle=":", linewidth=1.3,
                 label=f"Budget ceiling ({_BLUE_BUDGET_EIS:,} EIS)", zorder=1)
    ax_b.axhline(_BLUE_DECISIVE_WIN, color=BLUE_COL, linestyle="--", linewidth=1.5,
                 alpha=0.75,
                 label=f"Decisive-win threshold ({_BLUE_DECISIVE_WIN})", zorder=2)
    ax_b.set_title("Defender (Blue) Training Curves")
    ax_b.set_xlabel("Environment Interaction Steps (EIS)")
    ax_b.set_ylabel("Avg Step Reward (smoothed)")
    ax_b.legend(fontsize=9, frameon=True, ncol=2)
    ax_b.grid(True, axis="y", alpha=0.4)
    ax_b.set_xlim(left=0, right=_BLUE_BUDGET_EIS * 1.06)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Red-team early-termination timeline
# ---------------------------------------------------------------------------

def _load_red_exit(run_dir: Path) -> tuple[int | None, int | None, str | None]:
    """Return (total_num_steps, n_unique_honeypots, exit_reason) or (None, None, None)."""
    state_path = run_dir / "training_state.json"
    reason_path = run_dir / "exit_reason.txt"

    steps: int | None = None
    hp: int | None = None
    reason: str | None = None

    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            steps = int(data["total_num_steps"])
            accessed = data.get("accessed_honeypots") or []
            hp = len(accessed) if accessed is not None else None
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    if reason_path.exists():
        try:
            reason = reason_path.read_text().strip() or None
        except OSError:
            reason = None

    return steps, hp, reason


def plot_red_termination_timeline(
    results: list[tuple[str, str]],
    out_path: str | Path,
    honeypot_universe: int = 22,
) -> Path:
    """
    Bar chart of red EIS-at-exit per self-play iteration, with stopping reason
    and per-iter unique-honeypot count labeled above each bar. Reference lines
    show blue's 8000-EIS budget ceiling (compute asymmetry) and the 1000-step
    novelty window that triggers red's exit.

    Args:
        results:           [(label, selfplay_dir), ...] — first entry used.
        out_path:          PNG destination.
        honeypot_universe: Denominator for "N/M hp" labels above bars.
    """
    out_path = Path(out_path)

    if len(results) > 1:
        print("[plot_red_termination_timeline] Multiple runs given; using first only.",
              file=sys.stderr)
    _label, selfplay_dir = results[0]

    iters_data = discover_iterations(selfplay_dir)
    if not iters_data:
        print(f"[plot_red_termination_timeline] No iterations in {selfplay_dir}.",
              file=sys.stderr)
        return out_path

    rows: list[tuple[int, int, int | None, str]] = []
    for entry in iters_data:
        red_dir = entry.get("red_dir")
        if not red_dir:
            continue
        run_dir = find_run_dir(red_dir)
        if not run_dir:
            continue
        steps, hp, reason = _load_red_exit(run_dir)
        if steps is None:
            continue
        rows.append((int(entry["iter"]), steps, hp, reason or "unknown"))

    if not rows:
        print("[plot_red_termination_timeline] No red training_state.json found; "
              "skipping.", file=sys.stderr)
        return out_path

    iters_n = [r[0] for r in rows]
    steps_n = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    bars = ax.bar(iters_n, steps_n, color=RED_COL, width=0.7, zorder=3,
                  edgecolor="#5e0a0a", linewidth=0.6)

    ax.axhline(_BLUE_BUDGET_EIS, color=BLUE_COL, linestyle="--", linewidth=1.3,
               alpha=0.8, zorder=2,
               label=f"Blue budget ceiling ({_BLUE_BUDGET_EIS:,} EIS)")
    ax.axhline(1000, color="#888888", linestyle=":", linewidth=1.1, alpha=0.8,
               zorder=2, label="Red novelty window (1k EIS)")

    for bar, (_, steps, hp, reason) in zip(bars, rows):
        hp_label = f"{hp}/{honeypot_universe} hp" if hp is not None else "?/? hp"
        ax.text(bar.get_x() + bar.get_width() / 2,
                steps + max(steps_n) * 0.02,
                f"{steps:,}\n{hp_label}",
                ha="center", va="bottom", fontsize=9, zorder=4)

    unique_reasons = {r[3] for r in rows}
    if len(unique_reasons) == 1:
        ax.set_title(
            f"Red-team exit: novelty-exhaustion across all iterations\n"
            f"(all {len(rows)} iters → '{next(iter(unique_reasons))}')",
            fontsize=12,
        )
    else:
        ax.set_title("Red-team EIS at exit — mixed stopping reasons", fontsize=12)

    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Red EIS at exit")
    ax.set_xticks(iters_n)
    ax.set_ylim(0, max(steps_n + [_BLUE_BUDGET_EIS]) * 1.20)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-iteration training dynamics (reward vs EIS)."
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument("--out", default="figures/training_dynamics.png")
    parser.add_argument(
        "--smooth-window", type=int, default=7,
        help="Causal rolling-mean window for reward smoothing (default: 7).",
    )
    parser.add_argument(
        "--red-termination-out", default="figures/red_termination.png",
        help="Destination PNG for the red EIS-at-exit timeline "
             "(Pillar 5, compute-asymmetry evidence).",
    )
    parser.add_argument(
        "--skip-red-termination", action="store_true",
        help="Skip the red-termination timeline panel (main figure only).",
    )
    parser.add_argument(
        "--honeypot-universe", type=int, default=22,
        help="Denominator for honeypot-coverage labels (default: 22).",
    )
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out = plot_training_dynamics(results, args.out, smooth_window=args.smooth_window)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")

    if not args.skip_red_termination:
        term_out = plot_red_termination_timeline(
            results, args.red_termination_out,
            honeypot_universe=args.honeypot_universe,
        )
        print(f"[{RED_TERMINATION_DESCRIPTION[:100]}...]\n  → {term_out}")


if __name__ == "__main__":
    main()
