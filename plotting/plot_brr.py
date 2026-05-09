"""
BRR (Benign Refusal Rate) vs. self-play iteration.

Fraction of legitimate user queries incorrectly refused by the blue team.
Lower BRR = better utility preservation.

Sources (each produces its own PNG):
  train_rollouts  — BRR from training-time benign turns in
                    iter_*/blueteam/**/debug_logs/reward_debug.jsonl.
                    Same corpus and denial predicate as plot_per_style_refusal.
                    Headline figure; ~13–15% for the canonical run.
  benign_eval     — BRR re-aggregated from benign_eval/benign_only/blue_*/
                    reward_debug.jsonl using the same denial predicate.
                    Near-floor (~1–3%) because that corpus is plain-heavy.
  human_eval      — BRR directly from iter_N/summary.json (0–1 scale, %).

CLI:
    python plotting/plot_brr.py --results results-<ID>[:Label] \\
        [--sources train_rollouts,benign_eval] [--out-dir figures/]
Or imported:
    from plotting.plot_brr import plot_brr, DESCRIPTION
    paths = plot_brr(results, out_dir="figures/")  # dict[source, Path]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style,
        denial_rate_with_ci,
        load_benign_eval_per_turn,
        load_human_eval_summaries,
        load_train_rollout_benign_turns,
        parse_results_arg,
        wilson_ci_pct,
        RUN_COLORS, FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        denial_rate_with_ci,
        load_benign_eval_per_turn,
        load_human_eval_summaries,
        load_train_rollout_benign_turns,
        parse_results_arg,
        wilson_ci_pct,
        RUN_COLORS, FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = (
    "BRR (Benign Refusal Rate) vs. self-play iteration. "
    "train_rollouts source: population-weighted denial rate across plain / "
    "multi_turn / adversarial-framed benign turns seen during training — "
    "matches plot_per_style_refusal population (~13–15% for the canonical run). "
    "benign_eval source: same denial predicate on the held-out plain-heavy eval "
    "corpus — near-floor (~1–3%) because adversarial-framed turns are under-represented. "
    "human_eval source: from iter_N/summary.json."
)

_SOURCE_TITLE = {
    "train_rollouts": "BRR — training-time rollouts (all benign styles)",
    "benign_eval":    "BRR — held-out benign eval corpus (plain-heavy)",
    "human_eval":     "BRR — human evaluation",
}
_SOURCE_YLABEL = "BRR (%)"


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------

def plot_brr(
    results: list[tuple[str, str]],
    out_dir: str | Path = "figures/",
    filename_prefix: str = "brr",
    sources: tuple[str, ...] = ("train_rollouts",),
    human_eval_parent: str | None = None,
    cross_eval_subdir: str = "cross_eval",
) -> dict[str, Path]:
    """
    Plot BRR (%) vs. self-play iteration; one PNG per source.

    Args:
        results:           [(label, selfplay_dir), ...]
        out_dir:           Directory for output PNGs.
        filename_prefix:   PNG files are named <prefix>_<source>.png.
        sources:           Which sources to plot. Subset of
                           ("train_rollouts", "benign_eval", "human_eval").
                           "cross_eval" is accepted as a deprecated alias for
                           "benign_eval" (emits a warning).
        human_eval_parent: Parent dir with iter_N/summary.json (human_eval only).
        cross_eval_subdir: Unused; kept for call-site compatibility.

    Returns dict mapping source name → Path of written PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize deprecated alias
    normalized: list[str] = []
    for src in sources:
        if src == "cross_eval":
            print(
                "  [plot_brr] WARNING: source 'cross_eval' is deprecated — "
                "using 'benign_eval' instead (re-aggregated from per-turn JSONL "
                "with consistent denial definition).",
                file=sys.stderr,
            )
            normalized.append("benign_eval")
        else:
            normalized.append(src)
    sources = tuple(dict.fromkeys(normalized))  # dedup, preserve order

    colors = RUN_COLORS * (len(results) // len(RUN_COLORS) + 1)
    produced: dict[str, Path] = {}

    for src in sources:
        out_path = out_dir / f"{filename_prefix}_{src}.png"
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

        for run_idx, (label, selfplay_dir) in enumerate(results):
            color = colors[run_idx]

            if src == "train_rollouts":
                rows_by_iter = load_train_rollout_benign_turns(selfplay_dir)
                if not rows_by_iter:
                    print(
                        f"  [plot_brr] No training-rollout benign turns found for "
                        f"{selfplay_dir}",
                        file=sys.stderr,
                    )
                    continue
                iters = sorted(rows_by_iter)
                vals, lo_errs, hi_errs = [], [], []
                for i in iters:
                    rate, lo, hi, _, _ = denial_rate_with_ci(rows_by_iter[i])
                    vals.append(rate)
                    lo_errs.append(max(0.0, rate - lo) if not np.isnan(rate) else 0.0)
                    hi_errs.append(max(0.0, hi - rate) if not np.isnan(rate) else 0.0)
                yerr = [lo_errs, hi_errs] if any(v > 0 for v in lo_errs + hi_errs) else None
                ax.errorbar(
                    iters, vals, yerr=yerr,
                    label=label, color=color,
                    marker="o", linewidth=2, markersize=7, capsize=4,
                )

            elif src == "benign_eval":
                rows_by_iter = load_benign_eval_per_turn(selfplay_dir)
                if not rows_by_iter:
                    print(
                        f"  [plot_brr] No benign_eval per-turn data found for "
                        f"{selfplay_dir}",
                        file=sys.stderr,
                    )
                    continue
                iters = sorted(rows_by_iter)
                vals, lo_errs, hi_errs = [], [], []
                for i in iters:
                    rate, lo, hi, _, _ = denial_rate_with_ci(rows_by_iter[i])
                    vals.append(rate)
                    lo_errs.append(max(0.0, rate - lo) if not np.isnan(rate) else 0.0)
                    hi_errs.append(max(0.0, hi - rate) if not np.isnan(rate) else 0.0)
                yerr = [lo_errs, hi_errs] if any(v > 0 for v in lo_errs + hi_errs) else None
                ax.errorbar(
                    iters, vals, yerr=yerr,
                    label=label, color=color,
                    marker="s", linewidth=2, markersize=7, capsize=4,
                )

            elif src == "human_eval":
                if not human_eval_parent:
                    print(
                        "  [plot_brr] --human-eval-parent required for human_eval source",
                        file=sys.stderr,
                    )
                    break
                summaries = load_human_eval_summaries(human_eval_parent)
                if not summaries:
                    print(
                        f"  [plot_brr] No human-eval summaries in {human_eval_parent}",
                        file=sys.stderr,
                    )
                    continue
                iters = sorted(summaries)
                vals = [summaries[i].get("BRR", float("nan")) * 100 for i in iters]
                ci_raw = [summaries[i].get("BRR_ci") for i in iters]
                ci_pct = [[c * 100 for c in ci] if ci else None for ci in ci_raw]
                lo_errs, hi_errs = [], []
                for val, ci in zip(vals, ci_pct):
                    if ci is None or np.isnan(val):
                        lo_errs.append(0.0)
                        hi_errs.append(0.0)
                    else:
                        lo_errs.append(max(0.0, val - ci[0]))
                        hi_errs.append(max(0.0, ci[1] - val))
                yerr = [lo_errs, hi_errs] if any(v > 0 for v in lo_errs + hi_errs) else None
                ax.errorbar(
                    iters, vals, yerr=yerr,
                    label=label, color=color,
                    marker="*", linewidth=2, markersize=9, capsize=4,
                )

        ax.set_xlabel("Self-play iteration")
        ax.set_ylabel(_SOURCE_YLABEL)
        ax.set_title(_SOURCE_TITLE.get(src, f"BRR — {src}"))
        ax.set_ylim(0, 105)
        _integer_xticks(ax)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True)
        ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout(pad=0.4)
        fig.savefig(out_path)
        plt.close(fig)
        produced[src] = out_path

    return produced


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _integer_xticks(ax: plt.Axes) -> None:
    xmin, xmax = ax.get_xlim()
    ticks = list(range(max(0, int(np.floor(xmin))), int(np.ceil(xmax)) + 1))
    if ticks:
        ax.set_xticks(ticks)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BRR (Benign Refusal Rate) vs. self-play iteration."
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
        help="One or more selfplay result dirs, optionally with :Label suffix.",
    )
    parser.add_argument(
        "--out-dir", default="figures/", metavar="DIR",
        help="Output directory for PNG files (default: figures/).",
    )
    parser.add_argument(
        "--filename-prefix", default="brr", metavar="PREFIX",
        help="PNG files are named <PREFIX>_<source>.png (default: brr).",
    )
    parser.add_argument(
        "--sources", default="train_rollouts",
        help=(
            "Comma-separated data sources: train_rollouts, benign_eval, human_eval "
            "(default: train_rollouts). 'cross_eval' is a deprecated alias for 'benign_eval'."
        ),
    )
    parser.add_argument("--human-eval-parent", default=None, metavar="DIR")
    parser.add_argument("--cross-eval-subdir", default="cross_eval",
                        help="Unused; kept for call-site compatibility.")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    paths = plot_brr(
        results,
        out_dir=args.out_dir,
        filename_prefix=args.filename_prefix,
        sources=sources,
        human_eval_parent=args.human_eval_parent,
        cross_eval_subdir=args.cross_eval_subdir,
    )
    for src, path in paths.items():
        print(f"[{src}] {path}")


if __name__ == "__main__":
    main()
