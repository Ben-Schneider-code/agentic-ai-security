"""
PVR (Policy Violation Rate) vs. self-play iteration — turn-level and conversation-level.

Shows how the red team's attack success rate evolves as both teams co-train.
Lower PVR = stronger defense.
  Cross-eval source: diagonal (red_i vs blue_i) from cross_eval_results.json.
  Human-eval source: evaluated against 32 human-crafted jailbreaks.

Can be run standalone:
    python plotting/plot_pvr.py --results results-<ID> [--sources cross_eval,human_eval]
Or imported:
    from plotting.plot_pvr import plot_pvr, plot_pvr_conv, DESCRIPTION, DESCRIPTION_CONV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Support both package import and direct execution
try:
    from ._data import (
        apply_paper_style, load_cross_eval_results, extract_diagonal_metrics,
        load_human_eval_summaries, parse_results_arg,
        RED_COL, BLUE_COL, GREEN_COL, GRAY_COL, HUMAN_COL,
        RUN_COLORS, FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, load_cross_eval_results, extract_diagonal_metrics,
        load_human_eval_summaries, parse_results_arg,
        RED_COL, BLUE_COL, GREEN_COL, GRAY_COL, HUMAN_COL,
        RUN_COLORS, FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = (
    "PVR_turn (turn-level Policy Violation Rate) vs. self-play iteration. "
    "Fraction of individual attack turns where the blue team incorrectly allowed a denied query. "
    "Cross-eval source: diagonal red_i vs blue_i pairings from cross_eval_results.json. "
    "Human-eval source: blue-team checkpoints evaluated against 32 human jailbreaks."
)

DESCRIPTION_WF = (
    "Work Factor (WF = 1/PVR_turn) vs. self-play iteration. "
    "Expected number of resource-accessing attacker turns required to achieve one policy violation. "
    "Higher WF = stronger defense. Plotted on a log scale. "
    "Points where PVR_turn = 0 (no violations observed) are omitted. "
    "Cross-eval source: diagonal red_i vs blue_i pairings from cross_eval_results.json."
)

DESCRIPTION_CONV = (
    "PVR_conv (conversation-level Policy Violation Rate) vs. self-play iteration. "
    "Fraction of attack conversations where the red team succeeded at least once. "
    "Cross-eval source: 'asr' field from diagonal red_i vs blue_i pairings in cross_eval_results.json. "
    "Human-eval source: blue-team checkpoints evaluated against 32 human jailbreaks."
)

_LINESTYLE = {"cross_eval": "-", "human_eval": "--"}
_MARKER    = {"cross_eval": "o", "human_eval": "s"}


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------

def _plot_pvr_impl(
    results: list[tuple[str, str]],
    out_path: Path,
    sources: tuple[str, ...],
    human_eval_parent: str | None,
    cross_eval_subdir: str,
    xeval_metric_key: str,
    xeval_ci_key: str,
    heval_metric_key: str,
    heval_ci_key: str,
    ylabel: str,
    title: str,
    log_tag: str,
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    colors = RUN_COLORS * (len(results) // len(RUN_COLORS) + 1)

    for run_idx, (label, selfplay_dir) in enumerate(results):
        color = colors[run_idx]

        if "cross_eval" in sources:
            xeval = load_cross_eval_results(selfplay_dir, cross_eval_subdir)
            if xeval is not None:
                diag = extract_diagonal_metrics(xeval)
                if diag:
                    iters = sorted(diag)
                    vals = [diag[i].get(xeval_metric_key, float("nan")) for i in iters]
                    yerr = _ci_to_yerr(vals, [diag[i].get(xeval_ci_key) for i in iters])
                    lbl = f"{label} (cross-eval)" if "human_eval" in sources else label
                    ax.errorbar(
                        iters, vals, yerr=yerr,
                        label=lbl, color=color,
                        linestyle=_LINESTYLE["cross_eval"],
                        marker=_MARKER["cross_eval"],
                        linewidth=2, markersize=7, capsize=4,
                    )
                else:
                    print(f"  [{log_tag}] No diagonal pairings found in {selfplay_dir}",
                          file=sys.stderr)
            else:
                print(f"  [{log_tag}] cross_eval_results.json not found in {selfplay_dir}",
                      file=sys.stderr)

        if "human_eval" in sources:
            if not human_eval_parent:
                print(f"  [{log_tag}] --human-eval-parent required for human_eval source",
                      file=sys.stderr)
            else:
                summaries = load_human_eval_summaries(human_eval_parent)
                if summaries:
                    iters = sorted(summaries)
                    vals = [summaries[i].get(heval_metric_key, float("nan")) * 100
                            for i in iters]
                    ci_raw = [summaries[i].get(heval_ci_key) for i in iters]
                    ci_pct = [[c * 100 for c in ci] if ci else None for ci in ci_raw]
                    yerr = _ci_to_yerr(vals, ci_pct)
                    lbl = f"{label} (human-eval)" if "cross_eval" in sources else label
                    ax.errorbar(
                        iters, vals, yerr=yerr,
                        label=lbl, color=color,
                        linestyle=_LINESTYLE["human_eval"],
                        marker=_MARKER["human_eval"],
                        linewidth=2, markersize=7, capsize=4,
                    )
                else:
                    print(f"  [{log_tag}] No human-eval summaries found in {human_eval_parent}",
                          file=sys.stderr)

    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 105)
    _integer_xticks(ax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public plotting functions (deterministic, idempotent)
# ---------------------------------------------------------------------------

def plot_pvr(
    results: list[tuple[str, str]],
    out_path: str | Path,
    sources: tuple[str, ...] = ("cross_eval",),
    human_eval_parent: str | None = None,
    cross_eval_subdir: str = "cross_eval",
) -> Path:
    """Plot PVR_turn (%) vs. self-play iteration. Returns the resolved Path written."""
    return _plot_pvr_impl(
        results, Path(out_path), sources, human_eval_parent, cross_eval_subdir,
        xeval_metric_key="pvr_turn",
        xeval_ci_key="pvr_turn_ci",
        heval_metric_key="PVR_turn",
        heval_ci_key="PVR_turn_ci",
        ylabel=r"$\mathrm{PVR}_{\mathrm{turn}}$ (%)",
        title="Turn-level Policy Violation Rate vs. Iteration",
        log_tag="plot_pvr",
    )


def plot_pvr_conv(
    results: list[tuple[str, str]],
    out_path: str | Path,
    sources: tuple[str, ...] = ("cross_eval",),
    human_eval_parent: str | None = None,
    cross_eval_subdir: str = "cross_eval",
) -> Path:
    """Plot PVR_conv (%) vs. self-play iteration. Returns the resolved Path written."""
    return _plot_pvr_impl(
        results, Path(out_path), sources, human_eval_parent, cross_eval_subdir,
        xeval_metric_key="asr",
        xeval_ci_key="asr_ci",
        heval_metric_key="PVR_conv",
        heval_ci_key="PVR_conv_ci",
        ylabel=r"$\mathrm{PVR}_{\mathrm{conv}}$ (%)",
        title="Conversation-level Policy Violation Rate vs. Iteration",
        log_tag="plot_pvr_conv",
    )


def plot_work_factor(
    results: list[tuple[str, str]],
    out_path: str | Path,
    sources: tuple[str, ...] = ("cross_eval",),
    human_eval_parent: str | None = None,
    cross_eval_subdir: str = "cross_eval",
) -> Path:
    """Plot Work Factor (WF = 1/PVR_turn) vs. self-play iteration. Returns the resolved Path written."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    colors = RUN_COLORS * (len(results) // len(RUN_COLORS) + 1)

    for run_idx, (label, selfplay_dir) in enumerate(results):
        color = colors[run_idx]

        if "cross_eval" in sources:
            xeval = load_cross_eval_results(selfplay_dir, cross_eval_subdir)
            if xeval is not None:
                diag = extract_diagonal_metrics(xeval)
                if diag:
                    iters = sorted(diag)
                    pvr_vals = [diag[i].get("pvr_turn", float("nan")) for i in iters]
                    wf_vals  = [_pvr_pct_to_wf(p) for p in pvr_vals]
                    pvr_cis  = [diag[i].get("pvr_turn_ci") for i in iters]
                    wf_cis   = [_invert_pvr_pct_ci(ci) for ci in pvr_cis]
                    lbl = f"{label} (cross-eval)" if "human_eval" in sources else label
                    _errorbar_wf(ax, iters, wf_vals, wf_cis, color=color,
                                 linestyle=_LINESTYLE["cross_eval"],
                                 marker=_MARKER["cross_eval"], label=lbl)
                else:
                    print(f"  [plot_work_factor] No diagonal pairings found in {selfplay_dir}",
                          file=sys.stderr)
            else:
                print(f"  [plot_work_factor] cross_eval_results.json not found in {selfplay_dir}",
                      file=sys.stderr)

        if "human_eval" in sources:
            if not human_eval_parent:
                print("  [plot_work_factor] --human-eval-parent required for human_eval source",
                      file=sys.stderr)
            else:
                summaries = load_human_eval_summaries(human_eval_parent)
                if summaries:
                    iters = sorted(summaries)
                    # human-eval PVR_turn is on 0–1 scale → convert to 0–100 for consistency
                    pvr_vals = [summaries[i].get("PVR_turn", float("nan")) * 100 for i in iters]
                    wf_vals  = [_pvr_pct_to_wf(p) for p in pvr_vals]
                    pvr_cis_raw = [summaries[i].get("PVR_turn_ci") for i in iters]
                    pvr_cis  = [[c * 100 for c in ci] if ci else None for ci in pvr_cis_raw]
                    wf_cis   = [_invert_pvr_pct_ci(ci) for ci in pvr_cis]
                    lbl = f"{label} (human-eval)" if "cross_eval" in sources else label
                    _errorbar_wf(ax, iters, wf_vals, wf_cis, color=color,
                                 linestyle=_LINESTYLE["human_eval"],
                                 marker=_MARKER["human_eval"], label=lbl)
                else:
                    print(f"  [plot_work_factor] No human-eval summaries found in {human_eval_parent}",
                          file=sys.stderr)

    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel(r"$\mathrm{WF} = 1\,/\,\mathrm{PVR}_{\mathrm{turn}}$")
    ax.set_title("Work Factor vs. Iteration")
    ax.set_yscale("log")
    _integer_xticks(ax)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pvr_pct_to_wf(pvr_pct: float) -> float:
    """WF = 100 / pvr_pct (pvr on 0–100 scale). Returns NaN when pvr_pct <= 0."""
    if np.isnan(pvr_pct) or pvr_pct <= 0.0:
        return float("nan")
    return 100.0 / pvr_pct


def _invert_pvr_pct_ci(ci: list[float] | None) -> list[float] | None:
    """Convert pvr_turn CI [lo%, hi%] → WF CI [100/hi, 100/lo]. Returns None if bounds are zero."""
    if ci is None:
        return None
    lo, hi = ci
    if hi <= 0.0 or lo <= 0.0:
        return None
    return [100.0 / hi, 100.0 / lo]


def _errorbar_wf(
    ax: "plt.Axes",
    iters: list[int],
    wf_vals: list[float],
    wf_cis: list[list[float] | None],
    **kwargs: object,
) -> None:
    """Plot work-factor errorbar, filtering out NaN points (PVR_turn = 0)."""
    valid = [(i, w, ci) for i, w, ci in zip(iters, wf_vals, wf_cis) if not np.isnan(w)]
    if not valid:
        return
    vi, vw, vci = zip(*valid)
    yerr = _ci_to_yerr(list(vw), list(vci))
    ax.errorbar(vi, vw, yerr=yerr, linewidth=2, markersize=7, capsize=4, **kwargs)


def _ci_to_yerr(
    vals: list[float],
    cis: list[list[float] | None],
) -> list[list[float]] | None:
    """Convert [(lo, hi)] CI pairs → [[lo_err, ...], [hi_err, ...]] for errorbar."""
    if not any(c is not None for c in cis):
        return None
    lo_errs: list[float] = []
    hi_errs: list[float] = []
    for val, ci in zip(vals, cis):
        if ci is None or np.isnan(val):
            lo_errs.append(0.0)
            hi_errs.append(0.0)
        else:
            lo_errs.append(max(0.0, val - ci[0]))
            hi_errs.append(max(0.0, ci[1] - val))
    return [lo_errs, hi_errs]


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
        description="Plot PVR (Policy Violation Rate) vs. self-play iteration."
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
        help="One or more selfplay result dirs, optionally with :Label suffix.",
    )
    parser.add_argument("--out", default="figures/pvr.png", metavar="PATH")
    parser.add_argument(
        "--sources", default="cross_eval",
        help="Comma-separated data sources: cross_eval,human_eval (default: cross_eval).",
    )
    parser.add_argument("--human-eval-parent", default=None, metavar="DIR",
                        help="Parent dir with iter_N/summary.json for human-eval source.")
    parser.add_argument("--cross-eval-subdir", default="cross_eval")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    out = plot_pvr(
        results, args.out, sources,
        args.human_eval_parent, args.cross_eval_subdir,
    )
    print(f"[{DESCRIPTION}]\n  → {out}")


if __name__ == "__main__":
    main()
