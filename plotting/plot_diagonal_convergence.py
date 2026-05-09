"""
Diagonal convergence — PVR_conv and PVR_turn along the training trajectory.

Plots each adjacent checkpoint pairing on the x-axis in training order:
  red_0 vs blue_0  →  red_1 vs blue_0  →  red_1 vs blue_1  →  red_2 vs blue_1  →  …

Points are colored red when red was trained last (red_N vs blue_{N-1}) and blue
when blue was trained last (red_N vs blue_N).

Preferred source: diagonal_eval/ (400 ep/cell); falls back to cross_eval/.

Can be run standalone:
    python plotting/plot_diagonal_convergence.py --results results-<ID>
Or imported:
    from plotting.plot_diagonal_convergence import plot_diagonal_convergence, DESCRIPTION
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style,
        load_pairing_metrics_with_decomposed,
        parse_results_arg,
        write_sidecar,
        RED_COL,
        BLUE_COL,
        GRAY_COL,
        FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_pairing_metrics_with_decomposed,
        parse_results_arg,
        write_sidecar,
        RED_COL,
        BLUE_COL,
        GRAY_COL,
        FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = (
    "Diagonal convergence: PVR_conv and PVR_turn for each adjacent checkpoint "
    "pairing along the self-play training trajectory "
    "(red_0 vs blue_0 → red_1 vs blue_0 → red_1 vs blue_1 → …). "
    "Red points = red trained last (red_N vs blue_{N-1}); "
    "blue points = blue trained last (red_N vs blue_N). "
    "Preferred source: diagonal_eval/; fallback: cross_eval/."
)

_SUBDIR_PRIORITY = ("diagonal_eval", "cross_eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adjacent_sequence(pairings: dict) -> list[tuple[int, int]]:
    """
    Return adjacent (red_iter, blue_iter) tuples in trajectory order.

    Pattern: (0,0), (1,0), (1,1), (2,1), (2,2), …
    Only includes entries present in the pairing data.
    """
    available = {(v["red_iter"], v["blue_iter"]) for v in pairings.values()}
    if not available:
        return []
    max_r = max(r for r, _ in available)
    seq: list[tuple[int, int]] = []
    if (0, 0) in available:
        seq.append((0, 0))
    for n in range(1, max_r + 1):
        if (n, n - 1) in available:
            seq.append((n, n - 1))
        if (n, n) in available:
            seq.append((n, n))
    return seq


def _point_color(red_iter: int, blue_iter: int) -> str:
    return RED_COL if red_iter > blue_iter else BLUE_COL


def _extract_series(
    seq: list[tuple[int, int]],
    pairing_lookup: dict[tuple[int, int], dict],
) -> tuple[
    list[float], list[float], list[float],
    list[float], list[float], list[float],
]:
    """
    Extract (val, ci_lo, ci_hi) arrays for PVR_conv and PVR_turn.
    All values are on the 0–100 scale; NaN where data is absent.
    """
    pvr_conv, pvr_conv_lo, pvr_conv_hi = [], [], []
    pvr_turn, pvr_turn_lo, pvr_turn_hi = [], [], []

    for r_iter, b_iter in seq:
        p  = pairing_lookup.get((r_iter, b_iter), {})
        m  = p.get("metrics", {})
        ci = p.get("confidence_intervals", {})

        # PVR_conv (asr field, 0–100)
        pvr_conv.append(m.get("asr", float("nan")))
        asr_ci = ci.get("asr")
        pvr_conv_lo.append(asr_ci[0] if asr_ci else float("nan"))
        pvr_conv_hi.append(asr_ci[1] if asr_ci else float("nan"))

        # PVR_turn (0–100)
        pvr_turn.append(m.get("pvr_turn", float("nan")))
        pt_ci = ci.get("pvr_turn")
        pvr_turn_lo.append(pt_ci[0] if pt_ci else float("nan"))
        pvr_turn_hi.append(pt_ci[1] if pt_ci else float("nan"))

    return (
        pvr_conv, pvr_conv_lo, pvr_conv_hi,
        pvr_turn, pvr_turn_lo, pvr_turn_hi,
    )


def _render_panel(
    ax: plt.Axes,
    x: np.ndarray,
    x_labels: list[str],
    vals: list[float],
    lo: list[float],
    hi: list[float],
    colors: list[str],
    ylabel: str,
    title: str,
) -> None:
    # Thin connecting line behind the points
    valid_pairs = [(xi, vi) for xi, vi in zip(x, vals) if not np.isnan(vi)]
    if len(valid_pairs) >= 2:
        xs, ys = zip(*valid_pairs)
        ax.plot(xs, ys, color=GRAY_COL, linewidth=1.0, zorder=0, alpha=0.5)

    for xi, vi, li, hi_i, col in zip(x, vals, lo, hi, colors):
        if np.isnan(vi):
            continue
        yerr_lo = max(0.0, vi - li) if not np.isnan(li) else 0.0
        yerr_hi = max(0.0, hi_i - vi) if not np.isnan(hi_i) else 0.0
        ax.errorbar(
            xi, vi,
            yerr=[[yerr_lo], [yerr_hi]],
            fmt="o",
            color=col,
            ecolor=col,
            markersize=8,
            capsize=4,
            linewidth=1.5,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.4)

    red_patch  = mpatches.Patch(color=RED_COL,  label="Red trained last")
    blue_patch = mpatches.Patch(color=BLUE_COL, label="Blue trained last")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="upper right")


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------


def plot_diagonal_convergence(
    results: list[tuple[str, str]],
    out_path: str | Path,
    *,
    eval_subdir: str | None = None,
) -> tuple[Path, dict]:
    """
    Plot PVR_conv and PVR_turn for adjacent pairings along the training diagonal.

    Args:
        results:     [(label, selfplay_dir), …] — only first entry is used.
        out_path:    Destination PNG path.
        eval_subdir: Subdir to load pairing data from. Auto-selects
                     diagonal_eval then cross_eval when None.

    Returns (path, metrics_dict). metrics_dict carries per-cell PVR_conv /
    PVR_turn / 99% Wilson CIs along the trajectory order.
    """
    out_path = Path(out_path)
    if len(results) > 1:
        print(
            "[plot_diagonal_convergence] Multiple runs given; using first run only.",
            file=sys.stderr,
        )

    label, selfplay_dir = results[0]

    # Load pairing data
    cross_eval = None
    chosen_subdir = eval_subdir
    if chosen_subdir is not None:
        cross_eval = load_pairing_metrics_with_decomposed(selfplay_dir, subdir=chosen_subdir)
    else:
        for sd in _SUBDIR_PRIORITY:
            cross_eval = load_pairing_metrics_with_decomposed(selfplay_dir, subdir=sd)
            if cross_eval is not None:
                chosen_subdir = sd
                break

    def _empty_figure(msg: str) -> tuple[Path, dict]:
        fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)
        for ax in axes:
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                    color=GRAY_COL)
        fig.tight_layout(pad=0.4)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path, {}

    if cross_eval is None:
        print(
            f"[plot_diagonal_convergence] No cross-eval data found in {selfplay_dir}.",
            file=sys.stderr,
        )
        return _empty_figure("No data")

    pairings = cross_eval.get("pairings", {})
    seq = _adjacent_sequence(pairings)

    if not seq:
        print(
            f"[plot_diagonal_convergence] No adjacent pairings in "
            f"{selfplay_dir}/{chosen_subdir}.",
            file=sys.stderr,
        )
        return _empty_figure("No adjacent pairings")

    pairing_lookup: dict[tuple[int, int], dict] = {
        (v["red_iter"], v["blue_iter"]): v
        for v in pairings.values()
        if "red_iter" in v and "blue_iter" in v
    }

    (
        pvr_conv, pvr_conv_lo, pvr_conv_hi,
        pvr_turn, pvr_turn_lo, pvr_turn_hi,
    ) = _extract_series(seq, pairing_lookup)

    x        = np.arange(len(seq))
    x_labels = [f"R{r}·B{b}" for r, b in seq]
    colors   = [_point_color(r, b) for r, b in seq]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    _render_panel(
        axes[0], x, x_labels,
        pvr_conv, pvr_conv_lo, pvr_conv_hi, colors,
        r"$\mathrm{PVR_{conv}}$ (%)",
        r"$\mathrm{PVR_{conv}}$ vs. Training Trajectory",
    )
    _render_panel(
        axes[1], x, x_labels,
        pvr_turn, pvr_turn_lo, pvr_turn_hi, colors,
        r"$\mathrm{PVR_{turn}}$ (%)",
        r"$\mathrm{PVR_{turn}}$ vs. Training Trajectory",
    )

    fig.suptitle(
        f"Convergence Along Training Trajectory — {label}  (source: {chosen_subdir})",
        fontsize=12,
    )
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    metrics = {
        "source_subdir": chosen_subdir,
        "label": label,
        "selfplay_dir": selfplay_dir,
        "trajectory_labels": x_labels,
        "trajectory_seq": [{"red_iter": r, "blue_iter": b} for r, b in seq],
        "pvr_conv_pct": [round(v, 2) if not np.isnan(v) else None for v in pvr_conv],
        "pvr_conv_ci_99": [
            [round(l, 2) if not np.isnan(l) else None,
             round(h, 2) if not np.isnan(h) else None]
            for l, h in zip(pvr_conv_lo, pvr_conv_hi)
        ],
        "pvr_turn_pct": [round(v, 2) if not np.isnan(v) else None for v in pvr_turn],
        "pvr_turn_ci_99": [
            [round(l, 2) if not np.isnan(l) else None,
             round(h, 2) if not np.isnan(h) else None]
            for l, h in zip(pvr_turn_lo, pvr_turn_hi)
        ],
    }
    return out_path, metrics


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PVR_conv, PVR_turn, and BRR for adjacent checkpoint pairings "
            "along the self-play training trajectory."
        )
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
        help="Selfplay result directory, optionally with :Label suffix.",
    )
    parser.add_argument("--out", default="figures/diagonal_convergence.png", metavar="PATH")
    parser.add_argument(
        "--eval-subdir", default=None, metavar="SUBDIR",
        help="Subdir for pairing data (default: tries diagonal_eval then cross_eval).",
    )
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out, metrics = plot_diagonal_convergence(
        results, args.out,
        eval_subdir=args.eval_subdir,
    )
    write_sidecar(out, DESCRIPTION, results, metrics)
    print(f"[{DESCRIPTION}]\n  → {out}")


if __name__ == "__main__":
    main()
