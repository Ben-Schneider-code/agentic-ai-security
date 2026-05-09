"""
Cross-evaluation heatmaps — generic (subdir, metric) renderer.

Generates PVR_conv and PVR_turn heatmaps from cross_eval_results.json for
any of the three run subdirectories:

  cross_eval/        — full N×M pairings (100 ep/cell)
  cross_eval_quick/  — quick subset, diag_plus_base (40 ep/cell)
  diagonal_eval/     — diagonal + adjacent pairings (400 ep/cell)

The `training` mode reads per-iteration blueteam training logs instead of
cross_eval_results.json and produces a PVR_conv diagonal for diagnostics.

Reading the heatmap:
  - Going DOWN a column (more red training) → darkens (red improves).
  - Going RIGHT in a row (more blue training) → lightens (blue defends).
  - Diagonal cells (black border) = co-evolved pairings.

CLI usage:
    python plotting/plot_cross_eval_heatmap.py --results results-<ID>
    python plotting/plot_cross_eval_heatmap.py --results results-<ID> --mode cross_eval_pvr_turn
    python plotting/plot_cross_eval_heatmap.py --results results-<ID> --mode training
    python plotting/plot_cross_eval_heatmap.py --results results-<ID> --mode all
Or imported:
    from plotting.plot_cross_eval_heatmap import plot_heatmap, plot_training_diagonal, HEATMAP_JOBS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches
from matplotlib.transforms import blended_transform_factory

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style,
        load_cross_eval_results,
        load_pairing_metrics_with_decomposed,
        parse_results_arg,
        FIG_SIZE_SINGLE,
        build_diagonal_matrix,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_cross_eval_results,
        load_pairing_metrics_with_decomposed,
        parse_results_arg,
        FIG_SIZE_SINGLE,
        build_diagonal_matrix,
    )

apply_paper_style()

# ---------------------------------------------------------------------------
# Metric and subdir metadata
# ---------------------------------------------------------------------------

_METRIC_META: dict[str, dict] = {
    "asr": {
        "display": r"$\mathrm{PVR_{conv}}$",
        "label": r"$\mathrm{PVR_{conv}}$ (%)",
        "cmap": "YlOrRd",
        "note": "darker = higher attack success",
    },
    "pvr_turn": {
        "display": r"$\mathrm{PVR_{turn}}$",
        "label": r"$\mathrm{PVR_{turn}}$ (%)",
        "cmap": "YlOrRd",
        "note": "darker = higher attack success",
    },
    "pvr_sql_turn": {
        "display": r"$\mathrm{PVR_{sql\text{-}turn}}$",
        "label": r"$\mathrm{PVR_{sql\text{-}turn}}$ (%)",
        "cmap": "YlOrRd",
        "note": "PVR conditioned on SQL emission; darker = blue lets more SQL through",
    },
    "work_factor": {
        "display": r"Work factor",
        "label": r"SQL turns per violation",
        "cmap": "YlGnBu",
        "note": "attack SQL turns per successful violation; higher = blue wins",
        "unbounded": True,
    },
}

_SUBDIR_META: dict[str, str] = {
    "cross_eval": "full N×M pairings",
    "cross_eval_quick": "quick subset",
    "diagonal_eval": "diagonal + adjacent pairings",
}

# ---------------------------------------------------------------------------
# Canonical job list: (subdir, metric, output_filename)
# Consumed by --mode all and plot_paper_figures.py.
# ---------------------------------------------------------------------------

HEATMAP_JOBS: list[tuple[str, str, str]] = [
    ("cross_eval",       "asr",          "cross_eval_pvr_conv.png"),
    ("cross_eval",       "pvr_turn",     "cross_eval_pvr_turn.png"),
    ("cross_eval",       "pvr_sql_turn", "cross_eval_pvr_sql_turn.png"),
    ("cross_eval",       "work_factor",  "cross_eval_work_factor.png"),
    ("cross_eval_quick", "asr",          "quick_pvr_conv.png"),
    ("cross_eval_quick", "pvr_turn",     "quick_pvr_turn.png"),
    ("cross_eval_quick", "pvr_sql_turn", "quick_pvr_sql_turn.png"),
    ("cross_eval_quick", "work_factor",  "quick_work_factor.png"),
    ("diagonal_eval",    "asr",          "diagonal_eval_pvr_conv.png"),
    ("diagonal_eval",    "pvr_turn",     "diagonal_eval_pvr_turn.png"),
    ("diagonal_eval",    "pvr_sql_turn", "diagonal_eval_pvr_sql_turn.png"),
    ("diagonal_eval",    "work_factor",  "diagonal_eval_work_factor.png"),
]

# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------


def build_pvr_matrix(
    cross_eval: dict,
    metric: str = "asr",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Build (n_red × n_blue) matrices for the given metric and its Wilson CIs.

    metric: key in pairing["metrics"] and pairing["confidence_intervals"].
            Use "asr" for PVR_conv, "pvr_turn" for PVR_turn.
    Missing pairings are NaN.
    Returns (mat, ci_lo_mat, ci_hi_mat, sorted_red_iters, sorted_blue_iters).
    """
    pairings = cross_eval.get("pairings", {})
    red_iters = sorted({v["red_iter"] for v in pairings.values()})
    blue_iters = sorted({v["blue_iter"] for v in pairings.values()})

    ri_idx = {r: i for i, r in enumerate(red_iters)}
    bi_idx = {b: i for i, b in enumerate(blue_iters)}

    mat = np.full((len(red_iters), len(blue_iters)), np.nan)
    ci_lo_mat = np.full((len(red_iters), len(blue_iters)), np.nan)
    ci_hi_mat = np.full((len(red_iters), len(blue_iters)), np.nan)

    for pairing in pairings.values():
        ri = pairing.get("red_iter")
        bi = pairing.get("blue_iter")
        val = pairing.get("metrics", {}).get(metric)
        ci = pairing.get("confidence_intervals", {}).get(metric)
        if ri is not None and bi is not None and val is not None:
            mat[ri_idx[ri], bi_idx[bi]] = val
            if ci is not None and len(ci) == 2:
                ci_lo_mat[ri_idx[ri], bi_idx[bi]] = ci[0]
                ci_hi_mat[ri_idx[ri], bi_idx[bi]] = ci[1]

    return mat, ci_lo_mat, ci_hi_mat, red_iters, blue_iters


# ---------------------------------------------------------------------------
# Shared renderer
# ---------------------------------------------------------------------------


def _render_heatmap_axes(
    fig,
    ax,
    mat: np.ndarray,
    ci_lo_mat: np.ndarray,
    ci_hi_mat: np.ndarray,
    red_iters: list[int],
    blue_iters: list[int],
    *,
    title: str,
    metric_label: str = r"$\mathrm{PVR_{conv}}$ (%)",
    cmap: str = "YlOrRd",
    show_marginals: bool = True,
    unbounded: bool = False,
) -> None:
    """
    Render a metric heatmap onto an existing (fig, ax).

    Handles cell annotation (value + Wilson CI + asterisk), diagonal borders,
    optional row/col means, axis labels, title, and colorbar.
    All parameters after the matrices are keyword-only.
    """
    row_means = np.nanmean(mat, axis=1)
    col_means = np.nanmean(mat, axis=0)
    grand_mean = float(np.nanmean(mat)) if not np.all(np.isnan(mat)) else 0.0

    vmin = float(np.nanmin(mat)) if not np.all(np.isnan(mat)) else 0.0
    vmax = float(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 100.0
    if unbounded:
        im_vmin = max(0.0, vmin - (vmax - vmin) * 0.05)
        im_vmax = vmax + (vmax - vmin) * 0.05
    else:
        im_vmin = max(0.0, vmin - 2.0)
        im_vmax = min(100.0, vmax + 2.0)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=im_vmin,
        vmax=im_vmax,
        origin="upper",
    )

    mid_val = (vmin + vmax) / 2
    for i in range(len(red_iters)):
        for j in range(len(blue_iters)):
            val = mat[i, j]
            if np.isnan(val):
                ax.text(
                    j, i, "—", ha="center", va="center", fontsize=9, color="#aaaaaa"
                )
                continue
            text_col = "white" if val > mid_val + 5 else "black"
            ax.text(
                j,
                i - 0.18,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_col,
                fontweight="bold",
            )
            lo, hi = ci_lo_mat[i, j], ci_hi_mat[i, j]
            if not (np.isnan(lo) or np.isnan(hi)):
                ci_col = "#aaaaaa" if val > mid_val + 5 else "#777777"
                ax.text(
                    j,
                    i + 0.25,
                    f"[{lo:.0f}–{hi:.0f}]",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=ci_col,
                )
                if hi < grand_mean or lo > grand_mean:
                    ax.text(
                        j + 0.35,
                        i - 0.35,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#333333",
                    )

    # Black border on diagonal cells (co-evolved pairings)
    for k in range(min(len(red_iters), len(blue_iters))):
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (k - 0.5, k - 0.5),
                1,
                1,
                linewidth=2.0,
                edgecolor="#111111",
                fill=False,
                zorder=3,
            )
        )

    if show_marginals:
        trans_r = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            1.02,
            -0.55,
            "mean",
            transform=trans_r,
            ha="left",
            va="center",
            fontsize=7,
            color="#888888",
            style="italic",
            clip_on=False,
        )
        for i, rmean in enumerate(row_means):
            if not np.isnan(rmean):
                ax.text(
                    1.02,
                    i,
                    f"{rmean:.0f}%",
                    transform=trans_r,
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#444444",
                    clip_on=False,
                )

        trans_b = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(
            -0.55,
            -0.06,
            "mean",
            transform=trans_b,
            ha="center",
            va="top",
            fontsize=7,
            color="#888888",
            style="italic",
            clip_on=False,
        )
        for j, cmean in enumerate(col_means):
            if not np.isnan(cmean):
                ax.text(
                    j,
                    -0.06,
                    f"{cmean:.0f}%",
                    transform=trans_b,
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#444444",
                    clip_on=False,
                )

    # Hatch cells whose CI width exceeds the warning threshold.
    _CI_WIDE_PP = 20.0
    _wide_ci_seen = False
    for i in range(len(red_iters)):
        for j in range(len(blue_iters)):
            lo, hi = ci_lo_mat[i, j], ci_hi_mat[i, j]
            if np.isnan(lo) or np.isnan(hi):
                continue
            if (hi - lo) > _CI_WIDE_PP:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="///",
                        linewidth=0, edgecolor="#555555", alpha=0.55, zorder=4,
                    )
                )
                _wide_ci_seen = True

    ax.set_xticks(range(len(blue_iters)))
    ax.set_yticks(range(len(red_iters)))
    ax.set_xticklabels([f"Blue {b}" for b in blue_iters], fontsize=10)
    ax.set_yticklabels([f"Red {r}" for r in red_iters], fontsize=10)
    ax.set_xlabel("Blue team iteration  (→ increasing blue compute)")
    ax.set_ylabel("Red team iteration  (↓ increasing red compute)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(metric_label, fontsize=10)

    legend_lines = ["■ diagonal =\nco-evolved"]
    if _wide_ci_seen:
        legend_lines.append("/// CI\nwidth >20 pp")
    ax.text(
        1.18,
        0.5,
        "\n\n".join(legend_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="center",
        ha="center",
        color="#444444",
    )


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_heatmap(
    results: list[tuple[str, str]],
    out_path: str | Path,
    *,
    subdir: str = "cross_eval",
    metric: str = "asr",
) -> Path:
    """
    Generic cross-eval heatmap for any subdir × metric combination.

    subdir: one of "cross_eval", "cross_eval_quick", "diagonal_eval".
    metric: "asr" (PVR_conv) or "pvr_turn" (PVR_turn).
    Only the first result entry is used.
    """
    out_path = Path(out_path)
    if len(results) > 1:
        print(
            "[plot_heatmap] Multiple runs given; using first run only.",
            file=sys.stderr,
        )

    _label, selfplay_dir = results[0]
    cross_eval = load_pairing_metrics_with_decomposed(selfplay_dir, subdir=subdir)
    if cross_eval is None:
        print(
            f"[plot_heatmap] No data for subdir={subdir!r} in {selfplay_dir}.",
            file=sys.stderr,
        )
        return out_path

    mat, ci_lo_mat, ci_hi_mat, red_iters, blue_iters = build_pvr_matrix(
        cross_eval, metric=metric
    )
    mm = _METRIC_META[metric]
    sd_desc = _SUBDIR_META.get(subdir, subdir)

    # Derive actual ep/cell from data (median n_attack_episodes across pairings)
    pairings_data = cross_eval.get("pairings", {})
    ep_counts = [v.get("n_attack_episodes", 0) for v in pairings_data.values() if v.get("n_attack_episodes")]
    ep_note = f"{int(sorted(ep_counts)[len(ep_counts)//2])} ep/cell" if ep_counts else ""
    sd_full = f"{sd_desc} ({ep_note})" if ep_note else sd_desc

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    _render_heatmap_axes(
        fig,
        ax,
        mat,
        ci_lo_mat,
        ci_hi_mat,
        red_iters,
        blue_iters,
        title=f"Heatmap: {mm['display']} — {sd_full}\n({mm['note']})",
        metric_label=mm["label"],
        cmap=mm["cmap"],
        show_marginals=True,
        unbounded=mm.get("unbounded", False),
    )
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_training_diagonal(
    results: list[tuple[str, str]],
    out_path: str | Path,
) -> Path:
    """
    Diagonal-only PVR_conv heatmap from per-iteration blueteam training logs.

    Each cell (N, N) shows PVR_conv computed by compute_pairing_metrics on all
    non-eval training records (is_eval=False) — identical episode-level
    definition to the cross_eval heatmap. Uses 99% Wilson CIs.
    Only the first result entry is used.
    """
    out_path = Path(out_path)
    if len(results) > 1:
        print(
            "[plot_training_diagonal] Multiple runs given; using first run only.",
            file=sys.stderr,
        )

    _label, selfplay_dir = results[0]
    mat, ci_lo_mat, ci_hi_mat, iter_nums, metric_label = build_diagonal_matrix(
        selfplay_dir, source="training"
    )
    if not iter_nums:
        print(
            f"[plot_training_diagonal] No blueteam training data in {selfplay_dir}.",
            file=sys.stderr,
        )
        return out_path

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    _render_heatmap_axes(
        fig,
        ax,
        mat,
        ci_lo_mat,
        ci_hi_mat,
        iter_nums,
        iter_nums,
        title=(
            r"Training Diagonal: $\mathrm{PVR_{conv}}$ (%) — co-evolved pairings"
            "\n(all training episodes, is_eval=False; 99% CI; darker = higher attack success)"
        ),
        metric_label=f"{metric_label} (%)",
        cmap="YlOrRd",
        show_marginals=False,
    )
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# Descriptions for external consumers (e.g. plot_paper_figures.py).
def _heatmap_desc(subdir: str, metric: str) -> str:
    mm = _METRIC_META[metric]
    sd = _SUBDIR_META.get(subdir, subdir)
    unit = "" if mm.get("unbounded") else " (%)"
    return (
        f"Cross-eval heatmap: {mm['display']}{unit} from {subdir}/ ({sd}). "
        "Rows = red team iterations, columns = blue team iterations. "
        f"{mm['note'].capitalize()}. "
        "Diagonal cells (black border) = co-evolved pairings."
    )


DESCRIPTION_TRAINING = (
    "Diagonal-only heatmap: PVR_conv (%) from training-time attack episodes. "
    "Each diagonal cell (iter N, iter N) shows PVR_conv from compute_pairing_metrics "
    "on all non-eval records (is_eval=False) — identical episode-level definition "
    "to the cross-eval heatmap. 99% Wilson CIs; darker = higher attack success."
)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

# Mode key = filename stem (e.g. "cross_eval_pvr_conv", "diagonal_eval_pvr_turn").
_HEATMAP_MODE_KEYS: dict[str, tuple[str, str, str]] = {
    Path(f).stem: (s, m, f) for s, m, f in HEATMAP_JOBS
}

_TRAINING_MODE = ("training", "figures/diagonal_training.png")


def main() -> None:
    all_mode_keys = list(_HEATMAP_MODE_KEYS) + ["training"]
    parser = argparse.ArgumentParser(
        description="Plot cross-eval heatmaps for PVR_conv and PVR_turn across all run subdirs."
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument(
        "--mode",
        choices=[*all_mode_keys, "all"],
        default="all",
        help="Which heatmap(s) to generate (default: all).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path. Ignored when --mode all; uses per-mode defaults.",
    )
    parser.add_argument(
        "--override-subdir",
        default=None,
        metavar="NAME",
        help="Override the subdir component of the selected mode(s). Useful for "
             "pointing the cross_eval.* modes at e.g. cross_eval_long/. "
             "Ignored for mode=training.",
    )
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    modes = all_mode_keys if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "training":
            out = args.out if (args.out and args.mode != "all") else _TRAINING_MODE[1]
            saved = plot_training_diagonal(results, out)
            print(f"[{DESCRIPTION_TRAINING[:80]}...]\n  → {saved}")
        else:
            subdir, metric, fname = _HEATMAP_MODE_KEYS[mode]
            if args.override_subdir:
                subdir = args.override_subdir
            out = args.out if (args.out and args.mode != "all") else f"figures/{fname}"
            saved = plot_heatmap(results, out, subdir=subdir, metric=metric)
            desc = _heatmap_desc(subdir, metric)
            print(f"[{desc[:80]}...]\n  → {saved}")


if __name__ == "__main__":
    main()
