"""
Diagonal convergence — PVR_conv, PVR_turn, and BRR along the training trajectory.

Emits a family of three single-panel PNGs into the output directory:
  pvr_conv_trajectory.png
  pvr_turn_trajectory.png
  brr_trajectory.png

Each plots adjacent checkpoint pairings on the x-axis in training order:
  red_0 vs blue_0  →  red_1 vs blue_0  →  red_1 vs blue_1  →  red_2 vs blue_1  →  …

Points are colored red when red was trained last (red_N vs blue_{N-1}) and blue
when blue was trained last (red_N vs blue_N).

PVR_conv / PVR_turn: per-pairing from cross_eval pairings/.
BRR: per blue checkpoint — at trajectory point (r, b) we show BRR(blue_b).
     Default source: cross_eval/benign_only/blue_*/reward_debug.jsonl.

Preferred attack source: diagonal_eval/ (400 ep/cell); falls back to cross_eval/.

Can be run standalone:
    python plotting/plot_diagonal_convergence.py --results results-<ID> --out-dir figures/
Or imported:
    from plotting.plot_diagonal_convergence import (
        plot_diagonal_convergence,
        DESCRIPTION_PVR_CONV_TRAJ,
        DESCRIPTION_PVR_TURN_TRAJ,
        DESCRIPTION_BRR_TRAJ,
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style,
        denial_rate_with_ci,
        load_benign_eval_per_turn,
        load_cross_eval_benign_per_turn,
        load_pairing_metrics_with_decomposed,
        load_train_rollout_benign_turns,
        parse_results_arg,
        write_sidecar,
        RED_COL,
        BLUE_COL,
        GRAY_COL,
        FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        denial_rate_with_ci,
        load_benign_eval_per_turn,
        load_cross_eval_benign_per_turn,
        load_pairing_metrics_with_decomposed,
        load_train_rollout_benign_turns,
        parse_results_arg,
        write_sidecar,
        RED_COL,
        BLUE_COL,
        GRAY_COL,
        FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION_PVR_CONV_TRAJ = (
    "PVR_conv along the self-play training trajectory: each adjacent "
    "checkpoint pairing (red_0 vs blue_0 → red_1 vs blue_0 → red_1 vs blue_1 → …). "
    "Red markers = red trained last; blue markers = blue trained last. "
    "Attack source: diagonal_eval/ preferred, else cross_eval/."
)

DESCRIPTION_PVR_TURN_TRAJ = (
    "PVR_turn along the self-play training trajectory (adjacent pairings). "
    "Marker fill encodes who trained last. "
    "Attack source: diagonal_eval/ preferred, else cross_eval/."
)

DESCRIPTION_BRR_TRAJ = (
    "BRR along the self-play training trajectory (adjacent pairings). "
    "For each (r, b) point, BRR is taken from blue checkpoint b. "
    "Default source: cross_eval/benign_only/."
)

_SUBDIR_PRIORITY = ("diagonal_eval", "cross_eval")
_BRR_SOURCES = ("cross_eval", "benign_eval", "train_rollouts")

# Distinct per-replicate line colors. Deliberately avoids RED_COL / BLUE_COL,
# which are reserved for the "who trained last" marker fill.
_REP_COLORS = ("#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b")

_FILENAMES = {
    "pvr_conv": "pvr_conv_trajectory.png",
    "pvr_turn": "pvr_turn_trajectory.png",
    "brr":      "brr_trajectory.png",
}

_PANEL_TITLES = {
    "pvr_conv": r"$\mathrm{PVR_{conv}}$ vs. Training Trajectory",
    "pvr_turn": r"$\mathrm{PVR_{turn}}$ vs. Training Trajectory",
    "brr":      r"BRR vs. Training Trajectory",
}

_PANEL_YLABELS = {
    "pvr_conv": r"$\mathrm{PVR_{conv}}$ (%)",
    "pvr_turn": r"$\mathrm{PVR_{turn}}$ (%)",
    "brr":      r"BRR (%)",
}


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


def _extract_attack_series(
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

        pvr_conv.append(m.get("asr", float("nan")))
        asr_ci = ci.get("asr")
        pvr_conv_lo.append(asr_ci[0] if asr_ci else float("nan"))
        pvr_conv_hi.append(asr_ci[1] if asr_ci else float("nan"))

        pvr_turn.append(m.get("pvr_turn", float("nan")))
        pt_ci = ci.get("pvr_turn")
        pvr_turn_lo.append(pt_ci[0] if pt_ci else float("nan"))
        pvr_turn_hi.append(pt_ci[1] if pt_ci else float("nan"))

    return (
        pvr_conv, pvr_conv_lo, pvr_conv_hi,
        pvr_turn, pvr_turn_lo, pvr_turn_hi,
    )


def _extract_brr_series(
    seq: list[tuple[int, int]],
    brr_rows_by_blue: dict[int, list[dict]],
) -> tuple[list[float], list[float], list[float]]:
    """
    For each (r, b) in seq, compute BRR (and 99% Wilson CI) from the rows
    associated with blue checkpoint b. Returns three lists in trajectory order.
    """
    vals, lo, hi = [], [], []
    for _r_iter, b_iter in seq:
        rows = brr_rows_by_blue.get(b_iter)
        if not rows:
            vals.append(float("nan"))
            lo.append(float("nan"))
            hi.append(float("nan"))
            continue
        rate, lo_pct, hi_pct, _k, _n = denial_rate_with_ci(rows)
        vals.append(rate)
        lo.append(lo_pct)
        hi.append(hi_pct)
    return vals, lo, hi


def _load_brr_rows(
    selfplay_dir: str,
    source: str,
    cross_eval_subdir: str,
) -> dict[int, list[dict]]:
    if source == "cross_eval":
        return load_cross_eval_benign_per_turn(selfplay_dir, subdir=cross_eval_subdir)
    if source == "benign_eval":
        return load_benign_eval_per_turn(selfplay_dir)
    if source == "train_rollouts":
        return load_train_rollout_benign_turns(selfplay_dir)
    raise ValueError(f"Unknown brr_source: {source!r} (choose from {_BRR_SOURCES})")


def _render_panel(
    ax: plt.Axes,
    x: np.ndarray,
    x_labels: list[str],
    series: list[tuple[str, str, list[float], list[float], list[float]]],
    point_colors: list[str],
    ylabel: str,
    title: str,
    show_ci: bool = True,
) -> None:
    """
    Render one trajectory panel with one line per replicate.

    series:       [(rep_label, rep_color, vals, lo, hi), …]; every value list is
                  aligned to `x` (0–100 scale, NaN where the pairing is absent).
    point_colors: per-x marker fill color — red/blue = who trained last.
    """
    n = len(series)
    offsets = np.linspace(-0.15, 0.15, n) if n > 1 else np.zeros(max(n, 1))

    rep_handles: list[Line2D] = []
    for (rep_label, rep_color, vals, lo, hi), off in zip(series, offsets):
        xs = x + off
        valid = [(xi, vi) for xi, vi in zip(xs, vals) if not np.isnan(vi)]
        if len(valid) >= 2:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, color=rep_color, linewidth=1.2, alpha=0.8, zorder=1)
        rep_handles.append(Line2D([0], [0], color=rep_color, linewidth=2,
                                  label=rep_label))

        for xi, vi, li, hi_i, pcol in zip(xs, vals, lo, hi, point_colors):
            if np.isnan(vi):
                continue
            yerr_lo = max(0.0, vi - li) if not np.isnan(li) else 0.0
            yerr_hi = max(0.0, hi_i - vi) if not np.isnan(hi_i) else 0.0
            ax.errorbar(
                xi, vi,
                yerr=([[yerr_lo], [yerr_hi]] if show_ci else None),
                fmt="o",
                mfc=pcol,
                mec=rep_color,
                ecolor=rep_color,
                markersize=7,
                markeredgewidth=1.6,
                capsize=3,
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
    ax.legend(handles=rep_handles + [red_patch, blue_patch],
              fontsize=8, loc="upper right")


def _save_individual(
    out_path: Path,
    x: np.ndarray,
    x_labels: list[str],
    series: list[tuple[str, str, list[float], list[float], list[float]]],
    point_colors: list[str],
    ylabel: str,
    title: str,
    show_ci: bool,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    _render_panel(ax, x, x_labels, series, point_colors, ylabel, title,
                  show_ci=show_ci)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _load_replicate(
    label: str,
    selfplay_dir: str,
    eval_subdir: str | None,
    brr_source: str,
) -> dict | None:
    """
    Load one replicate's pairing lookup and BRR rows. Returns None when the
    replicate has no usable cross-eval data.
    """
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

    if cross_eval is None:
        print(
            f"[plot_diagonal_convergence] No cross-eval data found in {selfplay_dir}.",
            file=sys.stderr,
        )
        return None

    pairings = cross_eval.get("pairings", {})
    seq = _adjacent_sequence(pairings)
    if not seq:
        print(
            f"[plot_diagonal_convergence] No adjacent pairings in "
            f"{selfplay_dir}/{chosen_subdir}.",
            file=sys.stderr,
        )
        return None

    pairing_lookup: dict[tuple[int, int], dict] = {
        (v["red_iter"], v["blue_iter"]): v
        for v in pairings.values()
        if "red_iter" in v and "blue_iter" in v
    }

    brr_rows_by_blue = _load_brr_rows(
        selfplay_dir,
        brr_source,
        cross_eval_subdir=chosen_subdir if brr_source == "cross_eval" else "cross_eval",
    )
    if not brr_rows_by_blue:
        print(
            f"[plot_diagonal_convergence] No BRR rows found for source "
            f"{brr_source!r} in {selfplay_dir}.",
            file=sys.stderr,
        )

    return {
        "label": label,
        "selfplay_dir": selfplay_dir,
        "chosen_subdir": chosen_subdir,
        "seq": seq,
        "pairing_lookup": pairing_lookup,
        "brr_rows": brr_rows_by_blue,
    }


def _empty_placeholder(out_dir: Path, msg: str) -> dict[str, Path]:
    """Write three single-panel placeholder PNGs and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, fname in _FILENAMES.items():
        p = out_dir / fname
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                color=GRAY_COL)
        ax.set_title(_PANEL_TITLES[key])
        ax.set_ylabel(_PANEL_YLABELS[key])
        fig.tight_layout(pad=0.4)
        fig.savefig(p)
        plt.close(fig)
        paths[key] = p
    return paths


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------


def compute_diagonal_convergence(
    results: list[tuple[str, str]],
    *,
    eval_subdir: str | None = None,
    brr_source: str = "cross_eval",
    **kwargs,
) -> dict:
    """Pure data load + metric math for the diagonal-convergence trajectories.

    Loads every replicate's adjacent-pairing PVR_conv / PVR_turn series and the
    BRR rows, computes the per-replicate trajectory arrays (with 99% Wilson CIs)
    on the master x-axis, and returns the JSON-serializable metrics dict that the
    three trajectory renderers are fully driven by. Returns ``{}`` when no
    replicate has usable cross-eval data or when there are no adjacent pairings
    (mirrors the plot's placeholder branches). No matplotlib, no file writes.

    Per-replicate arrays are stored at full precision under
    ``<metric>_pct_raw`` / ``<metric>_ci_99_raw`` (NaN where data is absent) so
    the renderer reproduces the figure exactly; the rounded ``<metric>_pct`` /
    ``<metric>_ci_99`` keys are kept for the sidecar.

    Accepts and ignores unknown kwargs (e.g. render-only ``show_ci``).
    """
    if brr_source not in _BRR_SOURCES:
        raise ValueError(f"brr_source must be one of {_BRR_SOURCES}, got {brr_source!r}")

    reps = [
        rep for rep in (
            _load_replicate(label, selfplay_dir, eval_subdir, brr_source)
            for label, selfplay_dir in results
        )
        if rep is not None
    ]

    if not reps:
        print(
            "[plot_diagonal_convergence] No usable cross-eval data in any replicate.",
            file=sys.stderr,
        )
        return {}

    master_seq = sorted({pair for rep in reps for pair in rep["seq"]})
    if not master_seq:
        return {}

    x_labels = [f"R{r}·B{b}" for r, b in master_seq]

    def _r(v: float) -> float | None:
        return round(v, 2) if not np.isnan(v) else None

    rep_metrics: dict[str, dict] = {}

    for i, rep in enumerate(reps):
        col = _REP_COLORS[i % len(_REP_COLORS)]
        (
            cv, cv_lo, cv_hi,
            tv, tv_lo, tv_hi,
        ) = _extract_attack_series(master_seq, rep["pairing_lookup"])
        bv, bv_lo, bv_hi = _extract_brr_series(master_seq, rep["brr_rows"])

        rep_metrics[rep["label"]] = {
            "selfplay_dir": rep["selfplay_dir"],
            "source_subdir": rep["chosen_subdir"],
            # Render-driver color + full-precision arrays (NaN where absent).
            "rep_color": col,
            "pvr_conv_pct_raw": cv,
            "pvr_conv_ci_99_raw": [[l, h] for l, h in zip(cv_lo, cv_hi)],
            "pvr_turn_pct_raw": tv,
            "pvr_turn_ci_99_raw": [[l, h] for l, h in zip(tv_lo, tv_hi)],
            "brr_pct_raw": bv,
            "brr_ci_99_raw": [[l, h] for l, h in zip(bv_lo, bv_hi)],
            # Rounded copies for the JSON sidecar.
            "pvr_conv_pct": [_r(v) for v in cv],
            "pvr_conv_ci_99": [[_r(l), _r(h)] for l, h in zip(cv_lo, cv_hi)],
            "pvr_turn_pct": [_r(v) for v in tv],
            "pvr_turn_ci_99": [[_r(l), _r(h)] for l, h in zip(tv_lo, tv_hi)],
            "brr_pct": [_r(v) for v in bv],
            "brr_ci_99": [[_r(l), _r(h)] for l, h in zip(bv_lo, bv_hi)],
        }

    chosen_subdir = reps[0]["chosen_subdir"]
    metrics = {
        "source_subdir": chosen_subdir,
        "brr_source": brr_source,
        "trajectory_labels": x_labels,
        "trajectory_seq": [{"red_iter": r, "blue_iter": b} for r, b in master_seq],
        "replicates": rep_metrics,
    }
    return metrics


def plot_diagonal_convergence(
    results: list[tuple[str, str]],
    out_dir: str | Path,
    *,
    eval_subdir: str | None = None,
    show_ci: bool = True,
    brr_source: str = "cross_eval",
    precomputed: dict | None = None,
) -> tuple[dict[str, Path], dict]:
    """
    Plot PVR_conv, PVR_turn, and BRR for adjacent pairings along the training
    diagonal — one single-panel PNG per metric.

    Every entry in `results` is rendered as its own trajectory line; the
    x-axis is the union of all replicates' adjacent pairings in trajectory
    order. Lines are colored per replicate; markers keep the red/blue
    "who trained last" fill.

    Args:
        results:     [(label, selfplay_dir), …] — all entries are plotted.
        out_dir:     Destination directory. Writes pvr_conv_trajectory.png,
                     pvr_turn_trajectory.png, and brr_trajectory.png.
        eval_subdir: Subdir for attack pairings (auto: diagonal_eval, cross_eval).
        show_ci:     Toggle 99% Wilson CI error bars.
        brr_source:  "cross_eval" (default), "benign_eval", or "train_rollouts".
        precomputed: Optional metrics dict from compute_diagonal_convergence; when
                     None it is computed here.

    Returns:
        (paths, metrics) where paths is {"pvr_conv": Path, "pvr_turn": Path,
        "brr": Path} and metrics is the shared metrics dict for all three.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if precomputed is None:
        metrics = compute_diagonal_convergence(
            results, eval_subdir=eval_subdir, brr_source=brr_source
        )
    else:
        metrics = precomputed

    if not metrics:
        # Distinguish the two empty branches by message, matching the originals.
        msg = "No data"
        return _empty_placeholder(out_dir, msg), {}

    # Reconstruct everything the renderer needs straight from the metrics dict.
    master_seq = [(d["red_iter"], d["blue_iter"]) for d in metrics["trajectory_seq"]]
    x        = np.arange(len(master_seq))
    x_labels = metrics["trajectory_labels"]
    point_colors = [_point_color(r, b) for r, b in master_seq]

    conv_series: list[tuple] = []
    turn_series: list[tuple] = []
    brr_series:  list[tuple] = []
    for label, rep in metrics["replicates"].items():
        col = rep["rep_color"]
        cv = rep["pvr_conv_pct_raw"]
        cv_lo = [c[0] for c in rep["pvr_conv_ci_99_raw"]]
        cv_hi = [c[1] for c in rep["pvr_conv_ci_99_raw"]]
        tv = rep["pvr_turn_pct_raw"]
        tv_lo = [c[0] for c in rep["pvr_turn_ci_99_raw"]]
        tv_hi = [c[1] for c in rep["pvr_turn_ci_99_raw"]]
        bv = rep["brr_pct_raw"]
        bv_lo = [c[0] for c in rep["brr_ci_99_raw"]]
        bv_hi = [c[1] for c in rep["brr_ci_99_raw"]]

        conv_series.append((label, col, cv, cv_lo, cv_hi))
        turn_series.append((label, col, tv, tv_lo, tv_hi))
        brr_series.append((label, col, bv, bv_lo, bv_hi))

    panels = {
        "pvr_conv": conv_series,
        "pvr_turn": turn_series,
        "brr":      brr_series,
    }
    paths: dict[str, Path] = {}
    for key, series in panels.items():
        p = out_dir / _FILENAMES[key]
        _save_individual(
            p, x, x_labels, series, point_colors,
            _PANEL_YLABELS[key], _PANEL_TITLES[key],
            show_ci=show_ci,
        )
        paths[key] = p

    return paths, metrics


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


_DESC_BY_KEY = {
    "pvr_conv": DESCRIPTION_PVR_CONV_TRAJ,
    "pvr_turn": DESCRIPTION_PVR_TURN_TRAJ,
    "brr":      DESCRIPTION_BRR_TRAJ,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PVR_conv, PVR_turn, and BRR for adjacent checkpoint pairings "
            "along the self-play training trajectory. Emits three single-panel "
            "PNGs into --out-dir."
        )
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
        help="Selfplay result directory, optionally with :Label suffix.",
    )
    parser.add_argument("--out-dir", default="figures/", metavar="DIR")
    parser.add_argument(
        "--eval-subdir", default=None, metavar="SUBDIR",
        help="Subdir for pairing data (default: tries diagonal_eval then cross_eval).",
    )
    parser.add_argument(
        "--brr-source", default="cross_eval", choices=list(_BRR_SOURCES),
        help="BRR data source (default: cross_eval).",
    )
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    paths, metrics = plot_diagonal_convergence(
        results, args.out_dir,
        eval_subdir=args.eval_subdir,
        brr_source=args.brr_source,
    )
    for key, p in paths.items():
        write_sidecar(p, _DESC_BY_KEY[key], results, metrics)
        print(f"[{_DESC_BY_KEY[key]}]\n  → {p}")


if __name__ == "__main__":
    main()
