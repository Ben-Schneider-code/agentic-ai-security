"""
Compute efficiency: PVR_conv vs each team's cumulative training compute.

Two independent sweeps, each with a fixed strongest opponent:

  Left  — Red team learning curve:
    For each red checkpoint (red_0 = base model, red_1, ..., red_N):
      evaluate against the LATEST (strongest) blue checkpoint.
    X-axis: cumulative EIS spent on red team training only.
    Y-axis: PVR_conv (%) — higher means red is more capable.

  Right — Blue team learning curve:
    For each blue checkpoint (blue_0 = base model, blue_1, ..., blue_N):
      evaluate against the LATEST (strongest) red checkpoint.
    X-axis: cumulative EIS spent on blue team training only.
    Y-axis: PVR_conv (%) — lower means blue is defending better;
            BRR (%) overlaid — measures utility cost of defense.

Using the fixed strongest opponent (rather than the co-evolved opponent at each
iteration) isolates each team's improvement as a function of its own training
budget, independent of the other team's simultaneous training.

Data source: all pairings in cross_eval_results.json.
EIS per checkpoint: total_num_steps from training_state.json.

Can be run standalone:
    python plotting/plot_compute_efficiency.py --results results-<ID>
Or imported:
    from plotting.plot_compute_efficiency import plot_compute_efficiency, DESCRIPTION
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

try:
    from ._data import (
        apply_paper_style, discover_iterations, find_run_dir, load_training_state,
        load_cross_eval_results, load_benign_only,
        parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, discover_iterations, find_run_dir, load_training_state,
        load_cross_eval_results, load_benign_only,
        parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = (
    "Compute efficiency: PVR_conv vs each team's own cumulative training EIS, "
    "evaluated against the strongest fixed opponent. "
    "Left: red_N vs blue_last — PVR_conv as a function of red team training compute "
    "(higher = better attack). "
    "Right: red_last vs blue_N — PVR_conv (lower = better defense) and BRR as a "
    "function of blue team training compute. "
    "Isolates each team's improvement from its own training budget."
)

_CROSS_EVAL_SUBDIRS = ("cross_eval", "cross_eval_quick", "cross_eval_old")

# Metric key aliases inside each pairing dict
_ASR_KEY   = "asr"    # PVR_conv: conversation-level attack success rate (%)
_TPR_KEY   = "tpr"    # True positive rate on benign turns (%)
_ASR_CI    = "asr_ci"
_TPR_CI    = "tpr_ci"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cross_eval(selfplay_dir: str) -> dict | None:
    for subdir in _CROSS_EVAL_SUBDIRS:
        result = load_cross_eval_results(selfplay_dir, subdir=subdir)
        if result is not None:
            return result
    return None


def _pairing_key(red_iter: int, blue_iter: int) -> str:
    return f"red_{red_iter}_blue_{blue_iter}"


def load_team_eis(selfplay_dir: str) -> tuple[dict[int, int], dict[int, int]]:
    """
    Return (red_cumulative_eis, blue_cumulative_eis) dicts keyed by iter number.

    iter 0 (base model, no training) maps to 0 for both.
    iter N maps to the cumulative EIS spent training that team through iteration N.
    """
    iters_data = discover_iterations(selfplay_dir)

    red_cum: dict[int, int] = {0: 0}
    blue_cum: dict[int, int] = {0: 0}
    red_running = 0
    blue_running = 0

    for entry in iters_data:
        n = entry["iter"]

        red_dir = entry.get("red_dir")
        if red_dir:
            run_dir = find_run_dir(red_dir)
            if run_dir:
                try:
                    state = load_training_state(run_dir)
                    red_running += state.get("total_num_steps", 0)
                except FileNotFoundError:
                    pass
        red_cum[n] = red_running

        blue_dir = entry.get("blue_dir")
        if blue_dir:
            run_dir = find_run_dir(blue_dir)
            if run_dir:
                try:
                    state = load_training_state(run_dir)
                    blue_running += state.get("total_num_steps", 0)
                except FileNotFoundError:
                    pass
        blue_cum[n] = blue_running

    return red_cum, blue_cum


def _get_metric(pairing: dict, key: str) -> float | None:
    return pairing.get("metrics", {}).get(key)


def _get_ci(pairing: dict, key: str) -> list[float] | None:
    return pairing.get("confidence_intervals", {}).get(key)


def _get_tpr(pairing: dict, benign_only: dict, blue_iter: int) -> float:
    """Prefer benign-only TPR (evaluated on benign-only episodes) for BRR calculation."""
    bo = benign_only.get(f"blue_{blue_iter}", {})
    return bo.get("tpr") or _get_metric(pairing, _TPR_KEY) or 100.0


def build_red_sweep(
    cross_eval: dict,
    red_cum: dict[int, int],
    max_blue_iter: int,
) -> list[dict]:
    """
    red_N vs blue_last for N = 0 .. max_red_iter.
    Returns list of {iter, eis, pvr_conv, pvr_conv_lo, pvr_conv_hi}.
    """
    pairings = cross_eval.get("pairings", {})
    points = []
    for red_iter in sorted(red_cum.keys()):
        key = _pairing_key(red_iter, max_blue_iter)
        pairing = pairings.get(key)
        if pairing is None:
            continue
        pvr = _get_metric(pairing, _ASR_KEY)
        if pvr is None:
            continue
        ci = _get_ci(pairing, _ASR_CI) or [pvr, pvr]
        eis = red_cum.get(red_iter, 0)
        points.append({
            "iter": red_iter,
            "eis":        eis,
            "pvr_conv":   pvr,
            "pvr_conv_lo": ci[0],
            "pvr_conv_hi": ci[1],
        })
    return points


def build_blue_sweep(
    cross_eval: dict,
    blue_cum: dict[int, int],
    max_red_iter: int,
    benign_only: dict | None = None,
) -> list[dict]:
    """
    red_last vs blue_N for N = 0 .. max_blue_iter.
    Returns list of {iter, eis, pvr_conv, pvr_conv_lo, pvr_conv_hi, brr, brr_lo, brr_hi}.
    """
    pairings = cross_eval.get("pairings", {})
    bo_data  = benign_only or {}
    points = []
    for blue_iter in sorted(blue_cum.keys()):
        key = _pairing_key(max_red_iter, blue_iter)
        pairing = pairings.get(key)
        if pairing is None:
            continue
        pvr = _get_metric(pairing, _ASR_KEY)
        if pvr is None:
            continue
        pvr_ci = _get_ci(pairing, _ASR_CI) or [pvr, pvr]
        tpr = _get_tpr(pairing, bo_data, blue_iter)
        tpr_ci = _get_ci(pairing, _TPR_CI) or [tpr, tpr]
        brr    = 100.0 - tpr
        brr_lo = max(0.0, 100.0 - tpr_ci[1])
        brr_hi = 100.0 - tpr_ci[0]
        eis = blue_cum.get(blue_iter, 0)
        points.append({
            "iter": blue_iter,
            "eis":        eis,
            "pvr_conv":   pvr,
            "pvr_conv_lo": pvr_ci[0],
            "pvr_conv_hi": pvr_ci[1],
            "brr":    brr,
            "brr_lo": brr_lo,
            "brr_hi": brr_hi,
        })
    return points


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_sweep(
    ax: plt.Axes,
    points: list[dict],
    pvr_key: str,
    color: str,
    marker: str,
    label: str,
    annotate: bool = True,
) -> None:
    if not points:
        return
    eis = [p["eis"] for p in points]
    pvr = [p[pvr_key] for p in points]
    lo  = [p[f"{pvr_key}_lo"] for p in points]
    hi  = [p[f"{pvr_key}_hi"] for p in points]
    yerr = np.array([
        [v - l for v, l in zip(pvr, lo)],
        [h - v for v, h in zip(pvr, hi)],
    ])
    ax.errorbar(
        eis, pvr, yerr=yerr,
        fmt=f"-{marker}", color=color, linewidth=1.8, markersize=6,
        capsize=4, capthick=1.2, elinewidth=1.0,
        label=label, zorder=3,
    )
    if annotate:
        for p, x, y in zip(points, eis, pvr):
            ax.annotate(
                f"Iter {p['iter']}",
                xy=(x, y), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=8, color="#555555",
            )


_EIS_FMT = matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{int(v)}"
)


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------

def plot_compute_efficiency(
    results: list[tuple[str, str]],
    out_path: str | Path,
) -> Path:
    """
    Plot PVR_conv vs each team's own cumulative EIS using fixed-opponent sweeps.

    Left panel:  red_N vs blue_last  — red team learning curve.
    Right panel: red_last vs blue_N  — blue team defense curve + BRR.

    Args:
        results:  [(label, selfplay_dir), ...]
        out_path: Destination PNG path.

    Returns the resolved Path that was written.
    """
    out_path = Path(out_path)
    fig, (ax_red, ax_blue) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    for _label, selfplay_dir in results:
        cross_eval = _load_cross_eval(selfplay_dir)
        if cross_eval is None:
            print(f"[plot_compute_efficiency] No cross-eval data in {selfplay_dir}.",
                  file=sys.stderr)
            continue

        pairings = cross_eval.get("pairings", {})
        all_red_iters  = sorted({v["red_iter"]  for v in pairings.values()})
        all_blue_iters = sorted({v["blue_iter"] for v in pairings.values()})
        max_red_iter   = max(all_red_iters)
        max_blue_iter  = max(all_blue_iters)

        red_cum, blue_cum = load_team_eis(selfplay_dir)

        benign_only = load_benign_only(selfplay_dir)
        red_pts  = build_red_sweep(cross_eval, red_cum,  max_blue_iter)
        blue_pts = build_blue_sweep(cross_eval, blue_cum, max_red_iter, benign_only)

        # ── Left: red team learning curve ────────────────────────────────────
        _plot_sweep(ax_red, red_pts, "pvr_conv", RED_COL, "o",
                    f"PVR_conv (vs blue_{max_blue_iter})")

        # ── Right: blue team defense curve ───────────────────────────────────
        _plot_sweep(ax_blue, blue_pts, "pvr_conv", RED_COL, "o",
                    f"PVR_conv (vs red_{max_red_iter})")
        _plot_sweep(ax_blue, blue_pts, "brr", BLUE_COL, "s",
                    "BRR", annotate=False)

    # ── Shared reference: PVR of base model at 0 EIS ─────────────────────────
    # Mark 0 EIS explicitly so readers see the pre-training baseline
    for ax in (ax_red, ax_blue):
        ax.axvline(0, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)

    # ── Red panel formatting ──────────────────────────────────────────────────
    ax_red.set_title(
        r"Red Team: $\mathrm{PVR_{conv}}$ vs Red Training Compute"
        f"\n(evaluated against strongest blue)"
    )
    ax_red.set_xlabel("Cumulative Red Team EIS")
    ax_red.set_ylabel(r"$\mathrm{PVR_{conv}}$ (%)")
    ax_red.set_ylim(bottom=0)
    ax_red.xaxis.set_major_formatter(_EIS_FMT)
    ax_red.legend(fontsize=9, frameon=True)
    ax_red.grid(True, axis="y", alpha=0.4)

    # ── Blue panel formatting ─────────────────────────────────────────────────
    ax_blue.set_title(
        r"Blue Team: $\mathrm{PVR_{conv}}$ and BRR vs Blue Training Compute"
        f"\n(evaluated against strongest red)"
    )
    ax_blue.set_xlabel("Cumulative Blue Team EIS")
    ax_blue.set_ylabel("(%)")
    ax_blue.set_ylim(bottom=0)
    ax_blue.xaxis.set_major_formatter(_EIS_FMT)
    ax_blue.legend(fontsize=9, frameon=True)
    ax_blue.grid(True, axis="y", alpha=0.4)

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
        description=(
            "Plot PVR_conv/BRR vs each team's cumulative EIS "
            "(fixed strongest-opponent evaluation)."
        )
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument("--out", default="figures/compute_efficiency.png")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out = plot_compute_efficiency(results, args.out)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
