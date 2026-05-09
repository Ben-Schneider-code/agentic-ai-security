"""
Direct compute-efficiency comparison: ΔPVRconv vs normalized training budget.

Both teams' learning curves are placed on a SHARED panel with a normalized
x-axis (0 = no training, 1 = full training budget used).  Y-axis shows
ΔPVRconv from the no-training baseline (iter 0):

  Red  (positive = better attack):  PVR_conv rose — red improved.
  Blue (negative = better defense): PVR_conv fell — blue improved.

If the red curve rises while the blue curve stays near 0, red training is
more compute-efficient per unit of its own budget regardless of the absolute
EIS asymmetry between teams.

Data source: fixed-strongest-opponent evaluation from cross_eval_results.json
(same pairings as compute_efficiency.py: red_N vs blue_last, blue_N vs red_last).

Can be run standalone:
    python plotting/plot_pvr_vs_normalized_eis.py --results results-<ID>
Or imported:
    from plotting.plot_pvr_vs_normalized_eis import plot_pvr_vs_normalized_eis, DESCRIPTION
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
        apply_paper_style, parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_SINGLE,
    )
    from .plot_compute_efficiency import (
        _load_cross_eval, load_team_eis, build_red_sweep, build_blue_sweep,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_SINGLE,
    )
    from plotting.plot_compute_efficiency import (
        _load_cross_eval, load_team_eis, build_red_sweep, build_blue_sweep,
    )

apply_paper_style()

DESCRIPTION = (
    "Compute efficiency comparison: ΔPVRconv vs normalized training budget (0→1) "
    "for red and blue teams on shared axes. "
    "Red ΔPVRconv > 0 means better attack per unit compute. "
    "Blue ΔPVRconv < 0 means better defense per unit compute. "
    "A larger red-curve slope vs. near-flat blue curve shows attack is more compute-efficient."
)


def _normalize_eis(pts: list[dict]) -> list[float]:
    """Normalize EIS to [0, 1] using each team's own max EIS."""
    max_eis = max((p["eis"] for p in pts), default=1)
    return [p["eis"] / max(max_eis, 1) for p in pts]


def _delta_pvr(pts: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Compute ΔPVRconv and CI deltas relative to the iter-0 baseline."""
    if not pts:
        return [], [], []
    baseline = pts[0]["pvr_conv"]
    vals = [p["pvr_conv"]    - baseline for p in pts]
    lo   = [p["pvr_conv_lo"] - baseline for p in pts]
    hi   = [p["pvr_conv_hi"] - baseline for p in pts]
    return vals, lo, hi


def plot_pvr_vs_normalized_eis(
    results: list[tuple[str, str]],
    out_path: str | Path,
) -> Path:
    """
    Plot ΔPVRconv vs normalized EIS for red and blue teams on shared axes.

    Args:
        results:  [(label, selfplay_dir), ...]
        out_path: Destination PNG path.

    Returns the resolved Path that was written.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

    all_delta_vals: list[float] = []

    for _label, selfplay_dir in results:
        cross_eval = _load_cross_eval(selfplay_dir)
        if cross_eval is None:
            print(f"[plot_pvr_vs_normalized_eis] No cross-eval data in {selfplay_dir}.",
                  file=sys.stderr)
            continue

        pairings = cross_eval.get("pairings", {})
        all_red_iters  = sorted({v["red_iter"]  for v in pairings.values()})
        all_blue_iters = sorted({v["blue_iter"] for v in pairings.values()})
        max_red_iter   = max(all_red_iters)
        max_blue_iter  = max(all_blue_iters)

        red_cum, blue_cum = load_team_eis(selfplay_dir)

        red_pts  = build_red_sweep(cross_eval, red_cum,  max_blue_iter)
        blue_pts = build_blue_sweep(cross_eval, blue_cum, max_red_iter)

        # ── Red team ──────────────────────────────────────────────────────────
        if red_pts:
            x_red = _normalize_eis(red_pts)
            d_pvr, d_lo, d_hi = _delta_pvr(red_pts)
            all_delta_vals.extend(d_pvr)
            yerr = np.array([
                [v - l for v, l in zip(d_pvr, d_lo)],
                [h - v for v, h in zip(d_pvr, d_hi)],
            ])
            ax.errorbar(
                x_red, d_pvr, yerr=yerr,
                fmt="-o", color=RED_COL, linewidth=2.0, markersize=6,
                capsize=4, capthick=1.2, elinewidth=1.0,
                label=f"Red team  (vs blue_{max_blue_iter})", zorder=3,
            )
            for p, x, y in zip(red_pts, x_red, d_pvr):
                ax.annotate(
                    f"Iter {p['iter']}",
                    xy=(x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color="#555555",
                )

        # ── Blue team ─────────────────────────────────────────────────────────
        if blue_pts:
            x_blue = _normalize_eis(blue_pts)
            d_pvr, d_lo, d_hi = _delta_pvr(blue_pts)
            all_delta_vals.extend(d_pvr)
            yerr = np.array([
                [v - l for v, l in zip(d_pvr, d_lo)],
                [h - v for v, h in zip(d_pvr, d_hi)],
            ])
            ax.errorbar(
                x_blue, d_pvr, yerr=yerr,
                fmt="-s", color=BLUE_COL, linewidth=2.0, markersize=6,
                capsize=4, capthick=1.2, elinewidth=1.0,
                label=f"Blue team  (vs red_{max_red_iter})", zorder=3,
            )

    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1,
               label="Baseline (iter 0, no training)")

    # Direction annotations, placed at fixed fractions of the y-range
    y_lo, y_hi = ax.get_ylim()
    y_span = max(y_hi - y_lo, 1.0)
    ax.annotate(
        "← better defense (blue, PVR_conv fell)",
        xy=(0.98, y_lo + 0.08 * y_span), xycoords="data",
        fontsize=8, color=BLUE_COL, ha="right", va="bottom",
    )
    ax.annotate(
        "better attack (red, PVR_conv rose) →",
        xy=(0.98, y_hi - 0.08 * y_span), xycoords="data",
        fontsize=8, color=RED_COL, ha="right", va="top",
    )

    ax.set_xlabel("Normalized training budget  (0 = no training, 1 = full budget)")
    ax.set_ylabel(r"$\Delta\mathrm{PVR_{conv}}$ from baseline (pp)")
    ax.set_title(
        r"Compute Efficiency: $\Delta\mathrm{PVR_{conv}}$ vs Normalized EIS"
        "\n(fixed strongest-opponent evaluation)"
    )
    ax.set_xlim(-0.05, 1.10)
    ax.legend(fontsize=9, frameon=True)
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
        description=(
            "Plot ΔPVRconv vs normalized EIS for red and blue teams on shared axes."
        )
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument("--out", default="figures/pvr_vs_normalized_eis.png")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out = plot_pvr_vs_normalized_eis(results, args.out)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
