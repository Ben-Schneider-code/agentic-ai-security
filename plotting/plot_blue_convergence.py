"""
Blue team defense convergence: plateau analysis and marginal efficiency.

Demonstrates that the blue team's PVR_conv has saturated across iterations
despite increasing compute, so additional training budget would not close the
gap against the strongest red.

Left panel — PVR_conv vs cumulative blue EIS:
  Raw values with error bars, an estimated plateau band (mean ± 1σ of the
  final ceil(N/2) iterations), and an extrapolation arrow to show the
  asymptote extends forward.

Right panel — Marginal ΔPVR_conv per 1,000 additional blue EIS:
  Bar chart of the discrete derivative between consecutive blue checkpoints.
  Red bars = PVR_conv rose (blue regressed / attack worsened for blue).
  Blue bars = PVR_conv fell (blue defended / attack improved for blue).
  Near-zero or sign-alternating bars in later iterations = diminishing returns.

Data source: blue sweep from cross_eval_results.json (blue_N vs red_last),
same as plot_compute_efficiency.py.

Can be run standalone:
    python plotting/plot_blue_convergence.py --results results-<ID>
Or imported:
    from plotting.plot_blue_convergence import plot_blue_convergence, DESCRIPTION
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
        apply_paper_style, parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_1x2,
    )
    from .plot_compute_efficiency import (
        _load_cross_eval, load_team_eis, build_blue_sweep,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, parse_results_arg,
        RED_COL, BLUE_COL, FIG_SIZE_1x2,
    )
    from plotting.plot_compute_efficiency import (
        _load_cross_eval, load_team_eis, build_blue_sweep,
    )

apply_paper_style()

DESCRIPTION = (
    "Blue team convergence: PVR_conv vs cumulative blue EIS plateaus despite "
    "increasing compute. Left: raw PVR_conv with plateau asymptote band "
    "(mean ± 1σ of the final half of iterations). "
    "Right: marginal ΔPVR_conv per 1,000 additional blue EIS — "
    "near-zero or sign-alternating bars in later iterations show diminishing returns."
)

_EIS_FMT = matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{int(v)}"
)


def _plateau_estimate(pts: list[dict]) -> tuple[float, float, int]:
    """
    Estimate convergence plateau from the final ceil(N/2) points.
    Returns (mean_pvr, std_pvr, n_tail_points).
    """
    n_tail = max(1, (len(pts) + 1) // 2)
    tail_vals = [p["pvr_conv"] for p in pts[-n_tail:]]
    return float(np.mean(tail_vals)), float(np.std(tail_vals)), n_tail


def plot_blue_convergence(
    results: list[tuple[str, str]],
    out_path: str | Path,
) -> Path:
    """
    Plot blue team PVR_conv plateau and marginal efficiency bars.

    Args:
        results:  [(label, selfplay_dir), ...]
        out_path: Destination PNG path.

    Returns the resolved Path that was written.
    """
    out_path = Path(out_path)
    fig, (ax_pvr, ax_marg) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    last_max_red_iter: int | None = None

    for _label, selfplay_dir in results:
        cross_eval = _load_cross_eval(selfplay_dir)
        if cross_eval is None:
            print(f"[plot_blue_convergence] No cross-eval data in {selfplay_dir}.",
                  file=sys.stderr)
            continue

        pairings = cross_eval.get("pairings", {})
        all_red_iters = sorted({v["red_iter"] for v in pairings.values()})
        max_red_iter  = max(all_red_iters)
        last_max_red_iter = max_red_iter

        _, blue_cum = load_team_eis(selfplay_dir)
        blue_pts = build_blue_sweep(cross_eval, blue_cum, max_red_iter)
        if not blue_pts:
            continue

        eis = [p["eis"]         for p in blue_pts]
        pvr = [p["pvr_conv"]    for p in blue_pts]
        lo  = [p["pvr_conv_lo"] for p in blue_pts]
        hi  = [p["pvr_conv_hi"] for p in blue_pts]

        yerr = np.array([
            [v - l for v, l in zip(pvr, lo)],
            [h - v for v, h in zip(pvr, hi)],
        ])

        # ── Left: PVR_conv with plateau band ─────────────────────────────────
        ax_pvr.errorbar(
            eis, pvr, yerr=yerr,
            fmt="-o", color=RED_COL, linewidth=1.8, markersize=6,
            capsize=4, capthick=1.2, elinewidth=1.0,
            label=f"PVR_conv (vs red_{max_red_iter})", zorder=3,
        )
        for p, x, y in zip(blue_pts, eis, pvr):
            ax_pvr.annotate(
                f"Iter {p['iter']}",
                xy=(x, y), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=8, color="#555555",
            )

        plat_mean, plat_std, n_tail = _plateau_estimate(blue_pts)
        plateau_start_eis = blue_pts[-n_tail]["eis"]
        max_eis = max(eis)
        extrap_end = max_eis * 1.18

        ax_pvr.axhline(
            plat_mean, color=BLUE_COL, linewidth=1.5, linestyle="--", zorder=2,
            label=f"Plateau mean ({plat_mean:.1f}%)",
        )
        ax_pvr.fill_between(
            [plateau_start_eis, extrap_end],
            plat_mean - plat_std,
            plat_mean + plat_std,
            color=BLUE_COL, alpha=0.15, zorder=1,
            label=f"±1σ  ({plat_std:.1f} pp)",
        )
        # Extrapolation arrow: "more compute doesn't push this lower"
        ax_pvr.annotate(
            "",
            xy=(extrap_end, plat_mean),
            xytext=(max_eis * 0.99, plat_mean),
            arrowprops=dict(arrowstyle="->", color=BLUE_COL, lw=1.5),
        )
        ax_pvr.text(
            extrap_end * 1.002, plat_mean + 0.4,
            "asymptote",
            fontsize=8, color=BLUE_COL, va="bottom",
        )

        # ── Right: marginal ΔPVR_conv per 1k EIS ─────────────────────────────
        marg_vals:   list[float] = []
        marg_labels: list[str]   = []
        for i in range(1, len(blue_pts)):
            d_eis = blue_pts[i]["eis"] - blue_pts[i - 1]["eis"]
            d_pvr = blue_pts[i]["pvr_conv"] - blue_pts[i - 1]["pvr_conv"]
            if d_eis <= 0:
                continue
            marg_vals.append(d_pvr / d_eis * 1000)
            marg_labels.append(f"{blue_pts[i-1]['iter']}→{blue_pts[i]['iter']}")

        if marg_vals:
            bar_colors = [RED_COL if v > 0 else BLUE_COL for v in marg_vals]
            bars = ax_marg.bar(
                range(len(marg_vals)), marg_vals,
                color=bar_colors, alpha=0.80, edgecolor="none",
            )
            ax_marg.axhline(0, color="#888888", linewidth=1.0)
            ax_marg.set_xticks(range(len(marg_labels)))
            ax_marg.set_xticklabels(
                [f"Iter\n{lbl}" for lbl in marg_labels], fontsize=9,
            )
            for bar, val in zip(bars, marg_vals):
                va     = "bottom" if val >= 0 else "top"
                offset = 0.04 * max(abs(v) for v in marg_vals) if marg_vals else 0.05
                offset = offset if val >= 0 else -offset
                ax_marg.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + offset,
                    f"{val:+.2f}",
                    ha="center", va=va, fontsize=9, color="#222222",
                )

    # ── Left panel formatting ─────────────────────────────────────────────────
    title_suffix = f" (vs red_{last_max_red_iter})" if last_max_red_iter is not None else ""
    ax_pvr.set_title(
        r"Blue Defense: $\mathrm{PVR_{conv}}$ vs Cumulative Blue EIS"
        + f"\n{title_suffix}"
    )
    ax_pvr.set_xlabel("Cumulative Blue Team EIS")
    ax_pvr.set_ylabel(r"$\mathrm{PVR_{conv}}$ (%)")
    ax_pvr.set_ylim(bottom=0)
    ax_pvr.xaxis.set_major_formatter(_EIS_FMT)
    ax_pvr.legend(fontsize=9, frameon=True, loc="upper right")
    ax_pvr.grid(True, axis="y", alpha=0.4)

    # ── Right panel formatting ────────────────────────────────────────────────
    ax_marg.set_title(
        r"Marginal Blue Efficiency: $\Delta\mathrm{PVR_{conv}}$ per 1k EIS"
    )
    ax_marg.set_xlabel("Iteration transition")
    ax_marg.set_ylabel(r"$\Delta\mathrm{PVR_{conv}}$ / 1k EIS  (pp / k)")
    ax_marg.grid(True, axis="y", alpha=0.4)
    ax_marg.text(
        0.97, 0.97, "↑ PVR_conv rose\n(blue regressed)",
        transform=ax_marg.transAxes, fontsize=8,
        ha="right", va="top", color=RED_COL,
    )
    ax_marg.text(
        0.97, 0.03, "↓ PVR_conv fell\n(blue defended)",
        transform=ax_marg.transAxes, fontsize=8,
        ha="right", va="bottom", color=BLUE_COL,
    )

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
            "Plot blue team PVR_conv convergence plateau and marginal efficiency."
        )
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument("--out", default="figures/blue_convergence.png")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out = plot_blue_convergence(results, args.out)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
