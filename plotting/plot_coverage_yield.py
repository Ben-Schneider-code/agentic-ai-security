"""
Attempted vs. successful breach per (Red_i × Blue_j).

Attempted breach (referenced) = fraction of the honeypot universe red's SQL
*referenced* — parser-level intent: the query touched a honeypot row/column.
Successful breach (accessed)  = fraction red actually *violated* — executor-level
realization: the honeypot row materialized in the result set (accessed=True).

The story: as Blue evolves, attempted-breach stays high (red keeps targeting
honeypots) but successful-breach drops (blue filters execution) — a widening
gap proves blue is blocking at the execution layer, not the intent layer.

Underlying data fields in pairing metrics JSON are still named `coverage_pct`
and `yield_pct` (unchanged for backward compatibility); only the user-facing
labels and output filenames have been renamed.

CLI usage:
    python plotting/plot_coverage_yield.py --results results-<ID>
    python plotting/plot_coverage_yield.py --results results-<ID> --subdir cross_eval_quick
Or imported:
    from plotting.plot_coverage_yield import plot_coverage_yield, COVERAGE_YIELD_JOBS, DESCRIPTION
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
        load_pairing_metrics_with_decomposed,
        parse_results_arg,
        RED_COL,
        GRAY_COL,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_pairing_metrics_with_decomposed,
        parse_results_arg,
        RED_COL,
        GRAY_COL,
    )

apply_paper_style()

DESCRIPTION = (
    "Attempted vs. successful breach per (Red_i × Blue_j). Attempted breach = "
    "fraction of the honeypot universe red's SQL referenced (parser-level intent). "
    "Successful breach = fraction red actually violated (executor-level "
    "realization; accessed=True). A widening gap between the lines means blue is "
    "filtering at execution, not intent — attempted-breach stays high while "
    "successful-breach drops."
)

COVERAGE_YIELD_JOBS: list[tuple[str, str]] = [
    ("cross_eval",       "cross_eval_attempted_vs_successful_breach.png"),
    ("cross_eval_quick", "quick_attempted_vs_successful_breach.png"),
    ("diagonal_eval",    "diagonal_eval_attempted_vs_successful_breach.png"),
]


def plot_coverage_yield(
    results: list[tuple[str, str]],
    out_path: str | Path,
    *,
    subdir: str = "cross_eval",
    show_ci: bool = True,
) -> Path:
    """
    Coverage vs. yield line chart: one subplot per Red iter, x=Blue iter.

    Two errorbar lines per subplot: coverage (solid grey) and yield (dashed red).
    Error bars are 99% Wilson CIs. Only the first result entry is used.
    """
    out_path = Path(out_path)
    if len(results) > 1:
        print(
            "[plot_coverage_yield] Multiple runs given; using first run only.",
            file=sys.stderr,
        )

    _label, selfplay_dir = results[0]
    cross_eval = load_pairing_metrics_with_decomposed(selfplay_dir, subdir=subdir)
    if cross_eval is None:
        print(
            f"[plot_coverage_yield] No data for subdir={subdir!r} in {selfplay_dir}.",
            file=sys.stderr,
        )
        return out_path

    pairings = cross_eval.get("pairings", {})
    red_iters = sorted({v["red_iter"] for v in pairings.values()})
    blue_iters = sorted({v["blue_iter"] for v in pairings.values()})

    if not red_iters:
        print("[plot_coverage_yield] No pairings found.", file=sys.stderr)
        return out_path

    # Check if coverage_pct is available (requires MARFT import at metric-compute time).
    has_coverage = any(
        pairings.get(f"red_{red_iters[0]}_blue_{bi}", {})
        .get("metrics", {})
        .get("coverage_pct") is not None
        for bi in blue_iters
    )

    n_red = len(red_iters)
    fig_w = max(4.0, 3.5 * n_red)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_red,
        figsize=(fig_w, 4.0),
        sharey=True,
        squeeze=False,
    )
    axes_flat = [axes[0, j] for j in range(n_red)]

    legend_added = False
    for ax, ri in zip(axes_flat, red_iters):
        cov_vals, cov_lo, cov_hi = [], [], []
        yld_vals, yld_lo, yld_hi = [], [], []
        valid_blue = []

        for bi in blue_iters:
            key = f"red_{ri}_blue_{bi}"
            pairing = pairings.get(key)
            if pairing is None:
                continue
            m = pairing.get("metrics", {})
            ci = pairing.get("confidence_intervals", {})

            cov = m.get("coverage_pct")
            yld = m.get("yield_pct")
            if yld is None:
                continue

            valid_blue.append(bi)
            yld_vals.append(yld)
            yld_ci = ci.get("yield_pct") or [yld, yld]
            yld_lo.append(yld - yld_ci[0])
            yld_hi.append(yld_ci[1] - yld)

            if has_coverage and cov is not None:
                cov_vals.append(cov)
                cov_ci = ci.get("coverage_pct") or [cov, cov]
                cov_lo.append(cov - cov_ci[0])
                cov_hi.append(cov_ci[1] - cov)
            else:
                cov_vals.append(np.nan)
                cov_lo.append(0.0)
                cov_hi.append(0.0)

        x = np.array(valid_blue, dtype=float)

        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#888888")
            ax.set_title(f"Red {ri}", fontsize=11)
            continue

        # Successful-breach line (executor-level: accessed=True)
        kwargs_base = dict(marker="o", markersize=5, linewidth=1.5, capsize=3)
        yld_arr = np.array(yld_vals, dtype=float)
        ax.errorbar(
            x, yld_arr,
            yerr=([np.array(yld_lo), np.array(yld_hi)] if show_ci else None),
            color=RED_COL, linestyle="--",
            label="Successful breach (accessed)" if not legend_added else "_nolegend_",
            **kwargs_base,
        )

        # Attempted-breach line (parser-level: SQL referenced honeypot)
        cov_arr = np.array(cov_vals, dtype=float)
        if has_coverage and not np.all(np.isnan(cov_arr)):
            ax.errorbar(
                x, cov_arr,
                yerr=([np.array(cov_lo), np.array(cov_hi)] if show_ci else None),
                color=GRAY_COL, linestyle="-",
                label="Attempted breach (referenced)" if not legend_added else "_nolegend_",
                **kwargs_base,
            )
        elif not has_coverage:
            ax.text(
                0.5, 0.97,
                "Attempted-breach N/A\n(MARFT not imported)",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7, color="#888888",
                style="italic",
            )

        ax.set_xticks(valid_blue)
        ax.set_xlabel("Blue iteration", fontsize=10)
        ax.set_title(f"Red {ri}", fontsize=11)
        ax.set_ylim(0, 105)

        if not legend_added:
            ax.legend(fontsize=9, loc="upper right")
            legend_added = True

    axes_flat[0].set_ylabel("Fraction of honeypot universe (%)", fontsize=10)
    fig.suptitle(
        "Attempted vs. successful breach per pairing\n"
        "(attempted stays high → red keeps aiming at honeypots; successful drops → blue blocks at execution)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot attempted vs. successful breach from cross-eval pairings."
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument(
        "--subdir",
        default="cross_eval",
        help="Cross-eval subdir to plot (default: cross_eval). "
             "Known values with built-in output filenames: cross_eval, "
             "cross_eval_quick, diagonal_eval. Any other subdir (e.g. "
             "cross_eval_long) is accepted and writes to "
             "figures/<subdir>_attempted_vs_successful_breach.png unless --out is set.",
    )
    parser.add_argument("--out", default=None, help="Output path.")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    subdir = args.subdir
    default_fname = {
        "cross_eval": "cross_eval_attempted_vs_successful_breach.png",
        "cross_eval_quick": "quick_attempted_vs_successful_breach.png",
        "diagonal_eval": "diagonal_eval_attempted_vs_successful_breach.png",
    }.get(subdir, f"{subdir}_attempted_vs_successful_breach.png")
    out = args.out or f"figures/{default_fname}"
    saved = plot_coverage_yield(results, out, subdir=subdir)
    print(f"[{DESCRIPTION[:80]}...]\n  → {saved}")


if __name__ == "__main__":
    main()
