"""
Attempts-to-compromise CDF — SQL turns before first honeypot violation.

For each (Red_i × Blue_j) pairing, plots the empirical CDF over hit-attack-episodes
of "how many SQL-emitting turns red needed before the first successful violation".
As Blue evolves, the CDF should shift right — red must try more SQL before breaking
through. Only hit-episodes contribute to the CDF; clean episodes are reported as
annotation text.

CLI usage:
    python plotting/plot_attempts_cdf.py --results results-<ID>
    python plotting/plot_attempts_cdf.py --results results-<ID> --subdir cross_eval_quick
Or imported:
    from plotting.plot_attempts_cdf import plot_attempts_cdf, ATTEMPTS_CDF_JOBS, DESCRIPTION
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
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_pairing_metrics_with_decomposed,
        parse_results_arg,
    )

apply_paper_style()

DESCRIPTION = (
    "CDF of attack-SQL turns emitted before the first successful honeypot violation, "
    "per (Red_i × Blue_j) pairing. CDF shifting right with Blue evolution means "
    "red must emit more SQL before breaking through — the core asymmetry plot. "
    "Only hit-episodes contribute; annotation shows n_hit/n_total per pairing."
)

ATTEMPTS_CDF_JOBS: list[tuple[str, str]] = [
    ("cross_eval",       "cross_eval_attempts_cdf.png"),
    ("cross_eval_quick", "quick_attempts_cdf.png"),
    ("diagonal_eval",    "diagonal_eval_attempts_cdf.png"),
]


def _hit_idx_summary(hit_indices: list[int]) -> dict:
    """Summarize a list of first-hit SQL indices (quantiles + histogram + mean).

    `_raw` carries the full sorted array so the renderer can reproduce the exact
    empirical step-CDF; JSON-facing keys are the compact summary.
    """
    arr = np.array(sorted(hit_indices), dtype=float)
    if arr.size == 0:
        return {
            "quantiles": {k: None for k in ("p10", "p25", "p50", "p75", "p90", "max")},
            "hist": {"bin_edges": [], "counts": []},
            "mean": None,
            "_raw": [],
        }
    qs = np.percentile(arr, [10, 25, 50, 75, 90])
    # Integer-aligned histogram over the observed range (1 bin per SQL-turn index).
    lo, hi = int(arr.min()), int(arr.max())
    bin_edges = np.arange(lo, hi + 2) - 0.5  # centers on integers lo..hi
    counts, edges = np.histogram(arr, bins=bin_edges)
    return {
        "quantiles": {
            "p10": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]),
            "p75": float(qs[3]), "p90": float(qs[4]), "max": float(arr.max()),
        },
        "hist": {
            "bin_edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        },
        "mean": float(arr.mean()),
        "_raw": [int(v) for v in arr],
    }


def compute_attempts_cdf(
    results: list[tuple[str, str]],
    subdir: str = "cross_eval",
    **kwargs,
) -> dict:
    """
    Pure compute of per-pairing attempts-to-compromise summaries (no matplotlib).

    Uses the first result entry (the plot is single-run). For each (red_i, blue_j)
    pairing returns, keyed "red_i_blue_j":

        {
          "red_i_blue_j": {
            "red_iter": int, "blue_iter": int,
            "n_hit": int, "n_total": int,
            "hit_idx": {
              "quantiles": {"p10","p25","p50","p75","p90","max"},
              "hist": {"bin_edges": [...], "counts": [...]},
              "mean": float | None,
              "_raw": [int, ...],   # internal: full array for exact CDF rendering
            },
          }, ...
        }

    Also carries top-level "_axes": {"red_iters": [...], "blue_iters": [...]} so
    the renderer can lay out subplots. Returns {} on no data.
    """
    if not results:
        return {}
    _label, selfplay_dir = results[0]
    cross_eval = load_pairing_metrics_with_decomposed(selfplay_dir, subdir=subdir)
    if cross_eval is None:
        return {}

    pairings = cross_eval.get("pairings", {})
    red_iters = sorted({v["red_iter"] for v in pairings.values()})
    blue_iters = sorted({v["blue_iter"] for v in pairings.values()})
    if not red_iters:
        return {}

    cells: dict = {}
    for ri in red_iters:
        for bi in blue_iters:
            key = f"red_{ri}_blue_{bi}"
            pairing = pairings.get(key)
            if pairing is None:
                continue
            ep_stats = pairing.get("episode_stats", {})
            n_total = pairing.get("n_attack_episodes", 0)
            indices = ep_stats.get("per_ep_first_hit_sql_idx", [])
            hit_indices = [x for x in indices if x is not None]
            cells[key] = {
                "red_iter": ri,
                "blue_iter": bi,
                "n_hit": len(hit_indices),
                "n_total": n_total,
                "hit_idx": _hit_idx_summary(hit_indices),
            }

    if not cells:
        return {}
    cells["_axes"] = {"red_iters": red_iters, "blue_iters": blue_iters}
    return cells


def plot_attempts_cdf(
    results: list[tuple[str, str]],
    out_path: str | Path,
    *,
    subdir: str = "cross_eval",
    precomputed: dict | None = None,
) -> Path:
    """
    CDF of SQL turns before first violation, per (Red_i × Blue_j) pairing.

    One subplot per Red iter (rows); within each subplot, one CDF line per Blue iter
    colored along viridis. Only the first result entry is used.

    precomputed: optional dict from compute_attempts_cdf(results, subdir=...);
        when None it is computed here. Fully drives rendering via the per-cell
        summaries' internal `_raw` arrays.
    """
    out_path = Path(out_path)
    if len(results) > 1:
        print(
            "[plot_attempts_cdf] Multiple runs given; using first run only.",
            file=sys.stderr,
        )

    if precomputed is None:
        precomputed = compute_attempts_cdf(results, subdir=subdir)

    _label, selfplay_dir = results[0]
    if not precomputed:
        # Distinguish "no subdir data" from "no pairings" to preserve messages.
        if load_pairing_metrics_with_decomposed(selfplay_dir, subdir=subdir) is None:
            print(
                f"[plot_attempts_cdf] No data for subdir={subdir!r} in {selfplay_dir}.",
                file=sys.stderr,
            )
        else:
            print("[plot_attempts_cdf] No pairings found.", file=sys.stderr)
        return out_path

    cells = precomputed
    axes_info = cells.get("_axes", {})
    red_iters = axes_info.get("red_iters", [])
    blue_iters = axes_info.get("blue_iters", [])

    if not red_iters:
        print("[plot_attempts_cdf] No pairings found.", file=sys.stderr)
        return out_path

    cmap = plt.get_cmap("viridis")
    n_blue = len(blue_iters)
    bi_to_color = {bi: cmap(j / max(n_blue - 1, 1)) for j, bi in enumerate(blue_iters)}

    fig, axes = plt.subplots(
        nrows=len(red_iters),
        ncols=1,
        figsize=(6.5, 2.6 * len(red_iters)),
        sharex=True,
        squeeze=False,
    )
    axes_flat = [axes[i, 0] for i in range(len(red_iters))]

    for ax, ri in zip(axes_flat, red_iters):
        any_data = False
        for bi in blue_iters:
            key = f"red_{ri}_blue_{bi}"
            cell = cells.get(key)
            if cell is None:
                continue

            n_total = cell["n_total"]
            hit_indices = cell["hit_idx"]["_raw"]
            n_hit = cell["n_hit"]

            color = bi_to_color[bi]
            if n_hit == 0:
                # Add invisible point so legend entry appears; annotate later.
                ax.plot([], [], color=color, label=f"Blue {bi} (0/{n_total} hit)")
                continue

            sorted_vals = sorted(hit_indices)
            n = len(sorted_vals)
            # Empirical CDF: x=value, y=fraction of hit episodes with ≤x SQL turns before hit
            x_vals = np.array(sorted_vals)
            y_vals = np.arange(1, n + 1) / n

            ax.step(
                np.append(x_vals, x_vals[-1] + 1),
                np.append(y_vals, y_vals[-1]),
                where="post",
                color=color,
                label=f"Blue {bi} ({n_hit}/{n_total} hit)",
                linewidth=1.5,
            )
            any_data = True

        ax.set_ylabel("Fraction of hit episodes", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)
        ax.set_title(f"Red iter {ri}", fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="lower right", ncol=min(n_blue, 3))

        if not any_data:
            ax.text(
                0.5, 0.5, "No hit episodes",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888",
            )

    axes_flat[-1].set_xlabel("SQL turns before first violation (0 = first SQL attempt succeeded)")
    fig.suptitle(
        "Attempts-to-compromise CDF\n"
        "(right shift → blue forces more SQL attempts per violation)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot attempts-to-compromise CDF from cross-eval pairings."
    )
    parser.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    parser.add_argument(
        "--subdir",
        default="cross_eval",
        help="Cross-eval subdir to plot (default: cross_eval). "
             "Known values with built-in output filenames: cross_eval, "
             "cross_eval_quick, diagonal_eval. Any other subdir (e.g. "
             "cross_eval_long) is accepted and writes to "
             "figures/<subdir>_attempts_cdf.png unless --out is set.",
    )
    parser.add_argument("--out", default=None, help="Output path.")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    subdir = args.subdir
    default_fname = {
        "cross_eval": "cross_eval_attempts_cdf.png",
        "cross_eval_quick": "quick_attempts_cdf.png",
        "diagonal_eval": "diagonal_eval_attempts_cdf.png",
    }.get(subdir, f"{subdir}_attempts_cdf.png")
    out = args.out or f"figures/{default_fname}"
    saved = plot_attempts_cdf(results, out, subdir=subdir)
    print(f"[{DESCRIPTION[:80]}...]\n  → {saved}")


if __name__ == "__main__":
    main()
