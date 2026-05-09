"""
Cross-eval rank-invariance analysis: diagonal vs off-diagonal PVR_conv.

Tests whether co-evolved checkpoint pairs (diagonal cells) show lower PVR_conv
than other pairings (off-diagonal cells). Under the combinatorial-ceiling thesis,
no such diagonal dominance should exist — PVR is a property of |Q^P_deny|, not
of the co-evolved pair.

Canonical 8×8 result: diagonal mean = 16.94%, off-diagonal mean = 16.24%,
z = 0.76, p = 0.45 → null hypothesis (no dominance) NOT rejected.
One anomalous cell r7×b4 = 7.4% at n=216 (all others n≈400) is flagged.

Produces:
  figures/cross_eval_rank_invariance.png  — distribution violin + diagonal highlight
  figures/cross_eval_rank_invariance.json — full statistics sidecar

CLI:
    python plotting/cross_eval_rank_invariance.py \\
        --results results-<ID>[:Label] [--cross-eval-subdir cross_eval] \\
        [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ._data import (
        apply_paper_style,
        load_cross_eval_results,
        parse_results_arg,
        FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_cross_eval_results,
        parse_results_arg,
        FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = "Rank-invariance test: diagonal vs off-diagonal PVR_conv distribution"
_OUTLIER_CELL = (7, 4)  # r7×b4, n=216 (truncated)


def _z_stat(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Two-proportion z-statistic (unpooled) and two-tailed p-value."""
    import scipy.stats as st  # type: ignore
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_val = 2 * st.norm.sf(abs(z))
    return z, p_val


def _z_stat_fallback(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-statistic without scipy."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 else 1e-9
    z = (p1 - p2) / se
    # approximate two-tailed p from normal
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_val


def compute_rank_invariance(
    selfplay_dir: str, subdir: str = "cross_eval", outlier: tuple = _OUTLIER_CELL
) -> dict:
    data = load_cross_eval_results(selfplay_dir, subdir)
    if data is None:
        return {}

    pairings = data.get("pairings", {})
    cells: dict[tuple[int, int], dict] = {}
    for k, v in pairings.items():
        parts = k.split("_")
        if len(parts) != 4:
            continue
        ri, bi = int(parts[1]), int(parts[3])
        # Use C*_R (n_eps_with_sql) as the per-cell sample size for the z-test —
        # this matches the ASR denominator (problem_statement.tex eq. 49) and
        # avoids inflating the test's effective sample with all-refusal episodes.
        n_eps_with_sql = (
            v.get("episode_stats", {}).get("n_eps_with_sql")
            or v.get("n_attack_episodes", 400)
        )
        cells[(ri, bi)] = {
            "asr": v["metrics"]["asr"],
            "n": n_eps_with_sql,
        }

    iters = sorted(set(i for (i, _) in cells))
    diag_cells = {(i, i): cells[(i, i)] for i in iters if (i, i) in cells}
    offdiag_cells = {
        k: v for k, v in cells.items()
        if k[0] != k[1] and k != outlier
    }
    outlier_cell = cells.get(outlier)

    d_asrs = [v["asr"] for v in diag_cells.values()]
    d_ns = [v["n"] for v in diag_cells.values()]
    o_asrs = [v["asr"] for v in offdiag_cells.values()]
    o_ns = [v["n"] for v in offdiag_cells.values()]

    d_mean = sum(d_asrs) / len(d_asrs) if d_asrs else 0
    o_mean = sum(o_asrs) / len(o_asrs) if o_asrs else 0
    d_n_total = sum(d_ns)
    o_n_total = sum(o_ns)

    try:
        z, p_val = _z_stat(d_mean / 100, d_n_total, o_mean / 100, o_n_total)
    except Exception:
        z, p_val = _z_stat_fallback(d_mean / 100, d_n_total, o_mean / 100, o_n_total)

    return {
        "n_iters": len(iters),
        "n_diag_cells": len(diag_cells),
        "n_offdiag_cells": len(offdiag_cells),
        "outlier_cell": f"r{outlier[0]}xb{outlier[1]}" if outlier_cell else None,
        "outlier_asr": outlier_cell["asr"] if outlier_cell else None,
        "outlier_n": outlier_cell["n"] if outlier_cell else None,
        "diag_mean_pct": round(d_mean, 3),
        "offdiag_mean_pct": round(o_mean, 3),
        "diag_values": sorted(d_asrs),
        "offdiag_values": sorted(o_asrs),
        "diag_n_total": d_n_total,
        "offdiag_n_total": o_n_total,
        "z_stat": round(z, 4),
        "p_value": round(p_val, 4),
        "significant_95": abs(z) > 1.96,
        "significant_99": abs(z) > 2.576,
        "verdict": "NO diagonal dominance (null not rejected)" if abs(z) <= 1.96 else "DIAGONAL DOMINANCE DETECTED",
        "cells": {f"r{k[0]}xb{k[1]}": {"asr": v["asr"], "n": v["n"], "on_diag": k[0] == k[1]}
                  for k, v in cells.items()},
    }


def plot_rank_invariance(
    results: list[tuple[str, str]],
    cross_eval_subdir: str = "cross_eval",
    out_dir: str = "figures/",
) -> Path:
    out_path = Path(out_dir) / "cross_eval_rank_invariance.png"
    sidecar_path = Path(out_dir) / "cross_eval_rank_invariance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label, selfplay_dir = results[0]
    stats = compute_rank_invariance(selfplay_dir, cross_eval_subdir)
    if not stats:
        print(f"[rank_invariance] No data.", file=sys.stderr)
        return out_path

    with open(sidecar_path, "w") as f:
        json.dump(stats, f, indent=2)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    # Left: violin / strip plot of diagonal vs off-diagonal
    diag_vals = stats["diag_values"]
    offdiag_vals = stats["offdiag_values"]

    positions = [1, 2]
    parts = ax_l.violinplot(
        [diag_vals, offdiag_vals],
        positions=positions,
        showmeans=True,
        showextrema=True,
    )
    for pc, color in zip(parts["bodies"], ["#2196F3", "#9E9E9E"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    # Scatter individual points
    import random
    random.seed(42)
    for i, vals in enumerate([diag_vals, offdiag_vals], 1):
        jitter = [i + random.uniform(-0.06, 0.06) for _ in vals]
        ax_l.scatter(jitter, vals, s=15, alpha=0.5, color="#2196F3" if i == 1 else "#9E9E9E", zorder=3)

    # Mark outlier separately if present
    if stats.get("outlier_asr") is not None:
        ax_l.scatter([2 + 0.1], [stats["outlier_asr"]], s=40, color="red", zorder=5,
                     marker="x", label=f"Outlier r7×b4 (n={stats['outlier_n']})")
        ax_l.legend(fontsize=6, loc="upper right")

    ax_l.set_xticks(positions)
    ax_l.set_xticklabels([
        f"Diagonal\n(n={stats['n_diag_cells']}, mean={stats['diag_mean_pct']:.1f}%)",
        f"Off-diagonal\n(excl. outlier, mean={stats['offdiag_mean_pct']:.1f}%)",
    ], fontsize=8)
    ax_l.set_ylabel("PVR_conv (%)")
    ax_l.set_title("Diagonal vs off-diagonal PVR_conv")
    z = stats["z_stat"]
    p = stats["p_value"]
    sig_str = "n.s." if not stats["significant_95"] else ("*" if not stats["significant_99"] else "**")
    ax_l.text(
        0.5, 0.02,
        f"z = {z:.2f}, p = {p:.2f} ({sig_str})\n{stats['verdict']}",
        transform=ax_l.transAxes,
        ha="center", va="bottom", fontsize=7, style="italic",
    )

    # Right: 8×8 heatmap with diagonal highlighted
    cells = stats["cells"]
    n_iters = stats["n_iters"]
    matrix = [[float("nan")] * n_iters for _ in range(n_iters)]
    for key, cv in cells.items():
        ri = int(key[1])  # r{ri}xb{bi}
        bi = int(key.split("xb")[1])
        if ri < n_iters and bi < n_iters:
            matrix[ri][bi] = cv["asr"]

    import numpy as np  # type: ignore
    mat = np.array(matrix)
    im = ax_r.imshow(mat, cmap="YlOrRd", aspect="auto",
                     vmin=max(0, stats["offdiag_mean_pct"] - 8),
                     vmax=stats["offdiag_mean_pct"] + 8)
    fig.colorbar(im, ax=ax_r, label="PVR_conv (%)")

    # Highlight diagonal with blue borders
    for i in range(n_iters):
        ax_r.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#2196F3", linewidth=2, zorder=2
        ))

    # Annotate cells with values
    for ri in range(n_iters):
        for bi in range(n_iters):
            v = matrix[ri][bi]
            if not math.isnan(v):
                is_outlier = (ri == _OUTLIER_CELL[0] and bi == _OUTLIER_CELL[1])
                ax_r.text(bi, ri, f"{v:.0f}", ha="center", va="center",
                          fontsize=5, color="black",
                          fontweight="bold" if is_outlier else "normal")

    ax_r.set_xticks(range(n_iters))
    ax_r.set_yticks(range(n_iters))
    ax_r.set_xticklabels([f"b{i}" for i in range(n_iters)], fontsize=7)
    ax_r.set_yticklabels([f"r{i}" for i in range(n_iters)], fontsize=7)
    ax_r.set_xlabel("Blue checkpoint")
    ax_r.set_ylabel("Red checkpoint")
    ax_r.set_title(f"8×8 PVR_conv matrix (diagonal highlighted)")

    fig.suptitle(f"Rank-invariance: PVR_conv plateau = {stats['diag_mean_pct']:.1f}% (diagonal) ≈ {stats['offdiag_mean_pct']:.1f}% (off-diagonal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rank_invariance] saved {out_path}  z={z:.2f} p={p:.2f}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--cross-eval-subdir", default="cross_eval")
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_rank_invariance(results, args.cross_eval_subdir, args.out_dir)


if __name__ == "__main__":
    main()
