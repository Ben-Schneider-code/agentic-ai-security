"""
cross_eval_statistics.py — SQL-error rate heatmap across cross-evaluation pairings.

For each pairing directory that contains a summary.json, reads reward_debug.jsonl
and computes the fraction of turns where the model produced a response that was
neither a refusal nor valid parseable SQL (outcome_tier == 'sql_error').

The result is rendered as a heatmap over (red_iter × blue_iter) pairings,
following the same axis convention as plot_cross_eval_heatmap.py.

Usage:
    python plotting/cross_eval_statistics.py --results results-<ID>
    python plotting/cross_eval_statistics.py \\
        --results results-A:Label-A results-B:Label-B \\
        --cross-eval-subdir cross_eval \\
        --out figures/sql_error_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from plotting._data import apply_paper_style, parse_results_arg, write_sidecar
    from util.metrics import wilson_ci
except ImportError:
    from plotting._data import apply_paper_style, parse_results_arg, write_sidecar
    from util.metrics import wilson_ci

apply_paper_style()

DESCRIPTION = (
    "SQL-error rate heatmap: per (red_iter × blue_iter) pairing, the fraction of "
    "turns where the model produced output that was neither a refusal nor valid "
    "parseable SQL (outcome_tier == 'sql_error').  High rates indicate the model "
    "is struggling to emit well-formed SQL under adversarial or benign pressure."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _iter_pair_from_dirname(name: str) -> tuple[int, int] | None:
    """Parse 'red_N_blue_M' → (N, M), or return None if no match."""
    m = re.fullmatch(r"red_(\d+)_blue_(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _count_sql_errors_in_jsonl(jsonl_path: Path) -> tuple[int, int]:
    """
    Stream reward_debug.jsonl and count (n_sql_errors, n_total) records.

    Uses a streaming read to avoid loading multi-MB files into memory.
    Records with malformed JSON are skipped.
    """
    n_errors = n_total = 0
    with open(jsonl_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if rec.get("outcome_tier") == "sql_error":
                n_errors += 1
    return n_errors, n_total


def load_sql_error_matrix(
    selfplay_dir: str | Path,
    cross_eval_subdir: str = "cross_eval",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Build (n_red × n_blue) matrices of sql_error rate and Wilson CIs.

    Only pairings that have *both* a summary.json and a reward_debug.jsonl
    are included; missing cells are filled with NaN.

    Returns:
        rate_mat:  shape (n_red, n_blue), sql_error % in [0, 100]
        ci_lo_mat: lower 99 % Wilson CI
        ci_hi_mat: upper 99 % Wilson CI
        red_iters: sorted list of red iteration indices present
        blue_iters: sorted list of blue iteration indices present
    """
    pairings_dir = Path(selfplay_dir) / cross_eval_subdir / "pairings"
    if not pairings_dir.exists():
        raise FileNotFoundError(f"Pairings directory not found: {pairings_dir}")

    counts: dict[tuple[int, int], tuple[int, int]] = {}  # (r, b) → (n_err, n_total)

    for pairing_dir in sorted(pairings_dir.iterdir()):
        if not pairing_dir.is_dir():
            continue
        pair = _iter_pair_from_dirname(pairing_dir.name)
        if pair is None:
            continue
        if not (pairing_dir / "summary.json").exists():
            continue
        jsonl = pairing_dir / "reward_debug.jsonl"
        if not jsonl.exists():
            continue
        counts[pair] = _count_sql_errors_in_jsonl(jsonl)

    if not counts:
        raise ValueError(
            f"No valid pairings found under {pairings_dir}. "
            "Each pairing needs summary.json + reward_debug.jsonl."
        )

    red_iters = sorted({r for r, _ in counts})
    blue_iters = sorted({b for _, b in counts})
    nr, nb = len(red_iters), len(blue_iters)

    rate_mat = np.full((nr, nb), np.nan)
    ci_lo_mat = np.full((nr, nb), np.nan)
    ci_hi_mat = np.full((nr, nb), np.nan)

    r_idx = {r: i for i, r in enumerate(red_iters)}
    b_idx = {b: j for j, b in enumerate(blue_iters)}

    for (r, b), (n_err, n_total) in counts.items():
        i, j = r_idx[r], b_idx[b]
        if n_total == 0:
            continue
        rate_mat[i, j] = n_err / n_total * 100.0
        lo, hi = wilson_ci(n_err, n_total)
        ci_lo_mat[i, j] = lo
        ci_hi_mat[i, j] = hi

    return rate_mat, ci_lo_mat, ci_hi_mat, red_iters, blue_iters


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(
    fig: plt.Figure,
    ax: plt.Axes,
    rate_mat: np.ndarray,
    ci_lo_mat: np.ndarray,
    ci_hi_mat: np.ndarray,
    red_iters: list[int],
    blue_iters: list[int],
    *,
    title: str,
) -> None:
    """Render a sql_error-rate heatmap onto an existing axes."""
    from matplotlib.transforms import blended_transform_factory

    row_means = np.nanmean(rate_mat, axis=1)
    col_means = np.nanmean(rate_mat, axis=0)
    grand_mean = float(np.nanmean(rate_mat)) if not np.all(np.isnan(rate_mat)) else 0.0

    vmin = float(np.nanmin(rate_mat)) if not np.all(np.isnan(rate_mat)) else 0.0
    vmax = float(np.nanmax(rate_mat)) if not np.all(np.isnan(rate_mat)) else 100.0
    im_vmin = max(0.0, vmin - 2.0)
    im_vmax = min(100.0, vmax + 2.0)

    im = ax.imshow(
        rate_mat,
        aspect="auto",
        cmap="Blues",
        vmin=im_vmin,
        vmax=im_vmax,
        origin="upper",
    )

    mid_val = (vmin + vmax) / 2
    _wide_ci_seen = False

    for i in range(len(red_iters)):
        for j in range(len(blue_iters)):
            val = rate_mat[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#aaaaaa")
                continue
            text_col = "white" if val > mid_val + 5 else "black"
            ax.text(
                j, i - 0.18, f"{val:.1f}",
                ha="center", va="center", fontsize=9,
                color=text_col, fontweight="bold",
            )
            lo, hi = ci_lo_mat[i, j], ci_hi_mat[i, j]
            if not (np.isnan(lo) or np.isnan(hi)):
                ci_col = "#aaaaaa" if val > mid_val + 5 else "#777777"
                ax.text(
                    j, i + 0.25, f"[{lo:.0f}–{hi:.0f}]",
                    ha="center", va="center", fontsize=7, color=ci_col,
                )
                if hi < grand_mean or lo > grand_mean:
                    ax.text(
                        j + 0.35, i - 0.35, "*",
                        ha="center", va="center", fontsize=9, color="#333333",
                    )
                if (hi - lo) > 20.0:
                    ax.add_patch(matplotlib.patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="///", linewidth=0,
                        edgecolor="#555555", alpha=0.55, zorder=4,
                    ))
                    _wide_ci_seen = True

    # Black border on diagonal (co-evolved) cells
    for k in range(min(len(red_iters), len(blue_iters))):
        ax.add_patch(matplotlib.patches.Rectangle(
            (k - 0.5, k - 0.5), 1, 1,
            linewidth=2.0, edgecolor="#111111", fill=False, zorder=3,
        ))

    # Row marginals
    trans_r = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.02, -0.55, "mean", transform=trans_r,
        ha="left", va="center", fontsize=7, color="#888888", style="italic", clip_on=False,
    )
    for i, rmean in enumerate(row_means):
        if not np.isnan(rmean):
            ax.text(
                1.02, i, f"{rmean:.0f}%", transform=trans_r,
                ha="left", va="center", fontsize=8, color="#444444", clip_on=False,
            )

    # Column marginals
    trans_b = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        -0.55, -0.06, "mean", transform=trans_b,
        ha="center", va="top", fontsize=7, color="#888888", style="italic", clip_on=False,
    )
    for j, cmean in enumerate(col_means):
        if not np.isnan(cmean):
            ax.text(
                j, -0.06, f"{cmean:.0f}%", transform=trans_b,
                ha="center", va="top", fontsize=8, color="#444444", clip_on=False,
            )

    ax.set_xticks(range(len(blue_iters)))
    ax.set_yticks(range(len(red_iters)))
    ax.set_xticklabels([f"Blue {b}" for b in blue_iters], fontsize=10)
    ax.set_yticklabels([f"Red {r}" for r in red_iters], fontsize=10)
    ax.set_xlabel("Blue team iteration  (→ increasing blue compute)")
    ax.set_ylabel("Red team iteration  (↓ increasing red compute)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("SQL-error rate (%)", fontsize=10)

    legend_lines = ["■ diagonal =\nco-evolved"]
    if _wide_ci_seen:
        legend_lines.append("/// CI\nwidth >20 pp")
    ax.text(
        1.18, 0.5, "\n\n".join(legend_lines),
        transform=ax.transAxes, fontsize=8,
        va="center", ha="center", color="#444444",
    )


# ---------------------------------------------------------------------------
# Public plot function
# ---------------------------------------------------------------------------


def plot_sql_error_heatmap(
    results: list[tuple[str, str]],
    out_path: str | Path,
    cross_eval_subdir: str = "cross_eval",
) -> Path:
    """
    Plot sql_error rate heatmap for each selfplay run in *results*.

    One panel per run; panels share the same layout as plot_cross_eval_heatmap.py.

    Args:
        results:            [(label, selfplay_dir), ...]
        out_path:           Destination PNG path.
        cross_eval_subdir:  Subdirectory within selfplay_dir that holds pairings/.

    Returns:
        The resolved output path.
    """
    out_path = Path(out_path)
    n_panels = len(results)
    fig_w = max(6.0, 6.5 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 6.0), squeeze=False)

    for col, (label, selfplay_dir) in enumerate(results):
        ax = axes[0, col]
        try:
            rate_mat, ci_lo_mat, ci_hi_mat, red_iters, blue_iters = load_sql_error_matrix(
                selfplay_dir, cross_eval_subdir
            )
        except (FileNotFoundError, ValueError) as exc:
            ax.text(
                0.5, 0.5, f"No data\n{exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
            )
            ax.set_title(label)
            continue

        _render(
            fig, ax,
            rate_mat, ci_lo_mat, ci_hi_mat,
            red_iters, blue_iters,
            title=label,
        )

    fig.suptitle(
        "SQL-error rate: turns with neither refusal nor valid SQL output",
        y=1.01, fontsize=12,
    )
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a SQL-error rate heatmap from cross-evaluation pairings. "
            "Reads reward_debug.jsonl from each pairing that has a summary.json."
        )
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
        help="One or more selfplay result directories (optionally labelled with :Label).",
    )
    parser.add_argument(
        "--out", default="figures/sql_error_heatmap.png", metavar="PATH",
        help="Output PNG path (default: figures/sql_error_heatmap.png).",
    )
    parser.add_argument(
        "--cross-eval-subdir", default="cross_eval", metavar="SUBDIR",
        help="Subdirectory within selfplay_dir that holds pairings/ (default: cross_eval).",
    )

    args = parser.parse_args()
    results = parse_results_arg(args.results)
    print("Results:")
    for label, sd in results:
        print(f"  {label!r:30s}  ←  {sd}")

    path = plot_sql_error_heatmap(results, args.out, args.cross_eval_subdir)
    write_sidecar(path, DESCRIPTION, results)
    print(f"\n[{DESCRIPTION[:80]}…]\n  → {path}")


if __name__ == "__main__":
    main()
