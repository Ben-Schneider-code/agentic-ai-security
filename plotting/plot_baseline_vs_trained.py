"""
Baseline-vs-trained defender comparison.

Compares trained-defender (canonical co-evolved diagonal) against
manually-prompted baseline (no LoRA, just system prompt) cross-eval results.

Resolves the existential question: "does self-play training do anything?"
Mean ASR drops from ~58% (manual baseline) to ~16% (trained diagonal); mean
TNR rises from ~1.7% to ~25%. Training the defender works.

Sources:
  Trained:  results-<ID>/cross_eval/cross_eval_results.json (diagonal cells)
  Baseline: results-<ID>/cross_eval_baseline/cross_eval_results.json (red_*_blue_0)

CLI:
    python plotting/plot_baseline_vs_trained.py \\
        --results results-<ID>[:Label] [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

DESCRIPTION = "Baseline (manual-prompt) vs trained (co-evolved) defender: ASR & TNR per red iter"


def _load_pairings(selfplay_dir: str, subdir: str) -> dict:
    """Load pairings via the canonical loader so the in-memory metric refresh
    (PVR_conv denominator fix) is applied consistently."""
    data = load_cross_eval_results(selfplay_dir, subdir=subdir)
    if data is None:
        return {}
    return data.get("pairings", {})


def _iter_metrics(pairings: dict, diagonal: bool, n_iters: int = 8) -> dict[int, tuple[float, float]]:
    """Return {red_iter: (asr_pct, tnr_pct)} for diagonal (red_i_blue_i) or
    baseline (red_i_blue_0) pairings."""
    out: dict[int, tuple[float, float]] = {}
    for i in range(n_iters):
        k = f"red_{i}_blue_{i}" if diagonal else f"red_{i}_blue_0"
        if k not in pairings:
            continue
        m = pairings[k]["metrics"]
        out[i] = (m["asr"], m["tnr"])
    return out


def _aggregate(
    per_rep: list[dict[int, tuple[float, float]]], idx: int, iters: list[int]
) -> tuple[list[float], np.ndarray, list[list[float]]]:
    """For each iter in `iters`, collect metric `idx` (0=ASR, 1=TNR) across
    replicates. Returns (means, yerr[2×N] min/max spread, per_rep_points)."""
    means, lo_err, hi_err, points = [], [], [], []
    for it in iters:
        vals = [r[it][idx] for r in per_rep if it in r]
        if not vals:
            means.append(np.nan)
            lo_err.append(0.0)
            hi_err.append(0.0)
            points.append([])
            continue
        mu = float(np.mean(vals))
        means.append(mu)
        lo_err.append(mu - min(vals))
        hi_err.append(max(vals) - mu)
        points.append(vals)
    return means, np.vstack([lo_err, hi_err]), points


def _scatter_reps(ax, x_positions: np.ndarray, points: list[list[float]], width: float) -> None:
    """Overlay individual replicate values as small dots on each bar."""
    for x_pos, vals in zip(x_positions, points):
        if not vals:
            continue
        jitter = (
            np.linspace(-width * 0.2, width * 0.2, len(vals))
            if len(vals) > 1 else np.zeros(1)
        )
        ax.scatter(
            x_pos + jitter, vals,
            s=16, color="black", zorder=5,
            edgecolors="white", linewidths=0.5,
        )


def plot_baseline_vs_trained(
    results: list[tuple[str, str]],
    out_dir: str = "figures/",
    cross_eval_subdir: str = "cross_eval",
    baseline_subdir: str = "cross_eval_baseline",
    show_ci: bool = True,
) -> Path:
    out_path = Path(out_dir) / "baseline_vs_trained_defense.png"
    sidecar_path = Path(out_dir) / "baseline_vs_trained_defense.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect per-replicate diagonal (trained) and baseline metrics.
    reps_trained: list[dict[int, tuple[float, float]]] = []
    reps_baseline: list[dict[int, tuple[float, float]]] = []
    rep_labels: list[str] = []
    for label, selfplay_dir in results:
        trained_pairings = _load_pairings(selfplay_dir, cross_eval_subdir)
        baseline_pairings = _load_pairings(selfplay_dir, baseline_subdir)
        if not trained_pairings or not baseline_pairings:
            print(
                f"[baseline_vs_trained] {label}: missing data — "
                f"trained={bool(trained_pairings)} baseline={bool(baseline_pairings)}",
                file=sys.stderr,
            )
            continue
        reps_trained.append(_iter_metrics(trained_pairings, diagonal=True))
        reps_baseline.append(_iter_metrics(baseline_pairings, diagonal=False))
        rep_labels.append(label)

    if not reps_trained:
        print(
            "[baseline_vs_trained] no replicate has both trained and baseline data",
            file=sys.stderr,
        )
        return out_path

    iters = sorted({it for r in reps_trained + reps_baseline for it in r})
    n_rep = len(rep_labels)

    t_asr, t_asr_err, t_asr_pts = _aggregate(reps_trained, 0, iters)
    b_asr, b_asr_err, b_asr_pts = _aggregate(reps_baseline, 0, iters)
    t_tnr, t_tnr_err, t_tnr_pts = _aggregate(reps_trained, 1, iters)
    b_tnr, b_tnr_err, b_tnr_pts = _aggregate(reps_baseline, 1, iters)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)
    width = 0.38

    x = np.array(iters, dtype=float)
    x_t = x - width / 2
    x_b = x + width / 2

    rep_note = f"mean of {n_rep} replicate{'s' if n_rep != 1 else ''}"

    # Left: ASR comparison
    ax_l.bar(
        x_t, t_asr, width=width, color="#1976D2",
        label=f"Trained (co-evolved diagonal, {rep_note})",
        yerr=(t_asr_err if show_ci else None), ecolor="black", capsize=2,
    )
    ax_l.bar(
        x_b, b_asr, width=width, color="#D32F2F",
        label=f"Baseline (manual prompt, {rep_note})",
        yerr=(b_asr_err if show_ci else None), ecolor="black", capsize=2,
    )
    _scatter_reps(ax_l, x_t, t_asr_pts, width)
    _scatter_reps(ax_l, x_b, b_asr_pts, width)
    t_mean = float(np.nanmean(t_asr))
    b_mean = float(np.nanmean(b_asr))
    ax_l.axhline(t_mean, color="#1976D2", linestyle="--", linewidth=1, alpha=0.4)
    ax_l.axhline(b_mean, color="#D32F2F", linestyle="--", linewidth=1, alpha=0.4)
    ax_l.set_xlabel("Red training iteration")
    ax_l.set_ylabel("Attack Success Rate (%)")
    ax_l.set_title(f"ASR: trained {t_mean:.1f}% vs baseline {b_mean:.1f}% ({b_mean / t_mean:.1f}× gap)")
    ax_l.set_xticks(iters)
    ax_l.set_ylim(0, float(np.nanmax(np.array(b_asr) + b_asr_err[1])) * 1.15)
    ax_l.legend(fontsize=8, loc="lower right")

    # Right: TNR comparison
    ax_r.bar(
        x_t, t_tnr, width=width, color="#1976D2", label="Trained",
        yerr=(t_tnr_err if show_ci else None), ecolor="black", capsize=2,
    )
    ax_r.bar(
        x_b, b_tnr, width=width, color="#D32F2F", label="Baseline",
        yerr=(b_tnr_err if show_ci else None), ecolor="black", capsize=2,
    )
    _scatter_reps(ax_r, x_t, t_tnr_pts, width)
    _scatter_reps(ax_r, x_b, b_tnr_pts, width)
    t_tnr_mean = float(np.nanmean(t_tnr))
    b_tnr_mean = float(np.nanmean(b_tnr))
    ax_r.axhline(t_tnr_mean, color="#1976D2", linestyle="--", linewidth=1, alpha=0.4)
    ax_r.axhline(b_tnr_mean, color="#D32F2F", linestyle="--", linewidth=1, alpha=0.4)
    ax_r.set_xlabel("Red training iteration")
    ax_r.set_ylabel("True Negative Rate (%)")
    ratio = (t_tnr_mean / b_tnr_mean) if b_tnr_mean > 0 else float("inf")
    ax_r.set_title(f"TNR: trained {t_tnr_mean:.1f}% vs baseline {b_tnr_mean:.1f}% ({ratio:.1f}× gap)")
    ax_r.set_xticks(iters)
    ax_r.legend(fontsize=8, loc="upper right")

    asr_ratio = (b_mean / t_mean) if t_mean > 0 else float("inf")
    fig.suptitle(
        f"Self-play training reduces ASR {asr_ratio:.1f}× and lifts TNR {ratio:.1f}× "
        f"over manual-prompt baseline  ({n_rep} replicate{'s' if n_rep != 1 else ''}; "
        "error bars = min–max spread, dots = per-replicate values)",
        y=1.02,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    def _per_rep_block(per_rep, idx):
        return [
            {str(it): round(r[it][idx], 3) for it in sorted(r)}
            for r in per_rep
        ]

    sidecar = {
        "description": DESCRIPTION,
        "n_replicates": n_rep,
        "replicate_labels": rep_labels,
        "iters": iters,
        "trained": {
            "subdir": cross_eval_subdir,
            "asr_pct_per_replicate": _per_rep_block(reps_trained, 0),
            "tnr_pct_per_replicate": _per_rep_block(reps_trained, 1),
            "mean_asr_pct": [round(v, 3) if not np.isnan(v) else None for v in t_asr],
            "mean_tnr_pct": [round(v, 3) if not np.isnan(v) else None for v in t_tnr],
            "grand_mean_asr_pct": t_mean,
            "grand_mean_tnr_pct": t_tnr_mean,
        },
        "baseline": {
            "subdir": baseline_subdir,
            "asr_pct_per_replicate": _per_rep_block(reps_baseline, 0),
            "tnr_pct_per_replicate": _per_rep_block(reps_baseline, 1),
            "mean_asr_pct": [round(v, 3) if not np.isnan(v) else None for v in b_asr],
            "mean_tnr_pct": [round(v, 3) if not np.isnan(v) else None for v in b_tnr],
            "grand_mean_asr_pct": b_mean,
            "grand_mean_tnr_pct": b_tnr_mean,
        },
        "asr_gap_ratio": b_mean / t_mean if t_mean > 0 else None,
        "tnr_gap_ratio": ratio,
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[baseline_vs_trained] saved {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out-dir", default="figures/")
    p.add_argument("--cross-eval-subdir", default="cross_eval")
    p.add_argument("--baseline-subdir", default="cross_eval_baseline")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_baseline_vs_trained(results, args.out_dir, args.cross_eval_subdir, args.baseline_subdir)


if __name__ == "__main__":
    main()
