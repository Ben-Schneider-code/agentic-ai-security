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


def _diagonal_metrics(pairings: dict, n_iters: int = 8) -> tuple[list[int], list[float], list[tuple[float, float]], list[float], list[tuple[float, float]]]:
    iters, asrs, asr_cis, tnrs, tnr_cis = [], [], [], [], []
    for i in range(n_iters):
        k = f"red_{i}_blue_{i}"
        if k not in pairings:
            continue
        m = pairings[k]["metrics"]
        ci = pairings[k]["confidence_intervals"]
        iters.append(i)
        asrs.append(m["asr"])
        asr_cis.append(tuple(ci["asr"]))
        tnrs.append(m["tnr"])
        tnr_cis.append(tuple(ci["tnr"]))
    return iters, asrs, asr_cis, tnrs, tnr_cis


def _baseline_metrics(pairings: dict, n_iters: int = 8) -> tuple[list[int], list[float], list[tuple[float, float]], list[float], list[tuple[float, float]]]:
    iters, asrs, asr_cis, tnrs, tnr_cis = [], [], [], [], []
    for i in range(n_iters):
        k = f"red_{i}_blue_0"
        if k not in pairings:
            continue
        m = pairings[k]["metrics"]
        ci = pairings[k]["confidence_intervals"]
        iters.append(i)
        asrs.append(m["asr"])
        asr_cis.append(tuple(ci["asr"]))
        tnrs.append(m["tnr"])
        tnr_cis.append(tuple(ci["tnr"]))
    return iters, asrs, asr_cis, tnrs, tnr_cis


def _yerr_from_cis(values: list[float], cis: list[tuple[float, float]]) -> np.ndarray:
    lo = np.array([v - c[0] for v, c in zip(values, cis)])
    hi = np.array([c[1] - v for v, c in zip(values, cis)])
    return np.vstack([lo, hi])


def plot_baseline_vs_trained(
    results: list[tuple[str, str]],
    out_dir: str = "figures/",
    cross_eval_subdir: str = "cross_eval",
    baseline_subdir: str = "cross_eval_baseline",
) -> Path:
    out_path = Path(out_dir) / "baseline_vs_trained_defense.png"
    sidecar_path = Path(out_dir) / "baseline_vs_trained_defense.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label, selfplay_dir = results[0]

    trained_pairings = _load_pairings(selfplay_dir, cross_eval_subdir)
    baseline_pairings = _load_pairings(selfplay_dir, baseline_subdir)

    if not trained_pairings or not baseline_pairings:
        print(
            f"[baseline_vs_trained] missing data: trained={bool(trained_pairings)} "
            f"baseline={bool(baseline_pairings)}",
            file=sys.stderr,
        )
        return out_path

    t_iters, t_asr, t_asr_ci, t_tnr, t_tnr_ci = _diagonal_metrics(trained_pairings)
    b_iters, b_asr, b_asr_ci, b_tnr, b_tnr_ci = _baseline_metrics(baseline_pairings)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)
    width = 0.38

    # Left: ASR comparison
    x_t = np.array(t_iters, dtype=float) - width / 2
    x_b = np.array(b_iters, dtype=float) + width / 2
    ax_l.bar(
        x_t,
        t_asr,
        width=width,
        color="#1976D2",
        label=f"Trained (co-evolved diagonal, n=400/cell)",
        yerr=_yerr_from_cis(t_asr, t_asr_ci),
        ecolor="black",
        capsize=2,
    )
    ax_l.bar(
        x_b,
        b_asr,
        width=width,
        color="#D32F2F",
        label=f"Baseline (manual prompt, n=200/cell)",
        yerr=_yerr_from_cis(b_asr, b_asr_ci),
        ecolor="black",
        capsize=2,
    )
    t_mean = float(np.mean(t_asr))
    b_mean = float(np.mean(b_asr))
    ax_l.axhline(t_mean, color="#1976D2", linestyle="--", linewidth=1, alpha=0.4)
    ax_l.axhline(b_mean, color="#D32F2F", linestyle="--", linewidth=1, alpha=0.4)
    ax_l.set_xlabel("Red training iteration")
    ax_l.set_ylabel("Attack Success Rate (%)")
    ax_l.set_title(f"ASR: trained {t_mean:.1f}% vs baseline {b_mean:.1f}% ({b_mean / t_mean:.1f}× gap)")
    ax_l.set_xticks(sorted(set(t_iters) | set(b_iters)))
    ax_l.set_ylim(0, max(b_asr) * 1.15)
    ax_l.legend(fontsize=8, loc="lower right")

    # Right: TNR comparison
    ax_r.bar(
        x_t,
        t_tnr,
        width=width,
        color="#1976D2",
        label=f"Trained",
        yerr=_yerr_from_cis(t_tnr, t_tnr_ci),
        ecolor="black",
        capsize=2,
    )
    ax_r.bar(
        x_b,
        b_tnr,
        width=width,
        color="#D32F2F",
        label=f"Baseline",
        yerr=_yerr_from_cis(b_tnr, b_tnr_ci),
        ecolor="black",
        capsize=2,
    )
    t_tnr_mean = float(np.mean(t_tnr))
    b_tnr_mean = float(np.mean(b_tnr))
    ax_r.axhline(t_tnr_mean, color="#1976D2", linestyle="--", linewidth=1, alpha=0.4)
    ax_r.axhline(b_tnr_mean, color="#D32F2F", linestyle="--", linewidth=1, alpha=0.4)
    ax_r.set_xlabel("Red training iteration")
    ax_r.set_ylabel("True Negative Rate (%)")
    ratio = (t_tnr_mean / b_tnr_mean) if b_tnr_mean > 0 else float("inf")
    ax_r.set_title(f"TNR: trained {t_tnr_mean:.1f}% vs baseline {b_tnr_mean:.1f}% ({ratio:.1f}× gap)")
    ax_r.set_xticks(sorted(set(t_iters) | set(b_iters)))
    ax_r.legend(fontsize=8, loc="upper right")

    t_asr_mean = float(np.mean(t_asr))
    b_asr_mean = float(np.mean(b_asr))
    asr_ratio = (b_asr_mean / t_asr_mean) if t_asr_mean > 0 else float("inf")
    tnr_ratio_for_title = (t_tnr_mean / b_tnr_mean) if b_tnr_mean > 0 else float("inf")
    fig.suptitle(
        f"Self-play training reduces ASR {asr_ratio:.1f}× and lifts TNR {tnr_ratio_for_title:.1f}× over manual-prompt baseline",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sidecar = {
        "description": DESCRIPTION,
        "trained": {
            "subdir": cross_eval_subdir,
            "iters": t_iters,
            "asr_pct": t_asr,
            "asr_ci": t_asr_ci,
            "tnr_pct": t_tnr,
            "tnr_ci": t_tnr_ci,
            "mean_asr_pct": t_mean,
            "mean_tnr_pct": t_tnr_mean,
        },
        "baseline": {
            "subdir": baseline_subdir,
            "iters": b_iters,
            "asr_pct": b_asr,
            "asr_ci": b_asr_ci,
            "tnr_pct": b_tnr,
            "tnr_ci": b_tnr_ci,
            "mean_asr_pct": b_mean,
            "mean_tnr_pct": b_tnr_mean,
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
