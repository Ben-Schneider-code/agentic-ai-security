"""
Per-benign-style refusal rate across self-play iterations.

Shows that the dedicated benign_eval corpus (simple queries) hides a
utility problem: when the blue team is shown *adversarially-framed* legitimate
queries during training, it refuses ~25–35 % of them, vs ~10 % for plain
queries — even though benign_eval reports TPR≈100 %. This is evidence that the
utility signal in training is too easy, and blue trades utility on harder
benigns for security against attacks.

Source: <selfplay_dir>/per_style_pud/trend.csv (produced by
util/per_style_pud_trend.py). Wilson 99% CIs are derived from per-iteration
sample sizes in the same file.

CLI:
    python plotting/plot_per_style_refusal.py --results <dir>[:label]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style, parse_results_arg, write_sidecar,
        wilson_ci_pct,
        RED_COL, BLUE_COL, GREEN_COL, GRAY_COL, FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, parse_results_arg, write_sidecar,
        wilson_ci_pct,
        RED_COL, BLUE_COL, GREEN_COL, GRAY_COL, FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = (
    "Per-style benign denial rate across self-play iterations. The dedicated "
    "benign_eval corpus is trivial (plain-style queries) and reports TPR≈100% "
    "for every blue checkpoint. But the training-time per-style breakdown "
    "shows ~25–35% denial on adversarially-framed legitimate requests — a "
    "utility failure mode that the headline BRR plot masks. Source: "
    "<selfplay_dir>/per_style_pud/trend.csv. 99% Wilson CIs."
)

_STYLE_META = [
    ("plain",        BLUE_COL,  "o",  "Plain benign"),
    ("multi_turn",   GREEN_COL, "s",  "Multi-turn benign"),
    ("adversarial",  RED_COL,   "^",  "Adversarial-framed benign"),
]



def plot_per_style_refusal(
    results: list[tuple[str, str]],
    out_path: str | Path,
    show_ci: bool = True,
) -> Path:
    out_path = Path(out_path)
    if len(results) > 1:
        print("[plot_per_style_refusal] Multiple runs given; using first only.",
              file=sys.stderr)
    label, selfplay_dir = results[0]
    trend_path = Path(selfplay_dir) / "per_style_pud" / "trend.csv"
    if not trend_path.is_file():
        print(f"[plot_per_style_refusal] Missing {trend_path}", file=sys.stderr)
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        ax.text(0.5, 0.5, "per_style_pud/trend.csv not found",
                ha="center", va="center", transform=ax.transAxes, color=GRAY_COL)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    rows: list[dict] = []
    with open(trend_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r["iter"]))
    iters = [int(r["iter"]) for r in rows]

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    for style_key, color, marker, display in _STYLE_META:
        n_col = f"{style_key}_n"
        p_col = f"{style_key}_pud"
        rates: list[float] = []
        ci_lo: list[float] = []
        ci_hi: list[float] = []
        present_iters: list[int] = []
        for r in rows:
            try:
                n = int(r[n_col])
                p = float(r[p_col])
            except (ValueError, KeyError):
                continue
            if math.isnan(p) or n == 0:
                continue
            k = int(round(p * n))
            lo, hi = wilson_ci_pct(k, n)
            rates.append(p * 100.0)
            ci_lo.append(lo)
            ci_hi.append(hi)
            present_iters.append(int(r["iter"]))
        if not present_iters:
            continue
        yerr = np.array([[v - l for v, l in zip(rates, ci_lo)],
                         [h - v for v, h in zip(rates, ci_hi)]]) if show_ci else None
        ax.errorbar(present_iters, rates, yerr=yerr,
                    fmt=f"-{marker}", color=color, label=display,
                    linewidth=1.8, markersize=7, capsize=4, capthick=1.0,
                    elinewidth=1.0)

    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Benign denial rate (%)")
    ax.set_xticks(iters)
    ax.set_ylim(0, max(50, 5))
    ax.set_title(
        "Per-style benign denial during training\n"
        "(benign_eval held-out reports ≈99% TPR; training-time rollouts disagree)"
    )
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_per_style_refusal] source={trend_path}", file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    ap.add_argument("--out", default="figures/per_style_refusal.png")
    args = ap.parse_args()
    results = parse_results_arg(args.results)
    out = plot_per_style_refusal(results, args.out)
    write_sidecar(out, DESCRIPTION, results)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
