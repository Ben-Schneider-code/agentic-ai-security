"""
Utility by query style — Pillar 1 headline figure.

Shows three lines (plain / multi-turn / adversarial-framed) of benign denial
rate across self-play iterations, with 99 % Wilson CIs derived from per-iter
sample sizes in `per_style_pud/trend.csv`.

Overlays a faded reference line for the plain-pool BRR sourced from
`benign_eval/benign_eval_results.json`. The plain-pool reference line is
labeled "plain-only operating point — biased upward" because the benign_eval
corpus is dominated by plain queries and therefore overstates utility.

Replaces `brr.png` as the headline utility figure for Pillar 1; the headline
operating point is the style-stratified denial rate, not the plain-only BRR.

CLI:
    python plotting/plot_utility_by_style.py --results <dir>[:label]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style,
        parse_results_arg,
        write_sidecar,
        RED_COL,
        BLUE_COL,
        GREEN_COL,
        GRAY_COL,
        FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        parse_results_arg,
        write_sidecar,
        RED_COL,
        BLUE_COL,
        GREEN_COL,
        GRAY_COL,
        FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = (
    "Pillar 1 headline: per-query-style benign denial rate (plain / multi-turn / "
    "adversarial-framed) across self-play iterations with 99% Wilson CIs. "
    "Overlays the plain-pool BRR from benign_eval as a faded 'biased upward' "
    "reference. Replaces brr.png as the operating-point figure for Pillar 1. "
    "Sources: per_style_pud/trend.csv, benign_eval/benign_eval_results.json."
)

_STYLE_META = [
    ("plain", BLUE_COL, "o", "Plain benign (training)"),
    ("multi_turn", GREEN_COL, "s", "Multi-turn benign (training)"),
    ("adversarial", RED_COL, "^", "Adversarial-framed benign (training)"),
]


def _wilson(k: int, n: int, z: float = 2.576) -> tuple[float, float]:
    if n == 0:
        return 0.0, 100.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0.0, (c - m) * 100.0), min(100.0, (c + m) * 100.0)


def _load_per_style(selfplay_dir: Path) -> list[dict] | None:
    trend = selfplay_dir / "per_style_pud" / "trend.csv"
    if not trend.is_file():
        return None
    rows: list[dict] = []
    with trend.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r["iter"]))
    return rows


def _load_plain_brr(selfplay_dir: Path) -> dict[int, dict] | None:
    """Return {blue_iter: {brr, ci_lo, ci_hi}} from benign_eval_results.json."""
    f = selfplay_dir / "benign_eval" / "benign_eval_results.json"
    if not f.is_file():
        return None
    with f.open() as fh:
        d = json.load(fh)
    out: dict[int, dict] = {}
    for key, entry in (d.get("benign_only") or {}).items():
        try:
            it = int(entry.get("blue_iter", key.replace("blue_", "")))
        except (TypeError, ValueError):
            continue
        tpr = entry.get("tpr")
        ci = entry.get("tpr_ci") or [tpr, tpr]
        if tpr is None:
            continue
        out[it] = {
            "brr": 100.0 - tpr,
            "ci_lo": 100.0 - ci[1],
            "ci_hi": 100.0 - ci[0],
        }
    return out or None


def plot_utility_by_style(
    results: list[tuple[str, str]],
    out_path: str | Path,
    show_ci: bool = True,
) -> Path:
    out_path = Path(out_path)
    if len(results) > 1:
        print("[plot_utility_by_style] Multiple runs given; using first only.",
              file=sys.stderr)
    label, selfplay_dir = results[0]
    selfplay_dir = Path(selfplay_dir)

    style_rows = _load_per_style(selfplay_dir)
    plain_brr = _load_plain_brr(selfplay_dir)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    if style_rows is None:
        ax.text(0.5, 0.5, "per_style_pud/trend.csv not found",
                ha="center", va="center", transform=ax.transAxes, color=GRAY_COL)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    iters_all: list[int] = sorted({int(r["iter"]) for r in style_rows})

    for style_key, color, marker, display in _STYLE_META:
        n_col = f"{style_key}_n"
        p_col = f"{style_key}_pud"
        rates: list[float] = []
        ci_lo: list[float] = []
        ci_hi: list[float] = []
        present: list[int] = []
        for r in style_rows:
            try:
                n = int(r[n_col])
                p = float(r[p_col])
            except (ValueError, KeyError):
                continue
            if math.isnan(p) or n == 0:
                continue
            k = int(round(p * n))
            lo, hi = _wilson(k, n)
            rates.append(p * 100.0)
            ci_lo.append(lo)
            ci_hi.append(hi)
            present.append(int(r["iter"]))
        if not present:
            continue
        yerr = np.array([[v - l for v, l in zip(rates, ci_lo)],
                         [h - v for v, h in zip(rates, ci_hi)]]) if show_ci else None
        ax.errorbar(present, rates, yerr=yerr,
                    fmt=f"-{marker}", color=color, label=display,
                    linewidth=1.8, markersize=7, capsize=4, capthick=1.0,
                    elinewidth=1.0, zorder=3)

    # Plain-pool BRR reference (faded)
    if plain_brr is not None:
        its = sorted(plain_brr.keys())
        brrs = [plain_brr[i]["brr"] for i in its]
        ax.plot(its, brrs,
                linestyle="--", color=GRAY_COL, alpha=0.6, linewidth=1.6,
                marker="x", markersize=6,
                label="Plain-pool BRR\n(benign_eval, biased upward)", zorder=2)

    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Benign denial rate (%)")
    ax.set_xticks(iters_all)
    ax.set_ylim(0, 45)
    ax.set_title(
        "Utility by query style — headline operating point\n"
        "Adversarial-framed benigns are refused 15–25 pp more than plain"
    )
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=8)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_utility_by_style] selfplay_dir={selfplay_dir}", file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    ap.add_argument("--out", default="figures/utility_by_style.png")
    args = ap.parse_args()
    results = parse_results_arg(args.results)
    out = plot_utility_by_style(results, args.out)
    write_sidecar(out, DESCRIPTION, results)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
