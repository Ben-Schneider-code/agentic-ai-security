"""
Held-out per-style benign refusal rate vs blue checkpoint iteration.

Reads from cross_eval/benign_only/blue_*/reward_debug.jsonl (canonical held-out
benign evaluation, n≈600–900 episodes per blue iter). Computes per-style refusal
rates with 95% Wilson CIs for styles: plain, multi_turn, adversarial.

This replaces utility_by_style.png (training-time) as the Pillar 1 headline figure.
Held-out adversarial: ~3.6% vs plain: ~1.4% — a 2.5× ratio significant at 95%.

CLI:
    python plotting/plot_held_out_per_style_refusal.py \\
        --results results-<ID>[:Label] [--cross-eval-subdir cross_eval] \\
        [--out-dir figures/]

Returns:
    Path to saved PNG (and JSON sidecar).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ._data import (
        apply_paper_style,
        parse_results_arg,
        wilson_ci_pct,
        FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        parse_results_arg,
        wilson_ci_pct,
        FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = "Held-out per-style benign refusal rate by blue checkpoint (Pillar 1 headline)"

STYLE_LABELS = {
    "plain": "Plain",
    "multi_turn": "Multi-turn",
    "adversarial": "Adversarial-framed",
}
STYLE_COLORS = {
    "plain": "#2196F3",
    "multi_turn": "#FF9800",
    "adversarial": "#F44336",
}
STYLE_MARKERS = {"plain": "o", "multi_turn": "s", "adversarial": "^"}
STYLES = ["plain", "multi_turn", "adversarial"]


def load_held_out_per_style(
    selfplay_dir: str, cross_eval_subdir: str = "cross_eval"
) -> dict[int, dict[str, dict]]:
    """
    Returns {blue_iter: {style: {"n": int, "refused": int}}}
    from <selfplay_dir>/<cross_eval_subdir>/benign_only/blue_*/reward_debug.jsonl.
    """
    base = Path(selfplay_dir) / cross_eval_subdir / "benign_only"
    if not base.is_dir():
        return {}
    result: dict[int, dict[str, dict]] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        import re
        m = re.match(r"^blue_(\d+)$", entry.name)
        if not m:
            continue
        blue_iter = int(m.group(1))
        jsonl = entry / "reward_debug.jsonl"
        if not jsonl.is_file():
            continue
        counts: dict[str, dict] = {s: {"n": 0, "refused": 0} for s in STYLES}
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("turn_type") != "benign":
                    continue
                style = r.get("benign_style", "plain")
                if style not in counts:
                    counts[style] = {"n": 0, "refused": 0}
                counts[style]["n"] += 1
                if r.get("is_refusal", False):
                    counts[style]["refused"] += 1
        result[blue_iter] = counts
    return result


def plot_held_out_per_style_refusal(
    results: list[tuple[str, str]],
    cross_eval_subdir: str = "cross_eval",
    out_dir: str = "figures/",
    show_ci: bool = True,
) -> Path:
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

    out_path = Path(out_dir) / "held_out_per_style_refusal.png"
    sidecar_path = Path(out_dir) / "held_out_per_style_refusal.json"

    label, selfplay_dir = results[0]
    data = load_held_out_per_style(selfplay_dir, cross_eval_subdir)
    if not data:
        print(
            f"[held_out_per_style_refusal] No data found at "
            f"{selfplay_dir}/{cross_eval_subdir}/benign_only/",
            file=sys.stderr,
        )
        return out_path

    blue_iters = sorted(data.keys())
    sidecar: dict = {"selfplay_dir": selfplay_dir, "per_iter": {}, "pooled": {}}

    for style in STYLES:
        rates, lo_errs, hi_errs = [], [], []
        for bi in blue_iters:
            counts = data[bi].get(style, {"n": 0, "refused": 0})
            n, ref = counts["n"], counts["refused"]
            if n == 0:
                rates.append(float("nan"))
                lo_errs.append(0.0)
                hi_errs.append(0.0)
            else:
                rate = ref / n * 100
                lo, hi = wilson_ci_pct(ref, n, z=1.96)
                rates.append(rate)
                lo_errs.append(max(0.0, rate - lo))
                hi_errs.append(max(0.0, hi - rate))
        ax.errorbar(
            blue_iters,
            rates,
            yerr=([lo_errs, hi_errs] if show_ci else None),
            label=STYLE_LABELS[style],
            color=STYLE_COLORS[style],
            marker=STYLE_MARKERS[style],
            capsize=3,
            linewidth=1.5,
            markersize=5,
        )
        sidecar["per_iter"][style] = {
            str(bi): {"n": data[bi].get(style, {"n": 0})["n"],
                      "refused": data[bi].get(style, {"n": 0, "refused": 0})["refused"],
                      "rate_pct": r}
            for bi, r in zip(blue_iters, rates)
        }

    # Pooled values annotation
    pooled_rows = []
    grand_total_n = 0
    for style in STYLES:
        total_n = sum(data[bi].get(style, {"n": 0})["n"] for bi in blue_iters)
        total_ref = sum(data[bi].get(style, {"n": 0, "refused": 0})["refused"] for bi in blue_iters)
        grand_total_n += total_n
        if total_n > 0:
            rate = total_ref / total_n * 100
            lo, hi = wilson_ci_pct(total_ref, total_n, z=1.96)
            pooled_rows.append(f"{STYLE_LABELS[style]}: {rate:.1f}% [{lo:.1f}, {hi:.1f}] (n={total_n:,})")
            sidecar["pooled"][style] = {"n": total_n, "refused": total_ref,
                                         "rate_pct": rate, "ci95_lo": lo, "ci95_hi": hi}

    ax.set_xlabel("Blue checkpoint iteration")
    ax.set_ylabel("Benign refusal rate (%)")
    ax.set_title("Held-out benign refusal by query style (95% Wilson CI)")
    ax.set_xticks(blue_iters)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7)

    note = f"Pooled across {grand_total_n:,} held-out queries (95% Wilson CI): " + " | ".join(pooled_rows)
    ax.text(
        0.01, 0.99, note,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=6,
        style="italic",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[held_out_per_style_refusal] saved {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--cross-eval-subdir", default="cross_eval")
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_held_out_per_style_refusal(results, args.cross_eval_subdir, args.out_dir)


if __name__ == "__main__":
    main()
