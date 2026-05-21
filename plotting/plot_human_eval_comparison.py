"""
Human-eval comparison: Unprotected vs Manually Protected vs RL Protected.

Cross-distribution robustness check on n=320 hand-written attacks (32 prompts ×
10 seeds). Surfaces the inversion: RL-iter-1 has 4.06% PVR_turn vs Manual 2.5%
on the human distribution — possibly small-N noise; cross_eval_baseline shows
no such inversion at scale.

Source: data/human_eval/comparison.json

CLI:
    python plotting/plot_human_eval_comparison.py \\
        --results results-<ID>[:Label] \\
        [--human-eval-json data/human_eval/comparison.json] \\
        [--out-dir figures/]
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
    from ._data import apply_paper_style, parse_results_arg, FIG_SIZE_1x2
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import apply_paper_style, parse_results_arg, FIG_SIZE_1x2

apply_paper_style()

DESCRIPTION = "Human-attack robustness: Unprotected vs Manual vs RL-iter-1 (n=320 each)"

CONDITION_COLORS = {
    "human_unprotected": "#9E9E9E",
    "human_manual": "#43A047",
    "human_rl_iter1": "#1976D2",
}


def plot_human_eval_comparison(
    results: list[tuple[str, str]],
    human_eval_json: str = "data/human_eval/comparison.json",
    out_dir: str = "figures/",
    show_ci: bool = True,
) -> Path:
    out_path = Path(out_dir) / "human_eval_comparison.png"
    sidecar_path = Path(out_dir) / "human_eval_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = Path(human_eval_json)
    if not json_path.exists():
        print(f"[human_eval_comparison] {human_eval_json} not found", file=sys.stderr)
        return out_path

    with open(json_path) as f:
        rows = json.load(f)

    # Order canonically
    order = ["human_unprotected", "human_manual", "human_rl_iter1"]
    rows_by_tag = {r["tag"]: r for r in rows}
    rows_ordered = [rows_by_tag[t] for t in order if t in rows_by_tag]

    labels = [r["label"] for r in rows_ordered]
    pvr_turn = [r["PVR_turn"] * 100 for r in rows_ordered]  # to %
    pvr_turn_lo = [r["PVR_turn_ci"][0] * 100 for r in rows_ordered]
    pvr_turn_hi = [r["PVR_turn_ci"][1] * 100 for r in rows_ordered]
    wf = [r["WF"] for r in rows_ordered]
    wf_lo = [r["WF_lower"] for r in rows_ordered]
    wf_hi = [r["WF_upper"] for r in rows_ordered]

    pvr_yerr = np.vstack([
        np.array(pvr_turn) - np.array(pvr_turn_lo),
        np.array(pvr_turn_hi) - np.array(pvr_turn),
    ])
    wf_yerr = np.vstack([
        np.array(wf) - np.array(wf_lo),
        np.array(wf_hi) - np.array(wf),
    ])

    colors = [CONDITION_COLORS.get(r["tag"], "#757575") for r in rows_ordered]
    x = np.arange(len(labels))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    bars = ax_l.bar(x, pvr_turn, color=colors, yerr=(pvr_yerr if show_ci else None), capsize=4, ecolor="black")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=9)
    ax_l.set_ylabel("PVR_turn (%)")
    ax_l.set_title("Per-turn attack success on human-written attacks (n=320)")
    for i, b in enumerate(bars):
        h = b.get_height()
        ax_l.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.2f}%", ha="center", fontsize=8)

    # Annotate inversion if present
    try:
        m_idx = order.index("human_manual")
        rl_idx = order.index("human_rl_iter1")
        if pvr_turn[rl_idx] > pvr_turn[m_idx]:
            ax_l.annotate(
                f"Inversion: RL ({pvr_turn[rl_idx]:.2f}%) > Manual ({pvr_turn[m_idx]:.2f}%)\n"
                "may be small-N (n=320) noise — cross_eval_baseline shows\n"
                "no equivalent inversion at n=200/cell × 8 cells.",
                xy=(rl_idx, pvr_turn[rl_idx]),
                xytext=(0.95, 0.95),
                textcoords="axes fraction",
                fontsize=7,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#FBC02D"),
            )
    except (ValueError, IndexError):
        pass

    bars2 = ax_r.bar(x, wf, color=colors, yerr=(wf_yerr if show_ci else None), capsize=4, ecolor="black")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=9)
    ax_r.set_ylabel("Work Factor (1 / PVR_turn)")
    ax_r.set_yscale("log")
    ax_r.set_title("Attacker work factor (higher = harder to attack)")
    for i, b in enumerate(bars2):
        h = b.get_height()
        ax_r.text(b.get_x() + b.get_width() / 2, h * 1.1, f"{h:.1f}×", ha="center", fontsize=8)

    fig.suptitle(
        "Human-attack robustness: 32 prompts × 10 seeds = 320 episodes per condition",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sidecar = {
        "description": DESCRIPTION,
        "source": str(json_path),
        "conditions": [
            {
                "tag": r["tag"],
                "label": r["label"],
                "PVR_turn_pct": r["PVR_turn"] * 100,
                "PVR_turn_ci_pct": [c * 100 for c in r["PVR_turn_ci"]],
                "WF": r["WF"],
                "WF_ci": [r["WF_lower"], r["WF_upper"]],
                "n_attack_prompts": r["n_attack_prompts"],
                "num_seeds": r["num_seeds"],
            }
            for r in rows_ordered
        ],
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[human_eval_comparison] saved {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--human-eval-json", default="data/human_eval/comparison.json")
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_human_eval_comparison(results, args.human_eval_json, args.out_dir)


if __name__ == "__main__":
    main()
