#!/usr/bin/env python3
"""Iter-6 novelty recovery: per-iter honeypot discovery decomposed by tier.

Loads each redteam reward_debug.jsonl in `results-<ID>/iter_N/redteam/...`,
counts unique honeypots discovered during training (via `new_honeypots_accessed`),
and buckets them by tier from `figures/honeypot_tiers.json`.

Hypothesis under test (Pillar 2, fourth corroborating signal): the iter-6 EIS
bump is a *transient defender regression* — blue_5's policy shift briefly
re-opened rare-tier honeypots that earlier blueteams had been blocking. If
iter-6 excess hits are PII-dominant rather than rare, the framing weakens
and Pillar 2 narrative needs revision.

Usage:
    python util/iter6_novelty_breakdown.py \\
        --results-dir results-20260408-1726-t9s16 \\
        --tiers-json figures/honeypot_tiers.json \\
        --out figures/iter6_novelty_recovery.png \\
        --iters 5 6 7
"""

# TODO: See if this is still necessary

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Reuse helpers from honeypot_stats.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from util.honeypot_stats import find_iteration_logs, load_records  # noqa: E402

DESCRIPTION = (
    "Per-iter unique honeypots discovered during training, decomposed by "
    "diagonal-eval tier (pii_dominant / harvestable / rare / training_only). "
    "Tests whether the iter-6 EIS bump is a rare-tier resurgence (supporting "
    "transient defender regression) or PII-dominant (weakening that framing)."
)

# Tier order for stacking (bottom → top of bar)
TIER_ORDER = ["pii_dominant", "harvestable", "rare", "training_only", "never_breached"]
TIER_COLORS = {
    "pii_dominant": "#b2182b",       # deep red
    "harvestable":  "#ef8a62",       # orange-red
    "rare":         "#fddbc7",       # light salmon
    "training_only": "#92c5de",       # light blue
    "never_breached": "#bababa",      # gray
}


def load_tier_map(tiers_json: Path) -> dict[str, str]:
    """honeypot_id -> tier from figures/honeypot_tiers.json."""
    with tiers_json.open() as f:
        data = json.load(f)
    return {hp["id"]: hp["tier"] for hp in data["honeypots"]}


def normalize_honeypot_id(raw: str) -> str:
    """new_honeypots_accessed entries match honeypot_tiers.json IDs already."""
    return str(raw)


def per_iter_breakdown(
    iter_logs: dict[int, Path],
    tier_map: dict[str, str],
    iters: list[int],
) -> dict[int, dict[str, int]]:
    """For each requested iter, return {tier: count_of_honeypot_hits}.

    Counts every `new_honeypots_accessed` event during training mode (mode='train').
    Uses raw event counts (not unique IDs per iter) so that the bar height tracks
    the per-iter EIS-bump intensity, not just discovery diversity.
    """
    out: dict[int, dict[str, int]] = {}
    for k in iters:
        if k not in iter_logs:
            print(f"[warn] iter_{k} not found in results dir; skipping", file=sys.stderr)
            continue
        records = load_records(iter_logs[k])
        tier_counts: dict[str, int] = defaultdict(int)
        for r in records:
            if r.get("mode") != "train":
                continue
            for hp in r.get("new_honeypots_accessed") or []:
                hp_id = normalize_honeypot_id(hp)
                tier = tier_map.get(hp_id, "training_only")
                tier_counts[tier] += 1
        out[k] = dict(tier_counts)
    return out


def per_iter_unique_honeypots(
    iter_logs: dict[int, Path],
    tier_map: dict[str, str],
    iters: list[int],
) -> dict[int, dict[str, set[str]]]:
    """For each requested iter, return {tier: set_of_unique_honeypot_ids}."""
    out: dict[int, dict[str, set[str]]] = {}
    for k in iters:
        if k not in iter_logs:
            continue
        records = load_records(iter_logs[k])
        tier_sets: dict[str, set[str]] = defaultdict(set)
        for r in records:
            if r.get("mode") != "train":
                continue
            for hp in r.get("new_honeypots_accessed") or []:
                hp_id = normalize_honeypot_id(hp)
                tier = tier_map.get(hp_id, "training_only")
                tier_sets[tier].add(hp_id)
        out[k] = {t: s for t, s in tier_sets.items()}
    return out


def plot(
    breakdown: dict[int, dict[str, int]],
    unique_breakdown: dict[int, dict[str, set[str]]],
    out_path: Path,
    title_suffix: str = "",
) -> Path:
    iters = sorted(unique_breakdown.keys())
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)

    bottom = [0.0] * len(iters)
    totals: list[int] = []
    for tier in TIER_ORDER:
        heights = [len(unique_breakdown.get(k, {}).get(tier, set())) for k in iters]
        if not any(heights):
            continue
        ax.bar(
            [str(k) for k in iters],
            heights,
            bottom=bottom,
            color=TIER_COLORS[tier],
            edgecolor="black",
            linewidth=0.4,
            label=tier,
        )
        bottom = [b + h for b, h in zip(bottom, heights)]
    totals = [int(round(b)) for b in bottom]

    # Total-count labels above each bar
    for i, total in enumerate(totals):
        ax.text(i, total + 0.4, str(total), ha="center", va="bottom", fontsize=9)

    # Annotate the iter-6 bump
    if 6 in iters and 5 in iters:
        i6 = iters.index(6)
        i5 = iters.index(5)
        if totals[i6] > totals[i5]:
            ax.annotate(
                f"iter-6 broad-tier recovery\n(+{totals[i6] - totals[i5]} unique honeypots)",
                xy=(i6, totals[i6]),
                xytext=(i5 - 0.3, totals[i6] + 3.0),
                ha="left",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"),
            )

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Unique honeypots discovered (training-time)")
    ax.set_title(f"Iter-6 novelty recovery decomposed by tier{title_suffix}", fontsize=11)
    ax.set_ylim(0, max(totals) + 8)
    ax.legend(loc="upper right", fontsize=8, frameon=False, title="Tier (eval-time)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument(
        "--tiers-json",
        default=Path("figures/honeypot_tiers.json"),
        type=Path,
        help="Path to honeypot_tiers.json (default: figures/honeypot_tiers.json).",
    )
    p.add_argument(
        "--iters",
        nargs="+",
        type=int,
        default=[5, 6, 7],
        help="Iterations to include (default: 5 6 7).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figures/iter6_novelty_recovery.png"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional JSON sidecar (default: <out>.json).",
    )
    args = p.parse_args()

    iter_logs = find_iteration_logs(args.results_dir)
    if not iter_logs:
        sys.exit(f"No redteam reward_debug.jsonl found under {args.results_dir}")

    tier_map = load_tier_map(args.tiers_json)
    breakdown = per_iter_breakdown(iter_logs, tier_map, args.iters)
    unique_breakdown = per_iter_unique_honeypots(iter_logs, tier_map, args.iters)

    out_path = plot(breakdown, unique_breakdown, args.out)

    # JSON sidecar for the decision gate
    sidecar = args.out_json or args.out.with_suffix(".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w") as f:
        json.dump(
            {
                "description": DESCRIPTION,
                "results_dir": str(args.results_dir),
                "tiers_json": str(args.tiers_json),
                "iters": args.iters,
                "events_per_iter_by_tier": {str(k): v for k, v in breakdown.items()},
                "unique_honeypots_per_iter_by_tier": {
                    str(k): {t: sorted(s) for t, s in v.items()}
                    for k, v in unique_breakdown.items()
                },
            },
            f,
            indent=2,
        )

    # Print decision-gate summary
    print(f"\n[iter6_novelty_breakdown] wrote {out_path} + {sidecar}")
    print("\nPer-iter event counts by tier:")
    for k in sorted(breakdown.keys()):
        tier_str = ", ".join(f"{t}={breakdown[k].get(t, 0)}" for t in TIER_ORDER if breakdown[k].get(t, 0))
        print(f"  iter_{k}: {tier_str}")
    print("\nUnique honeypots per iter by tier:")
    for k in sorted(unique_breakdown.keys()):
        for t in TIER_ORDER:
            ids = sorted(unique_breakdown[k].get(t, set()))
            if ids:
                print(f"  iter_{k} {t}: {ids}")


if __name__ == "__main__":
    main()
