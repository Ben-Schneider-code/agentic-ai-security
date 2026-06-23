"""
Defender-concentration analysis: blue block rate stability + breach tier distribution.

Reads from iter_*/blueteam/.../debug_logs/reward_debug.jsonl (training-time).
Produces a two-panel figure:
  Left:  per-iter attack-turn block rate (stable ~92%)
  Right: per-iter stacked bar of honeypot-breach tier distribution

Key finding: blue maintains a stable ~92% attack-turn block rate across all iters.
Training-time breaches spread across harvestable (50%) and PII-dominant (36%) tiers;
PII-dominant concentration strengthens at eval time (60.5%) as the attacker
specializes toward high-value targets.

CLI:
    python plotting/plot_defender_concentration.py \\
        --results results-<ID>[:Label] [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ._data import (
        apply_paper_style,
        parse_results_arg,
        FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        parse_results_arg,
        FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = "Defender concentration: block-rate stability + breach-tier distribution"

TIER_COLORS = {
    "pii_dominant": "#F44336",
    "harvestable": "#FF9800",
    "rare": "#9E9E9E",
    "unknown": "#607D8B",
}
TIER_LABELS = {
    "pii_dominant": "PII-dominant",
    "harvestable": "Harvestable",
    "rare": "Rare",
}
TIERS = ["pii_dominant", "harvestable", "rare"]


def _load_tier_map(selfplay_dir: str, out_dir: str | Path | None = None) -> dict[str, str]:
    """Load honeypot_id -> tier from honeypot_tiers.json.

    The figure out_dir (where plot_honeypot_difficulty writes the sidecar) is checked
    first; a custom ``--out-dir`` is the common case and was previously missed, leaving
    the map empty so every breach fell into an 'unknown' tier (and cum_total=0 raised
    ZeroDivisionError downstream).
    """
    candidates = []
    if out_dir is not None:
        candidates.append(Path(out_dir) / "honeypot_tiers.json")
    candidates += [
        Path(selfplay_dir).parent / "figures" / "honeypot_tiers.json",
        Path("figures/honeypot_tiers.json"),
    ]
    for tiers_path in candidates:
        if tiers_path.exists():
            with open(tiers_path) as f:
                data = json.load(f)
            return {h["id"]: h["tier"] for h in data.get("honeypots", [])}
    return {}


def _find_blue_jsonl(iter_dir: Path) -> Path | None:
    candidates = list(iter_dir.glob("blueteam/**/debug_logs/reward_debug.jsonl"))
    return candidates[0] if candidates else None


def compute_defender_concentration(selfplay_dir: str, out_dir: str | Path | None = None) -> dict[int, dict]:
    tier_map = _load_tier_map(selfplay_dir, out_dir=out_dir)
    base = Path(selfplay_dir)
    result: dict[int, dict] = {}

    for iter_dir in sorted(base.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        try:
            iter_num = int(iter_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        jsonl = _find_blue_jsonl(iter_dir)
        if jsonl is None:
            continue

        total_attack = 0
        total_refused = 0
        tier_counts: dict[str, int] = {t: 0 for t in TIERS + ["unknown"]}

        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("turn_type") != "attack":
                    continue
                total_attack += 1
                if r.get("is_refusal"):
                    total_refused += 1
                for hp_raw in r.get("accessed_honeypots", []):
                    try:
                        hp = ast.literal_eval(hp_raw) if isinstance(hp_raw, str) else hp_raw
                        if hp.get("accessed"):
                            hp_id = f"{hp.get('type', '')}:{hp.get('identifier', '')}"
                            tier = tier_map.get(hp_id, "unknown")
                            tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    except Exception:
                        pass

        if total_attack == 0:
            continue
        result[iter_num] = {
            "total_attack": total_attack,
            "refused": total_refused,
            "block_rate_pct": total_refused / total_attack * 100,
            "tier_counts": tier_counts,
            "total_breaches": sum(tier_counts.values()),
        }
    return result


def plot_defender_concentration(
    results: list[tuple[str, str]],
    out_dir: str = "figures/",
) -> Path:
    out_path = Path(out_dir) / "defender_concentration.png"
    sidecar_path = Path(out_dir) / "defender_concentration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label, selfplay_dir = results[0]
    data = compute_defender_concentration(selfplay_dir, out_dir=out_dir)
    if not data:
        print(f"[defender_concentration] No data found.", file=sys.stderr)
        return out_path

    with open(sidecar_path, "w") as f:
        json.dump({str(k): v for k, v in data.items()}, f, indent=2)

    iters = sorted(data.keys())
    block_rates = [data[i]["block_rate_pct"] for i in iters]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    # Left: block rate line
    ax_l.plot(iters, block_rates, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax_l.axhline(sum(block_rates) / len(block_rates), color="#2196F3", linestyle="--",
                 linewidth=1, alpha=0.5, label=f"Mean {sum(block_rates)/len(block_rates):.1f}%")
    ax_l.set_xlabel("Blue training iteration")
    ax_l.set_ylabel("Attack-turn block rate (%)")
    ax_l.set_title("Blue team block rate stability")
    ax_l.set_xticks(iters)
    ax_l.set_ylim(85, 100)
    ax_l.legend(fontsize=8)

    # Right: stacked bar of breach tier distribution
    bottoms = [0.0] * len(iters)
    for tier in TIERS:
        heights = []
        for i in iters:
            tc = data[i]["tier_counts"]
            total = data[i]["total_breaches"]
            heights.append(tc.get(tier, 0) / total * 100 if total > 0 else 0)
        ax_r.bar(iters, heights, bottom=bottoms,
                 label=TIER_LABELS.get(tier, tier),
                 color=TIER_COLORS.get(tier, "#607D8B"))
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax_r.set_xlabel("Blue training iteration")
    ax_r.set_ylabel("% of training-time breaches")
    ax_r.set_title("Breach tier distribution (training-time)")
    ax_r.set_xticks(iters)
    ax_r.legend(loc="upper right", fontsize=8)

    # Add cumulative annotation
    cum_pii = sum(data[i]["tier_counts"].get("pii_dominant", 0) for i in iters)
    cum_harv = sum(data[i]["tier_counts"].get("harvestable", 0) for i in iters)
    cum_rare = sum(data[i]["tier_counts"].get("rare", 0) for i in iters)
    cum_total = cum_pii + cum_harv + cum_rare
    if cum_total > 0:
        note = (f"Cumulative: PII {cum_pii/cum_total*100:.0f}% | "
                f"Harvestable {cum_harv/cum_total*100:.0f}% | "
                f"Rare {cum_rare/cum_total*100:.0f}%")
    else:
        note = "Cumulative: no tier-classified breaches"
    ax_r.text(0.01, 1.01, note, transform=ax_r.transAxes,
              va="bottom", ha="left", fontsize=6, style="italic")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[defender_concentration] saved {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_defender_concentration(results, args.out_dir)


if __name__ == "__main__":
    main()
