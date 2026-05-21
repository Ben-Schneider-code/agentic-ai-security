#!/usr/bin/env python3
"""Per-honeypot training-time vs eval-time hits, decomposed by tier.

Walks training-time `iter_N/redteam/.../reward_debug.jsonl` for the canonical
22-honeypot universe (from `iter_1/redteam/.../reward_config.yaml`) and
eval-time `diagonal_eval/pairings/red_i_blue_i/reward_debug.jsonl`.

Headline:
- 22 declared honeypots; 17 observed at eval time on the diagonal.
- 5 are "training-only" (companyname, namestyle, shiptoaddressid,
  billtoaddressid, purchaseordernumber) — attacker-eval-coverage gap, not
  defender protection.
- 3 (middlename, suffix, title) never breached at eval — but middlename and
  suffix DO appear in training (low-information PII columns blue learns to
  block; attacker rarely retries at eval).

Usage:
    python util/honeypot_training_vs_eval.py \\
        --results-dir results-20260408-1726-t9s16 \\
        --tiers-json figures/honeypot_tiers.json \\
        --out figures/honeypot_training_vs_eval.png
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from util.honeypot_stats import (  # noqa: E402
    find_iteration_logs,
    load_records,
    load_reward_config,
)

DESCRIPTION = (
    "Training-time vs eval-time hits per honeypot, grouped by diagonal-eval "
    "tier. Stars mark the 5 training-only honeypots (eval coverage 0) and "
    "the 3 never-breached at eval. Supports the framing that the "
    "eval-time coverage gap (17/22) is attacker-budget limited, not "
    "defender-protected."
)

TIER_ORDER = ["pii_dominant", "harvestable", "rare", "training_only", "never_breached"]
TIER_COLORS = {
    "pii_dominant": "#b2182b",
    "harvestable":  "#ef8a62",
    "rare":         "#fddbc7",
    "training_only": "#92c5de",
    "never_breached": "#bababa",
}


def load_tier_map(tiers_json: Path) -> dict[str, str]:
    with tiers_json.open() as f:
        data = json.load(f)
    return {hp["id"]: hp["tier"] for hp in data["honeypots"]}


def declared_universe_from_reward_config(results_dir: Path) -> list[str]:
    """Honeypot universe from iter_1/redteam/.../reward_config.yaml.

    Returns canonical honeypot IDs: type:identifier (matches tier_map keys).
    """
    candidates = list(
        (results_dir / "iter_1" / "redteam").glob("**/reward_config.yaml")
    )
    if not candidates:
        sys.exit(f"reward_config.yaml not found under {results_dir}/iter_1/redteam")
    cfg = load_reward_config(candidates[0])

    ids: list[str] = []
    for tbl in cfg.get("honeypot_tables", []) or []:
        ids.append(f"table_access:{tbl}")
    for col in cfg.get("honeypot_columns", []) or []:
        ids.append(f"column_access:{col}")
    for col in cfg.get("honeypot_salesorder_columns", []) or []:
        ids.append(f"salesorder_column_access:{col}")
    for oid in cfg.get("honeypot_order_ids", []) or []:
        ids.append(f"salesorderid:{oid}")
    return ids


def count_training_hits(results_dir: Path) -> Counter:
    """Count training-time honeypot hits across all iters via new_honeypots_accessed."""
    counter: Counter = Counter()
    iter_logs = find_iteration_logs(results_dir)
    for k, path in sorted(iter_logs.items()):
        records = load_records(path)
        for r in records:
            if r.get("mode") != "train":
                continue
            for hp in r.get("new_honeypots_accessed") or []:
                counter[str(hp)] += 1
    return counter


def _parse_accessed_entry(entry) -> tuple[str, str, bool] | None:
    """Diagonal-eval logs store accessed_honeypots as a list of repr-strings."""
    try:
        d = ast.literal_eval(entry) if isinstance(entry, str) else entry
    except (ValueError, SyntaxError):
        return None
    if not isinstance(d, dict):
        return None
    typ = d.get("type")
    ident = d.get("identifier")
    accessed = bool(d.get("accessed"))
    if typ is None or ident is None:
        return None
    return str(typ), str(ident), accessed


def count_eval_hits(results_dir: Path) -> Counter:
    """Count eval-time honeypot hits from diagonal_eval pairings (accessed=True only)."""
    counter: Counter = Counter()
    diag_dir = results_dir / "diagonal_eval" / "pairings"
    if not diag_dir.exists():
        print(f"[warn] {diag_dir} not found", file=sys.stderr)
        return counter
    for pairing_dir in sorted(diag_dir.iterdir()):
        if not pairing_dir.is_dir():
            continue
        log = pairing_dir / "reward_debug.jsonl"
        if not log.exists():
            continue
        with log.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ah = d.get("accessed_honeypots") or []
                for entry in ah:
                    parsed = _parse_accessed_entry(entry)
                    if parsed is None:
                        continue
                    typ, ident, accessed = parsed
                    if accessed:
                        counter[f"{typ}:{ident}"] += 1
    return counter


def classify_tier(
    hp_id: str,
    tier_map: dict[str, str],
    eval_count: int,
    train_count: int,
) -> str:
    if hp_id in tier_map:
        return tier_map[hp_id]
    # Not in tier_map → not observed at eval
    if train_count > 0 and eval_count == 0:
        return "training_only"
    if eval_count == 0 and train_count == 0:
        return "never_breached"
    # Defensive default — should rarely hit
    return "rare"


def plot(
    declared: list[str],
    train_counts: Counter,
    eval_counts: Counter,
    tier_map: dict[str, str],
    out_path: Path,
) -> Path:
    rows: list[dict] = []
    for hp in declared:
        t = train_counts.get(hp, 0)
        e = eval_counts.get(hp, 0)
        tier = classify_tier(hp, tier_map, e, t)
        rows.append({"id": hp, "train": t, "eval": e, "tier": tier})

    # Order: by tier, then by descending eval-count, then by descending train-count
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    rows.sort(key=lambda r: (tier_rank.get(r["tier"], 99), -r["eval"], -r["train"]))

    fig_h = max(7.5, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(11, fig_h), constrained_layout=True)

    y = np.arange(len(rows))
    bar_h = 0.38
    train_vals = [r["train"] for r in rows]
    eval_vals = [r["eval"] for r in rows]

    # Horizontal bars: training (top) and eval (bottom) per honeypot
    ax.barh(y - bar_h / 2, train_vals, bar_h, color="#cc4c2c", label="training-time hits", edgecolor="black", linewidth=0.3)
    ax.barh(y + bar_h / 2, eval_vals, bar_h, color="#3a7ab8", label="eval-time hits (diagonal)", edgecolor="black", linewidth=0.3)

    # Tick labels with tier color stripe
    labels = []
    for r in rows:
        marker = ""
        if r["tier"] == "training_only":
            marker = " ★"  # 5 missing-from-eval
        elif r["tier"] == "never_breached" and r["train"] == 0 and r["eval"] == 0:
            marker = " ✗"  # truly never breached
        elif r["tier"] == "never_breached":
            marker = " ◇"  # never_breached at eval but seen at training
        labels.append(f"{r['id']}{marker}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    # Color per-tier bands
    for i, r in enumerate(rows):
        ax.axhspan(i - 0.5, i + 0.5, alpha=0.10, color=TIER_COLORS.get(r["tier"], "#cccccc"), zorder=0)

    ax.invert_yaxis()
    ax.set_xlabel("Honeypot hits (count)")
    ax.set_xscale("symlog", linthresh=2)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    # Tier legend (color bands)
    from matplotlib.patches import Patch
    handles, labels_ = ax.get_legend_handles_labels()
    tier_handles = [
        Patch(facecolor=TIER_COLORS[t], alpha=0.4, label=t)
        for t in TIER_ORDER
        if any(r["tier"] == t for r in rows)
    ]
    leg2 = ax.legend(
        handles=tier_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        title="Tier (eval-time)",
    )
    ax.add_artist(leg2)
    ax.legend(handles=handles, labels=labels_, loc="lower right", fontsize=9)

    n_train_only = sum(1 for r in rows if r["tier"] == "training_only")
    n_never = sum(1 for r in rows if r["tier"] == "never_breached")
    fig.suptitle(
        f"Per-honeypot training vs eval hits  ★ training-only ({n_train_only})  "
        f"◇/✗ never breached at eval ({n_never})",
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--tiers-json", default=Path("figures/honeypot_tiers.json"), type=Path)
    p.add_argument("--out", default=Path("figures/honeypot_training_vs_eval.png"), type=Path)
    p.add_argument("--out-json", default=None, type=Path)
    args = p.parse_args()

    declared = declared_universe_from_reward_config(args.results_dir)
    train_counts = count_training_hits(args.results_dir)
    eval_counts = count_eval_hits(args.results_dir)
    tier_map = load_tier_map(args.tiers_json)

    out_path = plot(declared, train_counts, eval_counts, tier_map, args.out)

    sidecar = args.out_json or args.out.with_suffix(".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for hp in declared:
        t = train_counts.get(hp, 0)
        e = eval_counts.get(hp, 0)
        rows.append({"id": hp, "train_hits": t, "eval_hits": e, "tier": classify_tier(hp, tier_map, e, t)})
    with sidecar.open("w") as f:
        json.dump(
            {
                "description": DESCRIPTION,
                "results_dir": str(args.results_dir),
                "n_declared": len(declared),
                "n_observed_eval": sum(1 for r in rows if r["eval_hits"] > 0),
                "n_observed_train": sum(1 for r in rows if r["train_hits"] > 0),
                "rows": rows,
            },
            f,
            indent=2,
        )

    # Verification print
    train_only = [r for r in rows if r["tier"] == "training_only"]
    never = [r for r in rows if r["tier"] == "never_breached"]
    print(f"[honeypot_training_vs_eval] wrote {out_path} + {sidecar}")
    print(f"  declared: {len(declared)} | observed at eval: {sum(1 for r in rows if r['eval_hits'] > 0)} | observed at train: {sum(1 for r in rows if r['train_hits'] > 0)}")
    print("  training_only (eval=0, train>0):")
    for r in train_only:
        print(f"    {r['id']}: train={r['train_hits']} eval={r['eval_hits']}")
    print("  never_breached (eval=0):")
    for r in never:
        print(f"    {r['id']}: train={r['train_hits']} eval={r['eval_hits']}")
    # Verification gate: every training_only must have train > 0
    bad = [r for r in train_only if r["train_hits"] == 0]
    if bad:
        print(f"\n[WARNING] {len(bad)} training_only honeypots have ZERO training hits — narrative invalid for these:", file=sys.stderr)
        for r in bad:
            print(f"  {r['id']}", file=sys.stderr)


if __name__ == "__main__":
    main()
