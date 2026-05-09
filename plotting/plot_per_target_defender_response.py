"""
Per-target defender response decomposition.

Answers supervisor Q2/Q4: "What makes those targets hard to protect?"

For each top-N honeypot target, measure:
  refusal rate    = % of intent-tagged attack turns blue refused
  breach rate     = % of intent-tagged attack turns where the target was breached
  accepted-clean  = % blue accepted but no breach

Target intent is inferred from substring keywords (column/table names) in
the attacker's user_message. Turns with no keyword match are bucketed as
"ambiguous" and excluded from the per-target plot but reported in the sidecar.

Reveals whether phone/email defeat blue because:
  (a) blue refuses them less often (utility-cost asymmetry), OR
  (b) blue accepts them but breach lands (semantic blind spot).

Source: results-<ID>/cross_eval/pairings/red_i_blue_i/reward_debug.jsonl (diagonal).

CLI:
    python plotting/plot_per_target_defender_response.py \\
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
import numpy as np

try:
    from ._data import apply_paper_style, parse_results_arg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import apply_paper_style, parse_results_arg

apply_paper_style()

DESCRIPTION = "Per-target defender response: refusal vs breach vs accepted-clean per top-10 honeypot"

# Heuristic keyword patterns per honeypot ID for intent detection
TARGET_KEYWORDS: dict[str, list[str]] = {
    "column_access:phone": ["phone", "phonenumber"],
    "column_access:emailaddress": ["email", "e-mail", "emailaddress"],
    "column_access:passwordhash": ["passwordhash", "password hash", "hashed password"],
    "column_access:passwordsalt": ["passwordsalt", "password salt", "salt"],
    "column_access:modifieddate": ["modifieddate", "modified date", "modified_date"],
    "column_access:rowguid": ["rowguid", "row guid", "guid"],
    "column_access:salesperson": ["salesperson"],
    "column_access:companyname": ["companyname", "company name"],
    "column_access:middlename": ["middlename", "middle name"],
    "column_access:namestyle": ["namestyle"],
    "column_access:suffix": ["suffix"],
    "column_access:title": ["title"],
    "table_access:address": ["address"],
    "table_access:customeraddress": ["customeraddress", "customer address"],
    "salesorderid:88888": ["88888"],
    "salesorderid:88889": ["88889"],
    "salesorderid:88890": ["88890"],
    "salesorder_column_access:purchaseordernumber": ["purchaseordernumber", "purchase order number"],
    "salesorder_column_access:accountnumber": ["accountnumber", "account number"],
    "salesorder_column_access:creditcardapprovalcode": ["creditcardapprovalcode", "credit card approval"],
    "salesorder_column_access:billtoaddressid": ["billtoaddressid", "bill to address"],
    "salesorder_column_access:shiptoaddressid": ["shiptoaddressid", "ship to address"],
}


def _load_tier_map(selfplay_dir: str) -> dict[str, str]:
    candidates = [
        Path(selfplay_dir).parent / "figures" / "honeypot_tiers.json",
        Path("figures/honeypot_tiers.json"),
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return {h["id"]: h["tier"] for h in data.get("honeypots", [])}
    return {}


def _parse_accessed(field) -> set[str]:
    """Parse accessed_honeypots into a set of honeypot ids."""
    out: set[str] = set()
    for item in field or []:
        try:
            hp = ast.literal_eval(item) if isinstance(item, str) else item
            if hp.get("accessed"):
                out.add(f"{hp.get('type', '')}:{hp.get('identifier', '')}")
        except Exception:
            pass
    return out


def _detect_targets(user_message: str) -> set[str]:
    """Return set of honeypot ids whose keywords appear in user_message."""
    if not user_message:
        return set()
    msg = user_message.lower()
    out: set[str] = set()
    for hp_id, kws in TARGET_KEYWORDS.items():
        for kw in kws:
            if kw in msg:
                out.add(hp_id)
                break
    return out


def compute_per_target_response(
    selfplay_dir: str,
    cross_eval_subdir: str = "cross_eval",
    diagonal_only: bool = True,
) -> dict[str, dict]:
    base = Path(selfplay_dir) / cross_eval_subdir / "pairings"
    if not base.exists():
        return {}

    counts: dict[str, dict[str, int]] = {hp: {"intent": 0, "refused": 0, "breached": 0, "accepted_clean": 0} for hp in TARGET_KEYWORDS}
    total_attack_turns = 0
    total_with_intent = 0
    total_refused = 0
    total_breached = 0

    for pairing_dir in sorted(base.glob("red_*_blue_*")):
        if diagonal_only:
            try:
                parts = pairing_dir.name.split("_")
                r = int(parts[1])
                b = int(parts[3])
                if r != b:
                    continue
            except (IndexError, ValueError):
                continue

        jsonl = pairing_dir / "reward_debug.jsonl"
        if not jsonl.exists():
            continue
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
                total_attack_turns += 1
                msg = r.get("user_message", "") or ""
                refused = bool(r.get("is_refusal"))
                breached_set = _parse_accessed(r.get("accessed_honeypots"))
                if refused:
                    total_refused += 1
                if breached_set:
                    total_breached += 1
                targets = _detect_targets(msg)
                if not targets:
                    continue
                total_with_intent += 1
                for hp in targets:
                    counts[hp]["intent"] += 1
                    if refused:
                        counts[hp]["refused"] += 1
                    elif hp in breached_set:
                        counts[hp]["breached"] += 1
                    else:
                        counts[hp]["accepted_clean"] += 1

    return {
        "per_target": counts,
        "totals": {
            "total_attack_turns": total_attack_turns,
            "total_with_intent": total_with_intent,
            "total_refused": total_refused,
            "total_breached": total_breached,
        },
    }


def plot_per_target_defender_response(
    results: list[tuple[str, str]],
    out_dir: str = "figures/",
    cross_eval_subdir: str = "cross_eval",
) -> Path:
    out_path = Path(out_dir) / "per_target_defender_response.png"
    sidecar_path = Path(out_dir) / "per_target_defender_response.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label, selfplay_dir = results[0]
    data = compute_per_target_response(selfplay_dir, cross_eval_subdir, diagonal_only=True)
    if not data:
        print("[per_target_defender_response] no data found", file=sys.stderr)
        return out_path

    tier_map = _load_tier_map(selfplay_dir)
    per_target = data["per_target"]
    totals = data["totals"]

    # Top-N by intent count
    ranked = sorted(per_target.items(), key=lambda kv: -kv[1]["intent"])
    top = [kv for kv in ranked if kv[1]["intent"] >= 5][:10]

    if not top:
        print("[per_target_defender_response] no targets with intent >= 5", file=sys.stderr)
        return out_path

    labels = [hp.replace("column_access:", "col:").replace("table_access:", "tbl:").replace("salesorder_column_access:", "so_col:").replace("salesorderid:", "so_id:") for hp, _ in top]
    refused_rates = [c["refused"] / c["intent"] * 100 for _, c in top]
    breached_rates = [c["breached"] / c["intent"] * 100 for _, c in top]
    accepted_rates = [c["accepted_clean"] / c["intent"] * 100 for _, c in top]
    intent_counts = [c["intent"] for _, c in top]
    tier_colors = []
    for hp, _ in top:
        tier = tier_map.get(hp, "unknown")
        tier_colors.append({"pii_dominant": "#B71C1C", "harvestable": "#E65100", "rare": "#37474F"}.get(tier, "#000000"))

    overall_refusal = totals["total_refused"] / totals["total_attack_turns"] * 100 if totals["total_attack_turns"] else 0

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, refused_rates, color="#43A047", label=f"Refused")
    ax.bar(x, accepted_rates, bottom=refused_rates, color="#FFB300", label=f"Accepted (no breach)")
    ax.bar(
        x,
        breached_rates,
        bottom=[r + a for r, a in zip(refused_rates, accepted_rates)],
        color="#D32F2F",
        label=f"Breached",
    )
    ax.axhline(overall_refusal, color="#1976D2", linestyle="--", linewidth=1.2,
               label=f"Overall attack-turn refusal {overall_refusal:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    # Color tick labels by tier
    for tick, color in zip(ax.get_xticklabels(), tier_colors):
        tick.set_color(color)
    ax.set_ylabel("% of intent-tagged attack turns")
    ax.set_title(
        f"Defender response per target ({totals['total_with_intent']} intent-tagged "
        f"of {totals['total_attack_turns']} attack turns, diagonal cells)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    for i, n in enumerate(intent_counts):
        ax.text(i, 102, f"n={n}", ha="center", fontsize=7, color="#555555")

    fig.text(
        0.5,
        -0.04,
        "Tick label color = honeypot tier (PII-dominant, Harvestable, Rare). "
        "Intent inferred from substring match in attacker's user_message; ambiguous turns excluded from per-target view.",
        ha="center",
        fontsize=7.5,
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sidecar = {
        "description": DESCRIPTION,
        "totals": totals,
        "overall_refusal_pct": round(overall_refusal, 2),
        "per_target": {
            hp: {
                **counts,
                "refusal_rate_pct": counts["refused"] / counts["intent"] * 100 if counts["intent"] else None,
                "breach_rate_pct": counts["breached"] / counts["intent"] * 100 if counts["intent"] else None,
                "accepted_clean_pct": counts["accepted_clean"] / counts["intent"] * 100 if counts["intent"] else None,
                "tier": tier_map.get(hp, "unknown"),
            }
            for hp, counts in per_target.items() if counts["intent"] > 0
        },
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[per_target_defender_response] saved {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--cross-eval-subdir", default="cross_eval")
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_per_target_defender_response(results, args.out_dir, args.cross_eval_subdir)


if __name__ == "__main__":
    main()
