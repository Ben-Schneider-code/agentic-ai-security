"""Build figures/utility_ablation.json from raw ablation eval data.

util/plot_ablations.py was written for an A1-A5 multi-variant layout that doesn't
match the current 2-variant (none, plain-only) shape, so we publish a JSON sidecar
directly. All ASRs are computed with the corrected PVR_conv denominator
(n_eps_with_sql per problem_statement.tex eq.49).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MARFT"))

from util.metrics import compute_pairing_metrics  # type: ignore  # noqa: E402

RUNS = {
    "canonical": "results-20260408-1726-t9s16/cross_eval/pairings/red_1_blue_1",
    "none":      "ablations/none/none-20260425-0424-rok62/eval_view/cross_eval/pairings",
    "plain_only": "ablations/plain-only/plain-only-20260425-1524-ln81m/eval_view/cross_eval/pairings",
}

# Per-style benign refusal numbers come from the held-out per-style eval and
# are independent of the PVR_conv denominator fix (different metric).
PER_STYLE_REFUSAL = {
    "canonical": {"plain": 1.4, "adversarial": 3.6, "multi_turn": 1.1, "adv_plain_ratio": 2.5},
    "none":      {"plain": 4.0, "adversarial": 3.0, "multi_turn": 0.4, "adv_plain_ratio": 0.75},
    "plain_only": {"plain": 0.0, "adversarial": 5.0, "multi_turn": 0.8, "adv_plain_ratio": None},
}


def _compute_asr(pairings_dir: str) -> dict:
    p = Path(pairings_dir)
    if not p.exists():
        return {"available": False, "reason": f"{p} not found"}
    if (p / "reward_debug.jsonl").exists():
        jsonls = [p / "reward_debug.jsonl"]
    else:
        jsonls = sorted(p.glob("*/reward_debug.jsonl"))
    if not jsonls:
        return {"available": False, "reason": f"no reward_debug.jsonl under {p}"}
    j = jsonls[0]
    records = []
    with open(j) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    res = compute_pairing_metrics(records)
    return {
        "available": True,
        "source_pairing": j.parent.name,
        "asr_pct": res["metrics"]["asr"],
        "pvr_turn_pct": res["metrics"]["pvr_turn"],
        "n_eps_with_sql": res["episode_stats"]["n_eps_with_sql"],
        "n_attack_episodes": res["n_attack_episodes"],
        "n_hit_eps": res["episode_stats"]["n_hit_eps"],
    }


def main() -> None:
    out = {
        "description": (
            "B3 utility ablation: per-variant refusal-by-style + ASR at red_1×blue_1, "
            "under corrected PVR_conv denominator (n_eps_with_sql per problem_statement.tex eq.49). "
            "Refusal rates are from the held-out per-style benign eval (independent of the denominator fix); "
            "ASR is recomputed from raw reward_debug.jsonl in eval_view/cross_eval/."
        ),
        "denominator_fix_applied": "n_eps_with_sql (= C*_R), per metrics.py:158",
        "variants": {},
    }
    for variant, pdir in RUNS.items():
        asr_block = _compute_asr(pdir)
        ref_block = PER_STYLE_REFUSAL.get(variant, {})
        out["variants"][variant] = {
            "asr": asr_block,
            "per_style_refusal_pct": ref_block,
        }

    out_path = Path("figures/utility_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    # Echo summary
    for v, info in out["variants"].items():
        a = info["asr"]
        if a.get("available"):
            print(f"  {v}: ASR={a['asr_pct']}%  pvr_turn={a['pvr_turn_pct']}%  "
                  f"n_eps_with_sql={a['n_eps_with_sql']}/{a['n_attack_episodes']}  "
                  f"per-style refusal: {info['per_style_refusal_pct']}")
        else:
            print(f"  {v}: NOT AVAILABLE — {a.get('reason')}")


if __name__ == "__main__":
    main()
