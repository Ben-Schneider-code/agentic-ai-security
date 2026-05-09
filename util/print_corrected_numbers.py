"""Print all key numbers needed to update evaluation.md after the PVR_conv fix.

Reads the regenerated sidecar JSONs in figures/, plus recomputes ablation ASRs
from raw pairings/, plus per-cell column-0 / row-0 numbers for the
generalization plot. Output is meant to be eyeballed alongside evaluation.md.
"""
from __future__ import annotations
import json, sys, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "MARFT"))

from plotting._data import load_cross_eval_results  # type: ignore  # noqa: E402
from util.metrics import compute_pairing_metrics  # type: ignore  # noqa: E402

CANONICAL = "results-20260408-1726-t9s16"
FIGS = Path("figures")


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    # 1. Headline plateau & diagonal mean from pvr_asymptote.json
    _section("PVR_conv / PVR_turn — pvr_asymptote.json")
    p = json.load(open(FIGS / "pvr_asymptote.json"))
    m = p.get("metrics", {})
    if m:
        print(f"source_subdir: {m.get('source_subdir')}")
        print(f"PVR_conv per-iter %:   {m['pvr_conv']['per_iter_pct']}")
        print(f"PVR_conv 99% CIs:      {m['pvr_conv']['per_iter_ci_99']}")
        print(f"PVR_conv diag mean %:  {m['pvr_conv']['diag_mean_pct']}")
        print(f"PVR_conv plateau %:    {m['pvr_conv']['plateau_tail_mean_pct']} ± {m['pvr_conv']['plateau_tail_std_pp']} pp (tail-{m['n_tail_for_plateau']})")
        print(f"PVR_turn per-iter %:   {m['pvr_turn']['per_iter_pct']}")
        print(f"PVR_turn 99% CIs:      {m['pvr_turn']['per_iter_ci_99']}")
        print(f"PVR_turn diag mean %:  {m['pvr_turn']['diag_mean_pct']}")
        print(f"PVR_turn plateau %:    {m['pvr_turn']['plateau_tail_mean_pct']} ± {m['pvr_turn']['plateau_tail_std_pp']} pp (tail-{m['n_tail_for_plateau']})")
    else:
        print("(empty metrics — sidecar not refreshed)")

    # 2. Rank-invariance
    _section("Rank invariance — cross_eval_rank_invariance.json")
    ri = json.load(open(FIGS / "cross_eval_rank_invariance.json"))
    if "diag_mean_pct" in ri:
        print(f"diag mean = {ri['diag_mean_pct']}%, off-diag mean (excl outlier) = {ri['offdiag_mean_pct']}%")
        print(f"z = {ri['z_stat']}, p = {ri['p_value']}, verdict = {ri['verdict']}")
        print(f"outlier {ri.get('outlier_cell')}: asr={ri.get('outlier_asr')}, n={ri.get('outlier_n')}")
        print(f"diag_n_total={ri['diag_n_total']}, offdiag_n_total={ri['offdiag_n_total']}")
        # Off-diag mean WITH outlier
        all_off = ri["offdiag_values"]
        if ri.get("outlier_asr") is not None:
            all_off_incl = all_off + [ri["outlier_asr"]]
            print(f"off-diag mean (incl outlier) = {statistics.mean(all_off_incl):.2f}%")
    else:
        print("(no metrics)")

    # 3. Baseline-vs-trained
    _section("Baseline vs trained — baseline_vs_trained_defense.json")
    bv = json.load(open(FIGS / "baseline_vs_trained_defense.json"))
    print(json.dumps(bv.get("metrics", bv), indent=2)[:2200])

    # 4. Tier decomposition
    _section("Tier decomposition — tier_pvr_decomposition.json")
    td = json.load(open(FIGS / "tier_pvr_decomposition.json"))
    print(json.dumps(td.get("metrics", td), indent=2)[:2200])

    # 5. Per-target defender response (refusal rates)
    _section("Per-target defender response — per_target_defender_response.json")
    pt = json.load(open(FIGS / "per_target_defender_response.json"))
    print(json.dumps(pt.get("metrics", pt), indent=2)[:2200])

    # 6. Honeypot per-iter heatmap (top-2 share, total)
    _section("Honeypot per-iter heatmap — honeypot_per_iter_heatmap.json")
    hp = json.load(open(FIGS / "honeypot_per_iter_heatmap.json"))
    print(f"total_hits={hp.get('total_hits')}  top2_share_pct={hp.get('top2_share_pct')}  n_iters={hp.get('n_iterations')}  n_honeypots={hp.get('n_honeypots')}")
    for h in hp.get("honeypots", [])[:8]:
        print(f"  {h['id']:50s} tier={h['tier']:14s}  total={h['total_hits']:>4}  iters_with_hit={h['iters_with_hit']}/8  per_iter={list(h['per_iter_hits'].values())}")

    # 7. Honeypot tiers (PII share etc)
    _section("Honeypot tiers — honeypot_tiers.json")
    ht = json.load(open(FIGS / "honeypot_tiers.json"))
    print(json.dumps({k: v for k, v in ht.items() if k != 'honeypots'}, indent=2)[:1200])

    # 8. Generalization (b0 column / r0 row)
    _section("Generalization — column-b0 + row-r0 ASRs (recomputed)")
    ce = load_cross_eval_results(CANONICAL, "cross_eval")
    pairings = ce.get("pairings", {})
    col_b0 = []
    row_r0 = []
    for k, v in pairings.items():
        ri = v.get("red_iter"); bi = v.get("blue_iter")
        if bi == 0:
            col_b0.append((ri, v["metrics"]["asr"]))
        if ri == 0:
            row_r0.append((bi, v["metrics"]["asr"]))
    col_b0.sort(); row_r0.sort()
    print("Column b0 (iter red 0..7 vs frozen blue_0):", [f"{a:.1f}" for _, a in col_b0])
    print("Row r0 (frozen red_0 vs iter blue 0..7):    ", [f"{a:.1f}" for _, a in row_r0])

    # 9. Diagonal cell ASRs (fresh)
    _section("Diagonal cells (fresh) — for 7.4–20.5% range claim")
    diag = []
    for k, v in pairings.items():
        ri = v.get("red_iter"); bi = v.get("blue_iter")
        if ri == bi:
            diag.append((ri, v["metrics"]["asr"], v["metrics"]["pvr_turn"]))
    diag.sort()
    for i, asr, pvrt in diag:
        print(f"  iter {i}: asr={asr:.2f}%  pvr_turn={pvrt:.2f}%")
    asrs = [a for _, a, _ in diag]
    print(f"  diag mean: {statistics.mean(asrs):.2f}%  range: {min(asrs):.2f}–{max(asrs):.2f}%")

    # 10. Cross-eval matrix-wide range (for "values 7.4–20.5%")
    _section("Cross-eval all 64 cells (fresh) — full matrix range")
    all_asrs = [(v.get("red_iter"), v.get("blue_iter"), v["metrics"]["asr"]) for v in pairings.values()]
    asrs = [a for _, _, a in all_asrs]
    print(f"  matrix range: {min(asrs):.2f}–{max(asrs):.2f}%  mean: {statistics.mean(asrs):.2f}%  std: {statistics.stdev(asrs):.2f}")
    # Find min/max cells
    minc = min(all_asrs, key=lambda t: t[2])
    maxc = max(all_asrs, key=lambda t: t[2])
    print(f"  min cell: r{minc[0]}xb{minc[1]} = {minc[2]:.2f}%")
    print(f"  max cell: r{maxc[0]}xb{maxc[1]} = {maxc[2]:.2f}%")

    # 11. Ablation ASRs (fresh) — read raw pairings via metrics.compute_pairing_metrics
    _section("Ablation ASRs (fresh)")
    for ab_name, ab_dir in [
        ("canonical (red_1 vs blue_1)", f"{CANONICAL}/cross_eval/pairings/red_1_blue_1"),
        ("none ablation", "ablations/none/none-20260425-0424-rok62/eval_view/cross_eval/pairings"),
        ("plain-only ablation", "ablations/plain-only/plain-only-20260425-1524-ln81m/eval_view/cross_eval/pairings"),
    ]:
        p = Path(ab_dir)
        if not p.exists():
            print(f"  {ab_name}: NOT FOUND ({ab_dir})")
            continue
        if (p / "reward_debug.jsonl").exists():
            jsonls = [p / "reward_debug.jsonl"]
        else:
            jsonls = sorted(p.glob("*/reward_debug.jsonl"))
        for j in jsonls[:2]:
            recs = []
            with open(j) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln: continue
                    try: recs.append(json.loads(ln))
                    except: pass
            r = compute_pairing_metrics(recs)
            print(f"  {ab_name} [{j.parent.name}]: asr={r['metrics']['asr']:.2f}%  pvr_turn={r['metrics']['pvr_turn']:.2f}%  n_eps_with_sql={r['episode_stats']['n_eps_with_sql']}/{r['n_attack_episodes']}")

    # 12. Baseline cross_eval (fresh) — for 58.25 → ?
    _section("Baseline (cross_eval_baseline) ASRs (fresh)")
    ceb = load_cross_eval_results(CANONICAL, "cross_eval_baseline")
    if ceb:
        bv_asrs = []
        bv_tnrs = []
        for k, v in ceb.get("pairings", {}).items():
            bv_asrs.append((v.get("red_iter"), v["metrics"]["asr"]))
            bv_tnrs.append((v.get("red_iter"), v["metrics"]["tnr"]))
        bv_asrs.sort()
        print(f"  per-red-iter ASR: {[f'{a:.1f}' for _, a in bv_asrs]}")
        print(f"  mean ASR: {statistics.mean(a for _,a in bv_asrs):.2f}%")
        print(f"  mean TNR: {statistics.mean(t for _,t in bv_tnrs):.2f}%")
    else:
        print("  baseline cross_eval not loadable")


if __name__ == "__main__":
    main()
