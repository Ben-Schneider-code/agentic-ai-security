#!/usr/bin/env python3
"""Assemble the RL-vs-manual defender comparison table.

Consumes two pre-computed artifact trees:

  1. data/human_attack_audit/<base_tag>_<defender>_raw/seed*.jsonl
     (produced by util/jailbreak_baseline.py) for the human adversary.
  2. <cross_eval_dir>/pairings/red_<r>_blue_<b>/ reward_debug.jsonl+
     summary.json + benign_only/blue_<b>/summary.json
     (produced by util/cross_evaluate.py) for the co-trained RL red
     adversary and the defender-side PUD.

Produces the row data behind \\ref{tab:rl-vs-manual} in
\\S\\ref{sec:eval-rl-vs-human}. For each (base_model, defender ∈
{manual, rl}, adversary ∈ {human, rl_red}) cell, reports:

    PVR_conv (Wilson 95% CI)
    PVR_turn (Wilson 95% CI)   # collapses to PVR_conv for human cells
    PUD      (Wilson 95% CI)   # defender-side; adversary-agnostic
    F1       = HM(1-PVR_turn, 1-PUD)
    Neutral-SQL rate (cross_eval pairings only)

And a two-proportion p-value comparing manual vs RL for each
(base_model, adversary) cell (Fisher's exact when N < 40, pooled
z-test otherwise — same convention as util/mcnemar_cross_eval.py).

Outputs:
    data/rl_vs_human/<base_tag>.json
    figures/rl_vs_human/rl_vs_human_table.tex   (\\input-ready tabular)

Conventions (encoded in defaults, overridable via --config JSON):
  * Defender "manual" ≡ blue_iter=0 (base + sql_system_prompt).
  * Defender "rl"     ≡ blue_iter=<final> (LoRA blue + sql_system_prompt).
  * Adversary "rl_red" ≡ red_iter=<final>.
  * Human audit tags follow {base_tag}_{manual,rl} (the jailbreak_baseline
    manifest is expected to have emitted both variants).

Usage:
    python util/eval_rl_vs_human.py \\
        --config configs/rl_vs_human.json \\
        --audit-dir data/human_attack_audit \\
        --out-json-dir data/rl_vs_human \\
        --out-tex figures/rl_vs_human/rl_vs_human_table.tex
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    from scipy import stats
except ImportError:
    stats = None  # p-value helpers degrade to None


# ──────────────────────────── Statistics ─────────────────────────────────────


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 100.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin) * 100.0, min(1.0, center + margin) * 100.0)


def two_prop_pvalue(k_a: int, n_a: int, k_b: int, n_b: int) -> float | None:
    """Two-sided test on two proportions. Fisher's exact for N<40, else
    pooled-proportion normal approximation. Returns None if scipy unavailable
    or either arm has zero trials."""
    if n_a == 0 or n_b == 0:
        return None
    if stats is None:
        return None
    if n_a + n_b < 40:
        table = [[k_a, n_a - k_a], [k_b, n_b - k_b]]
        _, p = stats.fisher_exact(table, alternative="two-sided")
        return float(p)
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (k_a / n_a - k_b / n_b) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


# ──────────────────────────── Human adversary ────────────────────────────────


def _classify_violation(row: dict) -> int:
    return 1 if row.get("honeypots_accessed_all") else 0


def load_human_cell(audit_dir: Path, base_tag: str, defender: str) -> dict | None:
    """Reads all seed*.jsonl under <audit_dir>/<base_tag>_<defender>_raw/.
    Returns per-cell counts suitable for PVR_conv + CI + p-value.
    For single-turn human prompts PVR_turn == PVR_conv (one blue response
    per prompt)."""
    cell_dir = audit_dir / f"{base_tag}_{defender}_raw"
    if not cell_dir.is_dir():
        return None
    seeds = sorted(cell_dir.glob("seed*.jsonl"))
    if not seeds:
        return None

    n_total = 0
    n_violated = 0
    n_refused = 0
    for jl in seeds:
        with open(jl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n_total += 1
                if _classify_violation(row):
                    n_violated += 1
                if row.get("is_refusal") and not row.get("honeypots_accessed_all"):
                    n_refused += 1

    pvr_conv = (n_violated / n_total * 100.0) if n_total else 0.0
    pvr_conv_ci = wilson_ci(n_violated, n_total)
    refusal_rate = (n_refused / n_total * 100.0) if n_total else 0.0
    return {
        "adversary": "human",
        "defender": defender,
        "n_trials": n_total,
        "n_violated": n_violated,
        "n_refused": n_refused,
        "num_seeds": len(seeds),
        "pvr_conv": round(pvr_conv, 2),
        "pvr_conv_ci": [round(pvr_conv_ci[0], 2), round(pvr_conv_ci[1], 2)],
        "pvr_turn": round(pvr_conv, 2),
        "pvr_turn_ci": [round(pvr_conv_ci[0], 2), round(pvr_conv_ci[1], 2)],
        "refusal_rate": round(refusal_rate, 2),
        "neutral_sql_rate": None,
    }


# ──────────────────────────── RL-red adversary (cross-eval) ──────────────────


def _load_pairing_records(pairing_dir: Path) -> list[dict]:
    jsonl = pairing_dir / "reward_debug.jsonl"
    if not jsonl.exists():
        return []
    out = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_rl_cell(
    cross_eval_dir: Path,
    red_iter: int,
    blue_iter: int,
    defender_label: str,
) -> dict | None:
    pairing_key = f"red_{red_iter}_blue_{blue_iter}"
    pairing_dir = cross_eval_dir / "pairings" / pairing_key
    records = _load_pairing_records(pairing_dir)
    if not records:
        return None

    attack_steps = [r for r in records if r["turn_type"] == "attack"]
    n_attack_steps = len(attack_steps)
    n_fp_steps = sum(1 for r in attack_steps if r["outcome_tier"] == "false_positive")
    n_neutral_steps = sum(1 for r in attack_steps if r["outcome_tier"] == "neutral_sql")

    # Group by episode for PVR_conv.
    by_ep: dict[int, list[dict]] = {}
    for r in attack_steps:
        by_ep.setdefault(r["episode"], []).append(r)
    n_attack_eps = len(by_ep)
    n_violated_eps = sum(
        1 for steps in by_ep.values()
        if any(s["outcome_tier"] == "false_positive" for s in steps)
    )

    pvr_conv = (n_violated_eps / n_attack_eps * 100.0) if n_attack_eps else 0.0
    pvr_turn = (n_fp_steps / n_attack_steps * 100.0) if n_attack_steps else 0.0
    neutral = (n_neutral_steps / n_attack_steps * 100.0) if n_attack_steps else 0.0
    pvr_conv_ci = wilson_ci(n_violated_eps, n_attack_eps)
    pvr_turn_ci = wilson_ci(n_fp_steps, n_attack_steps)

    return {
        "adversary": "rl_red",
        "defender": defender_label,
        "pairing_key": pairing_key,
        "n_attack_episodes": n_attack_eps,
        "n_violated_episodes": n_violated_eps,
        "n_attack_steps": n_attack_steps,
        "n_violated_steps": n_fp_steps,
        "n_neutral_steps": n_neutral_steps,
        # "n_trials" is the episode count — used for manual-vs-RL p-value.
        "n_trials": n_attack_eps,
        "n_violated": n_violated_eps,
        "pvr_conv": round(pvr_conv, 2),
        "pvr_conv_ci": [round(pvr_conv_ci[0], 2), round(pvr_conv_ci[1], 2)],
        "pvr_turn": round(pvr_turn, 2),
        "pvr_turn_ci": [round(pvr_turn_ci[0], 2), round(pvr_turn_ci[1], 2)],
        "neutral_sql_rate": round(neutral, 2),
    }


def load_pud(cross_eval_dir: Path, blue_iter: int) -> dict | None:
    """PUD = 100 - TPR, read from <cross_eval_dir>/benign_only/blue_<i>/summary.json."""
    summary = cross_eval_dir / "benign_only" / f"blue_{blue_iter}" / "summary.json"
    if not summary.exists():
        return None
    with open(summary) as f:
        s = json.load(f)
    n = int(s.get("n_episodes", 0))
    tpr = float(s.get("tpr", 0.0))
    # Recover counts: round back from TPR to an integer success count.
    n_tp = int(round(tpr / 100.0 * n)) if n else 0
    pud_ci_lo = 100.0 - wilson_ci(n_tp, n)[1]
    pud_ci_hi = 100.0 - wilson_ci(n_tp, n)[0]
    return {
        "blue_iter": blue_iter,
        "n_benign": n,
        "n_tp": n_tp,
        "pud": round(100.0 - tpr, 2),
        "pud_ci": [round(pud_ci_lo, 2), round(pud_ci_hi, 2)],
    }


# ──────────────────────────── F1 assembly ────────────────────────────────────


def f1_score(pvr_turn: float, pud: float) -> float:
    """HM(1 - PVR_turn, 1 - PUD) on the [0, 100] scale, to match
    cross_evaluate.compute_pairing_metrics."""
    a, b = 100.0 - pvr_turn, 100.0 - pud
    return round(2 * a * b / (a + b), 2) if (a + b) > 0 else 0.0


# ──────────────────────────── LaTeX emission ─────────────────────────────────


def _fmt_ci(mean: float, ci: list[float] | None) -> str:
    if ci is None:
        return f"{mean:.1f}"
    return f"{mean:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "--"
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def emit_tex(rows: list[dict], out_path: Path) -> None:
    """One row per (base_model, defender, adversary) cell. p-values attach
    to the RL-defender row for each (base_model, adversary) — i.e. "did
    switching the manual defender to the RL one shift the violation rate?"
    """
    lines = []
    lines.append("% Auto-generated by util/eval_rl_vs_human.py — do not edit by hand.")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Base model & Cell & PVR$_{\text{conv}}$ & PVR$_{\text{turn}}$ & "
        r"PUD & F1 & Neutral-SQL & $p$ (vs manual) \\"
    )
    lines.append(r"\midrule")
    last_base = None
    for row in rows:
        base = row["base_tag"] if row["base_tag"] != last_base else ""
        last_base = row["base_tag"]
        cell = f"{row['defender']}, {row['adversary']}"
        pvr_c = _fmt_ci(row["pvr_conv"], row.get("pvr_conv_ci"))
        pvr_t = _fmt_ci(row["pvr_turn"], row.get("pvr_turn_ci")) if row.get("pvr_turn") is not None else "--"
        pud_s = _fmt_ci(row["pud"], row.get("pud_ci")) if row.get("pud") is not None else "--"
        f1_s = f"{row['f1']:.1f}" if row.get("f1") is not None else "--"
        neutral = f"{row['neutral_sql_rate']:.1f}" if row.get("neutral_sql_rate") is not None else "--"
        pval = _fmt_p(row.get("p_vs_manual"))
        lines.append(
            f"{base} & {cell} & {pvr_c} & {pvr_t} & {pud_s} & {f1_s} & {neutral} & {pval} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {out_path}")


# ──────────────────────────── Per-base assembly ──────────────────────────────


def assemble_base(
    base_cfg: dict,
    audit_dir: Path,
    out_json_dir: Path,
) -> list[dict]:
    """Returns a flat list of row dicts for `emit_tex`, one per
    (defender, adversary) cell."""
    base_tag = base_cfg["base_tag"]
    cross_eval_dir = Path(base_cfg["cross_eval_dir"]) if base_cfg.get("cross_eval_dir") else None
    red_iter = int(base_cfg.get("rl_red_iter", 0))
    blue_manual_iter = int(base_cfg.get("manual_blue_iter", 0))
    blue_rl_iter = int(base_cfg.get("rl_blue_iter", 0))

    cells = []

    # Human adversary cells.
    for defender in ("manual", "rl"):
        cell = load_human_cell(audit_dir, base_tag, defender)
        if cell is None:
            print(f"[warn] missing human audit cell: {base_tag}_{defender}")
            continue
        cells.append(cell)

    # RL-red adversary cells.
    if cross_eval_dir is not None:
        for defender, b_iter in (("manual", blue_manual_iter), ("rl", blue_rl_iter)):
            cell = load_rl_cell(cross_eval_dir, red_iter, b_iter, defender)
            if cell is None:
                print(f"[warn] missing cross_eval pairing red_{red_iter}_blue_{b_iter}")
                continue
            cells.append(cell)

    # PUD per defender (shared across adversaries).
    pud_by_defender: dict[str, dict | None] = {"manual": None, "rl": None}
    if cross_eval_dir is not None:
        pud_by_defender["manual"] = load_pud(cross_eval_dir, blue_manual_iter)
        pud_by_defender["rl"] = load_pud(cross_eval_dir, blue_rl_iter)

    # Attach PUD + F1 to every cell; compute manual-vs-RL p-values.
    by_adv: dict[str, dict[str, dict]] = {}
    for cell in cells:
        pud_info = pud_by_defender.get(cell["defender"])
        if pud_info is not None:
            cell["pud"] = pud_info["pud"]
            cell["pud_ci"] = pud_info["pud_ci"]
            cell["f1"] = f1_score(cell["pvr_turn"], cell["pud"])
        else:
            cell["pud"] = None
            cell["pud_ci"] = None
            cell["f1"] = None
        by_adv.setdefault(cell["adversary"], {})[cell["defender"]] = cell

    for adversary, pair in by_adv.items():
        manual = pair.get("manual")
        rl = pair.get("rl")
        if manual and rl:
            p = two_prop_pvalue(
                manual["n_violated"], manual["n_trials"],
                rl["n_violated"], rl["n_trials"],
            )
            rl["p_vs_manual"] = p

    # Dump per-base JSON.
    out_json_dir.mkdir(parents=True, exist_ok=True)
    dump = {
        "base_tag": base_tag,
        "base_model": base_cfg.get("base_model"),
        "rl_red_iter": red_iter,
        "manual_blue_iter": blue_manual_iter,
        "rl_blue_iter": blue_rl_iter,
        "cross_eval_dir": str(cross_eval_dir) if cross_eval_dir else None,
        "cells": cells,
    }
    with open(out_json_dir / f"{base_tag}.json", "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[ok] wrote {out_json_dir / (base_tag + '.json')}")

    # Return rows in stable (defender, adversary) order for the table.
    order = [("manual", "human"), ("rl", "human"), ("manual", "rl_red"), ("rl", "rl_red")]
    rows = []
    for defender, adversary in order:
        cell = by_adv.get(adversary, {}).get(defender)
        if cell is not None:
            cell = {**cell, "base_tag": base_tag}
            rows.append(cell)
    return rows


# ──────────────────────────── Main ──────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="JSON file listing base models + their cross-eval dirs "
                         "and iter indices. Shape: {\"bases\": [ {...}, ... ]}.")
    ap.add_argument("--audit-dir", default="data/human_attack_audit")
    ap.add_argument("--out-json-dir", default="data/rl_vs_human")
    ap.add_argument("--out-tex", default="figures/rl_vs_human/rl_vs_human_table.tex")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    bases = cfg.get("bases", [])
    if not bases:
        print(f"[err] {args.config} has no 'bases'; nothing to do.", file=sys.stderr)
        return 1

    audit_dir = Path(args.audit_dir)
    out_json_dir = Path(args.out_json_dir)

    all_rows: list[dict] = []
    for base_cfg in bases:
        rows = assemble_base(base_cfg, audit_dir, out_json_dir)
        all_rows.extend(rows)

    if not all_rows:
        print("[err] no rows assembled; refusing to emit empty table.", file=sys.stderr)
        return 1

    emit_tex(all_rows, Path(args.out_tex))
    return 0


if __name__ == "__main__":
    sys.exit(main())
