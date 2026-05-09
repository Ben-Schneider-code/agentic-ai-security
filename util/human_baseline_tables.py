#!/usr/bin/env python3
"""Emit the three LaTeX tabulars that draft.tex §sec:eval-human expects.

Consumes aggregate JSON files written by util/jailbreak_baseline.py under
<audit-dir>/<tag>_<variant>.json plus the raw seed jsonl reports under
<audit-dir>/<tag>_<variant>_raw/ (the jailbreak_baseline driver already
produces both per-cell). PUD numbers come from a user-supplied PUD map
(--pud-json), since PUD is computed from benign-only passes in a cross-
evaluation run rather than from the human-attack audit itself.

Emits into --out-dir:
  q1a_unprotected_llama31.tex     # 32-row per-prompt table
  q1b_transfer_unprotected.tex    # 4-row (model × PVR_conv, refusal, PUD)
  q2_patched.tex                  # 4-row patched defender (+ F1)

Strategy categories and per-prompt target labels are parsed from
new_jailbreaks.txt headers (`# --- N. CATEGORY ---` / `# Target: ...`).

Usage:
    python util/human_baseline_tables.py \\
        --audit-dir data/human_attack_audit \\
        --dataset new_jailbreaks.txt \\
        --config configs/human_baseline_tables.json \\
        --out-dir figures/human_baseline
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


# ──────────────────────────── Dataset metadata parser ───────────────────────


def parse_prompt_metadata(dataset_path: Path) -> list[dict]:
    """Returns [{index, strategy, target, prompt}] in block order."""
    text = dataset_path.read_text()
    lines = text.split("\n")

    entries: list[dict] = []
    current_strategy = "unclassified"
    current_target = None
    pending_block: list[str] = []

    strategy_re = re.compile(r"^#\s*---\s*(?:\d+\.\s*)?(.+?)\s*---\s*$")
    target_re = re.compile(r"^#\s*Target\s*:\s*(.+?)\s*$", re.IGNORECASE)

    def flush():
        if pending_block:
            prompt = "\n".join(pending_block).strip()
            if prompt:
                entries.append({
                    "index": len(entries),
                    "strategy": current_strategy,
                    "target": current_target or "",
                    "prompt": prompt,
                })

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            pending_block = []
            current_target = None
            continue
        if stripped.startswith("#"):
            m = strategy_re.match(stripped)
            if m:
                current_strategy = m.group(1).strip()
                continue
            m = target_re.match(stripped)
            if m:
                current_target = m.group(1).strip()
                continue
            # any other comment line — skip
            continue
        pending_block.append(line)

    flush()
    return entries


# ──────────────────────────── Aggregate loaders ──────────────────────────────


def load_aggregate(audit_dir: Path, tag: str, variant: str) -> dict | None:
    path = audit_dir / f"{tag}_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_raw_rows(audit_dir: Path, tag: str, variant: str) -> list[list[dict]]:
    """Returns per-seed list of rows for the given cell, or [] if missing."""
    cell_dir = audit_dir / f"{tag}_{variant}_raw"
    out = []
    if not cell_dir.is_dir():
        return out
    for jl in sorted(cell_dir.glob("seed*.jsonl")):
        rows = []
        with open(jl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out.append(rows)
    return out


# ──────────────────────────── Violation classifier (per prompt) ─────────────


def per_prompt_violation_fraction(reports: list[list[dict]]) -> dict[int, dict]:
    """Returns {conv_idx: {n_seeds, n_hit, fraction, first_violation_type}}.

    `first_violation_type` is taken from the (type, identifier) of the
    first honeypot observed for that prompt across seeds — useful for
    the Q1a "violation type" column. Empty string if never violated.
    """
    by_conv: dict[int, dict] = {}
    for rows in reports:
        for r in rows:
            idx = int(r.get("conversation_index", -1))
            if idx < 0:
                continue
            slot = by_conv.setdefault(idx, {"n_seeds": 0, "n_hit": 0, "violation_type": ""})
            slot["n_seeds"] += 1
            accessed = r.get("honeypots_accessed_all") or []
            if accessed:
                slot["n_hit"] += 1
                if not slot["violation_type"]:
                    first = accessed[0]
                    if isinstance(first, dict):
                        slot["violation_type"] = f"{first.get('type', '?')}:{first.get('identifier', '?')}"
                    else:
                        slot["violation_type"] = str(first)
    for slot in by_conv.values():
        slot["fraction"] = slot["n_hit"] / slot["n_seeds"] if slot["n_seeds"] else 0.0
    return by_conv


# ──────────────────────────── Metric helpers ─────────────────────────────────


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 100.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin) * 100.0, min(1.0, center + margin) * 100.0)


def f1_from(pvr_conv: float, pud: float) -> float:
    a, b = 100.0 - pvr_conv, 100.0 - pud
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.1f} [{lo:.1f}, {hi:.1f}]"


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("_", r"\_")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


# ──────────────────────────── Table emitters ─────────────────────────────────


def emit_q1a(
    meta: list[dict],
    stability: dict[int, dict],
    out_path: Path,
    title_tag: str,
) -> None:
    lines = [
        f"% Auto-generated by util/human_baseline_tables.py ({title_tag}). Do not edit.",
        r"\begin{tabular}{rllll}",
        r"\toprule",
        r"ID & Strategy & Target & Violation type & Stability \\",
        r"\midrule",
    ]
    for entry in meta:
        idx = entry["index"]
        stab = stability.get(idx, {"n_seeds": 0, "n_hit": 0, "fraction": 0.0, "violation_type": ""})
        stab_str = f"{stab['n_hit']}/{stab['n_seeds']}" if stab["n_seeds"] else "--"
        lines.append(
            f"{idx} & {latex_escape(entry['strategy'])} & "
            f"{latex_escape(entry['target'])} & "
            f"{latex_escape(stab['violation_type'] or '--')} & {stab_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {out_path}")


def emit_transfer_row(
    agg: dict | None,
    pud_entry: dict | None,
    include_f1: bool,
) -> tuple[str, str, str, str | None]:
    if agg is None:
        pvr = "--"
        refusal = "--"
        pvr_conv_val = None
    else:
        pvr_conv_val = float(agg.get("pvr_conv_mean", 0.0))
        pvr = fmt_ci(
            pvr_conv_val,
            float(agg.get("pvr_conv_ci_lo", 0.0)),
            float(agg.get("pvr_conv_ci_hi", 0.0)),
        )
        refusal = f"{float(agg.get('refusal_rate_mean', 0.0)):.1f}"

    if pud_entry is None:
        pud_str = "--"
        pud_val = None
    else:
        pud_val = float(pud_entry["pud"])
        ci = pud_entry.get("pud_ci") or [pud_val, pud_val]
        pud_str = fmt_ci(pud_val, float(ci[0]), float(ci[1]))

    if not include_f1:
        return pvr, refusal, pud_str, None
    if pvr_conv_val is None or pud_val is None:
        return pvr, refusal, pud_str, "--"
    return pvr, refusal, pud_str, f"{f1_from(pvr_conv_val, pud_val):.1f}"


def emit_q1b(
    base_rows: list[dict],
    audit_dir: Path,
    pud_map: dict,
    out_path: Path,
) -> None:
    lines = [
        "% Auto-generated by util/human_baseline_tables.py (Q1b — unprotected transfer).",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Base model & PVR$_{\text{conv}}$ & Refusal rate & PUD \\",
        r"\midrule",
    ]
    for base in base_rows:
        tag = base["base_tag"]
        label = base.get("label", tag)
        agg = load_aggregate(audit_dir, tag, "unprotected")
        pud_entry = pud_map.get(tag, {}).get("unprotected")
        pvr, refusal, pud_str, _ = emit_transfer_row(agg, pud_entry, include_f1=False)
        lines.append(f"{latex_escape(label)} & {pvr} & {refusal} & {pud_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {out_path}")


def emit_q2(
    base_rows: list[dict],
    audit_dir: Path,
    pud_map: dict,
    out_path: Path,
) -> None:
    lines = [
        "% Auto-generated by util/human_baseline_tables.py (Q2 — patched defender).",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Base model & PVR$_{\text{conv}}$ & Refusal rate & PUD & F1 \\",
        r"\midrule",
    ]
    for base in base_rows:
        tag = base["base_tag"]
        label = base.get("label", tag)
        agg = load_aggregate(audit_dir, tag, "patched")
        pud_entry = pud_map.get(tag, {}).get("patched")
        pvr, refusal, pud_str, f1_str = emit_transfer_row(agg, pud_entry, include_f1=True)
        lines.append(
            f"{latex_escape(label)} & {pvr} & {refusal} & {pud_str} & {f1_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {out_path}")


# ──────────────────────────── Main ──────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", default="data/human_attack_audit")
    ap.add_argument("--dataset", default="new_jailbreaks.txt")
    ap.add_argument("--config", required=True,
                    help="JSON config with keys: bases=[{base_tag,label}], "
                         "q1a_focus_tag (default llama31_8b_instruct), "
                         "pud_map={tag:{variant:{pud,pud_ci}}}.")
    ap.add_argument("--out-dir", default="figures/human_baseline")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[err] --config {cfg_path} does not exist", file=sys.stderr)
        return 1
    with open(cfg_path) as f:
        cfg = json.load(f)
    bases = cfg.get("bases", [])
    q1a_focus = cfg.get("q1a_focus_tag", "llama31_8b_instruct")
    pud_map = cfg.get("pud_map", {})

    audit_dir = Path(args.audit_dir)
    out_dir = Path(args.out_dir)

    meta = parse_prompt_metadata(Path(args.dataset))
    print(f"[info] parsed {len(meta)} prompts from {args.dataset}")

    # Q1a: per-prompt table for the unprotected focus model.
    raw = load_raw_rows(audit_dir, q1a_focus, "unprotected")
    if not raw:
        print(f"[warn] no raw reports for {q1a_focus}_unprotected; Q1a table will mark every row '--/--'")
    stability = per_prompt_violation_fraction(raw)
    emit_q1a(meta, stability, out_dir / "q1a_unprotected_llama31.tex",
             title_tag=f"Q1a — {q1a_focus} unprotected")

    # Q1b + Q2: transfer across base models.
    if not bases:
        print("[warn] config has no 'bases'; skipping Q1b/Q2 tables.")
        return 0
    emit_q1b(bases, audit_dir, pud_map, out_dir / "q1b_transfer_unprotected.tex")
    emit_q2(bases, audit_dir, pud_map, out_dir / "q2_patched.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
