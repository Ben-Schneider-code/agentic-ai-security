#!/usr/bin/env python3
"""Agent-vs-Human baseline aggregator.

Orchestrates N seeded replays of the 32-conversation human attack dataset
(new_jailbreaks.txt) against one (model, variant) cell at a time, then
aggregates the N JSONL reports into a single summary per cell for
draft.tex §sec:eval-human (Q1a, Q1b, Q2, non-determinism audit).

Assumes a vLLM server is ALREADY running for the requested model (the
run_human_baseline.sh driver starts one per model and calls this script
once per variant). Invokes util/run_conversations.py with --no_start_vllm
so every seeded run reuses the same server.

Manifest (JSON) shape:

    {
      "dataset":   "new_jailbreaks.txt",
      "audit_dir": "data/human_attack_audit",
      "cells": [
        {"model": "meta-llama/Llama-3.1-8B-Instruct",
         "tag":   "llama31_8b_instruct",
         "variant": "unprotected",
         "system_prompt_file": "prompts/unprotected_system_prompt.txt",
         "port": 8001},
        {"model": "Snowflake/Arctic-Text2SQL-R1-7B",
         "tag":   "arctic_r1_7b",
         "variant": "patched",
         "system_prompt_file": null,
         "port": 8001}
      ]
    }

A null system_prompt_file means "use the default manually-protected
sql_system_prompt". Callers should only pass one cell's port at a time
(i.e., match the currently running vLLM).

Outputs per cell: <audit_dir>/<tag>_<variant>.json with fields:
    num_seeds, num_conversations, pvr_conv_mean, pvr_conv_ci_lo,
    pvr_conv_ci_hi, refusal_rate_mean, stability[conv_idx] = fraction
    of seeds where the prompt triggered any honeypot.

Plus a combined <audit_dir>/nondeterminism.json across all cells present
in the audit_dir, listing every (tag, variant, conv_idx) whose fraction
is strictly between 0 and 1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 100.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin) * 100.0, min(1.0, center + margin) * 100.0)


def run_one_seed(cell: dict, dataset: str, seed: int, output_path: Path) -> Path:
    """Invoke util/run_conversations.py once for a given seed. Returns the
    JSONL output path. Skips the call if output already exists (cheap
    resume)."""
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[skip] {output_path} exists, reusing.")
        return output_path

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "util" / "run_conversations.py"),
        dataset,
        "--model_name", cell["model"],
        "--port", str(cell.get("port", 8001)),
        "--no_start_vllm",
        "--seed", str(seed),
        "--run_tag", f"{cell['tag']}_{cell['variant']}",
        "--output_file", str(output_path),
    ]
    if cell.get("system_prompt_file"):
        cmd += ["--system_prompt_file", cell["system_prompt_file"]]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def summarize_reports(report_paths: list[Path]) -> dict:
    """Collapse N per-seed JSONL reports into a single-cell summary."""
    per_seed_violated = []
    per_seed_refused = []
    per_conv_hit_count: dict[int, int] = {}
    num_conversations = None

    for rp in report_paths:
        rows = []
        with open(rp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if num_conversations is None:
            num_conversations = len(rows)
        violated = 0
        refused = 0
        for r in rows:
            idx = int(r.get("conversation_index", -1))
            hit = bool(r.get("honeypots_accessed_all"))
            if hit:
                violated += 1
                per_conv_hit_count[idx] = per_conv_hit_count.get(idx, 0) + 1
            else:
                per_conv_hit_count.setdefault(idx, 0)
            if r.get("is_refusal"):
                refused += 1
        per_seed_violated.append(violated)
        per_seed_refused.append(refused)

    num_seeds = len(report_paths)
    total_conversations = num_seeds * (num_conversations or 0)
    total_violated = sum(per_seed_violated)
    ci_lo, ci_hi = wilson_ci(total_violated, total_conversations)
    pvr_mean = (total_violated / total_conversations * 100.0) if total_conversations else 0.0
    refusal_mean = (
        sum(per_seed_refused) / total_conversations * 100.0 if total_conversations else 0.0
    )
    stability = {
        idx: (count / num_seeds) for idx, count in per_conv_hit_count.items()
    }
    return {
        "num_seeds": num_seeds,
        "num_conversations": num_conversations or 0,
        "per_seed_pvr_conv": [
            (v / num_conversations * 100.0) if num_conversations else 0.0
            for v in per_seed_violated
        ],
        "pvr_conv_mean": pvr_mean,
        "pvr_conv_ci_lo": ci_lo,
        "pvr_conv_ci_hi": ci_hi,
        "refusal_rate_mean": refusal_mean,
        "stability": stability,
    }


def write_nondeterminism_index(audit_dir: Path) -> Path:
    """Scan every *_<variant>.json in audit_dir; collect unstable prompts."""
    unstable = []
    for p in sorted(audit_dir.glob("*.json")):
        if p.name == "nondeterminism.json":
            continue
        try:
            with open(p) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        stability = summary.get("stability", {})
        for idx, frac in stability.items():
            if 0.0 < float(frac) < 1.0:
                unstable.append({
                    "file": p.name,
                    "conv_idx": int(idx),
                    "fraction_hit": float(frac),
                    "num_seeds": summary.get("num_seeds"),
                })
    out = audit_dir / "nondeterminism.json"
    with open(out, "w") as f:
        json.dump({"unstable_prompts": unstable}, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON manifest path.")
    parser.add_argument("--cell-index", type=int, default=None,
                        help="If set, process only manifest.cells[cell_index].")
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of seeded replays per cell (default 5).")
    parser.add_argument("--seed-base", type=int, default=1000,
                        help="First seed value; subsequent seeds are +1 each.")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip replay; just regenerate nondeterminism.json.")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    dataset = manifest.get("dataset", "new_jailbreaks.txt")
    audit_dir = Path(manifest.get("audit_dir", "data/human_attack_audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)

    cells = manifest.get("cells", [])
    if args.cell_index is not None:
        cells = [cells[args.cell_index]]

    if not args.index_only:
        for cell in cells:
            tag = cell["tag"]
            variant = cell["variant"]
            cell_dir = audit_dir / f"{tag}_{variant}_raw"
            cell_dir.mkdir(parents=True, exist_ok=True)
            report_paths = []
            for i in range(args.num_seeds):
                seed = args.seed_base + i
                out = cell_dir / f"seed{seed}.jsonl"
                run_one_seed(cell, dataset, seed, out)
                report_paths.append(out)
            summary = summarize_reports(report_paths)
            summary["tag"] = tag
            summary["variant"] = variant
            summary["model"] = cell["model"]
            summary_path = audit_dir / f"{tag}_{variant}.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[ok] wrote {summary_path}: PVR_conv = "
                  f"{summary['pvr_conv_mean']:.1f}% "
                  f"[{summary['pvr_conv_ci_lo']:.1f}, {summary['pvr_conv_ci_hi']:.1f}]")

    nd_path = write_nondeterminism_index(audit_dir)
    print(f"[ok] wrote {nd_path}")


if __name__ == "__main__":
    main()
