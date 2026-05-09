#!/usr/bin/env python3
"""
Aggregate human-baseline summary.json files for the three defender configurations
and emit a paper-ready comparison table.

Usage:
  python3 util/aggregate_human_baseline.py \
      --tags human_unprotected human_manual human_rl_iter1 \
      --labels "Unprotected" "Manually Protected" "RL Protected" \
      --human-eval-dir data/human_eval \
      --output-dir data/human_eval

Outputs:
  <output-dir>/comparison.json   — machine-readable metrics per config
  <output-dir>/comparison.tex    — LaTeX tabular (3 data rows)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def pct(v: float) -> str:
    return f"{v * 100:.1f}"


def wf_str(s: dict) -> str:
    wf = s.get("WF")
    wf_lower = s.get("WF_lower")
    if wf is not None:
        return f"{wf:.1f}"
    if wf_lower is not None:
        return f">{wf_lower:.1f}"
    return r"$\infty$"


def fmt_metric(val: float, ci: list[float]) -> str:
    """Format as '12.3 (9.1, 16.4)' in percent."""
    return f"{pct(val)} ({pct(ci[0])}, {pct(ci[1])})"


def load_summary(human_eval_dir: Path, tag: str) -> dict:
    path = human_eval_dir / tag / "summary.json"
    if not path.exists():
        print(f"ERROR: Missing {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def sanity_check(summaries: list[dict], tags: list[str]) -> None:
    ref = summaries[0]
    for key in ("n_attack_prompts", "n_benign_turns", "num_seeds"):
        vals = [s[key] for s in summaries]
        if len(set(vals)) != 1:
            mapping = dict(zip(tags, vals))
            print(
                f"ERROR: Inconsistent '{key}' across configs: {mapping}. "
                "The three runs must use the same attack set, benign set, and seed count.",
                file=sys.stderr,
            )
            sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs=3,
                   default=["human_unprotected", "human_manual", "human_rl_iter1"],
                   metavar=("UNPROTECTED", "MANUAL", "RL"),
                   help="Output-dir tags in order: unprotected, manual, rl (default: human_*)")
    p.add_argument("--labels", nargs=3,
                   default=["Unprotected", "Manually Protected", "RL Protected"],
                   metavar=("LABEL1", "LABEL2", "LABEL3"))
    p.add_argument("--human-eval-dir", default="data/human_eval",
                   help="Directory containing per-tag subdirs (default: data/human_eval)")
    p.add_argument("--output-dir", default="data/human_eval",
                   help="Where to write comparison.{json,tex} (default: data/human_eval)")
    args = p.parse_args()

    human_eval_dir = Path(args.human_eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [load_summary(human_eval_dir, tag) for tag in args.tags]
    sanity_check(summaries, args.tags)

    # ── Monotonicity check ──────────────────────────────────────────────────
    pvrs = [s["PVR_turn"] for s in summaries]
    if not (pvrs[0] >= pvrs[1] >= pvrs[2]):
        print(
            f"WARNING: PVR_turn does not strictly decrease across configs: "
            f"{[f'{v*100:.1f}%' for v in pvrs]}. "
            "Check that the tags are in the right order and that the RL run used the correct LoRA.",
            file=sys.stderr,
        )

    # ── Build machine-readable output ───────────────────────────────────────
    rows = []
    for tag, label, s in zip(args.tags, args.labels, summaries):
        rows.append({
            "tag": tag,
            "label": label,
            "model_id": s.get("model_id"),
            "adapter_path": s.get("adapter_path"),
            "system_prompt_path": s.get("system_prompt_path"),
            "num_seeds": s["num_seeds"],
            "n_attack_prompts": s["n_attack_prompts"],
            "n_benign_turns": s["n_benign_turns"],
            "PVR_turn": s["PVR_turn"],
            "PVR_turn_ci": s["PVR_turn_ci"],
            "PVR_conv": s["PVR_conv"],
            "PVR_conv_ci": s["PVR_conv_ci"],
            "BRR": s["BRR"],
            "BRR_ci": s["BRR_ci"],
            "WF": s.get("WF"),
            "WF_lower": s.get("WF_lower"),
            "WF_upper": s.get("WF_upper"),
        })

    comparison_json = output_dir / "comparison.json"
    with open(comparison_json, "w") as f:
        json.dump(rows, f, indent=2)

    # ── Build LaTeX table ───────────────────────────────────────────────────
    tex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{PVR\textsubscript{turn} (\%)} & \textbf{PVR\textsubscript{conv} (\%)} & \textbf{BRR (\%)} & \textbf{WF} \\",
        r"\midrule",
    ]
    for row in rows:
        label = row["label"]
        pvr_t = fmt_metric(row["PVR_turn"], row["PVR_turn_ci"])
        pvr_c = fmt_metric(row["PVR_conv"], row["PVR_conv_ci"])
        brr = fmt_metric(row["BRR"], row["BRR_ci"])
        wf = wf_str(row)
        tex_lines.append(f"  {label} & {pvr_t} & {pvr_c} & {brr} & {wf} \\\\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex_body = "\n".join(tex_lines)

    comparison_tex = output_dir / "comparison.tex"
    with open(comparison_tex, "w") as f:
        f.write(tex_body + "\n")

    # ── Stdout summary ──────────────────────────────────────────────────────
    col_w = max(len(r["label"]) for r in rows) + 2
    header = f"{'Config':<{col_w}} {'PVR_turn':>10} {'PVR_conv':>10} {'BRR':>8} {'WF':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        wf = wf_str(row)
        print(
            f"{row['label']:<{col_w}} "
            f"{pct(row['PVR_turn']):>9}% "
            f"{pct(row['PVR_conv']):>9}% "
            f"{pct(row['BRR']):>7}% "
            f"{wf:>8}"
        )
    print("=" * len(header))
    print(f"\nWrote: {comparison_json}")
    print(f"Wrote: {comparison_tex}")


if __name__ == "__main__":
    main()
