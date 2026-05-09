#!/usr/bin/env python3
"""Per-style PUD across self-play iterations (post-hoc).

``plot_blueteam_results.py`` already exposes per-style benign refusal
series (plain / adversarial / multi_turn) within a single iteration.
The paper's §sec:per-style additionally claims a per-iteration trend.
This script adds the missing across-iteration roll-up without touching
training code: it reads each ``iter_*/blueteam/.../reward_debug.jsonl``,
computes the per-style benign denial rate (false-negative / refusal
tier) aggregated over the iteration's training-phase benign turns, and
plots the three series against self-play iteration index.

Notes:
    The ``benign_style`` field was added to the training log partway
    through the project; older iterations that predate the logging have
    style=None and are reported in an ``unlabeled`` bucket.

Run:
    python util/per_style_pud_trend.py <results_dir>

Outputs (in ``<results_dir>/per_style_pud/``):
    trend.csv
    trend.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLES = ["plain", "adversarial", "multi_turn", "unlabeled"]
COLORS = {
    "plain": "#2980b9",
    "adversarial": "#c0392b",
    "multi_turn": "#27ae60",
    "unlabeled": "#888",
}


def find_iteration_logs(results_dir: Path) -> dict[int, Path]:
    iters: dict[int, Path] = {}
    for iter_dir in sorted(results_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        try:
            k = int(iter_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        candidates = list(iter_dir.glob("blueteam/**/debug_logs/reward_debug.jsonl"))
        if candidates:
            iters[k] = candidates[0]
    return iters


def denial_rates_by_style(jsonl_path: Path) -> dict[str, tuple[int, int]]:
    """Return {style: (n_refused, n_total)} across benign training turns."""
    counts = defaultdict(lambda: [0, 0])  # [refused, total]
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("turn_type") != "benign":
                continue
            if r.get("is_eval"):
                continue
            style = r.get("benign_style") or "unlabeled"
            if style not in STYLES:
                style = "unlabeled"
            counts[style][1] += 1
            if r.get("is_refusal") or r.get("outcome_tier") == "false_negative":
                counts[style][0] += 1
    return {s: (c[0], c[1]) for s, c in counts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    logs = find_iteration_logs(args.results_dir)
    if not logs:
        print(f"No iter_*/blueteam debug logs found under {args.results_dir}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (args.results_dir / "per_style_pud")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, path in logs.items():
        counts = denial_rates_by_style(path)
        rows.append({
            "iter": k,
            **{
                f"{s}_n": counts.get(s, (0, 0))[1] for s in STYLES
            },
            **{
                f"{s}_pud": (counts.get(s, (0, 0))[0] / counts[s][1])
                if counts.get(s, (0, 0))[1] > 0 else float("nan")
                for s in STYLES
            },
        })

    csv_path = out_dir / "trend.csv"
    with csv_path.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    iters = [r["iter"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in STYLES:
        ys = [r[f"{s}_pud"] for r in rows]
        if all(y != y for y in ys):  # all NaN
            continue
        ax.plot(iters, ys, "o-", color=COLORS[s], label=s, lw=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Benign denial rate (PUD proxy)")
    ax.set_title("Per-style PUD across self-play iterations (training benign turns)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "trend.png", dpi=120)
    plt.close(fig)

    print(f"Wrote {csv_path} and trend.png")
    for r in rows:
        parts = []
        for s in STYLES:
            n = r[f"{s}_n"]
            pud = r[f"{s}_pud"]
            if n > 0:
                parts.append(f"{s}={pud:.2f} (n={n})")
        print(f"  iter {r['iter']:>2}: " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
