#!/usr/bin/env python3
"""Detect blanket-refusal collapse during blue-team training.

A ``blanket-refusal'' collapse is the failure mode where $\\mathcal{A}$
reduces $\\mathrm{PVR}$ trivially by denying \\emph{every} request,
pushing PUD to 1. The paper's §sec:diagnostics promises an early-warning
log entry when the per-episode denial rate on benign turns exceeds 0.7
for a sustained window. This script computes the sliding rate post-hoc
from the training ``reward_debug.jsonl`` that the blue environment
already emits, and writes a CSV trace plus a PNG plot per iteration.

The 200-episode window in the paper is narrated at that nominal scale;
each blue training phase is capped at ~100 episodes, so the sustained-
window check is parameterized by ``--window`` (default 30) so the
script produces useful output at the real training-phase length. The
original 0.7 threshold is retained.

Inputs:
    results_dir: path to a self-play run containing
      ``iter_*/blueteam/.../debug_logs/reward_debug.jsonl``.

Outputs (written to ``<results_dir>/collapse_monitor/``):
    iter_<k>_benign_denial.csv   — per-episode benign denial rate
    iter_<k>_benign_denial.png   — line plot with threshold overlay
    summary.json                 — warning flags per iteration
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


def find_iteration_logs(results_dir: Path) -> dict[int, Path]:
    """Map iteration index → reward_debug.jsonl path.

    The training harness nests logs under
    ``iter_<k>/blueteam/.../run_*/debug_logs/reward_debug.jsonl``; we
    pick the first match per iteration (typically one exists).
    """
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


def per_episode_benign_denial(jsonl_path: Path) -> list[tuple[int, float]]:
    """Return [(episode_idx, denial_rate)] over benign-turn steps only.

    Denial rate is the fraction of benign steps in that episode where
    ``is_refusal`` is True. Episodes with zero benign steps are skipped.
    """
    by_episode: dict[int, list[bool]] = defaultdict(list)
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("turn_type") != "benign":
                continue
            if r.get("is_eval"):  # exclude eval passes from training trace
                continue
            by_episode[int(r["episode"])].append(bool(r.get("is_refusal", False)))

    out = []
    for ep in sorted(by_episode):
        flags = by_episode[ep]
        if not flags:
            continue
        out.append((ep, sum(flags) / len(flags)))
    return out


def sliding_mean(values: list[float], window: int) -> list[float]:
    """Right-aligned sliding mean; output[i] uses values[i-window+1 .. i]."""
    if window <= 1 or not values:
        return list(values)
    out = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def flag_sustained_high(
    denials: list[float], window: int, threshold: float
) -> list[tuple[int, int, float]]:
    """Return (start_idx, end_idx, mean_rate) for every run of ``window``
    consecutive episodes whose sliding mean stays above ``threshold``.
    Adjacent/overlapping windows are merged.
    """
    if len(denials) < window:
        return []
    over = [
        i for i in range(window - 1, len(denials))
        if sum(denials[i - window + 1 : i + 1]) / window > threshold
    ]
    if not over:
        return []
    flags = []
    run_start = over[0] - window + 1
    run_end = over[0]
    for i in over[1:]:
        if i == run_end + 1:
            run_end = i
        else:
            flags.append(
                (run_start, run_end,
                 sum(denials[run_start : run_end + 1]) / (run_end - run_start + 1))
            )
            run_start = i - window + 1
            run_end = i
    flags.append(
        (run_start, run_end,
         sum(denials[run_start : run_end + 1]) / (run_end - run_start + 1))
    )
    return flags


def process_iteration(
    iter_idx: int,
    jsonl_path: Path,
    out_dir: Path,
    window: int,
    threshold: float,
) -> dict:
    pairs = per_episode_benign_denial(jsonl_path)
    if not pairs:
        return {"iter": iter_idx, "n_benign_episodes": 0, "warnings": []}

    episodes = [ep for ep, _ in pairs]
    rates = [r for _, r in pairs]
    rolling = sliding_mean(rates, window)

    csv_path = out_dir / f"iter_{iter_idx}_benign_denial.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "benign_denial_rate", f"rolling_mean_w{window}"])
        for ep, r, m in zip(episodes, rates, rolling):
            w.writerow([ep, f"{r:.4f}", f"{m:.4f}"])

    flags = flag_sustained_high(rates, window, threshold)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, rates, color="#888", alpha=0.4, label="per-episode")
    ax.plot(episodes, rolling, color="#c0392b", lw=2, label=f"rolling mean (w={window})")
    ax.axhline(threshold, color="k", ls="--", lw=1, label=f"threshold {threshold:.2f}")
    for start, end, mean_rate in flags:
        ax.axvspan(episodes[start], episodes[end], color="#c0392b", alpha=0.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training episode (benign only)")
    ax.set_ylabel("Benign denial rate")
    ax.set_title(
        f"Iteration {iter_idx}: benign denial rate "
        f"({len(flags)} sustained-high window{'s' if len(flags) != 1 else ''})"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"iter_{iter_idx}_benign_denial.png", dpi=120)
    plt.close(fig)

    warnings = [
        {
            "start_episode": int(episodes[s]),
            "end_episode": int(episodes[e]),
            "mean_rate": round(m, 4),
        }
        for s, e, m in flags
    ]
    return {
        "iter": iter_idx,
        "n_benign_episodes": len(episodes),
        "overall_denial_rate": round(sum(rates) / len(rates), 4),
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="Path to a self-play run directory")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Override output directory (useful when results_dir is read-only)")
    ap.add_argument("--window", type=int, default=30,
                    help="Sliding-window size in episodes (default 30 to fit a ~100-episode phase)")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Denial-rate threshold above which a window triggers a warning")
    args = ap.parse_args()

    logs = find_iteration_logs(args.results_dir)
    if not logs:
        print(f"No iter_*/blueteam debug logs found under {args.results_dir}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (args.results_dir / "collapse_monitor")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for k, path in logs.items():
        print(f"[iter {k}] {path}")
        summary.append(process_iteration(k, path, out_dir, args.window, args.threshold))

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {"window": args.window, "threshold": args.threshold, "iterations": summary},
            f, indent=2,
        )
    print(f"\nWrote {summary_path}")
    total_warnings = sum(len(s["warnings"]) for s in summary)
    print(f"{total_warnings} sustained-high window(s) across {len(summary)} iterations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
