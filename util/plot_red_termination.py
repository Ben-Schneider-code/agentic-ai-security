"""
Red-team termination analysis: training effort and novelty discovery per iteration.

All red-team runs exit with `no_new_honeypot_for_1000_steps`, confirming that
the saturation criterion (not a wall-clock limit) drives convergence. The figure
shows two panels:

  Left:  Training gradient steps at termination per red iteration.
         Later iters terminate faster because the hardened defender constrains
         the exploitable surface — except iter 6, which recovers broad-tier
         coverage and therefore requires ~2× more training steps.

  Right: Unique novel honeypot IDs first discovered per red iteration.
         Marginal discovery falls monotonically (iters 1→5) before iter 6
         recovers a suite of previously-suppressed honeypots.

The exit-reason file path is:
  {selfplay_dir}/iter_{i}/redteam/**/exit_reason.txt

CLI:
    python util/plot_red_termination.py \\
        --results results-<ID>[:Label] [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from plotting._data import apply_paper_style, parse_results_arg, FIG_SIZE_1x2
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import apply_paper_style, parse_results_arg, FIG_SIZE_1x2

apply_paper_style()

DESCRIPTION = "Red-team termination: training steps + novel honeypot discovery per iteration"


def _find_red_run_dir(iter_dir: Path) -> Path | None:
    matches = list(iter_dir.glob("redteam/**/exit_reason.txt"))
    return matches[0].parent if matches else None


def _last_checkpoint_step(run_dir: Path) -> int | None:
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    steps = []
    for d in ckpt_root.iterdir():
        if d.is_dir() and d.name.startswith("steps_"):
            try:
                steps.append(int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return max(steps) if steps else None


def _novel_honeypots(run_dir: Path) -> list[str]:
    jsonl = run_dir / "debug_logs" / "reward_debug.jsonl"
    if not jsonl.is_file():
        return []
    seen: set[str] = set()
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for hp_id in r.get("new_honeypots_accessed", []):
            seen.add(str(hp_id))
    return sorted(seen)


def compute_termination_stats(selfplay_dir: str) -> dict[int, dict]:
    base = Path(selfplay_dir)
    result: dict[int, dict] = {}
    for iter_dir in sorted(base.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        try:
            iter_num = int(iter_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        run_dir = _find_red_run_dir(iter_dir)
        if run_dir is None:
            continue

        exit_reason_path = run_dir / "exit_reason.txt"
        exit_reason = exit_reason_path.read_text().strip() if exit_reason_path.is_file() else "unknown"
        final_step = _last_checkpoint_step(run_dir)
        novel_hps = _novel_honeypots(run_dir)

        result[iter_num] = {
            "exit_reason": exit_reason,
            "final_step": final_step,
            "novel_honeypots": novel_hps,
            "n_novel": len(novel_hps),
        }
    return result


def plot_red_termination(
    results: list[tuple[str, str]],
    out_dir: str = "figures/",
) -> Path:
    out_path = Path(out_dir) / "red_termination.png"
    sidecar_path = Path(out_dir) / "red_termination.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label, selfplay_dir = results[0]
    data = compute_termination_stats(selfplay_dir)
    if not data:
        print("[red_termination] No data found.", file=sys.stderr)
        return out_path

    with open(sidecar_path, "w") as f:
        json.dump({str(k): {**v, "novel_honeypots": v["novel_honeypots"]} for k, v in data.items()}, f, indent=2)

    iters = sorted(data.keys())
    all_saturated = all(data[i]["exit_reason"] == "no_new_honeypot_for_1000_steps" for i in iters)

    steps = [data[i]["final_step"] or 0 for i in iters]
    n_novel = [data[i]["n_novel"] for i in iters]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    # Left: training steps at termination
    bar_colors = ["#F44336" if data[i]["exit_reason"] != "no_new_honeypot_for_1000_steps"
                  else "#2196F3" for i in iters]
    ax_l.bar(iters, steps, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_l.set_xlabel("Red training iteration")
    ax_l.set_ylabel("Gradient steps at termination")
    ax_l.set_title("Red training effort per iteration")
    ax_l.set_xticks(iters)
    if all_saturated:
        ax_l.text(0.5, 0.98,
                  "All iters: no_new_honeypot_for_1000_steps",
                  transform=ax_l.transAxes, va="top", ha="center",
                  fontsize=7, style="italic", color="#2196F3")

    # Right: unique novel honeypots per iter
    ax_r.bar(iters, n_novel, color="#FF9800", edgecolor="white", linewidth=0.5)
    ax_r.set_xlabel("Red training iteration")
    ax_r.set_ylabel("Unique novel honeypot IDs discovered")
    ax_r.set_title("Marginal honeypot discovery per iteration")
    ax_r.set_xticks(iters)

    # Add a cumulative secondary axis
    ax_r2 = ax_r.twinx()
    cumulative = []
    seen: set[str] = set()
    for i in iters:
        seen.update(data[i]["novel_honeypots"])
        cumulative.append(len(seen))
    ax_r2.plot(iters, cumulative, "k--o", linewidth=1, markersize=4, alpha=0.6, label="Cumulative")
    ax_r2.set_ylabel("Cumulative unique honeypots")
    ax_r2.set_ylim(0, max(cumulative) * 1.3)
    ax_r2.legend(loc="lower right", fontsize=7)

    fig.suptitle(
        f"{label}: Red saturation — training steps and novel discovery per iteration",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[red_termination] saved {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_red_termination(results, args.out_dir)


if __name__ == "__main__":
    main()
