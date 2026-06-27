"""
Compute cost (EIS) per self-play iteration for red and blue teams.

EIS = Environment Interaction Steps = total_num_steps from training_state.json.
This is a GPU-invariant proxy for training compute cost.

Left subplot:  grouped bars (red vs. blue) per iteration with ratio annotation.
Right subplot: cumulative EIS lines showing total compute invested over iterations.

Ported and simplified from util/compute_cost_analysis.py.

Can be run standalone:
    python plotting/plot_running_time.py --results results-<ID>
Or imported:
    from plotting.plot_running_time import plot_running_time, DESCRIPTION
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style, discover_iterations, find_run_dir, load_training_state,
        parse_results_arg,
        RED_COL, BLUE_COL, RUN_COLORS, FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, discover_iterations, find_run_dir, load_training_state,
        parse_results_arg,
        RED_COL, BLUE_COL, RUN_COLORS, FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = (
    "Compute cost (EIS = Environment Interaction Steps) per self-play iteration. "
    "Left: grouped bars showing red vs. blue team EIS per iteration, with blue/red "
    "ratio annotated above each pair — blue team consistently exhausts its training "
    "budget while red team exits early via convergence signals (no_new_honeypot, "
    "all_honeypots_accessed), reflecting that adaptive defense is computationally "
    "harder than adaptive offense. "
    "Right: cumulative EIS lines showing total compute invested across iterations. "
    "EIS is read from total_num_steps in training_state.json (GPU-invariant)."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eis_per_iteration(
    selfplay_dir: str,
) -> list[dict]:
    """
    Load EIS for each iteration's red and blue training phase.

    Returns a list of dicts:
        {"iter": int, "red_eis": int | None, "blue_eis": int | None}
    """
    iters_data = discover_iterations(selfplay_dir)
    records: list[dict] = []

    for entry in iters_data:
        n = entry["iter"]
        rec: dict = {"iter": n, "red_eis": None, "blue_eis": None}

        for team, key in (("red_dir", "red_eis"), ("blue_dir", "blue_eis")):
            team_dir = entry.get(team)
            if team_dir is None:
                continue
            run_dir = find_run_dir(team_dir)
            if run_dir is None:
                continue
            try:
                state = load_training_state(run_dir)
                rec[key] = state.get("total_num_steps", 0)
            except FileNotFoundError:
                pass

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Pure compute (data loading + math only; no matplotlib, no side effects)
# ---------------------------------------------------------------------------

def compute_running_time(
    results: list[tuple[str, str]],
    **kwargs,
) -> dict:
    """
    Compute per-iteration and cumulative EIS for each run (pure, JSON-serializable).

    For each (label, selfplay_dir) in `results`, EIS per iteration is sourced from
    `<selfplay_dir>/compute_cost_analysis.json` when present (its
    `per_iteration[*].red_eis/blue_eis` + `aggregate`); otherwise derived via the
    same loaders the plot uses (`discover_iterations` + `load_training_state`'s
    `total_num_steps`, through `load_eis_per_iteration`).

    Returns:
        {
          "runs": [
            {
              "label": str,
              "selfplay_dir": str,
              "source": "compute_cost_analysis.json" | "training_state",
              "per_iteration": [{"iter", "red_eis", "blue_eis", "ratio"}],
              "cumulative": {"red": [...], "blue": [...]},
              "aggregate": {"mean_red_eis", "mean_blue_eis", ...},
            }, ...
          ]
        }
    A run with no iteration data is omitted (matches the plot's `continue`).
    Returns {} when no run has data.
    """
    runs: list[dict] = []

    for label, selfplay_dir in results:
        records, source = _load_eis_records(selfplay_dir)
        if not records:
            continue

        per_iteration: list[dict] = []
        red_seq: list[int] = []
        blue_seq: list[int] = []
        for r in records:
            re_ = r["red_eis"] or 0
            be_ = r["blue_eis"] or 0
            red_seq.append(re_)
            blue_seq.append(be_)
            per_iteration.append({
                "iter": r["iter"],
                "red_eis": re_,
                "blue_eis": be_,
                # blue/red ratio (None when red is 0 — matches the plot's `re_ > 0` guard)
                "ratio": (be_ / re_) if re_ > 0 else None,
            })

        cum_red = list(np.cumsum(red_seq).astype(int)) if red_seq else []
        cum_blue = list(np.cumsum(blue_seq).astype(int)) if blue_seq else []
        # Cast numpy ints to plain ints for JSON-serializability.
        cum_red = [int(v) for v in cum_red]
        cum_blue = [int(v) for v in cum_blue]

        red_nonzero = [v for v in red_seq if v > 0]
        blue_nonzero = [v for v in blue_seq if v > 0]
        ratios = [p["ratio"] for p in per_iteration if p["ratio"] is not None]
        aggregate = {
            "mean_red_eis": float(np.mean(red_nonzero)) if red_nonzero else 0.0,
            "mean_blue_eis": float(np.mean(blue_nonzero)) if blue_nonzero else 0.0,
            "cumulative_red_eis": cum_red[-1] if cum_red else 0,
            "cumulative_blue_eis": cum_blue[-1] if cum_blue else 0,
            "mean_eis_ratio": float(np.mean(ratios)) if ratios else 0.0,
        }

        runs.append({
            "label": label,
            "selfplay_dir": selfplay_dir,
            "source": source,
            "per_iteration": per_iteration,
            "cumulative": {"red": cum_red, "blue": cum_blue},
            "aggregate": aggregate,
        })

    if not runs:
        return {}
    return {"runs": runs}


def _load_eis_records(selfplay_dir: str) -> tuple[list[dict], str]:
    """
    Return ([{"iter", "red_eis", "blue_eis"}, ...], source_str).

    Prefers <selfplay_dir>/compute_cost_analysis.json when it has a populated
    per_iteration list; otherwise derives from training_state.json via
    load_eis_per_iteration. Mirrors the plot's exact numeric source.
    """
    cca = Path(selfplay_dir) / "compute_cost_analysis.json"
    if cca.is_file():
        try:
            with open(cca) as f:
                data = json.load(f)
            per_iter = data.get("per_iteration") or []
            if per_iter:
                records = [
                    {
                        "iter": entry.get("iter"),
                        "red_eis": entry.get("red_eis"),
                        "blue_eis": entry.get("blue_eis"),
                    }
                    for entry in per_iter
                ]
                return records, "compute_cost_analysis.json"
        except (json.JSONDecodeError, OSError):
            pass
    return load_eis_per_iteration(selfplay_dir), "training_state"


# ---------------------------------------------------------------------------
# Public plotting function (deterministic, idempotent)
# ---------------------------------------------------------------------------

def plot_running_time(
    results: list[tuple[str, str]],
    out_path: str | Path,
    precomputed: dict | None = None,
) -> Path:
    """
    Plot EIS per iteration (grouped bars) and cumulative EIS (line chart).

    Args:
        results:  [(label, selfplay_dir), ...]
        out_path: Destination PNG path.
        precomputed: optional dict from compute_running_time(results); when None
            it is computed here. Fully drives rendering.

    Returns the resolved Path that was written.
    """
    out_path = Path(out_path)
    if precomputed is None:
        precomputed = compute_running_time(results)
    run_data = precomputed.get("runs", []) if precomputed else []
    # Index computed runs by position in `results` so per-run styling (colors,
    # offsets) matches the original loop even when some runs were dropped (no data).
    by_dir: dict[str, dict] = {r["selfplay_dir"]: r for r in run_data}

    fig, (ax_bar, ax_cum) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    colors = RUN_COLORS * (len(results) // len(RUN_COLORS) + 1)

    iters: list[int] = []
    for run_idx, (label, selfplay_dir) in enumerate(results):
        run = by_dir.get(selfplay_dir)
        if run is None:
            print(f"  [plot_running_time] No data found in {selfplay_dir}", file=sys.stderr)
            continue

        per_iter = run["per_iteration"]
        iters = [r["iter"] for r in per_iter]
        red_eis = [r["red_eis"] for r in per_iter]
        blue_eis = [r["blue_eis"] for r in per_iter]

        n_runs = len(results)
        bar_w = 0.35 / max(n_runs, 1)
        x = np.arange(len(iters))
        offset = (run_idx - (n_runs - 1) / 2) * bar_w * 2

        red_color  = RED_COL  if n_runs == 1 else colors[run_idx]
        blue_color = BLUE_COL if n_runs == 1 else _lighten(colors[run_idx])

        red_label  = ("Red team"  if n_runs == 1 else f"{label} red")
        blue_label = ("Blue team" if n_runs == 1 else f"{label} blue")

        # --- Grouped bar chart ---
        ax_bar.bar(
            x + offset - bar_w / 2, red_eis,
            width=bar_w, color=red_color, alpha=0.85, label=red_label,
        )
        ax_bar.bar(
            x + offset + bar_w / 2, blue_eis,
            width=bar_w, color=blue_color, alpha=0.85, label=blue_label,
        )

        # Annotate blue/red ratio above each pair (single run only to avoid clutter)
        if n_runs == 1:
            for i, (re_, be_) in enumerate(zip(red_eis, blue_eis)):
                if re_ > 0:
                    ratio = be_ / re_
                    ax_bar.text(
                        x[i], max(re_, be_) * 1.03,
                        f"{ratio:.1f}× blue",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="#555555",
                    )

        # Mean lines
        if red_eis:
            mean_r = np.mean([v for v in red_eis if v > 0] or [0])
            ax_bar.axhline(mean_r, color=red_color, linestyle=":",
                           linewidth=1.2, alpha=0.7)
        if blue_eis:
            mean_b = np.mean([v for v in blue_eis if v > 0] or [0])
            ax_bar.axhline(mean_b, color=blue_color, linestyle=":",
                           linewidth=1.2, alpha=0.7)

        # --- Cumulative EIS ---
        cum_red  = np.asarray(run["cumulative"]["red"])
        cum_blue = np.asarray(run["cumulative"]["blue"])

        ax_cum.plot(iters, cum_red, marker="o", color=red_color, linewidth=2,
                    label=red_label)
        ax_cum.plot(iters, cum_blue, marker="s", color=blue_color, linewidth=2,
                    label=blue_label)
        ax_cum.fill_between(iters, 0, cum_red, color=red_color, alpha=0.1)
        ax_cum.fill_between(iters, 0, cum_blue, color=blue_color, alpha=0.15)

    # Bar chart formatting
    ax_bar.set_xticks(x if len(iters) > 0 else [])
    ax_bar.set_xticklabels(iters)
    ax_bar.set_xlabel("Self-play iteration")
    ax_bar.set_ylabel("EIS")
    ax_bar.set_title("EIS per Iteration (Red vs. Blue)")
    ax_bar.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    ax_bar.legend(fontsize=9, frameon=True)
    ax_bar.grid(True, axis="y", alpha=0.4)

    # Cumulative chart formatting
    ax_cum.set_xlabel("Self-play iteration")
    ax_cum.set_ylabel("Cumulative EIS")
    ax_cum.set_title("Cumulative EIS across Iterations")
    ax_cum.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    if iters:
        ax_cum.set_xticks(iters)
    ax_cum.legend(fontsize=9, frameon=True)
    ax_cum.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lighten(hex_color: str, factor: float = 0.5) -> str:
    """Return a lighter shade of hex_color by blending towards white."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02x}{g2:02x}{b2:02x}"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot EIS per self-play iteration (running time proxy)."
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
    )
    parser.add_argument("--out", default="figures/running_time.png")
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out = plot_running_time(results, args.out)
    print(f"[{DESCRIPTION}]\n  → {out}")


if __name__ == "__main__":
    main()
