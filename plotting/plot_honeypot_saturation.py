"""
Honeypot saturation across self-play — "attack-capacity-limited" evidence.

Left panel (training-time): red EIS spent per iteration + red yield on the
22-honeypot universe during training. Every iteration exits via
`no_new_honeypot_for_1000_steps`, so EIS shrinks as red runs out of novel
targets even though the compute budget is the same.

Right panel (eval-time, n=200 attack eps/cell): coverage_pct and yield_pct on
the co-evolved diagonal from diagonal_eval (or cross_eval once the new 800-ep
run is aggregated). Coverage = fraction of the 22-honeypot universe red's SQL
references; yield = fraction actually executed past blue.

CLI:
    python plotting/plot_honeypot_saturation.py --results <dir>[:label]
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
    from ._data import (
        apply_paper_style,
        load_pairing_metrics_with_decomposed,
        extract_diagonal_metrics,
        parse_results_arg,
        write_sidecar,
        RED_COL, BLUE_COL, GRAY_COL, GREEN_COL,
        FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_pairing_metrics_with_decomposed,
        extract_diagonal_metrics,
        parse_results_arg,
        write_sidecar,
        RED_COL, BLUE_COL, GRAY_COL, GREEN_COL,
        FIG_SIZE_1x2,
    )

from util.compute_cost_analysis import (
    discover_iterations,
    load_phase_data,
    compute_metrics,
)

apply_paper_style()

DESCRIPTION = (
    "Honeypot saturation across self-play. Left: per-iteration red EIS "
    "(training compute spent) and training-time yield (accessed/total honeypots). "
    "Every red iteration exits via `no_new_honeypot_for_1000_steps`, so EIS "
    "shrinks as the honeypot universe is exhausted. Right: eval-time coverage "
    "and yield on the co-evolved diagonal — if the attack surface saturates, "
    "these flatten regardless of additional blue compute."
)

_SUBDIR_PRIORITY = ("cross_eval", "diagonal_eval", "cross_eval_old2")


def _load_or_compute_cost(selfplay_dir: str) -> dict | None:
    """Return compute_cost metrics dict, loading the cached JSON if present,
    else computing inline from raw iter_N artifacts (<1s, no GPU)."""
    p = Path(selfplay_dir) / "compute_cost_analysis.json"
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    raw_iters = discover_iterations(selfplay_dir)
    if not raw_iters:
        return None
    loaded = []
    for it in raw_iters:
        entry = {"iter": it["iter"]}
        if it["red_dir"]:
            entry["red"] = load_phase_data(it["red_dir"], "redteam")
        if it["blue_dir"]:
            entry["blue"] = load_phase_data(it["blue_dir"], "blueteam")
        loaded.append(entry)
    return compute_metrics(loaded)


def _load_diagonal(selfplay_dir: str) -> tuple[dict[int, dict], str] | None:
    # load_pairing_metrics_with_decomposed re-computes coverage_pct/yield_pct
    # from raw JSONL when they are missing from the cached JSON.
    for sd in _SUBDIR_PRIORITY:
        ce = load_pairing_metrics_with_decomposed(selfplay_dir, subdir=sd)
        if ce is None:
            continue
        diag = extract_diagonal_metrics(ce)
        if diag:
            return diag, sd
    return None


def plot_honeypot_saturation(
    results: list[tuple[str, str]],
    out_path: str | Path,
    show_ci: bool = True,
) -> Path:
    out_path = Path(out_path)
    if len(results) > 1:
        print("[plot_honeypot_saturation] Multiple runs given; using first only.",
              file=sys.stderr)
    label, selfplay_dir = results[0]

    # Training-side data
    train_iters: list[int] = []
    red_eis: list[int] = []
    red_yield: list[float] = []
    red_accessed: list[int] = []
    red_exits: list[str] = []
    cc = _load_or_compute_cost(selfplay_dir)
    if cc:
        for row in cc.get("per_iteration", []):
            train_iters.append(row["iter"])
            red_eis.append(row["red_eis"])
            red_yield.append(100.0 * row["red_yield"])
            red_accessed.append(row["red_accessed_hp"])
            red_exits.append(row.get("red_exit", "?"))

    # Eval-side data
    diag_loaded = _load_diagonal(selfplay_dir)
    eval_iters: list[int] = []
    cov: list[float] = []
    cov_ci: list[list[float] | None] = []
    yld: list[float] = []
    yld_ci: list[list[float] | None] = []
    source = "—"
    if diag_loaded is not None:
        diag, source = diag_loaded
        for i in sorted(diag):
            eval_iters.append(i)
            m = diag[i]
            cov.append(m.get("coverage_pct") if m.get("coverage_pct") is not None
                       else float("nan"))
            cov_ci.append(m.get("coverage_pct_ci"))
            yld.append(m.get("yield_pct") if m.get("yield_pct") is not None
                       else float("nan"))
            yld_ci.append(m.get("yield_pct_ci"))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    # ---- Left: training-side ----
    if train_iters:
        ax2 = ax_l.twinx()
        bars = ax_l.bar(train_iters, red_eis, color=RED_COL, alpha=0.35,
                        edgecolor=RED_COL, label="Red EIS (training)")
        for x, v, acc in zip(train_iters, red_eis, red_accessed):
            ax_l.text(x, v + max(red_eis) * 0.02,
                      f"{v}\n({acc}/22 hp)",
                      ha="center", va="bottom", fontsize=8, color="#444")
        ax2.plot(train_iters, red_yield, color=GREEN_COL, marker="o",
                 linewidth=1.8, markersize=7, label="Training yield (%)")
        ax_l.set_xlabel("Self-play iteration")
        ax_l.set_ylabel("Red EIS (training compute)", color=RED_COL)
        ax2.set_ylabel("Training yield on 22 honeypots (%)", color=GREEN_COL)
        ax_l.set_xticks(train_iters)
        ax_l.set_title("Training-time saturation\n(all exits: no_new_honeypot_for_1000_steps)")
        ax_l.grid(True, axis="y", alpha=0.3)
        ax_l.tick_params(axis="y", labelcolor=RED_COL)
        ax2.tick_params(axis="y", labelcolor=GREEN_COL)
        # Combined legend
        h1, l1 = ax_l.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_l.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9, frameon=True)
    else:
        ax_l.text(0.5, 0.5, "compute_cost_analysis.json missing",
                  ha="center", va="center", transform=ax_l.transAxes, color=GRAY_COL)

    # ---- Right: eval-side ----
    if eval_iters:
        def _yerr(vals, cis):
            if not any(c is not None for c in cis):
                return None
            lo = [max(0, v - (c[0] if c else v)) for v, c in zip(vals, cis)]
            hi = [max(0, (c[1] if c else v) - v) for v, c in zip(vals, cis)]
            return [lo, hi]
        ax_r.errorbar(eval_iters, cov,
                      yerr=(_yerr(cov, cov_ci) if show_ci else None),
                      fmt="-o", color=BLUE_COL, linewidth=1.8, markersize=7,
                      capsize=4, label="Coverage (referenced / 22)")
        ax_r.errorbar(eval_iters, yld,
                      yerr=(_yerr(yld, yld_ci) if show_ci else None),
                      fmt="--s", color=RED_COL, linewidth=1.8, markersize=7,
                      capsize=4, label="Yield (accessed / 22)")
        ax_r.set_xlabel("Self-play iteration (co-evolved diagonal)")
        ax_r.set_ylabel("Fraction of honeypot universe (%)")
        ax_r.set_xticks(eval_iters)
        ax_r.set_ylim(0, max(100, max((c or 0) for c in cov) * 1.1))
        ax_r.set_ylim(0, 100)
        ax_r.set_title(f"Eval-time saturation (source: {source})")
        ax_r.grid(True, axis="y", alpha=0.3)
        ax_r.legend(loc="upper right", fontsize=9, frameon=True)
    else:
        ax_r.text(0.5, 0.5, "No diagonal eval data",
                  ha="center", va="center", transform=ax_r.transAxes, color=GRAY_COL)

    fig.suptitle(
        f"Attack-capacity-limited equilibrium — {label} "
        "(22-honeypot universe: 2 tables + 12 cols + 5 so-cols + 3 order_ids)",
        fontsize=12,
    )
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    # stderr summary for paper use
    print(f"[plot_honeypot_saturation] label={label} source={source}", file=sys.stderr)
    if train_iters:
        for i, e, a, y, x in zip(train_iters, red_eis, red_accessed, red_yield, red_exits):
            print(f"  train iter={i} red_eis={e} accessed={a}/22 yield={y:.1f}% exit={x}",
                  file=sys.stderr)
    if eval_iters:
        for i, c, y in zip(eval_iters, cov, yld):
            print(f"  eval iter={i} coverage={c}% yield={y}%", file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    ap.add_argument("--out", default="figures/honeypot_saturation.png")
    args = ap.parse_args()
    results = parse_results_arg(args.results)
    out = plot_honeypot_saturation(results, args.out)
    write_sidecar(out, DESCRIPTION, results)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
