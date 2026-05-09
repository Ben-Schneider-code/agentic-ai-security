"""
Security–Utility Pareto frontier across self-play checkpoints.

Shows every (utility, security) point for each trained checkpoint.
x-axis: Utility = 1 − BRR = TPR (%) — higher is more helpful on benign queries.
y-axis: Security = 1 − PVR_conv (%) — higher means fewer attack conversations succeed.
Upper-right is ideal.  Pareto-optimal checkpoints are highlighted with a frontier
line; dominated checkpoints are shown faded in the same colour.

Can be run standalone:
    python plotting/plot_security_utility_pareto.py --results results-<ID>
Or imported:
    from plotting.plot_security_utility_pareto import (
        plot_security_utility_pareto, DESCRIPTION
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style,
        load_cross_eval_results,
        load_benign_only,
        extract_diagonal_metrics,
        parse_results_arg,
        RUN_COLORS,
        FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_cross_eval_results,
        load_benign_only,
        extract_diagonal_metrics,
        parse_results_arg,
        RUN_COLORS,
        FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = (
    "Security–Utility Pareto frontier across self-play checkpoints. "
    "x = 1 − BRR = TPR (%): blue-team utility on benign queries (higher is better). "
    "y = 1 − PVR_conv (%): blue-team security against attack conversations (higher is better). "
    "Pareto-optimal checkpoints are highlighted with a frontier line; "
    "dominated checkpoints are shown faded. "
    "Source: diagonal red_i vs blue_i pairings from cross_eval_results.json."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ci_to_yerr(
    vals: list[float],
    cis: list[list[float] | None],
) -> list[list[float]] | None:
    """Convert [(lo, hi)] CI pairs → [[lo_err, ...], [hi_err, ...]] for errorbar."""
    if not any(c is not None for c in cis):
        return None
    lo_errs: list[float] = []
    hi_errs: list[float] = []
    for val, ci in zip(vals, cis):
        if ci is None or np.isnan(val):
            lo_errs.append(0.0)
            hi_errs.append(0.0)
        else:
            lo_errs.append(max(0.0, val - ci[0]))
            hi_errs.append(max(0.0, ci[1] - val))
    return [lo_errs, hi_errs]


def _pareto_indices(points: list[dict]) -> set[int]:
    """Return indices of points not dominated on (utility, security) — both maximized.

    A point p is dominated iff some other point q has q.utility >= p.utility
    AND q.security >= p.security with at least one strict inequality.
    Ties are treated as non-dominated (both kept).
    """
    dominated: set[int] = set()
    n = len(points)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ui, si = points[i]["utility"], points[i]["security"]
            uj, sj = points[j]["utility"], points[j]["security"]
            if uj >= ui and sj >= si and (uj > ui or sj > si):
                dominated.add(i)
                break
    return set(range(n)) - dominated


# ---------------------------------------------------------------------------
# Public plotting function (deterministic, idempotent)
# ---------------------------------------------------------------------------

_FALLBACK_SUBDIR_PRIORITY = ("cross_eval", "diagonal_eval", "cross_eval_old2", "cross_eval_old")


def plot_security_utility_pareto(
    results: list[tuple[str, str]],
    out_path: str | Path,
    cross_eval_subdir: str = "cross_eval",
    annotate_iters: bool = True,
) -> Path:
    """Plot Security–Utility Pareto frontier for all checkpoints. Returns resolved Path.

    If ``cross_eval_results.json`` is missing under ``cross_eval_subdir`` for a run,
    falls back through ``_FALLBACK_SUBDIR_PRIORITY`` (cross_eval → diagonal_eval →
    cross_eval_old2 → cross_eval_old) and prints which subdir was actually used.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    colors = RUN_COLORS * (len(results) // len(RUN_COLORS) + 1)
    any_plotted = False
    used_subdirs: list[tuple[str, str]] = []

    for run_idx, (label, selfplay_dir) in enumerate(results):
        color = colors[run_idx]

        # Try the requested subdir first, then fall through the priority list.
        candidates = (cross_eval_subdir,) + tuple(
            s for s in _FALLBACK_SUBDIR_PRIORITY if s != cross_eval_subdir
        )
        xeval = None
        chosen = None
        for sd in candidates:
            xeval = load_cross_eval_results(selfplay_dir, sd)
            if xeval is not None:
                chosen = sd
                break
        if xeval is None:
            print(
                f"[plot_security_utility_pareto] cross_eval_results.json not found in "
                f"any of {candidates} under {selfplay_dir} — skipping run.",
                file=sys.stderr,
            )
            continue
        if chosen != cross_eval_subdir:
            print(
                f"[plot_security_utility_pareto] {selfplay_dir}: requested subdir "
                f"{cross_eval_subdir!r} unavailable; using fallback {chosen!r}.",
                file=sys.stderr,
            )
        used_subdirs.append((label, chosen))

        diag = extract_diagonal_metrics(xeval, load_benign_only(selfplay_dir, chosen))
        points = [
            {
                "iter": it,
                "utility":  m["tpr"],
                "security": 100.0 - m["asr"],
                "utility_ci": m.get("tpr_ci"),
                "security_ci": (
                    [100.0 - m["asr_ci"][1], 100.0 - m["asr_ci"][0]]
                    if m.get("asr_ci") is not None else None
                ),
            }
            for it, m in sorted(diag.items())
            if m.get("tpr") is not None and m.get("asr") is not None
        ]

        if not points:
            print(
                f"[plot_security_utility_pareto] No usable checkpoints in {selfplay_dir} — skipping run.",
                file=sys.stderr,
            )
            continue

        pareto_idx = _pareto_indices(points)

        # Draw all points faded first
        all_u = [p["utility"] for p in points]
        all_s = [p["security"] for p in points]
        ax.scatter(all_u, all_s, color=color, alpha=0.25, s=40, edgecolors="none", zorder=2)

        # Pareto subset — prominent
        pareto_pts = [points[i] for i in sorted(pareto_idx)]
        pu = [p["utility"] for p in pareto_pts]
        ps = [p["security"] for p in pareto_pts]
        ax.scatter(
            pu, ps,
            color=color, alpha=1.0, s=80,
            edgecolors="black", linewidths=0.6,
            zorder=4, label=label,
        )

        # Frontier line — sort Pareto points by ascending utility
        frontier = sorted(pareto_pts, key=lambda p: p["utility"])
        fu = [p["utility"] for p in frontier]
        fs = [p["security"] for p in frontier]
        ax.plot(fu, fs, color=color, linewidth=2, alpha=0.9, zorder=3)

        # Error bars on Pareto points only
        sec_yerr = _ci_to_yerr(ps, [p["security_ci"] for p in pareto_pts])
        util_xerr = _ci_to_yerr(pu, [p["utility_ci"] for p in pareto_pts])
        if sec_yerr is not None or util_xerr is not None:
            ax.errorbar(
                pu, ps,
                yerr=sec_yerr,
                xerr=util_xerr,
                fmt="none",
                color=color,
                capsize=4,
                linewidth=1.2,
                zorder=3,
            )

        # Iteration labels on Pareto points only
        if annotate_iters:
            for p in pareto_pts:
                ax.annotate(
                    str(p["iter"]),
                    (p["utility"], p["security"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )

        any_plotted = True

    ax.set_xlabel(r"Utility: $1 - \mathrm{BRR}$ (%)")
    ax.set_ylabel(r"Security: $1 - \mathrm{PVR}_{\mathrm{conv}}$ (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.4)
    if any_plotted:
        ax.legend(frameon=True, loc="lower left")
    if used_subdirs:
        srcs = ", ".join(f"{lbl}: {sd}" for lbl, sd in used_subdirs)
        ax.set_title(f"Source: {srcs}", fontsize=8, color="gray", loc="right")
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Security–Utility Pareto frontier across self-play checkpoints."
    )
    parser.add_argument(
        "--results", nargs="+", required=True, metavar="DIR[:LABEL]",
        help="One or more selfplay result dirs, optionally with :Label suffix.",
    )
    parser.add_argument(
        "--out", default="figures/security_utility_pareto.png", metavar="PATH",
    )
    parser.add_argument(
        "--cross-eval-subdir", default="cross_eval",
        help="Subdirectory within selfplay_dir for cross-eval results (default: cross_eval).",
    )
    parser.add_argument(
        "--no-annotate-iters", dest="annotate_iters", action="store_false",
        help="Suppress iteration-number annotations on Pareto points.",
    )
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out = plot_security_utility_pareto(
        results,
        args.out,
        cross_eval_subdir=args.cross_eval_subdir,
        annotate_iters=args.annotate_iters,
    )
    print(f"[{DESCRIPTION}]\n  → {out}")


if __name__ == "__main__":
    main()
