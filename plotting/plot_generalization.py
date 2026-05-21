"""
Generalization plot — does training one team improve it against the frozen
opponent from the start of self-play?

Two panels, both with 99 % Wilson CIs:

  (a) Column 0 in the cross-eval matrix: **iterative red vs frozen blue_0**.
      If red's PVR_conv rises on this column, red is actually getting
      stronger, not merely co-evolving to the latest blue it trained with.
      A flat or declining column means red is only locally sharp — it won't
      transfer to a defender it has not seen.

  (b) Row 0 in the cross-eval matrix: **iterative blue vs frozen red_0**.
      Symmetric interpretation: if blue's PVR_conv falls across this row,
      blue is actually generalising defensive behaviour. If it stays flat
      or rises, later blues are not robust to the original attacker.

We also overlay the co-evolved diagonal as a dashed reference — the policy
pair each team was optimised against.

Source priority: cross_eval (800 ep/cell once the re-run completes) →
diagonal_eval (200 ep/cell) → cross_eval_old2 (50 ep/cell; CIs are wide).

CLI:
    python plotting/plot_generalization.py --results <dir>[:label]
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
        parse_results_arg,
        write_sidecar,
        RED_COL, BLUE_COL, GRAY_COL,
        FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_cross_eval_results,
        parse_results_arg,
        write_sidecar,
        RED_COL, BLUE_COL, GRAY_COL,
        FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = (
    "Generalization: PVR_conv for iterative-red vs frozen blue_0 (column 0) "
    "and iterative-blue vs frozen red_0 (row 0) in the cross-eval matrix. "
    "Dashed line: co-evolved diagonal as reference. Distinguishes "
    "co-evolution from actual generalisation. 99 % Wilson CIs. "
    "Source priority: cross_eval → diagonal_eval → cross_eval_old2."
)

_SUBDIR_PRIORITY = ("cross_eval", "cross_eval_old2", "diagonal_eval")


def _best_grid(
    selfplay_dir: str,
    subdir: str | None = None,
) -> tuple[dict, str] | None:
    """Pick the richest full-matrix source available.

    Prefer sources that actually have enough (red_i, blue_j) pairs to populate
    at least one full row or column.

    If `subdir` is given, only that subdir is considered (used to force e.g.
    `cross_eval_long/` once the long-episode eval completes).
    """
    candidates = (subdir,) if subdir else _SUBDIR_PRIORITY
    best = None
    for sd in candidates:
        ce = load_cross_eval_results(selfplay_dir, subdir=sd)
        if ce is None:
            continue
        pairings = ce.get("pairings", {})
        reds = sorted({v["red_iter"] for v in pairings.values()})
        blues = sorted({v["blue_iter"] for v in pairings.values()})
        # Score: prefer sources with a full col-0 or row-0 plus more iterations
        has_col0 = sum(1 for v in pairings.values() if v.get("blue_iter") == 0)
        has_row0 = sum(1 for v in pairings.values() if v.get("red_iter") == 0)
        score = (has_col0 + has_row0, len(reds) + len(blues))
        if best is None or score > best[2]:
            best = (ce, sd, score)
    if best is None:
        return None
    return best[0], best[1]


def _series(pairings: dict, fix_key: str, fix_val: int,
            var_key: str, metric: str = "asr") -> list[tuple[int, float, list[float]]]:
    """
    Return [(var_iter, value, [ci_lo, ci_hi])] for pairings where fix_key==fix_val,
    sorted by var_iter. Missing CIs fall back to [nan, nan].
    """
    rows = []
    for v in pairings.values():
        if v.get(fix_key) != fix_val:
            continue
        var = v.get(var_key)
        m = v.get("metrics", {}).get(metric)
        ci = v.get("confidence_intervals", {}).get(metric) or [float("nan"), float("nan")]
        if var is None or m is None:
            continue
        rows.append((var, m, ci))
    rows.sort(key=lambda r: r[0])
    return rows


def _plot_series(ax, rows, color, label, marker, show_ci: bool = True):
    if not rows:
        return
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    los = [r[2][0] for r in rows]
    his = [r[2][1] for r in rows]
    yerr = [
        [max(0.0, y - (l if not np.isnan(l) else y)) for y, l in zip(ys, los)],
        [max(0.0, (h if not np.isnan(h) else y) - y) for y, h in zip(ys, his)],
    ] if show_ci else None
    ax.errorbar(xs, ys, yerr=yerr,
                fmt=f"-{marker}", color=color, linewidth=1.8, markersize=7,
                capsize=4, capthick=1.1, elinewidth=1.0, label=label)


def plot_generalization(
    results: list[tuple[str, str]],
    out_path: str | Path,
    subdir: str | None = None,
    show_ci: bool = True,
) -> Path:
    out_path = Path(out_path)
    if len(results) > 1:
        print("[plot_generalization] Multiple runs given; using first only.",
              file=sys.stderr)
    label, selfplay_dir = results[0]

    loaded = _best_grid(selfplay_dir, subdir=subdir)
    if loaded is None:
        print(f"[plot_generalization] No cross-eval grid in {selfplay_dir}.",
              file=sys.stderr)
        fig, ax = plt.subplots(figsize=FIG_SIZE_1x2)
        ax.text(0.5, 0.5, "no grid data", ha="center", va="center",
                transform=ax.transAxes, color=GRAY_COL)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    ce, source = loaded
    pairings = ce.get("pairings", {})
    reds = sorted({v["red_iter"] for v in pairings.values()})
    blues = sorted({v["blue_iter"] for v in pairings.values()})
    n_attack = ce.get("metadata", {}).get("n_attack_episodes_per_cell", None)

    # Column 0 = iterative red vs frozen blue_0 (vary red_iter)
    col0 = _series(pairings, fix_key="blue_iter", fix_val=0, var_key="red_iter")
    # Row 0 = iterative blue vs frozen red_0 (vary blue_iter)
    row0 = _series(pairings, fix_key="red_iter", fix_val=0, var_key="blue_iter")
    # Diagonal
    diag_rows = [(v["red_iter"], v["metrics"]["asr"],
                  v["confidence_intervals"].get("asr") or [float("nan")] * 2)
                 for v in pairings.values()
                 if v.get("red_iter") == v.get("blue_iter")
                 and v.get("metrics", {}).get("asr") is not None]
    diag_rows.sort(key=lambda r: r[0])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)

    # ---- (a) Column 0: iterative red vs frozen blue_0 ----
    _plot_series(ax_l, col0, RED_COL,
                 r"iterative red vs frozen blue$_0$", "o", show_ci=show_ci)
    if diag_rows:
        xs = [r[0] for r in diag_rows]
        ys = [r[1] for r in diag_rows]
        ax_l.plot(xs, ys, linestyle="--", color=GRAY_COL, linewidth=1.3,
                  marker="x", markersize=6,
                  label="co-evolved diagonal (reference)")
    ax_l.set_xlabel("Red-team iteration")
    ax_l.set_ylabel(r"$\mathrm{PVR}_{\mathrm{conv}}$ (%)")
    ax_l.set_title(
        "Iterative-red vs frozen blue$_0$\n"
        "rising ⇒ red transfers; flat ⇒ red only co-evolved with its partner"
    )
    ax_l.set_ylim(0, max(35, max([r[1] for r in col0] + [r[1] for r in diag_rows] + [1]) + 5))
    if col0:
        ax_l.set_xticks(sorted(set(r[0] for r in col0 + diag_rows)))
    ax_l.legend(fontsize=9, frameon=True, loc="upper left")
    ax_l.grid(True, axis="y", alpha=0.4)

    # ---- (b) Row 0: iterative blue vs frozen red_0 ----
    _plot_series(ax_r, row0, BLUE_COL,
                 r"iterative blue vs frozen red$_0$", "s", show_ci=show_ci)
    if diag_rows:
        xs = [r[0] for r in diag_rows]
        ys = [r[1] for r in diag_rows]
        ax_r.plot(xs, ys, linestyle="--", color=GRAY_COL, linewidth=1.3,
                  marker="x", markersize=6,
                  label="co-evolved diagonal (reference)")
    ax_r.set_xlabel("Blue-team iteration")
    ax_r.set_ylabel(r"$\mathrm{PVR}_{\mathrm{conv}}$ (%)")
    ax_r.set_title(
        "Iterative-blue vs frozen red$_0$\n"
        "falling ⇒ blue generalises; flat ⇒ blue only memorised later reds"
    )
    ax_r.set_ylim(0, max(35, max([r[1] for r in row0] + [r[1] for r in diag_rows] + [1]) + 5))
    if row0:
        ax_r.set_xticks(sorted(set(r[0] for r in row0 + diag_rows)))
    ax_r.legend(fontsize=9, frameon=True, loc="upper left")
    ax_r.grid(True, axis="y", alpha=0.4)

    ep_note = f" ({n_attack} ep/cell)" if n_attack else ""
    fig.suptitle(
        f"Generalisation of self-play — {label}  (source: {source}{ep_note}, 99 % Wilson CI)",
        fontsize=12,
    )
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    # Stderr: dump numeric table
    print(f"[plot_generalization] source={source}  reds={reds}  blues={blues}",
          file=sys.stderr)
    print("  col_0 (iterative red vs frozen blue_0):", file=sys.stderr)
    for x, y, ci in col0:
        print(f"    red={x}  PVR_conv={y:.2f}  [{ci[0]:.1f},{ci[1]:.1f}]",
              file=sys.stderr)
    print("  row_0 (iterative blue vs frozen red_0):", file=sys.stderr)
    for x, y, ci in row0:
        print(f"    blue={x}  PVR_conv={y:.2f}  [{ci[0]:.1f},{ci[1]:.1f}]",
              file=sys.stderr)
    print("  diagonal (reference):", file=sys.stderr)
    for x, y, _ in diag_rows:
        print(f"    iter={x}  PVR_conv={y:.2f}", file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    ap.add_argument("--out", default="figures/generalization.png")
    ap.add_argument(
        "--subdir", default=None, metavar="NAME",
        help="Force a specific cross-eval subdir (e.g. cross_eval_long). "
             "If omitted, the richest of "
             f"{list(_SUBDIR_PRIORITY)} is used.",
    )
    args = ap.parse_args()
    results = parse_results_arg(args.results)
    out = plot_generalization(results, args.out, subdir=args.subdir)
    write_sidecar(out, DESCRIPTION, results)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
