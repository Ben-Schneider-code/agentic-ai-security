"""
PVR (conv + turn) along the co-evolved diagonal with 99% CIs and asymptote band.

This replaces the previous pvr.png/pvr_conv.png, which were silently empty when
the aggregated cross_eval_results.json was absent. The new plot:
  * Reads from cross_eval/ first (800 ep/cell once the re-run lands) and falls
    back to diagonal_eval/ (200 ep/cell) or cross_eval_old2/ (50 ep/cell).
  * Renders both PVR_conv and PVR_turn side-by-side.
  * Overlays a plateau mean ± 1σ band (tail half of iterations) to make the
    "asymptote" claim explicit.
  * Labels every point with its 99% Wilson CI and prints the numeric table
    to stderr for paper use.

CLI:
    python plotting/plot_pvr_asymptote.py --results <dir>[:label]
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
        extract_diagonal_metrics,
        parse_results_arg,
        write_sidecar,
        BLUE_COL, RED_COL, GRAY_COL,
        FIG_SIZE_1x2,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        load_cross_eval_results,
        extract_diagonal_metrics,
        parse_results_arg,
        write_sidecar,
        BLUE_COL, RED_COL, GRAY_COL,
        FIG_SIZE_1x2,
    )

apply_paper_style()

DESCRIPTION = (
    "PVR_conv and PVR_turn along the co-evolved diagonal (red_i vs blue_i) "
    "with 99% Wilson CIs and a plateau mean ± 1σ band over the tail half of "
    "iterations. Source priority: cross_eval → diagonal_eval → cross_eval_old2."
)

_SUBDIR_PRIORITY = ("cross_eval", "diagonal_eval", "cross_eval_old2")


def _load_diagonal(selfplay_dir: str) -> tuple[dict[int, dict], str] | None:
    for sd in _SUBDIR_PRIORITY:
        ce = load_cross_eval_results(selfplay_dir, subdir=sd)
        if ce is None:
            continue
        diag = extract_diagonal_metrics(ce)
        if diag:
            return diag, sd
    return None


def _plateau(vals: list[float]) -> tuple[float, float, int]:
    n_tail = max(1, (len(vals) + 1) // 2)
    tail = vals[-n_tail:]
    return float(np.mean(tail)), float(np.std(tail)), n_tail


def _render(ax, iters, vals, lo, hi, metric_label: str, color: str,
            show_ci: bool = True) -> None:
    iters = np.array(iters)
    vals = np.array(vals)
    yerr = np.array([[v - l for v, l in zip(vals, lo)],
                     [h - v for v, h in zip(vals, hi)]]) if show_ci else None
    ax.errorbar(
        iters, vals, yerr=yerr,
        fmt="-o", color=color, linewidth=1.8, markersize=7,
        capsize=4, capthick=1.2, elinewidth=1.0, zorder=3,
        label="per-iter PVR" + (" (99% Wilson CI)" if show_ci else ""),
    )
    # Plateau band — only meaningful when the tail has ≥2 points. With a single
    # tail point (≤2 iterations) the std is identically 0 and a "plateau / bounded
    # equilibrium" claim is not estimable, so we suppress the band and say so.
    plat_mean, plat_std, n_tail = _plateau(vals.tolist())
    estimable = n_tail >= 2
    x0 = iters[-n_tail]
    x1 = (iters[-1] + (iters[-1] - iters[0]) * 0.18) if len(iters) > 1 else iters[-1] + 0.5
    if estimable:
        ax.axhline(plat_mean, color=GRAY_COL, linewidth=1.5, linestyle="--", zorder=2,
                   label=f"plateau mean = {plat_mean:.1f}%")
        if show_ci:
            ax.fill_between([x0, x1], plat_mean - plat_std, plat_mean + plat_std,
                            color=GRAY_COL, alpha=0.18, zorder=1,
                            label=f"±1σ ({plat_std:.1f} pp, tail-{n_tail})")
        ax.annotate(
            "", xy=(x1, plat_mean),
            xytext=(iters[-1] + 1e-6, plat_mean),
            arrowprops=dict(arrowstyle="->", color=GRAY_COL, lw=1.3),
        )
    else:
        ax.text(0.5, 0.97, f"plateau not estimable (N={len(iters)} iter)",
                ha="center", va="top", transform=ax.transAxes, fontsize=8,
                color="#B00020",
                bbox=dict(boxstyle="round", fc="#FFF0F0", ec="#B00020", lw=0.6))
    if show_ci:
        for x, v, l, h in zip(iters, vals, lo, hi):
            ax.text(x, v + (h - l) * 0.55 + 0.5,
                    f"[{l:.0f}, {h:.0f}]",
                    ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_xlabel("Self-play iteration (co-evolved diagonal)")
    ax.set_ylabel(metric_label)
    ax.set_xticks(iters)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(fontsize=9, frameon=True, loc="upper left")


def plot_pvr_asymptote(
    results: list[tuple[str, str]],
    out_path: str | Path,
    show_ci: bool = True,
) -> tuple[Path, dict]:
    out_path = Path(out_path)
    if len(results) > 1:
        print("[plot_pvr_asymptote] Multiple runs given; using first only.",
              file=sys.stderr)
    label, selfplay_dir = results[0]

    loaded = _load_diagonal(selfplay_dir)
    if loaded is None:
        print(f"[plot_pvr_asymptote] No diagonal metrics found in {selfplay_dir}.",
              file=sys.stderr)
        fig, ax = plt.subplots(figsize=FIG_SIZE_1x2)
        ax.text(0.5, 0.5, "No diagonal cross-eval data",
                ha="center", va="center", transform=ax.transAxes, color=GRAY_COL)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path, {}

    diag, source = loaded
    iters = sorted(diag)
    pvr_conv = [diag[i].get("asr", float("nan")) for i in iters]
    pvr_conv_ci = [diag[i].get("asr_ci") or [float("nan"), float("nan")] for i in iters]
    pvr_turn = [diag[i].get("pvr_turn", float("nan")) for i in iters]
    pvr_turn_ci = [diag[i].get("pvr_turn_ci") or [float("nan"), float("nan")] for i in iters]
    # Raw numerator/denominator behind each PVR% — so a single breach over a tiny
    # denominator cannot be read as a trend. PVR_conv = n_honeypot_eps / n_eps_with_sql;
    # PVR_turn = n_fp_steps / n_attack_steps.
    raw = [diag[i].get("raw_counts", {}) for i in iters]
    conv_counts = [{"k": rc.get("n_honeypot_eps"), "n": rc.get("n_eps_with_sql")} for rc in raw]
    turn_counts = [{"k": rc.get("n_fp_steps"), "n": rc.get("n_attack_steps")} for rc in raw]

    print(f"[plot_pvr_asymptote] source={source}  diagonal N={len(iters)}",
          file=sys.stderr)
    print(f"{'iter':<6}{'PVR_conv':<10}{'k/n':<9}{'CI_conv':<16}"
          f"{'PVR_turn':<10}{'k/n':<9}{'CI_turn':<16}", file=sys.stderr)
    for i, vc, cc, vt, ct, kc, kt in zip(
        iters, pvr_conv, pvr_conv_ci, pvr_turn, pvr_turn_ci, conv_counts, turn_counts
    ):
        kn_c = f"{kc['k']}/{kc['n']}"
        kn_t = f"{kt['k']}/{kt['n']}"
        print(f"{i:<6}{vc:<10.2f}{kn_c:<9}[{cc[0]:.1f}, {cc[1]:.1f}]   "
              f"{vt:<10.2f}{kn_t:<9}[{ct[0]:.1f}, {ct[1]:.1f}]",
              file=sys.stderr)

    fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=FIG_SIZE_1x2)
    _render(ax_c, iters, pvr_conv,
            [c[0] for c in pvr_conv_ci],
            [c[1] for c in pvr_conv_ci],
            r"$\mathrm{PVR}_{\mathrm{conv}}$ (%)", RED_COL, show_ci=show_ci)
    _render(ax_t, iters, pvr_turn,
            [c[0] for c in pvr_turn_ci],
            [c[1] for c in pvr_turn_ci],
            r"$\mathrm{PVR}_{\mathrm{turn}}$ (%)", BLUE_COL, show_ci=show_ci)

    ax_c.set_title(r"Co-evolved $\mathrm{PVR}_{\mathrm{conv}}$")
    ax_t.set_title(r"Co-evolved $\mathrm{PVR}_{\mathrm{turn}}$")
    eq_phrase = (
        "Bounded equilibrium on the diagonal" if len(iters) >= 3
        else f"Diagonal PVR (N={len(iters)} iter — equilibrium not estimable)"
    )
    fig.suptitle(
        f"{eq_phrase} — {label}  (source: {source}, 99% Wilson CI)",
        fontsize=12,
    )
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    pvr_conv_plat_mean, pvr_conv_plat_std, n_tail = _plateau(pvr_conv)
    pvr_turn_plat_mean, pvr_turn_plat_std, _ = _plateau(pvr_turn)
    # A plateau/asymptote is only estimable when the tail has ≥2 points (N≥3 iters);
    # otherwise std≡0 is a degenerate artifact, not evidence of an equilibrium.
    plateau_estimable = n_tail >= 2

    def _plateau_block(per_iter, per_iter_ci, plat_mean, plat_std, counts):
        return {
            "per_iter_pct": [round(v, 2) for v in per_iter],
            "per_iter_ci_99": [[round(c[0], 2), round(c[1], 2)] for c in per_iter_ci],
            "per_iter_counts": counts,
            "diag_mean_pct": round(float(np.mean(per_iter)), 2),
            "plateau_tail_mean_pct": round(plat_mean, 2),
            "plateau_tail_std_pp": round(plat_std, 2) if plateau_estimable else None,
        }

    metrics = {
        "source_subdir": source,
        "label": label,
        "selfplay_dir": selfplay_dir,
        "iters": iters,
        "n_iterations": len(iters),
        "n_tail_for_plateau": n_tail,
        "plateau_estimable": plateau_estimable,
        "pvr_conv": _plateau_block(pvr_conv, pvr_conv_ci, pvr_conv_plat_mean,
                                   pvr_conv_plat_std, conv_counts),
        "pvr_turn": _plateau_block(pvr_turn, pvr_turn_ci, pvr_turn_plat_mean,
                                   pvr_turn_plat_std, turn_counts),
    }
    return out_path, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    ap.add_argument("--out", default="figures/pvr_asymptote.png")
    args = ap.parse_args()
    results = parse_results_arg(args.results)
    out, metrics = plot_pvr_asymptote(results, args.out)
    write_sidecar(out, DESCRIPTION, results, metrics)
    print(f"[{DESCRIPTION[:100]}...]\n  → {out}")


if __name__ == "__main__":
    main()
