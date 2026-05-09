#!/usr/bin/env python3
"""Aggregate ablation self-play runs into comparison plots per methodology §sec:ablations.

Expects a directory laid out by ``run_ablations.sh``::

    ablations/
      A1_flat_decay/selfplay/cross_eval/cross_eval_results.json
      A1_baseline/selfplay/cross_eval/cross_eval_results.json
      A2_plain_only/...
      A2_baseline/...
      A3_fixed_0.5/...
      A3_baseline/...
      A4_reversed/...
      A4_baseline/...
      A5_K1/selfplay/cross_eval/cross_eval_results.json
      A5_K2/...
      A5_K4/...
      A5_K8/...

For each ablation it extracts tail-iteration PVR_conv, PVR_turn, and PUD
from the held-out cross-eval JSON and renders a bar-chart comparison.

Usage:  python util/plot_ablations.py <ablations_dir> [--output-dir DIR]

Naming note: the JSON field ``asr`` is an alias for ``PVR_conv`` in
problem_statement.tex; this script reads ``asr`` from JSON but renders it
as ``PVR_conv`` in all titles and legends.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_cross_eval(path: Path) -> dict | None:
    candidates = [
        path / "cross_eval_results.json",
        path / "selfplay" / "cross_eval" / "cross_eval_results.json",
    ]
    for c in candidates:
        if c.exists():
            with c.open() as f:
                return json.load(f)
    return None


def extract_tail_metrics(ce: dict) -> dict:
    """Mean metrics over the final iteration's blue pairings."""
    pairings = ce.get("pairings", {})
    if not pairings:
        return {}
    blues = {p["blue_iter"] for p in pairings.values()}
    tail_blue = max(blues)
    filtered = [p for p in pairings.values() if p["blue_iter"] == tail_blue]
    if not filtered:
        return {}
    keys = ["asr", "pvr_turn", "tpr", "dominance"]
    out = {}
    for k in keys:
        vals = [p["metrics"].get(k) for p in filtered if p["metrics"].get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    # PUD = 100 - TPR (in percent)
    out["pud"] = 100 - out.get("tpr", 0) if not np.isnan(out.get("tpr", np.nan)) else float("nan")
    return out


def plot_pair(ax, labels, vals_a, vals_b, title, ylabel, condition_names):
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, vals_a, w, label=condition_names[0], color="#3498db")
    ax.bar(x + w / 2, vals_b, w, label=condition_names[1], color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def render_two_condition(root: Path, ablation: str, treatment_dir: str, out_path: Path) -> None:
    baseline = load_cross_eval(root / f"{ablation}_baseline")
    treatment = load_cross_eval(root / treatment_dir)
    if baseline is None or treatment is None:
        print(f"[{ablation}] skipping — missing cross_eval JSON "
              f"(baseline={baseline is not None}, treatment={treatment is not None})")
        return
    b = extract_tail_metrics(baseline)
    t = extract_tail_metrics(treatment)

    metrics = ["asr", "pvr_turn", "pud"]
    titles = {
        "asr": r"$\mathrm{PVR}_{\mathrm{conv}}$ (episode-level, %)",
        "pvr_turn": r"$\mathrm{PVR}_{\mathrm{turn}}$ (turn-level, %)",
        "pud": r"PUD (utility degradation, %)",
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4))
    for ax, m in zip(axes, metrics):
        plot_pair(
            ax,
            labels=[ablation],
            vals_a=[b.get(m, 0)],
            vals_b=[t.get(m, 0)],
            title=titles[m],
            ylabel="%",
            condition_names=["baseline", treatment_dir.split("_", 1)[1]],
        )
    fig.suptitle(f"Ablation {ablation}: baseline vs {treatment_dir}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), dpi=150)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"[{ablation}] saved {out_path}")


def render_a5(root: Path, out_path: Path) -> None:
    k_dirs = sorted(root.glob("A5_K*"))
    if not k_dirs:
        print("[A5] skipping — no A5_K* directories")
        return
    ks, asr, pvr_turn, pud = [], [], [], []
    for d in k_dirs:
        ce = load_cross_eval(d)
        if ce is None:
            continue
        try:
            k = int(d.name.split("A5_K")[1])
        except ValueError:
            continue
        m = extract_tail_metrics(ce)
        ks.append(k)
        asr.append(m.get("asr", float("nan")))
        pvr_turn.append(m.get("pvr_turn", float("nan")))
        pud.append(m.get("pud", float("nan")))
    if not ks:
        print("[A5] skipping — no loadable A5 results")
        return
    order = np.argsort(ks)
    ks = np.array(ks)[order]
    asr = np.array(asr)[order]
    pvr_turn = np.array(pvr_turn)[order]
    pud = np.array(pud)[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, asr, "o-", label=r"$\mathrm{PVR}_{\mathrm{conv}}$", color="#e74c3c")
    ax.plot(ks, pvr_turn, "s-", label=r"$\mathrm{PVR}_{\mathrm{turn}}$", color="#f39c12")
    ax.plot(ks, pud, "^-", label="PUD", color="#3498db")
    ax.set_xlabel("Self-play iterations K")
    ax.set_ylabel("%")
    ax.set_title(r"A5 — Adaptation budget: $\mathrm{PVR}$/PUD vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), dpi=150)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"[A5] saved {out_path}")


def _load_summary(path: Path) -> dict | None:
    """Load a summary.json if it exists, else None. Silent on missing."""
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] cannot read {path}: {exc}")
        return None


def _latest_utility_ablation_tag(variant_dir: Path) -> Path | None:
    """Return the most recent <tag> subdir under ablations/<variant>/,
    or None if none complete enough to analyze.

    A tag is "complete enough" if eval_view/cross_eval/pairings/.../summary.json
    exists. Still counts as complete if some per-style benign dirs are missing —
    we render whatever's present.
    """
    if not variant_dir.is_dir():
        return None
    candidates = []
    for tag in sorted(variant_dir.iterdir()):
        if not tag.is_dir():
            continue
        ce_pairings = list(
            (tag / "eval_view" / "cross_eval" / "pairings").glob("red_*_blue_*/summary.json")
        )
        if ce_pairings:
            candidates.append(tag)
    return candidates[-1] if candidates else None


def render_utility_ablation(
    ablations_root: Path,
    out_dir: Path,
    variants: tuple[str, ...] = ("none", "plain-only"),
    styles: tuple[str, ...] = ("plain", "adversarial", "multi_turn"),
) -> None:
    """
    Resumable utility-ablation renderer keyed on `run_utility_ablation.sh`
    layout:

        <ablations_root>/
          none/<tag>/eval_view/cross_eval/pairings/red_1_blue_1/summary.json
          none/<tag>/benign_style_{plain,adversarial,multi_turn}/benign_only/blue_1/summary.json
          plain-only/<tag>/...

    Emits two idempotent artifacts:

      * figures/utility_ablation.png — grouped bar chart:
            per-variant × per-style BRR (from benign_style_*/benign_only/)
            plus a separate panel with PVR_conv / PVR_turn (from cross_eval).
      * figures/utility_ablation.json — all raw numbers + cross-ablation
            honeypot diff via metrics.compare_pairings.

    Safe to re-run as variants complete: skips missing inputs, renders
    whatever's present, marks incomplete variants in the JSON.
    """
    # Local import so util/ doesn't pull this in when not needed.
    try:
        from util.metrics import compare_pairings
    except ImportError:
        _here = Path(__file__).resolve().parent.parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        from util.metrics import compare_pairings  # type: ignore

    per_variant: dict[str, dict] = {}
    attack_summaries: dict[str, dict] = {}  # variant -> attack summary for diff

    for variant in variants:
        vdir = ablations_root / variant
        tag = _latest_utility_ablation_tag(vdir)
        if tag is None:
            print(f"[utility_ablation] {variant}: no completed tag under {vdir} — skipping.")
            per_variant[variant] = {"status": "pending", "tag_dir": None}
            continue

        # Attack eval (cross_eval) — there's exactly one pairing (red_1_blue_1)
        attack_dir = tag / "eval_view" / "cross_eval" / "pairings"
        attack_summary_path = next(attack_dir.glob("red_*_blue_*/summary.json"), None)
        attack_summary = _load_summary(attack_summary_path) if attack_summary_path else None

        # Per-style benign eval
        per_style: dict[str, dict] = {}
        for s in styles:
            style_dir = tag / f"benign_style_{s}" / "benign_only"
            # first blue iter (blue_1 under the eval_view symlink structure)
            blue_dirs = sorted(style_dir.glob("blue_*/summary.json")) if style_dir.is_dir() else []
            if blue_dirs:
                per_style[s] = _load_summary(blue_dirs[0]) or {}
            else:
                per_style[s] = {}

        per_variant[variant] = {
            "status": "complete" if attack_summary else "partial",
            "tag_dir": str(tag),
            "attack": attack_summary or {},
            "benign_per_style": per_style,
        }
        if attack_summary is not None:
            attack_summaries[variant] = attack_summary

    # ---- cross-ablation honeypot diff (whichever two variants are present) ----
    diffs: dict[str, dict] = {}
    present = list(attack_summaries)
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            diffs[f"{a}_vs_{b}"] = compare_pairings(
                attack_summaries[a], attack_summaries[b]
            )

    # ---- write JSON artifact (single source of truth) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "utility_ablation.json"
    with json_path.open("w") as f:
        json.dump({
            "description": (
                "Utility-ablation comparison. Variants: 'none' removes all "
                "benign training signal (BLUETEAM_FIXED_ATTACK_PROB=1.0); "
                "'plain-only' restricts the training benign pool to plain "
                "single-turn queries (BLUETEAM_PLAIN_ONLY=1). For each "
                "variant we record attack PVR from eval_view/cross_eval and "
                "per-style TPR from benign_style_*/benign_only. "
                "Resumable: partial/missing inputs are marked in the "
                "status field. Cross-variant honeypot diff uses "
                "util/metrics.compare_pairings."
            ),
            "variants": per_variant,
            "cross_variant_honeypot_diff": diffs,
        }, f, indent=2)
    print(f"[utility_ablation] wrote {json_path}")

    # ---- bar chart ----
    complete_variants = [v for v in variants if per_variant[v]["status"] != "pending"]
    if not complete_variants:
        print("[utility_ablation] no complete variants — skipping bar chart.")
        return

    fig, (ax_brr, ax_pvr) = plt.subplots(
        1, 2, figsize=(4.2 * 2 + 1.5, 4.5),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    # BRR per style per variant
    x = np.arange(len(styles))
    w = 0.8 / max(len(complete_variants), 1)
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for i, variant in enumerate(complete_variants):
        benign = per_variant[variant]["benign_per_style"]
        # "benign_denial_rate" or derive 100 - TPR
        vals = []
        for s in styles:
            bs = benign.get(s, {}) or {}
            tpr = bs.get("metrics", {}).get("tpr")
            if tpr is None:
                vals.append(float("nan"))
            else:
                vals.append(100.0 - float(tpr))
        ax_brr.bar(x + (i - (len(complete_variants) - 1) / 2) * w, vals, w,
                   label=variant, color=palette[i % len(palette)])
    ax_brr.set_xticks(x)
    ax_brr.set_xticklabels(styles)
    ax_brr.set_ylabel("BRR (%)")
    ax_brr.set_title("Utility ablation: benign refusal by style")
    ax_brr.legend(fontsize=10)
    ax_brr.grid(True, axis="y", alpha=0.3)

    # PVR per variant
    for i, variant in enumerate(complete_variants):
        attack = per_variant[variant]["attack"].get("metrics", {})
        pvr_conv = attack.get("asr", float("nan"))
        pvr_turn = attack.get("pvr_turn", float("nan"))
        xs = np.array([0, 1]) + (i - (len(complete_variants) - 1) / 2) * w
        ax_pvr.bar(xs, [pvr_conv, pvr_turn], w,
                   label=variant, color=palette[i % len(palette)])
    ax_pvr.set_xticks([0, 1])
    ax_pvr.set_xticklabels([r"$\mathrm{PVR}_{\mathrm{conv}}$",
                            r"$\mathrm{PVR}_{\mathrm{turn}}$"])
    ax_pvr.set_ylabel("PVR (%)")
    ax_pvr.set_title("Utility ablation: attack success")
    ax_pvr.legend(fontsize=10)
    ax_pvr.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Utility ablation ({', '.join(complete_variants)}) — red_1 vs blue_1",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(pad=0.5)
    png = out_dir / "utility_ablation.png"
    fig.savefig(png, dpi=150)
    fig.savefig(png.with_suffix(".pdf"), dpi=150)
    plt.close(fig)
    print(f"[utility_ablation] wrote {png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate ablation self-play runs into comparison plots.",
    )
    parser.add_argument(
        "ablations_dir",
        help="Root directory laid out by run_ablations.sh OR run_utility_ablation.sh.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write figures into. Defaults to <ablations_dir>/figures.",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "broad", "utility"],
        help="'broad' = original A1–A5 layout (run_ablations.sh). "
             "'utility' = run_utility_ablation.sh layout (none/plain-only). "
             "'auto' (default) picks whichever layout is detected.",
    )
    args = parser.parse_args()

    root = Path(args.ablations_dir)
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory")
        sys.exit(1)

    fig_dir = Path(args.output_dir) if args.output_dir else root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Detect layout when --mode auto
    mode = args.mode
    if mode == "auto":
        has_broad = any((root / f"{a}_baseline").is_dir() for a in ("A1", "A2", "A3", "A4"))
        has_utility = (root / "none").is_dir() or (root / "plain-only").is_dir()
        if has_utility and not has_broad:
            mode = "utility"
        elif has_broad and not has_utility:
            mode = "broad"
        elif has_utility and has_broad:
            print("[auto] both layouts detected; running both.")
            mode = "both"
        else:
            print("[auto] neither layout detected; exiting.")
            sys.exit(1)

    if mode in ("broad", "both"):
        for ablation, treatment_suffix in [
            ("A1", "A1_flat_decay"),
            ("A2", "A2_plain_only"),
            ("A4", "A4_reversed"),
        ]:
            render_two_condition(root, ablation, treatment_suffix, fig_dir / f"ablation_{ablation}")

        a3_candidates = sorted(root.glob("A3_fixed_*"))
        if a3_candidates:
            render_two_condition(root, "A3", a3_candidates[0].name, fig_dir / "ablation_A3")
        else:
            print("[A3] skipping — no A3_fixed_* treatment directory")

        render_a5(root, fig_dir / "ablation_A5")

    if mode in ("utility", "both"):
        # Emit into the global figures/ dir when writing the utility ablation
        # so evaluation.md can cite figures/utility_ablation.png directly.
        utility_out = Path(args.output_dir) if args.output_dir else (
            Path(__file__).resolve().parent.parent / "figures"
        )
        render_utility_ablation(root, utility_out)

    print(f"\nFigures written under {fig_dir}")


if __name__ == "__main__":
    main()
