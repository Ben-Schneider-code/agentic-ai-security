#!/usr/bin/env python3
"""Emit the ablation master table for methodology.tex (sec:ablations).

Reads every ``cross_eval_results.json`` under an ablation root and writes a
LaTeX ``tabular`` block summarising the tail-iteration PVR/PUD metrics per
variant. Each row compares a treatment directory against the matching
``*_baseline`` row (or, for A5, against the full K sweep).

Usage:
    python util/ablation_table.py <ablations_dir> [--output FILE]

The ``asr`` field in JSON is the legacy name for ``PVR_conv`` — this script
reads ``asr`` from disk but labels it ``PVR_conv`` in the rendered table.

Per-style PUD (plain / adversarial / multi-turn) is not currently emitted by
``util/cross_evaluate.py``; those columns are omitted with a note. If a
``significance_matrix.json`` (from ``util/mcnemar_cross_eval.py``) is present
inside each variant's cross-eval dir, cells whose pairing-level p-value falls
below 0.05 are starred (*). Without that file the script still runs and
prints a warning — cells are left unstarred.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from plot_ablations import extract_tail_metrics, load_cross_eval  # type: ignore


METRIC_KEYS = ["asr", "pvr_turn", "pud", "cfr", "f1", "dominance"]
METRIC_HEADERS = {
    "asr": r"$\mathrm{PVR}_{\mathrm{conv}}$",
    "pvr_turn": r"$\mathrm{PVR}_{\mathrm{turn}}$",
    "pud": r"$\mathrm{PUD}$",
    "cfr": r"$\mathrm{CFR}$",
    "f1": r"$\mathrm{F1}$",
    "dominance": r"$\mathrm{Dom}$",
}


def load_significance(variant_dir: Path) -> dict | None:
    """Locate significance_matrix.json inside variant_dir or its selfplay/cross_eval."""
    for candidate in (
        variant_dir / "significance_matrix.json",
        variant_dir / "cross_eval" / "significance_matrix.json",
        variant_dir / "selfplay" / "cross_eval" / "significance_matrix.json",
    ):
        if candidate.exists():
            with candidate.open() as f:
                return json.load(f)
    return None


def any_significant(sig: dict | None, threshold: float = 0.05) -> bool:
    """Return True if any pairing p-value in the matrix falls below threshold."""
    if not sig:
        return False
    for section in ("blue_vs_blue", "red_vs_red"):
        for entries in sig.get(section, {}).values():
            for row in entries:
                p = row.get("p_mcnemar")
                if p is None:
                    p = row.get("p_two_prop")
                if p is not None and p < threshold:
                    return True
    return False


def get_tail_metrics(variant_dir: Path) -> tuple[dict, bool]:
    """Return (metrics dict, has_significant_result)."""
    ce = load_cross_eval(variant_dir)
    if ce is None:
        return {}, False
    m = extract_tail_metrics(ce)
    # extract_tail_metrics returns asr/pvr_turn/tpr/pud/dominance; add missing.
    if ce_pairings := ce.get("pairings"):
        blues = {p["blue_iter"] for p in ce_pairings.values()}
        tail_blue = max(blues)
        filtered = [p for p in ce_pairings.values() if p["blue_iter"] == tail_blue]
        for extra in ("cfr", "f1"):
            vals = [p["metrics"].get(extra) for p in filtered if p["metrics"].get(extra) is not None]
            m[extra] = float(np.mean(vals)) if vals else float("nan")
    sig = load_significance(variant_dir)
    return m, any_significant(sig)


def fmt(val: float, star: bool) -> str:
    if val != val:  # NaN check
        return "--"
    s = f"{val:.1f}"
    return f"{s}*" if star else s


def render_row(variant_label: str, metrics: dict, starred: bool) -> str:
    cells = [variant_label]
    for k in METRIC_KEYS:
        cells.append(fmt(metrics.get(k, float("nan")), starred and k in ("asr", "pvr_turn", "pud")))
    return " & ".join(cells) + r" \\"


def collect_variants(root: Path) -> list[tuple[str, Path]]:
    """Return (display_label, dir) pairs in a stable order.

    Groups are emitted in the ablation sequence A1..A5, with baseline always
    listed first inside each pair (A3/A5 fan out across multiple variants).
    """
    groups: list[tuple[str, Path]] = []
    pairs = [
        ("A1", "A1_baseline", [("A1 flat reward decay", "A1_flat_decay")]),
        ("A2", "A2_baseline", [("A2 plain benign only", "A2_plain_only")]),
        ("A4", "A4_baseline", [("A4 reversed rewards", "A4_reversed")]),
    ]
    for tag, baseline_name, treatments in pairs:
        if (root / baseline_name).is_dir():
            groups.append((f"{tag} baseline", root / baseline_name))
        for label, treat_name in treatments:
            if (root / treat_name).is_dir():
                groups.append((label, root / treat_name))

    # A3: match any A3_fixed_* treatment
    if (root / "A3_baseline").is_dir():
        groups.append(("A3 baseline", root / "A3_baseline"))
    for p in sorted(root.glob("A3_fixed_*")):
        prob = p.name.removeprefix("A3_fixed_")
        groups.append((f"A3 fixed prob={prob}", p))

    # A5: sweep over K
    for p in sorted(root.glob("A5_K*"), key=lambda d: int(d.name.removeprefix("A5_K") or "0")):
        k = p.name.removeprefix("A5_K")
        groups.append((f"A5 K={k}", p))
    return groups


def render_table(root: Path) -> tuple[str, bool]:
    """Return (latex_string, had_any_significance_info)."""
    variants = collect_variants(root)
    if not variants:
        raise SystemExit(f"No ablation variant directories found under {root}")

    any_sig_info = False
    lines: list[str] = []
    lines.append("% Ablation master table — generated by util/ablation_table.py")
    lines.append("% Columns: PVR_conv, PVR_turn, PUD, CFR, F1, Dominance (tail iteration).")
    lines.append("% Per-style PUD (plain/adversarial/multi-turn) is not yet emitted by")
    lines.append("% cross_evaluate.py; once those keys land here, extend METRIC_KEYS.")
    lines.append(r"\begin{tabular}{l" + "c" * len(METRIC_KEYS) + "}")
    lines.append(r"\toprule")
    header_cells = ["Variant"] + [METRIC_HEADERS[k] for k in METRIC_KEYS]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")
    for label, variant_dir in variants:
        metrics, starred = get_tail_metrics(variant_dir)
        if not metrics:
            print(f"[warn] {variant_dir}: cross_eval_results.json missing, emitting blank row")
            lines.append(render_row(label, {}, False))
            continue
        # Track whether any variant brought a significance matrix to the party.
        if load_significance(variant_dir) is not None:
            any_sig_info = True
        lines.append(render_row(label, metrics, starred))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines), any_sig_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablations_dir", help="Root of the ablation results tree.")
    parser.add_argument(
        "--output",
        default=None,
        help="Destination file. Defaults to stdout.",
    )
    args = parser.parse_args()

    root = Path(args.ablations_dir)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    latex, had_sig = render_table(root)
    if not had_sig:
        print(
            "[warn] no significance_matrix.json found in any variant — "
            "cells will not be starred. Run util/mcnemar_cross_eval.py first.",
            file=sys.stderr,
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex + "\n")
        print(f"Saved: {out_path}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
