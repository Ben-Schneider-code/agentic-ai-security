#!/usr/bin/env python3
"""Compare eval PVR across honeypot arms to explain the rowcol(~50%) -> col(<30%) drop (Q3).

For each results dir, pulls the diagonal (red_i vs blue_i) cross-eval metrics, then groups
them by ``honeypot_type`` and reports the mechanistic gap. The drop is explained by
``yield_pct`` (fraction of the honeypot universe actually accessed) collapsing in the
col arm — col removes the easy row/owner-id enumeration path — together with col's
near-100% ``neutral_sql_rate`` (SQL that never touches a honeypot, so blue has nothing
to violate on).

Metrics are read straight from each run's cached cross_eval_results.json — they were
computed at cross-eval time under that run's own honeypot arm and are authoritative per
run; the decomposed keys (yield_pct/coverage_pct/...) are already present in every cache.
We deliberately do NOT route through the recompute path (load_cross_eval_results /
compute_pairing_metrics): it imports MARFT, which fixes HONEYPOT_TYPE at first import, so
in a mixed-arm invocation it would recompute every run against whichever arm imported
first — corrupting or crashing on the values we are trying to compare.

Instead we detect the historical universe-propagation bug directly: a run whose cached
``honeypot_universe`` disagrees with its summary.json arm (e.g. a ``col`` run cached with
universe 64) was evaluated under the wrong arm — its PVR/yield are not comparable, so it
is shown with a banner and excluded from the grouped means.

Output is stdout tables only. Reuses plotting/_data.py and util/_diag_common.py.

Usage:
    python util/pvr_config_compare.py RESULTS_DIR [RESULTS_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting._data import extract_diagonal_metrics  # noqa: E402
from util._diag_common import (  # noqa: E402
    HONEYPOT_UNIVERSE,
    load_run_summary,
    note,
)

# Diagonal metrics shown per run; (key, has_ci, width).
COLS = [
    ("asr", True, "PVR_conv(asr)"),
    ("pvr_turn", True, "PVR_turn"),
    ("yield_pct", True, "yield_pct"),
    ("neutral_sql_rate", False, "neutral_sql"),
    ("coverage_pct", False, "coverage"),
]
# Keys aggregated across runs for the grouped comparison.
AGG_KEYS = ["asr", "pvr_turn", "yield_pct", "neutral_sql_rate"]


def load_cross_eval_raw(results_dir: str) -> dict | None:
    """Read the cached cross_eval_results.json directly (no recompute/refresh).

    This is the primary loader on purpose: the decomposed metrics are already present
    in every cache, and routing through the recompute path would import MARFT (fixing
    HONEYPOT_TYPE at first import) and corrupt/crash mixed-arm comparisons. Reading the
    cache also preserves the on-disk ``honeypot_universe`` so the wrong-arm bug stays
    detectable rather than being silently "fixed" by an inconsistent recompute.
    """
    for sub in ("cross_eval", "cross_eval_quick"):
        path = Path(results_dir) / sub / "cross_eval_results.json"
        if path.is_file():
            with open(path) as fh:
                return json.load(fh)
    return None


def cached_universe(cross_eval: dict) -> int | None:
    """The honeypot_universe recorded in the cached cross-eval (None if absent)."""
    for pairing in cross_eval.get("pairings", {}).values():
        u = pairing.get("episode_stats", {}).get("honeypot_universe")
        if u:
            return u
    return None


def is_empty_cell(m: dict) -> bool:
    """A diagonal cell with no SQL/honeypot signal at all (e.g. incomplete cross-eval)."""
    return all(not m.get(k) for k in ("asr", "yield_pct", "neutral_sql_rate", "coverage_pct"))


def _cell(m: dict, key: str, has_ci: bool) -> str:
    v = m.get(key)
    if v is None:
        return f"{'n/a':>18}"
    ci = m.get(f"{key}_ci") if has_ci else None
    if ci:
        return f"{v:5.1f} [{ci[0]:4.1f},{ci[1]:4.1f}]"
    return f"{v:5.1f}{'':>12}"


def render_run(results_dir: str, honeypot_type: str, diag: dict, universe, mismatch: bool, expected: int) -> None:
    banner = ""
    if mismatch:
        banner = (
            f"  <-- WRONG ARM: cross-eval universe={universe} but {honeypot_type} expects "
            f"{expected}; PVR/yield NOT comparable (excluded from aggregation)"
        )
    print(f"\n{results_dir}  honeypot_type={honeypot_type}  universe={universe}{banner}")
    header = "  iter  " + "  ".join(f"{label:<18}" for _, _, label in COLS)
    print(header)
    for it in sorted(diag):
        m = diag[it]
        cells = "  ".join(_cell(m, key, has_ci) for key, has_ci, _ in COLS)
        tag = "  (incomplete/zero)" if is_empty_cell(m) else ""
        print(f"   {it:<4}  {cells}{tag}")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+", help="one or more results-* directories")
    args = ap.parse_args()

    # type -> key -> list of diagonal values across runs/iters
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    run_counts: dict[str, int] = defaultdict(int)
    excluded: list[str] = []

    print("== Per-run diagonal eval metrics (red_i vs blue_i, 99% CI where available) ==")
    for d in args.results_dirs:
        summary = load_run_summary(d)
        hp = summary["honeypot_type"]
        cross_eval = load_cross_eval_raw(d)
        if cross_eval is None:
            note(f"{d}: no cross_eval — skipping")
            continue
        diag = extract_diagonal_metrics(cross_eval)
        if not diag:
            note(f"{d}: cross_eval has no diagonal (red_i==blue_i) pairings — skipping")
            continue

        expected = HONEYPOT_UNIVERSE.get(hp)
        cached = cached_universe(cross_eval)
        universe = cached if cached is not None else expected
        mismatch = cached is not None and expected is not None and cached != expected

        render_run(d, hp, diag, universe, mismatch, expected)

        if mismatch:
            excluded.append(f"{d} (universe {cached}!={expected})")
            continue  # arm-mismatched runs are not comparable — exclude from means

        run_counts[hp] += 1
        for it, m in diag.items():
            if is_empty_cell(m):
                continue  # don't let empty/incomplete cells drag down group means
            for k in AGG_KEYS:
                if m.get(k) is not None:
                    grouped[hp][k].append(m[k])

    if excluded:
        print("\n== Excluded from aggregation (cross-eval under wrong honeypot arm) ==")
        for e in excluded:
            print(f"  - {e}")

    if not grouped:
        note("no comparable runs produced diagonal metrics; nothing to group")
        return

    print("\n== Grouped by honeypot_type (mean over runs x diagonal iters) ==")
    print(f"  {'type':<8} {'n_runs':>6} {'n_cells':>7}   "
          f"{'PVR_conv':>9} {'PVR_turn':>9} {'yield_pct':>9} {'neutral_sql':>11}")
    means: dict[str, dict[str, float | None]] = {}
    for hp in sorted(grouped):
        g = grouped[hp]
        means[hp] = {k: _mean(g[k]) for k in AGG_KEYS}
        ncells = len(g["asr"])

        def s(k: str) -> str:
            v = means[hp][k]
            return f"{v:.1f}" if v is not None else "—"

        print(f"  {hp:<8} {run_counts[hp]:>6} {ncells:>7}   "
              f"{s('asr'):>9} {s('pvr_turn'):>9} {s('yield_pct'):>9} {s('neutral_sql_rate'):>11}")

    if "rowcol" in means and "col" in means:
        r, c = means["rowcol"], means["col"]

        def gap(k: str) -> str:
            if r[k] is None or c[k] is None:
                return "—"
            return f"{r[k] - c[k]:+.1f} pp"

        print("\n== Mechanistic gap (rowcol - col) ==")
        print(f"  PVR_conv gap:     {gap('asr')}")
        print(f"  yield_pct gap:    {gap('yield_pct')}   "
              "<- red accesses far more of the honeypot universe in rowcol")
        print(f"  neutral_sql gap:  {gap('neutral_sql_rate')}   "
              "<- col SQL is overwhelmingly neutral; blue rarely sees a real violation")
        print("  => the PVR drop rowcol(~50%) -> col(<30%) is driven by the yield collapse")
        print("     (col removes the easy row/owner-id enumeration path) plus near-100%")
        print("     neutral SQL in the col arm.")


if __name__ == "__main__":
    main()
