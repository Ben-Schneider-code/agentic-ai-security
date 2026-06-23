#!/usr/bin/env python3
"""Cross-eval honeypot referenced-vs-accessed gap (the PVR-mechanism diagnostic).

For each results dir, read cross_eval_results.json and, per red/blue pairing, report
how many honeypots the attack SQL *referenced* (coverage) vs how many actually got
*accessed* (yield == PVR surface). The gap localizes WHY PVR is low:

  - low coverage              -> red never references honeypot columns (attacker problem)
  - high coverage, ~0 yield   -> red references them but the queries don't convert to
                                 access (execution errors / schema / defender) — a
                                 different fix than "make the red attack more"

Read-only; reuses plotting/_data.load_cross_eval_results and util/_diag_common.

Usage:
    python util/crosseval_pvr_gap.py RESULTS_DIR [RESULTS_DIR ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting._data import load_cross_eval_results  # noqa: E402
from util._diag_common import load_run_summary, note  # noqa: E402

# Thresholds for the localizing flags (percentage points of the honeypot universe).
LOW_COVERAGE_PCT = 5.0
CONVERSION_GAP_COVERAGE_PCT = 10.0
CONVERSION_GAP_YIELD_PCT = 1.0


def _num(v) -> float:
    return v if isinstance(v, (int, float)) else float("nan")


def render_dir(results_dir: str) -> None:
    summary = load_run_summary(results_dir)
    ce = load_cross_eval_results(results_dir)
    print(f"\n== {results_dir}  [honeypot_type={summary['honeypot_type']}] ==")
    if not ce or not ce.get("pairings"):
        note(f"{results_dir}: no cross_eval_results.json pairings — skipping")
        return

    print(
        f"  {'pairing':<14} {'eps':>4} {'cover%':>7} {'yield%':>7} "
        f"{'pvr_turn':>8} {'ref':>4} {'acc':>4} {'ref-not-acc':>11}"
    )
    universe = None
    for key, p in ce["pairings"].items():
        m = p.get("metrics", {})
        stats = p.get("episode_stats", {})
        ref = set(stats.get("referenced_honeypots") or [])
        acc = set(stats.get("accessed_honeypots") or [])
        universe = stats.get("honeypot_universe", universe)
        cov, yld = m.get("coverage_pct"), m.get("yield_pct")

        flag = ""
        if isinstance(cov, (int, float)) and cov < LOW_COVERAGE_PCT:
            flag = "  <- red barely references honeypots"
        elif (
            isinstance(cov, (int, float))
            and isinstance(yld, (int, float))
            and cov >= CONVERSION_GAP_COVERAGE_PCT
            and yld < CONVERSION_GAP_YIELD_PCT
        ):
            flag = "  <- references don't convert to access"

        print(
            f"  {key:<14} {p.get('n_attack_episodes', 0):>4} "
            f"{_num(cov):>7.2f} {_num(yld):>7.2f} {_num(m.get('pvr_turn')):>8.2f} "
            f"{len(ref):>4} {len(acc):>4} {len(ref - acc):>11}{flag}"
        )
    print(
        f"  (universe = {universe} honeypots; "
        f"cover%=referenced/universe, yield%=accessed/universe)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+", help="one or more results-* directories")
    args = ap.parse_args()
    for d in args.results_dirs:
        render_dir(d)


if __name__ == "__main__":
    main()
