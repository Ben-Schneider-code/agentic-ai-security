#!/usr/bin/env python3
"""Cross-replicate comparison of redteam honeypot stats.

Strictly validates that N replicate runs (same experiment, different seeds)
share the same iteration set and the same ``total_honeypots`` universe, then
juxtaposes their per-iteration stats and surfaces cross-run deviations:

  - Per-iteration side-by-side metrics with min/max/range spread.
  - Per-iteration honeypot membership grid (which reps reached which honeypot).
  - Per-iteration "all conversations" length distribution juxtaposed.
  - Aggregate across reps:
      * unique honeypots per rep + coverage,
      * honeypots found in ALL reps,
      * honeypots found in SOME but not all reps (grouped by membership pattern,
        excluding singletons which are listed separately),
      * honeypots found in ONLY ONE rep (replicate-unique discoveries),
      * honeypots not found by ANY rep (when universe size is known),
      * cumulative coverage progression per rep with spread,
      * global new-access frequency comparison (top 20).

Usage:
    python util/compare_honeypot_stats.py \\
        --results-dir results-A --results-dir results-B [--results-dir ...]

Reps are auto-labeled rep0/rep1/... in argv order.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running as a script (``python util/compare_honeypot_stats.py``) from
# the repo root, matching the convention used by sibling util scripts.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from util.honeypot_stats import load_run_stats  # noqa: E402


_SEP = "=" * 72
_SUB = "-" * 72


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_strict(runs: list[dict]) -> tuple[list[int], int | None]:
    """Hard-fail on mismatched iteration sets or total_honeypots universes.

    Returns (sorted_iter_keys, shared_total_honeypots_or_None).
    """
    iter_sets = [frozenset(r["iter_stats"].keys()) for r in runs]
    if len(set(iter_sets)) != 1:
        print("ERROR: iteration indices differ across runs:", file=sys.stderr)
        for i, r in enumerate(runs):
            ks = sorted(r["iter_stats"].keys())
            print(f"  rep{i} ({r['results_dir'].name}): {ks}", file=sys.stderr)
        raise SystemExit(2)
    iters = sorted(next(iter(iter_sets)))

    totals = [r["total_honeypots"] for r in runs]
    known = [t for t in totals if t is not None]
    if known and len(set(known)) > 1:
        print("ERROR: total_honeypots differ across runs:", file=sys.stderr)
        for i, (r, t) in enumerate(zip(runs, totals)):
            print(f"  rep{i} ({r['results_dir'].name}): {t}", file=sys.stderr)
        raise SystemExit(2)
    total_hp = known[0] if known else None
    return iters, total_hp


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _spread(values: list[int]) -> str:
    nums = [v for v in values if v is not None]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    return f"[min={lo} max={hi} range={hi - lo}]"


def _fmt_pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "  - %"


def _rep_labels(n: int) -> list[str]:
    return [f"rep{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def print_header(runs: list[dict], iters: list[int], total_hp: int | None) -> None:
    print(_SEP)
    print("CROSS-RUN HONEYPOT COMPARISON")
    print(_SEP)
    for i, r in enumerate(runs):
        print(f"  rep{i}  ->  {r['results_dir']}")
    print()
    print(f"  iterations       : {iters}")
    if total_hp:
        print(f"  total_honeypots  : {total_hp}")
    else:
        print(f"  total_honeypots  : (unknown — no reward_config.yaml parsed)")


def print_iter_juxtaposition(
    k: int, runs: list[dict], total_hp: int | None
) -> None:
    n_reps = len(runs)
    stats = [r["iter_stats"][k] for r in runs]
    labels = _rep_labels(n_reps)

    print()
    print(_SEP)
    print(f"ITERATION {k}")
    print(_SEP)

    # --- Summary metric table -------------------------------------------
    label_w = 30
    col_w = 22

    rows: list[tuple[str, list[str], list[int]]] = []

    n_conv = [s["n_conversations"] for s in stats]
    rows.append(("Conversations total", [str(n) for n in n_conv], n_conv))

    n_sql = [s["n_with_sql"] for s in stats]
    rows.append(
        (
            "Conversations with SQL",
            [f"{c} ({_fmt_pct(c, n)})" for c, n in zip(n_sql, n_conv)],
            n_sql,
        )
    )

    n_hp = [s["n_with_honeypot"] for s in stats]
    rows.append(
        (
            "Conversations with honeypot",
            [f"{c} ({_fmt_pct(c, n)})" for c, n in zip(n_hp, n_conv)],
            n_hp,
        )
    )

    uniq = [len(s["unique_honeypots"]) for s in stats]
    if total_hp:
        uniq_cells = [f"{u}/{total_hp} ({100 * u / total_hp:.1f}%)" for u in uniq]
    else:
        uniq_cells = [str(u) for u in uniq]
    rows.append(("Unique honeypots reached", uniq_cells, uniq))

    # Header
    hdr = " " * 2 + " " * label_w + "".join(f"{lbl:<{col_w}}" for lbl in labels) + "spread"
    print(hdr)
    print("  " + "-" * (label_w + col_w * n_reps + len("spread")))
    for label, cells, nums in rows:
        print(
            f"  {label:<{label_w}}"
            + "".join(f"{c:<{col_w}}" for c in cells)
            + _spread(nums)
        )

    # --- Honeypot membership grid ---------------------------------------
    rep_hp_sets = [set(s["unique_honeypots"]) for s in stats]
    union_hp = set().union(*rep_hp_sets)
    print()
    print(f"  Honeypots reached this iter (n_union={len(union_hp)}):")
    if not union_hp:
        print("    (none)")
    else:
        hp_w = max(36, min(60, max(len(h) for h in union_hp)))
        col_label_w = 5  # "rep0 " width
        hdr = "    " + " " * hp_w + "  " + "  ".join(f"{lbl:<{col_label_w}}" for lbl in labels) + "  cnt"
        print(hdr)
        scored = sorted(
            union_hp,
            key=lambda hp: (
                -sum(1 for s in rep_hp_sets if hp in s),
                hp,
            ),
        )
        for hp in scored:
            cells = ["  ✓  " if hp in s else "  ·  " for s in rep_hp_sets]
            n_seen = sum(1 for s in rep_hp_sets if hp in s)
            row_cells = "  ".join(f"{c.strip():^{col_label_w}}" for c in cells)
            print(f"    {hp:<{hp_w}}  {row_cells}  {n_seen}/{n_reps}")

    # --- Conversation length distribution (all) -------------------------
    print()
    print(f"  Conversation length distribution (all):")
    all_lens = sorted(set().union(*(s["conv_length_dist"].keys() for s in stats)))
    if not all_lens:
        print("    (no conversations)")
        return
    cell_w = 14
    hdr = "    " + f"{'Length':<8}" + "  ".join(f"{lbl:^{cell_w}}" for lbl in labels)
    print(hdr)
    totals_per_rep = [sum(s["conv_length_dist"].values()) for s in stats]
    for L in all_lens:
        cells = []
        for s, tot in zip(stats, totals_per_rep):
            cnt = s["conv_length_dist"].get(L, 0)
            pct = 100 * cnt / tot if tot else 0.0
            cells.append(f"{cnt:>3} ({pct:5.1f}%)")
        print(f"    {L:<8}" + "  ".join(f"{c:^{cell_w}}" for c in cells))


def print_aggregate(
    runs: list[dict], iters: list[int], total_hp: int | None
) -> None:
    n_reps = len(runs)
    labels = _rep_labels(n_reps)

    rep_hp_sets: list[set[str]] = []
    rep_hp_freq: list[Counter] = []
    for r in runs:
        hp_set: set[str] = set()
        freq: Counter = Counter()
        for k in iters:
            s = r["iter_stats"][k]
            hp_set.update(s["unique_honeypots"])
            for hp, c in s["honeypot_freq"].items():
                freq[hp] += c
        rep_hp_sets.append(hp_set)
        rep_hp_freq.append(freq)

    print()
    print(_SEP)
    print("AGGREGATE ACROSS REPS")
    print(_SEP)

    # --- Per-rep unique discovery counts --------------------------------
    print("\n  Unique honeypots discovered (across all iterations):")
    counts = [len(s) for s in rep_hp_sets]
    for i, c in enumerate(counts):
        cov = f"  ({100 * c / total_hp:.1f}% of {total_hp})" if total_hp else ""
        print(f"    rep{i}: {c}{cov}")
    print(f"    {_spread(counts)}")

    # --- Membership breakdown ------------------------------------------
    all_seen = set().union(*rep_hp_sets) if rep_hp_sets else set()

    def mask_of(hp: str) -> tuple[bool, ...]:
        return tuple(hp in s for s in rep_hp_sets)

    found_all = sorted(hp for hp in all_seen if all(mask_of(hp)))
    print(f"\n  Found in ALL {n_reps} reps: {len(found_all)}")
    for hp in found_all:
        print(f"    {hp}")

    # Partial: in 2..n_reps-1 reps (i.e., not unanimous, not singleton)
    partial: dict[tuple[bool, ...], list[str]] = {}
    singletons: dict[int, list[str]] = {i: [] for i in range(n_reps)}
    for hp in sorted(all_seen):
        m = mask_of(hp)
        s = sum(m)
        if s == n_reps:
            continue
        if s == 1:
            singletons[m.index(True)].append(hp)
        else:
            partial.setdefault(m, []).append(hp)

    if partial:
        n_partial = sum(len(v) for v in partial.values())
        print(f"\n  Found in SOME but not all reps: {n_partial}")
        # Sort patterns by descending popularity, then mask order.
        for mask in sorted(partial, key=lambda m: (-sum(m), m)):
            members = ",".join(labels[i] for i, x in enumerate(mask) if x)
            hps = partial[mask]
            print(f"    [{members}]  ({len(hps)}):")
            for hp in hps:
                print(f"      {hp}")
    else:
        print(f"\n  Found in SOME but not all reps: 0")

    n_singletons = sum(len(v) for v in singletons.values())
    print(f"\n  Found in ONLY ONE rep (replicate-unique): {n_singletons}")
    for i in range(n_reps):
        if singletons[i]:
            print(f"    rep{i} (×{len(singletons[i])}):")
            for hp in singletons[i]:
                print(f"      {hp}")
        else:
            print(f"    rep{i}: (none)")

    if total_hp:
        n_missed = total_hp - len(all_seen)
        print(
            f"\n  Not found in ANY rep: {n_missed}"
            f"  (universe={total_hp}, union_seen={len(all_seen)})"
        )

    # --- Coverage progression -------------------------------------------
    print("\n  Coverage progression (cumulative unique honeypots per rep):")
    cell_w = 10
    print(
        "    "
        + f"{'iter':<6}"
        + "  ".join(f"{lbl:^{cell_w}}" for lbl in labels)
        + "  spread"
    )
    running: list[set[str]] = [set() for _ in range(n_reps)]
    for k in iters:
        cur = []
        for i, r in enumerate(runs):
            running[i].update(r["iter_stats"][k]["unique_honeypots"])
            cur.append(len(running[i]))
        cells = []
        for c in cur:
            if total_hp:
                cells.append(f"{c}/{total_hp}")
            else:
                cells.append(str(c))
        print(
            "    "
            + f"{k:<6}"
            + "  ".join(f"{c:^{cell_w}}" for c in cells)
            + f"  {_spread(cur)}"
        )

    # --- Global frequency comparison ------------------------------------
    union_freq: Counter = Counter()
    for f in rep_hp_freq:
        for hp, c in f.items():
            union_freq[hp] += c
    top_n = min(20, len(union_freq))
    print(f"\n  Global new-access frequency (top {top_n} by total across reps):")
    if not union_freq:
        print("    (no honeypot accesses recorded)")
        return
    hp_w = max(36, min(60, max(len(h) for h in union_freq)))
    cell_w = 6
    print(
        "    "
        + f"{'honeypot':<{hp_w}}  "
        + "  ".join(f"{lbl:^{cell_w}}" for lbl in labels)
        + "  total"
    )
    for hp, _tot in union_freq.most_common(top_n):
        per_rep = [f.get(hp, 0) for f in rep_hp_freq]
        cells = "  ".join(f"{c:^{cell_w}}" for c in per_rep)
        print(f"    {hp:<{hp_w}}  {cells}  {sum(per_rep):>5}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--results-dir",
        action="append",
        type=Path,
        required=True,
        metavar="DIR",
        help="Replicate run directory; pass once per rep (>= 2 required).",
    )
    args = ap.parse_args()
    paths: list[Path] = args.results_dir
    if len(paths) < 2:
        print(
            "ERROR: need at least 2 --results-dir arguments to compare.",
            file=sys.stderr,
        )
        return 2

    runs: list[dict] = []
    for i, p in enumerate(paths):
        print(f"[rep{i}] loading {p} ...", flush=True)
        try:
            runs.append(load_run_stats(p))
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 1

    iters, total_hp = validate_strict(runs)
    print()
    print_header(runs, iters, total_hp)
    for k in iters:
        print_iter_juxtaposition(k, runs, total_hp)
    print_aggregate(runs, iters, total_hp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
