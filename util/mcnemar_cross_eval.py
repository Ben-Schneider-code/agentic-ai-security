#!/usr/bin/env python3
"""Pairwise significance tests on conversation-level violation outcomes
across blue-team versions (for fixed red) and across red-team versions
(for fixed blue) from cross-evaluation output.

Background. The cross-evaluation RNG seed is
``seed + red_iter * 1000 + blue_iter`` (``cross_evaluate.py:416``), so
episode-level attack/benign assignments differ across pairings. Attack
content additionally varies because red-team generation is stochastic
(temperature 0.7). A strict matched-pair McNemar assumption therefore
does not hold post-hoc. We report:

  1. McNemar (with continuity correction) on the intersection of episode
     indices where both pairings happened to sample the same turn_type.
     This is the closest proxy to a matched-pair test the existing
     artifacts support; power is reduced by the size of the aligned
     subset.

  2. A two-sided two-proportion z-test on the full per-pairing
     episode-level violation rate (unpaired). This is the design-honest
     headline test.

Both are written to CSV and to a JSON matrix keyed by pairing so that
``util/plot_cross_eval.py`` can annotate heatmap cells with ``p > 0.05``
in a distinct style (the paper's §sec:episode-protocol promise).

Run:
    python util/mcnemar_cross_eval.py <cross_eval_dir>

Outputs (in <cross_eval_dir>):
    significance_tests.csv
    significance_matrix.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from scipy import stats


def load_pairing(pairing_dir: Path) -> dict:
    """Load one pairing's reward_debug.jsonl → per-episode attack outcomes.

    Returns a dict with:
        episodes: {episode_idx: {"turn_type": str, "violation": 0|1}}
            For attack episodes, violation=1 if any step has outcome_tier
            ``false_positive`` (the honeypot-access tier). For benign
            episodes, violation is 0 by construction; we still record
            turn_type so callers can align.
        red_iter: int; blue_iter: int
    """
    jsonl = pairing_dir / "reward_debug.jsonl"
    summary = pairing_dir / "summary.json"
    episodes: dict[int, dict] = {}
    with jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            ep = r["episode"]
            slot = episodes.setdefault(
                ep, {"turn_type": r["turn_type"], "violation": 0}
            )
            if r["turn_type"] == "attack" and r.get("outcome_tier") == "false_positive":
                slot["violation"] = 1

    with summary.open() as f:
        s = json.load(f)

    return {
        "pairing_key": s["pairing_key"],
        "red_iter": s["red_iter"],
        "blue_iter": s["blue_iter"],
        "episodes": episodes,
    }


def mcnemar_on_aligned(a: dict, b: dict) -> tuple[int, int, int, float | None, str]:
    """McNemar on the subset of episode indices where both pairings saw
    the same turn_type (attack). Returns (n_aligned, disc_b, disc_c,
    p_value_or_None, method_tag).
    """
    aligned = [
        ep for ep, r in a["episodes"].items()
        if ep in b["episodes"]
        and r["turn_type"] == "attack"
        and b["episodes"][ep]["turn_type"] == "attack"
    ]
    disc_b = sum(
        1 for ep in aligned
        if a["episodes"][ep]["violation"] == 0 and b["episodes"][ep]["violation"] == 1
    )
    disc_c = sum(
        1 for ep in aligned
        if a["episodes"][ep]["violation"] == 1 and b["episodes"][ep]["violation"] == 0
    )
    n = disc_b + disc_c
    if n == 0:
        return len(aligned), disc_b, disc_c, None, "mcnemar:no_disc"
    if n < 25:
        p = stats.binomtest(min(disc_b, disc_c), n, 0.5, alternative="two-sided").pvalue
        return len(aligned), disc_b, disc_c, p, "mcnemar:exact"
    chi2 = (abs(disc_b - disc_c) - 1) ** 2 / n
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return len(aligned), disc_b, disc_c, p, "mcnemar:cc_chi2"


def two_proportion_z(a: dict, b: dict) -> tuple[int, int, int, int, float | None, str]:
    """Unpaired two-sided two-proportion test on attack-episode violation
    rates. Returns (n_a, k_a, n_b, k_b, p_value_or_None, method_tag).
    Uses Fisher's exact for n<40, else normal approximation of the pooled
    proportion.
    """
    a_attack = [r for r in a["episodes"].values() if r["turn_type"] == "attack"]
    b_attack = [r for r in b["episodes"].values() if r["turn_type"] == "attack"]
    n_a, n_b = len(a_attack), len(b_attack)
    k_a = sum(r["violation"] for r in a_attack)
    k_b = sum(r["violation"] for r in b_attack)
    if n_a == 0 or n_b == 0:
        return n_a, k_a, n_b, k_b, None, "two_prop:insufficient"
    if n_a + n_b < 40:
        table = [[k_a, n_a - k_a], [k_b, n_b - k_b]]
        _, p = stats.fisher_exact(table, alternative="two-sided")
        return n_a, k_a, n_b, k_b, p, "two_prop:fisher"
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return n_a, k_a, n_b, k_b, 1.0, "two_prop:zero_se"
    z = (k_a / n_a - k_b / n_b) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return n_a, k_a, n_b, k_b, p, "two_prop:normal"


def run_tests(cross_eval_dir: Path, output_dir: Path | None = None) -> None:
    pairings_dir = cross_eval_dir / "pairings"
    output_dir = output_dir or cross_eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pairings_dir.is_dir():
        print(f"ERROR: {pairings_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    pairings = sorted(p for p in pairings_dir.iterdir() if p.is_dir())
    if not pairings:
        print(f"ERROR: no pairings found in {pairings_dir}", file=sys.stderr)
        sys.exit(1)

    data = {}
    for p in pairings:
        try:
            d = load_pairing(p)
            data[d["pairing_key"]] = d
        except Exception as e:
            print(f"  [warn] skipping {p.name}: {e}", file=sys.stderr)

    rows = []
    matrix = {
        "blue_vs_blue": {},  # keyed by red_iter → list of {a,b,p,...}
        "red_vs_red": {},    # keyed by blue_iter
    }

    # Blue-vs-blue at fixed red
    by_red: dict[int, list[dict]] = {}
    for d in data.values():
        by_red.setdefault(d["red_iter"], []).append(d)
    for red_iter, group in by_red.items():
        group = sorted(group, key=lambda x: x["blue_iter"])
        entries = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                n_al, db, dc, p_mc, tag_mc = mcnemar_on_aligned(a, b)
                n_a, k_a, n_b, k_b, p_un, tag_un = two_proportion_z(a, b)
                row = {
                    "comparison": "blue_vs_blue",
                    "fixed": f"red_iter={red_iter}",
                    "a": a["pairing_key"],
                    "b": b["pairing_key"],
                    "n_aligned_attack": n_al,
                    "disc_b": db,
                    "disc_c": dc,
                    "p_mcnemar": p_mc,
                    "mcnemar_method": tag_mc,
                    "n_a_attack": n_a,
                    "k_a_violations": k_a,
                    "n_b_attack": n_b,
                    "k_b_violations": k_b,
                    "p_two_prop": p_un,
                    "two_prop_method": tag_un,
                }
                rows.append(row)
                entries.append(row)
        matrix["blue_vs_blue"][str(red_iter)] = entries

    # Red-vs-red at fixed blue
    by_blue: dict[int, list[dict]] = {}
    for d in data.values():
        by_blue.setdefault(d["blue_iter"], []).append(d)
    for blue_iter, group in by_blue.items():
        group = sorted(group, key=lambda x: x["red_iter"])
        entries = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                n_al, db, dc, p_mc, tag_mc = mcnemar_on_aligned(a, b)
                n_a, k_a, n_b, k_b, p_un, tag_un = two_proportion_z(a, b)
                row = {
                    "comparison": "red_vs_red",
                    "fixed": f"blue_iter={blue_iter}",
                    "a": a["pairing_key"],
                    "b": b["pairing_key"],
                    "n_aligned_attack": n_al,
                    "disc_b": db,
                    "disc_c": dc,
                    "p_mcnemar": p_mc,
                    "mcnemar_method": tag_mc,
                    "n_a_attack": n_a,
                    "k_a_violations": k_a,
                    "n_b_attack": n_b,
                    "k_b_violations": k_b,
                    "p_two_prop": p_un,
                    "two_prop_method": tag_un,
                }
                rows.append(row)
                entries.append(row)
        matrix["red_vs_red"][str(blue_iter)] = entries

    csv_path = output_dir / "significance_tests.csv"
    with csv_path.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    json_path = output_dir / "significance_matrix.json"
    with json_path.open("w") as f:
        json.dump(matrix, f, indent=2)

    n_sig_mc = sum(1 for r in rows if r["p_mcnemar"] is not None and r["p_mcnemar"] < 0.05)
    n_sig_un = sum(1 for r in rows if r["p_two_prop"] is not None and r["p_two_prop"] < 0.05)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(
        f"{len(rows)} pairwise comparisons · "
        f"{n_sig_mc} McNemar p<0.05 · "
        f"{n_sig_un} two-prop p<0.05"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cross_eval_dir", type=Path, help="Path to cross_eval output dir")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write significance_tests.csv and significance_matrix.json "
             "(defaults to cross_eval_dir; useful when cross_eval_dir is read-only)",
    )
    args = ap.parse_args()
    run_tests(args.cross_eval_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
