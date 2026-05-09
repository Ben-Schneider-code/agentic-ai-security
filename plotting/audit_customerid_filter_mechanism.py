"""
Customerid-filter conditional breach audit (B.2.1, supervisor Q2 closeout).

Tests the inferred mechanism "blue's denylist passes through customerid-filtered
queries" by computing P(breach | blue's emitted SQL contains customerid=29485)
versus P(breach | does not). If the mechanism holds, the conditional breach
rate should be substantially higher when the filter is present.

Per problem_statement.tex eq.38, "breach" = outcome_tier == false_positive
(any honeypot accessed). We restrict the denominator to attack turns that
emitted an extractable SQL block in blue_action (the population at risk of
breach), per eq.49 conditioning on resource-accessing turns.

Sidecar: figures/customerid_filter_audit.json
PNG: optional summary bar chart figures/customerid_filter_audit.png

CLI:
    python plotting/audit_customerid_filter_mechanism.py \\
        --results results-<ID>[:Label] [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ._data import apply_paper_style, parse_results_arg
    from .plot_top_target_mechanism import _extract_sql_block, _parse_accessed
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import apply_paper_style, parse_results_arg
    from plotting.plot_top_target_mechanism import _extract_sql_block, _parse_accessed

apply_paper_style()

DESCRIPTION = (
    "P(breach | blue SQL contains customerid=29485) vs P(breach | does not). "
    "Tests the inferred Pillar 6 (b) mechanism that blue's denylist gates on "
    "column-name keywords without consulting row-restriction structure."
)

CUSTOMERID_FILTER_PATTERN = re.compile(r"customerid\s*=\s*29485", re.IGNORECASE)
WHERE_PATTERN = re.compile(r"\bwhere\b", re.IGNORECASE)


def _wilson_ci(k: int, n: int, z: float = 2.576) -> tuple[float, float]:
    """99% Wilson CI on a proportion (returns lo, hi as percentages)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, (centre - half) * 100), min(100.0, (centre + half) * 100))


def audit_customerid_filter(
    selfplay_dir: str,
    cross_eval_subdir: str = "cross_eval",
) -> dict:
    base = Path(selfplay_dir) / cross_eval_subdir / "pairings"
    if not base.exists():
        raise FileNotFoundError(f"no pairings dir at {base}")

    n_with_filter = 0
    n_with_filter_breach = 0
    n_no_filter = 0
    n_no_filter_breach = 0
    n_no_sql = 0
    n_total_attack = 0

    for pairing_dir in sorted(base.glob("red_*_blue_*")):
        jsonl = pairing_dir / "reward_debug.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("turn_type") != "attack":
                    continue
                n_total_attack += 1
                blue_action = r.get("blue_action", "") or ""
                sql = _extract_sql_block(blue_action)
                if not sql:
                    n_no_sql += 1
                    continue
                breached = bool(_parse_accessed(r.get("accessed_honeypots")))
                if CUSTOMERID_FILTER_PATTERN.search(sql):
                    n_with_filter += 1
                    if breached:
                        n_with_filter_breach += 1
                else:
                    n_no_filter += 1
                    if breached:
                        n_no_filter_breach += 1

    rate_with = (n_with_filter_breach / n_with_filter * 100) if n_with_filter else 0.0
    rate_no = (n_no_filter_breach / n_no_filter * 100) if n_no_filter else 0.0
    ci_with = _wilson_ci(n_with_filter_breach, n_with_filter)
    ci_no = _wilson_ci(n_no_filter_breach, n_no_filter)
    relative_risk = (rate_with / rate_no) if rate_no > 0 else float("inf")

    return {
        "description": DESCRIPTION,
        "n_total_attack_turns": n_total_attack,
        "n_no_sql_emitted": n_no_sql,
        "with_customerid_29485": {
            "n": n_with_filter,
            "n_breaches": n_with_filter_breach,
            "breach_rate_pct": round(rate_with, 2),
            "ci99_pct": [round(ci_with[0], 2), round(ci_with[1], 2)],
        },
        "without_customerid_29485": {
            "n": n_no_filter,
            "n_breaches": n_no_filter_breach,
            "breach_rate_pct": round(rate_no, 2),
            "ci99_pct": [round(ci_no[0], 2), round(ci_no[1], 2)],
        },
        "relative_risk_with_vs_without": round(relative_risk, 3),
        "interpretation": (
            "If P(breach | filter present) > P(breach | absent), the inferred "
            "mechanism (blue passes through customerid-filtered queries) holds. "
            "Reverse direction or near-parity rejects the mechanism."
        ),
    }


def render_summary(audit: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    cats = ["with customerid=29485", "without"]
    rates = [
        audit["with_customerid_29485"]["breach_rate_pct"],
        audit["without_customerid_29485"]["breach_rate_pct"],
    ]
    cis = [
        audit["with_customerid_29485"]["ci99_pct"],
        audit["without_customerid_29485"]["ci99_pct"],
    ]
    err_lo = [rates[i] - cis[i][0] for i in range(2)]
    err_hi = [cis[i][1] - rates[i] for i in range(2)]
    ax.bar(cats, rates, yerr=[err_lo, err_hi], capsize=8, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("Breach rate (%) on SQL-emitting attack turns")
    ax.set_title(
        "P(breach | blue SQL contains customerid=29485) vs P(breach | absent)\n"
        f"with: n={audit['with_customerid_29485']['n']}, "
        f"without: n={audit['without_customerid_29485']['n']}; 99% Wilson CIs"
    )
    for i, r in enumerate(rates):
        ax.text(i, r + 1.5, f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--cross-eval-subdir", default="cross_eval")
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()

    results = parse_results_arg(args.results)
    label, selfplay_dir = results[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_customerid_filter(selfplay_dir, args.cross_eval_subdir)
    sidecar_path = out_dir / "customerid_filter_audit.json"
    with open(sidecar_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"[customerid_filter_audit] saved {sidecar_path}")

    png_path = out_dir / "customerid_filter_audit.png"
    render_summary(audit, png_path)
    print(f"[customerid_filter_audit] saved {png_path}")

    print()
    print(f"  with filter: n={audit['with_customerid_29485']['n']:,}, "
          f"breach rate = {audit['with_customerid_29485']['breach_rate_pct']:.2f}% "
          f"({audit['with_customerid_29485']['ci99_pct']})")
    print(f"  no filter:   n={audit['without_customerid_29485']['n']:,}, "
          f"breach rate = {audit['without_customerid_29485']['breach_rate_pct']:.2f}% "
          f"({audit['without_customerid_29485']['ci99_pct']})")
    print(f"  relative risk = {audit['relative_risk_with_vs_without']:.3f}×")


if __name__ == "__main__":
    main()
