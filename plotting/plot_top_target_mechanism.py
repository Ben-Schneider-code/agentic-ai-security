"""
Top-target attack mechanism: pattern decomposition + exemplar quotes.

Answers supervisor Q2 with concrete examples: what does a successful phone /
emailaddress / passwordhash breach actually look like?

For each top target:
  - Stacked bar of SQL pattern classes (direct / join / subquery / etc.)
    among breaches landing on that target.
  - Sidecar JSON with up to 5 representative (user_message, blue_action)
    exemplars per target — for quoting in the paper.

Source: results-<ID>/cross_eval/pairings/red_*_blue_*/reward_debug.jsonl,
filtered to rows where accessed_honeypots contains the target.

CLI:
    python plotting/plot_top_target_mechanism.py \\
        --results results-<ID>[:Label] \\
        [--targets column_access:phone column_access:emailaddress column_access:passwordhash] \\
        [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import apply_paper_style, parse_results_arg
    from .plot_attack_evolution import _classify_sql, PATTERN_LABELS, PATTERN_COLORS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import apply_paper_style, parse_results_arg
    from plotting.plot_attack_evolution import _classify_sql, PATTERN_LABELS, PATTERN_COLORS

apply_paper_style()

DESCRIPTION = "Top-target breach mechanism: SQL pattern composition + representative exemplars"

DEFAULT_TARGETS = [
    "column_access:phone",
    "column_access:emailaddress",
    "column_access:passwordhash",
]


def _parse_accessed(field) -> set[str]:
    out: set[str] = set()
    for item in field or []:
        try:
            hp = ast.literal_eval(item) if isinstance(item, str) else item
            if hp.get("accessed"):
                out.add(f"{hp.get('type', '')}:{hp.get('identifier', '')}")
        except Exception:
            pass
    return out


def _extract_sql_block(blue_action: str) -> str:
    if not blue_action:
        return ""
    m = re.search(r"```(?:sql)?\s*(.*?)```", blue_action, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _truncate(text: str, max_chars: int = 220) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def collect_breaches(
    selfplay_dir: str,
    targets: list[str],
    cross_eval_subdir: str = "cross_eval",
) -> dict[str, list[dict]]:
    base = Path(selfplay_dir) / cross_eval_subdir / "pairings"
    if not base.exists():
        return {t: [] for t in targets}

    bag: dict[str, list[dict]] = {t: [] for t in targets}
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
                accessed = _parse_accessed(r.get("accessed_honeypots"))
                if not accessed:
                    continue
                hits = accessed & set(targets)
                if not hits:
                    continue
                user_msg = r.get("user_message", "") or ""
                blue_action = r.get("blue_action", "") or ""
                sql = _extract_sql_block(blue_action)
                cls = _classify_sql(blue_action)
                for t in hits:
                    bag[t].append(
                        {
                            "pairing": pairing_dir.name,
                            "episode": r.get("episode"),
                            "step": r.get("step"),
                            "red_iter": r.get("red_iter"),
                            "blue_iter": r.get("blue_iter"),
                            "sql_pattern": cls,
                            "user_message_excerpt": _truncate(user_msg, 400),
                            "extracted_sql": _truncate(sql, 400),
                        }
                    )
    return bag


def plot_top_target_mechanism(
    results: list[tuple[str, str]],
    targets: list[str] | None = None,
    out_dir: str = "figures/",
    cross_eval_subdir: str = "cross_eval",
    n_exemplars: int = 5,
) -> Path:
    if targets is None:
        targets = DEFAULT_TARGETS
    out_path = Path(out_dir) / "top_target_attack_mechanism.png"
    sidecar_path = Path(out_dir) / "top_target_attack_mechanism.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label, selfplay_dir = results[0]
    bag = collect_breaches(selfplay_dir, targets, cross_eval_subdir)

    # Build pattern matrix
    pattern_classes = ["direct_select", "join", "subquery", "union", "catalog", "none"]
    pattern_pct = np.zeros((len(pattern_classes), len(targets)), dtype=float)
    n_per_target: list[int] = []
    for j, t in enumerate(targets):
        rows = bag[t]
        n = len(rows)
        n_per_target.append(n)
        if n == 0:
            continue
        cls_counts = Counter(r["sql_pattern"] for r in rows)
        for i, cls in enumerate(pattern_classes):
            pattern_pct[i, j] = cls_counts.get(cls, 0) / n * 100

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(targets))
    bottoms = np.zeros(len(targets))
    for i, cls in enumerate(pattern_classes):
        ax.bar(
            x,
            pattern_pct[i],
            bottom=bottoms,
            color=PATTERN_COLORS[cls],
            label=PATTERN_LABELS[cls],
        )
        bottoms += pattern_pct[i]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("column_access:", "col:").replace("table_access:", "tbl:") for t in targets],
        fontsize=9,
    )
    ax.set_ylabel("% of breaches on this target")
    ax.set_title(
        f"SQL pattern decomposition of successful breaches on top targets\n"
        f"(canonical cross_eval; n breaches per target shown above bars)"
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(0, 105)
    for j, n in enumerate(n_per_target):
        ax.text(j, 102, f"n={n}", ha="center", fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Mode-aware exemplar sampling: take one exemplar per sql_pattern in modal
    # order, then fill remaining slots from the modal pattern. Replaces the
    # earlier stride-by-pairing-sort selector which produced exemplar lists
    # that did not represent the modal mechanism.
    def _select_exemplars(rows: list[dict], n: int) -> list[dict]:
        if not rows:
            return []
        pat_counts = Counter(r["sql_pattern"] for r in rows)
        seen_patterns: set[str] = set()
        selected: list[dict] = []
        for pat, _ in pat_counts.most_common():
            for r in rows:
                if r["sql_pattern"] == pat:
                    selected.append(r)
                    seen_patterns.add(pat)
                    break
            if len(selected) >= n:
                return selected[:n]
        # Fill remainder from the modal pattern
        modal_pat = pat_counts.most_common(1)[0][0]
        for r in rows:
            if len(selected) >= n:
                break
            if r["sql_pattern"] == modal_pat and r not in selected:
                selected.append(r)
        return selected[:n]

    # Where-clause sub-classification for each target (B.2.1 / A.2.3 — replaces
    # inferred customerid-filter mechanism phrasing with measured percentages)
    def _where_breakdown(rows: list[dict]) -> dict[str, float]:
        if not rows:
            return {"with_customerid_29485": 0.0, "no_where_clause": 0.0, "other_where": 0.0, "n": 0}
        n_with_cust = 0
        n_no_where = 0
        n_other = 0
        for r in rows:
            sql = (r.get("extracted_sql") or "").lower()
            has_where = bool(re.search(r"\bwhere\b", sql))
            has_cust_29485 = bool(re.search(r"customerid\s*=\s*29485", sql))
            if has_cust_29485:
                n_with_cust += 1
            elif not has_where:
                n_no_where += 1
            else:
                n_other += 1
        n = len(rows)
        return {
            "with_customerid_29485": round(n_with_cust / n * 100, 2),
            "no_where_clause": round(n_no_where / n * 100, 2),
            "other_where": round(n_other / n * 100, 2),
            "n": n,
        }

    sidecar = {
        "description": DESCRIPTION,
        "targets": targets,
        "n_breaches_per_target": {t: len(bag[t]) for t in targets},
        "sql_pattern_pct_per_target": {
            t: {cls: round(float(pattern_pct[i, j]), 2) for i, cls in enumerate(pattern_classes)}
            for j, t in enumerate(targets)
        },
        "where_clause_breakdown_pct": {t: _where_breakdown(bag[t]) for t in targets},
        "exemplars": {t: _select_exemplars(bag[t], n_exemplars) for t in targets},
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[top_target_mechanism] saved {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--cross-eval-subdir", default="cross_eval")
    p.add_argument("--out-dir", default="figures/")
    p.add_argument("--n-exemplars", type=int, default=5)
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_top_target_mechanism(results, args.targets, args.out_dir, args.cross_eval_subdir, args.n_exemplars)


if __name__ == "__main__":
    main()
