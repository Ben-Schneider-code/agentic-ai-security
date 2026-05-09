#!/usr/bin/env python3
"""Count exact per-split, per-style sizes of the benign query pools.

Feeds the numbers that appear in the Benign Prompt Design section of the
paper. Also dumps the pools as a structured JSON artifact for release
alongside the code.

Self-contained: parses the two list literals directly out of
``blueteam_sql_env.py`` with ``ast`` so it does not need to import the
MARFT runtime (which pulls in torch/transformers).

Run:
    python util/benign_pool_stats.py

Writes to ``data/benign_pool_stats.json`` and prints a summary to stdout.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "MARFT" / "marft" / "envs" / "blueteam_sql" / "blueteam_sql_env.py"
OUTPUT = ROOT / "data" / "benign_pool_stats.json"

# Mirror of ``BlueTeamSQLEnv._classify_benign_style`` (kept in sync manually
# — if the live classifier changes, update this list).
_ADVERSARIAL_MARKERS = (
    "security team", "compliance", "audit", "gdpr", "urgent",
    "data protection", "dba team", "management has requested",
    "standard operating procedure", "reconciliation",
    "routine", "verification protocol", "identity check",
    "authentication confirmed", "access granted",
    "credentials", "system_notice", "debug_mode",
    "[usage analysis]", "[integrity check]", "[system diagnostic]",
    "[audit", "[system", "[data", "[account", "[order",
    "[line item", "[financial", "[product", "[final",
    "alert:", "pci", "legal hold", "emergency",
    "let me test", "testing the system", "testing query",
    "before we touch anything sensitive",
)


def classify(query) -> str:
    if isinstance(query, list):
        return "multi_turn"
    lower = query.lower()
    if any(marker in lower for marker in _ADVERSARIAL_MARKERS):
        return "adversarial"
    return "plain"


def load_pool_literals() -> tuple[list, list]:
    """Extract BENIGN_QUERIES and BENIGN_EVAL_QUERIES via ``ast``.

    ``ast.literal_eval`` handles nested list/str literals so the multi-turn
    sequences (list-of-str) round-trip correctly.
    """
    tree = ast.parse(ENV_FILE.read_text())
    pools: dict[str, list] = {}
    wanted = {"BENIGN_QUERIES", "BENIGN_EVAL_QUERIES"}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in wanted:
                pools[tgt.id] = ast.literal_eval(node.value)
    missing = wanted - pools.keys()
    if missing:
        raise RuntimeError(f"Could not find {missing} in {ENV_FILE}")
    return pools["BENIGN_QUERIES"], pools["BENIGN_EVAL_QUERIES"]


def split_stats(pool: list) -> dict:
    single = [q for q in pool if not isinstance(q, list)]
    multi = [q for q in pool if isinstance(q, list)]

    by_style = {"plain": 0, "adversarial": 0, "multi_turn": 0}
    for q in pool:
        by_style[classify(q)] += 1

    n_single = len(single) or 1
    return {
        "total_entries": len(pool),
        "single_turn_entries": len(single),
        "multi_turn_sequences": len(multi),
        "by_style": by_style,
        "single_turn_style_fractions": {
            "plain": round(by_style["plain"] / n_single, 4),
            "adversarial": round(by_style["adversarial"] / n_single, 4),
        },
    }


def dump_pool(pool: list) -> list:
    return [
        {"type": "multi_turn", "turns": q}
        if isinstance(q, list)
        else {"type": "single_turn", "text": q}
        for q in pool
    ]


def main() -> int:
    train_pool, eval_pool = load_pool_literals()

    summary = {
        "train": split_stats(train_pool),
        "eval": split_stats(eval_pool),
        "sampling_notes": {
            "style_is_single_turn_only": (
                "single-turn entries split between plain and adversarial; "
                "multi_turn is an episode-level construct (list-of-turn "
                "sequences) and is not comparable to the single-turn styles "
                "as a pool fraction."
            ),
        },
    }

    artifact = {
        "summary": summary,
        "train_pool": dump_pool(train_pool),
        "eval_pool": dump_pool(eval_pool),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
