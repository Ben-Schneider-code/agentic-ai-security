#!/usr/bin/env python3
"""Honeypot statistics from redteam training iterations.

Reads reward_debug.jsonl from each iter_*/redteam iteration and prints:
  - Conversations (correctly reconstructed per rank + step_count reset)
  - Unique honeypots achieved and their frequency
  - Conversation length distribution
  - Outcome tier distribution
  - Coverage progression across iterations

The ``episode`` field in reward_debug.jsonl is the outer training-loop
counter, not a per-conversation ID — multiple conversations per rank
can share the same episode value. Conversations are reconstructed by
grouping records per rank (sorted by timestamp) and splitting on
step_count == 1 (env reset).

Usage:
    python util/honeypot_stats.py --results-dir <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml

    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_iteration_logs(results_dir: Path) -> dict[int, Path]:
    """Map iteration index → reward_debug.jsonl for redteam."""
    iters: dict[int, Path] = {}
    for iter_dir in sorted(results_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        try:
            k = int(iter_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        candidates = list(iter_dir.glob("redteam/**/debug_logs/reward_debug.jsonl"))
        if candidates:
            iters[k] = candidates[0]
    return iters


def find_total_honeypots(results_dir: Path, iter_k: int) -> int | None:
    """Return total_honeypots from reward_config.yaml if available."""
    if not _HAVE_YAML:
        return None
    iter_dir = results_dir / f"iter_{iter_k}"
    candidates = list(iter_dir.glob("redteam/**/reward_config.yaml"))
    if not candidates:
        return None
    try:
        with candidates[0].open() as f:
            cfg = yaml.safe_load(f)
        return int(cfg.get("total_honeypots", 0)) or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Conversation reconstruction
# ---------------------------------------------------------------------------


def reconstruct_conversations(records: list[dict]) -> list[list[dict]]:
    """Split log records into conversations.

    Groups records by rank, sorts each group by timestamp, then splits
    on step_count == 1 (signals an env reset / new conversation start).
    """
    by_rank: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        rank = int(r.get("rank", 0))
        by_rank[rank].append(r)

    conversations: list[list[dict]] = []
    for rank in sorted(by_rank):
        rank_records = sorted(by_rank[rank], key=lambda r: r.get("timestamp", 0.0))
        current: list[dict] = []
        for r in rank_records:
            step = int(r.get("step_count", 1))
            if step == 1 and current:
                conversations.append(current)
                current = []
            current.append(r)
        if current:
            conversations.append(current)

    return conversations


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


_HONEYPOT_TIERS = {"honeypot_access", "repeated_honeypot"}
_NO_SQL_TIERS = {"no_sql", ""}


def compute_iter_stats(
    conversations: list[list[dict]], total_honeypots: int | None
) -> dict:
    n_conv = len(conversations)
    n_with_honeypot = 0
    n_with_sql = 0
    honeypot_counter: Counter = Counter()
    conv_length_counter: Counter = Counter()
    conv_length_honeypot_counter: Counter = Counter()
    conv_length_sql_counter: Counter = Counter()
    outcome_counter: Counter = Counter()

    for conv in conversations:
        max_step = 0
        had_honeypot = False
        had_sql = False
        for r in conv:
            step = int(r.get("step_count", 1))
            max_step = max(max_step, step)

            tier = r.get("outcome_tier", "")
            if tier:
                outcome_counter[tier] += 1
            if tier in _HONEYPOT_TIERS:
                had_honeypot = True
            if tier not in _NO_SQL_TIERS:
                had_sql = True

            for hp in r.get("new_honeypots_accessed") or []:
                if isinstance(hp, str) and hp:
                    honeypot_counter[hp] += 1

        conv_length_counter[max_step] += 1
        if had_honeypot:
            n_with_honeypot += 1
            conv_length_honeypot_counter[max_step] += 1
        if had_sql:
            n_with_sql += 1
            conv_length_sql_counter[max_step] += 1

    unique_honeypots = sorted(honeypot_counter.keys())
    coverage = (
        len(unique_honeypots) / total_honeypots * 100
        if total_honeypots
        else None
    )

    return {
        "n_conversations": n_conv,
        "n_with_honeypot": n_with_honeypot,
        "n_with_sql": n_with_sql,
        "unique_honeypots": unique_honeypots,
        "honeypot_freq": honeypot_counter,
        "conv_length_dist": conv_length_counter,
        "conv_length_honeypot_dist": conv_length_honeypot_counter,
        "conv_length_sql_dist": conv_length_sql_counter,
        "outcome_dist": outcome_counter,
        "total_honeypots": total_honeypots,
        "coverage_pct": coverage,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_SEP = "=" * 60


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "#" * filled


def print_iter_stats(k: int, path: Path, stats: dict) -> None:
    n = stats["n_conversations"]
    n_hp = stats["n_with_honeypot"]
    unique_hp = stats["unique_honeypots"]
    hp_freq = stats["honeypot_freq"]
    conv_dist = stats["conv_length_dist"]
    outcome_dist = stats["outcome_dist"]
    total_hp = stats["total_honeypots"]
    coverage = stats["coverage_pct"]

    print(f"\n{_SEP}")
    print(f"ITERATION {k}  [{path}]")
    print(_SEP)

    n_sql = stats["n_with_sql"]
    pct_hp = 100 * n_hp / n if n else 0.0
    pct_sql = 100 * n_sql / n if n else 0.0
    print(f"  Conversations total         : {n}")
    print(f"  Conversations with SQL      : {n_sql}  ({pct_sql:.1f}%)")
    print(f"  Conversations with honeypot : {n_hp}  ({pct_hp:.1f}%)")

    if total_hp:
        print(
            f"  Unique honeypots reached    : {len(unique_hp)} / {total_hp}"
            f"  ({coverage:.1f}% coverage)"
        )
    else:
        print(f"  Unique honeypots reached    : {len(unique_hp)}")

    if hp_freq:
        print(f"\n  Honeypot frequency (new first-access events):")
        for hp, count in hp_freq.most_common():
            print(f"    {hp:<48} {count:>4}")
    else:
        print("\n  No honeypots accessed in this iteration.")

    print(f"\n  Conversation length distribution (all):")
    total_conv = sum(conv_dist.values())
    for length in sorted(conv_dist):
        count = conv_dist[length]
        pct = 100 * count / total_conv if total_conv else 0.0
        print(
            f"    Length {length}: {count:>5}  ({pct:5.1f}%)  {_bar(pct)}"
        )

    sql_len_dist = stats["conv_length_sql_dist"]
    print(f"\n  Conversation length distribution (any SQL, n={n_sql}):")
    if sql_len_dist:
        for length in sorted(sql_len_dist):
            count = sql_len_dist[length]
            pct = 100 * count / n_sql if n_sql else 0.0
            print(f"    Length {length}: {count:>5}  ({pct:5.1f}%)  {_bar(pct)}")
    else:
        print("    (none)")

    hp_len_dist = stats["conv_length_honeypot_dist"]
    print(f"\n  Conversation length distribution (honeypot tier, n={n_hp}):")
    if hp_len_dist:
        for length in sorted(hp_len_dist):
            count = hp_len_dist[length]
            pct = 100 * count / n_hp if n_hp else 0.0
            print(f"    Length {length}: {count:>5}  ({pct:5.1f}%)  {_bar(pct)}")
    else:
        print("    (none)")

    if outcome_dist:
        print(f"\n  Outcome tier distribution (step-level):")
        for tier, count in outcome_dist.most_common():
            print(f"    {tier:<30} : {count:>5}")


def print_aggregate(iter_stats: dict[int, dict]) -> None:
    print(f"\n{_SEP}")
    print("AGGREGATE (all iterations)")
    print(_SEP)

    # Collect global honeypot info
    global_freq: Counter = Counter()
    first_seen: dict[str, int] = {}
    running_unique: list[tuple[int, int]] = []
    seen_so_far: set[str] = set()

    total_hp: int | None = None
    for k in sorted(iter_stats):
        stats = iter_stats[k]
        if total_hp is None and stats["total_honeypots"]:
            total_hp = stats["total_honeypots"]
        for hp, count in stats["honeypot_freq"].items():
            global_freq[hp] += count
            if hp not in first_seen:
                first_seen[hp] = k
        seen_so_far.update(stats["unique_honeypots"])
        running_unique.append((k, len(seen_so_far)))

    n_unique = len(global_freq)
    if total_hp:
        print(f"  Unique honeypots across all iterations: {n_unique} / {total_hp}  ({100*n_unique/total_hp:.1f}%)")
    else:
        print(f"  Unique honeypots across all iterations: {n_unique}")

    if first_seen:
        print(f"\n  First discovered in iteration:")
        for hp in sorted(first_seen, key=lambda h: (first_seen[h], h)):
            print(f"    {hp:<48}  →  iter {first_seen[hp]}")

        print(f"\n  Global honeypot frequency:")
        for hp, count in global_freq.most_common():
            print(f"    {hp:<48} {count:>4}")

    if running_unique:
        print(f"\n  Coverage progression:")
        for k, n in running_unique:
            cov = f"  ({100*n/total_hp:.1f}%)" if total_hp else ""
            print(f"    iter {k}: {n}{(' / ' + str(total_hp)) if total_hp else ''}{cov}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    logs = find_iteration_logs(results_dir)
    if not logs:
        print(
            f"No iter_*/redteam debug logs found under {results_dir}",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(logs)} iteration(s) in {results_dir}")

    iter_stats: dict[int, dict] = {}
    for k, path in sorted(logs.items()):
        print(f"  [iter {k}] reading {path} ...", end=" ", flush=True)
        records = load_records(path)
        conversations = reconstruct_conversations(records)
        total_hp = find_total_honeypots(results_dir, k)
        stats = compute_iter_stats(conversations, total_hp)
        iter_stats[k] = stats
        print(f"{len(records)} records → {len(conversations)} conversations")

    for k, stats in sorted(iter_stats.items()):
        print_iter_stats(k, logs[k], stats)

    print_aggregate(iter_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
