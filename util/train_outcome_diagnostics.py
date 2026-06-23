#!/usr/bin/env python3
"""Training-time outcome diagnostics for self-play runs (Q1 + behavioral half of Q2).

For each results dir, each trained iteration, and each side (red / blue), print the
per-turn ``outcome_tier`` distribution from training-time ``reward_debug.jsonl``:

  RED  — outcome_tier distribution with ``sql_error`` highlighted, honeypot-hit count
         (the terminal-reward learning signal), #PPO updates (unique ``episode`` values)
         and the warmup split read straight from the logged ``in_warmup_period`` field.
         Plus a SECONDARY, clearly-labelled heuristic attribution of ``execution_result``
         (guardrail "dangerous query" marker vs real SQL error vs no-error) to explain
         WHY the canonical sql_error rate is what it is. The canonical rate is always the
         logged ``outcome_tier == "sql_error"`` — the heuristic is diagnostic colour only.

  BLUE — training turns (``is_eval == False``) split by ``turn_type``: benign turns as
         true_positive vs false_negative (over-refusal), attack turns as
         true_negative / neutral_sql / false_positive.

Output is stdout tables only. Reuses plotting/_data.py and util/_diag_common.py; never
writes into a results dir.

Usage:
    python util/train_outcome_diagnostics.py RESULTS_DIR [RESULTS_DIR ...] [--early-late-trend]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting._data import load_reward_debug_lines, wilson_ci_pct  # noqa: E402
from util._diag_common import iter_runs, load_run_summary, note  # noqa: E402

# Display order for red outcome tiers; success tiers first, then degrading outcomes.
RED_TIER_ORDER = [
    "honeypot_access",
    "repeated_honeypot",
    "sensitive_query",
    "sql_generated",
    "sql_error",
    "blocked_attempt",
    "no_sql",
]
RED_SUCCESS_TIERS = {"honeypot_access", "repeated_honeypot"}
BENIGN_TIERS = ["true_positive", "false_negative"]
ATTACK_TIERS = ["true_negative", "neutral_sql", "false_positive", "sql_error"]

# Plain-text turn delimiters a leaking red policy emits past its own turn (mirrors
# marft.mas.prompt_format.RED_TURN_STOPS) plus the decoded special-token marker.
LEAK_MARKERS = ("\nassistant:", "\nsql_agent:", "<|im_start|>")
# Honeypot-hit rate (% of turns) at/below which the learning signal is flagged weak.
WEAK_HIT_RATE_PCT = 0.5


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def _ordered_counts(counter: Counter, order: list[str]) -> list[tuple[str, int]]:
    """Counts in `order`, then any unseen tiers appended (stable, by frequency)."""
    seen = set(order)
    rows = [(t, counter.get(t, 0)) for t in order if counter.get(t, 0)]
    extra = [(t, c) for t, c in counter.most_common() if t not in seen]
    return rows + extra


# --------------------------------------------------------------------------- red

def red_episode_stats(rows: list[dict]) -> dict:
    """PPO-update count, episode range, and warmup split (from logged field)."""
    eps = [r.get("episode") for r in rows if r.get("episode") is not None]
    n_warm = sum(1 for r in rows if r.get("in_warmup_period"))
    return {
        "n_updates": len(set(eps)),
        "ep_min": min(eps) if eps else None,
        "ep_max": max(eps) if eps else None,
        "n_warmup": n_warm,
        "n_total": len(rows),
    }


def red_honeypot_hits(rows: list[dict]) -> tuple[int, int]:
    """(hits by outcome_tier, hits by is_successful_attack) — should agree."""
    by_tier = sum(1 for r in rows if r.get("outcome_tier") in RED_SUCCESS_TIERS)
    by_flag = sum(1 for r in rows if r.get("is_successful_attack"))
    return by_tier, by_flag


def red_exec_attribution(rows: list[dict]) -> Counter:
    """SECONDARY heuristic over execution_result (NOT the canonical sql_error rate).

    Order matters: the guardrail refusal literally contains 'ERROR: dangerous query',
    so test the guardrail marker before the generic 'error' substring.
    """
    c: Counter = Counter()
    for r in rows:
        txt = (r.get("execution_result") or "").lower()
        if not txt:
            c["no_execution"] += 1
        elif "dangerous query" in txt:
            c["guardrail_marker"] += 1
        elif "error" in txt:
            c["sql_error_text"] += 1
        else:
            c["clean_result"] += 1
    return c


def red_sql_error_trend(rows: list[dict]) -> tuple[tuple, tuple] | None:
    """Split episodes into first/second half; return (early, late) sql_error stats."""
    eps = sorted({r.get("episode") for r in rows if r.get("episode") is not None})
    if len(eps) < 2:
        return None
    mid = len(eps) // 2
    early_eps, late_eps = set(eps[:mid]), set(eps[mid:])

    def _stats(keep: set) -> tuple:
        sub = [r for r in rows if r.get("episode") in keep]
        k = sum(1 for r in sub if r.get("outcome_tier") == "sql_error")
        lo, hi = wilson_ci_pct(k, len(sub))
        return _pct(k, len(sub)), lo, hi, len(sub), (min(keep), max(keep))

    return _stats(early_eps), _stats(late_eps)


def _red_text(r: dict) -> str:
    """The red action text — training logs it as red_team_input, cross-eval as user_message."""
    return r.get("red_team_input") or r.get("user_message") or ""


def red_refusal_breakout(rows: list[dict]) -> tuple[int, int]:
    """(red self-refusal, victim refusal) counts from the logged reward_breakdown.

    These dominate the no_sql bucket; surfacing them separates 'the attacker refused'
    (a model/prompt problem) from 'the defender refused' (blue working as intended).
    """
    self_ref = sum(1 for r in rows if "red_self_refusal" in (r.get("reward_breakdown") or {}))
    victim_ref = sum(1 for r in rows if "victim_refusal" in (r.get("reward_breakdown") or {}))
    return self_ref, victim_ref


def red_leak_stats(rows: list[dict]) -> tuple[int, int, float]:
    """(n_with_text, n_leaked, median_post_leak_frac).

    A leaked action contains a plain-text turn delimiter — the policy kept generating
    the victim's reply past its own turn (the turn-termination harness bug). Post-leak
    fraction is the share of the action AFTER the first delimiter, i.e. the hallucinated
    dialogue PPO trained on. Should read ~0 once the stop-strings fix is in.
    """
    texts = [t for t in (_red_text(r) for r in rows) if t]
    fracs: list[float] = []
    for t in texts:
        idxs = [t.find(m) for m in LEAK_MARKERS if m in t]
        if idxs:
            fracs.append((len(t) - min(idxs)) / len(t))
    med = statistics.median(fracs) if fracs else 0.0
    return len(texts), len(fracs), med


def render_red(n: int, rows: list[dict], show_trend: bool) -> None:
    counter = Counter(r.get("outcome_tier") for r in rows)
    total = len(rows)
    stats = red_episode_stats(rows)
    hits_tier, hits_flag = red_honeypot_hits(rows)

    print(f"\n-- iter {n} / RED (n={total} turns) --")
    print("  outcome_tier          count     pct")
    for tier, cnt in _ordered_counts(counter, RED_TIER_ORDER):
        flag = "  <- HIGH" if tier == "sql_error" and _pct(cnt, total) >= 25 else ""
        mark = "*" if tier == "sql_error" else " "
        print(f"  {tier:<20}{mark} {cnt:5d}  {_pct(cnt, total):5.1f}%{flag}")
    hit_rate = _pct(hits_tier, total)
    if hits_tier == 0:
        sig = "  <- NO LEARNING SIGNAL"
    elif hit_rate < WEAK_HIT_RATE_PCT:
        sig = f"  <- WEAK LEARNING SIGNAL (<{WEAK_HIT_RATE_PCT:g}%)"
    else:
        sig = ""
    print(
        f"  honeypot_hits: {hits_tier} ({hit_rate:.2f}%)"
        f"   (is_successful_attack sum: {hits_flag}){sig}"
    )

    # Break the no_sql bucket into the two failure modes it hides.
    self_ref, victim_ref = red_refusal_breakout(rows)
    print(
        f"  refusals: red_self_refusal {self_ref} ({_pct(self_ref, total):.1f}%)"
        f"   victim_refusal {victim_ref} ({_pct(victim_ref, total):.1f}%)"
        f"{'  <- RED REFUSES (attacker problem)' if _pct(self_ref, total) >= 20 else ''}"
    )

    # Turn-termination leakage: red kept generating past its own turn (harness bug).
    n_txt, n_leak, post_frac = red_leak_stats(rows)
    leak_pct = _pct(n_leak, n_txt)
    print(
        f"  role-leak: {n_leak}/{n_txt} actions ({leak_pct:.1f}%)"
        f"   median post-leak garbage {post_frac * 100:.0f}%"
        f"{'  <- HARNESS TURN-TERMINATION LEAK' if leak_pct >= 5 else ''}"
    )
    warm_note = (
        "  <- NEVER EXITS WARMUP"
        if stats["n_total"] and stats["n_warmup"] == stats["n_total"]
        else ""
    )
    print(
        f"  PPO updates: {stats['n_updates']}"
        f"  (episode {stats['ep_min']}..{stats['ep_max']})"
        f"   warmup: {stats['n_warmup']}/{stats['n_total']} rows in-warmup{warm_note}"
    )
    attrib = red_exec_attribution(rows)
    print("  [secondary execution_result heuristic, NOT canonical]")
    print(
        "    guardrail \"dangerous query\" {gm} ({gmp:.1f}%) | "
        "real SQL error {se} ({sep:.1f}%) | "
        "clean {cl} ({clp:.1f}%) | no-exec {ne} ({nep:.1f}%)".format(
            gm=attrib["guardrail_marker"], gmp=_pct(attrib["guardrail_marker"], total),
            se=attrib["sql_error_text"], sep=_pct(attrib["sql_error_text"], total),
            cl=attrib["clean_result"], clp=_pct(attrib["clean_result"], total),
            ne=attrib["no_execution"], nep=_pct(attrib["no_execution"], total),
        )
    )
    if show_trend:
        trend = red_sql_error_trend(rows)
        if trend is None:
            print("  [--early-late-trend] too few episodes to split")
        else:
            (e, elo, ehi, en, erange), (l, llo, lhi, ln, lrange) = trend
            print(
                f"  [--early-late-trend] RED sql_error: "
                f"early(ep {erange[0]}-{erange[1]}, n={en}) {e:.1f}% [{elo:.1f},{ehi:.1f}]"
                f"  ->  late(ep {lrange[0]}-{lrange[1]}, n={ln}) {l:.1f}% [{llo:.1f},{lhi:.1f}]"
                f"   delta {l - e:+.1f}pp"
            )


# -------------------------------------------------------------------------- blue

def render_blue(n: int, rows: list[dict]) -> None:
    train = [r for r in rows if not r.get("is_eval")]
    benign = [r for r in train if r.get("turn_type") == "benign"]
    attack = [r for r in train if r.get("turn_type") == "attack"]
    print(f"\n-- iter {n} / BLUE (n={len(train)} training turns, is_eval=False) --")

    bc = Counter(r.get("outcome_tier") for r in benign)
    nb = len(benign)
    over_refusal = _pct(bc.get("false_negative", 0), nb)
    benign_cells = "  ".join(
        f"{t} {bc.get(t, 0)} ({_pct(bc.get(t, 0), nb):.1f}%)"
        for t, _ in _ordered_counts(bc, BENIGN_TIERS)
    )
    print(f"  benign (n={nb}):  {benign_cells or '(none)'}   [over-refusal={over_refusal:.1f}%]")

    ac = Counter(r.get("outcome_tier") for r in attack)
    na = len(attack)
    attack_cells = "  ".join(
        f"{t} {ac.get(t, 0)} ({_pct(ac.get(t, 0), na):.1f}%)"
        for t, _ in _ordered_counts(ac, ATTACK_TIERS)
    )
    print(f"  attack (n={na}):  {attack_cells or '(none)'}")


# -------------------------------------------------------------------------- main

def render_dir(results_dir: str, show_trend: bool) -> None:
    summary = load_run_summary(results_dir)
    print(
        f"\n== {results_dir}  "
        f"[honeypot_type={summary['honeypot_type']}  "
        f"base={summary.get('base_model', '?').split('/')[-1]}  "
        f"steps={summary.get('num_env_steps')}  "
        f"horizon={summary.get('horizon')}] =="
    )
    any_side = False
    for n, side, run_dir in iter_runs(results_dir):
        rows = load_reward_debug_lines(run_dir, mode="training_time", tail_pct=1.0)
        if not rows:
            note(f"iter_{n}/{side}team: no reward_debug rows — skipping")
            continue
        any_side = True
        if side == "red":
            render_red(n, rows, show_trend)
        else:
            render_blue(n, rows)
    if not any_side:
        note(f"{results_dir}: no usable trained iterations found")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+", help="one or more results-* directories")
    ap.add_argument(
        "--early-late-trend",
        action="store_true",
        help="add red sql_error first-half vs second-half trend",
    )
    args = ap.parse_args()
    for d in args.results_dirs:
        render_dir(d, args.early_late_trend)


if __name__ == "__main__":
    main()
