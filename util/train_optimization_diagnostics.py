#!/usr/bin/env python3
"""Optimization-health diagnostics for self-play training (optimization half of Q2).

For each results dir, each trained iteration, each side (red / blue) report whether the
policy is actually being trained, from two sources:

  - ``run_dir/logs/summary.json`` (TensorBoard scalars; keys are full paths, value is a
    list of ``[timestamp, step, value]``). For average_step_rewards / policy_loss /
    value_loss / approx_kl / entropy / policy_grad_norm / value_grad_norm we report
    first / last / mean / delta / slope and n (= number of PPO updates).
  - ``args.yaml`` hyperparameters (lr, ppo_epoch, num_mini_batch, num_env_steps, horizon,
    warmup_steps).

Two distinct "warmup" mechanisms matter and are both surfaced:
  * env-reward warmup (RewardConfig.warmup_episodes): shaping rewards active; read from
    the per-row logged ``in_warmup_period`` field -> never_exits_warmup flag.
  * trainer actor warmup (args.yaml ``warmup_steps``): the ACTOR is frozen until this
    step, so policy_loss / approx_kl / entropy stay 0.0 (only the value head trains).
    The "actor activity" line counts how many updates actually moved the policy.

Plus a corroborating mean ``final_reward`` early-vs-late from reward_debug, and heuristic
FLAGS: few_updates, never_exits_warmup, actor_frozen_updates, kl~0,
reward_flat_or_declining, entropy_rising.

Output is stdout tables only. Reuses plotting/_data.py and util/_diag_common.py.

Usage:
    python util/train_optimization_diagnostics.py RESULTS_DIR [RESULTS_DIR ...]
        [--few-updates-threshold N] [--kl-eps F]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting._data import load_args_yaml, load_reward_debug_lines  # noqa: E402
from util._diag_common import iter_runs, load_run_summary, note  # noqa: E402

# Flag thresholds (overridable via CLI in main()).
FEW_THRESHOLD = 10
KL_EPS = 1e-4

METRICS = [
    "average_step_rewards",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "entropy",
    "policy_grad_norm",
    "value_grad_norm",
]
# Actor-side metrics that read 0.0 while the actor is frozen during trainer warmup.
ACTOR_METRICS = {"policy_loss", "approx_kl", "entropy", "policy_grad_norm"}


def load_optimizer_logs(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {metric_name: [(step, value), ...]} from logs/summary.json; {} if absent."""
    path = run_dir / "logs" / "summary.json"
    if not path.is_file():
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    out: dict[str, list[tuple[int, float]]] = {}
    for key, series in raw.items():
        name = key.split("/")[-1]
        if name in METRICS:
            out[name] = [
                (pt[1], pt[2])
                for pt in series
                if isinstance(pt, (list, tuple)) and len(pt) >= 3
            ]
    return out


def _slope(vals: list[float]) -> float | None:
    """Least-squares slope over the sample index; None if < 2 points."""
    n = len(vals)
    if n < 2:
        return None
    sx = sum(range(n))
    sy = sum(vals)
    sxx = sum(i * i for i in range(n))
    sxy = sum(i * v for i, v in enumerate(vals))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    return (n * sxy - sx * sy) / denom


def series_stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"n": 0}
    return {
        "first": vals[0],
        "last": vals[-1],
        "mean": sum(vals) / n,
        "delta": vals[-1] - vals[0],
        "slope": _slope(vals),
        "n": n,
    }


def actor_activity(series: list[tuple[int, float]]) -> dict:
    """Count updates that actually moved the actor (nonzero metric value)."""
    active = [(s, v) for s, v in series if v != 0.0]
    return {
        "n_active": len(active),
        "n_total": len(series),
        "first_step": active[0][0] if active else None,
        "mean_active": (sum(v for _, v in active) / len(active)) if active else None,
    }


def reward_early_late(rows: list[dict]) -> tuple[float, float] | None:
    """Mean final_reward over first-half vs last-half of episodes."""
    eps = sorted({r.get("episode") for r in rows if r.get("episode") is not None})
    if len(eps) < 2:
        return None
    mid = len(eps) // 2
    early_eps, late_eps = set(eps[:mid]), set(eps[mid:])

    def _mean(keep: set) -> float:
        vals = [r.get("final_reward", 0.0) for r in rows if r.get("episode") in keep]
        return sum(vals) / len(vals) if vals else float("nan")

    return _mean(early_eps), _mean(late_eps)


def warmup_state(rows: list[dict]) -> tuple[int, int]:
    return sum(1 for r in rows if r.get("in_warmup_period")), len(rows)


def compute_flags(
    stats: dict[str, dict],
    logs: dict[str, list[tuple[int, float]]],
    n_warmup: int,
    n_total: int,
) -> list[str]:
    flags: list[str] = []
    asr = stats.get("average_step_rewards", {})
    n_updates = asr.get("n", 0) or max((s.get("n", 0) for s in stats.values()), default=0)
    if 0 < n_updates < FEW_THRESHOLD:
        flags.append(f"few_updates({n_updates}<{FEW_THRESHOLD})")

    if n_total and n_warmup == n_total:
        flags.append(f"never_exits_warmup({n_warmup}/{n_total} rows)")

    kl_act = actor_activity(logs.get("approx_kl", []))
    if kl_act["n_total"] and kl_act["n_active"] < kl_act["n_total"]:
        flags.append(
            f"actor_frozen_updates({kl_act['n_total'] - kl_act['n_active']}"
            f"/{kl_act['n_total']}, active_from_step={kl_act['first_step']})"
        )
    if kl_act["mean_active"] is not None and kl_act["mean_active"] < KL_EPS:
        flags.append(f"kl~0(active mean {kl_act['mean_active']:.2g}<{KL_EPS:g})")

    if asr.get("n", 0) >= 2 and asr.get("delta", 0.0) <= 0:
        flags.append(f"reward_flat_or_declining(delta {asr['delta']:+.3g})")

    # entropy_rising judged over nonzero (post-warmup) points only, to avoid the
    # frozen-actor leading zeros faking an upward trend.
    ent_active = [v for _, v in logs.get("entropy", []) if v != 0.0]
    ent_slope = _slope(ent_active)
    if ent_slope is not None and ent_slope > 0 and len(ent_active) >= 2:
        flags.append(f"entropy_rising(slope {ent_slope:+.2g})")

    return flags or ["(none)"]


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4g}"


def render_side(n: int, side: str, run_dir: Path) -> None:
    logs = load_optimizer_logs(run_dir)
    rows = load_reward_debug_lines(run_dir, mode="training_time", tail_pct=1.0)
    if side == "blue":
        rows = [r for r in rows if not r.get("is_eval")]

    stats = {m: series_stats([v for _, v in logs.get(m, [])]) for m in METRICS}
    n_warm, n_tot = warmup_state(rows)

    print(f"\n-- iter {n} / {side.upper()} --")
    if not logs:
        note(f"iter_{n}/{side}team: no logs/summary.json — optimizer scalars unavailable")
    else:
        print(f"  {'metric':<20} {'first':>9} {'last':>9} {'mean':>9} {'delta':>9} {'slope':>9}  n")
        for m in METRICS:
            s = stats[m]
            if not s.get("n"):
                print(f"  {m:<20} {'—':>9} {'—':>9} {'—':>9} {'—':>9} {'—':>9}  0")
                continue
            print(
                f"  {m:<20} {_fmt(s['first']):>9} {_fmt(s['last']):>9} "
                f"{_fmt(s['mean']):>9} {_fmt(s['delta']):>9} {_fmt(s['slope']):>9}  {s['n']}"
            )
        kl_act = actor_activity(logs.get("approx_kl", []))
        print(
            f"  actor activity (approx_kl != 0): {kl_act['n_active']}/{kl_act['n_total']}"
            f" updates"
            + (f", first nonzero at step {kl_act['first_step']}" if kl_act["first_step"] else "")
            + ("  <- ACTOR ~FROZEN (value head trains, policy barely updates)"
               if kl_act["n_total"] and kl_act["n_active"] <= 1 else "")
        )

    el = reward_early_late(rows)
    if el is not None:
        print(f"  reward(debug) early->late: {el[0]:+.3g} -> {el[1]:+.3g}")
    elif rows:
        print("  reward(debug) early->late: n/a (single episode)")
    else:
        print("  reward(debug) early->late: n/a (no reward_debug rows)")

    # Shaping decay actually applied (decay_factor is keyed on the PPO-update episode
    # counter, not env steps — report it directly so a long run is not mis-blamed on
    # "decay zeroed the signal" when it only fell, say, 1.0 -> 0.61).
    decays = [r.get("decay_factor") for r in rows if r.get("decay_factor") is not None]
    if decays:
        print(
            f"  shaping decay_factor: first {decays[0]:.3f} -> last {decays[-1]:.3f}"
            f"  (min {min(decays):.3f})"
        )

    print(f"  FLAGS: {', '.join(compute_flags(stats, logs, n_warm, n_tot))}")


def render_dir(results_dir: str) -> None:
    summary = load_run_summary(results_dir)
    runs = list(iter_runs(results_dir))
    header_args = load_args_yaml(runs[0][2]) if runs else {}
    print(
        f"\n== {results_dir}  [{summary['honeypot_type']}]  "
        f"lr={header_args.get('lr', 'n/a')} "
        f"ppo_epoch={header_args.get('ppo_epoch', 'n/a')} "
        f"num_mini_batch={header_args.get('num_mini_batch', 'n/a')} "
        f"num_env_steps={header_args.get('num_env_steps', summary.get('num_env_steps'))} "
        f"horizon={header_args.get('horizon', summary.get('horizon'))} "
        f"warmup_steps={header_args.get('warmup_steps', 'n/a')} =="
    )
    if not runs:
        note(f"{results_dir}: no usable trained iterations found")
        return
    for n, side, run_dir in runs:
        render_side(n, side, run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+", help="one or more results-* directories")
    ap.add_argument("--few-updates-threshold", type=int, default=10)
    ap.add_argument("--kl-eps", type=float, default=1e-4)
    args = ap.parse_args()

    global FEW_THRESHOLD, KL_EPS
    FEW_THRESHOLD = args.few_updates_threshold
    KL_EPS = args.kl_eps

    for d in args.results_dirs:
        render_dir(d)


if __name__ == "__main__":
    main()
