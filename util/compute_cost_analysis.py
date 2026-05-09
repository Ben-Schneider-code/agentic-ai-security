#!/usr/bin/env python3
"""
Compute cost analysis for adversarial self-play training runs.

Measures and compares the compute cost of red team vs blue team training per
self-play iteration using GPU-invariant metrics derived from existing artifacts.

Primary metric: Environment Interaction Steps (EIS) — total_num_steps from
training_state.json, equal to training_episodes * episode_length * n_rollout_threads.
This directly measures what was fed into the RL algorithm.

Meta-metrics (supplementary):
  - Environment episodes completed (actual red-vs-blue interactions)
  - PPO update cycles
  - Estimated tokens generated (env_episodes * horizon * max_new_tokens * 2 agents)

Usage:
    python util/compute_cost_analysis.py --selfplay-dir results-20260322-1641-m92p4
    python util/compute_cost_analysis.py --selfplay-dir results-20260322-1641-m92p4 --json-out results.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ──────────────────────────────────────────────────────────────────────────────


def _find_run_dir(team_dir: Path) -> Path | None:
    """Walk into team_dir to find the deepest directory containing training_state.json."""
    for root, _dirs, files in os.walk(team_dir):
        if "training_state.json" in files:
            return Path(root)
    return None


def discover_iterations(selfplay_dir: str) -> list[dict]:
    """
    Discover all iter_N subdirectories in a self-play results dir.

    Returns a list of dicts sorted by iteration number:
        {"iter": int, "red_dir": Path | None, "blue_dir": Path | None}
    """
    base = Path(selfplay_dir)
    iters = {}
    for entry in sorted(base.iterdir()):
        m = re.match(r"^iter_(\d+)$", entry.name)
        if m and entry.is_dir():
            n = int(m.group(1))
            iters[n] = {"iter": n, "red_dir": None, "blue_dir": None}
            red = entry / "redteam"
            blue = entry / "blueteam"
            if red.is_dir():
                iters[n]["red_dir"] = red
            if blue.is_dir():
                iters[n]["blue_dir"] = blue
    return [iters[k] for k in sorted(iters)]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def _load_yaml_field(yaml_path: Path, field: str, default=None):
    """Read a single field from a YAML file using plain regex (avoids python/tuple tag issue)."""
    if not yaml_path.exists():
        return default
    text = yaml_path.read_text()
    m = re.search(rf"^{re.escape(field)}:\s*(.+)$", text, re.MULTILINE)
    if m:
        try:
            return int(m.group(1).strip())
        except ValueError:
            try:
                return float(m.group(1).strip())
            except ValueError:
                return m.group(1).strip()
    return default


def load_phase_data(team_dir: Path, team: str) -> dict:
    """
    Load all available metrics for a single training phase from existing artifacts.

    Returns dict with keys:
        team, run_dir, eis, ppo_episodes, env_episodes, accessed_honeypots,
        total_honeypots, exit_reason, final_quality, episode_length,
        n_rollout_threads, horizon, max_new_tokens, ppo_epoch
    """
    run_dir = _find_run_dir(team_dir)
    if run_dir is None:
        return {"team": team, "available": False}

    # --- training_state.json ---
    state_file = run_dir / "training_state.json"
    if not state_file.exists():
        return {"team": team, "available": False}
    with open(state_file) as f:
        state = json.load(f)

    eis = state.get("total_num_steps", 0)
    ppo_episodes = state.get("episode", 0) + 1  # 0-indexed → count
    all_returns = state.get("all_episodic_returns", [])
    env_episodes = len(all_returns)
    accessed_honeypots = len(state.get("accessed_honeypots", []))

    # Final defense quality: rolling avg of last 100 env episodes
    dw_window = 100
    final_quality = (
        float(np.mean(all_returns[-dw_window:]))
        if len(all_returns) >= dw_window
        else (float(np.mean(all_returns)) if all_returns else 0.0)
    )

    # --- exit_reason.txt ---
    exit_file = run_dir / "exit_reason.txt"
    exit_reason = exit_file.read_text().strip() if exit_file.exists() else "unknown"

    # --- args.yaml ---
    args_file = run_dir / "args.yaml"
    episode_length = _load_yaml_field(args_file, "episode_length", default=10)
    n_rollout_threads = _load_yaml_field(args_file, "n_rollout_threads", default=8)
    horizon = _load_yaml_field(args_file, "horizon", default=5)
    max_new_tokens = _load_yaml_field(args_file, "max_new_tokens", default=512)
    ppo_epoch = _load_yaml_field(args_file, "ppo_epoch", default=1)

    # --- reward_config.yaml (plain grep — avoids python/tuple YAML tag) ---
    reward_cfg = run_dir / "reward_config.yaml"
    total_honeypots = _load_yaml_field(reward_cfg, "total_honeypots", default=0)
    # blueteam always returns 0 from get_total_honeypots(); use redteam's value
    if total_honeypots == 0 and team == "blueteam":
        total_honeypots = None  # filled in later from redteam

    return {
        "team": team,
        "available": True,
        "run_dir": str(run_dir),
        "eis": eis,
        "ppo_episodes": ppo_episodes,
        "ppo_updates": ppo_episodes * ppo_epoch,
        "env_episodes": env_episodes,
        "accessed_honeypots": accessed_honeypots,
        "total_honeypots": total_honeypots,
        "exit_reason": exit_reason,
        "final_quality": final_quality,
        "episode_length": episode_length,
        "n_rollout_threads": n_rollout_threads,
        "horizon": horizon,
        "max_new_tokens": max_new_tokens,
        "ppo_epoch": ppo_epoch,
    }


def estimate_tokens(phase: dict) -> int:
    """
    Estimate total tokens generated during a training phase.

    Both the student (being trained) and opponent generate tokens every step,
    so multiply by 2 agents. Each environment episode lasts up to `horizon`
    turns; use `env_episodes * horizon * max_new_tokens * 2` as the upper bound.
    In practice episodes terminate early, so this is a conservative over-estimate.
    """
    if not phase.get("available"):
        return 0
    env_eps = phase["env_episodes"]
    horizon = phase["horizon"]
    max_tokens = phase["max_new_tokens"]
    # 2 agents generating per turn; horizon bounds the turns per env episode
    return env_eps * horizon * max_tokens * 2


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(iterations: list[dict]) -> dict:
    """
    Compute per-iteration and aggregate cost metrics.

    Each element of `iterations` must have "red" and "blue" phase dicts
    (as returned by load_phase_data) plus an "iter" key.
    """
    per_iter = []
    for it in iterations:
        red = it.get("red", {})
        blue = it.get("blue", {})
        if not red.get("available") or not blue.get("available"):
            continue

        # Propagate total_honeypots from red to blue (blue's reward_config stores 0)
        if blue.get("total_honeypots") is None:
            blue["total_honeypots"] = red.get("total_honeypots", 0)

        total_hp = red.get("total_honeypots") or 0
        ratio = blue["eis"] / red["eis"] if red["eis"] > 0 else float("inf")

        per_iter.append(
            {
                "iter": it["iter"],
                # --- Red phase ---
                "red_eis": red["eis"],
                "red_ppo_episodes": red["ppo_episodes"],
                "red_ppo_updates": red["ppo_updates"],
                "red_env_episodes": red["env_episodes"],
                "red_tokens_est": estimate_tokens(red),
                "red_accessed_hp": red["accessed_honeypots"],
                "red_total_hp": total_hp,
                "red_yield": red["accessed_honeypots"] / total_hp
                if total_hp > 0
                else None,
                "red_exit": red["exit_reason"],
                # --- Blue phase ---
                "blue_eis": blue["eis"],
                "blue_ppo_episodes": blue["ppo_episodes"],
                "blue_ppo_updates": blue["ppo_updates"],
                "blue_env_episodes": blue["env_episodes"],
                "blue_tokens_est": estimate_tokens(blue),
                "blue_quality": blue["final_quality"],
                "blue_exit": blue["exit_reason"],
                # --- Ratio ---
                "eis_ratio": ratio,
            }
        )

    if not per_iter:
        return {"per_iteration": [], "aggregate": {}}

    ratios = [r["eis_ratio"] for r in per_iter]
    red_eis_list = [r["red_eis"] for r in per_iter]
    blue_eis_list = [r["blue_eis"] for r in per_iter]
    red_tokens = [r["red_tokens_est"] for r in per_iter]
    blue_tokens = [r["blue_tokens_est"] for r in per_iter]
    red_yields = [r["red_yield"] for r in per_iter if r["red_yield"] is not None]
    blue_qualities = [r["blue_quality"] for r in per_iter]

    aggregate = {
        "n_iterations": len(per_iter),
        # EIS
        "mean_red_eis": float(np.mean(red_eis_list)),
        "std_red_eis": float(np.std(red_eis_list)),
        "mean_blue_eis": float(np.mean(blue_eis_list)),
        "std_blue_eis": float(np.std(blue_eis_list)),
        "cumulative_red_eis": int(sum(red_eis_list)),
        "cumulative_blue_eis": int(sum(blue_eis_list)),
        "cumulative_eis_ratio": sum(blue_eis_list) / sum(red_eis_list)
        if sum(red_eis_list) > 0
        else None,
        # Ratio distribution
        "mean_eis_ratio": float(np.mean(ratios)),
        "std_eis_ratio": float(np.std(ratios)),
        "min_eis_ratio": float(min(ratios)),
        "max_eis_ratio": float(max(ratios)),
        # Token estimates
        "mean_red_tokens_est": float(np.mean(red_tokens)),
        "mean_blue_tokens_est": float(np.mean(blue_tokens)),
        "cumulative_red_tokens_est": int(sum(red_tokens)),
        "cumulative_blue_tokens_est": int(sum(blue_tokens)),
        # Context
        "mean_red_yield": float(np.mean(red_yields)) if red_yields else None,
        "mean_blue_quality": float(np.mean(blue_qualities)),
        # Exit reason distribution
        "red_exit_counts": _count_values([r["red_exit"] for r in per_iter]),
        "blue_exit_counts": _count_values([r["blue_exit"] for r in per_iter]),
    }

    return {"per_iteration": per_iter, "aggregate": aggregate}


def _count_values(lst: list) -> dict:
    counts = {}
    for v in lst:
        counts[v] = counts.get(v, 0) + 1
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────────────

_EXIT_SHORT = {
    "no_new_honeypot_for_1000_steps": "no_new_hp_1000",
    "all_honeypots_accessed": "all_hp_found",
    "max_episodes_met": "max_ep",
    "blueteam_decisive_win": "decisive_win",
    "blueteam_plateaued": "plateau",
    "blueteam_max_episodes_reached": "max_eps",
    "forced_exit": "forced",
    "unknown": "?",
}


def _shorten_exit(reason: str) -> str:
    return _EXIT_SHORT.get(reason, reason[:16])


def format_table(metrics: dict) -> str:
    per_iter = metrics["per_iteration"]
    agg = metrics["aggregate"]
    if not per_iter:
        return "(no complete iterations found)\n"

    header = (
        f"{'Iter':>4}  {'RedEIS':>7}  {'RedEnvEp':>8}  "
        f"{'RedTok(est)':>10}  {'RedYield':>12}  {'RedExit':<15}  "
        f"{'BlueEIS':>7}  {'BlueEnvEp':>8}  {'BlueTok(est)':>10}  "
        f"{'BlueQual':>10}  {'BlueExit':<15}  {'B/R':>6}"
    )
    sep = "-" * len(header)

    rows = [header, sep]
    for r in per_iter:
        total_hp = r["red_total_hp"]
        yield_str = f"{r['red_accessed_hp']}/{total_hp}" if total_hp else "?"
        rows.append(
            f"{r['iter']:>4}  {r['red_eis']:>7,}  {r['red_env_episodes']:>8,}  "
            f"{r['red_tokens_est']:>10,}  {yield_str:>12}  {_shorten_exit(r['red_exit']):<15}  "
            f"{r['blue_eis']:>7,}  {r['blue_env_episodes']:>8,}  "
            f"{r['blue_tokens_est']:>10,}  "
            f"{r['blue_quality']:>10.3f}  {_shorten_exit(r['blue_exit']):<15}  "
            f"{r['eis_ratio']:>6.1f}x"
        )

    rows.append(sep)
    # Aggregate row
    n = agg["n_iterations"]
    yield_avg = (
        f"{agg['mean_red_yield']:.2f}" if agg["mean_red_yield"] is not None else "?"
    )
    rows.append(
        f"{'Avg':>4}  {agg['mean_red_eis']:>7,.0f}  "
        f"{'-':>8}  {agg['mean_red_tokens_est']:>10,.0f}  "
        f"{yield_avg:>12}  {'-':<15}  "
        f"{agg['mean_blue_eis']:>7,.0f}  {'-':>8}  "
        f"{agg['mean_blue_tokens_est']:>10,.0f}  "
        f"{agg['mean_blue_quality']:>10.3f}  {'-':<15}  "
        f"{agg['mean_eis_ratio']:>6.1f}x"
    )

    lines = ["\n=== Per-Iteration Cost Breakdown ===", "\n".join(rows)]

    lines.append("\n=== Aggregate Cost Metrics ===")
    lines.append(f"  Iterations analyzed         : {n}")
    lines.append(
        f"  Mean EIS ratio (blue/red)   : {agg['mean_eis_ratio']:.2f}x  ±{agg['std_eis_ratio']:.2f}  "
        f"[{agg['min_eis_ratio']:.1f}x – {agg['max_eis_ratio']:.1f}x]"
    )
    lines.append(
        f"  Cumulative EIS ratio (b/r)  : {agg['cumulative_eis_ratio']:.2f}x  "
        f"(blue={agg['cumulative_blue_eis']:,}, red={agg['cumulative_red_eis']:,})"
    )
    lines.append(
        f"  Cumul. token est (red/blue) : {agg['cumulative_red_tokens_est']:,} / {agg['cumulative_blue_tokens_est']:,}"
    )
    if agg["mean_red_yield"] is not None:
        lines.append(f"  Mean red honeypot yield     : {agg['mean_red_yield']:.1%}")
    lines.append(f"  Mean blue final quality     : {agg['mean_blue_quality']:.3f}")
    lines.append(f"  Red exit reasons            : {agg['red_exit_counts']}")
    lines.append(f"  Blue exit reasons           : {agg['blue_exit_counts']}")
    lines.append(
        "\nNote: 'Tokens(est)' = env_episodes × horizon × max_new_tokens × 2 agents "
        "(upper bound; actual tokens depend on early episode termination)."
    )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

_RED = "#e74c3c"
_BLUE = "#3498db"
_RED_LIGHT = "#f5b7b1"
_BLUE_LIGHT = "#aed6f1"


def plot_cost_metrics(
    metrics: dict, selfplay_dir: str, plot_out: str | None = None
) -> str:
    """
    Render a 2×2 figure summarising compute cost asymmetry.

    Subplots:
      (0,0)  EIS per iteration — grouped bar (red vs blue), ratio annotated above
      (0,1)  Cumulative EIS — stacked area showing growing gap
      (1,0)  EIS ratio per iteration — bar chart with mean line
      (1,1)  Outcome quality — red honeypot yield (bar) + blue final quality (line)

    Returns path to saved PNG.
    """
    per_iter = metrics["per_iteration"]
    agg = metrics["aggregate"]
    if not per_iter:
        print("No data to plot.")
        return ""

    iters = [r["iter"] for r in per_iter]
    red_eis = [r["red_eis"] for r in per_iter]
    blue_eis = [r["blue_eis"] for r in per_iter]
    ratios = [r["eis_ratio"] for r in per_iter]
    red_yields = [
        r["red_yield"] * 100 if r["red_yield"] is not None else 0 for r in per_iter
    ]
    blue_qualities = [r["blue_quality"] * 100 for r in per_iter]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    run_name = Path(selfplay_dir).name
    fig.suptitle(
        "Compute Cost Asymmetry",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(len(iters))
    bar_w = 0.38

    # ── (0,0) EIS per iteration ──────────────────────────────────────────────
    ax = axes[0, 0]
    bars_r = ax.bar(
        x - bar_w / 2, red_eis, width=bar_w, color=_RED, alpha=0.85, label="Red team"
    )
    bars_b = ax.bar(
        x + bar_w / 2, blue_eis, width=bar_w, color=_BLUE, alpha=0.85, label="Blue team"
    )

    # Annotate ratio above each pair
    for i, (re_, be_, ratio) in enumerate(zip(red_eis, blue_eis, ratios)):
        ax.text(
            x[i],
            max(re_, be_) * 1.03,
            f"{ratio:.1f}×",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#555555",
        )

    ax.set_title("EIS per Iteration (ratio annotated)")
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Environment interaction steps")
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(red_eis) * 1.18)

    # ── (0,1) Cumulative EIS ─────────────────────────────────────────────────
    ax = axes[0, 1]
    cum_red = np.cumsum(red_eis)
    cum_blue = np.cumsum(blue_eis)

    ax.fill_between(iters, 0, cum_red, color=_RED, alpha=0.25)
    ax.fill_between(iters, 0, cum_blue, color=_BLUE, alpha=0.35)
    ax.plot(
        iters, cum_red, marker="o", color=_RED, linewidth=2, label="Red (cumulative)"
    )
    ax.plot(
        iters, cum_blue, marker="s", color=_BLUE, linewidth=2, label="Blue (cumulative)"
    )

    # Annotate final cumulative ratio (blue/red)
    final_ratio = agg.get("cumulative_eis_ratio") or 0
    ax.annotate(
        f"Blue/Red: {final_ratio:.1f}×",
        xy=(iters[-1], cum_blue[-1]),
        xytext=(-40, 20),
        textcoords="offset points",
        fontsize=8,
        color=_BLUE,
        arrowprops=dict(arrowstyle="->", color=_BLUE, lw=1),
    )

    ax.set_title("Cumulative EIS across Iterations")
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Cumulative environment interaction steps")
    ax.set_xticks(iters)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) EIS ratio per iteration ────────────────────────────────────────
    ax = axes[1, 0]
    bar_colors = [_BLUE if r >= agg["mean_eis_ratio"] else _BLUE_LIGHT for r in ratios]
    ax.bar(
        x, ratios, width=0.55, color=bar_colors, alpha=0.85, label="Blue/Red EIS ratio"
    )

    mean_r = agg["mean_eis_ratio"]
    ax.axhline(
        mean_r,
        color="#555555",
        linewidth=1.4,
        linestyle="--",
        label=f"Mean {mean_r:.1f}×",
    )

    # ±1 std band
    std_r = agg["std_eis_ratio"]
    ax.axhspan(
        mean_r - std_r,
        mean_r + std_r,
        color="#dddddd",
        alpha=0.4,
        label=f"±1 SD ({std_r:.1f})",
    )

    for i, r in enumerate(ratios):
        ax.text(x[i], r + 0.1, f"{r:.1f}×", ha="center", va="bottom", fontsize=7.5)

    ax.set_title("EIS Ratio (Blue / Red) per Iteration")
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Ratio (×)")
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_ylim(0, max(ratios) * 1.25)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # ── (1,1) Outcome quality ─────────────────────────────────────────────────
    ax = axes[1, 1]
    ax2 = ax.twinx()

    ax.bar(
        x - bar_w / 2,
        red_yields,
        width=bar_w,
        color=_RED,
        alpha=0.7,
        label="Red yield (%)",
    )
    ax2.plot(
        iters,
        blue_qualities,
        marker="s",
        color=_BLUE,
        linewidth=2,
        label="Blue quality (×100)",
    )

    # Reference lines
    ax.axhline(
        100 * (agg.get("mean_red_yield") or 0),
        color=_RED,
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )
    ax2.axhline(
        np.mean(blue_qualities), color=_BLUE, linestyle=":", linewidth=1, alpha=0.7
    )

    ax.set_title("Outcome Quality per Iteration")
    ax.set_xlabel("Self-play iteration")
    ax.set_ylabel("Red honeypot yield (%)", color=_RED)
    ax2.set_ylabel("Blue rolling-avg return (×100)", color=_BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_ylim(0, 115)
    ax2.set_ylim(0, 115)
    ax.tick_params(axis="y", colors=_RED)
    ax2.tick_params(axis="y", colors=_BLUE)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    # Save
    if plot_out:
        out_path = Path(plot_out)
    else:
        candidate = Path(selfplay_dir) / "compute_cost_analysis.png"
        try:
            candidate.touch()
            out_path = candidate
        except PermissionError:
            out_path = Path(f"{Path(selfplay_dir).name}_compute_cost_analysis.png")

    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compute cost analysis for adversarial self-play training runs."
    )
    parser.add_argument(
        "--selfplay-dir",
        required=True,
        help="Path to a self-play results directory (e.g. results-20260322-1641-m92p4)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON output (default: <selfplay-dir>/compute_cost_analysis.json)",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="Optional path to write PNG plot (default: <selfplay-dir>/compute_cost_analysis.png)",
    )
    args = parser.parse_args()

    selfplay_dir = Path(args.selfplay_dir)
    if not selfplay_dir.exists():
        print(f"ERROR: Directory not found: {selfplay_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {selfplay_dir.resolve()}")
    raw_iters = discover_iterations(str(selfplay_dir))
    print(f"Found {len(raw_iters)} iteration(s).")

    # Load phase data for each iteration
    loaded_iters = []
    for it in raw_iters:
        entry = {"iter": it["iter"]}
        if it["red_dir"]:
            entry["red"] = load_phase_data(it["red_dir"], "redteam")
        else:
            print(f"  [iter {it['iter']}] WARNING: no redteam dir found")
        if it["blue_dir"]:
            entry["blue"] = load_phase_data(it["blue_dir"], "blueteam")
        else:
            print(f"  [iter {it['iter']}] WARNING: no blueteam dir found")
        loaded_iters.append(entry)

    metrics = compute_metrics(loaded_iters)
    print(format_table(metrics))

    # Write JSON output
    if args.json_out:
        json_out = Path(args.json_out)
    else:
        # Try writing next to the selfplay dir first; fall back to cwd
        candidate = selfplay_dir / "compute_cost_analysis.json"
        try:
            candidate.touch()
            json_out = candidate
        except PermissionError:
            json_out = Path(f"{selfplay_dir.name}_compute_cost_analysis.json")
    with open(json_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nJSON results written to: {json_out}")

    plot_path = plot_cost_metrics(metrics, str(selfplay_dir), args.plot_out)
    if plot_path:
        print(f"Plot saved to:          {plot_path}")


if __name__ == "__main__":
    main()
