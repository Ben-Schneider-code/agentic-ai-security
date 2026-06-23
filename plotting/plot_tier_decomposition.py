"""
Per-tier PVR_conv decomposition with 99 % Wilson CIs.

Each episode is assigned to the highest-priority tier of any honeypot it
accessed (pii_dominant > harvestable > rare), so the three-tier stack sums
exactly to the overall PVR_conv (= ASR).

Data source: canonical cross_eval diagonal cells
  cross_eval/pairings/red_{i}_blue_{i}/reward_debug.jsonl
Tier map:     figures/honeypot_tiers.json

CLI:
    python plotting/plot_tier_decomposition.py \\
        --results <dir>[:label] [--out figures/tier_pvr_decomposition.png]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style, parse_results_arg, write_sidecar,
        wilson_ci_pct, FIG_SIZE_SINGLE,
        RED_COL, BLUE_COL, GRAY_COL,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style, parse_results_arg, write_sidecar,
        wilson_ci_pct, FIG_SIZE_SINGLE,
        RED_COL, BLUE_COL, GRAY_COL,
    )

apply_paper_style()

DESCRIPTION = (
    "Per-tier PVR_conv decomposition: pii_dominant / harvestable / rare stack "
    "sums to total ASR; canonical cross_eval diagonal (400 ep/cell), 99 % Wilson CIs."
)

TIER_ORDER = ["pii_dominant", "harvestable", "rare"]
# Any accessed honeypot is a real breach; if it is absent from the tier map we floor
# it to the lowest tier rather than dropping it (silent undercount).
_FALLBACK_TIER = "rare"
TIER_COLORS = {
    "pii_dominant": "#C62828",
    "harvestable":  "#1565C0",
    "rare":         "#558B2F",
}
TIER_LABELS = {
    "pii_dominant": "PII-dominant",
    "harvestable":  "Harvestable",
    "rare":         "Rare",
}


def _load_tier_map(selfplay_dir: str, out_dir: str | Path | None = None) -> dict[str, str]:
    """Return {honeypot_id: tier} from honeypot_tiers.json.

    Search order: the figure output dir (where ``plot_honeypot_difficulty`` writes the
    sidecar during a ``plot_paper_figures`` run), then ``<selfplay_dir>/figures/``, then
    ``./figures/``. The output dir is checked FIRST because the orchestrator writes
    figures to a custom ``--out-dir`` (e.g. ``figures_<id>/``), not ``./figures``; before
    this was added the map silently came back empty and every breach was miscounted as
    ``no_breach`` (reported PVR 0 while cross_eval had real breaches).
    """
    candidates = []
    if out_dir is not None:
        candidates.append(Path(out_dir) / "honeypot_tiers.json")
    candidates += [
        Path(selfplay_dir) / "figures" / "honeypot_tiers.json",
        Path("figures") / "honeypot_tiers.json",
    ]
    for p in candidates:
        if p.is_file():
            data = json.loads(p.read_text())
            return {h["id"]: h["tier"] for h in data.get("honeypots", [])}
    print(
        "[plot_tier_decomposition] WARNING: no honeypot_tiers.json found in "
        f"{[str(c) for c in candidates]}; tier map is empty. Accessed honeypots will "
        "be floored to the lowest tier so breaches are not silently dropped.",
        file=sys.stderr,
    )
    return {}


def _parse_hp_entry(raw) -> dict | None:
    """Parse a honeypot record (dict or stringified-dict / JSON) into a dict.

    Mirrors ``plot_honeypot_difficulty._parse_hp`` so both plotters agree on the
    ``accessed_honeypots`` wire format — cross_eval writes a list of *stringified*
    dicts (e.g. ``["{'accessed': True, 'type': '...', 'identifier': '...'}"]``), so a
    str-vs-dict mismatch here was silently dropping every breach.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            d = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                d = json.loads(raw)
            except Exception:
                return None
        return d if isinstance(d, dict) else None
    return None


def _compute_tier_pvr(reward_debug_path: Path, tier_map: dict[str, str]) -> dict:
    """
    Read a reward_debug.jsonl and compute per-tier episode counts.

    Returns {"total": N, "pii_dominant": k, "harvestable": k, "rare": k,
             "any_breach": k, "no_breach": k}
    where the three tiers partition all breaching episodes by highest-priority tier.
    """
    # Collect per-episode honeypot accesses, filtering to attack episodes only.
    # reward_debug.jsonl mixes attack + benign eval episodes; turn_type distinguishes them.
    # We also track which episodes have ≥1 resource-accessing turn (a_t ≠ ⊥, i.e.
    # outcome_tier ∈ {false_positive, neutral_sql}) — that is C*_R per
    # problem_statement.tex eq.49 and is the correct PVR_conv denominator.
    _ATTACK_SQL_TIERS = {"false_positive", "neutral_sql"}
    episodes: dict[int, set[str]] = {}
    attack_ep_ids: set[int] = set()
    eps_with_sql: set[int] = set()
    with open(reward_debug_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = r.get("episode")
            if ep is None:
                continue
            if r.get("turn_type") == "attack":
                attack_ep_ids.add(ep)
                if r.get("outcome_tier") in _ATTACK_SQL_TIERS:
                    eps_with_sql.add(ep)
            if ep not in episodes:
                episodes[ep] = set()
            for raw in r.get("accessed_honeypots", []):
                hp = _parse_hp_entry(raw)
                if hp is None or not hp.get("accessed", True):
                    continue
                t, ident = hp.get("type", ""), hp.get("identifier", "")
                if t and ident:
                    episodes[ep].add(f"{t}:{ident}")
    # Restrict to attack episodes that have ≥1 resource-accessing turn (= C*_R).
    episodes = {ep: hps for ep, hps in episodes.items() if ep in eps_with_sql}

    counts = {t: 0 for t in TIER_ORDER}
    no_breach = 0
    untiered_breaches = 0
    for hp_set in episodes.values():
        if not hp_set:
            no_breach += 1  # resource-accessing SQL but no honeypot hit (genuine no-breach)
            continue
        tiers_hit = {tier_map[hp] for hp in hp_set if tier_map.get(hp) in TIER_ORDER}
        if tiers_hit:
            best = min(tiers_hit, key=lambda t: TIER_ORDER.index(t))
            counts[best] += 1
        else:
            # Honeypot(s) accessed but absent from the tier map (or classified
            # never_breached): still a real breach — floor to the lowest tier so the
            # stack total equals PVR_conv instead of silently vanishing into no_breach.
            counts[_FALLBACK_TIER] += 1
            untiered_breaches += 1

    total = len(episodes)
    # Invariant: every non-empty-hp_set episode lands in exactly one tier bucket.
    assert sum(counts.values()) == total - no_breach, (
        f"tier breach accounting mismatch in {reward_debug_path}: "
        f"sum(counts)={sum(counts.values())} != any_breach={total - no_breach}"
    )
    if untiered_breaches:
        print(
            f"[plot_tier_decomposition] {reward_debug_path.parent.name}: "
            f"{untiered_breaches} breach episode(s) had honeypots absent from the tier "
            f"map; floored to '{_FALLBACK_TIER}'.",
            file=sys.stderr,
        )
    return {
        "total": total,  # n_eps_with_sql = C*_R denominator
        "n_attack_ep": len(attack_ep_ids),
        "no_breach": no_breach,
        "any_breach": total - no_breach,
        **counts,
    }


def compute_decomposition(selfplay_dir: str, out_dir: str | Path | None = None) -> dict[int, dict]:
    base = Path(selfplay_dir)
    tier_map = _load_tier_map(selfplay_dir, out_dir=out_dir)
    ce_pairings = base / "cross_eval" / "pairings"
    result: dict[int, dict] = {}
    for i in range(8):
        p = ce_pairings / f"red_{i}_blue_{i}" / "reward_debug.jsonl"
        if not p.is_file():
            continue
        stats = _compute_tier_pvr(p, tier_map)
        result[i] = stats
    return result


def plot_tier_decomposition(
    results: list[tuple[str, str]],
    out_path: str | Path = "figures/tier_pvr_decomposition.png",
    show_ci: bool = True,
) -> Path:
    out_path = Path(out_path)
    label, selfplay_dir = results[0]

    # The orchestrator writes honeypot_tiers.json (via plot_honeypot_difficulty) into the
    # same out_dir as this figure, so look there first for the tier map.
    data = compute_decomposition(selfplay_dir, out_dir=out_path.parent)
    if not data:
        print("[plot_tier_decomposition] No diagonal cross_eval data found.", file=sys.stderr)
        return out_path

    iters = sorted(data.keys())
    totals = [data[i]["total"] for i in iters]

    # Per-tier PVR_conv (%) and Wilson CIs
    tier_pvr: dict[str, list[float]] = {t: [] for t in TIER_ORDER}
    tier_lo: dict[str, list[float]] = {t: [] for t in TIER_ORDER}
    tier_hi: dict[str, list[float]] = {t: [] for t in TIER_ORDER}
    for i in iters:
        n = data[i]["total"]
        for t in TIER_ORDER:
            k = data[i][t]
            lo, hi = wilson_ci_pct(k, n, z=2.576)
            pct = 100.0 * k / n if n else 0.0
            tier_pvr[t].append(pct)
            tier_lo[t].append(lo)
            tier_hi[t].append(hi)

    total_asr = [sum(tier_pvr[t][j] for t in TIER_ORDER) for j in range(len(iters))]

    # Sidecar JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = {
        str(i): {
            "total_episodes": data[i]["total"],
            "any_breach": data[i]["any_breach"],
            "pvr_conv_pct": round(total_asr[j], 2),
            **{t: {"k": data[i][t], "pct": round(tier_pvr[t][j], 2),
                    "ci_lo": round(tier_lo[t][j], 2), "ci_hi": round(tier_hi[t][j], 2)}
               for t in TIER_ORDER},
        }
        for j, i in enumerate(iters)
    }
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    bottom = np.zeros(len(iters))
    for t in TIER_ORDER:
        ys = np.array(tier_pvr[t])
        ax.bar(iters, ys, bottom=bottom,
               color=TIER_COLORS[t], label=TIER_LABELS[t],
               edgecolor="white", linewidth=0.4)
        # Error bars only for the largest tier (pii_dominant) for clarity
        if t == "pii_dominant" and show_ci:
            errs = [
                [max(0.0, pct - lo) for pct, lo in zip(tier_pvr[t], tier_lo[t])],
                [max(0.0, hi - pct) for pct, hi in zip(tier_pvr[t], tier_hi[t])],
            ]
            ax.errorbar(iters, bottom + ys, yerr=errs,
                        fmt="none", color="black", capsize=3, elinewidth=0.8, capthick=0.8)
        bottom += ys

    # Total ASR line
    ax.plot(iters, total_asr, "k--o", linewidth=1.3, markersize=5,
            label=r"Total PVR$_\mathrm{conv}$", zorder=5)

    plateau = float(np.mean(total_asr))
    ax.axhline(plateau, color=GRAY_COL, linewidth=0.9, linestyle=":",
               label=f"Mean plateau {plateau:.1f} %")

    ax.set_xlabel("Self-play iteration (diagonal red$_i$ vs blue$_i$)")
    ax.set_ylabel(r"PVR$_\mathrm{conv}$ (%) by tier")
    ax.set_title(
        f"Per-tier PVR decomposition — {label}\n"
        r"Stack $\Sigma$ = total ASR; pii$_\mathrm{dominant}$ > harvestable > rare priority"
    )
    ax.set_xticks(iters)
    ax.set_ylim(0, max(total_asr) * 1.35)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    ax.grid(True, axis="y", alpha=0.35)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Stderr summary
    print(f"[plot_tier_decomposition] iters={iters}", file=sys.stderr)
    for j, i in enumerate(iters):
        row = sidecar[str(i)]
        print(f"  iter {i}: total={row['total_episodes']}  ASR={row['pvr_conv_pct']:.1f}%  "
              + "  ".join(f"{t}={row[t]['pct']:.1f}%" for t in TIER_ORDER),
              file=sys.stderr)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--results", nargs="+", required=True, metavar="DIR[:LABEL]")
    ap.add_argument("--out", default="figures/tier_pvr_decomposition.png")
    args = ap.parse_args()
    results = parse_results_arg(args.results)
    out = plot_tier_decomposition(results, args.out)
    # NOTE: plot_tier_decomposition writes its own richer per-iter sidecar.
    print(f"[{DESCRIPTION[:80]}...]\n  → {out}")


if __name__ == "__main__":
    main()
