"""
figure_registry.py — single source of truth mapping each enabled paper figure to
its pure metric-compute function.

Both the plotting orchestrator (plot_paper_figures.run_all) and the JSON exporter
(util/export_results_json.py) drive off this registry, so the numbers in the
figures, their PNG sidecars, and the exported JSON tree all come from one
`compute_<key>(results, **kwargs) -> dict` per figure family. No metric is
re-derived in a second place.

This module sits one layer above plotting/_data.py (the loader layer) and below
plot_paper_figures.py. Compute callables are imported lazily inside thin thunks so
that (a) importing the registry stays cheap, and (b) a single broken plot module
does not break the whole registry import.

Section assignment (json_section) controls which export file a figure lands in:
  headline   — the paper "money numbers"
  cross_eval — N×N cross-eval matrices and per-cell decompositions
  training   — per-iteration training-time signals + cost
  derived    — secondary / appendix analyses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSpec:
    """One enabled figure family.

    key            matches the run_all() skip-key (e.g. "pvr_asymptote").
    compute        compute_<key>(results, **kwargs) -> dict (pure, JSON-serializable).
    json_section   "headline" | "cross_eval" | "training" | "derived".
    per_run        True  -> called once per (label, dir) target.
                   False -> takes the full results list at once (e.g. baseline).
    needs_marft    coverage/yield/universe degrade to None without the MARFT env.
    depends_on     figure keys that must compute first (e.g. honeypot_difficulty
                   writes honeypot_tiers.json that tier figures read).
    optional_deps  python packages the figure needs; absence -> recorded skip.
    default_kwargs base kwargs passed to compute (subdir names etc.).
    """

    key: str
    compute: Callable[..., dict]
    json_section: str
    per_run: bool = True
    needs_marft: bool = False
    depends_on: tuple[str, ...] = ()
    optional_deps: tuple[str, ...] = ()
    default_kwargs: dict = field(default_factory=dict)


def _lazy(module_path: str, fn_name: str) -> Callable[..., dict]:
    """Return a thunk that imports `module_path.fn_name` on first call."""

    def _thunk(results, **kwargs):
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, fn_name)(results, **kwargs)

    _thunk.__name__ = f"lazy_{fn_name}"
    return _thunk


# ---------------------------------------------------------------------------
# Adapters for figures whose pure compute predates this registry and takes a
# selfplay_dir rather than the (label, dir) results list, or whose figure-level
# percentages are assembled from raw counts via shared primitives.
# ---------------------------------------------------------------------------


def _compute_rank_invariance(results, *, cross_eval_subdir: str = "cross_eval", **_):
    """Diagonal vs off-diagonal PVR_conv; cells -> axis+matrix (drops bulky dups)."""
    from plotting.cross_eval_rank_invariance import compute_rank_invariance

    d = compute_rank_invariance(results[0][1], cross_eval_subdir)
    if not d:
        return {}
    cells = d.pop("cells", {})
    # diag/offdiag value lists are recoverable from the matrix below — drop them.
    d.pop("diag_values", None)
    d.pop("offdiag_values", None)
    n = int(d.get("n_iters") or 0)
    values = [[None] * n for _ in range(n)]
    for key, cv in cells.items():
        # key like "r3xb5"
        try:
            ri = int(key[1 : key.index("x")])
            bi = int(key[key.index("xb") + 2 :])
        except (ValueError, KeyError):
            continue
        if 0 <= ri < n and 0 <= bi < n:
            values[ri][bi] = cv.get("asr")
    d["matrix"] = {
        "red_iters": list(range(n)),
        "blue_iters": list(range(n)),
        "values": values,
    }
    return d


def _compute_tier_decomposition(results, *, out_dir=None, **_):
    """Per-tier PVR_conv stack + 99% Wilson CI (reuses util.metrics.wilson_ci_pct)."""
    from plotting.plot_tier_decomposition import compute_decomposition, TIER_ORDER
    from util.metrics import wilson_ci_pct

    data = compute_decomposition(results[0][1], out_dir=out_dir)
    if not data:
        return {}
    out: dict[str, dict] = {}
    for i in sorted(data):
        row = data[i]
        n = int(row.get("total", 0) or 0)
        entry: dict = {"total_episodes": n, "any_breach": row.get("any_breach")}
        total_pct = 0.0
        for t in TIER_ORDER:
            k = int(row.get(t, 0) or 0)
            lo, hi = wilson_ci_pct(k, n, z=2.576) if n else (0.0, 0.0)
            pct = 100.0 * k / n if n else 0.0
            total_pct += pct
            entry[t] = {"k": k, "pct": pct, "ci_lo": lo, "ci_hi": hi}
        entry["pvr_conv_pct"] = total_pct
        out[str(i)] = entry
    return out


def _compute_defender_concentration(results, *, out_dir=None, **_):
    from plotting.plot_defender_concentration import compute_defender_concentration

    data = compute_defender_concentration(results[0][1], out_dir=out_dir)
    if not data:
        return {}
    return {str(i): v for i, v in sorted(data.items())}


def _compute_per_target_response(results, *, cross_eval_subdir: str = "cross_eval", **_):
    """Per-target refusal/breach/accepted-clean counts + rates (rate = k/intent)."""
    from plotting.plot_per_target_defender_response import compute_per_target_response

    data = compute_per_target_response(results[0][1], cross_eval_subdir, diagonal_only=True)
    if not data:
        return {}
    enriched: dict[str, dict] = {}
    for hp, c in data.get("per_target", {}).items():
        intent = int(c.get("intent", 0) or 0)
        if intent == 0:
            continue
        enriched[hp] = {
            **c,
            "refusal_rate_pct": 100.0 * c.get("refused", 0) / intent,
            "breach_rate_pct": 100.0 * c.get("breached", 0) / intent,
            "accepted_clean_pct": 100.0 * c.get("accepted_clean", 0) / intent,
        }
    return {"totals": data.get("totals", {}), "per_target": enriched}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


REGISTRY: list[FigureSpec] = [
    # ---- headline -------------------------------------------------------
    FigureSpec(
        key="pvr_asymptote",
        compute=_lazy("plotting.plot_pvr_asymptote", "compute_pvr_asymptote"),
        json_section="headline",
    ),
    FigureSpec(
        key="diagonal_convergence",
        compute=_lazy("plotting.plot_diagonal_convergence", "compute_diagonal_convergence"),
        json_section="headline",
    ),
    FigureSpec(
        key="baseline_vs_trained",
        compute=_lazy("plotting.plot_baseline_vs_trained", "compute_baseline_vs_trained"),
        json_section="headline",
        per_run=False,
        default_kwargs={
            "cross_eval_subdir": "cross_eval",
            "baseline_subdir": "cross_eval_baseline",
        },
    ),
    FigureSpec(
        key="rank_invariance",
        compute=_compute_rank_invariance,
        json_section="headline",
        default_kwargs={"cross_eval_subdir": "cross_eval"},
    ),
    FigureSpec(
        key="tier_decomposition",
        compute=_compute_tier_decomposition,
        json_section="headline",
        depends_on=("honeypot_difficulty",),
    ),
    FigureSpec(
        key="held_out_per_style_refusal",
        compute=_lazy(
            "plotting.plot_held_out_per_style_refusal",
            "compute_held_out_per_style_refusal",
        ),
        json_section="headline",
        default_kwargs={"cross_eval_subdir": "cross_eval"},
    ),
    FigureSpec(
        key="honeypot_saturation",
        compute=_lazy("plotting.plot_honeypot_saturation", "compute_honeypot_saturation"),
        json_section="headline",
        needs_marft=True,
    ),
    # ---- cross_eval -----------------------------------------------------
    FigureSpec(
        key="heatmaps",
        compute=_lazy("plotting.plot_cross_eval_heatmap", "compute_heatmaps"),
        json_section="cross_eval",
        needs_marft=True,
    ),
    FigureSpec(
        key="attempts_cdf",
        compute=_lazy("plotting.plot_attempts_cdf", "compute_attempts_cdf"),
        json_section="cross_eval",
        default_kwargs={"subdir": "cross_eval"},
    ),
    FigureSpec(
        key="coverage_yield",
        compute=_lazy("plotting.plot_coverage_yield", "compute_coverage_yield"),
        json_section="cross_eval",
        needs_marft=True,
        default_kwargs={"subdir": "cross_eval"},
    ),
    FigureSpec(
        key="generalization",
        compute=_lazy("plotting.plot_generalization", "compute_generalization"),
        json_section="cross_eval",
    ),
    FigureSpec(
        key="per_target_defense",
        compute=_compute_per_target_response,
        json_section="cross_eval",
        default_kwargs={"cross_eval_subdir": "cross_eval"},
    ),
    FigureSpec(
        key="top_target_mechanism",
        compute=_lazy("plotting.plot_top_target_mechanism", "compute_top_target_mechanism"),
        json_section="cross_eval",
        default_kwargs={"cross_eval_subdir": "cross_eval"},
    ),
    # ---- training -------------------------------------------------------
    FigureSpec(
        key="running_time",
        compute=_lazy("plotting.plot_running_time", "compute_running_time"),
        json_section="training",
    ),
    FigureSpec(
        key="lora",
        compute=_lazy("plotting.plot_lora_diversity", "compute_lora"),
        json_section="training",
        optional_deps=(),  # prefers precomputed lora_delta_metrics.json; no torch
    ),
    FigureSpec(
        key="training_curves",
        compute=_lazy("plotting.plot_training_curves", "compute_selfplay_tail"),
        json_section="training",
    ),
    FigureSpec(
        key="defender_concentration",
        compute=_compute_defender_concentration,
        json_section="training",
        depends_on=("honeypot_difficulty",),
    ),
    FigureSpec(
        key="honeypot_difficulty",
        compute=_lazy("plotting.plot_honeypot_difficulty", "compute_honeypot_difficulty"),
        json_section="training",
        needs_marft=True,  # universe label only; has a fallback list
    ),
    # ---- derived --------------------------------------------------------
    FigureSpec(
        key="attack_evolution",
        compute=_lazy("plotting.plot_attack_evolution", "compute_attack_evolution"),
        json_section="derived",
        optional_deps=("sklearn",),  # TF-IDF matrix only; regex fields survive
    ),
    FigureSpec(
        key="honeypot_per_iter",
        compute=_lazy("plotting.plot_honeypot_per_iter_heatmap", "compute_honeypot_per_iter_heatmap"),
        json_section="derived",
        depends_on=("honeypot_difficulty",),
    ),
]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def ordered_registry(
    skip: set[str] | None = None,
    only: set[str] | None = None,
) -> list[FigureSpec]:
    """Return REGISTRY filtered by skip/only and topologically ordered by depends_on.

    A dependency that is itself skipped is simply not waited on (its dependents
    still run; they degrade via _load_tier_map's empty-map fallback).
    """
    skip = skip or set()
    selected = [
        s for s in REGISTRY
        if s.key not in skip and (only is None or s.key in only)
    ]
    selected_keys = {s.key for s in selected}

    ordered: list[FigureSpec] = []
    emitted: set[str] = set()
    remaining = list(selected)
    # Kahn-style: emit a spec once all its in-set deps are emitted. Guard against
    # cycles with a bounded number of passes.
    for _ in range(len(remaining) + 1):
        progressed = False
        still: list[FigureSpec] = []
        for s in remaining:
            unmet = [d for d in s.depends_on if d in selected_keys and d not in emitted]
            if unmet:
                still.append(s)
            else:
                ordered.append(s)
                emitted.add(s.key)
                progressed = True
        remaining = still
        if not remaining:
            break
        if not progressed:
            # Dependency cycle (shouldn't happen) — emit the rest in declared order.
            ordered.extend(remaining)
            break
    return ordered


# ---------------------------------------------------------------------------
# Glossary — definition/formula/source/ci_method for each metric key, so the
# exported JSON is self-describing and an LLM never needs to read the code.
# ---------------------------------------------------------------------------


GLOSSARY: dict[str, dict] = {
    "asr": {
        "alias": "PVR_conv",
        "definition": "Policy-violation rate, conversation-level (attack episodes that breach ≥1 honeypot).",
        "formula": "n_honeypot_eps / n_eps_with_sql * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "pvr_conv": {
        "alias": "asr",
        "definition": "Conversation-level policy-violation rate (same as asr).",
        "formula": "n_honeypot_eps / n_eps_with_sql * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "pvr_turn": {
        "definition": "Turn-level policy-violation rate over all attack turns.",
        "formula": "n_fp_steps / n_attack_steps * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "pvr_sql_turn": {
        "definition": "Turn-level violation rate over SQL-emitting attack turns only.",
        "formula": "n_fp_steps / n_sql_emitted * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "work_factor": {
        "definition": "Mean SQL attempts per successful breach (attacker effort).",
        "formula": "n_sql_emitted / n_fp_steps",
        "source": "util.metrics.compute_pairing_metrics",
        "note": "None when no breach; requires the MARFT env for honeypot detection.",
    },
    "coverage_pct": {
        "definition": "Fraction of the honeypot universe referenced in parsed SQL.",
        "formula": "|referenced_ids| / honeypot_universe * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
        "note": "Requires the MARFT env; None otherwise.",
    },
    "yield_pct": {
        "definition": "Fraction of the honeypot universe actually accessed by execution.",
        "formula": "|accessed_ids| / honeypot_universe * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "tnr": {
        "definition": "True-negative rate: attack episodes correctly refused.",
        "formula": "n_refused_eps / n_attack_eps * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "tpr": {
        "definition": "True-positive rate: benign turns correctly answered.",
        "formula": "n_tp / n_benign * 100",
        "source": "util.metrics.compute_pairing_metrics",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "brr": {
        "definition": "Benign refusal rate (over-refusal of benign queries).",
        "formula": "sum(is_benign_denial) / n_benign * 100",
        "source": "util.metrics.denial_rate_with_ci / is_benign_denial",
        "ci_method": "Wilson 99% (z=2.576)",
    },
    "held_out_refusal": {
        "definition": "Per-style benign refusal rate on held-out benign eval.",
        "formula": "refused / n * 100",
        "source": "plot_held_out_per_style_refusal",
        "ci_method": "Wilson 95% (z=1.96)",
    },
    "dominance": {
        "definition": "Self-play dominance scalar (>0 attacker-favored, <0 defender-favored).",
        "formula": "max(0, HM(1-pvr_turn, tpr) * (1 - 10*cfr)) - min(1, 5*cfr)",
        "source": "util.metrics.compute_pairing_metrics",
    },
    "eis": {
        "definition": "Environment Interaction Steps — GPU-invariant compute unit.",
        "formula": "training_state.json total_num_steps",
        "source": "plotting._data.load_training_state / compute_cost_analysis.json",
    },
    "block_rate_pct": {
        "definition": "Training-time defender block rate over attack turns.",
        "formula": "refused / total_attack * 100",
        "source": "plot_defender_concentration.compute_defender_concentration",
    },
}
