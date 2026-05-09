"""
plot_paper_figures.py — Master orchestrator for paper-quality figures.

Calls every plot in a fixed, reproducible order and writes all outputs to
figures/ (or --out-dir).  Prints each plot's description alongside its path.

Usage:
    python plotting/plot_paper_figures.py --results results-<ID>
    python plotting/plot_paper_figures.py \\
        --results results-A:Llama-8B results-B:Llama-70B \\
        --out-dir figures/ \\
        --pvr-sources cross_eval,human_eval \\
        --human-eval-parent data/human_eval/ \\
        --skip lora

Output order:
    figures/pvr.png
    figures/pvr_conv.png
    figures/work_factor.png
    figures/brr.png
    figures/semantic_diversity_*.png
    figures/running_time.png
    figures/training_dynamics.png
    figures/compute_efficiency.png
    figures/cross_eval_pvr_conv.png
    figures/cross_eval_pvr_turn.png
    figures/quick_pvr_conv.png          (if cross_eval_quick/ exists)
    figures/quick_pvr_turn.png          (if cross_eval_quick/ exists)
    figures/diagonal_eval_pvr_conv.png  (if diagonal_eval/ exists)
    figures/diagonal_eval_pvr_turn.png  (if diagonal_eval/ exists)
    figures/diagonal_training.png
    figures/pvr_vs_normalized_eis.png
    figures/blue_convergence.png
    figures/cross_eval_attempts_cdf.png
    figures/quick_attempts_cdf.png          (if cross_eval_quick/ exists)
    figures/diagonal_eval_attempts_cdf.png  (if diagonal_eval/ exists)
    figures/cross_eval_coverage_yield.png
    figures/quick_coverage_yield.png        (if cross_eval_quick/ exists)
    figures/diagonal_eval_coverage_yield.png (if diagonal_eval/ exists)
    figures/security_utility_pareto.png
    figures/diagonal_convergence.png
    figures/lora_drift.png
    figures/lora_delta.png
    figures/lora_cosine.png
    figures/sql_error_heatmap.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support both package import and direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from plotting._data import parse_results_arg, write_sidecar
    from plotting.plot_pvr import (
        plot_pvr,
        DESCRIPTION as DESC_PVR,
        plot_pvr_conv,
        DESCRIPTION_CONV as DESC_PVR_CONV,
        plot_work_factor,
        DESCRIPTION_WF as DESC_WF,
    )
    from plotting.plot_brr import plot_brr, DESCRIPTION as DESC_BRR
    from plotting.plot_semantic_diversity import (
        plot_semantic_diversity,
        DESCRIPTION as DESC_DIV,
    )
    from plotting.plot_running_time import plot_running_time, DESCRIPTION as DESC_RT
    from plotting.plot_lora_diversity import (
        plot_lora_diversity,
        DESCRIPTION as DESC_LORA,
    )
    from plotting.plot_training_dynamics import (
        plot_training_dynamics,
        DESCRIPTION as DESC_DYN,
    )
    from plotting.plot_compute_efficiency import (
        plot_compute_efficiency,
        DESCRIPTION as DESC_EFF,
    )
    from plotting.plot_cross_eval_heatmap import (
        plot_heatmap,
        plot_training_diagonal,
        HEATMAP_JOBS,
        DESCRIPTION_TRAINING as DESC_TRAINING_DIAG,
        _heatmap_desc,
    )
    from plotting.plot_pvr_vs_normalized_eis import (
        plot_pvr_vs_normalized_eis,
        DESCRIPTION as DESC_NORM_EIS,
    )
    from plotting.plot_blue_convergence import (
        plot_blue_convergence,
        DESCRIPTION as DESC_BLUE_CONV,
    )
    from plotting.plot_attempts_cdf import (
        plot_attempts_cdf,
        ATTEMPTS_CDF_JOBS,
        DESCRIPTION as DESC_ATTEMPTS,
    )
    from plotting.plot_coverage_yield import (
        plot_coverage_yield,
        COVERAGE_YIELD_JOBS,
        DESCRIPTION as DESC_COV,
    )
    from plotting.plot_security_utility_pareto import (
        plot_security_utility_pareto,
        DESCRIPTION as DESC_PARETO,
    )
    from plotting.plot_diagonal_convergence import (
        plot_diagonal_convergence,
        DESCRIPTION as DESC_DIAG_CONV,
    )
except ImportError:
    from plotting._data import parse_results_arg, write_sidecar
    from plotting.plot_pvr import (
        plot_pvr,
        DESCRIPTION as DESC_PVR,
        plot_pvr_conv,
        DESCRIPTION_CONV as DESC_PVR_CONV,
        plot_work_factor,
        DESCRIPTION_WF as DESC_WF,
    )
    from plotting.plot_brr import plot_brr, DESCRIPTION as DESC_BRR
    from plotting.plot_semantic_diversity import (
        plot_semantic_diversity,
        DESCRIPTION as DESC_DIV,
    )
    from plotting.plot_running_time import plot_running_time, DESCRIPTION as DESC_RT
    from plotting.plot_lora_diversity import (
        plot_lora_diversity,
        DESCRIPTION as DESC_LORA,
    )
    from plotting.plot_training_dynamics import (
        plot_training_dynamics,
        DESCRIPTION as DESC_DYN,
    )
    from plotting.plot_compute_efficiency import (
        plot_compute_efficiency,
        DESCRIPTION as DESC_EFF,
    )
    from plotting.plot_cross_eval_heatmap import (
        plot_heatmap,
        plot_training_diagonal,
        HEATMAP_JOBS,
        DESCRIPTION_TRAINING as DESC_TRAINING_DIAG,
        _heatmap_desc,
    )
    from plotting.plot_pvr_vs_normalized_eis import (
        plot_pvr_vs_normalized_eis,
        DESCRIPTION as DESC_NORM_EIS,
    )
    from plotting.plot_blue_convergence import (
        plot_blue_convergence,
        DESCRIPTION as DESC_BLUE_CONV,
    )
    from plotting.plot_attempts_cdf import (
        plot_attempts_cdf,
        ATTEMPTS_CDF_JOBS,
        DESCRIPTION as DESC_ATTEMPTS,
    )
    from plotting.plot_coverage_yield import (
        plot_coverage_yield,
        COVERAGE_YIELD_JOBS,
        DESCRIPTION as DESC_COV,
    )
    from plotting.plot_security_utility_pareto import (
        plot_security_utility_pareto,
        DESCRIPTION as DESC_PARETO,
    )
    from plotting.plot_diagonal_convergence import (
        plot_diagonal_convergence,
        DESCRIPTION as DESC_DIAG_CONV,
    )


# ---------------------------------------------------------------------------
# Core orchestration function (deterministic, idempotent)
# ---------------------------------------------------------------------------


def run_all(
    results: list[tuple[str, str]],
    out_dir: str | Path,
    pvr_sources: tuple[str, ...] = ("cross_eval",),
    brr_sources: tuple[str, ...] = ("train_rollouts", "benign_eval"),
    human_eval_parent: str | None = None,
    cross_eval_subdir: str = "cross_eval",
    human_queries: str = "new_jailbreaks.txt",
    benign_queries: str | None = "data/benign_pool_stats.json",
    query_mode: str = "training_time",
    tail_pct: float = 0.25,
    lora_base_model: str | None = None,
    skip: set[str] | None = None,
    training_dynamics_smooth: int = 7,
    baseline_cross_eval_subdir: str = "cross_eval_baseline",
    human_eval_json: str | None = "data/human_eval/comparison.json",
) -> list[tuple[str, Path]]:
    """
    Run all paper figure plots in a fixed order.

    Args:
        results:            [(label, selfplay_dir), ...]
        out_dir:            Output directory for all figures.
        pvr_sources:        Data sources for PVR plot.
        brr_sources:        Data sources for BRR plot.
        human_eval_parent:  Parent dir with iter_N/summary.json for human-eval.
        cross_eval_subdir:  Subdir within selfplay_dir for cross-eval data.
        human_queries:      Path to jailbreak queries file.
        benign_queries:     Path to benign_pool_stats.json (None to disable overlay).
        query_mode:         Query collection mode for semantic diversity.
        tail_pct:           Tail fraction for final_episode query mode.
        lora_base_model:    Override base model for LoRA diversity plot.
        skip:               Set of plot names to skip.
                            Valid names: pvr, pvr_conv, work_factor, brr, diversity,
                            running_time, training_dynamics, compute_efficiency,
                            security_utility_pareto, heatmaps, training_diagonal,
                            pvr_vs_normalized_eis, blue_convergence, attempts_cdf,
                            coverage_yield, diagonal_convergence, lora,
                            sql_error_heatmap.

    Returns:
        [(description, output_path), ...] for each plot produced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = skip or set()
    produced: list[tuple[str, Path]] = []

    # 1. PVR_turn
    if "pvr" not in skip:
        path = plot_pvr(
            results,
            out_dir / "pvr.png",
            sources=pvr_sources,
            human_eval_parent=human_eval_parent,
            cross_eval_subdir=cross_eval_subdir,
        )
        produced.append((DESC_PVR, path))
        write_sidecar(path, DESC_PVR, results)

    # 2. PVR_conv
    if "pvr_conv" not in skip:
        path = plot_pvr_conv(
            results,
            out_dir / "pvr_conv.png",
            sources=pvr_sources,
            human_eval_parent=human_eval_parent,
            cross_eval_subdir=cross_eval_subdir,
        )
        produced.append((DESC_PVR_CONV, path))
        write_sidecar(path, DESC_PVR_CONV, results)

    # 3. Work Factor (WF = 1/PVR_turn)
    if "work_factor" not in skip:
        path = plot_work_factor(
            results,
            out_dir / "work_factor.png",
            sources=pvr_sources,
            human_eval_parent=human_eval_parent,
            cross_eval_subdir=cross_eval_subdir,
        )
        produced.append((DESC_WF, path))
        write_sidecar(path, DESC_WF, results)

    # 4. BRR — one PNG per source
    if "brr" not in skip:
        brr_paths = plot_brr(
            results,
            out_dir=out_dir,
            filename_prefix="brr",
            sources=brr_sources,
            human_eval_parent=human_eval_parent,
            cross_eval_subdir=cross_eval_subdir,
        )
        for src, path in brr_paths.items():
            produced.append((DESC_BRR, path))
            write_sidecar(path, DESC_BRR, results)

    # 5. Semantic diversity
    if "diversity" not in skip:
        EMBEDDERS: list[tuple[str, str]] = [
            # ("MINILM", "all-MiniLM-L6-v2"),
            # # BAAI's BGE-M3: SOTA for mixed contexts. Handles multi-lingual, long-context (8k),
            # # and generates dense, sparse, and multi-vector embeddings simultaneously.
            # ("BGE_M3", "BAAI/bge-m3"),
            # # Nomic v1.5: Highly efficient 8k context model featuring Matryoshka Representation Learning.
            # # Allows you to truncate embedding dimensions without significant performance loss.
            # ("NOMIC_1_5", "nomic-ai/nomic-embed-text-v1.5"),
            # # Jina v3: The latest generation (late 2024) featuring task-specific LoRA adapters
            # # (retrieval, clustering, classification) built directly into the architecture. 8192 context.
            # ("JINA_V3", "jinaai/jina-embeddings-v3"),
            # Snowflake Arctic Embed v2: Explicitly optimized for enterprise data, SQL schemas,
            # and code retrieval tasks. Excellent performance-to-parameter ratio.
            ("ARCTIC_V2", "Snowflake/snowflake-arctic-embed-l-v2.0"),
            # Alibaba GTE Large v1.5: Consistently sits near the top of the MTEB leaderboard for
            # models under 1B parameters. Very strong on complex English and code reasoning.
            ("GTE_V1_5", "Alibaba-NLP/gte-large-en-v1.5"),
            # # --- LLM-Backed (Heavyweight Open) ---
            # # Salesforce SFR Mistral: If your security paper needs a benchmark for the absolute
            # # ceiling of open-weight performance. Uses a 7B LLM backbone for deep logical reasoning.
            # ("SFR_MISTRAL", "Salesforce/SFR-Embedding-Mistral"),
        ]

        for id, embedder in EMBEDDERS:
            path, sem_metrics = plot_semantic_diversity(
                results,
                out_dir / f"semantic_diversity_{id}.png",
                human_queries_path=human_queries,
                query_mode=query_mode,
                tail_pct=tail_pct,
                embedder=embedder,
                cross_eval_subdir=cross_eval_subdir,
                benign_queries_path=benign_queries,
            )
            produced.append((DESC_DIV, path))
            write_sidecar(path, DESC_DIV, results, sem_metrics)

    # 5. Running time (EIS)
    if "running_time" not in skip:
        path = plot_running_time(results, out_dir / "running_time.png")
        produced.append((DESC_RT, path))
        write_sidecar(path, DESC_RT, results)

    # 6. Training dynamics (reward-vs-EIS learning curves, process visualization)
    if "training_dynamics" not in skip:
        path = plot_training_dynamics(
            results,
            out_dir / "training_dynamics.png",
            smooth_window=training_dynamics_smooth,
        )
        produced.append((DESC_DYN, path))
        write_sidecar(path, DESC_DYN, results)

    # 7. Compute efficiency (PVR_conv and BRR vs cumulative EIS)
    if "compute_efficiency" not in skip:
        path = plot_compute_efficiency(results, out_dir / "compute_efficiency.png")
        produced.append((DESC_EFF, path))
        write_sidecar(path, DESC_EFF, results)

    # 7b. Security–Utility Pareto frontier across checkpoints
    if "security_utility_pareto" not in skip:
        path = plot_security_utility_pareto(
            results,
            out_dir / "security_utility_pareto.png",
            cross_eval_subdir=cross_eval_subdir,
        )
        produced.append((DESC_PARETO, path))
        write_sidecar(path, DESC_PARETO, results)

    # 8. Cross-eval heatmaps — PVR_conv + PVR_turn for all available subdirs.
    #    Each call is a no-op (prints a warning) if the subdir doesn't exist.
    if "heatmaps" not in skip:
        for subdir, metric, fname in HEATMAP_JOBS:
            path = plot_heatmap(results, out_dir / fname, subdir=subdir, metric=metric)
            desc = _heatmap_desc(subdir, metric)
            produced.append((desc, path))
            write_sidecar(path, desc, results)

    # 9. Training-log diagonal: PVR_conv from per-iteration blueteam logs.
    if "training_diagonal" not in skip:
        path = plot_training_diagonal(results, out_dir / "diagonal_training.png")
        produced.append((DESC_TRAINING_DIAG, path))
        write_sidecar(path, DESC_TRAINING_DIAG, results)

    # 10. ΔPVRconv vs normalized EIS — direct cross-team efficiency comparison
    if "pvr_vs_normalized_eis" not in skip:
        path = plot_pvr_vs_normalized_eis(
            results, out_dir / "pvr_vs_normalized_eis.png"
        )
        produced.append((DESC_NORM_EIS, path))
        write_sidecar(path, DESC_NORM_EIS, results)

    # 11. Blue team convergence — plateau analysis and marginal efficiency
    if "blue_convergence" not in skip:
        path = plot_blue_convergence(results, out_dir / "blue_convergence.png")
        produced.append((DESC_BLUE_CONV, path))
        write_sidecar(path, DESC_BLUE_CONV, results)

    # 12. Decomposed-PVR: attempts-to-compromise CDF
    if "attempts_cdf" not in skip:
        for subdir_name, fname in ATTEMPTS_CDF_JOBS:
            path = plot_attempts_cdf(results, out_dir / fname, subdir=subdir_name)
            produced.append((DESC_ATTEMPTS, path))
            write_sidecar(path, DESC_ATTEMPTS, results)

    # 13. Decomposed-PVR: honeypot coverage vs. yield
    if "coverage_yield" not in skip:
        for subdir_name, fname in COVERAGE_YIELD_JOBS:
            path = plot_coverage_yield(results, out_dir / fname, subdir=subdir_name)
            produced.append((DESC_COV, path))
            write_sidecar(path, DESC_COV, results)

    # 14. Diagonal convergence — PVR_conv, PVR_turn, BRR for adjacent pairings
    if "diagonal_convergence" not in skip:
        path, dc_metrics = plot_diagonal_convergence(
            results,
            out_dir / "diagonal_convergence.png",
            eval_subdir=cross_eval_subdir,
        )
        produced.append((DESC_DIAG_CONV, path))
        write_sidecar(path, DESC_DIAG_CONV, results, dc_metrics)

    # 15. LoRA diversity (three output files)
    if "lora" not in skip:
        drift, delta, cosine = plot_lora_diversity(
            results,
            out_dir,
            base_model=lora_base_model,
        )
        for p in (drift, delta, cosine):
            produced.append((DESC_LORA, p))
            write_sidecar(p, DESC_LORA, results)

    # 16. Pillar 1 headline: utility by query style (replaces brr.png as headline)
    if "utility_by_style" not in skip:
        try:
            from plotting.plot_utility_by_style import (
                plot_utility_by_style,
                DESCRIPTION as DESC_UTIL_STYLE,
            )
            path = plot_utility_by_style(results, out_dir / "utility_by_style.png")
            produced.append((DESC_UTIL_STYLE, path))
            write_sidecar(path, DESC_UTIL_STYLE, results)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] utility_by_style skipped: {e}", file=sys.stderr)

    # 17. Pillar 1 supporting: per-style refusal (training-time denial) — appendix
    if "per_style_refusal" not in skip:
        try:
            from plotting.plot_per_style_refusal import (
                plot_per_style_refusal,
                DESCRIPTION as DESC_PER_STYLE,
            )
            path = plot_per_style_refusal(results, out_dir / "per_style_refusal.png")
            produced.append((DESC_PER_STYLE, path))
            write_sidecar(path, DESC_PER_STYLE, results)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] per_style_refusal skipped: {e}", file=sys.stderr)

    # 18. Pillar 4: generalization (column-0 / row-0 + diagonal reference)
    if "generalization" not in skip:
        try:
            from plotting.plot_generalization import (
                plot_generalization,
                DESCRIPTION as DESC_GEN,
            )
            path = plot_generalization(
                results,
                out_dir / "generalization.png",
                subdir=cross_eval_subdir,
            )
            produced.append((DESC_GEN, path))
            write_sidecar(path, DESC_GEN, results)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] generalization skipped: {e}", file=sys.stderr)

    # 19. Pillar 3: per-honeypot difficulty + tier classification (canonical narrative)
    if "honeypot_difficulty" not in skip:
        try:
            from plotting.plot_honeypot_difficulty import (
                plot_honeypot_difficulty,
                DESCRIPTION as DESC_HP_DIFF,
            )
            path = plot_honeypot_difficulty(
                results, out_dir / "honeypot_difficulty.png"
            )
            produced.append((DESC_HP_DIFF, path))
            write_sidecar(path, DESC_HP_DIFF, results)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] honeypot_difficulty skipped: {e}", file=sys.stderr)

    # 20. Pillar 5: honeypot saturation (training-time vs eval-time)
    if "honeypot_saturation" not in skip:
        try:
            from plotting.plot_honeypot_saturation import (
                plot_honeypot_saturation,
                DESCRIPTION as DESC_HP_SAT,
            )
            path = plot_honeypot_saturation(
                results, out_dir / "honeypot_saturation.png"
            )
            produced.append((DESC_HP_SAT, path))
            write_sidecar(path, DESC_HP_SAT, results)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] honeypot_saturation skipped: {e}", file=sys.stderr)

    # 21. Pillar 3 appendix: per-honeypot training-vs-eval coverage
    if "honeypot_training_vs_eval" not in skip:
        try:
            import subprocess
            from pathlib import Path as _P
            script = _P(__file__).resolve().parent.parent / "util" / "honeypot_training_vs_eval.py"
            for label, selfplay_dir in results:
                out_png = out_dir / "honeypot_training_vs_eval.png"
                cmd = [
                    sys.executable, str(script),
                    "--results-dir", selfplay_dir,
                    "--out", str(out_png),
                ]
                rc = subprocess.run(cmd).returncode
                if rc == 0:
                    desc = (
                        "Per-honeypot training-time vs eval-time hits, grouped by tier. "
                        "Stars mark training-only honeypots and never-breached-at-eval. "
                        "Supports Pillar 3 attacker-coverage-gap framing."
                    )
                    produced.append((desc, out_png))
                    write_sidecar(out_png, desc, results)
                break  # single-run figure
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] honeypot_training_vs_eval skipped: {e}", file=sys.stderr)

    # 22. Pillar 2 fourth signal: iter-6 novelty recovery decomposed by tier
    if "iter6_novelty" not in skip:
        try:
            import subprocess
            from pathlib import Path as _P
            script = _P(__file__).resolve().parent.parent / "util" / "iter6_novelty_breakdown.py"
            for label, selfplay_dir in results:
                out_png = out_dir / "iter6_novelty_recovery.png"
                cmd = [
                    sys.executable, str(script),
                    "--results-dir", selfplay_dir,
                    "--out", str(out_png),
                    "--iters", "1", "2", "3", "4", "5", "6", "7",
                ]
                rc = subprocess.run(cmd).returncode
                if rc == 0:
                    desc = (
                        "Iter-6 novelty recovery decomposed by tier — supports "
                        "Pillar 2 'broad-tier transient defender regression' framing."
                    )
                    produced.append((desc, out_png))
                    write_sidecar(out_png, desc, results)
                break
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] iter6_novelty skipped: {e}", file=sys.stderr)

    # 23a. Held-out per-style benign refusal (Pillar 1 headline).
    if "held_out_per_style_refusal" not in skip:
        try:
            from plotting.plot_held_out_per_style_refusal import (
                plot_held_out_per_style_refusal,
                DESCRIPTION as DESC_HELD_OUT,
            )
            path = plot_held_out_per_style_refusal(
                results, cross_eval_subdir=cross_eval_subdir, out_dir=str(out_dir)
            )
            produced.append((DESC_HELD_OUT, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] held_out_per_style_refusal skipped: {e}", file=sys.stderr)

    # 23b. Rank-invariance figure (Pillar 4: diagonal vs off-diagonal PVR_conv).
    if "cross_eval_rank_invariance" not in skip:
        try:
            from plotting.cross_eval_rank_invariance import (
                plot_rank_invariance,
                DESCRIPTION as DESC_RANK,
            )
            path = plot_rank_invariance(
                results, cross_eval_subdir=cross_eval_subdir, out_dir=str(out_dir)
            )
            produced.append((DESC_RANK, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] cross_eval_rank_invariance skipped: {e}", file=sys.stderr)

    # 23c. Defender concentration figure (Pillar 3: block rate stability + breach tiers).
    if "defender_concentration" not in skip:
        try:
            from plotting.plot_defender_concentration import (
                plot_defender_concentration,
                DESCRIPTION as DESC_DEF,
            )
            path = plot_defender_concentration(results, out_dir=str(out_dir))
            produced.append((DESC_DEF, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] defender_concentration skipped: {e}", file=sys.stderr)

    # 23d. PVR asymptote (diagonal convergence plot) — previously not in orchestrator.
    if "pvr_asymptote" not in skip:
        try:
            from plotting.plot_pvr_asymptote import (
                plot_pvr_asymptote,
                DESCRIPTION as DESC_ASYM,
            )
            path, asym_metrics = plot_pvr_asymptote(results, out_dir / "pvr_asymptote.png")
            produced.append((DESC_ASYM, path))
            write_sidecar(path, DESC_ASYM, results, asym_metrics)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] pvr_asymptote skipped: {e}", file=sys.stderr)

    # 23e. LoRA orthogonality null check — previously not in orchestrator.
    if "lora_orthogonality" not in skip:
        try:
            from plotting.plot_lora_orthogonality import (
                plot_lora_orthogonality,
                DESCRIPTION as DESC_ORTH,
            )
            path = plot_lora_orthogonality(results, out_dir / "lora_orthogonality.png")
            produced.append((DESC_ORTH, path))
            write_sidecar(path, DESC_ORTH, results)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] lora_orthogonality skipped: {e}", file=sys.stderr)

    # 23f. Red-team termination: training steps + novel honeypot discovery.
    if "red_termination" not in skip:
        try:
            from util.plot_red_termination import (
                plot_red_termination,
            )
            DESC_RED_TERM = "Red saturation: training steps and novel discovery per iteration"
            path = plot_red_termination(results, out_dir=str(out_dir))
            produced.append((DESC_RED_TERM, path))
            # NOTE: plot_red_termination writes its own per-iter sidecar; do not overwrite.
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] red_termination skipped: {e}", file=sys.stderr)

    # 23g. Per-tier PVR decomposition: stack of pii_dominant / harvestable / rare.
    if "tier_decomposition" not in skip:
        try:
            from plotting.plot_tier_decomposition import plot_tier_decomposition
            DESC_TIER = "Per-tier PVR_conv decomposition with 99% Wilson CIs"
            path = plot_tier_decomposition(results, out_path=out_dir / "tier_pvr_decomposition.png")
            produced.append((DESC_TIER, path))
            # NOTE: plot_tier_decomposition writes its own per-iter sidecar; do not overwrite.
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] tier_decomposition skipped: {e}", file=sys.stderr)

    # 23h. Baseline-vs-trained defender comparison (resolves "did self-play do anything").
    if "baseline_compare" not in skip:
        try:
            from plotting.plot_baseline_vs_trained import (
                plot_baseline_vs_trained,
                DESCRIPTION as DESC_BASELINE,
            )
            path = plot_baseline_vs_trained(
                results,
                out_dir=str(out_dir),
                cross_eval_subdir=cross_eval_subdir,
                baseline_subdir=baseline_cross_eval_subdir,
            )
            produced.append((DESC_BASELINE, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] baseline_compare skipped: {e}", file=sys.stderr)

    # 23i. Per-honeypot × per-iter breach heatmap (composition stability across iters).
    if "honeypot_per_iter" not in skip:
        try:
            from plotting.plot_honeypot_per_iter_heatmap import (
                plot_honeypot_per_iter_heatmap,
                DESCRIPTION as DESC_HP_HM,
            )
            path = plot_honeypot_per_iter_heatmap(results, out_dir=str(out_dir))
            produced.append((DESC_HP_HM, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] honeypot_per_iter skipped: {e}", file=sys.stderr)

    # 23j. Per-target defender response (refusal vs breach vs accepted-clean).
    if "per_target_defense" not in skip:
        try:
            from plotting.plot_per_target_defender_response import (
                plot_per_target_defender_response,
                DESCRIPTION as DESC_PER_TARGET,
            )
            path = plot_per_target_defender_response(
                results, out_dir=str(out_dir), cross_eval_subdir=cross_eval_subdir
            )
            produced.append((DESC_PER_TARGET, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] per_target_defense skipped: {e}", file=sys.stderr)

    # 23k. Attack template + SQL pattern evolution across iters.
    if "attack_evolution" not in skip:
        try:
            from plotting.plot_attack_evolution import (
                plot_attack_evolution,
                DESCRIPTION as DESC_ATK_EVO,
            )
            tpl, pat = plot_attack_evolution(results, out_dir=str(out_dir))
            produced.append((DESC_ATK_EVO + " (template similarity)", tpl))
            produced.append((DESC_ATK_EVO + " (SQL pattern)", pat))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] attack_evolution skipped: {e}", file=sys.stderr)

    # 23l. Top-target attack mechanism (phone / email / passwordhash exemplars).
    if "top_target_mechanism" not in skip:
        try:
            from plotting.plot_top_target_mechanism import (
                plot_top_target_mechanism,
                DESCRIPTION as DESC_TT_MECH,
            )
            path = plot_top_target_mechanism(
                results, out_dir=str(out_dir), cross_eval_subdir=cross_eval_subdir
            )
            produced.append((DESC_TT_MECH, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] top_target_mechanism skipped: {e}", file=sys.stderr)

    # 23m. Human-eval comparison (Unprotected / Manual / RL-iter-1 on n=320 each).
    if "human_eval_compare" not in skip:
        try:
            human_eval_path = Path(human_eval_json) if human_eval_json else None
            if human_eval_path and human_eval_path.exists():
                from plotting.plot_human_eval_comparison import (
                    plot_human_eval_comparison,
                    DESCRIPTION as DESC_HUMAN,
                )
                path = plot_human_eval_comparison(
                    results, human_eval_json=str(human_eval_path), out_dir=str(out_dir)
                )
                produced.append((DESC_HUMAN, path))
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] human_eval_compare skipped: {e}", file=sys.stderr)

    # 23. SQL-error rate heatmap — diagnostic: fraction of turns with neither
    #     refusal nor valid parseable SQL, per (red_iter × blue_iter) pairing.
    if "sql_error_heatmap" not in skip:
        try:
            from plotting.cross_eval_statistics import (
                plot_sql_error_heatmap,
                DESCRIPTION as DESC_SQL_ERR,
            )
            path = plot_sql_error_heatmap(
                results,
                out_dir / "sql_error_heatmap.png",
                cross_eval_subdir=cross_eval_subdir,
            )
            produced.append((DESC_SQL_ERR, path))
            write_sidecar(path, DESC_SQL_ERR, results)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] sql_error_heatmap skipped: {e}", file=sys.stderr)

    return produced


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all paper figures for adversarial self-play results. "
            "Outputs are written to --out-dir (default: figures/)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="DIR[:LABEL]",
        help=(
            "One or more selfplay result directories. "
            "Optionally append :Label to override the auto-detected model name."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="figures/",
        metavar="DIR",
        help="Output directory for all figures (default: figures/).",
    )

    # Data sources for PVR / BRR
    parser.add_argument(
        "--pvr-sources",
        default="cross_eval",
        metavar="SRC[,SRC]",
        help="Data sources for PVR plot: cross_eval and/or human_eval (default: cross_eval).",
    )
    parser.add_argument(
        "--brr-sources",
        default="train_rollouts,benign_eval",
        metavar="SRC[,SRC]",
        help=(
            "Data sources for BRR plot: train_rollouts, benign_eval, human_eval "
            "(default: train_rollouts,benign_eval). 'cross_eval' is a deprecated alias "
            "for 'benign_eval'."
        ),
    )
    parser.add_argument(
        "--human-eval-parent",
        default=None,
        metavar="DIR",
        help=(
            "Parent directory containing iter_N/summary.json for human-eval source. "
            "Required when human_eval is included in --pvr-sources or --brr-sources."
        ),
    )
    parser.add_argument(
        "--cross-eval-subdir",
        default="cross_eval",
        metavar="SUBDIR",
        help="Subdirectory within selfplay_dir for cross-eval results (default: cross_eval).",
    )
    parser.add_argument(
        "--baseline-cross-eval-subdir",
        default="cross_eval_baseline",
        metavar="SUBDIR",
        help=(
            "Subdirectory within selfplay_dir for the manual-prompt baseline cross-eval "
            "(default: cross_eval_baseline). Consumed by plot_baseline_vs_trained."
        ),
    )
    parser.add_argument(
        "--human-eval-json",
        default="data/human_eval/comparison.json",
        metavar="PATH",
        help=(
            "Path to human-eval comparison.json (default: data/human_eval/comparison.json). "
            "Pass empty string to skip the human_eval comparison plot."
        ),
    )

    # Semantic diversity options
    parser.add_argument(
        "--human-queries",
        default="new_jailbreaks.txt",
        metavar="PATH",
        help="Path to human jailbreak queries file (default: new_jailbreaks.txt).",
    )
    parser.add_argument(
        "--benign-queries",
        default="data/benign_pool_stats.json",
        metavar="PATH",
        help=(
            "Path to benign queries pool (default: data/benign_pool_stats.json). "
            "Pass empty string to disable benign overlay on semantic diversity plots."
        ),
    )
    parser.add_argument(
        "--query-mode",
        default="training_time",
        choices=["training_time", "evaluation_time", "final_episode"],
        help="Query source for semantic diversity plot (default: training_time).",
    )
    parser.add_argument(
        "--tail-pct",
        type=float,
        default=0.25,
        help="Fraction of final training episodes for final_episode mode (default: 0.25).",
    )

    # LoRA options
    parser.add_argument(
        "--lora-base-model",
        default=None,
        metavar="MODEL_ID",
        help="Override base model for LoRA diversity plot.",
    )

    # Skip flags
    parser.add_argument(
        "--skip",
        default="",
        metavar="NAME[,NAME]",
        help=(
            "Comma-separated list of plots to skip. "
            "Valid: pvr, pvr_conv, work_factor, brr, diversity, running_time, "
            "training_dynamics, compute_efficiency, security_utility_pareto, heatmaps, "
            "training_diagonal, pvr_vs_normalized_eis, blue_convergence, attempts_cdf, "
            "coverage_yield, diagonal_convergence, lora, sql_error_heatmap, "
            "baseline_compare, honeypot_per_iter, per_target_defense, attack_evolution, "
            "top_target_mechanism, human_eval_compare."
        ),
    )

    args = parser.parse_args()

    # Resolve results with auto-labels
    results = parse_results_arg(args.results)
    print("Results:")
    for label, selfplay_dir in results:
        print(f"  {label!r:30s}  ←  {selfplay_dir}")

    # Parse sources and skip
    pvr_sources = tuple(s.strip() for s in args.pvr_sources.split(",") if s.strip())
    brr_sources = tuple(s.strip() for s in args.brr_sources.split(",") if s.strip())
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    produced = run_all(
        results=results,
        out_dir=args.out_dir,
        pvr_sources=pvr_sources,
        brr_sources=brr_sources,
        human_eval_parent=args.human_eval_parent,
        cross_eval_subdir=args.cross_eval_subdir,
        human_queries=args.human_queries,
        benign_queries=(args.benign_queries or None),
        query_mode=args.query_mode,
        tail_pct=args.tail_pct,
        lora_base_model=args.lora_base_model,
        skip=skip,
        baseline_cross_eval_subdir=args.baseline_cross_eval_subdir,
        human_eval_json=(args.human_eval_json or None),
    )

    print(f"\n{'=' * 60}")
    print(f"Generated {len(produced)} figure(s):")
    print(f"{'=' * 60}")
    for description, path in produced:
        print(f"\n[{description}]")
        print(f"  → {path}")
    print()


if __name__ == "__main__":
    main()
