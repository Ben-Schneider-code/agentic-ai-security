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
    figures/cross_eval_attempted_vs_successful_breach.png
    figures/quick_attempted_vs_successful_breach.png        (if cross_eval_quick/ exists)
    figures/diagonal_eval_attempted_vs_successful_breach.png (if diagonal_eval/ exists)
    figures/security_utility_pareto.png
    figures/pvr_conv_trajectory.png
    figures/pvr_turn_trajectory.png
    figures/brr_trajectory.png
    figures/lora_drift.png
    figures/lora_delta.png
    figures/lora_cosine.png
    figures/sql_error_heatmap.png
    figures/training_plots/red_reward_curve.png
    figures/training_plots/red_reward_vs_eis_scatter.png
    figures/training_plots/red_cumulative_reward_vs_eis.png
    figures/training_plots/red_outcome_composition.png
    figures/training_plots/red_fluency.png
    figures/training_plots/red_honeypot_discovery.png
    figures/training_plots/blue_prf1.png
    figures/training_plots/blue_outcome_rates.png
    figures/training_plots/selfplay_arms_race.png
    figures/training_plots/selfplay_dominance.png
    figures/training_plots/optimization_curves.png
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def _slug(label: str) -> str:
    """Filename-safe slug for a run label."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
    return s or "run"


def _per_path(base: Path, label: str, multi: bool) -> Path:
    """If multi-result, insert `__<slug>` before extension; else return base."""
    if not multi:
        return base
    return base.with_name(f"{base.stem}__{_slug(label)}{base.suffix}")


def _per_dir(base: Path, label: str, multi: bool) -> Path:
    """If multi-result, return base/<slug>; else return base. Creates the dir."""
    p = base / _slug(label) if multi else base
    p.mkdir(parents=True, exist_ok=True)
    return p


def _git_info(cwd: Path) -> tuple[str, bool | None]:
    """Best-effort git SHA + dirty flag. Returns ('unknown', None) on any failure."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"
    try:
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        ).stdout
        dirty: bool | None = bool(dirty_out.strip())
    except Exception:
        dirty = None
    return sha, dirty


def _export_honeypot_type(results: list[tuple[str, str]]) -> str | None:
    """Read honeypot_type from each result's summary.json and export it.

    Downstream metric refresh (compute_pairing_metrics → redteam_sql_env import)
    raises RuntimeError when HONEYPOT_TYPE is unset, which the orchestrator's
    per-plot try/except would silently swallow — dropping the trajectory and
    heatmap families. Setting it here keeps those plots renderable from a plain
    `python plot_paper_figures.py` invocation. Fail-fast on disagreement so a
    mixed-arm comparison can't silently fix to whichever arm appeared first.
    """
    arms: dict[str, str] = {}
    for label, selfplay_dir in results:
        summary = Path(selfplay_dir) / "summary.json"
        if not summary.is_file():
            continue
        try:
            arm = json.loads(summary.read_text()).get("honeypot_type")
        except (json.JSONDecodeError, OSError):
            continue
        if arm:
            arms[label] = arm
    if not arms:
        return None
    distinct = set(arms.values())
    if len(distinct) > 1:
        raise RuntimeError(
            "Mixed honeypot_type across --results: "
            + ", ".join(f"{lbl}={arm}" for lbl, arm in arms.items())
            + ". Refusing to set HONEYPOT_TYPE to a single arm."
        )
    arm = next(iter(distinct))
    existing = os.environ.get("HONEYPOT_TYPE")
    if existing and existing != arm:
        raise RuntimeError(
            f"HONEYPOT_TYPE={existing!r} is preset but results summaries say {arm!r}."
        )
    os.environ["HONEYPOT_TYPE"] = arm
    return arm


def _build_run_meta(run_all_kwargs: dict, cli_args: dict | None) -> dict:
    """Snapshot run-level metadata reused across every sidecar in this invocation."""
    sha, dirty = _git_info(_REPO_ROOT)
    return {
        "argv": list(sys.argv),
        "cli_args": cli_args or {},
        "run_all_kwargs": run_all_kwargs,
        "git_sha": sha,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


# Support both package import and direct execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from plotting._data import parse_results_arg, write_sidecar
    from plotting.plot_pvr import (
        plot_pvr,
        DESCRIPTION as DESC_PVR,
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
        DESCRIPTION_PVR_CONV_TRAJ as DESC_PVR_CONV_TRAJ,
        DESCRIPTION_PVR_TURN_TRAJ as DESC_PVR_TURN_TRAJ,
        DESCRIPTION_BRR_TRAJ      as DESC_BRR_TRAJ,
    )
except ImportError:
    from plotting._data import parse_results_arg, write_sidecar
    from plotting.plot_pvr import (
        plot_pvr,
        DESCRIPTION as DESC_PVR,
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
        DESCRIPTION_PVR_CONV_TRAJ as DESC_PVR_CONV_TRAJ,
        DESCRIPTION_PVR_TURN_TRAJ as DESC_PVR_TURN_TRAJ,
        DESCRIPTION_BRR_TRAJ      as DESC_BRR_TRAJ,
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
    show_ci: bool = True,
    cli_args: dict | None = None,
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
                            Valid names: pvr, work_factor, brr, diversity,
                            running_time, training_dynamics, compute_efficiency,
                            security_utility_pareto, heatmaps, training_diagonal,
                            pvr_vs_normalized_eis, blue_convergence, attempts_cdf,
                            coverage_yield, diagonal_convergence, lora,
                            sql_error_heatmap, training_curves.

    Returns:
        [(description, output_path), ...] for each plot produced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = skip or set(["diversity"]) # TODO: SKipping diversity for now
    produced: list[tuple[str, Path]] = []

    # Export HONEYPOT_TYPE from summary.json before any plot loader triggers
    # compute_pairing_metrics (which imports redteam_sql_env). Without this,
    # the trajectory / heatmap / coverage_yield families are silently dropped
    # by the per-plot try/except below.
    honeypot_arm = _export_honeypot_type(results)

    # Run-level metadata reused by every sidecar write.
    run_meta = _build_run_meta(
        run_all_kwargs={
            "results": [{"label": lbl, "selfplay_dir": d} for lbl, d in results],
            "out_dir": str(out_dir),
            "pvr_sources": list(pvr_sources),
            "brr_sources": list(brr_sources),
            "human_eval_parent": human_eval_parent,
            "cross_eval_subdir": cross_eval_subdir,
            "human_queries": human_queries,
            "benign_queries": benign_queries,
            "query_mode": query_mode,
            "tail_pct": tail_pct,
            "lora_base_model": lora_base_model,
            "skip": sorted(skip),
            "training_dynamics_smooth": training_dynamics_smooth,
            "baseline_cross_eval_subdir": baseline_cross_eval_subdir,
            "human_eval_json": human_eval_json,
            "show_ci": show_ci,
            "honeypot_type": honeypot_arm,
        },
        cli_args=cli_args,
    )

    # 1. PVR_turn
    # if "pvr" not in skip:
    #     try:
    #         path = plot_pvr(
    #             results,
    #             out_dir / "pvr.png",
    #             sources=pvr_sources,
    #             human_eval_parent=human_eval_parent,
    #             cross_eval_subdir=cross_eval_subdir,
    #             show_ci=show_ci,
    #         )
    #         produced.append((DESC_PVR, path))
    #         write_sidecar(path, DESC_PVR, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] pvr skipped: {e}", file=sys.stderr)

    # 3. Work Factor (WF = 1/PVR_turn)
    # if "work_factor" not in skip:
    #     try:
    #         path = plot_work_factor(
    #             results,
    #             out_dir / "work_factor.png",
    #             sources=pvr_sources,
    #             human_eval_parent=human_eval_parent,
    #             cross_eval_subdir=cross_eval_subdir,
    #             show_ci=show_ci,
    #         )
    #         produced.append((DESC_WF, path))
    #         write_sidecar(path, DESC_WF, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] work_factor skipped: {e}", file=sys.stderr)

    # 4. BRR — one PNG per source
    # if "brr" not in skip:
    #     try:
    #         brr_paths = plot_brr(
    #             results,
    #             out_dir=out_dir,
    #             filename_prefix="brr",
    #             sources=brr_sources,
    #             human_eval_parent=human_eval_parent,
    #             cross_eval_subdir=cross_eval_subdir,
    #             show_ci=show_ci,
    #         )
    #         for src, path in brr_paths.items():
    #             produced.append((DESC_BRR, path))
    #             write_sidecar(path, DESC_BRR, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] brr skipped: {e}", file=sys.stderr)

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

        try:
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
                write_sidecar(
                    path, DESC_DIV, results, sem_metrics,
                    plot_kwargs={
                        "human_queries_path": human_queries,
                        "query_mode": query_mode,
                        "tail_pct": tail_pct,
                        "embedder": embedder,
                        "embedder_id": id,
                        "cross_eval_subdir": cross_eval_subdir,
                        "benign_queries_path": benign_queries,
                    },
                    run_meta=run_meta,
                )
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] diversity skipped: {e}", file=sys.stderr)

    # 5. Running time (EIS)
    if "running_time" not in skip:
        try:
            path = plot_running_time(results, out_dir / "running_time.png")
            produced.append((DESC_RT, path))
            write_sidecar(path, DESC_RT, results, plot_kwargs={}, run_meta=run_meta)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] running_time skipped: {e}", file=sys.stderr)

    # 6. Training dynamics (reward-vs-EIS learning curves, process visualization)
    # if "training_dynamics" not in skip:
    #     multi = len(results) > 1
    #     for label, sd in results:
    #         try:
    #             path = plot_training_dynamics(
    #                 [(label, sd)],
    #                 _per_path(out_dir / "training_dynamics.png", label, multi),
    #                 smooth_window=training_dynamics_smooth,
    #             )
    #             produced.append((DESC_DYN, path))
    #             write_sidecar(path, DESC_DYN, [(label, sd)])
    #         except Exception as e:  # pragma: no cover
    #             print(f"[plot_paper_figures] training_dynamics [{label}] skipped: {e}", file=sys.stderr)

    # 7. Compute efficiency (PVR_conv and BRR vs cumulative EIS)
    # if "compute_efficiency" not in skip:
    #     try:
    #         path = plot_compute_efficiency(
    #             results, out_dir / "compute_efficiency.png", show_ci=show_ci,
    #         )
    #         produced.append((DESC_EFF, path))
    #         write_sidecar(path, DESC_EFF, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] compute_efficiency skipped: {e}", file=sys.stderr)

    # 7b. Security–Utility Pareto frontier across checkpoints
    # if "security_utility_pareto" not in skip:
    #     try:
    #         path = plot_security_utility_pareto(
    #             results,
    #             out_dir / "security_utility_pareto.png",
    #             cross_eval_subdir=cross_eval_subdir,
    #             show_ci=show_ci,
    #         )
    #         produced.append((DESC_PARETO, path))
    #         write_sidecar(path, DESC_PARETO, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] security_utility_pareto skipped: {e}", file=sys.stderr)

    # 8. Cross-eval heatmaps — PVR_conv + PVR_turn for all available subdirs.
    #    Each call is a no-op (prints a warning) if the subdir doesn't exist.
    if "heatmaps" not in skip:
        try:
            for subdir, metric, fname in HEATMAP_JOBS:
                path = plot_heatmap(results, out_dir / fname, subdir=subdir, metric=metric)
                if not Path(path).exists():
                    continue
                desc = _heatmap_desc(subdir, metric)
                produced.append((desc, path))
                write_sidecar(
                    path, desc, results,
                    plot_kwargs={"subdir": subdir, "metric": metric, "fname": fname},
                    run_meta=run_meta,
                )
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] heatmaps skipped: {e}", file=sys.stderr)

    # 9. Training-log diagonal: PVR_conv from per-iteration blueteam logs.
    # if "training_diagonal" not in skip:
    #     try:
    #         path = plot_training_diagonal(results, out_dir / "diagonal_training.png")
    #         produced.append((DESC_TRAINING_DIAG, path))
    #         write_sidecar(path, DESC_TRAINING_DIAG, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] training_diagonal skipped: {e}", file=sys.stderr)

    # 10. ΔPVRconv vs normalized EIS — direct cross-team efficiency comparison
    # if "pvr_vs_normalized_eis" not in skip:
    #     try:
    #         path = plot_pvr_vs_normalized_eis(
    #             results, out_dir / "pvr_vs_normalized_eis.png", show_ci=show_ci,
    #         )
    #         produced.append((DESC_NORM_EIS, path))
    #         write_sidecar(path, DESC_NORM_EIS, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] pvr_vs_normalized_eis skipped: {e}", file=sys.stderr)

    # 11. Blue team convergence — plateau analysis and marginal efficiency
    # if "blue_convergence" not in skip:
    #     try:
    #         path = plot_blue_convergence(
    #             results, out_dir / "blue_convergence.png", show_ci=show_ci,
    #         )
    #         produced.append((DESC_BLUE_CONV, path))
    #         write_sidecar(path, DESC_BLUE_CONV, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] blue_convergence skipped: {e}", file=sys.stderr)

    # 12. Decomposed-PVR: attempts-to-compromise CDF
    if "attempts_cdf" not in skip:
        multi = len(results) > 1
        for subdir_name, fname in ATTEMPTS_CDF_JOBS:
            for label, sd in results:
                try:
                    path = plot_attempts_cdf(
                        [(label, sd)],
                        _per_path(out_dir / fname, label, multi),
                        subdir=subdir_name,
                    )
                    if not Path(path).exists():
                        continue
                    produced.append((DESC_ATTEMPTS, path))
                    write_sidecar(
                        path, DESC_ATTEMPTS, [(label, sd)],
                        plot_kwargs={"subdir": subdir_name, "fname": fname},
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] attempts_cdf [{label}/{subdir_name}] skipped: {e}", file=sys.stderr)

    # 13. Decomposed-PVR: honeypot coverage vs. yield
    if "coverage_yield" not in skip:
        multi = len(results) > 1
        for subdir_name, fname in COVERAGE_YIELD_JOBS:
            for label, sd in results:
                try:
                    path = plot_coverage_yield(
                        [(label, sd)],
                        _per_path(out_dir / fname, label, multi),
                        subdir=subdir_name,
                        show_ci=show_ci,
                    )
                    if not Path(path).exists():
                        continue
                    produced.append((DESC_COV, path))
                    write_sidecar(
                        path, DESC_COV, [(label, sd)],
                        plot_kwargs={
                            "subdir": subdir_name,
                            "fname": fname,
                            "show_ci": show_ci,
                        },
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] coverage_yield [{label}/{subdir_name}] skipped: {e}", file=sys.stderr)

    # 14. Diagonal trajectory: PVR_conv, PVR_turn, BRR for adjacent pairings
    if "diagonal_convergence" not in skip:
        try:
            paths, dc_metrics = plot_diagonal_convergence(
                results,
                out_dir,
                eval_subdir=cross_eval_subdir,
                show_ci=show_ci,
            )
            desc_by_key = {
                "pvr_conv": DESC_PVR_CONV_TRAJ,
                "pvr_turn": DESC_PVR_TURN_TRAJ,
                "brr":      DESC_BRR_TRAJ,
            }
            dc_plot_kwargs = {
                "eval_subdir": cross_eval_subdir,
                "show_ci": show_ci,
            }
            for key, p in paths.items():
                produced.append((desc_by_key[key], p))
                write_sidecar(
                    p, desc_by_key[key], results, dc_metrics,
                    plot_kwargs={**dc_plot_kwargs, "trajectory_key": key},
                    run_meta=run_meta,
                )
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] diagonal_convergence skipped: {e}", file=sys.stderr)

    # 15. LoRA diversity (three output files)
    if "lora" not in skip:
        try:
            drift, delta, cosine = plot_lora_diversity(
                results,
                out_dir,
                base_model=lora_base_model,
            )
            for p in (drift, delta, cosine):
                produced.append((DESC_LORA, p))
                write_sidecar(
                    p, DESC_LORA, results,
                    plot_kwargs={"base_model": lora_base_model},
                    run_meta=run_meta,
                )
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] lora skipped: {e}", file=sys.stderr)

    # 16. Pillar 1 headline: utility by query style (replaces brr.png as headline)
    # if "utility_by_style" not in skip:
    #     try:
    #         from plotting.plot_utility_by_style import (
    #             plot_utility_by_style,
    #             DESCRIPTION as DESC_UTIL_STYLE,
    #         )
    #         multi = len(results) > 1
    #         for label, sd in results:
    #             try:
    #                 path = plot_utility_by_style(
    #                     [(label, sd)],
    #                     _per_path(out_dir / "utility_by_style.png", label, multi),
    #                     show_ci=show_ci,
    #                 )
    #                 produced.append((DESC_UTIL_STYLE, path))
    #                 write_sidecar(path, DESC_UTIL_STYLE, [(label, sd)])
    #             except Exception as e:  # pragma: no cover
    #                 print(f"[plot_paper_figures] utility_by_style [{label}] skipped: {e}", file=sys.stderr)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] utility_by_style skipped: {e}", file=sys.stderr)

    # 17. Pillar 1 supporting: per-style refusal (training-time denial) — appendix
    # if "per_style_refusal" not in skip:
    #     try:
    #         from plotting.plot_per_style_refusal import (
    #             plot_per_style_refusal,
    #             DESCRIPTION as DESC_PER_STYLE,
    #         )
    #         multi = len(results) > 1
    #         for label, sd in results:
    #             try:
    #                 path = plot_per_style_refusal(
    #                     [(label, sd)],
    #                     _per_path(out_dir / "per_style_refusal.png", label, multi),
    #                     show_ci=show_ci,
    #                 )
    #                 produced.append((DESC_PER_STYLE, path))
    #                 write_sidecar(path, DESC_PER_STYLE, [(label, sd)])
    #             except Exception as e:  # pragma: no cover
    #                 print(f"[plot_paper_figures] per_style_refusal [{label}] skipped: {e}", file=sys.stderr)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] per_style_refusal skipped: {e}", file=sys.stderr)

    # 18. Pillar 4: generalization (column-0 / row-0 + diagonal reference)
    if "generalization" not in skip:
        try:
            from plotting.plot_generalization import (
                plot_generalization,
                DESCRIPTION as DESC_GEN,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_generalization(
                        [(label, sd)],
                        _per_path(out_dir / "generalization.png", label, multi),
                        subdir=cross_eval_subdir,
                        show_ci=show_ci,
                    )
                    produced.append((DESC_GEN, path))
                    write_sidecar(
                        path, DESC_GEN, [(label, sd)],
                        plot_kwargs={"subdir": cross_eval_subdir, "show_ci": show_ci},
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] generalization [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] generalization skipped: {e}", file=sys.stderr)

    # 19. Pillar 3: per-honeypot difficulty + tier classification (canonical narrative)
    if "honeypot_difficulty" not in skip:
        try:
            from plotting.plot_honeypot_difficulty import (
                plot_honeypot_difficulty,
                DESCRIPTION as DESC_HP_DIFF,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_honeypot_difficulty(
                        [(label, sd)],
                        _per_path(out_dir / "honeypot_difficulty.png", label, multi),
                    )
                    produced.append((DESC_HP_DIFF, path))
                    write_sidecar(
                        path, DESC_HP_DIFF, [(label, sd)],
                        plot_kwargs={}, run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] honeypot_difficulty [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] honeypot_difficulty skipped: {e}", file=sys.stderr)

    # 20. Pillar 5: honeypot saturation (training-time vs eval-time)
    if "honeypot_saturation" not in skip:
        try:
            from plotting.plot_honeypot_saturation import (
                plot_honeypot_saturation,
                DESCRIPTION as DESC_HP_SAT,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path, sat_metrics = plot_honeypot_saturation(
                        [(label, sd)],
                        _per_path(out_dir / "honeypot_saturation.png", label, multi),
                        show_ci=show_ci,
                    )
                    produced.append((DESC_HP_SAT, path))
                    write_sidecar(
                        path, DESC_HP_SAT, [(label, sd)], metrics=sat_metrics,
                        plot_kwargs={"show_ci": show_ci}, run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] honeypot_saturation [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] honeypot_saturation skipped: {e}", file=sys.stderr)

    # 21. Pillar 3 appendix: per-honeypot training-vs-eval coverage
    # if "honeypot_training_vs_eval" not in skip:
    #     try:
    #         import subprocess
    #         from pathlib import Path as _P
    #         script = _P(__file__).resolve().parent.parent / "util" / "honeypot_training_vs_eval.py"
    #         multi = len(results) > 1
    #         for label, selfplay_dir in results:
    #             out_png = _per_path(out_dir / "honeypot_training_vs_eval.png", label, multi)
    #             cmd = [
    #                 sys.executable, str(script),
    #                 "--results-dir", selfplay_dir,
    #                 "--out", str(out_png),
    #             ]
    #             rc = subprocess.run(cmd).returncode
    #             if rc == 0:
    #                 desc = (
    #                     "Per-honeypot training-time vs eval-time hits, grouped by tier. "
    #                     "Stars mark training-only honeypots and never-breached-at-eval. "
    #                     "Supports Pillar 3 attacker-coverage-gap framing."
    #                 )
    #                 produced.append((desc, out_png))
    #                 write_sidecar(out_png, desc, [(label, selfplay_dir)])
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] honeypot_training_vs_eval skipped: {e}", file=sys.stderr)

    # 22. Pillar 2 fourth signal: iter-6 novelty recovery decomposed by tier
    # if "iter6_novelty" not in skip:
    #     try:
    #         import subprocess
    #         from pathlib import Path as _P
    #         script = _P(__file__).resolve().parent.parent / "util" / "iter6_novelty_breakdown.py"
    #         multi = len(results) > 1
    #         for label, selfplay_dir in results:
    #             out_png = _per_path(out_dir / "iter6_novelty_recovery.png", label, multi)
    #             cmd = [
    #                 sys.executable, str(script),
    #                 "--results-dir", selfplay_dir,
    #                 "--out", str(out_png),
    #                 "--iters", "1", "2", "3", "4", "5", "6", "7",
    #             ]
    #             rc = subprocess.run(cmd).returncode
    #             if rc == 0:
    #                 desc = (
    #                     "Iter-6 novelty recovery decomposed by tier — supports "
    #                     "Pillar 2 'broad-tier transient defender regression' framing."
    #                 )
    #                 produced.append((desc, out_png))
    #                 write_sidecar(out_png, desc, [(label, selfplay_dir)])
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] iter6_novelty skipped: {e}", file=sys.stderr)

    # 23a. Held-out per-style benign refusal (Pillar 1 headline).
    if "held_out_per_style_refusal" not in skip:
        try:
            from plotting.plot_held_out_per_style_refusal import (
                plot_held_out_per_style_refusal,
                DESCRIPTION as DESC_HELD_OUT,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_held_out_per_style_refusal(
                        [(label, sd)],
                        cross_eval_subdir=cross_eval_subdir,
                        out_dir=str(_per_dir(out_dir, label, multi)),
                        show_ci=show_ci,
                    )
                    produced.append((DESC_HELD_OUT, path))
                    write_sidecar(
                        Path(path), DESC_HELD_OUT, [(label, sd)],
                        plot_kwargs={
                            "cross_eval_subdir": cross_eval_subdir,
                            "show_ci": show_ci,
                        },
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] held_out_per_style_refusal [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] held_out_per_style_refusal skipped: {e}", file=sys.stderr)

    # 23b. Rank-invariance figure (Pillar 4: diagonal vs off-diagonal PVR_conv).
    if "cross_eval_rank_invariance" not in skip:
        try:
            from plotting.cross_eval_rank_invariance import (
                plot_rank_invariance,
                DESCRIPTION as DESC_RANK,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_rank_invariance(
                        [(label, sd)],
                        cross_eval_subdir=cross_eval_subdir,
                        out_dir=str(_per_dir(out_dir, label, multi)),
                    )
                    produced.append((DESC_RANK, path))
                    write_sidecar(
                        Path(path), DESC_RANK, [(label, sd)],
                        plot_kwargs={"cross_eval_subdir": cross_eval_subdir},
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] cross_eval_rank_invariance [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] cross_eval_rank_invariance skipped: {e}", file=sys.stderr)

    # 23c. Defender concentration figure (Pillar 3: block rate stability + breach tiers).
    if "defender_concentration" not in skip:
        try:
            from plotting.plot_defender_concentration import (
                plot_defender_concentration,
                DESCRIPTION as DESC_DEF,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_defender_concentration(
                        [(label, sd)],
                        out_dir=str(_per_dir(out_dir, label, multi)),
                    )
                    produced.append((DESC_DEF, path))
                    write_sidecar(
                        Path(path), DESC_DEF, [(label, sd)],
                        plot_kwargs={}, run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] defender_concentration [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] defender_concentration skipped: {e}", file=sys.stderr)

    # 23d. PVR asymptote (diagonal convergence plot) — previously not in orchestrator.
    if "pvr_asymptote" not in skip:
        try:
            from plotting.plot_pvr_asymptote import (
                plot_pvr_asymptote,
                DESCRIPTION as DESC_ASYM,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path, asym_metrics = plot_pvr_asymptote(
                        [(label, sd)],
                        _per_path(out_dir / "pvr_asymptote.png", label, multi),
                        show_ci=show_ci,
                    )
                    produced.append((DESC_ASYM, path))
                    write_sidecar(
                        path, DESC_ASYM, [(label, sd)], asym_metrics,
                        plot_kwargs={"show_ci": show_ci},
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] pvr_asymptote [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] pvr_asymptote skipped: {e}", file=sys.stderr)

    # 23e. LoRA orthogonality null check — previously not in orchestrator.
    # if "lora_orthogonality" not in skip:
    #     try:
    #         from plotting.plot_lora_orthogonality import (
    #             plot_lora_orthogonality,
    #             DESCRIPTION as DESC_ORTH,
    #         )
    #         multi = len(results) > 1
    #         for label, sd in results:
    #             try:
    #                 path = plot_lora_orthogonality(
    #                     [(label, sd)],
    #                     _per_path(out_dir / "lora_orthogonality.png", label, multi),
    #                 )
    #                 produced.append((DESC_ORTH, path))
    #                 write_sidecar(path, DESC_ORTH, [(label, sd)])
    #             except Exception as e:  # pragma: no cover
    #                 print(f"[plot_paper_figures] lora_orthogonality [{label}] skipped: {e}", file=sys.stderr)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] lora_orthogonality skipped: {e}", file=sys.stderr)

    # 23f. Red-team termination: training steps + novel honeypot discovery.
    # if "red_termination" not in skip:
    #     try:
    #         from util.plot_red_termination import (
    #             plot_red_termination,
    #         )
    #         DESC_RED_TERM = "Red saturation: training steps and novel discovery per iteration"
    #         multi = len(results) > 1
    #         for label, sd in results:
    #             try:
    #                 path = plot_red_termination(
    #                     [(label, sd)],
    #                     out_dir=str(_per_dir(out_dir, label, multi)),
    #                 )
    #                 produced.append((DESC_RED_TERM, path))
    #                 # NOTE: plot_red_termination writes its own per-iter sidecar; do not overwrite.
    #             except Exception as e:  # pragma: no cover
    #                 print(f"[plot_paper_figures] red_termination [{label}] skipped: {e}", file=sys.stderr)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] red_termination skipped: {e}", file=sys.stderr)

    # 23g. Per-tier PVR decomposition: stack of pii_dominant / harvestable / rare.
    if "tier_decomposition" not in skip:
        try:
            from plotting.plot_tier_decomposition import plot_tier_decomposition
            DESC_TIER = "Per-tier PVR_conv decomposition with 99% Wilson CIs"
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_tier_decomposition(
                        [(label, sd)],
                        out_path=_per_path(out_dir / "tier_pvr_decomposition.png", label, multi),
                        show_ci=show_ci,
                    )
                    produced.append((DESC_TIER, path))
                    # NOTE: plot_tier_decomposition writes its own per-iter sidecar; do not overwrite.
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] tier_decomposition [{label}] skipped: {e}", file=sys.stderr)
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
                show_ci=show_ci,
            )
            produced.append((DESC_BASELINE, path))
            write_sidecar(
                Path(path), DESC_BASELINE, results,
                plot_kwargs={
                    "cross_eval_subdir": cross_eval_subdir,
                    "baseline_subdir": baseline_cross_eval_subdir,
                    "show_ci": show_ci,
                },
                run_meta=run_meta,
            )
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] baseline_compare skipped: {e}", file=sys.stderr)

    # 23i. Per-honeypot × per-iter breach heatmap (composition stability across iters).
    if "honeypot_per_iter" not in skip:
        try:
            from plotting.plot_honeypot_per_iter_heatmap import (
                plot_honeypot_per_iter_heatmap,
                DESCRIPTION as DESC_HP_HM,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_honeypot_per_iter_heatmap(
                        [(label, sd)],
                        out_dir=str(_per_dir(out_dir, label, multi)),
                    )
                    produced.append((DESC_HP_HM, path))
                    write_sidecar(
                        Path(path), DESC_HP_HM, [(label, sd)],
                        plot_kwargs={}, run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] honeypot_per_iter [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] honeypot_per_iter skipped: {e}", file=sys.stderr)

    # 23j. Per-target defender response (refusal vs breach vs accepted-clean).
    if "per_target_defense" not in skip:
        try:
            from plotting.plot_per_target_defender_response import (
                plot_per_target_defender_response,
                DESCRIPTION as DESC_PER_TARGET,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_per_target_defender_response(
                        [(label, sd)],
                        out_dir=str(_per_dir(out_dir, label, multi)),
                        cross_eval_subdir=cross_eval_subdir,
                    )
                    produced.append((DESC_PER_TARGET, path))
                    write_sidecar(
                        Path(path), DESC_PER_TARGET, [(label, sd)],
                        plot_kwargs={"cross_eval_subdir": cross_eval_subdir},
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] per_target_defense [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] per_target_defense skipped: {e}", file=sys.stderr)

    # 23k. Attack template + SQL pattern evolution across iters.
    if "attack_evolution" not in skip:
        try:
            from plotting.plot_attack_evolution import (
                plot_attack_evolution,
                DESCRIPTION as DESC_ATK_EVO,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    tpl, pat = plot_attack_evolution(
                        [(label, sd)],
                        out_dir=str(_per_dir(out_dir, label, multi)),
                    )
                    produced.append((DESC_ATK_EVO + " (template similarity)", tpl))
                    produced.append((DESC_ATK_EVO + " (SQL pattern)", pat))
                    write_sidecar(
                        Path(tpl), DESC_ATK_EVO + " (template similarity)",
                        [(label, sd)],
                        plot_kwargs={"variant": "template_similarity"},
                        run_meta=run_meta,
                    )
                    write_sidecar(
                        Path(pat), DESC_ATK_EVO + " (SQL pattern)",
                        [(label, sd)],
                        plot_kwargs={"variant": "sql_pattern"},
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] attack_evolution [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] attack_evolution skipped: {e}", file=sys.stderr)

    # 23l. Top-target attack mechanism (phone / email / passwordhash exemplars).
    if "top_target_mechanism" not in skip:
        try:
            from plotting.plot_top_target_mechanism import (
                plot_top_target_mechanism,
                DESCRIPTION as DESC_TT_MECH,
            )
            multi = len(results) > 1
            for label, sd in results:
                try:
                    path = plot_top_target_mechanism(
                        [(label, sd)],
                        out_dir=str(_per_dir(out_dir, label, multi)),
                        cross_eval_subdir=cross_eval_subdir,
                    )
                    produced.append((DESC_TT_MECH, path))
                    write_sidecar(
                        Path(path), DESC_TT_MECH, [(label, sd)],
                        plot_kwargs={"cross_eval_subdir": cross_eval_subdir},
                        run_meta=run_meta,
                    )
                except Exception as e:  # pragma: no cover
                    print(f"[plot_paper_figures] top_target_mechanism [{label}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] top_target_mechanism skipped: {e}", file=sys.stderr)

    # 23m. Human-eval comparison (Unprotected / Manual / RL-iter-1 on n=320 each).
    # if "human_eval_compare" not in skip:
    #     try:
    #         human_eval_path = Path(human_eval_json) if human_eval_json else None
    #         if human_eval_path and human_eval_path.exists():
    #             from plotting.plot_human_eval_comparison import (
    #                 plot_human_eval_comparison,
    #                 DESCRIPTION as DESC_HUMAN,
    #             )
    #             path = plot_human_eval_comparison(
    #                 results, human_eval_json=str(human_eval_path), out_dir=str(out_dir),
    #                 show_ci=show_ci,
    #             )
    #             produced.append((DESC_HUMAN, path))
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] human_eval_compare skipped: {e}", file=sys.stderr)

    # 23. SQL-error rate heatmap — diagnostic: fraction of turns with neither
    #     refusal nor valid parseable SQL, per (red_iter × blue_iter) pairing.
    # if "sql_error_heatmap" not in skip:
    #     try:
    #         from plotting.cross_eval_statistics import (
    #             plot_sql_error_heatmap,
    #             DESCRIPTION as DESC_SQL_ERR,
    #         )
    #         path = plot_sql_error_heatmap(
    #             results,
    #             out_dir / "sql_error_heatmap.png",
    #             cross_eval_subdir=cross_eval_subdir,
    #         )
    #         produced.append((DESC_SQL_ERR, path))
    #         write_sidecar(path, DESC_SQL_ERR, results)
    #     except Exception as e:  # pragma: no cover
    #         print(f"[plot_paper_figures] sql_error_heatmap skipped: {e}", file=sys.stderr)

    # 24. Training-time learning curves — paper-style replications of the
    #     util/plot_results.py dashboards, written to a training_plots/ subdir
    #     per run. Reads only json/jsonl (no tensorboard / env / HONEYPOT_TYPE).
    if "training_curves" not in skip:
        try:
            from plotting.plot_training_curves import (
                plot_red_reward_curve, plot_red_reward_scatter,
                plot_red_cumulative_reward, plot_red_outcome_composition,
                plot_red_fluency, plot_red_honeypot_discovery,
                plot_blue_prf1, plot_blue_outcome_rates,
                plot_selfplay_arms_race, plot_selfplay_dominance,
                plot_optimization_curves,
                plot_rolling_pvr_train, plot_rolling_coverage_train, plot_edsr,
                DESC_RED_REWARD, DESC_RED_REWARD_SCATTER, DESC_RED_CUMULATIVE,
                DESC_RED_COMPOSITION, DESC_RED_FLUENCY,
                DESC_RED_HONEYPOT, DESC_BLUE_PRF1, DESC_BLUE_RATES,
                DESC_ARMS_RACE, DESC_DOMINANCE, DESC_OPT_CURVES,
                DESC_ROLLING_PVR_TRAIN, DESC_ROLLING_COVERAGE_TRAIN, DESC_EDSR,
            )
            multi = len(results) > 1
            tc_jobs = [
                (plot_red_reward_curve,        "red_reward_curve.png",        DESC_RED_REWARD,      {}),
                (plot_red_reward_scatter,      "red_reward_vs_eis_scatter.png",   DESC_RED_REWARD_SCATTER, {}),
                (plot_red_cumulative_reward,   "red_cumulative_reward_vs_eis.png", DESC_RED_CUMULATIVE,     {}),
                (plot_red_outcome_composition, "red_outcome_composition.png", DESC_RED_COMPOSITION, {}),
                (plot_red_fluency,             "red_fluency.png",             DESC_RED_FLUENCY,      {}),
                (plot_red_honeypot_discovery,  "red_honeypot_discovery.png",  DESC_RED_HONEYPOT,     {}),
                (plot_blue_prf1,               "blue_prf1.png",               DESC_BLUE_PRF1,        {}),
                (plot_blue_outcome_rates,      "blue_outcome_rates.png",      DESC_BLUE_RATES,       {"show_ci": show_ci}),
                (plot_selfplay_arms_race,      "selfplay_arms_race.png",      DESC_ARMS_RACE,        {"show_ci": show_ci}),
                (plot_selfplay_dominance,      "selfplay_dominance.png",      DESC_DOMINANCE,        {}),
                (plot_optimization_curves,     "optimization_curves.png",     DESC_OPT_CURVES,       {}),
                (plot_rolling_pvr_train,        "rolling_pvr_train.png",        DESC_ROLLING_PVR_TRAIN,      {}),
                (plot_rolling_coverage_train,   "rolling_coverage_train.png",   DESC_ROLLING_COVERAGE_TRAIN, {}),
                (plot_edsr,                     "edsr.png",                     DESC_EDSR,                   {}),
            ]
            for label, sd in results:
                tp_dir = _per_dir(out_dir, label, multi) / "training_plots"
                tp_dir.mkdir(parents=True, exist_ok=True)
                for fn, fname, desc, kwargs in tc_jobs:
                    try:
                        path = fn([(label, sd)], tp_dir / fname, **kwargs)
                        if not Path(path).exists():
                            continue
                        produced.append((desc, path))
                        write_sidecar(
                            Path(path), desc, [(label, sd)],
                            plot_kwargs=kwargs, run_meta=run_meta,
                        )
                    except Exception as e:  # pragma: no cover
                        print(f"[plot_paper_figures] training_curves [{label}/{fname}] skipped: {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"[plot_paper_figures] training_curves skipped: {e}", file=sys.stderr)

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

    # CI rendering toggle
    parser.add_argument(
        "--ci",
        default="true",
        type=lambda s: s.strip().lower(),
        choices=["true", "false"],
        metavar="true|false",
        help=(
            "Render confidence-interval overlays on plots (default: true). "
            "Pass --ci=false to omit error bars / Wilson CI bands / plateau bands. "
            "Sidecar JSON metrics are unaffected and always include CI fields."
        ),
    )

    # Skip flags
    parser.add_argument(
        "--skip",
        default="",
        metavar="NAME[,NAME]",
        help=(
            "Comma-separated list of plots to skip. "
            "Valid: pvr, work_factor, brr, diversity, running_time, "
            "training_dynamics, compute_efficiency, security_utility_pareto, heatmaps, "
            "training_diagonal, pvr_vs_normalized_eis, blue_convergence, attempts_cdf, "
            "coverage_yield, diagonal_convergence, lora, sql_error_heatmap, "
            "baseline_compare, honeypot_per_iter, per_target_defense, attack_evolution, "
            "top_target_mechanism, human_eval_compare, training_curves."
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
        show_ci=(args.ci == "true"),
        cli_args=vars(args),
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
