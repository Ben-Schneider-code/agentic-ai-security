#!/usr/bin/env python3
"""
Standalone benign-TPR sweep for every blue-team LoRA checkpoint in a self-play run.

Usage:
    python util/run_benign_eval.py \
        --selfplay-dir results-20260322-1641-m92p4 \
        --base-model Snowflake/Arctic-Text2SQL-R1-7B \
        --output-dir results-20260322-1641-m92p4/benign_eval \
        --episodes 200 --resume

    # Re-aggregate without re-running episodes:
    python util/run_benign_eval.py \
        --selfplay-dir results-20260322-1641-m92p4 \
        --output-dir results-20260322-1641-m92p4/benign_eval \
        --aggregate-only
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Add project root and MARFT to sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
for _p in [_project_root, os.path.join(_project_root, "MARFT")]:
    if _p not in sys.path:
        sys.path.append(_p)

from util.cross_evaluate import (
    discover_checkpoints,
    evaluate_benign_only,
    load_progress,
    save_progress,
)
from marft.envs.blueteam_sql.blueteam_sql_env import (
    BENIGN_EVAL_QUERIES,
    BlueTeamSQLEnv,
)
from marft.envs.redteam_sql.redteam_sql_env import OfflineLLM
from util.mcp_client import MCPClient


_STYLE_CHOICES = ("plain", "adversarial", "multi_turn")


def _filter_pool_by_style(
    pool: list, keep_styles: set[str]
) -> list:
    """Filter BENIGN_EVAL_QUERIES by the _classify_benign_style tag.

    A string query is classified as plain/adversarial; a list is multi_turn.
    """
    classify = BlueTeamSQLEnv._classify_benign_style  # staticmethod
    return [q for q in pool if classify(q) in keep_styles]


def _print_query_pool_summary(pool: list | None = None):
    pool = pool if pool is not None else BENIGN_EVAL_QUERIES
    single_turn = [q for q in pool if isinstance(q, str)]
    multi_turn  = [q for q in pool if isinstance(q, list)]
    horizon_counts: dict[int, int] = {}
    for seq in multi_turn:
        horizon_counts[len(seq)] = horizon_counts.get(len(seq), 0) + 1

    classify = BlueTeamSQLEnv._classify_benign_style
    by_style: dict[str, int] = {}
    for q in pool:
        by_style[classify(q)] = by_style.get(classify(q), 0) + 1

    print("=" * 60)
    print("Benign-eval pool (filtered to current --style-filter)")
    print(f"  Total queries : {len(pool)}")
    print(f"  Single-turn   : {len(single_turn)}  (horizon=1 each)")
    if multi_turn:
        breakdown = ", ".join(
            f"{count}×{turns}-turn"
            for turns, count in sorted(horizon_counts.items())
        )
        print(f"  Multi-turn    : {len(multi_turn)}  ({breakdown})")
    if by_style:
        parts = [f"{s}={by_style.get(s, 0)}" for s in _STYLE_CHOICES]
        print(f"  By style      : {', '.join(parts)}")
    print("=" * 60)


async def run_benign_sweep(args):
    """Iterate every blue-team checkpoint and run benign-only episodes."""
    # Apply --style-filter by monkey-patching the module-level pool that
    # evaluate_benign_only imports. Keeping it tightly scoped avoids
    # touching the training-side env.
    import marft.envs.blueteam_sql.blueteam_sql_env as _benv
    import util.cross_evaluate as _cxe

    keep_styles = set(args.style_filter)
    unknown = keep_styles - set(_STYLE_CHOICES)
    if unknown:
        print(f"ERROR: unknown style(s) in --style-filter: {sorted(unknown)}. "
              f"Choices: {_STYLE_CHOICES}", file=sys.stderr)
        sys.exit(2)

    if keep_styles != set(_STYLE_CHOICES):
        filtered = _filter_pool_by_style(list(BENIGN_EVAL_QUERIES), keep_styles)
        print(f"[run_benign_eval] style-filter={sorted(keep_styles)} → "
              f"{len(filtered)}/{len(BENIGN_EVAL_QUERIES)} queries remain")
        _benv.BENIGN_EVAL_QUERIES[:] = filtered  # in-place for import sites
        _cxe.BENIGN_EVAL_QUERIES = filtered
        pool_for_summary = filtered
    else:
        pool_for_summary = list(BENIGN_EVAL_QUERIES)

    _print_query_pool_summary(pool_for_summary)
    os.makedirs(args.output_dir, exist_ok=True)

    _, blue_loras = discover_checkpoints(args.selfplay_dir)
    print(f"Discovered {len(blue_loras)} blue LoRA checkpoints")

    if not blue_loras and not args.include_base:
        print("ERROR: No blue LoRA checkpoints found and --no-include-base set. Nothing to do.")
        sys.exit(1)

    blue_versions = ([0] if args.include_base else []) + sorted(blue_loras.keys())
    print(f"Blue versions to evaluate: {blue_versions}")
    print(f"Episodes per version: {args.episodes}")
    print(f"Output dir: {args.output_dir}")

    completed = load_progress(args.output_dir) if args.resume else set()
    if completed:
        print(f"Resuming: {len(completed)} iterations already completed")

    mcp = MCPClient(max_concurrent=args.concurrency)
    await mcp.connect_to_server("/app/mcp/postgres.py")
    print("MCP client connected.")

    # Verify vLLM server is up
    print("Verifying vLLM server...")
    _probe = OfflineLLM(
        model_name=args.base_model,
        vllm_base_url=args.blue_vllm_url,
        max_tokens=1,
    )
    del _probe
    print("vLLM server verified.")

    try:
        for blue_iter in blue_versions:
            benign_key = f"benign_blue_{blue_iter}"
            if benign_key in completed:
                print(f"  [{benign_key}] Skipping (already completed)")
                continue

            blue_model = f"blue_{blue_iter}" if blue_iter > 0 else args.base_model
            blue_llm = OfflineLLM(
                model_name=blue_model,
                mcp_client=None,
                vllm_base_url=args.blue_vllm_url,
                max_tokens=512,
                skip_health_check=True,
            )

            try:
                await evaluate_benign_only(
                    blue_iter=blue_iter,
                    blue_llm=blue_llm,
                    mcp_client=mcp,
                    n_episodes=args.episodes,
                    seed=args.seed,
                    output_dir=args.output_dir,
                    concurrency=args.concurrency,
                )
                completed.add(benign_key)
                save_progress(args.output_dir, completed)
            except Exception as e:
                print(f"  [{benign_key}] FAILED: {e}")
                traceback.print_exc()

    finally:
        print("Cleaning up MCP client...")
        await mcp.cleanup()

    print(f"\nSweep complete. {len(completed)} iterations finished.")


def aggregate(output_dir: str, selfplay_dir: str = "", base_model: str = ""):
    """Rebuild benign_eval_results.json from per-iter summary files."""
    benign_dir = os.path.join(output_dir, "benign_only")

    results = {
        "metadata": {
            "selfplay_dir": selfplay_dir,
            "base_model": base_model,
            "output_dir": output_dir,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "benign_only": {},
    }

    if os.path.isdir(benign_dir):
        for blue_name in sorted(os.listdir(benign_dir)):
            summary_path = os.path.join(benign_dir, blue_name, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    results["benign_only"][blue_name] = json.load(f)

    # Convenience rollup across iterations
    summaries = list(results["benign_only"].values())
    if summaries:
        tprs = [s["tpr"] for s in summaries]
        results["metadata"]["n_iters"] = len(summaries)
        results["metadata"]["per_iter_tpr"] = {
            s_name: s["tpr"]
            for s_name, s in results["benign_only"].items()
        }
        results["metadata"]["tpr_range"] = [round(min(tprs), 2), round(max(tprs), 2)]
        results["metadata"]["mean_tpr"] = round(sum(tprs) / len(tprs), 2)

    out_path = os.path.join(output_dir, "benign_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Aggregated {len(summaries)} iterations → {out_path}")
    if summaries:
        print(f"  TPR range: {results['metadata']['tpr_range']}%  "
              f"mean: {results['metadata']['mean_tpr']}%")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benign-TPR sweep over all blue-team LoRA checkpoints."
    )
    parser.add_argument("--selfplay-dir", required=True,
                        help="Self-play run dir containing iter_*/blueteam checkpoints")
    parser.add_argument("--base-model", default="",
                        help="Base model HF ID (auto-detected if omitted)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write benign_only/ + progress.json")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Benign episodes per blue iteration (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--blue-vllm-url", default="http://localhost:8002/v1",
                        help="Blue-team vLLM base URL")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent episodes (default: 8)")
    parser.add_argument("--include-base", action="store_true", default=True,
                        help="Include iter_0 (base model, no LoRA) as baseline")
    parser.add_argument("--no-include-base", action="store_false", dest="include_base")
    parser.add_argument("--resume", action="store_true",
                        help="Skip iterations already in progress.json")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Rebuild benign_eval_results.json without running episodes")
    parser.add_argument(
        "--style-filter",
        default=",".join(_STYLE_CHOICES),
        help=(
            "Comma-separated list of benign styles to keep from BENIGN_EVAL_QUERIES. "
            f"Choices: {','.join(_STYLE_CHOICES)}. "
            "Use --style-filter adversarial to evaluate the 'hard' benign corpus."
        ),
    )

    args = parser.parse_args()
    args.style_filter = [s.strip() for s in args.style_filter.split(",") if s.strip()]

    if args.aggregate_only:
        aggregate(args.output_dir, args.selfplay_dir, args.base_model)
    else:
        if not args.base_model:
            print("ERROR: --base-model is required unless --aggregate-only is set.")
            sys.exit(1)
        asyncio.run(run_benign_sweep(args))
        aggregate(args.output_dir, args.selfplay_dir, args.base_model)


if __name__ == "__main__":
    main()
