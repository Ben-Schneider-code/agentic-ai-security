"""
Semantic diversity of RL redteam attack queries vs. human jailbreaks.

Embeds queries using sentence-transformers, projects to 2D via PCA, and
visualises the coverage overlap between RL-trained attack strategies and
hand-crafted human jailbreaks. Per-iteration spread (std of PC coordinates)
is computed and printed.

Query source modes:
  training_time  — red_team_input from per-iteration training reward_debug.jsonl (default)
  evaluation_time — user_message from cross-eval diagonal pairings reward_debug.jsonl
  final_episode   — last tail_pct fraction of training episodes only

NOTE: The embedder defaults to a general-purpose sentence-transformer model (i.e. you should set a specific one yourself).
      semantic distances between SQL injection attack strategies. Candidates:
      - A model fine-tuned on cybersecurity text
      - A code/SQL embedding model (e.g. microsoft/codebert-base)
      - Custom embedding trained on adversarial SQL corpora

Can be run standalone:
    python plotting/plot_semantic_diversity.py --results results-<ID>
Or imported:
    from plotting.plot_semantic_diversity import plot_semantic_diversity, DESCRIPTION
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from ._data import (
        apply_paper_style,
        discover_iterations,
        find_run_dir,
        load_reward_debug_lines,
        get_attack_query,
        load_human_queries,
        load_benign_queries,
        load_cross_eval_results,
        parse_results_arg,
        HUMAN_COL,
        BENIGN_COL,
        BENIGN_MARKER,
        RUN_COLORS,
        FIG_SIZE_SINGLE,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import (
        apply_paper_style,
        discover_iterations,
        find_run_dir,
        load_reward_debug_lines,
        get_attack_query,
        load_human_queries,
        load_benign_queries,
        load_cross_eval_results,
        parse_results_arg,
        HUMAN_COL,
        BENIGN_COL,
        BENIGN_MARKER,
        FIG_SIZE_SINGLE,
    )

apply_paper_style()

DESCRIPTION = (
    "Semantic diversity of RL redteam attack queries vs. human jailbreaks and "
    "benign queries. Queries are embedded and projected to 2D via PCA. "
    "RL queries are coloured by self-play iteration; human jailbreaks are shown "
    "as orange stars; benign queries as small cyan triangles. Explained variance "
    "(PC1+PC2) and per-iteration PC spread are annotated."
)

_DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Query collection helpers
# ---------------------------------------------------------------------------


def _collect_training_queries(
    selfplay_dir: str,
    mode: str,
    tail_pct: float,
) -> dict[int, list[str]]:
    """
    Collect redteam attack queries from training reward_debug.jsonl files.

    Returns {iter_num: [query_text, ...]} (attack turns only).
    """
    iters_data = discover_iterations(selfplay_dir)
    queries_by_iter: dict[int, list[str]] = {}

    for entry in iters_data:
        n = entry["iter"]
        red_dir = entry.get("red_dir")
        if red_dir is None:
            continue
        run_dir = find_run_dir(red_dir)
        if run_dir is None:
            continue
        lines = load_reward_debug_lines(
            run_dir, mode=mode, tail_pct=tail_pct, turn_type="attack"
        )
        texts = [get_attack_query(ln) for ln in lines]
        texts = [t for t in texts if t]
        if texts:
            queries_by_iter[n] = texts

    return queries_by_iter


def _collect_eval_queries(
    selfplay_dir: str,
    cross_eval_subdir: str,
) -> dict[int, list[str]]:
    """
    Collect redteam attack queries from cross-eval diagonal pairings
    reward_debug.jsonl files.

    Returns {iter_num: [query_text, ...]} (attack turns only).
    """
    xeval = load_cross_eval_results(selfplay_dir, cross_eval_subdir)
    if xeval is None:
        return {}

    pairings = xeval.get("pairings", {})
    queries_by_iter: dict[int, list[str]] = {}

    for _key, pairing in pairings.items():
        ri = pairing.get("red_iter")
        bi = pairing.get("blue_iter")
        if ri is None or bi is None or ri != bi:
            continue
        # Locate the pairing's reward_debug.jsonl
        base = Path(selfplay_dir)
        pairing_dir = base / cross_eval_subdir / "pairings" / f"red_{ri}_blue_{bi}"
        if not pairing_dir.is_dir():
            continue
        lines = load_reward_debug_lines(
            pairing_dir, mode="evaluation_time", turn_type="attack"
        )
        texts = [get_attack_query(ln) for ln in lines]
        texts = [t for t in texts if t]
        if texts:
            queries_by_iter[ri] = texts

    return queries_by_iter


def collect_queries(
    selfplay_dir: str,
    mode: str,
    tail_pct: float,
    cross_eval_subdir: str = "cross_eval",
) -> dict[int, list[str]]:
    """
    Collect redteam attack queries for the given mode.

    Returns {iter_num: [query_text, ...]}.
    """
    if mode == "evaluation_time":
        return _collect_eval_queries(selfplay_dir, cross_eval_subdir)
    return _collect_training_queries(selfplay_dir, mode, tail_pct)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str], embedder: str) -> np.ndarray:
    """Embed a list of texts using sentence-transformers. Returns (N, D) array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedder, trust_remote_code=True)
    return model.encode(
        texts,
        show_progress_bar=len(texts) > 200,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


# ---------------------------------------------------------------------------
# Public plotting function (deterministic, idempotent)
# ---------------------------------------------------------------------------


def plot_semantic_diversity(
    results: list[tuple[str, str]],
    out_path: str | Path,
    human_queries_path: str = "new_jailbreaks.txt",
    query_mode: str = "training_time",
    tail_pct: float = 0.25,
    embedder: str = _DEFAULT_EMBEDDER,
    cross_eval_subdir: str = "cross_eval",
    benign_queries_path: str | None = "data/benign_pool_stats.json",
) -> Path:
    """
    Plot semantic diversity: RL queries vs. human jailbreaks vs. benign queries
    in PCA 2D space.

    Args:
        results:             [(label, selfplay_dir), ...]
        out_path:            Destination PNG path.
        human_queries_path:  Path to new_jailbreaks.txt (or equivalent).
        query_mode:          "training_time" | "evaluation_time" | "final_episode".
        tail_pct:            Fraction of final episodes used when mode="final_episode".
        embedder:            Sentence-transformers model ID.
        cross_eval_subdir:   Subdir for evaluation_time mode.
        benign_queries_path: Path to benign_pool_stats.json. Pass None to skip.

    Returns the resolved Path that was written.
    """
    from sklearn.decomposition import PCA

    out_path = Path(out_path)

    # Load human queries
    human_queries = load_human_queries(human_queries_path)
    print(
        f"  [semantic_diversity] Loaded {len(human_queries)} human jailbreak queries."
    )

    # Load benign queries (optional)
    benign_queries: list[str] = []
    if benign_queries_path:
        try:
            benign_queries = load_benign_queries(benign_queries_path)
            print(
                f"  [semantic_diversity] Loaded {len(benign_queries)} benign queries."
            )
        except FileNotFoundError as e:
            print(
                f"  [semantic_diversity] WARNING: {e} — skipping benign overlay.",
                file=sys.stderr,
            )

    # Collect RL queries per run
    run_queries: list[tuple[str, dict[int, list[str]]]] = []
    for label, selfplay_dir in results:
        qbi = collect_queries(selfplay_dir, query_mode, tail_pct, cross_eval_subdir)
        total = sum(len(v) for v in qbi.values())
        print(
            f"  [semantic_diversity] {label}: {total} RL queries across "
            f"{len(qbi)} iteration(s)."
        )
        if total > 0:
            run_queries.append((label, qbi))

    metrics: dict = {
        "query_mode": query_mode,
        "embedder": embedder,
        "human_query_count": len(human_queries),
        "benign_query_count": len(benign_queries),
        "runs": {},
    }

    if not run_queries:
        print(
            "  [semantic_diversity] No RL queries found — skipping plot.",
            file=sys.stderr,
        )
        fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
        ax.text(
            0.5,
            0.5,
            "No RL query data found.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return out_path, metrics

    # One subplot per run
    n_plots = len(run_queries)
    fig_w = FIG_SIZE_SINGLE[0] * n_plots
    fig, axes = plt.subplots(
        1, n_plots, figsize=(fig_w, FIG_SIZE_SINGLE[1]), squeeze=False
    )

    for plot_idx, (label, qbi) in enumerate(run_queries):
        ax = axes[0, plot_idx]

        # Flatten all texts: RL (tagged by iter) + human
        all_texts: list[str] = []
        iter_tags: list[int | None] = []

        sorted_iters = sorted(qbi.keys())
        for n in sorted_iters:
            texts = qbi[n]
            all_texts.extend(texts)
            iter_tags.extend([n] * len(texts))

        rl_count = len(all_texts)
        all_texts.extend(human_queries)
        human_count = len(human_queries)
        all_texts.extend(benign_queries)
        benign_count = len(benign_queries)

        # Embed and PCA
        print(
            f"  [semantic_diversity] Embedding {len(all_texts)} texts with {embedder}..."
        )
        embeddings = embed_texts(all_texts, embedder)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
        var = pca.explained_variance_ratio_

        rl_coords = coords[:rl_count]
        human_coords = coords[rl_count : rl_count + human_count]
        benign_coords = coords[rl_count + human_count :]
        rl_iters_arr = np.array(iter_tags[:rl_count], dtype=float)

        # Plot RL queries coloured by iteration
        min_iter = min(sorted_iters)
        max_iter = max(sorted_iters)
        cmap = plt.get_cmap("plasma")
        norm = plt.Normalize(vmin=min_iter, vmax=max_iter)
        sc = ax.scatter(
            rl_coords[:, 0],
            rl_coords[:, 1],
            c=rl_iters_arr,
            cmap=cmap,
            norm=norm,
            s=18,
            alpha=0.35,
            linewidths=0,
            zorder=2,
            label="RL queries",
        )

        # Plot benign queries (small, distinct marker — density-comparable to RL dots)
        if benign_count > 0:
            ax.scatter(
                benign_coords[:, 0],
                benign_coords[:, 1],
                c=BENIGN_COL,
                marker=BENIGN_MARKER,
                s=22,
                alpha=0.6,
                linewidths=0,
                zorder=3,
                label=f"Benign queries (n={benign_count})",
            )

        # Plot human queries
        ax.scatter(
            human_coords[:, 0],
            human_coords[:, 1],
            c=HUMAN_COL,
            marker="*",
            s=120,
            zorder=10,
            edgecolors="black",
            linewidths=0.5,
            label=f"Human jailbreaks (n={human_count})",
        )

        # Colorbar for iteration
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Iteration", fontsize=10)
        cbar.set_ticks(sorted_iters)

        ax.set_title(
            f"{label}\nPC1={var[0]:.1%}, PC2={var[1]:.1%}",
            fontsize=12,
        )
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)
        ax.legend(fontsize=9, loc="best", frameon=True)
        ax.grid(True, alpha=0.3)

        # Collect per-iteration spread statistics
        run_metrics: dict = {
            "total_rl_queries": rl_count,
            "iteration_count": len(sorted_iters),
            "explained_variance_pc1": float(var[0]),
            "explained_variance_pc2": float(var[1]),
            "per_iteration": {},
        }
        print(f"  [semantic_diversity] {label} — per-iteration PC std:")
        for n in sorted_iters:
            mask = rl_iters_arr == n
            qcount = int(mask.sum())
            entry: dict = {"query_count": qcount}
            if qcount >= 2:
                pc_std = rl_coords[mask].std(axis=0)
                spread = float(np.linalg.norm(pc_std))
                entry["std_pc1"] = float(pc_std[0])
                entry["std_pc2"] = float(pc_std[1])
                entry["spread_norm"] = spread
                print(
                    f"    iter {n:2d}: std(PC1)={pc_std[0]:.4f}, "
                    f"std(PC2)={pc_std[1]:.4f}, norm={spread:.4f}"
                )
            run_metrics["per_iteration"][str(n)] = entry
        metrics["runs"][label] = run_metrics

    fig.suptitle("Semantic Diversity: RL Queries vs. Human Jailbreaks", fontsize=13)
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path, metrics


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot semantic diversity of RL redteam queries vs. human jailbreaks."
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        metavar="DIR[:LABEL]",
    )
    parser.add_argument("--out", default="figures/semantic_diversity.png")
    parser.add_argument("--human-queries", default="new_jailbreaks.txt", metavar="PATH")
    parser.add_argument(
        "--query-mode",
        default="training_time",
        choices=["training_time", "evaluation_time", "final_episode"],
        help="Which queries to embed (default: training_time).",
    )
    parser.add_argument(
        "--tail-pct",
        type=float,
        default=0.25,
        help="Fraction of final episodes for final_episode mode.",
    )
    parser.add_argument(
        "--embedder", default=_DEFAULT_EMBEDDER, help="Sentence-transformers model ID."
    )
    parser.add_argument("--cross-eval-subdir", default="cross_eval")
    parser.add_argument(
        "--benign-queries",
        default="data/benign_pool_stats.json",
        metavar="PATH",
        help="Path to benign_pool_stats.json. Pass empty string to disable.",
    )
    args = parser.parse_args()

    results = parse_results_arg(args.results)
    out, _ = plot_semantic_diversity(
        results,
        args.out,
        args.human_queries,
        args.query_mode,
        args.tail_pct,
        args.embedder,
        args.cross_eval_subdir,
        benign_queries_path=args.benign_queries or None,
    )
    print(f"[{DESCRIPTION}]\n  → {out}")


if __name__ == "__main__":
    main()
