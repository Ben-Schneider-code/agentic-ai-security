#!/usr/bin/env python3
"""Compute red-team output diversity diagnostics post-hoc.

Promised in §sec:diagnostics of the paper: distinct-4-gram coverage and
average pairwise cosine dissimilarity on the final 100 episodes of each
red-team training phase. A sharp drop in either metric is a mode-
collapse signal. This script reads the training ``reward_debug.jsonl``
that ``redteam_sql_env.py`` already writes, so no re-training is
required.

We use TF-IDF (character 3--5-gram) cosine similarity as the embedding,
which is self-contained (no pretrained model to download) and captures
lexical/phrasing shifts that matter for a social-engineering attacker.
The paper's optional sentence-embedding variant can be swapped in later;
TF-IDF is sufficient for collapse detection.

A category-conditional violation rate is promised too; the current
training log does not persist the sampled strategy category, so that
breakdown is skipped with a note rather than fabricated. (Adding the
field would require editing ``redteam_sql_env.py``, which the project
plan prohibits.)

Run:
    python util/diversity_diagnostics.py <results_dir>

Outputs in ``<results_dir>/diversity/``:
    summary.json           — per-iteration distinct-4-gram + mean dissimilarity
    trend.png              — trend across iterations
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_iteration_logs(results_dir: Path) -> dict[int, Path]:
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


def load_attack_texts(jsonl_path: Path, last_n: int = 100) -> list[str]:
    """Return the red-team attack strings from the final ``last_n``
    episodes of this iteration. Episodes are identified by the
    ``episode`` field; the step with the largest index per episode is
    kept as the representative attack.
    """
    by_episode: dict[int, list[tuple[int, str]]] = {}
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = r.get("red_team_input") or ""
            if not text or not isinstance(text, str):
                continue
            ep = int(r.get("episode", -1))
            step = int(r.get("step_count", 0))
            by_episode.setdefault(ep, []).append((step, text.strip()))

    if not by_episode:
        return []

    selected_eps = sorted(by_episode)[-last_n:]
    return [by_episode[ep][-1][1] for ep in selected_eps if by_episode[ep]]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text)]


def distinct_ngram_ratio(texts: list[str], n: int = 4) -> float:
    """Type/token ratio over word n-grams across the corpus. 1.0 means
    every observed n-gram is unique; 0.0 means one n-gram dominates.
    """
    counter: Counter = Counter()
    total = 0
    for t in texts:
        toks = tokens(t)
        for i in range(len(toks) - n + 1):
            counter[tuple(toks[i : i + n])] += 1
            total += 1
    if total == 0:
        return float("nan")
    return len(counter) / total


def pairwise_mean_dissimilarity(texts: list[str]) -> float:
    """Mean (1 − cosine similarity) over all unordered pairs of texts,
    using TF-IDF on character 3--5-grams. Returns NaN on corpora too
    small to form a pair.
    """
    if len(texts) < 2:
        return float("nan")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    try:
        m = vec.fit_transform(texts)
    except ValueError:
        return float("nan")
    sim = cosine_similarity(m)
    n = len(texts)
    iu = np.triu_indices(n, k=1)
    if len(iu[0]) == 0:
        return float("nan")
    return float(1.0 - sim[iu].mean())


def process_iteration(
    iter_idx: int, jsonl_path: Path, last_n: int, n_gram: int
) -> dict:
    texts = load_attack_texts(jsonl_path, last_n=last_n)
    return {
        "iter": iter_idx,
        "n_episodes_used": len(texts),
        "distinct_ngram_ratio": distinct_ngram_ratio(texts, n=n_gram),
        "pairwise_mean_dissimilarity": pairwise_mean_dissimilarity(texts),
    }


def plot_trend(summary: list[dict], out_path: Path, n_gram: int) -> None:
    iters = [s["iter"] for s in summary]
    dist = [s["distinct_ngram_ratio"] for s in summary]
    diss = [s["pairwise_mean_dissimilarity"] for s in summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(iters, dist, "o-", color="#2980b9")
    ax1.set_ylim(0, max(1.0, max([d for d in dist if not np.isnan(d)], default=1.0) * 1.1))
    ax1.set_xlabel("Self-play iteration")
    ax1.set_ylabel(f"Distinct-{n_gram}-gram ratio")
    ax1.set_title(f"Lexical diversity (distinct-{n_gram}-gram / total)")
    ax1.grid(alpha=0.3)

    ax2.plot(iters, diss, "s-", color="#c0392b")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Self-play iteration")
    ax2.set_ylabel("Mean pairwise (1 − cos sim)")
    ax2.set_title("Semantic spread (TF-IDF char 3–5-gram)")
    ax2.grid(alpha=0.3)

    fig.suptitle("Red-team output diversity, last 100 episodes per iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--last-n", type=int, default=100)
    ap.add_argument("--n-gram", type=int, default=4)
    args = ap.parse_args()

    logs = find_iteration_logs(args.results_dir)
    if not logs:
        print(f"No iter_*/redteam debug logs found under {args.results_dir}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (args.results_dir / "diversity")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for k, path in logs.items():
        print(f"[iter {k}] {path}")
        summary.append(process_iteration(k, path, args.last_n, args.n_gram))

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "last_n_episodes": args.last_n,
                "n_gram": args.n_gram,
                "notes": (
                    "strategy_category is not persisted in reward_debug.jsonl "
                    "in the current training harness, so the promised "
                    "category-conditional violation rate is not reported here."
                ),
                "iterations": summary,
            },
            f, indent=2,
        )
    plot_trend(summary, out_dir / "trend.png", args.n_gram)
    print(f"\nWrote {summary_path} and trend.png")
    for s in summary:
        print(
            f"  iter {s['iter']:>2}: n={s['n_episodes_used']:>3} "
            f"distinct-{args.n_gram}={s['distinct_ngram_ratio']:.3f}  "
            f"mean_dissim={s['pairwise_mean_dissimilarity']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
