"""
Attack template evolution across iters (lexical + structural).

Answers supervisor Q3: "How do exploits at the equilibrium differ across epochs?"

Two figures:
  attack_template_evolution.png:
    Left  — 7×7 TF-IDF char 3–5-gram cosine similarity matrix across iters 1–7
            (computed on successful-attack red_team_input texts).
    Right — Per-iter top-3 modal n-grams (annotated table).

  attack_sql_pattern_evolution.png:
    Per-iter histogram of SQL pattern classes detected via regex on
    victim_full_response: JOIN / UNION / direct projection / subquery /
    catalog-lookup / refused-or-empty.

Source: results-<ID>/iter_{1..7}/redteam/.../debug_logs/reward_debug.jsonl

CLI:
    python plotting/plot_attack_evolution.py \\
        --results results-<ID>[:Label] [--out-dir figures/]
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

try:
    from ._data import apply_paper_style, parse_results_arg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting._data import apply_paper_style, parse_results_arg

apply_paper_style()

DESCRIPTION = "Attack template evolution: TF-IDF iter×iter similarity + per-iter SQL pattern histogram"

# SQL pattern classifiers (mutually exclusive, evaluated in order)
SQL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("union", re.compile(r"\bunion\s+(all\s+)?select\b", re.IGNORECASE)),
    ("subquery", re.compile(r"select\s+[^;]*?\(\s*select\b", re.IGNORECASE | re.DOTALL)),
    ("join", re.compile(r"\bjoin\b", re.IGNORECASE)),
    ("catalog", re.compile(r"\b(information_schema|pg_catalog|pg_class|pg_attribute|pg_tables|sqlite_master)\b", re.IGNORECASE)),
    ("direct_select", re.compile(r"\bselect\b.+\bfrom\b", re.IGNORECASE | re.DOTALL)),
]
PATTERN_LABELS = {
    "union": "UNION-based",
    "subquery": "Subquery",
    "join": "JOIN-based",
    "catalog": "Catalog lookup",
    "direct_select": "Direct projection",
    "none": "No SQL / refused",
}
PATTERN_COLORS = {
    "union": "#D32F2F",
    "subquery": "#7B1FA2",
    "join": "#1976D2",
    "catalog": "#F57C00",
    "direct_select": "#43A047",
    "none": "#9E9E9E",
}


def _classify_sql(text: str) -> str:
    if not text:
        return "none"
    sql_block = text
    # Try to extract from markdown code block
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        sql_block = m.group(1)
    for name, pat in SQL_PATTERNS:
        if pat.search(sql_block):
            return name
    return "none"


def _find_red_jsonl(iter_dir: Path) -> Path | None:
    candidates = list(iter_dir.glob("redteam/**/debug_logs/reward_debug.jsonl"))
    return candidates[0] if candidates else None


def _load_iter_attacks(jsonl_path: Path) -> tuple[list[str], list[str]]:
    """Return (successful_attack_inputs, victim_responses)."""
    inputs: list[str] = []
    responses: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("is_successful_attack"):
                continue
            txt = r.get("red_team_input")
            resp = r.get("victim_full_response") or ""
            if isinstance(txt, str) and txt.strip():
                inputs.append(txt.strip())
                responses.append(resp.strip() if isinstance(resp, str) else "")
    return inputs, responses


def _all_attack_responses(jsonl_path: Path) -> list[str]:
    """Return all victim_full_response (any outcome) for SQL pattern coverage."""
    out: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = r.get("victim_full_response") or ""
            if isinstance(resp, str):
                out.append(resp.strip())
    return out


def _top_ngrams(texts: list[str], n: int = 3, top_k: int = 3) -> list[tuple[str, int]]:
    token_re = re.compile(r"\w+", re.UNICODE)
    counter: Counter = Counter()
    for t in texts:
        toks = [w.lower() for w in token_re.findall(t)]
        for i in range(len(toks) - n + 1):
            counter[" ".join(toks[i : i + n])] += 1
    return counter.most_common(top_k)


def _tfidf_similarity_matrix(per_iter_texts: dict[int, list[str]]) -> tuple[list[int], np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    iters = sorted(per_iter_texts.keys())
    iter_corpus = [" ".join(per_iter_texts[i]) for i in iters]  # one doc per iter
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    m = vec.fit_transform(iter_corpus)
    sim = cosine_similarity(m)
    return iters, sim


def plot_attack_evolution(
    results: list[tuple[str, str]],
    out_dir: str = "figures/",
) -> tuple[Path, Path]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    template_path = out_dir_p / "attack_template_evolution.png"
    pattern_path = out_dir_p / "attack_sql_pattern_evolution.png"
    sidecar_path = out_dir_p / "attack_evolution.json"

    label, selfplay_dir = results[0]
    base = Path(selfplay_dir)

    per_iter_inputs: dict[int, list[str]] = {}
    per_iter_responses_succ: dict[int, list[str]] = {}
    per_iter_responses_all: dict[int, list[str]] = {}
    for iter_dir in sorted(base.glob("iter_*")):
        try:
            iter_num = int(iter_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if iter_num < 1:
            continue
        jsonl = _find_red_jsonl(iter_dir)
        if jsonl is None:
            continue
        inputs, responses = _load_iter_attacks(jsonl)
        if not inputs:
            continue
        per_iter_inputs[iter_num] = inputs
        per_iter_responses_succ[iter_num] = responses
        per_iter_responses_all[iter_num] = _all_attack_responses(jsonl)

    if len(per_iter_inputs) < 2:
        print(f"[attack_evolution] insufficient data (only {len(per_iter_inputs)} iters)", file=sys.stderr)
        return template_path, pattern_path

    # ------------------------------------------------------------------
    # FIGURE 1: TF-IDF similarity + top n-grams
    # ------------------------------------------------------------------
    iters, sim = _tfidf_similarity_matrix(per_iter_inputs)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    im = ax_l.imshow(sim, cmap="viridis", vmin=0.65, vmax=1.0, aspect="equal")
    ax_l.set_xticks(range(len(iters)))
    ax_l.set_xticklabels(iters)
    ax_l.set_yticks(range(len(iters)))
    ax_l.set_yticklabels(iters)
    ax_l.set_xlabel("Red iteration")
    ax_l.set_ylabel("Red iteration")
    ax_l.set_title(
        f"TF-IDF char 3–5-gram cosine similarity\n(successful-attack red_team_input texts, per iter)"
    )
    for i in range(len(iters)):
        for j in range(len(iters)):
            v = sim[i, j]
            color = "white" if v < 0.7 else "black"
            ax_l.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax_l, fraction=0.046, pad=0.04, label="Cosine similarity")

    # Per-iter top trigrams
    ax_r.axis("off")
    ax_r.set_title("Per-iter top-3 successful-attack trigrams")
    rows = []
    for it in iters:
        top = _top_ngrams(per_iter_inputs[it], n=3, top_k=3)
        rows.append([str(it)] + [f"{ng} ({c})" for ng, c in top] + [""] * (3 - len(top)))
    table = ax_r.table(
        cellText=rows,
        colLabels=["iter", "#1 trigram (count)", "#2", "#3"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    fig.suptitle(
        f"Per-iter template overlap (small-N caveat at iters 4, 5; N≤4 successful attacks); SQL pattern panel is the robust stability signal",
        y=1.02,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(template_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIGURE 2: SQL pattern histogram per iter (covers all attack turns)
    # ------------------------------------------------------------------
    pattern_iters = sorted(per_iter_responses_all.keys())
    pattern_classes = ["direct_select", "join", "subquery", "union", "catalog", "none"]
    pattern_matrix = np.zeros((len(pattern_classes), len(pattern_iters)), dtype=float)
    n_per_iter: list[int] = []
    for j, it in enumerate(pattern_iters):
        responses = per_iter_responses_all[it]
        n_per_iter.append(len(responses))
        if not responses:
            continue
        cls_counts = Counter(_classify_sql(r) for r in responses)
        for i, cls in enumerate(pattern_classes):
            pattern_matrix[i, j] = cls_counts.get(cls, 0) / len(responses) * 100

    fig2, ax2 = plt.subplots(figsize=(9.0, 5.0))
    bottoms = np.zeros(len(pattern_iters))
    for i, cls in enumerate(pattern_classes):
        ax2.bar(
            pattern_iters,
            pattern_matrix[i],
            bottom=bottoms,
            color=PATTERN_COLORS[cls],
            label=PATTERN_LABELS[cls],
        )
        bottoms += pattern_matrix[i]
    ax2.set_xticks(pattern_iters)
    ax2.set_xlabel("Red training iteration")
    ax2.set_ylabel("% of attack turns")
    ax2.set_title("SQL pattern class distribution across iters (all attack turns)")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.set_ylim(0, 105)
    for j, n in enumerate(n_per_iter):
        ax2.text(pattern_iters[j], 102, f"n={n}", ha="center", fontsize=7, color="#555555")

    fig2.tight_layout()
    fig2.savefig(pattern_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    sidecar = {
        "description": DESCRIPTION,
        "iters": iters,
        "tfidf_similarity_matrix": sim.tolist(),
        "n_successful_per_iter": {str(it): len(per_iter_inputs[it]) for it in iters},
        "n_total_attacks_per_iter": {str(it): len(per_iter_responses_all.get(it, [])) for it in iters},
        "sql_pattern_pct_per_iter": {
            str(it): {
                cls: round(float(pattern_matrix[i, j]), 2)
                for i, cls in enumerate(pattern_classes)
            }
            for j, it in enumerate(pattern_iters)
        },
        "top_ngrams_per_iter": {
            str(it): [{"trigram": ng, "count": c} for ng, c in _top_ngrams(per_iter_inputs[it], n=3, top_k=5)]
            for it in iters
        },
    }
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[attack_evolution] saved {template_path} and {pattern_path}")
    return template_path, pattern_path


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out-dir", default="figures/")
    args = p.parse_args()
    results = parse_results_arg(args.results)
    plot_attack_evolution(results, args.out_dir)


if __name__ == "__main__":
    main()
