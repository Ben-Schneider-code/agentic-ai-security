#!/usr/bin/env python3
"""Redteam eval-prompt manifest: build / load / validate.

The manifest pins the EXACT (question, per-turn strategies) sequence that
cross-eval feeds the red policy. With the manifest, every (red_iter, blue_iter)
pairing replays byte-identical prompts, and those prompts are byte-identical to
what training actually drew — so "training and evaluation use the same redteam
prompts" holds literally, not just distributionally.

Two ways to produce the attack inputs, which MUST agree (cross-checked here):

  1. From source (TRUE REPLAY): read the env-logged ``redteam_prompts.jsonl``
     that ``SQLEnv`` writes during training (mode == "train"). This is the
     ground truth — immune to any reconstruction drift.

  2. By reconstruction (CPU, no GPU): replay the env RNG exactly. After
     ``env.seed(red_seed + source_rank*1000)`` (train_sql.py:123 ->
     ``redteam_sql_env.SQLEnv.seed``) the env RNG is a FRESH ``random.Random``
     with NO constructor-time draw, and ``reset()`` pre-draws a FIXED count per
     episode: one ``randint`` for ``dataset.sample`` followed by ``horizon``
     ``generate_redteam_prompt`` draws. We mirror that order exactly.

The fixed per-episode draw count is what makes (2) faithful: the env pre-draws
all ``horizon`` strategies in ``reset()`` regardless of where the rollout
terminates, so the stream never desyncs on an early (honeypot) termination.
NOTE: this faithfulness only holds for runs produced by the pre-draw env. Runs
that predate it drew strategies lazily and cannot be reconstructed on CPU; build
their manifests from a re-run's logs instead.

HONEYPOT_TYPE must be exported before import (redteam_sql_env fail-fast contract).

CLI:
  HONEYPOT_TYPE=rowcol python util/build_redteam_manifest.py \
      --selfplay-dir results-<ID>                # build from logs + cross-check
  HONEYPOT_TYPE=rowcol python util/build_redteam_manifest.py \
      --selfplay-dir results-<ID> --reconstruct-only --n 200   # no logs (new-env only)
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "MARFT")):
    if _p not in sys.path:
        sys.path.append(_p)

import pandas as pd  # noqa: E402

from marft.envs.redteam_sql import redteam_sql_env as _env_mod  # noqa: E402
from marft.envs.redteam_sql.redteam_sql_env import (  # noqa: E402
    generate_redteam_prompt,
)

MANIFEST_FILENAME = "redteam_eval_manifest.json"
MANIFEST_SCHEMA = "redteam_eval_manifest/v1"

# Same CSV, same path coupling as training and cross-eval (env module dir).
_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(_env_mod.__file__)),
    "redteam_dataset_specific.csv",
)
_QUESTIONS_DF = pd.read_csv(_DATASET_PATH)


def dataset_metadata() -> dict:
    """SHA-256 + row-count + path of the question CSV (for audit parity)."""
    with open(_DATASET_PATH, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"path": _DATASET_PATH, "rows": int(len(_QUESTIONS_DF)), "sha256": sha}


# ─────────────────────────── attack-input sources ───────────────────────────


def reconstruct_attack_inputs(
    red_seed: int, source_rank: int, horizon: int, n_episodes: int
) -> list[dict]:
    """Replay SQLEnv's post-reseed RNG to reproduce training's prompt stream.

    Mirrors ``redteam_sql_env`` after ``env.seed(red_seed + source_rank*1000)``:
    a fresh ``random.Random`` with NO constructor draw, then per episode one
    ``randint`` (dataset sample) followed by ``horizon`` strategy draws — the
    exact order ``reset()`` pre-draws them in.
    """
    rng = random.Random(red_seed + source_rank * 1000)
    out: list[dict] = []
    for ep in range(n_episodes):
        sample_state = rng.randint(0, 2**31 - 1)
        row = _QUESTIONS_DF.sample(n=1, random_state=sample_state)
        strategies = [
            generate_redteam_prompt(current_turn=t, max_turns=horizon, rng=rng)
            for t in range(1, horizon + 1)
        ]
        out.append(
            {
                "episode": ep,
                "question_idx": int(row.index[0]),
                "question": str(row.iloc[0]["prompt"]),
                "strategies": strategies,
            }
        )
    return out


def load_logged_attack_inputs(
    selfplay_dir: str,
) -> tuple[list[str], dict[int, list[dict]]]:
    """Gather the env-logged prompt streams for ALL ranks across iterations.

    Globs every ``iter_*/redteam/**/redteam_prompts.jsonl`` under ``selfplay_dir``.
    Training uses ``n_rollout_threads`` parallel envs (ranks 0..R-1), each on a
    distinct RNG stream (``random.Random(red_seed + rank*1000)``); all ranks
    together are the COMPLETE set of redteam prompts training drew. (Rank 0 alone
    is far too few when num_env_steps is modest, so cross-eval pulls from all of
    them.) Each red iteration re-seeds identically and the per-episode draw count
    is fixed, so every iteration logs the SAME per-rank stream; we merge across
    iterations asserting agreement (a free cross-iteration consistency check).
    Per-rank episodes must form a contiguous 0..K range — fail fast otherwise so a
    truncated log can't yield a silently gapped manifest.

    Returns (log_paths, {rank: ordered_records}).
    """
    pattern = os.path.join(
        selfplay_dir, "iter_*", "redteam", "**", "redteam_prompts.jsonl"
    )
    paths = sorted(glob.glob(pattern, recursive=True))
    by_rank: dict[int, dict[int, dict]] = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rank = int(rec["rank"])
                ep = int(rec["episode"])
                cur = {
                    "rank": rank,
                    "episode": ep,
                    "question_idx": int(rec["question_idx"]),
                    "question": str(rec["question"]),
                    "strategies": list(rec["strategies"]),
                }
                d = by_rank.setdefault(rank, {})
                if ep in d and d[ep] != cur:
                    raise RuntimeError(
                        f"Conflicting rank-{rank} episode {ep} across iteration "
                        f"logs — prompt stream is non-deterministic."
                    )
                d[ep] = cur
    ranks: dict[int, list[dict]] = {}
    for rank in sorted(by_rank):
        d = by_rank[rank]
        eps = sorted(d)
        if eps != list(range(len(eps))):
            raise RuntimeError(
                f"rank-{rank} episodes are not contiguous 0..K "
                f"(got {eps[:5]}...{eps[-3:]}) — refusing a gapped manifest."
            )
        ranks[rank] = [d[e] for e in eps]
    return paths, ranks


def flatten_ranks(ranks: dict[int, list[dict]]) -> list[dict]:
    """Deterministic flat order across ranks: interleave by episode then rank.

    episode 0 of every rank, then episode 1 of every rank, ... This spreads the
    eval prefix evenly across all training seeds (so a 1-of-8-rank eval still
    samples all eight streams), and handles unequal per-rank lengths (a rank that
    ran fewer episodes is simply absent from the tail).
    """
    if not ranks:
        return []
    max_ep = max(len(v) for v in ranks.values())
    out: list[dict] = []
    for ep in range(max_ep):
        for rank in sorted(ranks):
            if ep < len(ranks[rank]):
                out.append(ranks[rank][ep])
    return out


def cross_check_ranks(ranks: dict[int, list[dict]], red_seed: int, horizon: int) -> int:
    """Cross-check every rank's logged stream against its RNG reconstruction."""
    total = 0
    for rank, logged in sorted(ranks.items()):
        recon = reconstruct_attack_inputs(red_seed, rank, horizon, len(logged))
        total += cross_check(logged, recon)
    return total


# ───────────────────────────── manifest I/O ─────────────────────────────────


def build_manifest_dict(
    red_seed: int, horizon: int, attack_inputs: list[dict], source: dict
) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "red_seed": int(red_seed),
        "horizon": int(horizon),
        "source": source,
        "n_attack_inputs": len(attack_inputs),
        "dataset": dataset_metadata(),
        "attack_inputs": [
            {
                "rank": a.get("rank"),
                "episode": a.get("episode"),
                "question_idx": int(a["question_idx"]),
                "question": a["question"],
                "strategies": list(a["strategies"]),
            }
            for a in attack_inputs
        ],
    }


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_manifest(manifest: dict, horizon: int, n_attack_needed: int) -> list[dict]:
    """Fail-fast checks before cross-eval consumes a manifest. Returns inputs."""
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(
            f"Unexpected manifest schema {manifest.get('schema')!r} "
            f"(expected {MANIFEST_SCHEMA!r})."
        )
    m_h = int(manifest["horizon"])
    if m_h != horizon:
        raise RuntimeError(
            f"Manifest horizon {m_h} != eval horizon {horizon}. Rebuild the "
            f"manifest or run cross-eval with --horizon {m_h}."
        )
    inputs = manifest["attack_inputs"]
    if len(inputs) < n_attack_needed:
        raise RuntimeError(
            f"Manifest has {len(inputs)} attack inputs but cross-eval needs "
            f"{n_attack_needed} (episodes // 2). Rebuild with more episodes."
        )
    for i, a in enumerate(inputs[:n_attack_needed]):
        if len(a["strategies"]) != horizon:
            raise RuntimeError(
                f"Manifest attack input {i} has {len(a['strategies'])} strategies, "
                f"expected horizon={horizon}."
            )
    return inputs


def cross_check(logged: list[dict], reconstructed: list[dict]) -> int:
    """Assert the logged stream equals independent reconstruction, byte-for-byte.

    This is the guard that makes the manifest trustworthy: if training's logged
    prompts ever diverge from the RNG reconstruction (e.g. generate_redteam_prompt
    changed, or the env RNG order drifted), the build fails loudly here.
    """
    n = min(len(logged), len(reconstructed))
    for i in range(n):
        keys = ("question_idx", "question", "strategies")
        if {k: logged[i][k] for k in keys} != {k: reconstructed[i][k] for k in keys}:
            raise RuntimeError(
                f"Manifest cross-check FAILED at episode {i}: env-logged prompts "
                f"!= RNG reconstruction. Investigate generate_redteam_prompt / "
                f"env RNG consumption order."
            )
    return n


# ──────────────────────────────── CLI ───────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Build redteam eval-prompt manifest.")
    ap.add_argument("--selfplay-dir", required=True, help="results-<ID> directory.")
    ap.add_argument(
        "--out",
        default=None,
        help=f"Manifest output path (default: <selfplay-dir>/{MANIFEST_FILENAME}).",
    )
    ap.add_argument("--source-rank", type=int, default=0)
    ap.add_argument(
        "--reconstruct-only",
        action="store_true",
        help="Skip logs; reconstruct from red_seed (valid only for pre-draw-env runs).",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of attack episodes (required with --reconstruct-only).",
    )
    args = ap.parse_args()

    summary_path = os.path.join(args.selfplay_dir, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    red_seed = int(summary["red_seed"])
    horizon = int(summary["horizon"])
    out_path = args.out or os.path.join(args.selfplay_dir, MANIFEST_FILENAME)

    if args.reconstruct_only:
        if args.n is None:
            raise SystemExit("--n is required with --reconstruct-only.")
        attack_inputs = reconstruct_attack_inputs(
            red_seed, args.source_rank, horizon, args.n
        )
        for a in attack_inputs:
            a["rank"] = args.source_rank
        source = {"mode": "reconstructed", "ranks": [args.source_rank]}
        print(
            f"[manifest] reconstructed {len(attack_inputs)} episodes from "
            f"red_seed={red_seed} rank={args.source_rank} (NO log cross-check)."
        )
    else:
        paths, ranks = load_logged_attack_inputs(args.selfplay_dir)
        if not ranks:
            raise SystemExit(
                f"No redteam_prompts.jsonl under {args.selfplay_dir} "
                f"(searched iter_*/redteam/**). This run predates env "
                f"prompt-logging; re-run training with the pre-draw env, or use "
                f"--reconstruct-only --n N if it already uses it."
            )
        n_ok = cross_check_ranks(ranks, red_seed, horizon)
        attack_inputs = flatten_ranks(ranks)
        per_rank = {r: len(v) for r, v in sorted(ranks.items())}
        source = {"mode": "logged-all-ranks", "ranks": sorted(ranks)}
        print(
            f"[manifest] cross-check OK on {n_ok} episodes (env-logged == "
            f"reconstruction) across ranks {per_rank} from {len(paths)} log file(s)."
        )

    manifest = build_manifest_dict(red_seed, horizon, attack_inputs, source)
    write_manifest(out_path, manifest)
    print(
        f"[manifest] wrote {len(attack_inputs)} attack inputs "
        f"({source['mode']}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
