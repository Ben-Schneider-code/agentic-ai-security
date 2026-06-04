#!/usr/bin/env python3
"""Byte-level verification that cross-eval replays training's exact prompts.

Two invariants, both required for "training and evaluation use the same redteam
prompts":

  1. RECONSTRUCTION FIDELITY — build_redteam_manifest.reconstruct_attack_inputs
     consumes the env RNG in the SAME order SQLEnv does AFTER env.seed():
       * fresh ``random.Random(red_seed + rank*1000)``  (train_sql.py:123 ->
         SQLEnv.seed)
       * NO constructor-time draw. SQLEnv.__init__ draws once on
         ``random.Random(seed)`` (line 1318) but env.seed() then installs a fresh
         RNG, discarding it — so it never feeds any episode.
       * per episode: one ``randint`` for ``dataset.sample`` then ``horizon``
         ``generate_redteam_prompt`` draws (pre-drawn in reset(); state_transition
         no longer draws — fixed count, immune to early termination).
     Checked against a hand-transcribed reference below. If the env RNG order
     ever drifts, this fails. (The OLD verify wrongly included a constructor draw
     and never modeled env.seed(), so it validated against a wrong model of
     training — that bug is what this rewrite fixes.)

  2. REPLAY FIDELITY — the cross-eval episode plan replays manifest attack inputs
     verbatim and is pairing-invariant (seeded from red_seed alone, no
     red_iter/blue_iter).

Optionally, with --selfplay-dir, also runs the real integration check:
env-logged ``redteam_prompts.jsonl`` == reconstruction.

We do NOT instantiate SQLEnv (needs MCP + vLLM) — we inline a reference of the
relevant lines. Run: HONEYPOT_TYPE=rowcol python util/verify_match_train_prompts.py
"""

import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "MARFT")):
    if _p not in sys.path:
        sys.path.append(_p)

# HONEYPOT_TYPE must be set before importing redteam_sql_env — same contract as
# the rest of the codebase. Fail fast if missing.
if "HONEYPOT_TYPE" not in os.environ:
    raise RuntimeError(
        "HONEYPOT_TYPE not set. Export rowcol|row|col before running this verify script."
    )

from marft.envs.redteam_sql.redteam_sql_env import (  # noqa: E402
    generate_redteam_prompt,
    redteam_initial_obs,
)
from marft.envs.obs_format import DISPLAY_CAP, cap_display  # noqa: E402
from marft.mas.prompt_format import build_agent_prompt  # noqa: E402
from util.build_redteam_manifest import (  # noqa: E402
    _QUESTIONS_DF,
    build_manifest_dict,
    cross_check_ranks,
    flatten_ranks,
    load_logged_attack_inputs,
    load_manifest,
    reconstruct_attack_inputs,
    validate_manifest,
)
from util.cross_evaluate import _build_match_train_episode_plan  # noqa: E402


def _reference_attack_inputs(red_seed, source_rank, horizon, n_episodes):
    """Hand-transcribed mirror of SQLEnv's post-env.seed() RNG consumption.

    Mirrors redteam_sql_env after ``env.seed(red_seed + source_rank*1000)``:
    a fresh ``random.Random`` with NO constructor draw, then per episode one
    randint (dataset sample) followed by ``horizon`` strategy draws — the exact
    order ``reset()`` pre-draws them in.
    """
    rng = random.Random(red_seed + source_rank * 1000)
    out = []
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


def check_reconstruction_fidelity() -> bool:
    cases = [
        # (red_seed, source_rank, horizon, n_episodes)
        (10, 0, 5, 6),    # canonical run: red_seed=10 (replicate-seed 0), rank 0
        (10, 1, 5, 6),    # rank-1 stream
        (10, 7, 5, 6),    # rank-7 stream (8 rollout threads, run_training.sh:366)
        (1010, 0, 5, 8),  # replicate-seed 1 -> red_seed = 10 + 1*1000
        (10, 0, 3, 5),    # different horizon
    ]
    ok = True
    for red_seed, rank, horizon, n in cases:
        recon = reconstruct_attack_inputs(red_seed, rank, horizon, n)
        ref = _reference_attack_inputs(red_seed, rank, horizon, n)
        recon2 = reconstruct_attack_inputs(red_seed, rank, horizon, n)  # determinism
        if recon != ref or recon != recon2:
            ok = False
            print(f"[FAIL] reconstruct seed={red_seed} rank={rank} horizon={horizon}")
            for i, (a, b) in enumerate(zip(recon, ref)):
                if a != b:
                    print(f"  episode {i}: recon={a!r} ref={b!r}")
                    break
        else:
            q = recon[0]
            print(
                f"[OK]   reconstruct seed={red_seed} rank={rank} horizon={horizon} "
                f"n={n} ep0 idx={q['question_idx']} q[:50]={q['question'][:50]!r}"
            )
    return ok


def check_replay_fidelity() -> bool:
    """Eval plan must replay manifest attack inputs verbatim, pairing-invariant."""
    red_seed, horizon, n_episodes = 10, 5, 20
    n_attack = n_episodes // 2
    attack_inputs = reconstruct_attack_inputs(red_seed, 0, horizon, n_attack)
    manifest = build_manifest_dict(
        red_seed, horizon, attack_inputs, {"mode": "reconstructed", "ranks": [0]}
    )
    inputs = validate_manifest(manifest, horizon, n_attack)

    plan = _build_match_train_episode_plan(inputs, n_episodes, red_seed)
    plan2 = _build_match_train_episode_plan(inputs, n_episodes, red_seed)

    ok = plan == plan2  # deterministic / pairing-invariant
    if not ok:
        print("[FAIL] replay: plan not deterministic across builds")

    attack_cfgs = [c for c in plan if c[1] == "attack"]
    if len(attack_cfgs) != n_attack:
        ok = False
        print(f"[FAIL] replay: {len(attack_cfgs)} attack cfgs, expected {n_attack}")
    for i, (cfg, a) in enumerate(zip(attack_cfgs, inputs)):
        # cfg = (ep_idx, "attack", None, question, question_idx, None, strategies)
        if (
            cfg[3] != a["question"]
            or cfg[4] != int(a["question_idx"])
            or cfg[6] != a["strategies"]
        ):
            ok = False
            print(f"[FAIL] replay attack {i}: plan cfg != manifest input")
            break
    if ok:
        print(
            f"[OK]   replay: {n_attack} attack episodes match manifest in order; "
            f"plan deterministic & pairing-invariant"
        )
    return ok


def check_prompt_construction_identity() -> bool:
    """Format-level identity (the B1/B2/B3 fix): the eval red prompt is built from
    the SAME helpers training uses, so turn-1 — deterministic, no stochastic blue
    text — is byte-equal to training's MAS template. Also guards the shared
    build_agent_prompt and cap_display primitives.
    """
    ok = True

    # build_agent_prompt must reproduce the MAS generation prompt exactly.
    got = build_agent_prompt("STRAT", "OBS", "sql_agent")
    want = "STRATOBS<|im_start|>sql_agent: "
    if got != want:
        ok = False
        print(f"[FAIL] build_agent_prompt: {got!r} != {want!r}")

    # cap_display: <=cap unchanged, >cap truncated with the marker.
    at_cap = "x" * DISPLAY_CAP
    over_cap = "y" * (DISPLAY_CAP + 50)
    if cap_display(at_cap) != at_cap:
        ok = False
        print("[FAIL] cap_display truncated a <=cap string")
    if cap_display(over_cap) != over_cap[:DISPLAY_CAP] + "... [truncated]":
        ok = False
        print("[FAIL] cap_display >cap result mismatch")

    # Turn-1 red prompt built the eval way == the literal training template, for a
    # real manifest attack input. (Turn >=2 embeds stochastic blue text, so only
    # the format/cap — not the full runtime trace — is checkable.)
    a = reconstruct_attack_inputs(10, 0, 5, 1)[0]
    strategy = a["strategies"][0]
    eval_prompt = build_agent_prompt(strategy, redteam_initial_obs(a["question"]), "sql_agent")
    train_prompt = (
        strategy
        + f"<|im_start|>question: {a['question']}<|im_end|>\n"
        + "<|im_start|>sql_agent: "
    )
    if eval_prompt != train_prompt:
        ok = False
        print(f"[FAIL] turn-1 red prompt:\n  eval ={eval_prompt!r}\n  train={train_prompt!r}")

    if ok:
        print(
            "[OK]   prompt construction: build_agent_prompt + cap_display + turn-1 "
            "red prompt byte-identical to training"
        )
    return ok


def check_against_run(selfplay_dir) -> bool:
    """Integration check: env-logged streams == reconstruction for a real run."""
    with open(os.path.join(selfplay_dir, "summary.json")) as f:
        summary = json.load(f)
    red_seed = int(summary["red_seed"])
    horizon = int(summary["horizon"])
    paths, ranks = load_logged_attack_inputs(selfplay_dir)
    if not ranks:
        print(f"[SKIP] no redteam_prompts.jsonl logs under {selfplay_dir}")
        return True
    n = cross_check_ranks(ranks, red_seed, horizon)
    per_rank = {r: len(v) for r, v in sorted(ranks.items())}
    print(
        f"[OK]   run {selfplay_dir}: {n} env-logged episodes across ranks "
        f"{per_rank} == reconstruction ({len(paths)} log file(s))"
    )
    return True


def check_manifest_plan_against_logged(
    selfplay_dir, manifest_path=None, episodes=160
) -> bool:
    """End-to-end on REAL run data, using the actual logged strategies (not a
    reconstruction): env-logged prompts == manifest == the single pairing-invariant
    episode plan cross-eval replays == the turn-1 string eval feeds the red policy.

    check_against_run() proves logged == reconstruction; the other checks prove
    reconstruction -> plan -> prompt. This closes the remaining link DIRECTLY:
    the logged training prompts themselves flow, unchanged, into the cross-eval
    planner and out as the turn-1 wire string — so the chain never relies on the
    reconstruction being a faithful stand-in for the logs.
    """
    with open(os.path.join(selfplay_dir, "summary.json")) as f:
        summary = json.load(f)
    red_seed = int(summary["red_seed"])
    horizon = int(summary["horizon"])

    paths, ranks = load_logged_attack_inputs(selfplay_dir)
    if not ranks:
        print(f"[SKIP] no redteam_prompts.jsonl logs under {selfplay_dir}")
        return True
    logged = flatten_ranks(ranks)  # the exact order build_redteam_manifest writes

    if manifest_path and os.path.isfile(manifest_path):
        manifest = load_manifest(manifest_path)
        src = f"on-disk manifest {manifest_path}"
    else:
        manifest = build_manifest_dict(
            red_seed, horizon, logged,
            {"mode": "logged-all-ranks", "ranks": sorted(ranks)},
        )
        src = "in-memory manifest (built from logs)"

    # Clamp eval size to what the manifest can serve (manifest is normally larger).
    episodes = min(episodes, 2 * len(manifest["attack_inputs"]))
    n_attack = episodes // 2
    attack_inputs = validate_manifest(manifest, horizon, n_attack)

    ok = True
    # (a) manifest attack inputs == flattened logged records, byte-for-byte.
    for i in range(n_attack):
        a, lg = attack_inputs[i], logged[i]
        if (
            int(a["question_idx"]) != int(lg["question_idx"])
            or a["question"] != lg["question"]
            or list(a["strategies"]) != list(lg["strategies"])
        ):
            ok = False
            print(f"[FAIL] manifest attack input {i} != logged record {i}")
            break

    # (b) the episode plan cross-eval replays — seeded from red_seed alone, so it is
    # identical for EVERY (red_iter, blue_iter) pairing.
    plan = _build_match_train_episode_plan(attack_inputs, episodes, red_seed)
    if plan != _build_match_train_episode_plan(attack_inputs, episodes, red_seed):
        ok = False
        print("[FAIL] episode plan is not deterministic / pairing-invariant")

    # (c) every attack config carries the logged (question, idx, strategies); the
    # turn-1 red prompt built from it equals training's exact wire format.
    attack_cfgs = [c for c in plan if c[1] == "attack"]
    if len(attack_cfgs) != n_attack:
        ok = False
        print(f"[FAIL] plan has {len(attack_cfgs)} attack cfgs, expected {n_attack}")
    for i, cfg in enumerate(attack_cfgs):
        # cfg = (ep_idx, "attack", None, question, question_idx, None, strategies)
        lg = logged[i]
        if (
            cfg[3] != lg["question"]
            or int(cfg[4]) != int(lg["question_idx"])
            or cfg[6] != list(lg["strategies"])
        ):
            ok = False
            print(f"[FAIL] plan attack {i} config != logged record {i}")
            break
        eval_turn1 = build_agent_prompt(
            cfg[6][0], redteam_initial_obs(cfg[3]), "sql_agent"
        )
        train_turn1 = (
            lg["strategies"][0]
            + f"<|im_start|>question: {lg['question']}<|im_end|>\n"
            + "<|im_start|>sql_agent: "
        )
        if eval_turn1 != train_turn1:
            ok = False
            print(f"[FAIL] turn-1 red prompt for plan attack {i} != training wire format")
            break

    if ok:
        print(
            f"[OK]   end-to-end: {n_attack} attack episodes — logged == {src} == "
            f"pairing-invariant plan; turn-1 red prompts byte-identical to training "
            f"({len(paths)} log file(s), ranks {sorted(ranks)})"
        )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--selfplay-dir",
        default=None,
        help="If given, also check env-logged prompts == reconstruction, and run the "
        "end-to-end logged -> manifest -> plan -> turn-1 prompt check.",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="Optional path to a built redteam_eval_manifest.json. If given, the "
        "end-to-end check ties the ON-DISK manifest to the logs; otherwise it builds "
        "an in-memory manifest from the logs.",
    )
    args = ap.parse_args()

    ok = True
    print("== Reconstruction fidelity (env RNG order, NO constructor draw) ==")
    ok &= check_reconstruction_fidelity()
    print("\n== Replay fidelity (manifest -> pairing-invariant plan) ==")
    ok &= check_replay_fidelity()
    print("\n== Prompt construction identity (shared helpers; turn-1 byte-equal) ==")
    ok &= check_prompt_construction_identity()
    if args.selfplay_dir:
        print("\n== Integration: env-logged == reconstruction ==")
        ok &= check_against_run(args.selfplay_dir)
        print("\n== End-to-end: logged == manifest == pairing-invariant plan == turn-1 prompt ==")
        ok &= check_manifest_plan_against_logged(args.selfplay_dir, args.manifest)

    if not ok:
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
