#!/bin/bash
# Agentic AI Security: Self-Play Orchestrator (redesign).
#
# Key changes vs. legacy script:
#   * Coach + SIL removed entirely (no --coach-* flags, no /tmp/vllm_coach_registry.json).
#   * Symmetric env-step budget: red and blue both get --num-env-steps per phase.
#   * No early-stop / convergence — termination is solely budget-driven.
#   * Blue trains against the *full history* of red checkpoints via vLLM
#     multi-LoRA. The pool is tracked in results-{ID}/red_lora_registry.json
#     and rebuilt per iteration.
#   * Per-cell ablation surface: --vanilla-size, --bordercase-size, --honeypot-type.
#   * --resume by cell directory; iteration/round inferred where possible.
#
# Defaults (per the redesign plan):
#   - num_env_steps_per_phase = 1600 (≈ 24 GPU-hr per phase × 2 phases × 2 iters
#                                    ≈ 48 GPU-hr per cell on 2 A100s.)
#   - num_iterations          = 2
#   - horizon                 = 5
#   - vanilla_size            = unset (env defaults to full plain pool)
#   - bordercase_size         = unset (env defaults to full adversarial pool)
#   - honeypot_type           = rowcol
#
# Usage:
#   ./run_selfplay.sh --num-iterations 2 --num-env-steps 1600 --horizon 5 \
#                     --vanilla-size 120 --bordercase-size 20 --honeypot-type rowcol \
#                     --redteam-gpu 0 --blueteam-gpu 1 --replicate-seed 0
#   (default --base-model is Snowflake/Arctic-Text2SQL-R1-7B)
#
# Resume:
#   ./run_selfplay.sh --continue results-<ID> --continue-iteration 2 --continue-round red

set -e
set -o pipefail

echo "========================================"
echo "Agentic AI Security: Self-Play Orchestrator (redesign)"
echo "========================================"

# --- Defaults ---
BASE_MODEL="Snowflake/Arctic-Text2SQL-R1-7B"
REDTEAM_BASE_MODEL=""    # red (attacker) base; defaults to BASE_MODEL (homogeneous)
BLUETEAM_BASE_MODEL=""   # blue (defender) base; defaults to BASE_MODEL (homogeneous)
NUM_ITERATIONS=2
NUM_ENV_STEPS=1600
HORIZON=5
REDTEAM_GPU=""
BLUETEAM_GPU=""
VANILLA_SIZE=""
BORDERCASE_SIZE=""
HONEYPOT_TYPE=""
SCORING_MODE=""
OPPONENT_SAMPLER_SEED=""
REPLICATE_SEED=0
RED_SEED=""
BLUE_SEED=""
RESULTS_ID_OVERRIDE=""

# --- Continue defaults ---
CONTINUE_DIR=""
CONTINUE_ITER=""
CONTINUE_ROUND=""
RESUME_BLUE_CHECKPOINT=false

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base-model) BASE_MODEL="$2"; shift ;;
        --redteam-base-model) REDTEAM_BASE_MODEL="$2"; shift ;;
        --blueteam-base-model) BLUETEAM_BASE_MODEL="$2"; shift ;;
        --num-iterations) NUM_ITERATIONS="$2"; shift ;;
        --num-env-steps) NUM_ENV_STEPS="$2"; shift ;;
        --horizon) HORIZON="$2"; shift ;;
        --redteam-gpu) REDTEAM_GPU="$2"; shift ;;
        --blueteam-gpu) BLUETEAM_GPU="$2"; shift ;;
        --vanilla-size) VANILLA_SIZE="$2"; shift ;;
        --bordercase-size) BORDERCASE_SIZE="$2"; shift ;;
        --honeypot-type) HONEYPOT_TYPE="$2"; shift ;;
        --scoring-mode) SCORING_MODE="$2"; shift ;;
        --opponent-sampler-seed) OPPONENT_SAMPLER_SEED="$2"; shift ;;
        --replicate-seed) REPLICATE_SEED="$2"; shift ;;
        --red-seed) RED_SEED="$2"; shift ;;
        --blue-seed) BLUE_SEED="$2"; shift ;;
        --results-id) RESULTS_ID_OVERRIDE="$2"; shift ;;
        --continue) CONTINUE_DIR="$2"; shift ;;
        --continue-iteration) CONTINUE_ITER="$2"; shift ;;
        --continue-round) CONTINUE_ROUND="$2"; shift ;;
        --resume-blue-checkpoint) RESUME_BLUE_CHECKPOINT=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate GPU locators (required; no silent fallback to 0/1) ------------
# Every GPU process below is pinned via CUDA_VISIBLE_DEVICES to exactly these
# two indices: each run_training.sh phase puts the trainer on one and the
# actor vLLM fleet on the other. An unset/duplicate index would silently land
# on another user's card, so fail fast here.
if [[ -z "$REDTEAM_GPU" || -z "$BLUETEAM_GPU" ]]; then
    echo "ERROR: --redteam-gpu and --blueteam-gpu are both required." >&2
    exit 1
fi
if ! [[ "$REDTEAM_GPU" =~ ^[0-9]+$ && "$BLUETEAM_GPU" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --redteam-gpu / --blueteam-gpu must be non-negative integers (got $REDTEAM_GPU / $BLUETEAM_GPU)." >&2
    exit 1
fi
if [[ "$REDTEAM_GPU" == "$BLUETEAM_GPU" ]]; then
    echo "ERROR: --redteam-gpu and --blueteam-gpu must differ (got $REDTEAM_GPU / $BLUETEAM_GPU)." >&2
    exit 1
fi

# --- Validate continue flags ---
if [[ -n "$CONTINUE_DIR" ]]; then
    [[ ! -d "$CONTINUE_DIR" ]] && { echo "ERROR: Continue dir '$CONTINUE_DIR' does not exist."; exit 1; }
    [[ -z "$CONTINUE_ITER" ]] && { echo "ERROR: --continue-iteration required."; exit 1; }
    [[ -z "$CONTINUE_ROUND" ]] && { echo "ERROR: --continue-round required."; exit 1; }
    [[ "$CONTINUE_ROUND" != "red" && "$CONTINUE_ROUND" != "blue" ]] && \
        { echo "ERROR: --continue-round must be 'red' or 'blue'."; exit 1; }
    if ! [[ "$CONTINUE_ITER" =~ ^[0-9]+$ ]] || [[ "$CONTINUE_ITER" -lt 1 ]]; then
        echo "ERROR: --continue-iteration must be a positive integer."; exit 1
    fi
    if [[ "$CONTINUE_ITER" -gt "$NUM_ITERATIONS" ]]; then
        echo "ERROR: --continue-iteration ($CONTINUE_ITER) exceeds --num-iterations ($NUM_ITERATIONS)."; exit 1
    fi
elif [[ -n "$CONTINUE_ITER" || -n "$CONTINUE_ROUND" ]]; then
    echo "ERROR: --continue-iteration and --continue-round require --continue <result_dir>."; exit 1
fi

if [[ "$RESUME_BLUE_CHECKPOINT" == "true" ]]; then
    if [[ -z "$CONTINUE_DIR" || "$CONTINUE_ROUND" != "blue" ]]; then
        echo "ERROR: --resume-blue-checkpoint requires --continue and --continue-round blue."; exit 1
    fi
fi

find_latest_checkpoint() {
    local dir="$1"
    find "$dir" -name "sql_agent" -type d 2>/dev/null | sort -V | tail -n 1
}

# --- Resolve + export honeypot arm (fail-fast at python import depends on this) ---
# train_sql.py also sets os.environ["HONEYPOT_TYPE"] from --honeypot-type, but
# exporting here in the shell is belt-and-braces: it guarantees every python
# process this script launches (including any future top-level imports of
# redteam_sql_env) sees the correct arm even before argparse runs.
HONEYPOT_TYPE="${HONEYPOT_TYPE:-rowcol}"
case "$HONEYPOT_TYPE" in
    rowcol|row|col) ;;
    *) echo "ERROR: --honeypot-type must be rowcol|row|col (got '$HONEYPOT_TYPE')." >&2; exit 1 ;;
esac
export HONEYPOT_TYPE

# --- Resolve + export the honeypot scoring mode (PVR/BRR detector). ----------
# Defaults to execution-grounded scoring for BOTH the red and blue halves; the
# shim (redteam_sql_env.detect_honeypot_access) reads AAS_SCORING_MODE at call
# time, so exporting here guarantees every python process this script launches
# scores with the corrected detector. train_sql.py also sets it from
# --scoring-mode (belt-and-braces). 'stored' is rejected (metrics-only concept).
SCORING_MODE="${SCORING_MODE:-execution}"
case "$SCORING_MODE" in
    legacy|projection|execution) ;;
    *) echo "ERROR: --scoring-mode must be legacy|projection|execution (got '$SCORING_MODE')." >&2; exit 1 ;;
esac
export AAS_SCORING_MODE="$SCORING_MODE"

# Heterogeneous red/blue: each team may use a different base model. Both default
# to --base-model, so single-model callers are unchanged.
REDTEAM_BASE_MODEL="${REDTEAM_BASE_MODEL:-$BASE_MODEL}"
BLUETEAM_BASE_MODEL="${BLUETEAM_BASE_MODEL:-$BASE_MODEL}"

echo "Base model:      $BASE_MODEL"
echo "Red base model:  $REDTEAM_BASE_MODEL"
echo "Blue base model: $BLUETEAM_BASE_MODEL"
echo "Num iterations:  $NUM_ITERATIONS"
echo "Num env steps:   $NUM_ENV_STEPS (per phase, applied symmetrically to red and blue)"
echo "Horizon:         $HORIZON"
echo "GPU layout:      redteam=GPU${REDTEAM_GPU}  blueteam=GPU${BLUETEAM_GPU}"
echo "Honeypot type:   ${HONEYPOT_TYPE}"
echo "Scoring mode:    ${SCORING_MODE} (PVR + BRR detector, both halves)"
[[ -n "$VANILLA_SIZE" ]]    && echo "Vanilla size:    $VANILLA_SIZE"
[[ -n "$BORDERCASE_SIZE" ]] && echo "Bordercase size: $BORDERCASE_SIZE"

# Each run_training.sh phase reaps its OWN setsid'd vLLM process group from its
# own EXIT/INT/TERM/HUP trap, so we never sweep vLLM by name (a broad
# `pkill -f vllm.entrypoints` would kill concurrent sibling runs on this shared
# host). But that delegation only holds while the child actually runs its trap.
# Two signal paths bypass it and orphan the fleet, so we handle them here:
#   * A targeted `kill <selfplay_pid>` (e.g. scancel) hits THIS script only, not
#     the foreground child — so we forward the teardown to the running phase.
#   * SIGHUP on SSH disconnect kills an un-trapped bash WITHOUT running its EXIT
#     trap, while the setsid'd fleet (own session) ignores the terminal hangup —
#     so we trap HUP too and reap through the child.
# Beyond that, our only direct responsibility is the ephemeral Postgres we own.
source "$(pwd)/script/pg_ephemeral.sh"
OWNS_DB=0
CHILD_TRAINING_PID=""   # PID of the in-flight ./run_training.sh phase, if any
_selfplay_cleanup() {
    # Preserve the real exit status — bash exits with the EXIT trap's last
    # command status, and a trailing `[[...]] &&` that is false would otherwise
    # mask a successful run as exit 1.
    local _rc=$?
    # Tear down an in-flight training phase so its setsid'd vLLM fleet can't
    # orphan. The child reaps its own fleet from its own trap, so we just signal
    # it and wait for that teardown to finish. Idempotent: the kill -0 guard
    # makes a second pass (signal handler + EXIT trap) a no-op.
    if [[ -n "$CHILD_TRAINING_PID" ]] && kill -0 "$CHILD_TRAINING_PID" 2>/dev/null; then
        echo "[run_selfplay] Forwarding teardown to in-flight training phase (PID ${CHILD_TRAINING_PID})..."
        kill -TERM "$CHILD_TRAINING_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$CHILD_TRAINING_PID" 2>/dev/null || break
            sleep 2
        done
        kill -KILL "$CHILD_TRAINING_PID" 2>/dev/null || true
    fi
    [[ "$OWNS_DB" == "1" ]] && pg_ephemeral_stop
    return $_rc
}
trap _selfplay_cleanup EXIT
trap '_selfplay_cleanup; exit 130' INT
trap '_selfplay_cleanup; exit 143' TERM
trap '_selfplay_cleanup; exit 129' HUP

BLUE_LATEST_CKPT=""
RED_LATEST_CKPT=""

if [[ -n "$CONTINUE_DIR" ]]; then
    CONTINUE_DIR="${CONTINUE_DIR%/}"
    SELFPLAY_ID="$(basename "$CONTINUE_DIR")"
    SELFPLAY_ID="${SELFPLAY_ID#results-}"
    echo "Continuing run with Selfplay ID: ${SELFPLAY_ID}"
    echo "  Resume at iteration ${CONTINUE_ITER}, round ${CONTINUE_ROUND}"

    # Recover checkpoints from the iteration before the continue point
    if [[ "$CONTINUE_ITER" -gt 1 ]]; then
        PREV_BLUE_DIR="results-${SELFPLAY_ID}/iter_$((CONTINUE_ITER - 1))/blueteam"
        if [[ ! -d "$PREV_BLUE_DIR" ]]; then
            echo "ERROR: blue dir missing — ${PREV_BLUE_DIR}"; exit 1
        fi
        BLUE_LATEST_CKPT=$(find_latest_checkpoint "${PREV_BLUE_DIR}")
        [[ -z "$BLUE_LATEST_CKPT" ]] && { echo "ERROR: no sql_agent checkpoint in ${PREV_BLUE_DIR}"; exit 1; }
        BLUE_LATEST_CKPT=$(realpath "${BLUE_LATEST_CKPT}")
        echo "Recovered Blue LoRA from previous iteration: ${BLUE_LATEST_CKPT}"

        PREV_RED_DIR="results-${SELFPLAY_ID}/iter_$((CONTINUE_ITER - 1))/redteam"
        if [[ -d "$PREV_RED_DIR" ]]; then
            RED_LATEST_CKPT=$(find_latest_checkpoint "${PREV_RED_DIR}")
            if [[ -n "$RED_LATEST_CKPT" ]]; then
                RED_LATEST_CKPT=$(realpath "${RED_LATEST_CKPT}")
                echo "Recovered Red LoRA from previous iteration: ${RED_LATEST_CKPT}"
            fi
        fi
    fi
else
    SELFPLAY_ID="${RESULTS_ID_OVERRIDE:-$(date +%Y%m%d-%H%M)-$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 5 || true)}"
fi

echo "Selfplay Run ID: ${SELFPLAY_ID}"

# --- Per-run namespace + ephemeral Postgres --------------------------------
# The whole self-play run (all iterations + both teams) shares one ephemeral
# Postgres container and one runtime dir, torn down when this script exits.
export AAS_RUN_ID="$SELFPLAY_ID"
export AAS_RUN_DIR="$(pwd)/.runtime/${AAS_RUN_ID}"
mkdir -p "$AAS_RUN_DIR"
if [[ -n "$AAS_PG_CONTAINER" ]]; then
    pg_ephemeral_require
    echo "[run_selfplay] Reusing ephemeral Postgres from parent: ${AAS_PG_CONTAINER}"
else
    # Mark ownership BEFORE bring-up so a signal mid-startup still triggers
    # teardown (pg_ephemeral_start exports AAS_PG_CONTAINER as its first step).
    OWNS_DB=1
    pg_ephemeral_start
fi

CELL_ROOT="results-${SELFPLAY_ID}"
mkdir -p "$CELL_ROOT"

RED_LORA_REGISTRY="${CELL_ROOT}/red_lora_registry.json"

# Initialize registry (or rebuild from existing iteration dirs on resume).
init_red_registry() {
    python3 - <<PY
import json, os, glob, re
root = "${CELL_ROOT}"
entries = []
for path in sorted(glob.glob(os.path.join(root, "iter_*"))):
    m = re.match(r".*/iter_(\d+)$", path)
    if not m:
        continue
    iter_n = int(m.group(1))
    red_dir = os.path.join(path, "redteam")
    if not os.path.isdir(red_dir) or not os.path.isfile(os.path.join(red_dir, ".success")):
        continue
    candidates = []
    for root_, dirs, _ in os.walk(red_dir):
        if os.path.basename(root_) == "sql_agent":
            candidates.append(root_)
    if not candidates:
        continue
    latest = sorted(candidates)[-1]
    entries.append({"iter": iter_n, "name": f"red_iter_{iter_n}", "path": os.path.realpath(latest)})
out = {"selfplay_id": "${SELFPLAY_ID}", "entries": entries}
with open(os.path.join(root, "red_lora_registry.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"[registry] {len(entries)} red entries: {[e['name'] for e in entries]}")
PY
}

append_red_to_registry() {
    local iter_n="$1"
    local ckpt_path="$2"
    python3 - <<PY
import json, os
reg_path = "${RED_LORA_REGISTRY}"
data = json.load(open(reg_path)) if os.path.exists(reg_path) else {"selfplay_id": "${SELFPLAY_ID}", "entries": []}
entries = [e for e in data.get("entries", []) if e.get("iter") != ${iter_n}]
entries.append({"iter": ${iter_n}, "name": "red_iter_${iter_n}", "path": "${ckpt_path}"})
entries.sort(key=lambda e: e["iter"])
data["entries"] = entries
with open(reg_path, "w") as f:
    json.dump(data, f, indent=2)
print(f"[registry] now {len(entries)} red entries (added/updated red_iter_${iter_n})")
PY
}

init_red_registry

if [[ -z "$OPPONENT_SAMPLER_SEED" ]]; then
    # Pure function of REPLICATE_SEED so two invocations with the same
    # --replicate-seed pick the same opponent/benign draws. Do NOT mix in
    # SELFPLAY_ID — it carries a /dev/urandom suffix and would re-randomize
    # the sampler on every re-run.
    OPPONENT_SAMPLER_SEED=$(( (1000003 * REPLICATE_SEED + 1234) & 0xFFFFFFFF ))
fi
[[ -z "$RED_SEED" ]]  && RED_SEED=$(( 10 + REPLICATE_SEED * 1000 ))
[[ -z "$BLUE_SEED" ]] && BLUE_SEED=$(( 12 + REPLICATE_SEED * 1000 ))
echo "Replicate seed:        $REPLICATE_SEED"
echo "Red seed:              $RED_SEED"
echo "Blue seed:             $BLUE_SEED"
echo "Opponent sampler seed: $OPPONENT_SAMPLER_SEED"

SUMMARY_PATH="${CELL_ROOT}/summary.json"
python3 - <<PY
import json
out = {
    "selfplay_id": "${SELFPLAY_ID}",
    "base_model": "${BASE_MODEL}",
    "redteam_base_model": "${REDTEAM_BASE_MODEL}",
    "blueteam_base_model": "${BLUETEAM_BASE_MODEL}",
    "num_iterations": ${NUM_ITERATIONS},
    "num_env_steps": ${NUM_ENV_STEPS},
    "horizon": ${HORIZON},
    "redteam_gpu": ${REDTEAM_GPU},
    "blueteam_gpu": ${BLUETEAM_GPU},
    "vanilla_size": ${VANILLA_SIZE:-null},
    "bordercase_size": ${BORDERCASE_SIZE:-null},
    "honeypot_type": "${HONEYPOT_TYPE}",
    "scoring_mode": "${SCORING_MODE}",
    "replicate_seed": ${REPLICATE_SEED},
    "red_seed": ${RED_SEED},
    "blue_seed": ${BLUE_SEED},
    "opponent_sampler_seed": ${OPPONENT_SAMPLER_SEED},
}
with open("${SUMMARY_PATH}", "w") as f:
    json.dump(out, f, indent=2)
PY

# Build per-phase common training args.
ABLATION_ARGS=()
[[ -n "$VANILLA_SIZE" ]]    && ABLATION_ARGS+=(--vanilla-size "$VANILLA_SIZE")
[[ -n "$BORDERCASE_SIZE" ]] && ABLATION_ARGS+=(--bordercase-size "$BORDERCASE_SIZE")
ABLATION_ARGS+=(--honeypot-type "$HONEYPOT_TYPE")
ABLATION_ARGS+=(--scoring-mode "$SCORING_MODE")
ABLATION_ARGS+=(--opponent-sampler-seed "$OPPONENT_SAMPLER_SEED")
ABLATION_ARGS+=(--num-env-steps "$NUM_ENV_STEPS")

for ITER in $(seq 1 $NUM_ITERATIONS); do
    if [[ -n "$CONTINUE_DIR" && "$ITER" -lt "$CONTINUE_ITER" ]]; then
        echo ""
        echo "[Iter ${ITER}/${NUM_ITERATIONS}] Skipping (already completed)."
        continue
    fi

    echo ""
    echo "========================================"
    echo "Self-Play Iteration ${ITER}/${NUM_ITERATIONS}"
    echo "========================================"

    ITER_ID="${SELFPLAY_ID}/iter_${ITER}"
    echo "Iteration Path: results-${ITER_ID}/"

    # --- Red Phase ---
    SKIP_RED=false
    if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" && "$CONTINUE_ROUND" == "blue" ]]; then
        SKIP_RED=true
    fi

    if [[ "$SKIP_RED" == "true" ]]; then
        RED_DIR="results-${ITER_ID}/redteam"
        [[ ! -d "$RED_DIR" ]] && { echo "ERROR: red dir missing for skip-red: $RED_DIR"; exit 1; }
        RED_LATEST_CKPT=$(find_latest_checkpoint "${RED_DIR}")
        [[ -z "$RED_LATEST_CKPT" ]] && { echo "ERROR: no sql_agent checkpoint in ${RED_DIR}"; exit 1; }
        RED_LATEST_CKPT=$(realpath "${RED_LATEST_CKPT}")
        echo "[Iter ${ITER}] Skipping Red Team. Using existing Red LoRA: ${RED_LATEST_CKPT}"
        append_red_to_registry "$ITER" "$RED_LATEST_CKPT"
    else
        if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" && "$CONTINUE_ROUND" == "red" ]]; then
            STALE_RED="results-${ITER_ID}/redteam"
            if [[ -d "$STALE_RED" ]]; then
                echo "[Iter ${ITER}] Removing stale partial red results: ${STALE_RED}"
                rm -rf "$STALE_RED"
            fi
        fi

        # Red phase: student = red base (trained), opponent/victim = blue base.
        RED_TRAIN_ARGS=(
            --target redteam
            --results-id "${ITER_ID}"
            --base-model "$BASE_MODEL"
            --student-base-model "$REDTEAM_BASE_MODEL"
            --opponent-base-model "$BLUETEAM_BASE_MODEL"
            --actor-gpu "$BLUETEAM_GPU"
            --training-gpu "$REDTEAM_GPU"
            --horizon "$HORIZON"
            --seed "$RED_SEED"
            "${ABLATION_ARGS[@]}"
        )
        if [[ -n "$BLUE_LATEST_CKPT" ]]; then
            RED_TRAIN_ARGS+=(--opponent-lora "${BLUE_LATEST_CKPT}")
            echo "Using Blue LoRA from previous iteration: ${BLUE_LATEST_CKPT}"
        fi
        if [[ -n "$RED_LATEST_CKPT" ]]; then
            RED_TRAIN_ARGS+=(--student-lora "${RED_LATEST_CKPT}")
            echo "Continuing Red LoRA from previous iteration: ${RED_LATEST_CKPT}"
        fi

        echo "[Iter ${ITER}] Training Red Team (ID: ${ITER_ID})..."
        # Background + tracked wait (not `if ! ...`): exposes the child PID to
        # the cleanup trap so a targeted kill / SIGHUP forwards teardown to the
        # phase instead of abandoning it (and its setsid'd fleet) as an orphan.
        ./run_training.sh "${RED_TRAIN_ARGS[@]}" &
        CHILD_TRAINING_PID=$!
        RED_RC=0; wait "$CHILD_TRAINING_PID" || RED_RC=$?
        CHILD_TRAINING_PID=""
        if [[ "$RED_RC" -ne 0 ]]; then
            echo "Red Team training failed on iteration ${ITER}!"; exit 1
        fi

        RED_DIR="results-${ITER_ID}/redteam"
        if [ ! -f "${RED_DIR}/.success" ]; then
            echo "ERROR: Red Team dir missing .success marker: ${RED_DIR}"; exit 1
        fi

        RED_LATEST_CKPT=$(find_latest_checkpoint "${RED_DIR}")
        [[ -z "$RED_LATEST_CKPT" ]] && { echo "ERROR: no sql_agent checkpoint in ${RED_DIR}"; exit 1; }
        RED_LATEST_CKPT=$(realpath "${RED_LATEST_CKPT}")
        echo "Using Red LoRA: ${RED_LATEST_CKPT}"

        append_red_to_registry "$ITER" "$RED_LATEST_CKPT"
        # Red-phase run_training.sh already reaped its own vLLM on exit.
    fi

    # --- Blue Phase ---
    BLUE_RESUME_RUN_DIR=""
    if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" ]]; then
        STALE_BLUE="results-${ITER_ID}/blueteam"
        if [[ "$RESUME_BLUE_CHECKPOINT" == "true" ]]; then
            [[ ! -d "$STALE_BLUE" ]] && { echo "ERROR: --resume-blue-checkpoint set but no blueteam dir at ${STALE_BLUE}"; exit 1; }
            BLUE_RESUME_RUN_DIR=$(find "$STALE_BLUE" -type f -name training_state.json -printf '%h\n' | head -n 1)
            [[ -z "$BLUE_RESUME_RUN_DIR" ]] && { echo "ERROR: No training_state.json under ${STALE_BLUE}"; exit 1; }
            BLUE_RESUME_RUN_DIR=$(realpath "$BLUE_RESUME_RUN_DIR")
            echo "[Iter ${ITER}] Resuming crashed blue run from: ${BLUE_RESUME_RUN_DIR}"
        elif [[ -d "$STALE_BLUE" ]]; then
            echo "[Iter ${ITER}] Removing stale partial blue results: ${STALE_BLUE}"
            rm -rf "$STALE_BLUE"
        fi
    fi

    # Blue phase: student = blue base (trained), opponent/attacker = red base.
    BLUE_TRAIN_ARGS=(
        --target blueteam
        --results-id "${ITER_ID}"
        --base-model "$BASE_MODEL"
        --student-base-model "$BLUETEAM_BASE_MODEL"
        --opponent-base-model "$REDTEAM_BASE_MODEL"
        --opponent-lora-pool "$RED_LORA_REGISTRY"
        --actor-gpu "$REDTEAM_GPU"
        --training-gpu "$BLUETEAM_GPU"
        --horizon "$HORIZON"
        --seed "$BLUE_SEED"
        "${ABLATION_ARGS[@]}"
    )
    if [[ -n "$BLUE_RESUME_RUN_DIR" ]]; then
        BLUE_TRAIN_ARGS+=(--resume-run-dir "${BLUE_RESUME_RUN_DIR}")
    elif [[ -n "$BLUE_LATEST_CKPT" ]]; then
        BLUE_TRAIN_ARGS+=(--student-lora "${BLUE_LATEST_CKPT}")
        echo "Continuing Blue LoRA from previous iteration: ${BLUE_LATEST_CKPT}"
    fi

    echo "[Iter ${ITER}] Training Blue Team against red pool (ID: ${ITER_ID})..."
    # Background + tracked wait (see red phase): keeps the child PID visible to
    # the cleanup trap so an interrupt forwards teardown to the phase.
    ./run_training.sh "${BLUE_TRAIN_ARGS[@]}" &
    CHILD_TRAINING_PID=$!
    BLUE_RC=0; wait "$CHILD_TRAINING_PID" || BLUE_RC=$?
    CHILD_TRAINING_PID=""
    if [[ "$BLUE_RC" -ne 0 ]]; then
        echo "Blue Team training failed on iteration ${ITER}!"; exit 1
    fi

    BLUE_DIR="results-${ITER_ID}/blueteam"
    [ ! -f "${BLUE_DIR}/.success" ] && { echo "ERROR: Blue Team dir missing .success marker: ${BLUE_DIR}"; exit 1; }
    echo "[Iter ${ITER}] Complete. Results at results-${ITER_ID}/"

    BLUE_LATEST_CKPT=$(find_latest_checkpoint "${BLUE_DIR}")
    [[ -z "$BLUE_LATEST_CKPT" ]] && { echo "ERROR: no sql_agent checkpoint in ${BLUE_DIR}"; exit 1; }
    BLUE_LATEST_CKPT=$(realpath "${BLUE_LATEST_CKPT}")
    echo "Saved Blue LoRA for next iteration: ${BLUE_LATEST_CKPT}"
    # Blue-phase run_training.sh already reaped its own vLLM on exit.

    if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" ]]; then
        CONTINUE_DIR=""
    fi
done

# --- Assemble the redteam eval-prompt manifest -----------------------------
# Persist the EXACT (question, per-turn strategies) sequence training drew so
# cross-eval (--match-train-seeds) replays byte-identical prompts across every
# pairing. Built from the env-logged redteam_prompts.jsonl and cross-checked
# against an independent RNG reconstruction (fails loudly on any drift).
echo ""
echo "[run_selfplay] Assembling redteam eval-prompt manifest..."
if ! python3 util/build_redteam_manifest.py --selfplay-dir "${CELL_ROOT}"; then
    echo "ERROR: failed to build redteam eval-prompt manifest for ${CELL_ROOT}"
    exit 1
fi
MANIFEST_PATH="${CELL_ROOT}/redteam_eval_manifest.json"

echo ""
echo "========================================"
echo "Self-Play Complete (${NUM_ITERATIONS} iterations)."
echo "========================================"
echo "Cell directory: ${CELL_ROOT}"
echo "Red registry:   ${RED_LORA_REGISTRY}"
echo "Summary:        ${SUMMARY_PATH}"
echo "Prompt manifest:${MANIFEST_PATH}"
