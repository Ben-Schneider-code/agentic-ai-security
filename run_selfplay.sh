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
NUM_ITERATIONS=2
NUM_ENV_STEPS=1600
HORIZON=5
REDTEAM_GPU=0
BLUETEAM_GPU=1
VANILLA_SIZE=""
BORDERCASE_SIZE=""
HONEYPOT_TYPE=""
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
        --num-iterations) NUM_ITERATIONS="$2"; shift ;;
        --num-env-steps) NUM_ENV_STEPS="$2"; shift ;;
        --horizon) HORIZON="$2"; shift ;;
        --redteam-gpu) REDTEAM_GPU="$2"; shift ;;
        --blueteam-gpu) BLUETEAM_GPU="$2"; shift ;;
        --vanilla-size) VANILLA_SIZE="$2"; shift ;;
        --bordercase-size) BORDERCASE_SIZE="$2"; shift ;;
        --honeypot-type) HONEYPOT_TYPE="$2"; shift ;;
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

echo "Base model:      $BASE_MODEL"
echo "Num iterations:  $NUM_ITERATIONS"
echo "Num env steps:   $NUM_ENV_STEPS (per phase, applied symmetrically to red and blue)"
echo "Horizon:         $HORIZON"
echo "GPU layout:      redteam=GPU${REDTEAM_GPU}  blueteam=GPU${BLUETEAM_GPU}"
echo "Honeypot type:   ${HONEYPOT_TYPE:-rowcol (default)}"
[[ -n "$VANILLA_SIZE" ]]    && echo "Vanilla size:    $VANILLA_SIZE"
[[ -n "$BORDERCASE_SIZE" ]] && echo "Bordercase size: $BORDERCASE_SIZE"

# Cleanup function used on both success and failure exits.
cleanup_all_vllm() {
    echo ""
    echo "[cleanup] Stopping all vLLM processes..."
    pkill -f start_vllm.py || true
    pkill -f vllm.entrypoints || true
    sleep 10
    if pgrep -f vllm.entrypoints > /dev/null 2>&1; then
        echo "[cleanup] WARNING: vLLM still alive after SIGTERM. Sending SIGKILL."
        echo "[cleanup] If a GPU is bricked: 'nvidia-smi --gpu-reset -i <id>'."
        pkill -9 -f vllm.entrypoints || true
        sleep 3
    fi
    rm -f /tmp/vllm_actor_registry.json
}
trap cleanup_all_vllm EXIT

pkill -f start_vllm.py || true
pkill -f vllm.entrypoints || true
sleep 5

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
    OPPONENT_SAMPLER_SEED=$(python3 -c "import zlib; print((1000003 * ${REPLICATE_SEED} + zlib.crc32(b'${SELFPLAY_ID}')) & 0xFFFFFFFF)")
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
    "num_iterations": ${NUM_ITERATIONS},
    "num_env_steps": ${NUM_ENV_STEPS},
    "horizon": ${HORIZON},
    "redteam_gpu": ${REDTEAM_GPU},
    "blueteam_gpu": ${BLUETEAM_GPU},
    "vanilla_size": ${VANILLA_SIZE:-null},
    "bordercase_size": ${BORDERCASE_SIZE:-null},
    "honeypot_type": "${HONEYPOT_TYPE:-rowcol}",
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
[[ -n "$HONEYPOT_TYPE" ]]   && ABLATION_ARGS+=(--honeypot-type "$HONEYPOT_TYPE")
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

        RED_TRAIN_ARGS=(
            --target redteam
            --results-id "${ITER_ID}"
            --base-model "$BASE_MODEL"
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
        if ! ./run_training.sh "${RED_TRAIN_ARGS[@]}"; then
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

        pkill -f "vllm.entrypoints.*--port 800[1-9]" || true
        rm -f /tmp/vllm_actor_registry.json
        sleep 3
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

    BLUE_TRAIN_ARGS=(
        --target blueteam
        --results-id "${ITER_ID}"
        --base-model "$BASE_MODEL"
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
    if ! ./run_training.sh "${BLUE_TRAIN_ARGS[@]}"; then
        echo "Blue Team training failed on iteration ${ITER}!"; exit 1
    fi

    BLUE_DIR="results-${ITER_ID}/blueteam"
    [ ! -f "${BLUE_DIR}/.success" ] && { echo "ERROR: Blue Team dir missing .success marker: ${BLUE_DIR}"; exit 1; }
    echo "[Iter ${ITER}] Complete. Results at results-${ITER_ID}/"

    BLUE_LATEST_CKPT=$(find_latest_checkpoint "${BLUE_DIR}")
    [[ -z "$BLUE_LATEST_CKPT" ]] && { echo "ERROR: no sql_agent checkpoint in ${BLUE_DIR}"; exit 1; }
    BLUE_LATEST_CKPT=$(realpath "${BLUE_LATEST_CKPT}")
    echo "Saved Blue LoRA for next iteration: ${BLUE_LATEST_CKPT}"

    pkill -f "vllm.entrypoints.*--port 800[1-9]" || true
    rm -f /tmp/vllm_actor_registry.json
    sleep 3

    if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" ]]; then
        CONTINUE_DIR=""
    fi
done

echo ""
echo "========================================"
echo "Self-Play Complete (${NUM_ITERATIONS} iterations)."
echo "========================================"
echo "Cell directory: ${CELL_ROOT}"
echo "Red registry:   ${RED_LORA_REGISTRY}"
echo "Summary:        ${SUMMARY_PATH}"
