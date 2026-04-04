#!/bin/bash
set -e
set -o pipefail

echo "========================================"
echo "Agentic AI Security: Self-Play Orchestrator"
echo "========================================"

# --- Defaults ---
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
COACH_MODEL=""
NUM_ITERATIONS=8
REDTEAM_GPU=0
BLUETEAM_GPU=1

# --- Continue defaults ---
CONTINUE_DIR=""
CONTINUE_ITER=""
CONTINUE_ROUND=""

# --- Parse arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --base-model) BASE_MODEL="$2"; shift ;;
        --coach-model) COACH_MODEL="$2"; shift ;;
        --num-iterations) NUM_ITERATIONS="$2"; shift ;;
        --continue) CONTINUE_DIR="$2"; shift ;;
        --continue-iteration) CONTINUE_ITER="$2"; shift ;;
        --continue-round) CONTINUE_ROUND="$2"; shift ;;
        --redteam-gpu) REDTEAM_GPU="$2"; shift ;;
        --blueteam-gpu) BLUETEAM_GPU="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate continue flags ---
if [[ -n "$CONTINUE_DIR" ]]; then
    if [[ ! -d "$CONTINUE_DIR" ]]; then
        echo "ERROR: Continue directory '$CONTINUE_DIR' does not exist."
        exit 1
    fi
    if [[ -z "$CONTINUE_ITER" ]]; then
        echo "ERROR: --continue-iteration is required when using --continue."
        exit 1
    fi
    if [[ -z "$CONTINUE_ROUND" ]]; then
        echo "ERROR: --continue-round is required when using --continue."
        exit 1
    fi
    if [[ "$CONTINUE_ROUND" != "red" && "$CONTINUE_ROUND" != "blue" ]]; then
        echo "ERROR: --continue-round must be 'red' or 'blue'."
        exit 1
    fi
    if ! [[ "$CONTINUE_ITER" =~ ^[0-9]+$ ]] || [[ "$CONTINUE_ITER" -lt 1 ]]; then
        echo "ERROR: --continue-iteration must be a positive integer."
        exit 1
    fi
    if [[ "$CONTINUE_ITER" -gt "$NUM_ITERATIONS" ]]; then
        echo "ERROR: --continue-iteration ($CONTINUE_ITER) exceeds --num-iterations ($NUM_ITERATIONS)."
        exit 1
    fi
elif [[ -n "$CONTINUE_ITER" || -n "$CONTINUE_ROUND" ]]; then
    echo "ERROR: --continue-iteration and --continue-round require --continue <result_dir>."
    exit 1
fi

find_latest_checkpoint() {
    local dir="$1"
    find "$dir" -name "sql_agent" -type d | sort -V | tail -n 1
}

echo "Base model:      $BASE_MODEL"
echo "Coach model:     ${COACH_MODEL:-<default from config>}"
echo "Num iterations:  $NUM_ITERATIONS"
echo "GPU layout:      redteam=GPU${REDTEAM_GPU}  blueteam=GPU${BLUETEAM_GPU}  coach=GPU2 (from experiments/sql_training.json)"

# Build optional coach arg (only pass if explicitly set)
COACH_ARGS=()
if [[ -n "$COACH_MODEL" ]]; then
    COACH_ARGS=(--coach-model "$COACH_MODEL")
fi

export SELFPLAY_COACH_PERSISTENT=1

# Cleanup function used on both success and failure exits.
# Uses SIGTERM (not SIGKILL) to avoid bricking GPUs with active CUDA contexts.
cleanup_all_vllm() {
    echo ""
    echo "[cleanup] Stopping all vLLM processes..."
    pkill -f start_vllm.py || true
    # Give vLLM processes time to shut down gracefully (CUDA context teardown)
    pkill -f vllm.entrypoints || true
    sleep 10
    # Check if any survived; only then escalate, with a warning
    if pgrep -f vllm.entrypoints > /dev/null 2>&1; then
        echo "[cleanup] WARNING: vLLM processes still alive after SIGTERM. Sending SIGKILL."
        echo "[cleanup] This may brick a GPU. Run 'nvidia-smi --gpu-reset -i <id>' if needed."
        pkill -9 -f vllm.entrypoints || true
        sleep 3
    fi
    rm -f /tmp/vllm_coach_registry.json /tmp/vllm_actor_registry.json
}
trap cleanup_all_vllm EXIT

pkill -f start_vllm.py || true
pkill -f vllm.entrypoints || true
sleep 5

BLUE_LATEST_CKPT=""
RED_LATEST_CKPT=""

if [[ -n "$CONTINUE_DIR" ]]; then
    # --- Continue mode: reuse existing SELFPLAY_ID ---
    # Strip trailing slash and "results-" prefix to recover the ID
    CONTINUE_DIR="${CONTINUE_DIR%/}"
    SELFPLAY_ID="${CONTINUE_DIR#results-}"
    # Handle absolute / relative paths: take basename then strip prefix
    SELFPLAY_ID="$(basename "$CONTINUE_DIR")"
    SELFPLAY_ID="${SELFPLAY_ID#results-}"
    echo "Continuing run with Selfplay ID: ${SELFPLAY_ID}"
    echo "  Resume at iteration ${CONTINUE_ITER}, round ${CONTINUE_ROUND}"

    # Recover checkpoints from the iteration before the continue point
    if [[ "$CONTINUE_ITER" -gt 1 ]]; then
        PREV_BLUE_DIR="results-${SELFPLAY_ID}/iter_$((CONTINUE_ITER - 1))/blueteam"
        if [[ ! -d "$PREV_BLUE_DIR" ]]; then
            echo "ERROR: Cannot recover blue LoRA — directory not found: ${PREV_BLUE_DIR}"
            exit 1
        fi
        BLUE_LATEST_CKPT=$(find_latest_checkpoint "${PREV_BLUE_DIR}")
        if [[ -z "$BLUE_LATEST_CKPT" ]]; then
            echo "ERROR: No sql_agent checkpoint found in ${PREV_BLUE_DIR}"
            exit 1
        fi
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
    # --- Fresh run ---
    SELFPLAY_ID="$(date +%Y%m%d-%H%M)-$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 5 || true)"
fi

echo "Selfplay Run ID: ${SELFPLAY_ID}"

for ITER in $(seq 1 $NUM_ITERATIONS); do
    # --- Skip completed iterations when continuing ---
    if [[ -n "$CONTINUE_DIR" && "$ITER" -lt "$CONTINUE_ITER" ]]; then
        echo ""
        echo "[Iter ${ITER}/${NUM_ITERATIONS}] Skipping (already completed)."
        continue
    fi

    echo ""
    echo "========================================"
    echo "Self-Play Iteration ${ITER}/${NUM_ITERATIONS}"
    echo "========================================"

    # Use a structured ID string that run_training.sh will append to "results-"
    ITER_ID="${SELFPLAY_ID}/iter_${ITER}"
    echo "Iteration Path: results-${ITER_ID}/"

    # --- Determine whether to skip the red phase ---
    SKIP_RED=false
    if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" && "$CONTINUE_ROUND" == "blue" ]]; then
        SKIP_RED=true
    fi

    if [[ "$SKIP_RED" == "true" ]]; then
        # Recover RED_LATEST_CKPT from this iteration's existing red results
        RED_DIR="results-${ITER_ID}/redteam"
        if [[ ! -d "$RED_DIR" ]]; then
            echo "ERROR: Cannot recover red LoRA — directory not found: ${RED_DIR}"
            exit 1
        fi
        RED_LATEST_CKPT=$(find_latest_checkpoint "${RED_DIR}")
        if [[ -z "$RED_LATEST_CKPT" ]]; then
            echo "ERROR: No sql_agent checkpoint found in ${RED_DIR}!"
            exit 1
        fi
        RED_LATEST_CKPT=$(realpath "${RED_LATEST_CKPT}")
        echo "[Iter ${ITER}] Skipping Red Team (continuing from blue). Using existing Red LoRA: ${RED_LATEST_CKPT}"
    else
        # --- Clean up stale partial red dir if re-running ---
        if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" && "$CONTINUE_ROUND" == "red" ]]; then
            STALE_RED="results-${ITER_ID}/redteam"
            if [[ -d "$STALE_RED" ]]; then
                echo "[Iter ${ITER}] Removing stale partial red results: ${STALE_RED}"
                rm -rf "$STALE_RED"
            fi
        fi

        RED_TRAIN_ARGS=(--target redteam --results-id "${ITER_ID}" --base-model "$BASE_MODEL" "${COACH_ARGS[@]}" --actor-gpu "$BLUETEAM_GPU" --training-gpu "$REDTEAM_GPU")
        if [[ -n "$BLUE_LATEST_CKPT" ]]; then
            RED_TRAIN_ARGS+=(--opponent-lora "${BLUE_LATEST_CKPT}")
            echo "Using Blue LoRA from previous iteration: ${BLUE_LATEST_CKPT}"
        fi
        if [[ -n "$RED_LATEST_CKPT" ]]; then
            RED_TRAIN_ARGS+=(--student-lora "${RED_LATEST_CKPT}")
            echo "Continuing Red LoRA from previous iteration: ${RED_LATEST_CKPT}"
        fi

        # --- Phase 1: Train Red Team ---
        echo "[Iter ${ITER}] Training Red Team (ID: ${ITER_ID})..."
        if ! ./run_training.sh "${RED_TRAIN_ARGS[@]}"; then
            echo "Red Team training failed on iteration ${ITER}!"
            exit 1
        fi

        # Exact red team dir — known because we set the ID
        RED_DIR="results-${ITER_ID}/redteam"

        # Verify the success marker
        if [ ! -f "${RED_DIR}/.success" ]; then
            echo "ERROR: Red Team dir missing .success marker: ${RED_DIR}"
            exit 1
        fi

        # Find the highest-step LoRA checkpoint within the known dir.
        RED_LATEST_CKPT=$(find_latest_checkpoint "${RED_DIR}")

        if [ -z "$RED_LATEST_CKPT" ]; then
            echo "ERROR: No sql_agent checkpoint found in ${RED_DIR}!"
            exit 1
        fi
        RED_LATEST_CKPT=$(realpath "${RED_LATEST_CKPT}")
        echo "Using Red LoRA: ${RED_LATEST_CKPT}"

        # Kill actor vLLMs before Blue run (coach can stay alive)
        pkill -f "vllm.entrypoints.*--port 800[1-9]" || true
        rm -f /tmp/vllm_actor_registry.json
        sleep 3
    fi

    # --- Clean up stale partial blue dir if re-running ---
    if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" ]]; then
        STALE_BLUE="results-${ITER_ID}/blueteam"
        if [[ -d "$STALE_BLUE" ]]; then
            echo "[Iter ${ITER}] Removing stale partial blue results: ${STALE_BLUE}"
            rm -rf "$STALE_BLUE"
        fi
    fi

    # --- Phase 2: Train Blue Team ---
    BLUE_TRAIN_ARGS=(--target blueteam --results-id "${ITER_ID}" --base-model "$BASE_MODEL" "${COACH_ARGS[@]}" --opponent-lora "${RED_LATEST_CKPT}" --actor-gpu "$REDTEAM_GPU" --training-gpu "$BLUETEAM_GPU")
    if [[ -n "$BLUE_LATEST_CKPT" ]]; then
        BLUE_TRAIN_ARGS+=(--student-lora "${BLUE_LATEST_CKPT}")
        echo "Continuing Blue LoRA from previous iteration: ${BLUE_LATEST_CKPT}"
    fi
    echo "[Iter ${ITER}] Training Blue Team against Red LoRA (ID: ${ITER_ID})..."
    if ! ./run_training.sh "${BLUE_TRAIN_ARGS[@]}"; then
        echo "Blue Team training failed on iteration ${ITER}!"
        exit 1
    fi

    BLUE_DIR="results-${ITER_ID}/blueteam"
    if [ ! -f "${BLUE_DIR}/.success" ]; then
        echo "ERROR: Blue Team dir missing .success marker: ${BLUE_DIR}"
        exit 1
    fi

    echo "[Iter ${ITER}] Complete. Results at results-${ITER_ID}/"

    # Find the highest-step LoRA checkpoint for the blue team
    BLUE_LATEST_CKPT=$(find_latest_checkpoint "${BLUE_DIR}")
    if [ -z "$BLUE_LATEST_CKPT" ]; then
        echo "ERROR: No sql_agent checkpoint found in ${BLUE_DIR}!"
        exit 1
    fi
    BLUE_LATEST_CKPT=$(realpath "${BLUE_LATEST_CKPT}")
    echo "Saved Blue LoRA for next iteration: ${BLUE_LATEST_CKPT}"

    # Kill actor vLLMs before next iteration (coach can stay alive)
    pkill -f "vllm.entrypoints.*--port 800[1-9]" || true
    rm -f /tmp/vllm_actor_registry.json
    sleep 3

    # Clear the continue state after the resumed iteration completes
    if [[ -n "$CONTINUE_DIR" && "$ITER" -eq "$CONTINUE_ITER" ]]; then
        CONTINUE_DIR=""
    fi
done

echo ""
echo "========================================"
echo "Self-Play Alignment Complete (${NUM_ITERATIONS} iterations)."
echo "========================================"

# Full cleanup handled by the EXIT trap (cleanup_all_vllm)
