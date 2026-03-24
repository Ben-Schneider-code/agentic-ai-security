#!/bin/bash
# Unified Training Script for Red Team and Blue Team RFT
#
# Usage: ./run_training.sh --target {redteam|blueteam} [--opponent-lora <path>] [--host-only]
#                          [--base-model <hf_model_id>] [--coach-model <hf_model_id>]
#
# --base-model   Base model for student/opponent actors (default: meta-llama/Llama-3.1-8B-Instruct)
# --coach-model  Coach model for trajectory augmentation (default: read from experiments/sql_training.json)
#

set -e
set -o pipefail

echo "========================================"
echo "Agentic AI Security: Unified Training Runner"
echo "========================================"

TARGET=""
OPPONENT_LORA=""
STUDENT_LORA=""
HOST_ONLY=false
RESULTS_ID=""
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LOAD_IN_4BIT=false

# Read coach model default from the config file; can be overridden via --coach-model
COACH_CONFIG="experiments/sql_training.json"
COACH_MODEL_NAME=$(python3 -c "import json; cfg = json.load(open('$COACH_CONFIG')); print([s['model'] for s in cfg['servers'] if s['id'] == 'coach'][0])")

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --opponent-lora) OPPONENT_LORA="$2"; shift ;;
        --student-lora) STUDENT_LORA="$2"; shift ;;
        --host-only) HOST_ONLY=true ;;
        --results-id) RESULTS_ID="$2"; shift ;;
        --base-model) BASE_MODEL="$2"; shift ;;
        --coach-model) COACH_MODEL_NAME="$2"; shift ;;
        --load-in-4bit) LOAD_IN_4BIT=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Base model:   $BASE_MODEL"
echo "Coach model:  $COACH_MODEL_NAME"

if [[ "$TARGET" != "redteam" && "$TARGET" != "blueteam" ]]; then
    echo "ERROR: --target must be 'redteam' or 'blueteam'"
    exit 1
fi

if [[ "$TARGET" == "blueteam" && -z "$OPPONENT_LORA" ]]; then
    echo "ERROR: --opponent-lora is required when target is blueteam!"
    exit 1
fi

ROOT_DIR="$(pwd)"

# If no --results-id was passed (manual run), generate a fresh one
# and guard the whole results-{ID}/ dir.
if [[ -z "$RESULTS_ID" ]]; then
    RESULTS_ID="$(date +%Y%m%d-%H%M)-$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 5 || true)"
    RESULTS_BASE_DIR="${ROOT_DIR}/results-${RESULTS_ID}"
    if [ -d "${RESULTS_BASE_DIR}" ]; then
        echo "ERROR: Collision — ${RESULTS_BASE_DIR} already exists. Exiting."
        exit 1
    fi
else
    # ID was provided by run_selfplay.sh — guard only the team subdir
    RESULTS_BASE_DIR="${ROOT_DIR}/results-${RESULTS_ID}"
    RESULTS_TEAM_DIR="${RESULTS_BASE_DIR}/${TARGET}"
    if [ -d "${RESULTS_TEAM_DIR}" ]; then
        echo "ERROR: Collision — ${RESULTS_TEAM_DIR} already exists. Exiting."
        exit 1
    fi
fi

RESULTS_TEAM_DIR="${RESULTS_BASE_DIR}/${TARGET}"
mkdir -p "${RESULTS_TEAM_DIR}"
echo "Experiment ID:  ${RESULTS_ID}"
echo "Results dir:    ${RESULTS_TEAM_DIR}"

# ============================================
# 1. Start Core Infrastructure (DB, MCP, Coach vLLM)
# ============================================
echo ""
echo "[1/3] Starting Core Fixed Infrastructure..."
if [ -f "/tmp/vllm_coach_registry.json" ]; then
    echo "Coach registry found — skipping coach vLLM startup (already running)."
else
    echo "Coach registry not found — starting services with coach model: $COACH_MODEL_NAME"
    export OVERRIDE_COACH_MODEL="$COACH_MODEL_NAME"
    source start_rft_services.sh
    unset OVERRIDE_COACH_MODEL
fi

# Always read the coach URL from the registry (works whether we just started it or it was already running)
COACH_VLLM_URL=$(python3 -c "import json; reg = json.load(open('/tmp/vllm_coach_registry.json')); print(reg['coach']['url'])")/v1
export COACH_VLLM_URL
echo "Coach vLLM:   $COACH_VLLM_URL (GPU 0)"

# ============================================
# 2. Generate and Start Dynamic Actor vLLM
# ============================================
echo ""
echo "[2/3] Configuring Actor Models for $TARGET..."

ACTOR_CONFIG_PATH="/tmp/actor_vllm_config.json"
ACTOR_REGISTRY_PATH="/tmp/vllm_actor_registry.json"
rm -f "$ACTOR_REGISTRY_PATH"

GEN_CMD=(python3 util/generate_vllm_config.py --target "$TARGET" --out-config "$ACTOR_CONFIG_PATH" --model "$BASE_MODEL")

[[ -n "$OPPONENT_LORA" ]] && GEN_CMD+=(--opponent-lora "$OPPONENT_LORA")
[[ -n "$STUDENT_LORA" ]] && GEN_CMD+=(--student-lora "$STUDENT_LORA")

"${GEN_CMD[@]}"

echo "Starting Actor vLLM Fleet..."
python3 start_vllm.py --config "$ACTOR_CONFIG_PATH" --timeout 600 --wait-only &
VLLM_ACTOR_PID=$!
if [[ "${SELFPLAY_COACH_PERSISTENT:-}" != "1" && -n "${VLLM_FLEET_PID:-}" ]]; then
    trap 'kill $VLLM_ACTOR_PID 2>/dev/null || true; kill $VLLM_FLEET_PID 2>/dev/null || true' EXIT
else
    trap 'kill $VLLM_ACTOR_PID 2>/dev/null || true' EXIT
fi

echo "Waiting for actor models to load..."
TIMEOUT=660
START_TIME=$(date +%s)
while true; do
    if ! kill -0 $VLLM_ACTOR_PID 2>/dev/null; then
        echo "ERROR: Actor vLLM fleet died unexpectedly"
        exit 1
    fi
    if [ -f "$ACTOR_REGISTRY_PATH" ]; then
        if python3 -c "import json; data = json.load(open('$ACTOR_REGISTRY_PATH')); exit(0 if data else 1)" 2>/dev/null; then
            echo "✓ Actor vLLM servers ready"
            break
        fi
    fi
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Actor vLLM servers timed out"
        kill $VLLM_ACTOR_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

# Read the registries
echo "Reading URLs from Registries..."
STUDENT_VLLM_URL=$(python3 -c "import json; reg = json.load(open('$ACTOR_REGISTRY_PATH')); print(reg['student']['url'])")/v1
export STUDENT_VLLM_URL
echo "Student Server URL: $STUDENT_VLLM_URL"

if [[ "$TARGET" == "blueteam" ]]; then
    REDTEAM_VLLM_URL=$(python3 -c "import json; reg = json.load(open('$ACTOR_REGISTRY_PATH')); print(reg['redteam']['url'])")/v1
    export REDTEAM_VLLM_URL
    echo "Red Team Opponent URL: $REDTEAM_VLLM_URL"
fi


if [ "$HOST_ONLY" = true ]; then
    echo ""
    echo "========================================"
    echo "HOST_ONLY MODE: Skipping Training."
    echo "Services are running. Terminate this script to clean up."
    echo "========================================"
    wait
    exit 0
fi

# ============================================
# 3. Training Process
# ============================================
echo ""
echo "[3/3] Starting Training..."
cd MARFT
# Memory optimization flags for CUDA allocator
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

EXTRA_TRAIN_ARGS=""
if [ "$LOAD_IN_4BIT" = true ]; then
    EXTRA_TRAIN_ARGS="--load_in_4bit"
fi

if [[ -n "$OPPONENT_LORA" ]]; then
    if [[ "$TARGET" == "redteam" ]]; then
        # Redteam attacks the opponent (blueteam)
        EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --opponent_model_name opponent_lora --opponent_lora_path $OPPONENT_LORA"
    else
        # Blueteam defends against the opponent (redteam)
        EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --opponent_model_name redteam --opponent_lora_path $OPPONENT_LORA"
    fi
fi

if [[ "$TARGET" == "redteam" ]]; then
    python3 marft/scripts/train_sql.py \
            --seed 10 \
            --env_name redteam_sql_env \
            --algorithm_name APPO \
            --experiment_name redteam_sql_experiment \
            --dataset_name None \
            --flag train \
            --num_mini_batch 10 \
            --ppo_epoch 1 \
            --lr 5e-7 \
            --critic_lr 5e-5 \
            --dataset_path None \
            --model_name_or_path "$BASE_MODEL" \
            --n_agents 1 \
            --agent_iteration_interval 800 \
            --n_rollout_threads 8 \
            --episode_length 10 \
            --gradient_cp_steps 8 \
            --context_window 2048 \
            --max_new_tokens 512 \
            --save_interval 400 \
            --entropy_coef 0.05 \
            --warmup_steps 500 \
            --horizon 5 \
            --coach_vllm_url "$COACH_VLLM_URL" \
            --coach_model_name "$COACH_MODEL_NAME" \
            --results_dir "${RESULTS_TEAM_DIR}" \
            $EXTRA_TRAIN_ARGS
else
    # NOTES:
    # - horizon must match redteam — blueteam attack episodes use multi-turn red LoRA
    python3 marft/scripts/train_sql.py \
            --seed 12 \
            --env_name blueteam_sql_env \
            --algorithm_name APPO \
            --experiment_name blueteam_sql_experiment \
            --dataset_name None \
            --flag train \
            --num_mini_batch 10 \
            --ppo_epoch 1 \
            --lr 5e-7 \
            --critic_lr 5e-5 \
            --dataset_path None \
            --model_name_or_path "$BASE_MODEL" \
            --n_agents 1 \
            --agent_iteration_interval 800 \
            --n_rollout_threads 8 \
            --episode_length 10 \
            --gradient_cp_steps 8 \
            --context_window 4096 \
            --max_new_tokens 512 \
            --save_interval 400 \
            --entropy_coef 0.05 \
            --warmup_steps 500 \
            --horizon 5 \
            --use_eval \
            --eval_interval 10 \
            --eval_episodes 20 \
            --n_eval_rollout_threads 2 \
            --coach_vllm_url "$COACH_VLLM_URL" \
            --coach_model_name "$COACH_MODEL_NAME" \
            --results_dir "${RESULTS_TEAM_DIR}" \
            $EXTRA_TRAIN_ARGS
fi

echo ""
echo "========================================"
echo "$TARGET Training Complete"
echo "========================================"
touch "${RESULTS_TEAM_DIR}/.success"
echo "Success marker: ${RESULTS_TEAM_DIR}/.success"
