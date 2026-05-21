#!/bin/bash
# Unified Training Script for Red Team and Blue Team RFT.
#
# Self-play redesign: NO coach, NO judge, NO degeneracy logic, NO early stop.
# Termination is driven solely by --num-env-steps. Blue can train against a
# pool of red LoRA checkpoints via --opponent-lora-pool.
#
# Usage:
#   ./run_training.sh --target {redteam|blueteam}
#                     [--opponent-lora <path>]
#                     [--opponent-lora-pool <red_lora_registry.json>]   # blueteam history mode
#                     [--student-lora <path>]
#                     [--base-model <hf_model_id>]
#                     [--actor-gpu N] [--training-gpu N]
#                     [--horizon N] [--num-env-steps N]
#                     [--vanilla-size N] [--bordercase-size N]
#                     [--honeypot-type {rowcol|row|col}]
#                     [--opponent-sampler-seed N]
#                     [--results-id <id>] [--resume-run-dir <dir>]

set -e
set -o pipefail

echo "========================================"
echo "Agentic AI Security: Unified Training Runner"
echo "========================================"

TARGET=""
OPPONENT_LORA=""
OPPONENT_LORA_POOL=""
STUDENT_LORA=""
HOST_ONLY=false
RESULTS_ID=""
BASE_MODEL="Snowflake/Arctic-Text2SQL-R1-7B"
LOAD_IN_4BIT=false
ACTOR_GPU=1
TRAINING_GPU=2
HORIZON=5
NUM_ENV_STEPS=1600
VANILLA_SIZE=""
BORDERCASE_SIZE=""
HONEYPOT_TYPE=""
OPPONENT_SAMPLER_SEED=""
RESUME_RUN_DIR=""
SEED=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --opponent-lora) OPPONENT_LORA="$2"; shift ;;
        --opponent-lora-pool) OPPONENT_LORA_POOL="$2"; shift ;;
        --student-lora) STUDENT_LORA="$2"; shift ;;
        --host-only) HOST_ONLY=true ;;
        --results-id) RESULTS_ID="$2"; shift ;;
        --base-model) BASE_MODEL="$2"; shift ;;
        --load-in-4bit) LOAD_IN_4BIT=true ;;
        --actor-gpu) ACTOR_GPU="$2"; shift ;;
        --training-gpu) TRAINING_GPU="$2"; shift ;;
        --horizon) HORIZON="$2"; shift ;;
        --num-env-steps) NUM_ENV_STEPS="$2"; shift ;;
        --vanilla-size) VANILLA_SIZE="$2"; shift ;;
        --bordercase-size) BORDERCASE_SIZE="$2"; shift ;;
        --honeypot-type) HONEYPOT_TYPE="$2"; shift ;;
        --opponent-sampler-seed) OPPONENT_SAMPLER_SEED="$2"; shift ;;
        --resume-run-dir) RESUME_RUN_DIR="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        # Deprecated coach flags — silently consume so old callers don't break.
        --coach-model) shift ;;
        --coach-gpu) shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

EPISODE_LENGTH=$(( 2 * HORIZON ))

echo "Base model:      $BASE_MODEL"
echo "Target:          $TARGET"
echo "Actor GPU:       $ACTOR_GPU"
echo "Training GPU:    $TRAINING_GPU"
echo "Horizon:         $HORIZON"
echo "Num env steps:   $NUM_ENV_STEPS"
echo "Episode length:  $EPISODE_LENGTH (=2*horizon)"
echo "Honeypot type:   ${HONEYPOT_TYPE:-rowcol (default)}"
echo "Vanilla size:    ${VANILLA_SIZE:-<unset>}"
echo "Bordercase size: ${BORDERCASE_SIZE:-<unset>}"
[[ -n "$OPPONENT_LORA_POOL" ]] && echo "Opponent pool:   $OPPONENT_LORA_POOL"

if [[ "$TARGET" != "redteam" && "$TARGET" != "blueteam" ]]; then
    echo "ERROR: --target must be 'redteam' or 'blueteam'"; exit 1
fi
if [[ "$TARGET" == "blueteam" && -z "$OPPONENT_LORA" && -z "$OPPONENT_LORA_POOL" ]]; then
    echo "ERROR: blueteam requires --opponent-lora or --opponent-lora-pool"; exit 1
fi

ROOT_DIR="$(pwd)"

if [[ -z "$RESULTS_ID" ]]; then
    RESULTS_ID="$(date +%Y%m%d-%H%M)-$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 5 || true)"
    RESULTS_BASE_DIR="${ROOT_DIR}/results-${RESULTS_ID}"
    if [ -d "${RESULTS_BASE_DIR}" ] && [[ -z "$RESUME_RUN_DIR" ]]; then
        echo "ERROR: Collision — ${RESULTS_BASE_DIR} already exists. Exiting."; exit 1
    fi
else
    RESULTS_BASE_DIR="${ROOT_DIR}/results-${RESULTS_ID}"
    RESULTS_TEAM_DIR="${RESULTS_BASE_DIR}/${TARGET}"
    if [ -d "${RESULTS_TEAM_DIR}" ] && [[ -z "$RESUME_RUN_DIR" ]]; then
        echo "ERROR: Collision — ${RESULTS_TEAM_DIR} already exists. Exiting."; exit 1
    fi
fi
RESULTS_TEAM_DIR="${RESULTS_BASE_DIR}/${TARGET}"
mkdir -p "${RESULTS_TEAM_DIR}"

if [[ -n "$RESUME_RUN_DIR" ]]; then
    if [[ ! -d "$RESUME_RUN_DIR" ]]; then
        echo "ERROR: --resume-run-dir does not exist: $RESUME_RUN_DIR"; exit 1
    fi
    if [[ ! -f "$RESUME_RUN_DIR/training_state.json" ]]; then
        echo "ERROR: --resume-run-dir has no training_state.json: $RESUME_RUN_DIR"; exit 1
    fi
    RESUME_RUN_DIR="$(realpath "$RESUME_RUN_DIR")"
    echo "Resuming run at: $RESUME_RUN_DIR"
fi
echo "Experiment ID:  ${RESULTS_ID}"
echo "Results dir:    ${RESULTS_TEAM_DIR}"

# ============================================
# 1. Start Core Infrastructure (DB, MCP)
# ============================================
echo ""
echo "[1/3] Starting Core Fixed Infrastructure..."
./script/init.sh

# ============================================
# 2. Generate and Start Dynamic Actor vLLM
# ============================================
echo ""
echo "[2/3] Configuring Actor Models for $TARGET..."

ACTOR_CONFIG_PATH="/tmp/actor_vllm_config.json"
ACTOR_REGISTRY_PATH="/tmp/vllm_actor_registry.json"
rm -f "$ACTOR_REGISTRY_PATH"

GEN_CMD=(python3 util/generate_vllm_config.py \
    --target "$TARGET" \
    --out-config "$ACTOR_CONFIG_PATH" \
    --model "$BASE_MODEL" \
    --actor-gpu "$ACTOR_GPU")

[[ -n "$OPPONENT_LORA" ]] && GEN_CMD+=(--opponent-lora "$OPPONENT_LORA")
[[ -n "$OPPONENT_LORA_POOL" ]] && GEN_CMD+=(--opponent-lora-pool "$OPPONENT_LORA_POOL")
[[ -n "$STUDENT_LORA" ]] && GEN_CMD+=(--student-lora "$STUDENT_LORA")

"${GEN_CMD[@]}"

echo "Starting Actor vLLM Fleet..."
python3 start_vllm.py --config "$ACTOR_CONFIG_PATH" --timeout 600 --wait-only &
VLLM_ACTOR_PID=$!
trap 'kill $VLLM_ACTOR_PID 2>/dev/null || true' EXIT

echo "Waiting for actor models to load..."
TIMEOUT=660
START_TIME=$(date +%s)
while true; do
    if ! kill -0 $VLLM_ACTOR_PID 2>/dev/null; then
        echo "ERROR: Actor vLLM fleet died unexpectedly"; exit 1
    fi
    if [ -f "$ACTOR_REGISTRY_PATH" ]; then
        if python3 -c "import json; data = json.load(open('$ACTOR_REGISTRY_PATH')); exit(0 if data else 1)" 2>/dev/null; then
            echo "✓ Actor vLLM servers ready"; break
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

echo "Reading URLs from Registries..."
STUDENT_VLLM_URL=$(python3 -c "import json; reg = json.load(open('$ACTOR_REGISTRY_PATH')); print(reg['student']['url'])")/v1
export STUDENT_VLLM_URL
echo "Student Server URL: $STUDENT_VLLM_URL"

if [[ "$TARGET" == "blueteam" ]]; then
    REDTEAM_VLLM_URL=$(python3 -c "import json; reg = json.load(open('$ACTOR_REGISTRY_PATH')); print(reg['redteam']['url'])")/v1
    export REDTEAM_VLLM_URL
    echo "Red Team Opponent URL: $REDTEAM_VLLM_URL"

    # Build a comma-separated REDTEAM_LORA_POOL from the pool registry so the
    # blue env knows which adapter names to sample from.
    if [[ -n "$OPPONENT_LORA_POOL" ]]; then
        REDTEAM_LORA_POOL=$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
entries = data['entries'] if isinstance(data, dict) and 'entries' in data else data
print(','.join(e['name'] for e in entries))
" "$OPPONENT_LORA_POOL")
        export REDTEAM_LORA_POOL
        echo "Red LoRA pool names: $REDTEAM_LORA_POOL"
    fi
fi

if [ "$HOST_ONLY" = true ]; then
    echo ""
    echo "HOST_ONLY MODE: Skipping Training. Services are running."
    wait
    exit 0
fi

# ============================================
# 3. Training Process
# ============================================
echo ""
echo "[3/3] Starting Training..."
cd MARFT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:False
export TRAINING_GPU="$TRAINING_GPU"
export TRAINING_DEVICE="cuda:$TRAINING_GPU"

EXTRA_TRAIN_ARGS=""
if [ "$LOAD_IN_4BIT" = true ]; then
    EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --load_in_4bit"
fi

# Tell the env about the opponent (single-LoRA case OR first entry of pool).
if [[ -n "$OPPONENT_LORA" ]]; then
    if [[ "$TARGET" == "redteam" ]]; then
        EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --opponent_model_name opponent_lora --opponent_lora_path $OPPONENT_LORA"
    else
        # Blueteam single-LoRA fallback (when no pool given): adapter name is "redteam".
        EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --opponent_model_name redteam --opponent_lora_path $OPPONENT_LORA"
    fi
elif [[ "$TARGET" == "blueteam" && -n "$OPPONENT_LORA_POOL" ]]; then
    # Use the first adapter name as the initial opponent_model_name. The env
    # will resample from REDTEAM_LORA_POOL each episode.
    FIRST_ADAPTER=$(echo "$REDTEAM_LORA_POOL" | cut -d, -f1)
    EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --opponent_model_name $FIRST_ADAPTER"
fi

if [[ -n "$STUDENT_LORA" ]]; then
    STUDENT_CKPT_DIR="$(dirname "$STUDENT_LORA")"
    EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --load_path $STUDENT_CKPT_DIR"
    echo "Initializing LoRA from prior checkpoint: $STUDENT_CKPT_DIR"
fi

if [[ -n "$RESUME_RUN_DIR" ]]; then
    EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --resume_run_dir $RESUME_RUN_DIR"
fi

# Ablation flags forwarded to train_sql.py (which translates them to env vars
# before env construction so the env reads the correct values).
[[ -n "$VANILLA_SIZE" ]] && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --vanilla-size $VANILLA_SIZE"
[[ -n "$BORDERCASE_SIZE" ]] && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --bordercase-size $BORDERCASE_SIZE"
[[ -n "$HONEYPOT_TYPE" ]] && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --honeypot-type $HONEYPOT_TYPE"
[[ -n "$REDTEAM_LORA_POOL" ]] && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --red-lora-pool $REDTEAM_LORA_POOL"
[[ -n "$OPPONENT_SAMPLER_SEED" ]] && EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --opponent-sampler-seed $OPPONENT_SAMPLER_SEED"

COMMON_ARGS=(
    --algorithm_name APPO
    --dataset_name None --dataset_path None
    --flag train
    --num_mini_batch 10 --ppo_epoch 1
    --lr 5e-7 --critic_lr 5e-5
    --model_name_or_path "$BASE_MODEL"
    --n_agents 1
    --agent_iteration_interval 800
    --n_rollout_threads 8
    --num_env_steps "$NUM_ENV_STEPS"
    --episode_length "$EPISODE_LENGTH"
    --gradient_cp_steps 8
    --context_window 16384
    --max_new_tokens 512 --victim_max_tokens 256
    --save_interval 400
    --entropy_coef 0.05
    --warmup_steps 500
    --horizon "$HORIZON"
    --results_dir "${RESULTS_TEAM_DIR}"
)

if [[ "$TARGET" == "redteam" ]]; then
    TARGET_ARGS=(
        --seed "${SEED:-10}"
        --env_name redteam_sql_env
        --experiment_name redteam_sql_experiment
    )
else
    TARGET_ARGS=(
        --seed "${SEED:-12}"
        --env_name blueteam_sql_env
        --experiment_name blueteam_sql_experiment
        --use_eval --eval_interval 10 --eval_episodes 20 --n_eval_rollout_threads 2
    )
fi

python3 marft/scripts/train_sql.py "${COMMON_ARGS[@]}" "${TARGET_ARGS[@]}" $EXTRA_TRAIN_ARGS

echo ""
echo "========================================"
echo "$TARGET Training Complete"
echo "========================================"
touch "${RESULTS_TEAM_DIR}/.success"
echo "Success marker: ${RESULTS_TEAM_DIR}/.success"
