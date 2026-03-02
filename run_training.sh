#!/bin/bash
# Unified Training Script for Red Team and Blue Team RFT
#
# Usage: ./run_training.sh --target {redteam|blueteam} [--opponent-lora <path>] [--host-only]
#

set -e

echo "========================================"
echo "Agentic AI Security: Unified Training Runner"
echo "========================================"

TARGET=""
OPPONENT_LORA=""
STUDENT_LORA=""
HOST_ONLY=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --opponent-lora) OPPONENT_LORA="$2"; shift ;;
        --student-lora) STUDENT_LORA="$2"; shift ;;
        --host-only) HOST_ONLY=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$TARGET" != "redteam" && "$TARGET" != "blueteam" ]]; then
    echo "ERROR: --target must be 'redteam' or 'blueteam'"
    exit 1
fi

if [[ "$TARGET" == "blueteam" && -z "$OPPONENT_LORA" ]]; then
    echo "ERROR: --opponent-lora is required when target is blueteam!"
    exit 1
fi

# ============================================
# 1. Start Core Infrastructure (DB, MCP, 32B Config)
# ============================================
echo ""
echo "[1/3] Starting Core Fixed Infrastructure..."
if [ -f "/tmp/vllm_coach_registry.json" ]; then
    echo "Coach registry path found so skipping starting..."
else
    echo "Coach registry path not found, starting services..."
    source start_rft_services.sh
fi
echo "Coach vLLM:   $COACH_VLLM_URL (GPU 0)"

# ============================================
# 2. Generate and Start Dynamic Actor vLLM
# ============================================
echo ""
echo "[2/3] Configuring Actor Models for $TARGET..."

ACTOR_CONFIG_PATH="/tmp/actor_vllm_config.json"
ACTOR_REGISTRY_PATH="/tmp/vllm_actor_registry.json"
rm -f "$ACTOR_REGISTRY_PATH"

GEN_CMD="python3 util/generate_vllm_config.py --target \"$TARGET\" --out-config \"$ACTOR_CONFIG_PATH\""

if [[ -n "$OPPONENT_LORA" ]]; then
    GEN_CMD="$GEN_CMD --opponent-lora \"$OPPONENT_LORA\""
fi

if [[ -n "$STUDENT_LORA" ]]; then
    GEN_CMD="$GEN_CMD --student-lora \"$STUDENT_LORA\""
fi

eval $GEN_CMD

echo "Starting Actor vLLM Fleet..."
python3 start_vllm.py --config "$ACTOR_CONFIG_PATH" --timeout 600 --wait-only &
VLLM_ACTOR_PID=$!

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
            --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
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
            --coach_model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
else
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
            --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
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
            --horizon 1 \
            --coach_vllm_url "$COACH_VLLM_URL" \
            --coach_model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
fi

echo ""
echo "========================================"
echo "$TARGET Training Complete"
echo "========================================"
