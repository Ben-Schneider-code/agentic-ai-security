#!/bin/bash
# Wrapper script to run standard SQL training with proper service initialization
# Usage: ./run_sql_training.sh [args for train_redteam_sql.py]
#
# Default algorithm: GRPO (can be overridden with --algorithm_name)

set -e

echo "========================================"
echo "SQL Training Runner"
echo "========================================"

# Check if vLLM services are already running by checking registry file
REGISTRY_PATH="/tmp/vllm_registry.json"

if [ -f "$REGISTRY_PATH" ]; then
    # Verify registry has student entry and is accessible
    if python3 -c "import json; reg = json.load(open('$REGISTRY_PATH')); exit(0 if 'student' in reg else 1)" 2>/dev/null; then
        echo "vLLM services already running (registry found), using existing instances"
        STUDENT_VLLM_URL=$(python3 -c "import json; reg = json.load(open('$REGISTRY_PATH')); print(reg['student']['url'])")/v1
        export STUDENT_VLLM_URL
    else
        echo "Registry file exists but invalid, starting services..."
        source /app/start_rft_services.sh
    fi
else
    echo "Starting services..."
    source /app/start_rft_services.sh
fi

# Check if user provided --algorithm_name argument
ALGORITHM_SET=false
for arg in "$@"; do
    if [[ "$arg" == "--algorithm_name"* ]]; then
        ALGORITHM_SET=true
        break
    fi
done


# Run SQL training
echo ""
echo "========================================"
echo "Starting SQL Training"
echo "========================================"
echo "Student URL: $STUDENT_VLLM_URL"
echo ""

cd /app/MARFT
# Memory optimization flags for CUDA allocator
# expandable_segments helps reduce fragmentation (per error message suggestion)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
# python3 marft/scripts/train_redteam_sql.py \
#         --seed 10 \
#         --lr 1e-6 \
#         --env_name redteam_sql_env \
#         --algorithm_name GRPO \
#         --experiment_name redteam_sql_experiment \
#         --dataset_name None \
#         --flag train \
#         --num_mini_batch 8 \
#         --ppo_epoch 1 \
#         --dataset_path None \
#         --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#         --n_agents 1 \
#         --agent_iteration_interval 1000 \
#         --n_rollout_threads 16 \
#         --episode_length 5 \
#         --gradient_cp_steps 8 \
#         --context_window 1536 \
#         --max_new_tokens 384 \
#         --save_interval 1000 \
#         --entropy_coef 0.0 \
#         --max_grad_norm 1.0 \
#         --gamma 1.0 \
#         --horizon 5 \
#         --group_size 8 \
#         --generation_temperature 0.8
python3 marft/scripts/train_redteam_sql.py \
        --seed 10 \
        --env_name redteam_sql_env \
        --algorithm_name APPO \
        --experiment_name redteam_sql_experiment \
        --dataset_name None \
        --flag train \
        --num_mini_batch 1 \
        --ppo_epoch 1 \
        --lr 5e-7 \
        --critic_lr 5e-5 \
        --dataset_path None \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --n_agents 1 \
        --agent_iteration_interval 1000 \
        --n_rollout_threads 1 \
        --episode_length 1 \ # TODO: REVIEW THIS
        --gradient_cp_steps 2 \
        --context_window 2048 \
        --max_new_tokens 512 \
        --save_interval 1000 \
        --entropy_coef 0.05 \
        --warmup_steps 500 \
        --horizon 5

echo ""
echo "========================================"
echo "SQL Training Complete"
echo "========================================"
