#!/bin/bash

# export CUDA_VISIBLE_DEVICES="6,7"
echo $CUDA_VISIBLE_DEVICES
echo HF_TOKEN:
echo $HF_TOKEN

# Stop if no token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)
# source "$SCRIPT_DIR/../../../.venv/bin/activate"

exec python3 $SCRIPT_DIR/train_redteam_sql.py \
        --seed 10 \
        --env_name redteam_sql_env \
        --algorithm_name APPO \
        --experiment_name sample_redteam_sql \
        --dataset_name None \
        --flag train \
        --num_mini_batch 1 \
        --ppo_epoch 1 \
        --lr 1e-6 \
        --critic_lr 5e-5 \
        --dataset_path None \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --n_agents 1 \
        --agent_iteration_interval 1000 \
        --profile_path $SCRIPT_DIR/profiles/redteam_sql.json \
        --n_rollout_threads 1 \
        --episode_length 1 \
        --gradient_cp_steps 2 \
        --context_window 2048 \
        --max_new_tokens 512 \
        --save_interval 1000
