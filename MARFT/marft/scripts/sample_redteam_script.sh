#!/bin/bash

# Redteam SQL Training Script
# Training auto-stops at 2000 episodes (set in REWARD_CONFIG.max_episodes). Feel free to configure rewards in code.

echo $CUDA_VISIBLE_DEVICES
echo HF_TOKEN:
echo $HF_TOKEN

# Stop if no token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &>/dev/null && pwd)

exec python3 $SCRIPT_DIR/train_redteam_sql.py \
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
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
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
