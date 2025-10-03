export CUDA_VISIBLE_DEVICES="2"
echo $CUDA_VISIBLE_DEVICES
python train_redteam_sql.py \
        --seed 10 \
        --env_name redteam_sql_env \
        --algorithm_name APPO \
        --experiment_name experiment_name \
        --dataset_name None \
        --flag train \
        --num_mini_batch 1 \
        --ppo_epoch 1 \
        --lr 1e-6 \
        --critic_lr 5e-5 \
        --dataset_path None \
        --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
        --n_agents 1 \
        --agent_iteration_interval 1000 \
        --profile_path profiles/redteam_sql.json \
        --n_rollout_threads 1 \
        --episode_length 1 \
        --gradient_cp_steps 2 \
        --context_window 2048 \
        --max_new_tokens 512 \
        --save_interval 1000