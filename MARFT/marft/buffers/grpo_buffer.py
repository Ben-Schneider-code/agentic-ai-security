import numpy as np
from .base_buffer import BaseBuffer

# TODO(Justin): This code might cause OOM by torch

class GRPOBuffer(BaseBuffer):
    """
    Buffer for GRPO (Group Relative Policy Optimization).

    Key differences from ActionBuffer:
    - Organizes data by groups (G episodes per unique prompt)
    - Stores episode-level returns for group-relative advantage computation
    - No value predictions needed (GRPO is critic-free)
    - Supports dynamic sampling (filtering zero-variance groups)

    Data organization:
    - episode_length = horizon * G (e.g., 5 * 8 = 40 for horizon=5, G=8)
    - Each "batch" contains multiple groups
    - Each group has G episodes starting from the same prompt
    """

    def __init__(self, args, num_agents):
        super().__init__(args, num_agents)
        self._shuffle_rng = np.random.default_rng(getattr(args, "seed", 0))
        self.group_size = args.group_size

        # Store episode-level cumulative rewards for GRPO
        # Shape: (max_batch, episode_length, n_rollout_threads, num_agents)
        # We'll accumulate rewards over each episode (horizon steps) into this
        self.episode_returns = np.zeros(
            (
                self.max_batch,
                self.episode_length,
                self.n_rollout_threads,
                self.num_agents,
            ),
            dtype=np.float32,
        )

        # GRPO advantages (group-relative)
        # Shape matches episode_returns since GRPO computes advantages per episode
        self.grpo_advantages = np.zeros_like(self.episode_returns)

        # Store log probs per episode (summed or averaged over turns)
        self.grpo_log_probs = np.zeros_like(self.episode_returns)

        # Track which groups have valid variance for dynamic sampling
        self.valid_groups = []

    def insert(
        self,
        next_obs,
        actions,
        rollout_obs,
        value_preds,
        rewards,
        masks,
        action_tokens,
        log_probs,
    ):
        """
        Insert transition data. For GRPO, value_preds are ignored (no critic).
        """
        self.obs[self.cur_batch_index, self.step + 1] = next_obs.copy()
        self.actions[self.cur_batch_index, self.step] = actions.copy()
        self.rollout_obs[self.cur_batch_index, self.step] = rollout_obs.copy()
        self.rewards[self.cur_batch_index, self.step] = rewards.copy()
        self.masks[self.cur_batch_index, self.step + 1] = masks.copy()
        self.action_tokens[self.cur_batch_index, self.step] = action_tokens.copy()
        self.grpo_log_probs[self.cur_batch_index, self.step] = log_probs.copy()
        self.step = (self.step + 1) % self.episode_length

    def compute_gae_and_returns(self, next_value):
        """
        Stub implementation to satisfy BaseBuffer abstract method.

        GRPO doesn't use GAE (Generalized Advantage Estimation) or value bootstrapping
        since it's critic-free. The actual advantage computation happens in
        compute_returns_and_advantages() which is called by the trainer.

        This method is never called for GRPO (see runner's before_update() method).
        """
        pass

    def compute_returns_and_advantages(
        self, use_dynamic_sampling=True, norm_by_std=False
    ):
        """
        Compute episode returns and group-relative advantages for GRPO.

        Process:
        1. Sum rewards over each episode (horizon steps) to get episode returns
        2. Group episodes by prompt (G episodes per group)
        3. Compute group-relative advantages: A_i = r_i - mean(r_group)
        4. Optionally filter out zero-variance groups (dynamic sampling)

        Args:
            use_dynamic_sampling: Filter out groups where all episodes have same reward
            norm_by_std: Normalize advantages by std (default False per Dr.GRPO)
        """
        # For GRPO, we need to accumulate rewards into episode returns
        # Assuming the runner calls this after collecting full episodes
        # The rewards are already stored in self.rewards

        # Compute episode returns by summing rewards within each episode
        # This assumes self.rewards contains per-step rewards for completed episodes
        # We'll compute cumulative return for each episode
        # Compute episode returns by summing rewards across the entire episode
        # For GRPO with sparse rewards, the return for EVERY step in the episode
        # should be the total cumulative reward of the episode.

        # 1. Sum rewards along the episode_length dimension (axis 1)
        # Shape: (max_batch, n_rollout_threads, num_agents)
        total_episode_rewards = self.rewards[self.cur_batch_index].sum(axis=0)

        # 2. Broadcast this total return to all steps in the episode
        # We assign the same total_episode_rewards to every step
        for step in range(self.episode_length):
            self.episode_returns[self.cur_batch_index, step] = total_episode_rewards

        # Compute group-relative advantages
        # Group structure: n_rollout_threads should be divisible by group_size
        # Groups are organized as: threads [0:G] = group 0, [G:2G] = group 1, etc.
        num_groups = self.n_rollout_threads // self.group_size

        self.valid_groups = []

        for group_idx in range(num_groups):
            start_thread = group_idx * self.group_size
            end_thread = start_thread + self.group_size

            # Get returns for this group (all steps, all agents)
            # Shape: (episode_length, group_size, num_agents)
            group_returns = self.episode_returns[
                self.cur_batch_index, :, start_thread:end_thread, :
            ]

            # Compute group statistics across episodes (dim=1 is the group_size dimension)
            # For each step and agent, compute mean and std across the G episodes
            group_mean = group_returns.mean(
                axis=1, keepdims=True
            )  # (episode_length, 1, num_agents)
            group_std = group_returns.std(
                axis=1, keepdims=True
            )  # (episode_length, 1, num_agents)

            # Check if this group has zero variance (all episodes got same reward)
            has_variance = np.any(group_std > 1e-8)

            if use_dynamic_sampling and not has_variance:
                # Skip this group - no learning signal
                # Set advantages to zero so they don't contribute to gradients
                self.grpo_advantages[
                    self.cur_batch_index, :, start_thread:end_thread, :
                ] = 0.0
                continue

            self.valid_groups.append(group_idx)

            # Compute advantages: A_i = r_i - mean(r_group)
            advantages = group_returns - group_mean

            # Optional: normalize by std (Dr.GRPO recommends against this)
            if norm_by_std and has_variance:
                advantages = advantages / (group_std + 1e-8)

            # Store advantages
            self.grpo_advantages[
                self.cur_batch_index, :, start_thread:end_thread, :
            ] = advantages

        self.cur_num_batch = (
            self.cur_num_batch + 1
            if self.cur_num_batch < self.max_batch
            else self.max_batch
        )

        # Return fraction of groups with zero variance for logging
        frac_zero_variance = 1.0 - (len(self.valid_groups) / max(num_groups, 1))
        return frac_zero_variance

    def sample(self, num_mini_batch: int = None, mini_batch_size: int = None):
        """
        Yield training data for GRPO.

        Returns same structure as ActionBuffer for compatibility, but:
        - value_preds are dummy (zeros)
        - returns are episode returns (not GAE-based)
        - advantages are group-relative
        """
        batch_size = self.n_rollout_threads * self.episode_length * self.cur_num_batch
        num_mini_batch *= self.cur_num_batch

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        rand = np.arange(batch_size)
        self._shuffle_rng.shuffle(rand)
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        # Flatten data
        obs = self.obs[:, :-1].reshape(-1, *self.obs.shape[3:])
        actions = self.actions.reshape(-1, *self.actions.shape[3:])
        rollout_obs = self.rollout_obs[:, :-1].reshape(-1, *self.rollout_obs.shape[3:])
        returns = self.episode_returns.reshape(-1, *self.episode_returns.shape[3:])
        advantages = self.grpo_advantages.reshape(-1, *self.grpo_advantages.shape[3:])
        log_prob = self.grpo_log_probs.reshape(-1, *self.grpo_log_probs.shape[3:])
        action_tokens = self.action_tokens.reshape(-1, *self.action_tokens.shape[3:])

        # Dummy value predictions (GRPO doesn't use values)
        value_preds = np.zeros_like(returns)

        for indices in sampler:
            obs_batch = obs[indices]
            action_batch = actions[indices]
            rollout_obs_batch = rollout_obs[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            advantages_batch = advantages[indices]
            log_prob_batch = log_prob[indices]
            action_tokens_batch = action_tokens[indices]

            yield (
                obs_batch,
                action_batch,
                rollout_obs_batch,
                log_prob_batch,
                value_preds_batch,
                return_batch,
                advantages_batch,
                action_tokens_batch,
            )
