import os
import torch
import torch.nn as nn
import numpy as np
from abc import ABC
from marft.mas import MAS
from marft.buffers.grpo_buffer import GRPOBuffer
from marft.utils.util import get_gard_norm, to_cuda

# TODO(Justin): This code might cause OOM by torch


class GRPOTrainer(ABC):
    """
    GRPO (Group Relative Policy Optimization) Trainer.

    Key features:
    - No critic network (critic-free)
    - Group-relative advantages: A_i = (r_i - mean(r_group)) / [optional: std(r_group)]
    - Asymmetric clipping: allows low-prob tokens to increase more readily
    - Single epoch updates (mu=1 per DeepSeek-Math)
    - Optional dynamic sampling to filter zero-variance groups
    - Per-token loss normalization (DAPO-style)
    """

    def __init__(self, args, mas: MAS):
        self.mas = mas
        self.num_agent = mas.num_agents
        self.agent_iteration_interval = args.agent_iteration_interval

        # GRPO-specific parameters
        self.clip_param = args.clip_param  # Lower bound (e.g., 0.2)
        self.clip_ratio_high = getattr(
            args, "clip_ratio_high", 0.3
        )  # Upper bound for asymmetric clipping
        self.group_size = args.group_size
        self.use_dynamic_sampling = getattr(args, "use_dynamic_sampling", True)
        self.norm_adv_by_std = getattr(args, "norm_adv_by_std_in_grpo", False)

        # Standard PPO parameters (adapted for GRPO)
        self.ppo_epoch = getattr(args, "ppo_epoch", 1)  # GRPO uses mu=1
        self.num_mini_batch = args.num_mini_batch
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = getattr(
            args, "entropy_coef", 0.0
        )  # GRPO typically doesn't use entropy bonus
        self._use_max_grad_norm = args.use_max_grad_norm
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.gradient_cp_steps = args.gradient_cp_steps

        # Initialize policy optimizers (no critic for GRPO)
        self.policy_optimizer = {}
        for agent in self.mas.agents:
            self.policy_optimizer[agent.role] = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, agent.parameters()),
                lr=self.lr,
                eps=1e-5,
                weight_decay=0,
            )

        # Load optimizer states if resuming
        if args.load_path is not None:
            self.load_optimizers(
                os.path.join(args.load_path, "optimizers.pt"), map_location="cpu"
            )

    def cal_policy_loss(
        self,
        log_prob_infer: torch.Tensor,
        log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        entropy: torch.Tensor,
    ):
        """
        Compute GRPO policy loss with asymmetric clipping.

        Asymmetric clipping (DAPO):
        - For advantage > 0: clip importance ratio to [1-eps_low, 1+eps_high]
          where eps_high > eps_low allows larger increases
        - For advantage < 0: standard symmetric clipping

        This prevents entropy collapse by allowing low-probability tokens
        (in good trajectories) to increase their probability more aggressively.
        """
        log_ratio = log_prob_infer - log_prob_batch
        imp_weights = torch.exp(log_ratio)
        approx_kl = ((imp_weights - 1) - log_ratio).mean()

        # Asymmetric clipping
        # When advantage > 0, we want to increase prob → allow ratio to go up to (1 + clip_ratio_high)
        # When advantage < 0, we want to decrease prob → standard clipping
        advantages_sign = (advantages_batch > 0).float()

        # Upper clip: use clip_ratio_high for positive advantages, clip_param for negative
        clip_upper = (
            1.0
            + self.clip_ratio_high * advantages_sign
            + self.clip_param * (1 - advantages_sign)
        )
        # Lower clip: always use clip_param (make it a tensor to match clip_upper)
        clip_lower = (1.0 - self.clip_param) * torch.ones_like(advantages_batch)

        clipped_ratio = torch.clamp(imp_weights, clip_lower, clip_upper)

        surr1 = -clipped_ratio * advantages_batch
        surr2 = -imp_weights * advantages_batch
        surr = torch.max(surr1, surr2)

        policy_loss = surr.mean() - self.entropy_coef * entropy.mean()
        return policy_loss, approx_kl

    def ppo_update(self, sample, global_steps: int):
        """
        Single GRPO update step.

        Note: GRPO uses mu=1 (single epoch), so this is called once per batch.
        """
        agent_to_train = None
        if self.agent_iteration_interval > 0:
            time_slice = global_steps // self.agent_iteration_interval
            agent_to_train = time_slice % self.num_agent

        (
            observations,
            actions,
            rollout_observations,
            log_probs,
            value_preds,  # Dummy for GRPO
            returns,  # Episode returns
            advantages,  # Group-relative advantages
            action_tokens,
        ) = sample

        # NO advantage normalization by mean/std (already done in buffer as group-relative)
        # This is a key difference from APPO

        (
            actions,
            rollout_observations,
            log_probs,
            returns,
            advantages,
            action_tokens,
        ) = to_cuda(
            (
                actions,
                rollout_observations,
                log_probs,
                returns,
                advantages,
                action_tokens,
            )
        )

        batch_size = rollout_observations.shape[0]
        cp_batch_size = int(batch_size // self.gradient_cp_steps)
        if cp_batch_size == 0:
            cp_batch_size = 1

        # Policy update (no critic for GRPO)
        for optimizer in self.policy_optimizer.values():
            optimizer.zero_grad()

        total_approx_kl = 0.0
        total_entropy = 0.0
        policy_loss = 0.0
        total_policy_grad_norm = 0.0

        for start in range(0, batch_size, cp_batch_size):
            end = start + cp_batch_size
            if end > batch_size:
                end = batch_size
            cp_weight = (end - start) / batch_size

            cp_obs_batch, cp_act_batch, cp_adv_batch, cp_log_probs_batch = (
                rollout_observations[start:end],
                action_tokens[start:end],
                advantages[start:end],
                log_probs[start:end],
            )

            log_prob_infer, cp_entropy = self.mas.get_joint_action_log_probs(
                cp_obs_batch, cp_act_batch, agent_to_train, batch_infer=True
            )

            if agent_to_train is not None:
                cp_log_probs_batch = cp_log_probs_batch[
                    :, agent_to_train : agent_to_train + 1
                ]
                cp_adv_batch = cp_adv_batch[:, agent_to_train : agent_to_train + 1]

            cp_policy_loss, approx_kl = self.cal_policy_loss(
                log_prob_infer, cp_log_probs_batch, cp_adv_batch, cp_entropy
            )

            total_approx_kl += approx_kl.item() * cp_weight
            total_entropy += cp_entropy.mean().item() * cp_weight
            cp_policy_loss = cp_policy_loss * cp_weight
            cp_policy_loss.backward()
            policy_loss += cp_policy_loss.item()

            # Explicitly delete intermediate tensors to free memory
            del log_prob_infer, cp_entropy, cp_policy_loss, approx_kl
            del cp_obs_batch, cp_act_batch, cp_adv_batch, cp_log_probs_batch

            # Clear CUDA cache every few checkpoints to prevent fragmentation
            if (start // cp_batch_size) % 2 == 0:
                torch.cuda.empty_cache()

        # No KL early stopping for GRPO (unlike APPO)
        # GRPO papers show KL penalty is unnecessary

        # Gradient clipping and optimizer step
        if agent_to_train is not None:
            agent = self.mas.agents[agent_to_train]
            if self._use_max_grad_norm:
                policy_grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), self.max_grad_norm
                )
            else:
                policy_grad_norm = get_gard_norm(agent.parameters())
            self.policy_optimizer[agent.role].step()
            total_policy_grad_norm = policy_grad_norm.item()
        else:
            for agent in self.mas.agents:
                if self._use_max_grad_norm:
                    policy_grad_norm = nn.utils.clip_grad_norm_(
                        agent.parameters(), self.max_grad_norm
                    )
                else:
                    policy_grad_norm = get_gard_norm(agent.parameters())
                self.policy_optimizer[agent.role].step()
                total_policy_grad_norm += policy_grad_norm.item()

        return (
            policy_loss,
            total_policy_grad_norm,
            total_approx_kl,
            total_entropy,
        )

    def train(self, buffer: GRPOBuffer, global_steps: int):
        """
        Perform GRPO training update.

        Args:
            buffer: GRPOBuffer containing training data
            global_steps: Current global step count

        Returns:
            train_info: dict with training metrics
        """
        train_info = {
            "policy_loss": 0,
            "policy_grad_norm": 0,
            "approx_kl": 0,
            "entropy": 0,
            "frac_reward_zero_std": buffer.compute_returns_and_advantages(
                use_dynamic_sampling=self.use_dynamic_sampling,
                norm_by_std=self.norm_adv_by_std,
            ),
        }

        update_time = 0
        for _ in range(self.ppo_epoch):  # Typically 1 for GRPO
            data_generator = buffer.sample(self.num_mini_batch)
            for sample in data_generator:
                (
                    policy_loss,
                    policy_grad_norm,
                    approx_kl,
                    entropy,
                ) = self.ppo_update(sample, global_steps)

                # Debug logging
                # print(f"[GRPOTrainer] Mini-batch update done. Loss: {policy_loss:.4f}, Grad: {policy_grad_norm:.4f}")

                train_info["policy_loss"] += policy_loss
                train_info["policy_grad_norm"] += policy_grad_norm
                train_info["approx_kl"] += approx_kl
                train_info["entropy"] += entropy
                update_time += 1

        print(f"[GRPOTrainer] Completed {update_time} updates for step {global_steps}")

        # Average metrics
        for k in ["policy_loss", "policy_grad_norm", "approx_kl", "entropy"]:
            train_info[k] /= max(update_time, 1)

        return train_info

    def save_optimizers(self, save_dir: str, steps: int) -> None:
        """Save optimizer states."""
        exp_path = os.path.join(save_dir, "steps_{:04d}".format(steps))
        os.makedirs(exp_path, exist_ok=True)
        torch.save(
            {
                "policy_opt_states": {
                    role: opt.state_dict()
                    for role, opt in self.policy_optimizer.items()
                },
            },
            os.path.join(exp_path, "optimizers.pt"),
        )
        print(f"[GRPOTrainer] optimizer states saved -> {exp_path}")

    def load_optimizers(self, path: str, map_location: str | torch.device = "cpu"):
        """Load optimizer states."""
        ckpt = torch.load(path, map_location=map_location)
        for role, opt_state in ckpt["policy_opt_states"].items():
            self.policy_optimizer[role].load_state_dict(opt_state)
        print(f"[GRPOTrainer] optimizer states loaded <- {path}")

    def prep_training(self):
        """Set models to training mode."""
        for agent in self.mas.agents:
            agent.train()

    def prep_rollout(self):
        """Set models to evaluation mode."""
        for agent in self.mas.agents:
            agent.model.zero_grad(set_to_none=True)
            agent.eval()
        torch.cuda.empty_cache()
