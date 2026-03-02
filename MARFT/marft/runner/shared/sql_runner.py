import os
import json
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from marft.mas import MAS


def _log_lure(
    log_dir: str,
    episode: int,
    thread_idx: int,
    honeypot_ids: list,
    reward: float,
    blueteam_context: tuple,
    red_team_actions: list,
):
    """Append a lure entry to lure_log.jsonl for future Blue Team training."""
    system_prompt, victim_conversation = blueteam_context
    entry = {
        "episode": episode,
        "thread": thread_idx,
        "honeypot_ids": honeypot_ids,
        "reward": reward,
        "blueteam_system_prompt": system_prompt,
        "victim_conversation": victim_conversation,
        "red_team_actions": red_team_actions,
    }
    lure_path = os.path.join(log_dir, "lure_log.jsonl")
    with open(lure_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


class SQLRunner:
    """Runner class to perform training, evaluation. and data collection. See parent class for details."""

    def __init__(self, config):
        self.num_agents = config["num_agents"]
        self.all_args = config["all_args"]
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.eval_interval = self.all_args.eval_interval
        self.algo = self.all_args.algorithm_name
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]

        # Always load profiles from the environment (dynamic generation)
        profiles = None
        if hasattr(self.envs, "envs") and len(self.envs.envs) > 0:
            env = self.envs.envs[0]
            if hasattr(env, "profiles"):
                profiles = env.profiles
                print(f"[Runner] Loaded {len(profiles)} profiles from environment")

        self.mas = MAS(
            model_path=self.all_args.model_name_or_path,
            context_window=self.all_args.context_window,
            max_new_tokens=self.all_args.max_new_tokens,
            num_agents=self.num_agents,
            profile_path=None,  # Ignore CLI arg, use dynamic profiles
            algo=self.algo,
            normalization_mode=self.all_args.normalization_mode,
            load_path=self.all_args.load_path,
            profiles=profiles,
        )

        if self.algo == "APPO":
            from marft.algorithms import APPOTrainer
            from marft.buffers.action_level_buffer import ActionBuffer

            self.trainer = APPOTrainer(self.all_args, self.mas)
            self.buffer = ActionBuffer(self.all_args, self.num_agents)
        elif self.algo == "TPPO":
            from marft.algorithms import TPPOTrainer
            from marft.buffers.token_level_buffer import TokenBuffer

            self.trainer = TPPOTrainer(self.all_args, self.mas)
            self.buffer = TokenBuffer(
                self.all_args, self.num_agents, self.mas.tokenizer.pad_token_id
            )
        elif self.algo == "GRPO":
            from marft.algorithms import GRPOTrainer
            from marft.buffers.grpo_buffer import GRPOBuffer

            self.trainer = GRPOTrainer(self.all_args, self.mas)
            self.buffer = GRPOBuffer(self.all_args, self.num_agents)
        else:
            raise NotImplementedError

        self.run_dir = config["run_dir"]
        self._make_log_dir()
        self.writter = SummaryWriter(self.log_dir)

        # Initialize trajectory augmenter (Phase 2) if coach URL is configured
        self.trajectory_augmenter = None
        coach_url = getattr(self.all_args, "coach_vllm_url", None)
        if coach_url and self.algo == "APPO":
            try:
                from marft.coach import TrajectoryAugmenter

                coach_model = getattr(
                    self.all_args,
                    "coach_model_name",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                )
                self.trajectory_augmenter = TrajectoryAugmenter(
                    coach_model_name=coach_model,
                    vllm_base_url=coach_url,
                )
                print(
                    f"[Runner] Phase 2 coach augmenter initialized: {coach_model} @ {coach_url}"
                )
            except Exception as e:
                print(
                    f"[Runner] WARNING: Failed to init coach augmenter, using naive duplication: {e}"
                )
                self.trajectory_augmenter = None

        # Store resume state for training loop
        self.resume_state = config.get("resume_state", None)

        # Store shared honeypot set reference for state saving/loading
        self.shared_honeypots = config.get("shared_honeypots", None)

        # Checkpointing state trackers
        self.current_episode = 0
        self.current_steps = 0
        self.current_returns = []
        self._should_stop_early = False

    def graceful_stop(self):
        """Signal to gracefully stop the training loop at the end of the current episode."""
        self._should_stop_early = True

    def emergency_save(self):
        """Immediately save the model and training state (useful for uncaught exceptions)."""
        print(
            f"\n[Runner] Executing emergency save at episode {self.current_episode}, steps {self.current_steps}..."
        )
        try:
            self.save(self.current_steps)
            self._save_training_state(
                self.current_episode, self.current_steps, self.current_returns
            )
            print("[Runner] Emergency save completed successfully.\n")
        except Exception as e:
            print(f"[Runner] WARNING: Emergency save failed: {e}\n")

    def _sync_profiles_to_mas(self):
        """Sync current profiles from environment to MAS.

        Called before each inference to ensure MAS has up-to-date profiles
        reflecting remaining honeypots.
        """
        if hasattr(self.envs, "envs") and len(self.envs.envs) > 0:
            env = self.envs.envs[0]
            if hasattr(env, "profiles"):
                self.mas.update_profiles(env.profiles)

    def run(self):
        """
        Main training loop.

        IMPORTANT TERMINOLOGY DISTINCTION:
        ---------------------------------
        - "Training Episode" (outer loop): One PPO update cycle. We collect
          `episode_length` steps of experience, then update the policy.
          The progress bar tracks these (e.g., "Ep 44/100").

        - "Environment Episode": One red team attack attempt. This terminates when:
          1. Terminal success (honeypot access) - early termination with reward
          2. `horizon` steps reached - normal termination

        Since episode_length >> horizon (e.g., 200 >> 5), many environment episodes
        complete within each training episode. The `all_episodic_returns` list
        tracks environment episodes, so it will have many more entries than the
        training episode count shown in the progress bar.

        This is expected behavior and NOT a bug.
        """
        # Dynamic config detection based on env_name
        env_name = getattr(self.all_args, "env_name", "")
        if "blueteam" in env_name:
            from marft.envs.blueteam_sql.blueteam_sql_env import CONFIG as REWARD_CONFIG

            try:
                from marft.envs.blueteam_sql.blueteam_sql_env import get_total_honeypots

                total_honeypots = get_total_honeypots()
            except ImportError:
                total_honeypots = 0
        else:
            from marft.envs.redteam_sql.redteam_sql_env import (
                REWARD_CONFIG,
                get_total_honeypots,
            )

            total_honeypots = get_total_honeypots()

        print("[Runner] Starting environment reset...")
        next_obs = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = next_obs.copy()

        calculated_episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        # Cap at max_episodes from frozen config (auto-stop at 2000)
        # USER_REQUEST: Fix total number of episodes to 2000 no matter what
        episodes = REWARD_CONFIG.max_episodes
        print(
            f"[Runner] Training for {episodes} TRAINING episodes (fixed to config max)"
        )
        if calculated_episodes < episodes:
            print(
                f"[Runner] WARNING: num_env_steps ({self.num_env_steps}) would imply fewer episodes ({calculated_episodes}). Forcing {episodes}."
            )

        print(
            f"[Runner] Each training episode = {self.episode_length} steps; "
            f"environment episode horizon = {self.all_args.horizon} steps"
        )
        print(
            f"[Runner] Expected ~{self.episode_length // self.all_args.horizon} environment episodes per training episode"
        )
        print(f"[Runner] Total honeypots to discover: {total_honeypots}")

        # Handle resume state
        start_episode = 0
        all_episodic_returns = []
        if self.resume_state:
            start_episode = (
                self.resume_state.get("episode", 0) + 1
            )  # Resume from next episode
            all_episodic_returns = self.resume_state.get("all_episodic_returns", [])
            # Restore accessed honeypots to the shared set
            resumed_honeypots = set(self.resume_state.get("accessed_honeypots", []))
            if resumed_honeypots and self.shared_honeypots is not None:
                self.shared_honeypots.update(resumed_honeypots)
                print(
                    f"[Runner] Restored {len(resumed_honeypots)} honeypots to shared tracking"
                )
            print(
                f"[Runner] Resuming from episode {start_episode} with {len(all_episodic_returns)} prior returns"
            )

        progress_bar = tqdm(
            total=episodes,
            initial=start_episode,
            desc="Training",
            position=0,
            leave=True,
        )

        for episode in range(start_episode, episodes):
            # Update trackers for checkpointing
            self.current_episode = episode
            self.current_returns = all_episodic_returns

            # Set current episode on all environments for decay calculation
            self._set_episode_on_envs(episode)

            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )
            self.current_steps = total_num_steps

            # Clear GPU cache once per training episode (not every step - expensive sync)
            torch.cuda.empty_cache()

            # --- Trajectory Harvesting: track successful trajectories ---
            enable_harvesting = getattr(
                self.all_args, "enable_trajectory_harvesting", True
            )
            oversample_factor = getattr(self.all_args, "oversample_factor", 5)
            harvested_trajectories = []  # list of (thread_idx, step, reward, honeypot_ids)

            for step in range(self.episode_length):
                # Pass generation params for GRPO (higher temperature for exploration)
                if self.algo == "GRPO":
                    temperature = getattr(self.all_args, "generation_temperature", 0.8)
                    top_k = getattr(self.all_args, "generation_top_k", 50)
                    rollout_obs, actions, action_tokens, values, log_probs = (
                        self.mas.infer_for_rollout(
                            self.buffer.obs[self.buffer.cur_batch_index, step],
                            temperature=temperature,
                            top_k=top_k,
                        )
                    )
                else:
                    rollout_obs, actions, action_tokens, values, log_probs = (
                        self.mas.infer_for_rollout(
                            self.buffer.obs[self.buffer.cur_batch_index, step]
                        )
                    )
                next_obs, rewards, dones, infos = self.envs.step(actions)

                # insert data into buffer
                data = (
                    next_obs,
                    rollout_obs,
                    rewards,
                    dones,
                    values,
                    actions,
                    action_tokens,
                    log_probs,
                )
                self.insert(data)

                for i in range(self.n_rollout_threads):
                    global_step = total_num_steps + step * self.n_rollout_threads + i

                    if dones[i, 0]:
                        # Environment episode terminated - log cumulative return
                        episodic_return = infos[i]["episodic_return"]
                        episode_length = infos[i].get("episode_length", "?")
                        terminal_success = infos[i].get("terminal_success", False)

                        self.writter.add_scalar(
                            "episodic_return", episodic_return, global_step
                        )
                        all_episodic_returns.append(episodic_return)

                        # Log terminal successes
                        if terminal_success:
                            self.writter.add_scalar(
                                "terminal_success", 1.0, global_step
                            )

                            # === TRAJECTORY HARVESTING: capture successful trajectory ===
                            if enable_harvesting and self.algo == "APPO":
                                harvested_trajectories.append(
                                    {
                                        "thread_idx": i,
                                        "success_step": step,
                                        "reward": float(episodic_return),
                                    }
                                )
                                print(
                                    f"[HARVEST] Captured successful trajectory: "
                                    f"thread={i}, step={step}, reward={episodic_return:.2f}"
                                )

                                # Log lure context for future Blue Team training
                                try:
                                    if hasattr(self.envs, "envs") and i < len(
                                        self.envs.envs
                                    ):
                                        env = self.envs.envs[i]
                                        if hasattr(env, "get_blueteam_context"):
                                            bt_context = env.get_blueteam_context()
                                            # Collect red team actions from buffer
                                            batch = self.buffer.cur_batch_index
                                            red_actions = []
                                            for s in range(step + 1):
                                                act = self.buffer.actions[
                                                    batch, s, i, :
                                                ]
                                                red_actions.append(
                                                    [str(a) for a in act]
                                                )
                                            _log_lure(
                                                self.log_dir,
                                                episode,
                                                i,
                                                list(self.shared_honeypots)
                                                if self.shared_honeypots
                                                else [],
                                                float(episodic_return),
                                                bt_context,
                                                red_actions,
                                            )
                                except Exception as e:
                                    print(f"[HARVEST] Warning: Failed to log lure: {e}")

                        # Log episode length for analysis
                        if episode_length is not None and episode_length != "?":
                            self.writter.add_scalar(
                                "env_episode_length", episode_length, global_step
                            )

                        # Sync profiles after episode reset (honeypots may have been accessed)
                        self._sync_profiles_to_mas()

                        # Plotting disabled - raw data saved via TensorBoard/debug logs
                        # if len(all_episodic_returns) % 5 == 0:
                        #     self._save_reward_plot(all_episodic_returns)

            # === TRAJECTORY HARVESTING: inject oversampled copies ===
            if harvested_trajectories and enable_harvesting and self.algo == "APPO":
                batch = self.buffer.cur_batch_index
                total_injected = 0
                for traj_info in harvested_trajectories:
                    tid = traj_info["thread_idx"]

                    # Phase 2: Use coach augmenter for diverse variations
                    if self.trajectory_augmenter is not None:
                        try:
                            # Collect the successful action texts
                            success_step = traj_info["success_step"]
                            action_texts = []
                            for s in range(self.episode_length):
                                act = self.buffer.actions[batch, s, tid, :]
                                action_texts.append([str(a) for a in act])

                            # Generate diverse variations via 32B coach
                            flat_actions = [
                                acts[0] for acts in action_texts[: success_step + 1]
                            ]
                            augmented = self.trajectory_augmenter.augment_redteam_trajectory_sync(
                                flat_actions, oversample_factor
                            )

                            # --- BATCHED scoring: all variations in one forward pass ---
                            # Build padded action lists for all variations at once
                            all_text_actions_2d = []
                            for var_actions in augmented:
                                padded_actions = (
                                    list(var_actions) + flat_actions[len(var_actions) :]
                                )
                                while len(padded_actions) < self.episode_length:
                                    padded_actions.append("")
                                all_text_actions_2d.append(
                                    [[a] * self.num_agents for a in padded_actions]
                                )

                            # Stack all variations into shape
                            # [n_variations * episode_length, num_agents] for a single batched call
                            n_vars = len(all_text_actions_2d)
                            step_obs = self.buffer.obs[
                                batch, : self.episode_length, tid, :
                            ].copy()
                            # Tile obs to match all variations
                            batched_obs = np.tile(step_obs, (n_vars, 1)).reshape(
                                n_vars * self.episode_length, self.num_agents
                            )
                            batched_actions = [
                                act for var_2d in all_text_actions_2d for act in var_2d
                            ]  # length = n_vars * episode_length

                            # Single forward pass for all variations
                            all_tokens, all_log_probs, all_value_preds = (
                                self.mas.tokenize_and_score_actions(
                                    batched_obs, batched_actions
                                )
                            )

                            # Slice back per-variation and inject
                            ep = self.episode_length
                            for var_idx, text_actions_2d in enumerate(
                                all_text_actions_2d
                            ):
                                sl = slice(var_idx * ep, (var_idx + 1) * ep)
                                trajectory_data = {
                                    "obs": self.buffer.obs[batch, :, tid, :].copy(),
                                    "actions": np.array(text_actions_2d, dtype=object),
                                    "rollout_obs": self.buffer.rollout_obs[
                                        batch, :, tid, :
                                    ].copy(),
                                    "rewards": self.buffer.rewards[
                                        batch, :, tid, :
                                    ].copy(),
                                    "masks": self.buffer.masks[batch, :, tid, :].copy(),
                                    "action_tokens": all_tokens[sl],
                                    "log_probs": all_log_probs[sl],
                                    "value_preds": all_value_preds[sl],
                                }
                                inj = self.buffer.inject_successful_trajectory(
                                    trajectory_data, 1
                                )
                                total_injected += inj

                            print(
                                f"[HARVEST] Phase 2: Injected {total_injected} augmented copies "
                                f"for thread {tid} (reward={traj_info['reward']:.2f})"
                            )
                        except Exception as e:
                            print(
                                f"[HARVEST] Phase 2 augmentation failed, falling back to naive: {e}"
                            )
                            # Fall back to Phase 1 naive duplication
                            trajectory_data = {
                                "obs": self.buffer.obs[batch, :, tid, :].copy(),
                                "actions": self.buffer.actions[batch, :, tid, :].copy(),
                                "rollout_obs": self.buffer.rollout_obs[
                                    batch, :, tid, :
                                ].copy(),
                                "rewards": self.buffer.rewards[batch, :, tid, :].copy(),
                                "masks": self.buffer.masks[batch, :, tid, :].copy(),
                                "action_tokens": self.buffer.action_tokens[
                                    batch, :, tid, :, :
                                ].copy(),
                                "log_probs": self.buffer.action_level_log_probs[
                                    batch, :, tid, :
                                ].copy(),
                                "value_preds": self.buffer.action_level_v_values[
                                    batch, : self.episode_length, tid, :
                                ].copy(),
                            }
                            injected = self.buffer.inject_successful_trajectory(
                                trajectory_data, oversample_factor
                            )
                            total_injected += injected
                    else:
                        # Phase 1: Naive duplication
                        trajectory_data = {
                            "obs": self.buffer.obs[batch, :, tid, :].copy(),
                            "actions": self.buffer.actions[batch, :, tid, :].copy(),
                            "rollout_obs": self.buffer.rollout_obs[
                                batch, :, tid, :
                            ].copy(),
                            "rewards": self.buffer.rewards[batch, :, tid, :].copy(),
                            "masks": self.buffer.masks[batch, :, tid, :].copy(),
                            "action_tokens": self.buffer.action_tokens[
                                batch, :, tid, :, :
                            ].copy(),
                            "log_probs": self.buffer.action_level_log_probs[
                                batch, :, tid, :
                            ].copy(),
                            "value_preds": self.buffer.action_level_v_values[
                                batch, : self.episode_length, tid, :
                            ].copy(),
                        }
                        injected = self.buffer.inject_successful_trajectory(
                            trajectory_data, oversample_factor
                        )
                        total_injected += injected
                        print(
                            f"[HARVEST] Phase 1: Injected {injected}/{oversample_factor} copies "
                            f"for thread {tid} (reward={traj_info['reward']:.2f})"
                        )

                self.writter.add_scalar(
                    "harvest/trajectories_captured",
                    len(harvested_trajectories),
                    total_num_steps,
                )
                self.writter.add_scalar(
                    "harvest/copies_injected", total_injected, total_num_steps
                )

            self.before_update()
            train_infos = self.trainer.train(self.buffer, total_num_steps)
            self.buffer.after_update()

            # save model and training state
            step_increment = self.episode_length * self.n_rollout_threads
            if (
                (episode == episodes - 1)
                or (
                    total_num_steps // self.all_args.save_interval
                    > (total_num_steps - step_increment) // self.all_args.save_interval
                )
                or self._should_stop_early
            ):
                self.save(total_num_steps)
                self._save_training_state(
                    episode, total_num_steps, all_episodic_returns
                )

            # log info
            if episode % self.log_interval == 0:
                avg_step_reward = np.mean(
                    self.buffer.rewards[self.buffer.pre_batch_index, :, :, -1]
                )

                # GRPO-specific: log fraction of zero-variance groups
                if self.algo == "GRPO" and "frac_reward_zero_std" in train_infos:
                    progress_bar.set_description(
                        f"Ep {episode}/{episodes} | steps: {total_num_steps} | reward: {avg_step_reward:.4f} | "
                        f"discovered: {len(self.shared_honeypots)}/{total_honeypots} | "
                        f"zero_var: {train_infos['frac_reward_zero_std']:.2%}"
                    )
                else:
                    progress_bar.set_description(
                        f"Ep {episode}/{episodes} | steps: {total_num_steps} | reward: {avg_step_reward:.4f} | discovered: {len(self.shared_honeypots)}/{total_honeypots}"
                    )
                train_infos["average_step_rewards"] = avg_step_reward
                self.log_train(train_infos, total_num_steps)
            progress_bar.update(1)

            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                self.eval(total_num_steps)

            if self._should_stop_early:
                print(
                    f"\n[Runner] Graceful early stop triggered at episode {episode}. Exiting training loop."
                )
                break

            # NOTE: Early stopping based on honeypots discovered has been removed
            # to force training for the full episode count.

    def _set_episode_on_envs(self, episode: int):
        """Set the current episode on all environments for decay calculation."""
        # ShareDummyVecEnv stores environments in self.envs list
        if hasattr(self.envs, "envs"):
            for env in self.envs.envs:
                if hasattr(env, "current_episode"):
                    env.current_episode = episode
        # Also handle eval_envs if present
        if self.eval_envs and hasattr(self.eval_envs, "envs"):
            for env in self.eval_envs.envs:
                if hasattr(env, "current_episode"):
                    env.current_episode = episode

    def _save_reward_plot(self, all_episodic_returns):
        """Save the episodic returns plot."""
        plt.figure()
        plt.plot(all_episodic_returns)
        plt.title("Total Rewards over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(os.path.join(self.log_dir, "episode_rewards.png"))
        plt.close()

    def _save_training_state(self, episode, total_num_steps, all_episodic_returns):
        """Save training state for resume capability."""
        # Use shared honeypots set directly
        accessed_honeypots = (
            list(self.shared_honeypots) if self.shared_honeypots else []
        )

        state = {
            "episode": episode,
            "total_num_steps": total_num_steps,
            "all_episodic_returns": all_episodic_returns,
            "accessed_honeypots": accessed_honeypots,
        }
        state_file = os.path.join(str(self.run_dir), "training_state.json")
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        print(
            f"[Runner] Training state saved: episode={episode}, steps={total_num_steps}, honeypots={len(accessed_honeypots)}"
        )

    def insert(self, data):
        (
            next_obs,
            rollout_obs,
            rewards,
            dones,
            values,
            actions,
            action_tokens,
            log_probs,
        ) = data
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents), dtype=np.float32
        )
        self.buffer.insert(
            next_obs,
            actions,
            rollout_obs,
            values,
            rewards,
            masks,
            action_tokens,
            log_probs,
        )

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        if self.algo == "GRPO":
            # GRPO doesn't use value bootstrapping, handled in buffer
            pass
        else:
            values = self.mas.get_next_values(
                self.buffer.obs[self.buffer.cur_batch_index, -1]
            )
            self.buffer.compute_gae_and_returns(values)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []

        eval_obs = self.eval_envs.reset()
        while True:
            eval_actions, _ = self.mas.get_actions(np.concatenate(eval_obs))
            eval_actions = np.array(np.split(eval_actions, self.n_eval_rollout_threads))
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions
            )

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(eval_rewards[eval_i])

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {"eval_average_episode_rewards": eval_episode_rewards}
                print("total_num_steps: ", total_num_steps)
                print("eval reward is {}.".format(np.mean(eval_episode_rewards)))
                self.log_eval(eval_env_infos, total_num_steps)
                break

    def _make_log_dir(self):
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.save_dir = str(self.run_dir / "checkpoints/")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def log_eval(self, eval_infos, total_num_steps):
        for k, v in eval_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def save(self, steps):
        """Save the MAS policies and critic networks."""
        self.mas.save(self.save_dir, steps)
        self.trainer.save_optimizers(self.save_dir, steps)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.mas.restore(model_dir)
