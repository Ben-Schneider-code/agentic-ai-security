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
            load_in_4bit=getattr(self.all_args, "load_in_4bit", False),
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

        # SIL coach augmenter REMOVED in redesign (was a major source of OOM
        # and complexity, often outweighing its marginal benefit).
        self.sil_augmenter = None

        # Store resume state for training loop
        self.resume_state = config.get("resume_state", None)

        # Store shared honeypot set reference for state saving/loading
        self.shared_honeypots = config.get("shared_honeypots", None)

        # Checkpointing state trackers
        self.current_episode = 0
        self.current_steps = 0
        self.current_returns = []
        self._should_stop_early = False
        self.exit_reason = "max_episodes_met"

    def graceful_stop(self):
        """Signal to gracefully stop the training loop at the end of the current episode."""
        self.exit_reason = "forced_exit"
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

            if getattr(self, "exit_reason", None) != "forced_exit":
                self.exit_reason = "error_uncaught_exception"
            exit_log_path = os.path.join(str(self.run_dir), "exit_reason.txt")
            with open(exit_log_path, "w") as f:
                f.write(self.exit_reason + "\n")
            print(f"[Runner] Wrote exit reason '{self.exit_reason}' to {exit_log_path}")

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
            from marft.envs.blueteam_sql.blueteam_sql_env import CONFIG as REWARD_CONFIG  # noqa: F401

        from marft.envs.redteam_sql.redteam_sql_env import (
            get_total_honeypots,
            get_honeypot_type,
        )
        total_honeypots = get_total_honeypots()
        honeypot_type = get_honeypot_type()

        print("[Runner] Starting environment reset...")
        next_obs = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = next_obs.copy()

        # Termination is now driven SOLELY by num_env_steps. No max_episodes cap,
        # no convergence-based early stop. The runner consumes the full env-step
        # budget and exits cleanly.
        episodes = max(
            1,
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads,
        )
        print(
            f"[Runner] Training for {episodes} PPO updates "
            f"(num_env_steps={self.num_env_steps}, episode_length={self.episode_length}, "
            f"n_rollout_threads={self.n_rollout_threads})"
        )

        print(
            f"[Runner] Each training episode = {self.episode_length} steps; "
            f"environment episode horizon = {self.all_args.horizon} steps"
        )
        print(
            f"[Runner] Expected ~{self.episode_length // max(self.all_args.horizon, 1)} environment episodes per training episode"
        )
        print(f"[Runner] HONEYPOT_TYPE arm: {honeypot_type} (universe size: {total_honeypots})")

        print("=" * 60)
        print("[Runner] === TRAINING CONFIGURATION ===")
        print(f"  env_name:          {env_name}")
        print(f"  algorithm:         {self.algo}")
        print(f"  model:             {self.all_args.model_name_or_path}")
        print(f"  n_rollout_threads: {self.n_rollout_threads}")
        print(f"  episode_length:    {self.episode_length}")
        print(f"  horizon:           {self.all_args.horizon}")
        print(f"  num_env_steps:     {self.num_env_steps}")
        print(f"  ppo_updates:       {episodes}")
        print(f"  honeypot_type:     {honeypot_type}")
        print(f"  total_honeypots:   {total_honeypots}")
        print(f"  normalization:     {self.all_args.normalization_mode}")
        print(f"  load_in_4bit:      {getattr(self.all_args, 'load_in_4bit', False)}")
        print(f"  load_path:         {self.all_args.load_path}")
        print(f"  save_interval:     {self.all_args.save_interval}")
        print(f"  log_dir:           {self.log_dir}")
        print(f"  run_dir:           {self.run_dir}")
        device = torch.device(self.mas.device)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            alloc_gb = torch.cuda.memory_allocated(device) / 1e9
            print(f"  gpu_device:        {device} ({props.name})")
            print(f"  gpu_total_mem:     {props.total_memory / 1e9:.1f} GB")
            print(f"  gpu_alloc_at_init: {alloc_gb:.1f} GB")
        print("=" * 60)

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

        self.last_honeypot_count = (
            len(self.shared_honeypots) if self.shared_honeypots else 0
        )
        self.last_honeypot_step = (
            start_episode * self.episode_length * self.n_rollout_threads
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
            self._log_gpu_memory("episode_start", total_num_steps)

            self.trainer.prep_rollout()
            for step in range(self.episode_length):
                print(
                    f"[Ep {episode+1}/{episodes} | Step {step+1}/{self.episode_length} | "
                    f"Global: {total_num_steps + step * self.n_rollout_threads}] rollout...",
                    flush=True,
                )
                # Sync profiles so MAS uses the correct turn counter
                self._sync_profiles_to_mas()
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
                        env_episode_length = infos[i].get("episode_length", "?")
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

                        # Log episode length for analysis
                        if env_episode_length is not None and env_episode_length != "?":
                            self.writter.add_scalar(
                                "env_episode_length", env_episode_length, global_step
                            )

                        # Sync profiles after episode reset (honeypots may have been accessed)
                        self._sync_profiles_to_mas()

                        # Plotting disabled - raw data saved via TensorBoard/debug logs
                        # if len(all_episodic_returns) % 5 == 0:
                        #     self._save_reward_plot(all_episodic_returns)

            self._log_gpu_memory("post_rollout", total_num_steps)

            print(
                f"[Ep {episode+1}/{episodes}] PPO update starting...",
                flush=True,
            )
            self.before_update()
            self.trainer.prep_training()
            train_infos = self.trainer.train(self.buffer, total_num_steps)
            self.buffer.after_update()
            self._log_gpu_memory("post_training", total_num_steps)
            print(
                f"[Ep {episode+1}/{episodes}] PPO update done. "
                f"v_loss={train_infos['value_loss']:.4f} p_loss={train_infos['policy_loss']:.4f} "
                f"kl={train_infos['approx_kl']:.4f}",
                flush=True,
            )

            # No early-stop / convergence checks. The runner consumes the entire
                # num_env_steps budget. Tracking honeypot discovery for stdout reporting only.
            current_honeypot_count = (
                len(self.shared_honeypots) if self.shared_honeypots else 0
            )

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
                print(
                    f"[Ep {episode+1}/{episodes}] Saving checkpoint at step {total_num_steps}...",
                    flush=True,
                )
                self.save(total_num_steps)
                self._save_training_state(
                    episode, total_num_steps, all_episodic_returns
                )

            # log info
            if episode % self.log_interval == 0:
                avg_step_reward = np.mean(
                    self.buffer.rewards[self.buffer.pre_batch_index, :, :, -1]
                )

                progress_bar.set_description(
                    f"Ep {episode}/{episodes} | steps: {total_num_steps} | "
                    f"reward: {avg_step_reward:.4f} | "
                    f"discovered: {current_honeypot_count}/{total_honeypots} | "
                    f"reward_avg: {float(np.mean(all_episodic_returns)) if all_episodic_returns else 0:.4f}"
                )
                train_infos["average_step_rewards"] = avg_step_reward
                self.log_train(train_infos, total_num_steps)
            progress_bar.update(1)

            # === Per-epoch security metrics: PVR / BRR / honeypot-found ===
            # Read from reward_debug.jsonl (env writes one line per turn) and
            # print + persist a one-line summary to summary.jsonl.
            try:
                from util.metrics import emit_per_epoch_metrics
                reward_debug_path = os.path.join(self.log_dir, "reward_debug.jsonl")
                summary = emit_per_epoch_metrics(
                    reward_debug_path,
                    prefix=f"[METRICS Ep {episode+1}/{episodes}]",
                )
                if summary:
                    summary["episode"] = episode
                    summary["total_num_steps"] = total_num_steps
                    summary["honeypot_type"] = honeypot_type
                    summary["env_name"] = env_name
                    sj_path = os.path.join(self.log_dir, "summary.jsonl")
                    with open(sj_path, "a") as f:
                        f.write(json.dumps(summary) + "\n")
            except Exception as metric_err:
                print(f"[Runner] WARNING: per-epoch metrics emit failed: {metric_err}")

            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                self.eval(total_num_steps)

            if self._should_stop_early:
                # Only fired by graceful_stop() / emergency_save(); no convergence-based stops.
                print(
                    f"\n[Runner] Forced stop at episode {episode}. Reason: {self.exit_reason}."
                )
                break

        # Log exit reason to file once training loop breaks naturally or early
        exit_log_path = os.path.join(str(self.run_dir), "exit_reason.txt")
        with open(exit_log_path, "w") as f:
            f.write(self.exit_reason + "\n")
        print(f"[Runner] Wrote exit reason '{self.exit_reason}' to {exit_log_path}")

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
        rewards = rewards.reshape(self.n_rollout_threads, self.num_agents)
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env] = 0.0
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

    def _log_gpu_memory(self, phase: str, total_num_steps: int):
        """Log GPU memory usage to TensorBoard and console (when utilization is high)."""
        device = torch.device(self.mas.device)
        allocated_gb = torch.cuda.memory_allocated(device) / 1e9
        reserved_gb = torch.cuda.memory_reserved(device) / 1e9
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        utilization = allocated_gb / total_gb if total_gb > 0 else 0

        self.writter.add_scalars(
            "gpu/allocated_gb", {"gpu/allocated_gb": allocated_gb}, total_num_steps
        )
        self.writter.add_scalars(
            "gpu/reserved_gb", {"gpu/reserved_gb": reserved_gb}, total_num_steps
        )

        if utilization > 0.90:
            print(
                f"[GPU] {phase}: alloc={allocated_gb:.1f}GB res={reserved_gb:.1f}GB "
                f"total={total_gb:.1f}GB ({utilization:.0%})",
                flush=True,
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []

        eval_obs = self.eval_envs.reset()
        while True:
            _, eval_actions, _ = self.mas.get_actions_sequential(eval_obs)
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions
            )

            eval_dones_env = np.all(eval_dones, axis=1)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(eval_infos[eval_i].get("episodic_return", eval_rewards[eval_i]))

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {"eval_average_episode_rewards": eval_episode_rewards}
                # print("total_num_steps: ", total_num_steps)  # in TensorBoard
                # print("eval reward is {}.".format(np.mean(eval_episode_rewards)))  # via self.log_eval()
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
