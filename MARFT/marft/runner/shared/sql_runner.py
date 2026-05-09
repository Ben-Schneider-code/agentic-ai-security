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


def _log_augmentation(
    log_dir: str,
    episode: int,
    thread_idx: int,
    coach_model: str,
    action_details: list,
    n_injected: int,
):
    """Append a coach augmentation event to coach_augment_log.jsonl.

    Each record stores the full per-action detail: original text, every raw
    variation the coach generated, and the quality-gate verdict (cosine_sim,
    jaccard_sim, accepted/rejected reason) for each one.  Existing records are
    never modified; the file is always opened in append mode.

    Schema (one JSON object per line)::

        {
          "episode": int,
          "thread": int,
          "timestamp": str,          # ISO-8601 UTC
          "coach_model": str,
          "n_injected": int,
          "actions": [
            {
              "original_action": str,
              "raw_variations": [str, ...],   # all strings returned by coach
              "stats": {
                "total": int,
                "passed": int,
                "rejected_semantic": int,
                "rejected_diversity": int,
                "padded": int,              # slots filled with original
                "details": [
                  {
                    "text":        str,
                    "cosine_sim":  float | null,
                    "jaccard_sim": float | null,
                    "verdict":     "accepted" | "rejected",
                    "reason":      "semantic" | "diversity" | null
                  }, ...
                ]
              }
            }, ...
          ]
        }
    """
    import datetime

    entry = {
        "episode": episode,
        "thread": thread_idx,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "coach_model": coach_model,
        "n_injected": n_injected,
        "actions": action_details,
    }
    log_path = os.path.join(log_dir, "coach_augment_log.jsonl")
    with open(log_path, "a") as f:
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

        # Initialize SIL coach augmenter (Phase 2) if coach URL is configured
        self.sil_augmenter = None
        coach_url = getattr(self.all_args, "coach_vllm_url", None)
        is_redteam = "blueteam" not in getattr(self.all_args, "env_name", "")
        if coach_url and self.algo == "APPO" and is_redteam:
            try:
                from marft.coach import SILCoachAugmenter

                coach_model = getattr(
                    self.all_args,
                    "coach_model_name",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                )
                self.sil_augmenter = SILCoachAugmenter(
                    coach_model_name=coach_model,
                    vllm_base_url=coach_url,
                )
                print(
                    f"[Runner] SIL Phase 2 coach augmenter initialized: {coach_model} @ {coach_url}"
                )
            except Exception as e:
                print(
                    f"[Runner] WARNING: Failed to init SIL coach augmenter, using naive duplication: {e}"
                )
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
        # Cap at max_episodes from frozen config (auto-stop at configured limit)
        # USER_REQUEST: Fix total number of episodes to max_episodes no matter what
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

        # Pre-compute SIL info for startup summary
        _sil_enabled = (
            getattr(self.all_args, "enable_sil", False)
            and "blueteam" not in env_name
        )
        if self.sil_augmenter:
            _coach_summary = (
                f"{getattr(self.all_args, 'coach_model_name', 'N/A')}"
                f" @ {getattr(self.all_args, 'coach_vllm_url', 'N/A')}"
            )
        else:
            _coach_summary = "None"
        print("=" * 60)
        print("[Runner] === TRAINING CONFIGURATION ===")
        print(f"  env_name:          {env_name}")
        print(f"  algorithm:         {self.algo}")
        print(f"  model:             {self.all_args.model_name_or_path}")
        print(f"  n_rollout_threads: {self.n_rollout_threads}")
        print(f"  episode_length:    {self.episode_length}")
        print(f"  horizon:           {self.all_args.horizon}")
        print(f"  max_episodes:      {episodes}")
        print(f"  total_honeypots:   {total_honeypots}")
        print(f"  enable_sil:        {_sil_enabled}")
        print(f"  sil_coach:         {_coach_summary}")
        print(f"  oversample_factor: {getattr(self.all_args, 'oversample_factor', 5)}")
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

            # --- Self-Imitation Learning: track successful trajectories ---
            enable_sil = (
                getattr(self.all_args, "enable_sil", False)
                and "blueteam" not in getattr(self.all_args, "env_name", "")
            )
            oversample_factor = getattr(self.all_args, "oversample_factor", 5)
            sil_successes = []  # list of (thread_idx, step, reward)

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

                            # === SELF-IMITATION LEARNING: capture successful trajectory ===
                            if enable_sil and self.algo == "APPO":
                                sil_successes.append(
                                    {
                                        "thread_idx": i,
                                        "success_step": step,
                                        "reward": float(episodic_return),
                                    }
                                )
                                # print(  # aggregate in TensorBoard sil/successes_captured
                                #     f"[SIL] Captured successful trajectory: "
                                #     f"thread={i}, step={step}, reward={episodic_return:.2f}"
                                # )

                                # Log lure context for future Blue Team training
                                try:
                                    if hasattr(self.envs, "envs") and i < len(
                                        self.envs.envs
                                    ):
                                        env = self.envs.envs[i]
                                        if hasattr(env, "get_blueteam_context"):
                                            bt_context = env.get_blueteam_context()
                                            # Collect red team actions from buffer
                                            # Use episode_length to slice only the
                                            # current env episode's steps, not all
                                            # training-window steps 0..step.
                                            batch = self.buffer.cur_batch_index
                                            red_actions = []
                                            ep_len = env_episode_length if isinstance(env_episode_length, int) else (step + 1)
                                            ep_start = max(0, step + 1 - ep_len)
                                            for s in range(ep_start, step + 1):
                                                act = self.buffer.actions[
                                                    batch, s, i, :
                                                ]
                                                red_actions.append(
                                                    [str(a) for a in act]
                                                )
                                            episode_hp_ids = []
                                            if hasattr(self.envs, "envs") and i < len(self.envs.envs):
                                                env_i = self.envs.envs[i]
                                                if hasattr(env_i, "episode_honeypot_ids"):
                                                    episode_hp_ids = list(env_i.episode_honeypot_ids)
                                            _log_lure(
                                                self.log_dir,
                                                episode,
                                                i,
                                                episode_hp_ids,
                                                float(episodic_return),
                                                bt_context,
                                                red_actions,
                                            )
                                except Exception as e:
                                    print(f"[SIL] Warning: Failed to log lure: {e}")

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

            # === SELF-IMITATION LEARNING: inject oversampled copies into on-policy batch ===
            if sil_successes and enable_sil and self.algo == "APPO":
                batch = self.buffer.cur_batch_index
                total_injected = 0
                for traj_info in sil_successes:
                    tid = traj_info["thread_idx"]
                    traj_injected = 0

                    # Phase 2: Use SIL coach augmenter for diverse variations
                    if self.sil_augmenter is not None:
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
                            augmented, action_details = (
                                self.sil_augmenter.augment_redteam_trajectory_sync(
                                    flat_actions, oversample_factor
                                )
                            )

                            # --- CHUNKED scoring: one variation at a time to avoid OOM ---
                            # GPU 2 already holds the student model + critic (~65 GB).
                            # Scoring all variations at once causes OOM, so we process
                            # each variation individually (episode_length items per pass).
                            torch.cuda.empty_cache()

                            step_obs = self.buffer.obs[
                                batch, : self.episode_length, tid, :
                            ].copy()

                            for var_actions in augmented:
                                padded_actions = (
                                    list(var_actions) + flat_actions[len(var_actions) :]
                                )
                                while len(padded_actions) < self.episode_length:
                                    padded_actions.append("")
                                text_actions_2d = [
                                    [a] * self.num_agents for a in padded_actions
                                ]

                                # Score this single variation (episode_length items)
                                var_tokens, var_log_probs, var_value_preds = (
                                    self.mas.tokenize_and_score_actions(
                                        step_obs, text_actions_2d
                                    )
                                )

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
                                    "action_tokens": var_tokens,
                                    "log_probs": var_log_probs,
                                    "value_preds": var_value_preds,
                                }
                                inj = self.buffer.sil_inject(
                                    trajectory_data, 1
                                )
                                traj_injected += inj
                                total_injected += inj

                            # print(  # aggregate in TensorBoard sil/copies_injected
                            #     f"[SIL] Phase 2: Injected {traj_injected} augmented copies "
                            #     f"for thread {tid} (reward={traj_info['reward']:.2f})"
                            # )

                            # Write structured debug log for this augmentation event
                            try:
                                coach_model_name = getattr(
                                    self.all_args,
                                    "coach_model_name",
                                    "unknown",
                                )
                                _log_augmentation(
                                    self.log_dir,
                                    episode,
                                    tid,
                                    coach_model_name,
                                    action_details,
                                    traj_injected,
                                )
                            except Exception as log_err:
                                print(
                                    f"[SIL] Warning: Failed to write coach_augment_log: {log_err}"
                                )
                        except Exception as e:
                            print(
                                f"[SIL] Phase 2 augmentation failed, falling back to naive: {e}"
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
                            injected = self.buffer.sil_inject(
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
                        injected = self.buffer.sil_inject(
                            trajectory_data, oversample_factor
                        )
                        total_injected += injected
                        # print(  # aggregate in TensorBoard sil/copies_injected
                        #     f"[SIL] Phase 1: Injected {injected}/{oversample_factor} copies "
                        #     f"for thread {tid} (reward={traj_info['reward']:.2f})"
                        # )

                self.writter.add_scalar(
                    "sil/successes_captured",
                    len(sil_successes),
                    total_num_steps,
                )
                self.writter.add_scalar(
                    "sil/copies_injected", total_injected, total_num_steps
                )

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

            current_honeypot_count = (
                len(self.shared_honeypots) if self.shared_honeypots else 0
            )
            if current_honeypot_count > self.last_honeypot_count:
                self.last_honeypot_count = current_honeypot_count
                self.last_honeypot_step = total_num_steps

            if total_honeypots > 0:
                if current_honeypot_count >= total_honeypots:
                    self.exit_reason = "all_honeypots_accessed"
                    self._should_stop_early = True
                elif total_num_steps - self.last_honeypot_step >= 1000:
                    self.exit_reason = "no_new_honeypot_for_1000_steps"
                    self._should_stop_early = True
            elif "blueteam" in env_name:
                # ──────────────────────────────────────────────────────────
                # Blueteam halting logic (total_honeypots == 0 branch)
                # All thresholds read from REWARD_CONFIG (single source of truth).
                # BLUETEAM_DISABLE_EARLY_STOP=1 bypasses convergence-based halts
                # (decisive_win and plateau) while keeping the hard episode cap,
                # so the full training budget is always consumed. Default (unset)
                # preserves existing behaviour exactly.
                # ──────────────────────────────────────────────────────────
                _disable_convergence = os.environ.get("BLUETEAM_DISABLE_EARLY_STOP", "") == "1"

                if not _disable_convergence:
                    _dw_thresh = REWARD_CONFIG.decisive_win_threshold
                    _dw_win = REWARD_CONFIG.decisive_win_window

                    if len(all_episodic_returns) >= _dw_win:
                        recent_returns = all_episodic_returns[-_dw_win:]
                        avg_return = float(np.mean(recent_returns))
                        if avg_return >= _dw_thresh:
                            self.exit_reason = "blueteam_decisive_win"
                            self._should_stop_early = True
                            print(
                                f"\n[Runner] blueteam_decisive_win: rolling-{_dw_win} avg "
                                f"= {avg_return:.4f} >= {_dw_thresh} → halting."
                            )

                    # Plateau logic
                    _plat_win = REWARD_CONFIG.plateau_window
                    _plat_min = REWARD_CONFIG.plateau_min_improvement
                    if (
                        not self._should_stop_early
                        and len(all_episodic_returns) >= _plat_win * 2
                    ):
                        recent_avg = float(np.mean(all_episodic_returns[-_plat_win:]))
                        past_avg = float(
                            np.mean(
                                all_episodic_returns[-_plat_win * 2 : -_plat_win]
                            )
                        )
                        if recent_avg - past_avg < _plat_min:
                            self.exit_reason = "blueteam_plateaued"
                            self._should_stop_early = True
                            print(
                                f"\n[Runner] blueteam_plateaued: recent={recent_avg:.4f} "
                                f"past={past_avg:.4f} improvement={recent_avg - past_avg:.4f} < {_plat_min} → halting."
                            )

                # Hard episode limit — always enforced regardless of BLUETEAM_DISABLE_EARLY_STOP.
                # BLUETEAM_MAX_TRAINING_EPISODES env var overrides the config default
                # (used by ablations to cap training at a reduced budget).
                _max_eps_override = os.environ.get("BLUETEAM_MAX_TRAINING_EPISODES", "")
                _max_eps = int(_max_eps_override) if _max_eps_override else REWARD_CONFIG.max_training_episodes
                if not self._should_stop_early and (episode + 1) >= _max_eps:
                    self.exit_reason = "blueteam_max_episodes_reached"
                    self._should_stop_early = True
                    print(
                        f"\n[Runner] blueteam_max_episodes_reached: {episode + 1} >= {_max_eps} → halting."
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

                # GRPO-specific: log fraction of zero-variance groups
                if self.algo == "GRPO" and "frac_reward_zero_std" in train_infos:
                    progress_bar.set_description(
                        f"Ep {episode}/{episodes} | steps: {total_num_steps} | reward: {avg_step_reward:.4f} | "
                        f"discovered: {len(self.shared_honeypots)}/{total_honeypots} | "
                        f"zero_var: {train_infos['frac_reward_zero_std']:.2%}"
                    )
                elif "blueteam" in env_name:
                    # Compute and show blueteam decisive-win metric in progress bar
                    _n_ep = len(all_episodic_returns)
                    _pbar_dw_win = REWARD_CONFIG.decisive_win_window
                    _pbar_dw_thr = REWARD_CONFIG.decisive_win_threshold
                    if _n_ep >= _pbar_dw_win:
                        _dw_avg = float(np.mean(all_episodic_returns[-_pbar_dw_win:]))
                        _dw_str = f"dw_avg={_dw_avg:.3f}/{_pbar_dw_thr}"
                    else:
                        _dw_avg = float(np.mean(all_episodic_returns)) if _n_ep > 0 else 0.0
                        _dw_str = f"dw_avg={_dw_avg:.3f}/{_pbar_dw_thr} ({_n_ep}<{_pbar_dw_win}ep)"
                    progress_bar.set_description(
                        f"Ep {episode}/{episodes} | steps: {total_num_steps} | reward: {avg_step_reward:.4f} | "
                        f"discovered: {len(self.shared_honeypots)}/{total_honeypots} | {_dw_str}"
                    )
                else:
                    progress_bar.set_description(
                        f"Ep {episode}/{episodes} | steps: {total_num_steps} | reward: {avg_step_reward:.4f} | "
                        f"reward_avg: {float(np.mean(all_episodic_returns)):.4f}"
                    )
                train_infos["average_step_rewards"] = avg_step_reward
                self.log_train(train_infos, total_num_steps)
            progress_bar.update(1)

            if self.all_args.use_eval and episode % self.all_args.eval_interval == 0:
                self.eval(total_num_steps)

            if self._should_stop_early:
                print(
                    f"\n[Runner] Graceful early stop triggered at episode {episode}. Reason: {self.exit_reason}. Exiting training loop."
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
