import dataclasses
import re
import sys
import time
import os
import signal
import random
import threading
import numpy as np
from pathlib import Path
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from marft.config import get_config


def get_env_components(env_name):
    if "blueteam" in env_name:
        from marft.envs.blueteam_sql.blueteam_sql_env import (
            BlueTeamSQLEnv as SQLEnv,
            CONFIG as REWARD_CONFIG,
        )

        try:
            from marft.envs.blueteam_sql.blueteam_sql_env import get_total_honeypots
        except ImportError:

            def get_total_honeypots():
                return 0

        return SQLEnv, REWARD_CONFIG, get_total_honeypots
    else:
        from marft.envs.redteam_sql.redteam_sql_env import (
            SQLEnv,
            REWARD_CONFIG,
            get_total_honeypots,
        )

        return SQLEnv, REWARD_CONFIG, get_total_honeypots


from marft.envs.env_wrappers import ShareDummyVecEnv
from marft.runner.shared.sql_runner import SQLRunner as Runner


class ThreadSafeSet:
    """A set wrapper that is safe for concurrent access from ThreadPoolExecutor workers."""

    def __init__(self, initial=()):
        self._set = set(initial)
        self._lock = threading.Lock()

    def add(self, item):
        with self._lock:
            self._set.add(item)

    def update(self, items):
        with self._lock:
            self._set.update(items)

    def __contains__(self, item):
        with self._lock:
            return item in self._set

    def __len__(self):
        with self._lock:
            return len(self._set)

    def __iter__(self):
        with self._lock:
            return iter(self._set.copy())

    def __bool__(self):
        with self._lock:
            return bool(self._set)


def make_train_env(all_args, shared_honeypots=None):
    """Create training environments with shared honeypot tracking.

    Args:
        all_args: Training arguments
        shared_honeypots: Shared set for tracking accessed honeypots across all envs.
                         If None, a new ThreadSafeSet is created.

    Returns:
        tuple: (vec_env, shared_honeypots_set)
    """
    if shared_honeypots is None:
        shared_honeypots = ThreadSafeSet()

    SQLEnv, _, _ = get_env_components(all_args.env_name)

    # vLLM URL must be provided explicitly by the orchestrator (run_training.sh
    # reads it from the per-run actor registry). No localhost fallback: a
    # missing value means we would silently train against the wrong server.
    vllm_url = os.environ.get("STUDENT_VLLM_URL")
    if not vllm_url:
        raise RuntimeError(
            "train_sql.py: STUDENT_VLLM_URL is not set. Launch training via "
            "run_training.sh so the actor vLLM endpoint is wired up."
        )

    def get_env_fn(rank):
        def init_env():
            env = SQLEnv(
                rank=rank,
                model_name=all_args.base_model,
                num_agents=all_args.n_agents,
                horizon=all_args.horizon,
                mode="train",
                dataset_path=all_args.dataset_path,
                log_dir=getattr(all_args, "debug_log_dir", None),
                shared_honeypots=shared_honeypots,  # Pass shared set
                vllm_base_url=vllm_url,
                max_tokens=all_args.victim_max_tokens,
                opponent_model_name=getattr(all_args, "opponent_model_name", None),
                opponent_lora_path=getattr(all_args, "opponent_lora_path", None),
                seed=all_args.seed,
            )
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    print(f"NUMBER OF ROLLOUT THREADS: {all_args.n_rollout_threads}")
    vec_env = ShareDummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
    )
    return vec_env, shared_honeypots


def make_eval_env(all_args):
    # vLLM URL must be provided explicitly by the orchestrator (run_training.sh
    # reads it from the per-run actor registry). No localhost fallback: a
    # missing value means we would silently train against the wrong server.
    vllm_url = os.environ.get("STUDENT_VLLM_URL")
    if not vllm_url:
        raise RuntimeError(
            "train_sql.py: STUDENT_VLLM_URL is not set. Launch training via "
            "run_training.sh so the actor vLLM endpoint is wired up."
        )

    SQLEnv, _, _ = get_env_components(all_args.env_name)

    def get_env_fn(rank):
        def init_env():
            env = SQLEnv(
                rank=rank,
                model_name=all_args.base_model,
                num_agents=all_args.n_agents,
                dataset_path=all_args.dataset_path,
                horizon=all_args.horizon,
                mode="test",
                log_dir=getattr(all_args, "debug_log_dir", None),
                vllm_base_url=vllm_url,
                max_tokens=all_args.victim_max_tokens,
                opponent_model_name=getattr(all_args, "opponent_model_name", None),
                opponent_lora_path=getattr(all_args, "opponent_lora_path", None),
                seed=all_args.seed,
                # Reward config now uses frozen REWARD_CONFIG - no CLI args
            )
            env.seed(all_args.seed + rank * 5000)
            return env

        return init_env

    return ShareDummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    )


def parse_args(args, parser):
    # Ablation knobs added by the redesign. We register them on the existing
    # parser so they show up in --help and feed through to env construction
    # (via env vars set immediately after parse).
    parser.add_argument(
        "--vanilla-size",
        type=int,
        default=None,
        help="Number of plain-style benign queries used for blue training.",
    )
    parser.add_argument(
        "--bordercase-size",
        type=int,
        default=None,
        help="Number of adversarial-style (legitimate) benign queries used for blue training.",
    )
    parser.add_argument(
        "--honeypot-type",
        choices=["rowcol", "row", "col"],
        default=None,
        help="Honeypot ablation arm. Default rowcol (full universe).",
    )
    parser.add_argument(
        "--red-lora-pool",
        type=str,
        default=None,
        help=(
            "Comma-separated list of red LoRA adapter names (matching --lora-modules) "
            "for blue history-based training. e.g. 'red_iter_1,red_iter_2'."
        ),
    )
    parser.add_argument(
        "--opponent-sampler-seed",
        type=int,
        default=None,
        help="Seed for the per-env opponent sampler RNG (combines with rank).",
    )
    all_args = parser.parse_known_args(args)[0]
    all_args.base_model = all_args.model_name_or_path

    # Translate ablation flags to env vars so the BlueTeamSQLEnv (constructed
    # below in make_train_env) picks them up at __init__.
    if all_args.honeypot_type is not None:
        os.environ["HONEYPOT_TYPE"] = all_args.honeypot_type
    if all_args.vanilla_size is not None:
        os.environ["VANILLA_BENIGN_SIZE"] = str(all_args.vanilla_size)
    if all_args.bordercase_size is not None:
        os.environ["BORDERCASE_BENIGN_SIZE"] = str(all_args.bordercase_size)
    if all_args.red_lora_pool:
        os.environ["REDTEAM_LORA_POOL"] = all_args.red_lora_pool
    if all_args.opponent_sampler_seed is not None:
        os.environ["OPPONENT_SAMPLER_SEED"] = str(all_args.opponent_sampler_seed)

    return all_args


def save_args_to_yaml(args, filename="args.yaml"):
    """Save argparse arguments to a YAML file."""
    with open(filename, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)


def _to_yaml_safe(obj):
    """Recursively convert tuples to lists so yaml.safe_dump accepts the value.

    RewardConfig uses tuple-typed fields for immutability. Without this,
    yaml.dump emits ``!!python/tuple`` tags that yaml.safe_load rejects.
    """
    if isinstance(obj, tuple):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    return obj


def save_reward_config_to_yaml(run_dir, all_args):
    """Save immutable reward config to YAML for reproducibility."""
    _, REWARD_CONFIG, get_total_honeypots = get_env_components(all_args.env_name)

    config_dict = dataclasses.asdict(REWARD_CONFIG)
    config_dict["total_honeypots"] = get_total_honeypots()
    config_dict = _to_yaml_safe(config_dict)

    with open(run_dir / "reward_config.yaml", "w") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _checkpoint_is_complete(folder: Path) -> bool:
    # A checkpoint is usable only if at least one role subdir contains the LoRA
    # adapter_config.json. Partial saves (process killed mid-save) can leave
    # role dirs with just README.md, which breaks PeftModel.from_pretrained.
    role_dirs = [p for p in folder.iterdir() if p.is_dir()]
    if not role_dirs:
        return False
    return all((role / "adapter_config.json").is_file() for role in role_dirs)


def find_latest_checkpoint(run_dir):
    """Find the latest *complete* checkpoint in a run directory.

    Skips partial / torn checkpoints left behind when a save was interrupted
    (e.g. by a CUDA crash between mkdir and adapter write).

    Returns:
        tuple: (checkpoint_path, steps) or (None, 0) if no checkpoint found
    """
    checkpoint_dir = Path(run_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None, 0

    checkpoints = []
    for folder in checkpoint_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("steps_"):
            try:
                steps = int(folder.name.split("_")[1])
                checkpoints.append((folder, steps))
            except (ValueError, IndexError):
                continue

    if not checkpoints:
        return None, 0

    checkpoints.sort(key=lambda x: x[1], reverse=True)
    for folder, steps in checkpoints:
        if _checkpoint_is_complete(folder):
            return str(folder), steps
        print(
            f">>> Skipping incomplete checkpoint {folder} (missing adapter_config.json)"
        )
    return None, 0


def load_training_state(run_dir):
    """Load training state from a run directory.

    Returns:
        dict with keys: start_episode, total_num_steps, all_episodic_returns
        or None if no state file found
    """
    import json

    state_file = Path(run_dir) / "training_state.json"
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return None


def build_run_dir(all_args):
    if getattr(all_args, "results_dir", None):
        base_path = Path(all_args.results_dir)
    else:
        base_path = Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/scripts/results"
        )
    run_dir = (
        base_path
        / all_args.experiment_name
        / all_args.base_model
        / all_args.dataset_name
        / all_args.algorithm_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        curr_run = "run_1"
    else:
        exst_run_nums = [
            int(m.group(1))
            for folder in run_dir.iterdir()
            if (m := re.match(r'^run_(\d+)', folder.name))
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run_1"
        else:
            curr_run = "run_%i" % (max(exst_run_nums) + 1)
    curr_run += f"_agent#{all_args.n_agents}_seed{all_args.seed}"
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Check for an optional experiment label file at the repo root.
    label_path = Path(__file__).resolve().parents[3] / "experiment_label.txt"
    if label_path.exists():
        try:
            import shutil

            shutil.copy(label_path, run_dir / "experiment_label.txt")
        except Exception as e:
            print(f"Warning: Failed to copy experiment label: {e}")

    print(f"Saving results to {run_dir}")
    return run_dir


def main(args):
    print(">>> Starting main execution of train_sql.py")
    parser = get_config()
    all_args = parse_args(args, parser)
    print(
        f">>> Arguments parsed. Experiment: {all_args.experiment_name}, Algorithm: {all_args.algorithm_name}"
    )

    # Handle resume vs new run
    resume_state = None
    if all_args.resume_run_dir:
        run_dir = Path(all_args.resume_run_dir)
        if not run_dir.exists():
            raise ValueError(f"Resume directory does not exist: {run_dir}")

        # Find the latest checkpoint
        checkpoint_path, checkpoint_steps = find_latest_checkpoint(run_dir)
        if checkpoint_path:
            print(
                f">>> Resuming from checkpoint: {checkpoint_path} (steps: {checkpoint_steps})"
            )
            # Set load_path so MAS and trainer load the checkpoint
            all_args.load_path = checkpoint_path

            # Load training state for episode tracking
            resume_state = load_training_state(run_dir)
            if resume_state:
                print(
                    f">>> Loaded training state: episode={resume_state.get('episode', 0)}, "
                    f"returns_count={len(resume_state.get('all_episodic_returns', []))}"
                )
        else:
            print(">>> No checkpoint found in resume directory, starting fresh")
    else:
        run_dir = build_run_dir(all_args)
        save_args_to_yaml(all_args, run_dir / "args.yaml")
        save_reward_config_to_yaml(run_dir, all_args)

    all_args.run_dir = str(run_dir)
    # Create debug logs directory next to logs (which is handled by runner)
    # We want it adjacent to logs/, so inside results/
    debug_log_dir = run_dir / "debug_logs"
    if not debug_log_dir.exists():
        debug_log_dir.mkdir(parents=True, exist_ok=True)
    all_args.debug_log_dir = str(debug_log_dir)
    print(f">>> Debug logs will be saved to: {all_args.debug_log_dir}")

    # seed
    # Only seed the training device — manual_seed_all initializes CUDA contexts on ALL
    # visible GPUs, leaking ~0.5-1GB onto other GPUs for the entire training duration.
    _training_gpu_env = os.environ.get("TRAINING_GPU")
    if not _training_gpu_env:
        raise RuntimeError(
            "train_sql.py: TRAINING_GPU is not set. Launch training via "
            "run_training.sh so the training GPU index is wired up."
        )
    _training_gpu = int(_training_gpu_env)
    print(f">>> Setting seed to {all_args.seed}")
    torch.manual_seed(all_args.seed)
    with torch.cuda.device(_training_gpu):
        torch.cuda.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)

    print(">>> Creating training environment...")
    envs, shared_honeypots = make_train_env(all_args)
    print(">>> Training environment created with shared honeypot tracking.")

    eval_envs = None
    if all_args.use_eval:
        print(">>> Creating eval environments...")
        eval_envs = make_eval_env(all_args)
        print(">>> Eval environments created.")

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": envs.n_agents if envs is not None else 1,
        "run_dir": run_dir,
        "resume_state": resume_state,
        "shared_honeypots": shared_honeypots,  # Pass to runner for state saving/loading
    }

    print(">>> Initializing Runner...")
    runner = Runner(config)
    print(">>> Runner initialized. Starting run() loop...")

    # Setup graceful stop signal handlers
    interrupt_count = [0]

    def signal_handler(signum, frame):
        interrupt_count[0] += 1
        sig_name = (
            signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        )
        if interrupt_count[0] >= 2:
            print(f"\n>>> Received {sig_name} multiple times. Forcing immediate exit!")
            raise KeyboardInterrupt(f"Forced exit from {sig_name}")

        print(
            f"\n>>> Received {sig_name}. Initiating graceful shutdown at end of current episode..."
        )
        runner.graceful_stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        runner.run()
        print(">>> Runner run() completed.")
    except Exception as e:
        import traceback

        print("\n>>> UNEXPECTED EXCEPTION DURING TRAINING:")
        traceback.print_exc()
        print(">>> Triggering emergency save before crashing...")
        runner.emergency_save()
        raise
    except KeyboardInterrupt:
        print("\n>>> FORCED IMMEDIATE EXIT (KEYBOARD INTERRUPT):")
        runner.exit_reason = "forced_exit"
        runner.emergency_save()
        raise
    finally:
        # post process
        if envs is not None:
            print(">>> Closing environments...")
            envs.close()

        print(">>> Exporting scalars and closing writer...")
        try:
            runner.writter.export_scalars_to_json(os.path.join(runner.log_dir, "summary.json"))
            runner.writter.close()
        except Exception as e:
            print(f">>> Failed to close writer: {e}")
        print(">>> Main execution completed.")


if __name__ == "__main__":
    _start_time = time.time()
    _start_wall = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_start_time))
    print(f">>> train_sql.py started at {_start_wall}")

    def _print_elapsed():
        elapsed = time.time() - _start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        end_wall = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(
            f">>> train_sql.py finished at {end_wall}  (total run time: {h}h {m}m {s}s)"
        )

    try:
        main(sys.argv[1:])
        _print_elapsed()
    except Exception:
        import traceback

        _print_elapsed()
        print("\n\n" + "=" * 50, file=sys.stderr)
        print("CRITICAL ERROR IN TRAIN_SQL.PY", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=" * 50 + "\n", file=sys.stderr)
        sys.exit(1)
