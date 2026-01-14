#!/usr/bin/env python
import sys
import os
import numpy as np
from pathlib import Path
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from marft.config import get_config
from marft.envs.redteam_sql.redteam_sql_env import SQLEnv
from marft.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from marft.runner.shared.redteam_sql_runner import RedTeamSQLRunner as Runner


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = SQLEnv(
                rank=rank,
                model_name=all_args.base_model,
                num_agents=all_args.n_agents,
                profile_path=all_args.profile_path,
                dataset_path=all_args.dataset_path,
                horizon=all_args.horizon,
                mode="train",
                log_dir=getattr(all_args, "debug_log_dir", None),
            )
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    print(f"NUMBER OF ROLLOUT THREADS: {all_args.n_rollout_threads}")
    return ShareDummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = SQLEnv(
                rank=rank,
                model_name=all_args.base_model,
                num_agents=all_args.n_agents,
                profile_path=all_args.profile_path,
                dataset_path=all_args.dataset_path,
                horizon=all_args.horizon,
                mode="test",
                log_dir=getattr(all_args, "debug_log_dir", None),
            )
            env.seed(all_args.seed + rank * 5000)
            return env

        return init_env

    return ShareDummyVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
    )


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    all_args.base_model = Path(all_args.model_name_or_path).parts[-1]
    return all_args


def save_args_to_yaml(args, filename="args.yaml"):
    """Save argparse arguments to a YAML file."""
    with open(filename, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)


def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint in a run directory.

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

    # Sort by steps and return the latest
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return str(checkpoints[0][0]), checkpoints[0][1]


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
    run_dir = (
        Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/scripts/results"
        )
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
            int(str(folder.name).split("_")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run_1"
        else:
            curr_run = "run_%i" % (max(exst_run_nums) + 1)
    curr_run += f"_agent#{all_args.n_agents}_seed{all_args.seed}"
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    print(f"Saving results to {run_dir}")
    return run_dir


def main(args):
    print(">>> Starting main execution of train_redteam_sql.py")
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

    all_args.run_dir = str(run_dir)
    # Create debug logs directory next to logs (which is handled by runner)
    # We want it adjacent to logs/, so inside results/
    debug_log_dir = run_dir / "debug_logs"
    if not debug_log_dir.exists():
        debug_log_dir.mkdir(parents=True, exist_ok=True)
    all_args.debug_log_dir = str(debug_log_dir)
    print(f">>> Debug logs will be saved to: {all_args.debug_log_dir}")

    # seed
    print(f">>> Setting seed to {all_args.seed}")
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    print(">>> Creating training environment...")
    envs = make_train_env(all_args)
    print(">>> Training environment created.")

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": envs.n_agents if envs is not None else 1,
        "run_dir": run_dir,
        "resume_state": resume_state,
    }

    print(">>> Initializing Runner...")
    runner = Runner(config)
    print(">>> Runner initialized. Starting run() loop...")
    runner.run()
    print(">>> Runner run() completed.")

    # post process
    if envs is not None:
        print(">>> Closing environments...")
        envs.close()

    print(">>> Exporting scalars and closing writer...")
    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()
    print(">>> Main execution completed successfully.")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        import traceback

        print("\n\n" + "=" * 50, file=sys.stderr)
        print("CRITICAL ERROR IN TRAIN_REDTEAM_SQL.PY", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=" * 50 + "\n", file=sys.stderr)
        sys.exit(1)
