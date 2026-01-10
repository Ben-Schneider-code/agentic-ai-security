# Run this with something like this:
# python util/parse_rewards.py results/sample_redteam_sql/Meta-Llama-3-8B-Instruct/None/APPO/run_1_agent#1_seed10/logs

import os
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def parse_rewards(log_dir):
    # Find tfevents file
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return

    event_files = [
        f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents")
    ]
    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    # Use the most recent file if multiple
    event_file = os.path.join(log_dir, sorted(event_files)[-1])
    print(f"Parsing {event_file}...")

    # Load the event file
    # size_guidance sets how many events to load. 0 means all.
    ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags()["scalars"]
    if "episodic_return" not in tags:
        print("Tag 'episodic_return' not found in logs.")
        print("Available tags:", tags)
        return

    events = ea.Scalars("episodic_return")

    steps = [e.step for e in events]
    rewards = [e.value for e in events]

    # Calculate cumulative sum
    cumulative_rewards = np.cumsum(rewards)

    print(f"Found {len(rewards)} episodes.")
    print(f"{'Step':<10} {'Reward':<10} {'Cumulative':<10}")
    for s, r, c in zip(steps, rewards, cumulative_rewards):
        print(f"{s:<10} {r:<10.4f} {c:<10.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cumulative_rewards, marker="o", linestyle="-")
    plt.title("Cumulative Rewards over Episodes")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)

    output_plot = os.path.join(log_dir, "cumulative_rewards.png")
    plt.savefig(output_plot)
    print(f"\nPlot saved to {output_plot}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_rewards.py <log_dir>")
        print("Example: python parse_rewards.py results/sample_redteam_sql/.../logs")
        sys.exit(1)

    log_dir = sys.argv[1]
    parse_rewards(log_dir)
