#!/usr/bin/env python3
"""
Generate all plots for a self-play run.

Usage:
    python util/plot_results.py <selfplay_dir>

Example:
    python util/plot_results.py results-20260322-1530-abc12

For each iteration, runs plot_redteam_results.py and plot_blueteam_results.py
on the respective subdirectories, then runs plot_selfplay_results.py on the
root directory.
"""

import glob
import os
import re
import subprocess
import sys


def find_run_dir(team_dir):
    """Find the run_* directory nested under a redteam/ or blueteam/ dir."""
    matches = glob.glob(os.path.join(team_dir, "**/run_*"), recursive=True)
    dirs = [m for m in matches if os.path.isdir(m)]
    if len(dirs) == 1:
        return dirs[0]
    if len(dirs) > 1:
        print(f"  Warning: multiple run_* dirs found in {team_dir}, using first")
        dirs.sort()
        return dirs[0]
    return None


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <selfplay_dir>")
        sys.exit(1)

    selfplay_dir = sys.argv[1]
    if not os.path.isdir(selfplay_dir):
        print(f"Error: '{selfplay_dir}' is not a directory")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    redteam_script = os.path.join(script_dir, "plot_redteam_results.py")
    blueteam_script = os.path.join(script_dir, "plot_blueteam_results.py")
    selfplay_script = os.path.join(script_dir, "plot_selfplay_results.py")

    # Discover iter_* directories sorted by iteration number
    iter_dirs = []
    for name in os.listdir(selfplay_dir):
        m = re.match(r"iter_(\d+)$", name)
        if m and os.path.isdir(os.path.join(selfplay_dir, name)):
            iter_dirs.append((int(m.group(1)), name))
    iter_dirs.sort()

    if not iter_dirs:
        print(f"Warning: no iter_* directories found in '{selfplay_dir}'")

    failures = []

    for iter_num, iter_name in iter_dirs:
        iter_path = os.path.join(selfplay_dir, iter_name)

        redteam_dir = os.path.join(iter_path, "redteam")
        if os.path.isdir(redteam_dir):
            run_dir = find_run_dir(redteam_dir)
            if run_dir:
                print(f"[{iter_name}] Running plot_redteam_results on {run_dir}")
                result = subprocess.run(
                    [sys.executable, redteam_script, run_dir],
                )
                if result.returncode != 0:
                    failures.append(f"{iter_name}/redteam (exit {result.returncode})")
            else:
                print(f"[{iter_name}] Warning: no run_* dir found in {redteam_dir}")

        blueteam_dir = os.path.join(iter_path, "blueteam")
        if os.path.isdir(blueteam_dir):
            run_dir = find_run_dir(blueteam_dir)
            if run_dir:
                print(f"[{iter_name}] Running plot_blueteam_results on {run_dir}")
                result = subprocess.run(
                    [sys.executable, blueteam_script, run_dir],
                )
                if result.returncode != 0:
                    failures.append(f"{iter_name}/blueteam (exit {result.returncode})")
            else:
                print(f"[{iter_name}] Warning: no run_* dir found in {blueteam_dir}")

    # Run selfplay plot on the root directory
    print(f"Running plot_selfplay_results on {selfplay_dir}")
    result = subprocess.run(
        [sys.executable, selfplay_script, selfplay_dir],
    )
    if result.returncode != 0:
        failures.append(f"selfplay (exit {result.returncode})")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()
