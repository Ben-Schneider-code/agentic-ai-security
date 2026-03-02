#!/usr/bin/env python3
import os
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Visualize successful honeypot examples from training logs."
    )
    parser.add_argument(
        "run_dir",
        help="Path to the training run directory containing reward_debug.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of examples to show (0 = unlimited)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save output to a markdown file instead of printing",
    )
    args = parser.parse_args()

    candidates = [
        os.path.join(args.run_dir, "debug_logs", "reward_debug.jsonl"),
        os.path.join(args.run_dir, "reward_debug.jsonl"),
    ]

    log_file = None
    for c in candidates:
        if os.path.exists(c):
            log_file = c
            break

    if not log_file:
        print(f"Error: Could not find reward_debug.jsonl in {args.run_dir}")
        sys.exit(1)

    successful_examples = []

    print(f"Parsing logs from: {log_file}")
    with open(log_file, "r") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)

                tier = data.get("outcome_tier", "")

                new_hps = data.get("new_honeypots_accessed")
                if new_hps is None:
                    old_hp = data.get("new_honeypot_accessed")
                    if old_hp:
                        new_hps = [old_hp]
                    else:
                        new_hps = []

                # We want any honeypot access (unique or repeated)
                # Sometime honeypots are not in the tier but in the breakdown
                breakdown = data.get("reward_breakdown", {})
                attempted_hps = breakdown.get("attempted_but_failed_honeypots", [])

                if new_hps or tier in ["honeypot_access", "repeated_honeypot"]:
                    honeypots_str = (
                        ", ".join(str(hp) for hp in new_hps if hp)
                        if new_hps
                        else "Unknown (Repeated?)"
                    )

                    exec_res = str(data.get("execution_result", ""))
                    lines_lst = exec_res.split("\n")
                    if len(lines_lst) > 15:
                        exec_res = (
                            "\n".join(lines_lst[:15])
                            + f"\n... [{len(lines_lst) - 15} more lines truncated]"
                        )
                    elif len(exec_res) > 800:
                        exec_res = exec_res[:800] + "\n... [truncated length]"

                    example = {
                        "episode": data.get("episode", data.get("step_count", i + 1)),
                        "tier": tier,
                        "honeypots": honeypots_str,
                        "red_team_input": data.get("red_team_input", ""),
                        "victim_response": data.get(
                            "victim_full_response", data.get("victim_response", "")
                        ),
                        "execution_result": exec_res,
                        "reward": data.get("final_reward", 0.0),
                    }
                    successful_examples.append(example)
            except json.JSONDecodeError:
                continue

    if not successful_examples:
        print("No successful honeypot accesses found in the log.")
        sys.exit(0)

    print(f"Found {len(successful_examples)} successful honeypot accesses.")

    if args.save:
        with open(args.save, "w") as f:
            f.write(f"# Successful Honeypot Examples ({len(successful_examples)})\n\n")
            for idx, ex in enumerate(successful_examples):
                if args.limit and idx >= args.limit:
                    break

                f.write(f"## Example {idx + 1} (Episode {ex['episode']})\n")
                f.write(f"- **Tier**: {ex['tier']}\n")
                f.write(f"- **Honeypots Accessed**: {ex['honeypots']}\n")
                f.write(f"- **Reward**: {ex['reward']:.4f}\n\n")

                f.write("### Red Team Input (The Attack)\n")
                f.write("```text\n")
                f.write(f"{ex['red_team_input']}\n")
                f.write("```\n\n")

                f.write("### Victim Response (Generated SQL)\n")
                f.write("```sql\n")
                f.write(f"{ex['victim_response']}\n")
                f.write("```\n\n")

                f.write("### Execution Result\n")
                f.write("```text\n")
                f.write(f"{ex['execution_result']}\n")
                f.write("```\n\n")
                f.write("---\n\n")
        print(f"Saved visualization to {args.save}")
    else:
        # Try rich for pretty terminal output, fallback to standard print
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax

            console = Console()

            for idx, ex in enumerate(successful_examples):
                if args.limit and idx >= args.limit:
                    break

                console.rule(
                    f"[bold green]Example {idx + 1} (Episode {ex['episode']})[/bold green]"
                )
                console.print(
                    f"[bold]Tier:[/bold] {ex['tier']} | [bold]Honeypots:[/bold] {ex['honeypots']} | [bold]Reward:[/bold] {ex['reward']:.4f}"
                )

                console.print("\n[bold cyan]Red Team Input (The Attack):[/bold cyan]")
                console.print(
                    Panel(
                        str(ex["red_team_input"]), title="Prompt", border_style="cyan"
                    )
                )

                console.print(
                    "\n[bold yellow]Victim Response (Generated SQL):[/bold yellow]"
                )
                console.print(
                    Panel(
                        Syntax(
                            str(ex["victim_response"]),
                            "sql",
                            theme="monokai",
                            word_wrap=True,
                        ),
                        title="SQL Response",
                        border_style="yellow",
                    )
                )

                console.print("\n[bold magenta]Execution Result:[/bold magenta]")
                console.print(
                    Panel(
                        str(ex["execution_result"]),
                        title="DB Output",
                        border_style="magenta",
                    )
                )
                console.print("\n")

        except ImportError:
            # Fallback
            for idx, ex in enumerate(successful_examples):
                if args.limit and idx >= args.limit:
                    break

                print("=" * 80)
                print(f"Example {idx + 1} (Episode {ex['episode']})")
                print(f"Tier: {ex['tier']}")
                print(f"Honeypots Accessed: {ex['honeypots']}")
                print(f"Reward: {ex['reward']:.4f}")
                print("-" * 40)
                print("RED TEAM INPUT:")
                print(ex["red_team_input"])
                print("-" * 40)
                print("VICTIM RESPONSE (SQL):")
                print(ex["victim_response"])
                print("-" * 40)
                print("EXECUTION RESULT:")
                print(ex["execution_result"])
                print("=" * 80)
                print("\n")


if __name__ == "__main__":
    main()
