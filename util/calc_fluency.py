import argparse
import sys
import os

# Add the project root and MARFT directory to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
marft_root = os.path.join(project_root, "MARFT")

sys.path.append(project_root)
sys.path.append(marft_root)

try:
    from marft.envs.redteam_sql.fluency_judge import FluencyJudge
except ImportError as e:
    print(f"Error importing FluencyJudge: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate fluency metrics for a text file."
    )
    parser.add_argument("file_path", help="Path to the text file to evaluate")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run the model on (default: cuda:0)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Model ID"
    )

    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        sys.exit(1)

    try:
        with open(args.file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"Evaluating text from {args.file_path} (Length: {len(text)} chars)")
    print("-" * 50)
    print(f"Text Preview: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
    print("-" * 50)

    print(f"Loading FluencyJudge (Model: {args.model}, Device: {args.device})...")
    try:
        judge = FluencyJudge.get_instance(model_id=args.model, device=args.device)

        info = judge.get_fluency_info(text)

        print("\n=== Fluency Results ===")
        print(f"Perplexity: {info['perplexity']:.2f}")
        print(f"Penalty:    {info['penalty']:.2f}")
        print(f"Threshold:  {info['threshold']}")
        print(f"Fluent:     {'Yes' if info['is_fluent'] else 'No'}")

        if "heuristic_gibberish" in info and info["heuristic_gibberish"]:
            print(f"Heuristic Gibberish: Yes ({info['heuristic_reason']})")

    except Exception as e:
        print(f"Error calculating fluency: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
