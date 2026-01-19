#!/usr/bin/env python
"""Calculate BERTScore between candidate and reference texts.

Requires: pip install bert-score

Usage:
    python calc_bertscore.py candidate.txt reference.txt
    python calc_bertscore.py --candidate "generated text" --reference "reference text"
"""

import argparse
import sys


def compute_bertscore(
    candidates: list[str], references: list[str], device: str = "cuda:0"
) -> dict:
    """
    Calculate BERTScore between candidates and references.

    Args:
        candidates: List of generated/hypothesis texts
        references: List of reference/ground truth texts
        device: Device to run model on

    Returns:
        dict with precision, recall, f1 scores
    """
    try:
        from bert_score import score
    except ImportError:
        print("Error: bert-score not installed. Run: pip install bert-score")
        sys.exit(1)

    P, R, F1 = score(candidates, references, lang="en", device=device, verbose=False)

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
        "precision_per_sample": P.tolist(),
        "recall_per_sample": R.tolist(),
        "f1_per_sample": F1.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BERTScore between candidate and reference texts."
    )
    parser.add_argument("candidate_file", nargs="?", help="Path to candidate text file")
    parser.add_argument("reference_file", nargs="?", help="Path to reference text file")
    parser.add_argument("--candidate", "-c", help="Candidate text string")
    parser.add_argument("--reference", "-r", help="Reference text string")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")

    args = parser.parse_args()

    # Get candidate text
    if args.candidate:
        candidate = args.candidate
    elif args.candidate_file:
        with open(args.candidate_file, "r", encoding="utf-8") as f:
            candidate = f.read().strip()
    else:
        print("Error: Provide candidate via file or --candidate flag")
        sys.exit(1)

    # Get reference text
    if args.reference:
        reference = args.reference
    elif args.reference_file:
        with open(args.reference_file, "r", encoding="utf-8") as f:
            reference = f.read().strip()
    else:
        print("Error: Provide reference via file or --reference flag")
        sys.exit(1)

    print(f"Candidate ({len(candidate)} chars): {candidate[:100]}...")
    print(f"Reference ({len(reference)} chars): {reference[:100]}...")
    print(f"Device: {args.device}")
    print("-" * 50)

    result = compute_bertscore([candidate], [reference], device=args.device)

    print("\n=== BERTScore Results ===")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")


if __name__ == "__main__":
    main()
