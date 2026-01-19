#!/usr/bin/env python
"""Calculate BLEU score between candidate and reference texts.

Usage:
    python calc_bleu.py candidate.txt reference.txt
    python calc_bleu.py --candidate "generated text" --reference "reference text"
"""

import argparse
import sys
import math
from collections import Counter


def get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-grams from token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(candidate: str, reference: str, max_n: int = 4) -> dict:
    """
    Calculate BLEU score between candidate and reference.

    Args:
        candidate: Generated/hypothesis text
        reference: Reference/ground truth text
        max_n: Maximum n-gram order (default 4 for BLEU-4)

    Returns:
        dict with bleu_1, bleu_2, bleu_3, bleu_4, and combined bleu score
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    if len(cand_tokens) == 0:
        return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    # Calculate precision for each n-gram level
    precisions = []
    for n in range(1, max_n + 1):
        if len(cand_tokens) < n or len(ref_tokens) < n:
            precisions.append(0.0)
            continue

        cand_ngrams = get_ngrams(cand_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        # Clipped counts
        clipped = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())

        precision = clipped / total if total > 0 else 0.0
        precisions.append(precision)

    # Brevity penalty
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = math.exp(1 - r / c) if c < r else 1.0

    # Geometric mean of precisions (avoid log(0))
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = bp * math.exp(log_avg)

    return {
        "bleu": bleu,
        "bleu_1": precisions[0] if len(precisions) > 0 else 0.0,
        "bleu_2": precisions[1] if len(precisions) > 1 else 0.0,
        "bleu_3": precisions[2] if len(precisions) > 2 else 0.0,
        "bleu_4": precisions[3] if len(precisions) > 3 else 0.0,
        "brevity_penalty": bp,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate BLEU score between candidate and reference texts."
    )
    parser.add_argument("candidate_file", nargs="?", help="Path to candidate text file")
    parser.add_argument("reference_file", nargs="?", help="Path to reference text file")
    parser.add_argument(
        "--candidate", "-c", help="Candidate text string (alternative to file)"
    )
    parser.add_argument(
        "--reference", "-r", help="Reference text string (alternative to file)"
    )

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
    print("-" * 50)

    result = bleu_score(candidate, reference)

    print("\n=== BLEU Results ===")
    print(f"BLEU (combined): {result['bleu']:.4f}")
    print(f"BLEU-1: {result['bleu_1']:.4f}")
    print(f"BLEU-2: {result['bleu_2']:.4f}")
    print(f"BLEU-3: {result['bleu_3']:.4f}")
    print(f"BLEU-4: {result['bleu_4']:.4f}")
    print(f"Brevity Penalty: {result['brevity_penalty']:.4f}")


if __name__ == "__main__":
    main()
