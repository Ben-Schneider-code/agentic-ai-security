#!/usr/bin/env python
"""Calculate embedding distance between candidate and reference texts.

Uses Jina Embeddings v3 for semantic similarity.

Requires: pip install transformers torch

Usage:
    python calc_embedding_distance.py candidate.txt reference.txt
    python calc_embedding_distance.py --candidate "text1" --reference "text2"
"""

import argparse
import sys


def compute_embedding_distance(
    candidates: list[str],
    references: list[str],
    model_name: str = "jinaai/jina-embeddings-v3",
    device: str = "cuda:0",
) -> dict:
    """
    Calculate embedding distance between candidates and references.

    Args:
        candidates: List of candidate texts
        references: List of reference texts
        model_name: HuggingFace model name
        device: Device to run model on

    Returns:
        dict with cosine_similarity, euclidean_distance
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print(
            "Error: transformers/torch not installed. Run: pip install transformers torch"
        )
        sys.exit(1)

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    def get_embeddings(texts: list[str]) -> torch.Tensor:
        """Get mean-pooled embeddings for texts."""
        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over token embeddings
            attention_mask = inputs["attention_mask"]
            embeddings = outputs.last_hidden_state
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            )
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

    cand_emb = get_embeddings(candidates)
    ref_emb = get_embeddings(references)

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(cand_emb, ref_emb, dim=1)

    # Euclidean distance
    euclidean_dist = torch.norm(cand_emb - ref_emb, dim=1)

    return {
        "cosine_similarity": cosine_sim.mean().item(),
        "euclidean_distance": euclidean_dist.mean().item(),
        "cosine_sim_per_sample": cosine_sim.tolist(),
        "euclidean_dist_per_sample": euclidean_dist.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate embedding distance between candidate and reference texts."
    )
    parser.add_argument("candidate_file", nargs="?", help="Path to candidate text file")
    parser.add_argument("reference_file", nargs="?", help="Path to reference text file")
    parser.add_argument("--candidate", "-c", help="Candidate text string")
    parser.add_argument("--reference", "-r", help="Reference text string")
    parser.add_argument(
        "--model", default="jinaai/jina-embeddings-v3", help="Model name"
    )
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
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print("-" * 50)

    result = compute_embedding_distance(
        [candidate], [reference], args.model, args.device
    )

    print("\n=== Embedding Distance Results ===")
    print(f"Cosine Similarity:   {result['cosine_similarity']:.4f}")
    print(f"Euclidean Distance:  {result['euclidean_distance']:.4f}")


if __name__ == "__main__":
    main()
