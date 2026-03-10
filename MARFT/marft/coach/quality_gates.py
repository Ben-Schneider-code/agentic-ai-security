"""
Quality gates for validating 32B coach-generated trajectory variations.

Two gates ensure augmented trajectories are useful:
1. Semantic Equivalence: cosine similarity of sentence embeddings >= threshold
2. Syntactic Diversity: word-level Jaccard similarity <= threshold

This prevents both semantically drifted variations AND near-duplicate copies.
"""

import numpy as np


class VariationQualityGate:
    """
    Validates 32B-generated variations for semantic equivalence + syntactic diversity.

    Uses a lightweight sentence-transformers model for dense embeddings and
    word-level Jaccard similarity for lexical diversity.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_cosine_sim: float = 0.7,
        max_jaccard_sim: float = 0.8,
        lazy_load: bool = True,
    ):
        """
        Args:
            embedding_model: HuggingFace model ID for sentence embeddings.
            min_cosine_sim: Minimum cosine similarity to original (semantic equivalence).
            max_jaccard_sim: Maximum word-level Jaccard similarity (syntactic diversity).
            lazy_load: If True, defer model loading until first use.
        """
        self.embedding_model_name = embedding_model
        self.min_cosine_sim = min_cosine_sim
        self.max_jaccard_sim = max_jaccard_sim
        self._embedder = None
        if not lazy_load:
            self._load_embedder()

    def _load_embedder(self):
        """Lazy-load the sentence transformer model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embedding_model_name)
                print(
                    f"[QualityGate] Loaded embedding model: {self.embedding_model_name}"
                )
            except ImportError:
                print(
                    "[QualityGate] WARNING: sentence-transformers not installed. "
                    "Falling back to Jaccard-only filtering."
                )
                self._embedder = None
            except Exception as e:
                print(f"[QualityGate] WARNING: Failed to load embedding model: {e}")
                self._embedder = None

    @staticmethod
    def word_jaccard(text_a: str, text_b: str) -> float:
        """Compute word-level Jaccard similarity between two texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        intersection = words_a & words_b
        union = words_a | words_b
        if not union:
            return 1.0  # Both empty = identical
        return len(intersection) / len(union)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def filter_variations(
        self, original: str, variations: list[str]
    ) -> tuple[list[str], dict]:
        """
        Return only variations that pass both quality gates.

        Args:
            original: The original successful action text.
            variations: List of 32B-generated variation texts.

        Returns:
            (passed_variations, stats_dict) where stats_dict contains
            counts of rejected variations and reasons.
        """
        if not variations:
            return [], {
                "total": 0,
                "passed": 0,
                "rejected_semantic": 0,
                "rejected_diversity": 0,
                "details": [],
            }

        stats = {
            "total": len(variations),
            "passed": 0,
            "rejected_semantic": 0,
            "rejected_diversity": 0,
            "details": [],  # per-variation: {text, cosine_sim, jaccard_sim, verdict, reason}
        }

        # Compute embeddings if model is available
        self._load_embedder()
        orig_emb = None
        var_embs = None
        if self._embedder is not None:
            try:
                all_texts = [original] + variations
                embeddings = self._embedder.encode(all_texts)
                orig_emb = embeddings[0]
                var_embs = embeddings[1:]
            except Exception as e:
                print(f"[QualityGate] WARNING: Embedding failed: {e}")
                orig_emb = None
                var_embs = None

        passed = []
        for idx, var_text in enumerate(variations):
            cosine_sim = None
            jaccard_sim = None

            # Gate 1: Semantic Equivalence (cosine similarity)
            if orig_emb is not None and var_embs is not None:
                cosine_sim = round(
                    float(self.cosine_similarity(orig_emb, var_embs[idx])), 4
                )
                if cosine_sim < self.min_cosine_sim:
                    stats["rejected_semantic"] += 1
                    stats["details"].append(
                        {
                            "text": var_text,
                            "cosine_sim": cosine_sim,
                            "jaccard_sim": None,
                            "verdict": "rejected",
                            "reason": "semantic",
                        }
                    )
                    continue
            # If no embedder, skip this gate

            # Gate 2: Syntactic Diversity (word-level Jaccard)
            jaccard_sim = round(float(self.word_jaccard(original, var_text)), 4)
            if jaccard_sim > self.max_jaccard_sim:
                stats["rejected_diversity"] += 1
                stats["details"].append(
                    {
                        "text": var_text,
                        "cosine_sim": cosine_sim,
                        "jaccard_sim": jaccard_sim,
                        "verdict": "rejected",
                        "reason": "diversity",
                    }
                )
                continue

            stats["details"].append(
                {
                    "text": var_text,
                    "cosine_sim": cosine_sim,
                    "jaccard_sim": jaccard_sim,
                    "verdict": "accepted",
                    "reason": None,
                }
            )
            passed.append(var_text)

        stats["passed"] = len(passed)
        return passed, stats
