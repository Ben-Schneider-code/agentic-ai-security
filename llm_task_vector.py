"""
Task vector computation for LLMs, analogous to the diffusion model implementation
in db_vs_crt/moderator-lib/task_vector.py and utils_task_vector.py.

Usage:
    # Compute task vector from pretrained and finetuned checkpoints
    tv = LLMTaskVector(pretrained_model_name_or_path, finetuned_model_name_or_path)

    # Apply to base model and save
    tv.apply_to(pretrained_model_name_or_path, output_dir, scaling_coef=1.0)

    # Or use CLI:
    python llm_task_vector.py \
        --pretrained meta-llama/Llama-3.1-8B \
        --finetuned ./my-finetuned-model \
        --output ./edited-model \
        --scaling_coef 0.6
"""

# claude --resume 9b7503cf-b973-44b3-ae11-27f5087e7a37

import argparse
import copy
import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Core TaskVector class (mirrors task_vector.py for diffusion models)
# ---------------------------------------------------------------------------

class LLMTaskVector:
    """
    Represents a task vector for a language model.

    A task vector is the element-wise difference between finetuned and pretrained
    model weights:  vector = finetuned_weights - pretrained_weights

    Supports scalar multiplication, addition, and negation so that multiple task
    vectors can be composed before being applied to a base model.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str | None = None,
        finetuned_model_name_or_path: str | None = None,
        vector: dict | None = None,
        vector_path: str | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.vector: dict[str, torch.Tensor] = {}

        if vector is not None:
            self.vector = vector
        elif vector_path is not None:
            self.vector = self._load(vector_path)
        else:
            assert pretrained_model_name_or_path is not None and finetuned_model_name_or_path is not None, (
                "Provide either (pretrained, finetuned) paths, a vector dict, or a vector_path."
            )
            self.vector = self._compute(pretrained_model_name_or_path, finetuned_model_name_or_path)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _compute(self, pretrained_path: str, finetuned_path: str) -> dict[str, torch.Tensor]:
        """Load both models and return their weight delta."""
        print(f"Loading pretrained model from: {pretrained_path}")
        pretrained_sd = self._load_state_dict(pretrained_path)

        print(f"Loading finetuned model from: {finetuned_path}")
        finetuned_sd = self._load_state_dict(finetuned_path)

        vector: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key in pretrained_sd:
                if key not in finetuned_sd:
                    print(f"Warning: key '{key}' missing from finetuned model — skipping.")
                    continue
                p = pretrained_sd[key]
                f = finetuned_sd[key]
                # Skip integer / boolean buffers (e.g. position_ids)
                if p.dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
                    continue
                vector[key] = f - p

        print(f"Task vector computed: {len(vector)} parameter tensors.")
        return vector

    @staticmethod
    def _load_state_dict(model_name_or_path: str) -> dict[str, torch.Tensor]:
        """
        Load model weights, preferring the safetensors shard if present,
        otherwise using HuggingFace's standard loader.

        If the path contains a PEFT/LoRA adapter (adapter_config.json), the
        adapter is merged into the base model weights before returning so that
        the task vector reflects the true weight delta.
        """
        import json

        # Fast path: single safetensors file (fully merged model)
        st_path = os.path.join(model_name_or_path, "model.safetensors")
        if os.path.isfile(st_path):
            return load_file(st_path, device="cpu")

        # Detect PEFT/LoRA adapter — must merge before computing task vector
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            from peft import PeftModel
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg["base_model_name_or_path"]
            print(f"  Detected LoRA adapter; loading base model '{base_model_name}' and merging...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            model = model.merge_and_unload()
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            del model
            return sd

        # Standard HF load (handles sharded checkpoints automatically)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        del model
        return sd

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save task vector to a .npy file (dict of tensors)."""
        np.save(path, self.vector)
        print(f"Task vector saved to: {path}")

    @staticmethod
    def _load(path: str) -> dict[str, torch.Tensor]:
        d = np.load(path, allow_pickle=True).item()
        # Ensure values are torch tensors
        return {k: torch.as_tensor(v) for k, v in d.items()}

    @classmethod
    def from_lora_adapter(cls, adapter_path: str, *, strict: bool = True) -> "LLMTaskVector":
        """
        Compute a task vector directly from a PEFT/LoRA adapter without loading the base model.

        The delta for each LoRA-targeted weight is:
            ΔW = (lora_alpha / r) · lora_B @ lora_A

        Non-targeted layers are omitted from the vector (their delta is zero).
        This is orders of magnitude cheaper than merging the full fp32 base model.

        Args:
            adapter_path: directory containing adapter_config.json and
                          adapter_model.safetensors.
            strict:       if True (default), any unrecognized key or shape
                          inconsistency raises ValueError immediately.

        Returns:
            LLMTaskVector whose vector keys mirror the base model's weight
            parameter names (e.g. "model.layers.0.self_attn.q_proj.weight").
        """
        import json

        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"adapter_config.json not found in: {adapter_path}")
        with open(config_path) as f:
            cfg = json.load(f)

        r = cfg.get("r")
        lora_alpha = cfg.get("lora_alpha")
        if r is None:
            raise ValueError(f"'r' missing from adapter_config.json: {config_path}")
        if lora_alpha is None:
            raise ValueError(f"'lora_alpha' missing from adapter_config.json: {config_path}")

        rank_pattern: dict[str, int] = cfg.get("rank_pattern") or {}
        alpha_pattern: dict[str, float] = cfg.get("alpha_pattern") or {}

        st_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.isfile(st_path):
            raise FileNotFoundError(f"adapter_model.safetensors not found in: {adapter_path}")
        adapter_sd = load_file(st_path, device="cpu")

        # Group A/B matrices by their corresponding full weight key.
        # PEFT key format: "base_model.model.<param_path>.lora_{A,B}.weight"
        PREFIX = "base_model.model."
        SUFFIX_A = ".lora_A.weight"
        SUFFIX_B = ".lora_B.weight"

        groups: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in adapter_sd.items():
            if not key.startswith(PREFIX):
                if strict:
                    raise ValueError(
                        f"Adapter key lacks expected prefix '{PREFIX}': {key!r}"
                    )
                continue
            stripped = key[len(PREFIX):]  # e.g. model.layers.0.self_attn.q_proj.lora_A.weight
            if stripped.endswith(SUFFIX_A):
                base_key = stripped[: -len(SUFFIX_A)] + ".weight"
                groups.setdefault(base_key, {})["lora_A"] = tensor
            elif stripped.endswith(SUFFIX_B):
                base_key = stripped[: -len(SUFFIX_B)] + ".weight"
                groups.setdefault(base_key, {})["lora_B"] = tensor
            elif strict:
                raise ValueError(
                    f"Unrecognized adapter key pattern (expected lora_A/lora_B suffix): {key!r}"
                )

        if strict and not groups:
            raise ValueError(f"No lora_A/lora_B pairs found in adapter: {adapter_path}")

        vector: dict[str, torch.Tensor] = {}
        for base_key, mats in groups.items():
            if set(mats.keys()) != {"lora_A", "lora_B"}:
                raise ValueError(
                    f"Incomplete lora_A/lora_B pair for '{base_key}': "
                    f"found {sorted(mats.keys())}"
                )
            A = mats["lora_A"].float()  # shape: (r, in_features)
            B = mats["lora_B"].float()  # shape: (out_features, r)

            if A.shape[0] != B.shape[1]:
                raise ValueError(
                    f"Rank mismatch for '{base_key}': "
                    f"lora_A rank dim={A.shape[0]}, lora_B rank dim={B.shape[1]}"
                )

            # Resolve per-module overrides (PEFT rank_pattern/alpha_pattern keys
            # are bare module names like "q_proj", not full dotted paths).
            module_name = base_key.rsplit(".", 2)[-2]  # e.g. "q_proj"
            effective_r = rank_pattern.get(module_name, r)
            effective_alpha = alpha_pattern.get(module_name, lora_alpha)
            scaling = effective_alpha / effective_r

            with torch.no_grad():
                vector[base_key] = scaling * (B @ A)  # (out_features, in_features)

        return cls(vector=vector)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __mul__(self, scale: float) -> "LLMTaskVector":
        with torch.no_grad():
            new_vector = {k: scale * v for k, v in self.vector.items()}
        return LLMTaskVector(vector=new_vector)

    def __rmul__(self, scale: float) -> "LLMTaskVector":
        return self.__mul__(scale)

    def __add__(self, other: "LLMTaskVector") -> "LLMTaskVector":
        with torch.no_grad():
            new_vector: dict[str, torch.Tensor] = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning: key '{key}' not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return LLMTaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self) -> "LLMTaskVector":
        with torch.no_grad():
            new_vector = {k: -v for k, v in self.vector.items()}
        return LLMTaskVector(vector=new_vector)

    def __sub__(self, other: "LLMTaskVector") -> "LLMTaskVector":
        return self.__add__(-other)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_to(
        self,
        pretrained_model_name_or_path: str,
        output_dir: str,
        scaling_coef: float = 1.0,
    ) -> None:
        """
        Apply the task vector to a pretrained model and save the result.

        new_weights = pretrained_weights + scaling_coef * task_vector
        """
        print(f"Applying task vector (α={scaling_coef}) to: {pretrained_model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        with torch.no_grad():
            state_dict = model.state_dict()
            for key in state_dict:
                if key not in self.vector:
                    print(f"Warning: key '{key}' not in task vector — leaving unchanged.")
                    continue
                state_dict[key] = state_dict[key] + scaling_coef * self.vector[key].to(state_dict[key].device)
            model.load_state_dict(state_dict)

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)

        # Copy tokenizer from pretrained so the output dir is self-contained
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"Note: could not copy tokenizer ({e})")

        print(f"Edited model saved to: {output_dir}")

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def cosine_similarity(self, other: "LLMTaskVector", per_layer: bool = True) -> float:
        """
        Compute the mean cosine similarity between two task vectors.
        Values near 1 mean vectors point in the same direction; 0 means orthogonal.

        Prints the number of layers skipped because a key was absent from one or
        both task vectors.
        """
        sims: list[float] = []
        skipped_missing_in_other = 0
        skipped_missing_in_self = 0
        all_keys = set(self.vector) | set(other.vector)
        for key in all_keys:
            in_self = key in self.vector
            in_other = key in other.vector
            if not in_self:
                skipped_missing_in_self += 1
                continue
            if not in_other:
                skipped_missing_in_other += 1
                continue
            v1 = self.vector[key].float().numpy().ravel()
            v2 = other.vector[key].float().numpy().ravel()
            norm1 = np.linalg.norm(v1) + 1e-12
            norm2 = np.linalg.norm(v2) + 1e-12
            if per_layer:
                sims.append(float(np.dot(v1 / norm1, v2 / norm2)))
        total_skipped = skipped_missing_in_self + skipped_missing_in_other
        if total_skipped:
            print(
                f"cosine_similarity: skipped {total_skipped} layer(s) "
                f"({skipped_missing_in_other} missing from other, "
                f"{skipped_missing_in_self} missing from self)"
            )
        return float(np.mean(sims)) if sims else 0.0

    def per_layer_cosine_similarity(self, other: "LLMTaskVector") -> dict[str, float]:
        """Return cosine similarity for every shared layer key.

        Prints the number of layers skipped because a key was absent from one or
        both task vectors.
        """
        result: dict[str, float] = {}
        skipped_missing_in_other = 0
        skipped_missing_in_self = 0
        all_keys = set(self.vector) | set(other.vector)
        for key in all_keys:
            in_self = key in self.vector
            in_other = key in other.vector
            if not in_self:
                skipped_missing_in_self += 1
                continue
            if not in_other:
                skipped_missing_in_other += 1
                continue
            v1 = self.vector[key].float().numpy().ravel()
            v2 = other.vector[key].float().numpy().ravel()
            norm1 = np.linalg.norm(v1) + 1e-12
            norm2 = np.linalg.norm(v2) + 1e-12
            result[key] = float(np.dot(v1 / norm1, v2 / norm2))
        total_skipped = skipped_missing_in_self + skipped_missing_in_other
        if total_skipped:
            print(
                f"per_layer_cosine_similarity: skipped {total_skipped} layer(s) "
                f"({skipped_missing_in_other} missing from other, "
                f"{skipped_missing_in_self} missing from self)"
            )
        return result

    def sign_agreement(self, other: "LLMTaskVector", print_counts: bool = False) -> dict[str, dict[int, int]]:
        """
        For each shared layer, compute sign(self) - sign(other) element-wise and
        count occurrences of each value in {-2, -1, 0, 1, 2}.

          0  : both vectors agree in direction (or both zero)
         ±1  : one vector is zero, the other is non-zero
         ±2  : vectors disagree in direction (opposite signs)

        Returns a dict mapping layer key → {value: count}.
        If print_counts is True, prints a table of counts per layer.
        """
        result: dict[str, dict[int, int]] = {}
        with torch.no_grad():
            for key in self.vector:
                if key not in other.vector:
                    continue
                s1 = torch.sign(self.vector[key].float())
                s2 = torch.sign(other.vector[key].float())
                diff = (s1 - s2).long()
                counts: dict[int, int] = {}
                for val in (-2, -1, 0, 1, 2):
                    counts[val] = int((diff == val).sum().item())
                result[key] = counts

        if print_counts:
            header = (
                f"{'layer':<60}  {'n':<9}"
                f"  {'-2':>8}  {'%-2':>7}"
                f"  {'-1':>8}  {'%-1':>7}"
                f"  {'0':>8}  {'%0':>7}"
                f"  {'+1':>8}  {'%+1':>7}"
                f"  {'+2':>8}  {'%+2':>7}"
                f"  {'agree%':>8}"
            )
            print(header)
            print("-" * len(header))
            for key, counts in result.items():
                total = sum(counts.values())
                denom = total if total > 0 else 1
                agree_pct = 100.0 * counts[0] / denom
                print(
                    f"{key:<60}  {total:<9d}"
                    f"  {counts[-2]:>8d}  {100*counts[-2]/denom:>6.2f}%"
                    f"  {counts[-1]:>8d}  {100*counts[-1]/denom:>6.2f}%"
                    f"  {counts[0]:>8d}  {100*counts[0]/denom:>6.2f}%"
                    f"  {counts[1]:>8d}  {100*counts[1]/denom:>6.2f}%"
                    f"  {counts[2]:>8d}  {100*counts[2]/denom:>6.2f}%"
                    f"  {agree_pct:>7.2f}%"
                )

        return result

    def l2_norm(self) -> float:
        """Global L2 norm across all task-vector parameters."""
        total = sum(v.float().norm().item() ** 2 for v in self.vector.values())
        return float(total ** 0.5)

    def projection_and_norms(self, other: "LLMTaskVector") -> tuple[float, float, float]:
        """
        Compute the norm of the projection of `other` onto `self`, and the
        global L2 norms of both task vectors.

        Returns:
            tuple: (proj_norm, norm_self, norm_other)
                - proj_norm  : norm of the projection of `other` onto `self`
                - norm_self  : global L2 norm of this task vector
                - norm_other : global L2 norm of `other`
        """
        dot_product = 0.0
        norm_self_sq = 0.0
        norm_other_sq = 0.0

        for k in self.vector:
            if k not in other.vector:
                continue
            v1 = self.vector[k].float().numpy()
            v2 = other.vector[k].float().numpy()

            dot_product += np.sum(v1 * v2)
            norm_self_sq += np.sum(v1 ** 2)
            norm_other_sq += np.sum(v2 ** 2)

        norm_self = float(np.sqrt(norm_self_sq))
        norm_other = float(np.sqrt(norm_other_sq))

        if norm_self > 1e-12:
            proj_norm = float(abs(dot_product) / norm_self)
        else:
            proj_norm = 0.0

        return proj_norm, norm_self, norm_other

    def stats(self) -> dict:
        """Return summary statistics useful for debugging."""
        norms = {k: float(v.float().norm().item()) for k, v in self.vector.items()}
        return {
            "num_params": len(self.vector),
            "global_l2": self.l2_norm(),
            "max_layer_norm": max(norms.values()),
            "min_layer_norm": min(norms.values()),
            "mean_layer_norm": float(np.mean(list(norms.values()))),
        }


# ---------------------------------------------------------------------------
# Utility functions (mirrors utils_task_vector.py)
# ---------------------------------------------------------------------------

def get_task_vector(
    finetuned_path: str,
    pretrained_path: str,
    operator: str = "+",
    device: str = "cpu",
) -> LLMTaskVector:
    """
    Compute the task vector and optionally negate it.

    operator="+"  →  finetuned - pretrained   (moves model toward the concept)
    operator="-"  →  pretrained - finetuned   (moves model away from the concept)
    """
    tv = LLMTaskVector(pretrained_path, finetuned_path, device=device)
    return -tv if operator == "-" else tv


def accumulate_task_vectors(task_vector_configs: list[dict]) -> LLMTaskVector:
    """
    Sum a list of saved task vectors.

    Each entry in task_vector_configs should be a dict with keys:
        path     : str   — path to the saved .npy task vector
        scale    : float — scaling coefficient (default 1.0)
    """
    final: LLMTaskVector | None = None
    for cfg in task_vector_configs:
        tv = LLMTaskVector(vector_path=cfg["path"])
        tv = tv * cfg.get("scale", 1.0)
        final = tv if final is None else final + tv
    assert final is not None, "task_vector_configs must not be empty"
    return final


def sign_conflict_merge(tv_a: LLMTaskVector, tv_b: LLMTaskVector) -> LLMTaskVector:
    """
    Merge two task vectors using sign-conflict resolution (mirrors our_merge in
    utils_task_vector.py).

    Where tv_a and tv_b agree in sign, keep tv_a's contribution.
    Where they conflict (sign difference == ±2), trust tv_b instead.
    """
    base = tv_a if len(tv_a.vector) >= len(tv_b.vector) else tv_b
    result = copy.deepcopy(base)

    with torch.no_grad():
        for k in base.vector:
            if k not in tv_a.vector or k not in tv_b.vector:
                continue
            v1 = tv_a.vector[k]
            v2 = tv_b.vector[k]

            sign_diff = torch.sign(v1) - torch.sign(v2)  # values in {-2, -1, 0, 1, 2}
            # mask = (torch.abs(sign_diff) != 2).int()      # 1 where no conflict
            mask = (torch.abs(sign_diff) == 0).int() 

            result.vector[k] = v1 * mask + v2

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two finetuned model task vectors against a shared pretrained base."
    )
    # p.add_argument("--pretrained", default="meta-llama/Llama-2-7b-chat-hf", help="Pretrained model path or HF hub ID")
    # p.add_argument("--finetuned1", default="/home/a38das/finetune_growl/models_no_beaver/old_backdoor_advbench", help="First finetuned model path or HF hub ID")
    # p.add_argument("--finetuned2", default="./mmlu-finetuned-llama2-7b", help="Second finetuned model path or HF hub ID")
    p.add_argument("--pretrained", default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Pretrained model path or HF hub ID")
    p.add_argument("--finetuned1", default="/home/a38das/finetune_growl/qwen_extra_aligned", help="First finetuned model path or HF hub ID")
    p.add_argument("--finetuned2", default="/home/a38das/emergent-misalignment/open_models/em_test", help="Second finetuned model path or HF hub ID")
    p.add_argument("--diff_origin", default=False, action='store_true')
    p.add_argument("--save_layers", default=None, help="Path to save top/bottom 10 layers by cosine similarity as JSON")
    return p.parse_args()


def main() -> None:
    import json
    args = parse_args()

    tv_a = get_task_vector(args.finetuned1, args.pretrained)
    if args.diff_origin:
        tv_b = get_task_vector(args.finetuned2, args.finetuned1)
    else:
        tv_b = get_task_vector(args.finetuned2, args.pretrained)

    print("\n=== Stats: finetuned1 task vector ===")
    print(json.dumps(tv_a.stats(), indent=2))

    print("\n=== Stats: finetuned2 task vector ===")
    print(json.dumps(tv_b.stats(), indent=2))

    proj_b_onto_a, norm_a, norm_b = tv_a.projection_and_norms(tv_b)
    proj_a_onto_b, _, _ = tv_b.projection_and_norms(tv_a)
    print(f"\n=== Projection and norms ===")
    print(f"  norm(tv_a):              {norm_a:.6f}")
    print(f"  norm(tv_b):              {norm_b:.6f}")
    print(f"  proj of tv_b onto tv_a:  {proj_b_onto_a:.6f}  ({100*proj_b_onto_a/norm_b:.2f}% of tv_b norm)")
    print(f"  proj of tv_a onto tv_b:  {proj_a_onto_b:.6f}  ({100*proj_a_onto_b/norm_a:.2f}% of tv_a norm)")

    global_cos = tv_a.cosine_similarity(tv_b)
    print(f"\n=== Global mean cosine similarity: {global_cos:.6f} ===\n")

    layer_cos = tv_a.per_layer_cosine_similarity(tv_b)
    sorted_layers = sorted(layer_cos.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  {'layer':<60}  {'cos_sim':>8}")
    print("  " + "-" * 70)
    for key, sim in sorted_layers:
        print(f"  {key:<60}  {sim:>8.4f}")

    if args.save_layers is not None:
        top10 = [{"layer": k, "cos_sim": v} for k, v in sorted_layers[:10]]
        bottom10 = [{"layer": k, "cos_sim": v} for k, v in sorted_layers[-10:]]
        with open(args.save_layers, "w") as f:
            json.dump({"top10": top10, "bottom10": bottom10}, f, indent=2)
        print(f"\nTop/bottom 10 layers saved to: {args.save_layers}")

    print(f"\n=== Per-layer sign agreement  (sign(tv1) - sign(tv2)) ===")
    tv_a.sign_agreement(tv_b, print_counts=True)


if __name__ == "__main__":
    main()
