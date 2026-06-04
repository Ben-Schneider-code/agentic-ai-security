"""MARFT agent prompt wire-format — the single source of truth for the exact
string an agent's policy is tokenized on.

Shared by the training path (``marft.mas.mas``) and the offline cross-evaluator
(``util.cross_evaluate``) so the red policy is prompted byte-identically in both.
Pure string ops, no heavy deps, so eval can reconstruct the training prompt
without importing the torch-laden MAS machinery.
"""

TURN_BEGIN = "<|im_start|>"
TURN_END = "<|im_end|>"


def build_agent_prompt(profile_prompt: str, obs: str, role: str) -> str:
    """The exact string MARFT tokenizes for one agent generation step.

    Concatenates the agent's profile (system) prompt, the accumulated
    observation, and the open ``<|im_start|>{role}: `` cue the policy completes.
    Mirrors ``mas.infer_for_rollout`` — keep it the only place this layout lives.
    """
    return f"{profile_prompt}{obs}{TURN_BEGIN}{role}: "
