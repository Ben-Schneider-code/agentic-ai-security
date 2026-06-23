"""MARFT agent prompt wire-format — the single source of truth for the exact
string an agent's policy is tokenized on.

Shared by the training path (``marft.mas.mas``) and the offline cross-evaluator
(``util.cross_evaluate``) so the red policy is prompted byte-identically in both.
Pure string ops, no heavy deps, so eval can reconstruct the training prompt
without importing the torch-laden MAS machinery.
"""

TURN_BEGIN = "<|im_start|>"
TURN_END = "<|im_end|>"

# Plain-text turn delimiters the running observation uses between turns
# (``redteam_append_turn``: ``sql_agent: ... \nassistant: ...``). A strong instruct
# policy imitates this format and keeps generating the *victim's* reply (and further
# turns) after its own instead of emitting ``<|im_end|>``. These are the generation
# stop strings — and the markers ``truncate_red_turn`` cuts on — so the red action is
# exactly one turn in both training (``mas.infer_for_rollout``) and cross-eval
# (``util.cross_evaluate``). Single source of truth; keep both paths importing these.
RED_TURN_STOPS = ["\nassistant:", "\nsql_agent:"]


def build_agent_prompt(profile_prompt: str, obs: str, role: str) -> str:
    """The exact string MARFT tokenizes for one agent generation step.

    Concatenates the agent's profile (system) prompt, the accumulated
    observation, and the open ``<|im_start|>{role}: `` cue the policy completes.
    Mirrors ``mas.infer_for_rollout`` — keep it the only place this layout lives.
    """
    return f"{profile_prompt}{obs}{TURN_BEGIN}{role}: "


def truncate_red_turn(text: str) -> str:
    """Cut a red action down to its own single turn.

    Generation stop strings (``RED_TURN_STOPS``) prevent most of the leaked-dialogue
    overrun at the source, but HF ``generate`` includes the matched stop string in its
    output and a missed boundary still slips through. This is the defensive post-cut
    that guarantees only the red's own turn reaches the victim, the reward, and the
    stored observation. Idempotent; returns ``text`` unchanged when there is no leak.
    """
    cut = len(text)
    for marker in (*RED_TURN_STOPS, TURN_END):
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()
