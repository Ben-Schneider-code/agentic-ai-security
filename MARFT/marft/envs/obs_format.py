"""Observation display cap shared by the SQL envs and the offline evaluator.

One definition so the red/blue conversation history is truncated identically
wherever it is rendered — both training envs (``redteam_sql_env``,
``blueteam_sql_env``) and ``util.cross_evaluate``. The cap is part of the prompt
the policies condition on, so every call site MUST produce the same bytes.
"""

DISPLAY_CAP = 1200


def cap_display(text: str, cap: int = DISPLAY_CAP) -> str:
    """Cap one observation/history entry, marking truncation so a policy can
    tell a cut-off turn from a naturally short one."""
    return text if len(text) <= cap else text[:cap] + "... [truncated]"
