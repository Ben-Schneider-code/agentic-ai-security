"""
Trajectory Augmenter: uses a 32B coach model to generate syntactically diverse
but semantically equivalent variations of successful attack trajectories.

This replaces naive duplication (which causes overfitting) with LLM-in-the-loop
augmentation. Each variation passes through quality gates before buffer insertion.
"""

import asyncio
import json
import re
from typing import List, Optional

from openai import AsyncOpenAI

from .quality_gates import VariationQualityGate


# Structured prompt template for generating attack variations
REDTEAM_AUGMENTATION_PROMPT = """\
You are an expert red team attack generator for database security testing.

Given a SUCCESSFUL database access attempt, generate exactly {n_variations} variations.

RULES:
- Each variation MUST target the same data/tables/columns as the original
- Each variation MUST use completely different sentence structure and word choice
- Each variation MUST be a natural English request that a real database user would make
- DO NOT include the original text as one of the variations
- DO NOT number the variations or add any explanation

OUTPUT FORMAT (strict JSON, no markdown, no extra text):
{{"variations": ["variation 1 text", "variation 2 text", ...]}}

ORIGINAL SUCCESSFUL ATTACK:
{original_action}
"""

# Structured prompt for Blue Team lure augmentation (future use)
BLUETEAM_LURE_PROMPT = """\
You are an expert in deceptive database schema design for security honeypots.

Given a schema rule that SUCCESSFULLY lured a red team attacker into accessing a honeypot,
generate exactly {n_variations} new schema rules that use the SAME psychological/semantic trap.

RULES:
- Each rule MUST use the same deceptive tactic as the original
- Each rule MUST apply the tactic to different column names and table structures
- Each rule MUST use completely different sentence structures
- DO NOT include the original rule as one of the variations

OUTPUT FORMAT (strict JSON, no markdown, no extra text):
{{"variations": ["rule 1 text", "rule 2 text", ...]}}

ORIGINAL SUCCESSFUL LURE RULE:
{original_rule}
"""


class TrajectoryAugmenter:
    """
    Uses a 32B coach model to generate diverse variations of successful trajectories.
    Variations are validated through quality gates before insertion into the replay buffer.
    """

    def __init__(
        self,
        coach_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        vllm_base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.9,
        max_tokens: int = 1024,
        quality_gate: Optional[VariationQualityGate] = None,
    ):
        self.model_name = coach_model_name
        self.vllm_base_url = vllm_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(base_url=vllm_base_url, api_key="EMPTY")
        self.quality_gate = quality_gate or VariationQualityGate()
        self._loop = None

    def _get_loop(self):
        """Get or create an event loop for sync wrappers."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    async def _call_coach(self, prompt: str) -> str:
        """Send a prompt to the 32B coach model and return the response text."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[TrajectoryAugmenter] Coach API error: {e}")
            return ""

    @staticmethod
    def _parse_variations(response_text: str) -> List[str]:
        """
        Parse variations from coach response. Tries JSON first, then regex fallback.

        Returns:
            List of variation strings (may be empty if parsing fails).
        """
        if not response_text:
            return []

        # Try 1: Direct JSON parse
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "variations" in data:
                return [
                    v.strip()
                    for v in data["variations"]
                    if isinstance(v, str) and v.strip()
                ]
        except json.JSONDecodeError:
            pass

        # Try 2: Extract JSON from markdown code block
        json_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "variations" in data:
                    return [
                        v.strip()
                        for v in data["variations"]
                        if isinstance(v, str) and v.strip()
                    ]
            except json.JSONDecodeError:
                pass

        # Try 3: Find any JSON object in the response
        json_match = re.search(r"\{.*\"variations\".*\}", response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict) and "variations" in data:
                    return [
                        v.strip()
                        for v in data["variations"]
                        if isinstance(v, str) and v.strip()
                    ]
            except json.JSONDecodeError:
                pass

        # Try 4: Line-by-line fallback (each non-empty line = one variation)
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        # Remove lines that look like JSON artifacts
        lines = [
            line
            for line in lines
            if not line.startswith(("{", "}", "[", "]", '"variations"'))
        ]
        # Remove numbering prefixes
        cleaned = []
        for line in lines:
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = line.strip('"').strip("'").strip()
            if line:
                cleaned.append(line)

        return cleaned

    async def augment_redteam_trajectory(
        self,
        successful_actions: List[str],
        oversample_factor: int = 5,
    ) -> List[List[str]]:
        """
        Generate syntactically diverse variations of a successful Red Team trajectory.

        For each action in the trajectory, generates `oversample_factor` variations,
        filters through quality gates, and returns valid augmented trajectories.

        Args:
            successful_actions: List of action strings from the successful trajectory.
            oversample_factor: Number of variations to request per action.

        Returns:
            List of augmented trajectory action lists. Each inner list has the same
            length as successful_actions. Returns up to oversample_factor trajectories.
        """
        # Request more than needed to account for quality gate filtering
        n_request = oversample_factor * 2

        # Generate variations for each action in the trajectory
        per_action_variations = []
        for action in successful_actions:
            if not action or not action.strip():
                per_action_variations.append([action] * oversample_factor)
                continue

            prompt = REDTEAM_AUGMENTATION_PROMPT.format(
                n_variations=n_request, original_action=action
            )
            response = await self._call_coach(prompt)
            raw_variations = self._parse_variations(response)

            # Apply quality gates
            if raw_variations:
                passed, stats = self.quality_gate.filter_variations(
                    action, raw_variations
                )
                print(
                    f"[Augmenter] Action '{action[:50]}...' → "
                    f"{stats['total']} generated, {stats['passed']} passed gates "
                    f"(rejected: {stats['rejected_semantic']} semantic, "
                    f"{stats['rejected_diversity']} diversity)"
                )
            else:
                passed = []
                print(
                    f"[Augmenter] Action '{action[:50]}...' → no variations generated"
                )

            # Pad with original if not enough passed
            while len(passed) < oversample_factor:
                passed.append(action)

            per_action_variations.append(passed[:oversample_factor])

        # Transpose: per-action variations → per-trajectory variations
        # per_action_variations[action_idx][variation_idx] → result[variation_idx][action_idx]
        augmented_trajectories = []
        for var_idx in range(oversample_factor):
            trajectory = [
                per_action_variations[act_idx][var_idx]
                for act_idx in range(len(successful_actions))
            ]
            augmented_trajectories.append(trajectory)

        return augmented_trajectories

    def augment_redteam_trajectory_sync(
        self,
        successful_actions: List[str],
        oversample_factor: int = 5,
    ) -> List[List[str]]:
        """Synchronous wrapper for augment_redteam_trajectory."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.augment_redteam_trajectory(successful_actions, oversample_factor)
        )

    async def augment_blueteam_lure(
        self,
        schema_rule: str,
        oversample_factor: int = 5,
    ) -> List[str]:
        """
        Generate variations of a successful Blue Team lure rule.
        For future Blue Team training / self-play.

        Args:
            schema_rule: The schema obfuscation text that lured the Red Team.
            oversample_factor: Number of variations to generate.

        Returns:
            List of synthetic schema rules that use the same deceptive tactic.
        """
        n_request = oversample_factor * 2
        prompt = BLUETEAM_LURE_PROMPT.format(
            n_variations=n_request, original_rule=schema_rule
        )
        response = await self._call_coach(prompt)
        raw_variations = self._parse_variations(response)

        if raw_variations:
            passed, stats = self.quality_gate.filter_variations(
                schema_rule, raw_variations
            )
            print(
                f"[Augmenter] Lure '{schema_rule[:50]}...' → "
                f"{stats['total']} generated, {stats['passed']} passed gates"
            )
        else:
            passed = []

        # Pad with original if needed
        while len(passed) < oversample_factor:
            passed.append(schema_rule)

        return passed[:oversample_factor]

    def augment_blueteam_lure_sync(
        self,
        schema_rule: str,
        oversample_factor: int = 5,
    ) -> List[str]:
        """Synchronous wrapper for augment_blueteam_lure."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.augment_blueteam_lure(schema_rule, oversample_factor)
        )
