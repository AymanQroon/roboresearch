from __future__ import annotations

import json
import re

import anthropic

from roboresearch.agents.client import MODELS, create_client

_MODEL = MODELS["quick_evaluator"]

_SYSTEM_PROMPT = """\
You are a fast evaluation agent for a robotics research system. You compare training run \
metrics against previous best results and decide whether to KEEP or DISCARD the new run.

## Decision Criteria (in priority order)
1. **success_rate** — the fraction of evaluation episodes where the robot completed the task. \
This is the primary metric. Any improvement here is significant.
2. **mean_reward** — average cumulative reward per episode. Use as a tiebreaker when \
success_rate is unchanged, or to detect positive trends even when success hasn't improved.

## Context
- Early runs (first 3-5 experiments) should be kept more liberally to build a baseline.
- If there have been many experiments without improvement, even marginal gains are worth keeping.
- A run that regresses on success_rate should almost always be discarded, even if mean_reward \
improves slightly.

## Output Format
Respond with ONLY a JSON object (no markdown fences, no extra text):
{
    "decision": "keep" or "discard",
    "reasoning": "2-3 sentence explanation of why",
    "confidence": <float between 0.0 and 1.0>
}\
"""


def _build_user_message(
    current_metrics: dict,
    best_metrics: dict | None,
    experiment_notes: str,
    num_experiments_without_improvement: int,
) -> str:
    parts = [
        "## Current Run Metrics",
        f"- success_rate: {current_metrics.get('success_rate', 'N/A')}",
        f"- mean_reward: {current_metrics.get('mean_reward', 'N/A')}",
        f"- std_reward: {current_metrics.get('std_reward', 'N/A')}",
        f"- mean_episode_length: {current_metrics.get('mean_episode_length', 'N/A')}",
    ]

    if best_metrics:
        parts.extend([
            "\n## Previous Best Metrics",
            f"- success_rate: {best_metrics.get('success_rate', 'N/A')}",
            f"- mean_reward: {best_metrics.get('mean_reward', 'N/A')}",
            f"- std_reward: {best_metrics.get('std_reward', 'N/A')}",
            f"- mean_episode_length: {best_metrics.get('mean_episode_length', 'N/A')}",
        ])
    else:
        parts.append("\n## Previous Best Metrics\nThis is the FIRST experiment — no previous best.")

    parts.append(f"\n## What Was Tried\n{experiment_notes}")
    parts.append(
        f"\n## Experiments Without Improvement\n{num_experiments_without_improvement} consecutive"
    )
    parts.append("\nShould this run be KEPT or DISCARDED?")

    return "\n".join(parts)


def _parse_json_response(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class QuickEvaluator:
    def __init__(self, **kwargs):
        self._client = create_client()

    def evaluate_decision(
        self,
        current_metrics: dict,
        best_metrics: dict | None,
        experiment_notes: str,
        num_experiments_without_improvement: int,
    ) -> dict:
        user_message = _build_user_message(
            current_metrics,
            best_metrics,
            experiment_notes,
            num_experiments_without_improvement,
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as exc:
            return self._fallback_decision(current_metrics, best_metrics, str(exc))

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        parsed = _parse_json_response(response.content[0].text)

        if parsed is None:
            result = self._fallback_decision(current_metrics, best_metrics, "Parse error")
            result["usage"] = usage
            return result

        decision = parsed.get("decision", "discard").lower()
        if decision not in ("keep", "discard"):
            decision = "discard"

        confidence = parsed.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.0, min(1.0, float(confidence)))

        return {
            "decision": decision,
            "reasoning": parsed.get("reasoning", ""),
            "confidence": confidence,
            "usage": usage,
        }

    def _fallback_decision(
        self,
        current_metrics: dict,
        best_metrics: dict | None,
        error: str,
    ) -> dict:
        if best_metrics is None:
            return {
                "decision": "keep",
                "reasoning": f"First experiment — keeping as baseline. (Fallback due to: {error})",
                "confidence": 1.0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

        current_sr = current_metrics.get("success_rate", 0.0)
        best_sr = best_metrics.get("success_rate", 0.0)

        if current_sr > best_sr:
            return {
                "decision": "keep",
                "reasoning": (
                    f"Success rate improved from {best_sr} to {current_sr}. "
                    f"(Fallback due to: {error})"
                ),
                "confidence": 0.8,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

        return {
            "decision": "discard",
            "reasoning": (
                f"Success rate did not improve ({current_sr} vs best {best_sr}). "
                f"(Fallback due to: {error})"
            ),
            "confidence": 0.7,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
