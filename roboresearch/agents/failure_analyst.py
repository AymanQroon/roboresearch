from __future__ import annotations

import json
import re
from collections import Counter


from roboresearch.agents.client import MODELS, create_client

MODEL = MODELS["failure_analyst"]
MAX_FRAMES_PER_EPISODE = 8

TASK_INFO = {
    "FetchReach-v4": {
        "description": (
            "The robot arm must move its end-effector to reach a target position in 3D space"
        ),
        "success_criteria": "End-effector is within 5cm of the target position",
    },
    "FetchPush-v4": {
        "description": "The robot arm must push a block on a table to a target position",
        "success_criteria": "Block is within 5cm of the target position",
    },
    "FetchSlide-v4": {
        "description": (
            "The robot arm must slide a puck across a table to a target position beyond its reach"
        ),
        "success_criteria": "Puck is within 5cm of the target position",
    },
    "FetchPickAndPlace-v4": {
        "description": (
            "The robot arm must pick up a block and place it at a target position in the air"
        ),
        "success_criteria": "Block is within 5cm of the target position",
    },
}

FAILURE_CATEGORIES = [
    "no_movement",
    "wrong_direction",
    "undershoot",
    "overshoot",
    "approach_angle",
    "no_contact",
    "premature_release",
    "grip_failure",
    "excessive_force",
    "oscillation",
    "timeout",
    "other",
]

_EPISODE_SYSTEM_PROMPT = """\
You are a robotics failure analyst. You examine sequences of simulation frames from a \
MuJoCo robot manipulation environment and diagnose why the robot failed to complete its task.

## Environment
{env_name}

## Task
{task_description}

## Success Criteria
{success_criteria}

## Failure Category Taxonomy
{categories}

## Instructions
- You will receive a sequence of frames captured at regular intervals during a single episode. \
Analyze the PROGRESSION across frames, not just individual snapshots.
- Describe the spatial relationships you observe: where is the robot arm, where is the target \
(shown as a small sphere or marker), and where is any object involved.
- Identify what went wrong by watching how the arm moves (or fails to move) \
across the frame sequence.
- Classify the failure into exactly one category from the taxonomy above.
- Explain the likely root cause in terms of the underlying policy behavior.
- Suggest concrete fixes that could be applied to the training process or reward function.
- Rate your confidence from 0.0 to 1.0 based on how clearly the frames reveal the failure mode.

## Output Format
Respond with ONLY a JSON object (no markdown fences, no extra text):
{{
    "diagnosis": "<2-4 sentence description of what happened in the episode>",
    "failure_category": "<one category from the taxonomy>",
    "root_cause": "<1-2 sentence explanation of the likely policy-level cause>",
    "suggested_fixes": ["<fix 1>", "<fix 2>"],
    "confidence": <float between 0.0 and 1.0>
}}\
"""

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a robotics failure analyst synthesizing patterns across multiple failed episodes \
from a robot manipulation environment.

## Environment
{env_name}

## Task
{task_description}

## Success Criteria
{success_criteria}

## Instructions
You will receive individual failure analyses for multiple episodes. Identify the dominant \
pattern across episodes and produce actionable recommendations.

## Output Format
Respond with ONLY a JSON object (no markdown fences, no extra text):
{{
    "pattern_summary": "<2-4 sentence summary of the common failure pattern>",
    "dominant_failure_category": "<most common category>",
    "overall_suggested_fixes": ["<fix 1>", "<fix 2>", "<fix 3>"]
}}\
"""


def _sample_frames(frames: list[str], max_frames: int = MAX_FRAMES_PER_EPISODE) -> list[str]:
    if len(frames) <= max_frames:
        return list(frames)

    # Always keep first and last, sample evenly between them
    middle_count = max_frames - 2
    total_middle = len(frames) - 2
    indices = [0]
    for i in range(middle_count):
        idx = 1 + round(i * (total_middle - 1) / (middle_count - 1))
        indices.append(idx)
    indices.append(len(frames) - 1)
    return [frames[i] for i in indices]


def _build_image_content(base64_data: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64_data,
        },
    }


def _parse_json_response(text: str) -> dict | None:
    # Try parsing the entire response as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Look for any JSON-like object in the response
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _extract_token_usage(response) -> dict:
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _add_token_usage(a: dict, b: dict) -> dict:
    return {
        "input_tokens": a.get("input_tokens", 0) + b.get("input_tokens", 0),
        "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
    }


class FailureAnalyst:
    def __init__(self, **kwargs):
        self._client = create_client()

    def analyze_episode(
        self,
        frames: list[str],
        task_description: str,
        success_criteria: str,
        episode_metrics: dict,
        env_name: str = "FetchReach-v4",
    ) -> dict:
        sampled = _sample_frames(frames)

        system_prompt = _EPISODE_SYSTEM_PROMPT.format(
            env_name=env_name,
            task_description=task_description,
            success_criteria=success_criteria,
            categories="\n".join(f"- {c}" for c in FAILURE_CATEGORIES),
        )

        content: list[dict] = []

        metrics_text = (
            f"Episode metrics:\n"
            f"- Reward: {episode_metrics.get('reward', 'N/A')}\n"
            f"- Episode length: {episode_metrics.get('episode_length', 'N/A')} steps\n"
            f"- Final distance to goal: {episode_metrics.get('final_distance', 'N/A')}\n"
            f"- Number of frames: {len(frames)} (showing {len(sampled)} sampled)\n\n"
            f"The frames below are in chronological order from start to end of episode:"
        )
        content.append({"type": "text", "text": metrics_text})

        for i, frame_data in enumerate(sampled):
            content.append({"type": "text", "text": f"Frame {i + 1}/{len(sampled)}:"})
            content.append(_build_image_content(frame_data))

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )

        raw_text = response.content[0].text
        token_usage = _extract_token_usage(response)

        parsed = _parse_json_response(raw_text)

        if parsed is None:
            parsed = {
                "diagnosis": raw_text[:500],
                "failure_category": "other",
                "root_cause": "Could not parse structured response from model.",
                "suggested_fixes": [],
                "confidence": 0.0,
            }

        if parsed.get("failure_category") not in FAILURE_CATEGORIES:
            parsed["failure_category"] = "other"

        confidence = parsed.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        parsed["confidence"] = max(0.0, min(1.0, float(confidence)))

        if not isinstance(parsed.get("suggested_fixes"), list):
            parsed["suggested_fixes"] = []

        parsed["token_usage"] = token_usage
        return parsed

    def analyze_batch(
        self,
        failed_episodes: list[dict],
        task_description: str,
        success_criteria: str,
        env_name: str = "FetchReach-v4",
        max_episodes: int = 3,
    ) -> dict:
        episodes_to_analyze = failed_episodes[:max_episodes]
        episode_analyses = []
        total_usage = {"input_tokens": 0, "output_tokens": 0}

        for episode in episodes_to_analyze:
            metrics = {
                "reward": episode.get("reward"),
                "episode_length": episode.get("episode_length"),
                "final_distance": episode.get("final_distance"),
            }
            analysis = self.analyze_episode(
                frames=episode.get("frames", []),
                task_description=task_description,
                success_criteria=success_criteria,
                episode_metrics=metrics,
                env_name=env_name,
            )
            episode_analyses.append(analysis)
            total_usage = _add_token_usage(total_usage, analysis.get("token_usage", {}))

        if not episode_analyses:
            return {
                "episode_analyses": [],
                "pattern_summary": "No failed episodes to analyze.",
                "dominant_failure_category": "other",
                "overall_suggested_fixes": [],
                "total_token_usage": total_usage,
            }

        synthesis = self._synthesize_patterns(
            episode_analyses=episode_analyses,
            task_description=task_description,
            success_criteria=success_criteria,
            env_name=env_name,
        )
        total_usage = _add_token_usage(total_usage, synthesis.get("token_usage", {}))

        return {
            "episode_analyses": episode_analyses,
            "pattern_summary": synthesis.get("pattern_summary", ""),
            "dominant_failure_category": synthesis.get("dominant_failure_category", "other"),
            "overall_suggested_fixes": synthesis.get("overall_suggested_fixes", []),
            "total_token_usage": total_usage,
        }

    def _synthesize_patterns(
        self,
        episode_analyses: list[dict],
        task_description: str,
        success_criteria: str,
        env_name: str,
    ) -> dict:
        system_prompt = _SYNTHESIS_SYSTEM_PROMPT.format(
            env_name=env_name,
            task_description=task_description,
            success_criteria=success_criteria,
        )

        summaries = []
        for i, analysis in enumerate(episode_analyses):
            summaries.append(
                f"Episode {i + 1}:\n"
                f"  Category: {analysis.get('failure_category', 'unknown')}\n"
                f"  Diagnosis: {analysis.get('diagnosis', 'N/A')}\n"
                f"  Root cause: {analysis.get('root_cause', 'N/A')}\n"
                f"  Confidence: {analysis.get('confidence', 0.0)}"
            )

        user_text = (
            f"Here are the individual failure analyses for {len(episode_analyses)} episodes:\n\n"
            + "\n\n".join(summaries)
        )

        response = self._client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}],
        )

        raw_text = response.content[0].text
        token_usage = _extract_token_usage(response)

        parsed = _parse_json_response(raw_text)

        if parsed is None:
            # Fall back to computing dominant category from individual analyses
            category_counts = Counter(
                a.get("failure_category", "other") for a in episode_analyses
            )
            dominant = category_counts.most_common(1)[0][0]
            parsed = {
                "pattern_summary": raw_text[:500],
                "dominant_failure_category": dominant,
                "overall_suggested_fixes": [],
            }

        if parsed.get("dominant_failure_category") not in FAILURE_CATEGORIES:
            category_counts = Counter(
                a.get("failure_category", "other") for a in episode_analyses
            )
            parsed["dominant_failure_category"] = category_counts.most_common(1)[0][0]

        if not isinstance(parsed.get("overall_suggested_fixes"), list):
            parsed["overall_suggested_fixes"] = []

        parsed["token_usage"] = token_usage
        return parsed
