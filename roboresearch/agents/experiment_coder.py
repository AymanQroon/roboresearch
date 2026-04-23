from __future__ import annotations

import json
import random
import re
from copy import deepcopy

import anthropic

from roboresearch.agents.client import MODELS, create_client
from roboresearch.training.configs import SUPPORTED_ALGORITHMS, get_default_config

SUPPORTED_ENVS = [
    "FetchReach-v4",
    "FetchPush-v4",
    "FetchSlide-v4",
    "FetchPickAndPlace-v4",
]

_MODEL = MODELS["experiment_coder"]

_VALID_RANGES = {
    "learning_rate": (1e-5, 1e-2),
    "batch_size": (32, 2048),
    "buffer_size": (10_000, 1_000_000),
    "gamma": (0.9, 0.999),
    "tau": (0.001, 0.05),
    "learning_starts": (100, 100_000),
    "n_steps": (64, 8192),
    "gae_lambda": (0.8, 1.0),
    "clip_range": (0.05, 0.5),
    "ent_coef": (0.0, 0.1),
}

_NET_ARCH_RANGE = (64, 512)

_ALGO_SPECIFIC_KEYS = {
    "SAC": {"buffer_size", "tau", "learning_starts"},
    "PPO": {"n_steps", "gae_lambda", "clip_range", "ent_coef"},
    "TD3": {"buffer_size", "tau", "learning_starts"},
}

_SYSTEM_PROMPT = """\
You are an expert reinforcement learning engineer generating training configurations \
for robot manipulation tasks in MuJoCo simulation.

## Supported Algorithms
SAC, PPO, TD3

## Supported Environments
FetchReach-v4, FetchPush-v4, FetchSlide-v4, FetchPickAndPlace-v4

## Config Schemas

### SAC
- algorithm: "SAC"
- env_name: one of the supported environments
- policy: "MultiInputPolicy"
- learning_rate: float (1e-5 to 1e-2). Controls step size for gradient updates.
- batch_size: int (32 to 2048). Number of transitions sampled per gradient step.
- buffer_size: int (10000 to 1000000). Replay buffer capacity.
- tau: float (0.001 to 0.05). Soft update coefficient for target networks.
- gamma: float (0.9 to 0.999). Discount factor for future rewards.
- learning_starts: int (100 to 100000). Steps of random exploration before training.
- policy_kwargs: {"net_arch": list of ints (each 64-512)}. Hidden layer sizes.

### PPO
- algorithm: "PPO"
- env_name: one of the supported environments
- policy: "MultiInputPolicy"
- learning_rate: float (1e-5 to 1e-2). Controls step size for gradient updates.
- batch_size: int (32 to 2048). Minibatch size for PPO updates.
- n_steps: int (64 to 8192). Steps collected per environment before each update.
- gamma: float (0.9 to 0.999). Discount factor for future rewards.
- gae_lambda: float (0.8 to 1.0). Bias-variance tradeoff for advantage estimation.
- clip_range: float (0.05 to 0.5). PPO clipping range for policy updates.
- ent_coef: float (0.0 to 0.1). Entropy bonus coefficient for exploration.
- policy_kwargs: {"net_arch": list of ints (each 64-512)}. Hidden layer sizes.

### TD3
- algorithm: "TD3"
- env_name: one of the supported environments
- policy: "MultiInputPolicy"
- learning_rate: float (1e-5 to 1e-2). Controls step size for gradient updates.
- batch_size: int (32 to 2048). Number of transitions sampled per gradient step.
- buffer_size: int (10000 to 1000000). Replay buffer capacity.
- tau: float (0.001 to 0.05). Soft update coefficient for target networks.
- gamma: float (0.9 to 0.999). Discount factor for future rewards.
- learning_starts: int (100 to 100000). Steps of random exploration before training.
- policy_kwargs: {"net_arch": list of ints (each 64-512)}. Hidden layer sizes.

## Output Format

Return EXACTLY this structure. The config JSON must be complete and valid. \
Do NOT include algorithm-specific parameters that don't belong to the chosen algorithm \
(e.g., no buffer_size/tau/learning_starts in PPO, no n_steps/gae_lambda/clip_range/ent_coef in SAC or TD3).

```json
{
  "config": {
    "algorithm": "...",
    "env_name": "...",
    "policy": "MultiInputPolicy",
    ...all relevant hyperparameters...
  },
  "changes": [
    "Changed X from A to B because ...",
    "Changed Y from C to D because ..."
  ],
  "reasoning": "One paragraph explaining why these changes as a whole should improve performance."
}
```
"""


def _build_user_message(
    experiment_plan: str,
    current_config: dict,
    past_experiments: list[dict],
    failure_analysis: str | None,
    env_name: str,
) -> str:
    parts = [
        f"## Experiment Plan\n{experiment_plan}",
        f"\n## Target Environment\n{env_name}",
        f"\n## Current Config\n```json\n{json.dumps(current_config, indent=2)}\n```",
    ]

    if past_experiments:
        recent = past_experiments[-5:]
        parts.append("\n## Past Experiments (most recent last)")
        for i, exp in enumerate(recent, 1):
            parts.append(f"\n### Experiment {i}")
            if "config" in exp:
                parts.append(f"Config: ```json\n{json.dumps(exp['config'], indent=2)}\n```")
            if "metrics" in exp:
                parts.append(f"Metrics: ```json\n{json.dumps(exp['metrics'], indent=2)}\n```")
            if "notes" in exp:
                parts.append(f"Notes: {exp['notes']}")

    if failure_analysis:
        parts.append(f"\n## Failure Analysis\n{failure_analysis}")

    parts.append(
        "\nGenerate a new training config based on the experiment plan. "
        "Respond with ONLY the JSON block in the format specified."
    )

    return "\n".join(parts)


def _extract_json(text: str) -> dict | None:
    pattern = r"```json\s*\n(.*?)\n\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _validate_config(config: dict, env_name: str) -> dict:
    algo = config.get("algorithm", "").upper()
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Invalid algorithm: {config.get('algorithm')}")

    resolved_env = config.get("env_name", env_name)
    if resolved_env not in SUPPORTED_ENVS:
        raise ValueError(f"Invalid env_name: {resolved_env}")

    defaults = get_default_config(algo, resolved_env)
    valid_keys = set(defaults.keys())

    validated = {"algorithm": algo, "env_name": resolved_env, "policy": "MultiInputPolicy"}

    for key in valid_keys:
        if key in ("algorithm", "env_name", "policy"):
            continue

        value = config.get(key, defaults[key])

        if key == "policy_kwargs":
            value = _validate_policy_kwargs(value, defaults[key])
        elif key in _VALID_RANGES:
            lo, hi = _VALID_RANGES[key]
            if isinstance(value, (int, float)):
                value = type(defaults[key])(_clamp(value, lo, hi))
            else:
                value = defaults[key]

        validated[key] = value

    return validated


def _validate_policy_kwargs(kwargs: dict | None, defaults: dict) -> dict:
    if not isinstance(kwargs, dict):
        return deepcopy(defaults)

    result = deepcopy(defaults)

    if "net_arch" in kwargs:
        arch = kwargs["net_arch"]
        if isinstance(arch, list) and all(isinstance(x, (int, float)) for x in arch):
            lo, hi = _NET_ARCH_RANGE
            result["net_arch"] = [int(_clamp(x, lo, hi)) for x in arch]

    return result


def _perturb_config(config: dict) -> dict:
    perturbed = deepcopy(config)
    lr = perturbed.get("learning_rate", 3e-4)
    factor = random.uniform(0.5, 2.0)
    lo, hi = _VALID_RANGES["learning_rate"]
    perturbed["learning_rate"] = _clamp(lr * factor, lo, hi)
    return perturbed


class ExperimentCoder:
    def __init__(self, **kwargs):
        self._client = create_client()

    def generate_config(
        self,
        experiment_plan: str,
        current_config: dict,
        past_experiments: list[dict] | None = None,
        failure_analysis: str | None = None,
        env_name: str = "FetchReach-v4",
    ) -> dict:
        if past_experiments is None:
            past_experiments = []

        user_message = _build_user_message(
            experiment_plan, current_config, past_experiments, failure_analysis, env_name
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as exc:
            return self._fallback_result(current_config, env_name, str(exc))

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        raw_text = response.content[0].text
        parsed = _extract_json(raw_text)

        if parsed is None:
            result = self._fallback_result(
                current_config, env_name, "Failed to parse JSON from model response"
            )
            result["usage"] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
            return result

        raw_config = parsed.get("config", parsed)
        changes = parsed.get("changes", [])
        reasoning = parsed.get("reasoning", "")

        try:
            validated_config = _validate_config(raw_config, env_name)
        except ValueError:
            result = self._fallback_result(
                current_config, env_name, "Validation failed on model-generated config"
            )
            result["usage"] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
            return result

        return {
            "config": validated_config,
            "changes": changes,
            "reasoning": reasoning,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }

    def _fallback_result(self, current_config: dict, env_name: str, error: str) -> dict:
        fallback = _perturb_config(current_config)
        fallback["env_name"] = env_name
        if "algorithm" not in fallback:
            fallback["algorithm"] = "SAC"
        return {
            "config": fallback,
            "changes": [f"Fallback: minor learning_rate perturbation due to error: {error}"],
            "reasoning": "Automatic fallback with small random perturbation of learning rate.",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
