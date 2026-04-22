from __future__ import annotations

import base64
import importlib
import io
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image
from sb3_contrib import ARS, CrossQ, TQC, TRPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

# Make project root importable so we can pull `make_env` / `success_from_info`
# from train.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRAME_INTERVAL = 5

ALGORITHMS = {
    "SAC": SAC, "TD3": TD3, "TQC": TQC, "CrossQ": CrossQ,
    "PPO": PPO, "DDPG": DDPG, "A2C": A2C, "TRPO": TRPO, "ARS": ARS,
}


def _train_module():
    if "train" in sys.modules:
        return importlib.reload(sys.modules["train"])
    return importlib.import_module("train")


def _build_env(env_name: str, render_mode: str | None = None) -> gym.Env:
    train = _train_module()
    if env_name == train.ENV_NAME:
        return train.make_env(render_mode=render_mode, with_reward_wrapper=False)
    if render_mode is None:
        return gym.make(env_name)
    return gym.make(env_name, render_mode=render_mode)


def _success_from_info(env_name: str, info: dict) -> bool:
    train = _train_module()
    if env_name == train.ENV_NAME:
        return train.success_from_info(info)
    return bool(info.get("is_success", False))


@dataclass
class _Episode:
    index: int
    success: bool
    total_reward: float
    length: int
    final_distance: float
    frames: list[str] = field(default_factory=list)


@dataclass
class _EvalRecord:
    eval_id: str
    model_path: str
    env_name: str
    timestamp: str
    episodes: list[_Episode] = field(default_factory=list)


mcp_server = FastMCP("RoboResearch Evaluation")
_store: dict[str, _EvalRecord] = {}


def _load_model(model_path: str, env: gym.Env):
    """Load an SB3 model, trying each algorithm class in turn. Passing the env
    is required when the model was saved with a wrapped obs space."""
    for cls in ALGORITHMS.values():
        try:
            return cls.load(model_path, env=env)
        except Exception:
            continue
    raise ValueError(f"Could not load model from {model_path}")


def _frame_to_b64(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _final_distance(obs) -> float:
    if isinstance(obs, dict):
        achieved = np.asarray(obs.get("achieved_goal", []))
        desired = np.asarray(obs.get("desired_goal", []))
        if achieved.size > 0 and desired.size > 0:
            return float(np.linalg.norm(achieved - desired))
    return -1.0


def _run_episode(model, env: gym.Env, env_name: str, idx: int, capture: bool = True) -> _Episode:
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    frames: list[str] = []
    succeeded = False

    if capture:
        frame = env.render()
        if frame is not None:
            frames.append(_frame_to_b64(frame))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        if _success_from_info(env_name, info):
            succeeded = True

        if capture and steps % FRAME_INTERVAL == 0:
            frame = env.render()
            if frame is not None:
                frames.append(_frame_to_b64(frame))

    if capture:
        frame = env.render()
        if frame is not None:
            frames.append(_frame_to_b64(frame))

    return _Episode(
        index=idx,
        success=succeeded,
        total_reward=total_reward,
        length=steps,
        final_distance=_final_distance(obs),
        frames=frames if not succeeded else [],
    )


@mcp_server.tool()
def run_evaluation(model_path: str, env_name: str, num_episodes: int = 10) -> dict:
    """Run evaluation episodes and return summary statistics.

    Args:
        model_path: Path to a saved SB3 model.
        env_name: Gymnasium environment ID.
        num_episodes: Number of evaluation episodes to run.
    """
    env = _build_env(env_name, render_mode="rgb_array")
    model = _load_model(model_path, env)

    eval_id = f"eval-{uuid.uuid4().hex[:12]}"
    record = _EvalRecord(
        eval_id=eval_id,
        model_path=model_path,
        env_name=env_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        for i in range(num_episodes):
            record.episodes.append(_run_episode(model, env, env_name, i))
    finally:
        env.close()

    _store[eval_id] = record

    successes = sum(1 for ep in record.episodes if ep.success)
    rewards = [ep.total_reward for ep in record.episodes]

    return {
        "evaluation_id": eval_id,
        "summary": {
            "success_rate": successes / len(record.episodes),
            "mean_reward": float(np.mean(rewards)),
            "num_episodes": len(record.episodes),
        },
    }


@mcp_server.tool()
def compute_metrics(evaluation_id: str) -> dict:
    """Compute aggregate metrics for a completed evaluation."""
    record = _store.get(evaluation_id)
    if record is None:
        raise ValueError(f"No evaluation found: {evaluation_id}")

    episodes = record.episodes
    rewards = [ep.total_reward for ep in episodes]
    lengths = [ep.length for ep in episodes]
    distances = [ep.final_distance for ep in episodes]
    successes = sum(1 for ep in episodes if ep.success)

    return {
        "evaluation_id": evaluation_id,
        "success_rate": successes / len(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_final_distance": float(np.mean(distances)),
        "num_episodes": len(episodes),
    }


@mcp_server.tool()
def compare_runs(eval_id_a: str, eval_id_b: str) -> dict:
    """Compare metrics from two evaluation runs side-by-side."""
    a = compute_metrics(eval_id_a)
    b = compute_metrics(eval_id_b)

    keys = ["success_rate", "mean_reward", "std_reward", "mean_episode_length",
            "min_reward", "max_reward", "mean_final_distance"]

    deltas = {}
    for k in keys:
        abs_d = b[k] - a[k]
        pct_d = (abs_d / abs(a[k]) * 100) if a[k] != 0 else float("inf")
        deltas[k] = {"absolute": abs_d, "percentage": pct_d}

    better_a = a["success_rate"] > b["success_rate"]
    if a["success_rate"] == b["success_rate"]:
        better_a = a["mean_reward"] > b["mean_reward"]

    return {
        "eval_a": a,
        "eval_b": b,
        "deltas": deltas,
        "which_is_better": eval_id_a if better_a else eval_id_b,
    }


@mcp_server.tool()
def get_failure_episodes(evaluation_id: str) -> dict:
    """Return detailed info and frame captures for failed episodes.

    Args:
        evaluation_id: ID from a previous run_evaluation call.
    """
    record = _store.get(evaluation_id)
    if record is None:
        raise ValueError(f"No evaluation found: {evaluation_id}")

    failures = []
    for ep in record.episodes:
        if ep.success:
            continue
        failures.append({
            "episode_index": ep.index,
            "frames": ep.frames,
            "reward": ep.total_reward,
            "episode_length": ep.length,
            "final_distance": ep.final_distance,
        })

    return {
        "evaluation_id": evaluation_id,
        "num_failures": len(failures),
        "total_episodes": len(record.episodes),
        "failure_rate": len(failures) / len(record.episodes),
        "episodes": failures,
    }


@mcp_server.tool()
def generate_report(evaluation_id: str) -> dict:
    """Generate a markdown evaluation report with metrics and failure analysis."""
    record = _store.get(evaluation_id)
    if record is None:
        raise ValueError(f"No evaluation found: {evaluation_id}")

    metrics = compute_metrics(evaluation_id)
    failures = get_failure_episodes(evaluation_id)

    report = f"""# Evaluation Report

- **ID:** `{evaluation_id}`
- **Model:** `{record.model_path}`
- **Environment:** `{record.env_name}`
- **Timestamp:** {record.timestamp}
- **Episodes:** {metrics['num_episodes']}

## Metrics

| Metric | Value |
|--------|-------|
| Success Rate | {metrics['success_rate']:.2%} |
| Mean Reward | {metrics['mean_reward']:.3f} |
| Std Reward | {metrics['std_reward']:.3f} |
| Mean Episode Length | {metrics['mean_episode_length']:.1f} |
| Mean Final Distance | {metrics['mean_final_distance']:.4f} |

## Failures

{failures['num_failures']} of {failures['total_episodes']} episodes failed ({failures['failure_rate']:.2%}).
"""

    return {"evaluation_id": evaluation_id, "report_markdown": report.strip()}


if __name__ == "__main__":
    mcp_server.run()
