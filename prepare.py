"""
Read-only infrastructure for RoboResearch.

Provides evaluation utilities, frame capture, reward component tracking,
and result logging. The AI agent should NOT modify this file — only
train.py is editable.

Env construction and success detection are imported from train.py so this
file stays in sync with whatever task train.py defines.
"""

from __future__ import annotations

import base64
import csv
import importlib
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_FILE = Path("results.tsv")
FAILURES_DIR = Path("failures")


def _train_module():
    if "train" in sys.modules:
        return importlib.reload(sys.modules["train"])
    return importlib.import_module("train")


def _build_env(env_name: str, render_mode: str | None = None,
               with_reward_wrapper: bool = False) -> gym.Env:
    train = _train_module()
    if env_name == train.ENV_NAME:
        return train.make_env(render_mode=render_mode, with_reward_wrapper=with_reward_wrapper)
    if render_mode is None:
        return gym.make(env_name)
    return gym.make(env_name, render_mode=render_mode)


def _success_from_info(env_name: str, info: dict) -> bool:
    train = _train_module()
    if env_name == train.ENV_NAME:
        return train.success_from_info(info)
    return bool(info.get("is_success", False))


def evaluate_model(model, env_name: str, num_episodes: int = 20) -> dict:
    """Run evaluation episodes and return detailed results."""
    env = _build_env(env_name)
    episodes = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward, step_count, done, succeeded = 0.0, 0, False, False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            if _success_from_info(env_name, info):
                succeeded = True

        episodes.append({
            "success": succeeded,
            "total_reward": total_reward,
            "episode_length": step_count,
            "final_distance": float(info.get("distance", 0.0)),
        })

    env.close()

    successes = [ep["success"] for ep in episodes]
    rewards = [ep["total_reward"] for ep in episodes]

    return {
        "episodes": episodes,
        "summary": {
            "success_rate": float(np.mean(successes)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_episode_length": float(np.mean([ep["episode_length"] for ep in episodes])),
        },
    }


def evaluate_with_reward_components(
    model, env, num_episodes: int = 5,
) -> dict:
    """Run evaluation on a reward-wrapped env and collect per-component breakdowns.

    The env should be a RewardWrapper that populates info['reward_components'].
    """
    train = _train_module()
    env_name = getattr(env.unwrapped.spec, "id", train.ENV_NAME)
    episodes = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, succeeded = False, False
        component_sums: dict[str, float] = {}
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
            if _success_from_info(env_name, info):
                succeeded = True

            components = info.get("reward_components", {})
            for k, v in components.items():
                component_sums[k] = component_sums.get(k, 0.0) + float(v)

        episodes.append({
            "success": succeeded,
            "step_count": step_count,
            "component_means": {k: v / max(step_count, 1) for k, v in component_sums.items()},
            "component_totals": component_sums,
        })

    successes = sum(1 for ep in episodes if ep["success"])
    return {
        "success_rate": successes / len(episodes),
        "num_episodes": len(episodes),
        "episodes": episodes,
        "aggregate_component_means": _aggregate_components(episodes),
    }


def _aggregate_components(episodes: list[dict]) -> dict[str, float]:
    """Average per-component means across episodes."""
    if not episodes:
        return {}
    all_keys = set()
    for ep in episodes:
        all_keys.update(ep["component_means"].keys())
    result = {}
    for k in sorted(all_keys):
        vals = [ep["component_means"].get(k, 0.0) for ep in episodes]
        result[k] = float(np.mean(vals))
    return result


def capture_failure_frames(
    model, env_name: str, num_episodes: int = 5, frame_interval: int = 5,
) -> list[list[np.ndarray]]:
    """Run episodes, return frames from failed ones for VLM analysis."""
    env = _build_env(env_name, render_mode="rgb_array")
    failure_frames = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        frames, done, succeeded = [], False, False

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if _success_from_info(env_name, info):
                succeeded = True

        final = env.render()
        if final is not None:
            frames.append(final)

        if not succeeded and frames:
            sampled = frames[::frame_interval]
            if frames[-1] is not sampled[-1]:
                sampled.append(frames[-1])
            failure_frames.append(sampled)

    env.close()
    return failure_frames


def frames_to_base64(frames: list[np.ndarray]) -> list[str]:
    """Convert numpy frames to base64-encoded PNGs."""
    result = []
    for frame in frames:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        result.append(b64)
    return result


def save_failure_frames(frames: list[np.ndarray], episode_id: int) -> Path:
    """Save failure frames as PNGs to the failures directory."""
    episode_dir = FAILURES_DIR / f"episode_{episode_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(episode_dir / f"frame_{i:03d}.png")

    return episode_dir


def log_result(
    run_id: str,
    algorithm: str,
    env_name: str,
    success_rate: float,
    mean_reward: float,
    notes: str = "",
) -> None:
    """Append a result row to results.tsv."""
    file_exists = RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "run_id", "timestamp", "algorithm", "env_name",
                "success_rate", "mean_reward", "notes",
            ])
        writer.writerow([
            run_id,
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            algorithm,
            env_name,
            f"{success_rate:.4f}",
            f"{mean_reward:.2f}",
            notes,
        ])


def load_results() -> list[dict]:
    """Load all results from results.tsv."""
    if not RESULTS_FILE.exists():
        return []

    results = []
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["success_rate"] = float(row["success_rate"])
            row["mean_reward"] = float(row["mean_reward"])
            results.append(row)
    return results


def get_best_result(env_name: str) -> dict | None:
    """Find the best result for a given environment."""
    results = load_results()
    matching = [r for r in results if r["env_name"] == env_name]
    if not matching:
        return None
    return max(matching, key=lambda r: r["success_rate"])
