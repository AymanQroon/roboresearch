from __future__ import annotations

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from roboresearch.training.env_utils import create_env


def evaluate_model(
    model: BaseAlgorithm,
    env_name: str,
    num_episodes: int,
    capture_frames: bool = False,
    frame_interval: int = 5,
) -> dict:
    render_mode = "rgb_array" if capture_frames else None
    env = create_env(env_name, render_mode=render_mode)

    episodes = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            done = terminated or truncated

            if capture_frames and step_count % frame_interval == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        episode_data = {
            "success": bool(info.get("is_success", False)),
            "total_reward": total_reward,
            "episode_length": step_count,
            "final_distance": float(info.get("distance", 0.0)),
        }
        if capture_frames:
            episode_data["frames"] = frames

        episodes.append(episode_data)

    env.close()

    summary = compute_summary_metrics(episodes)
    return {"episodes": episodes, "summary": summary}


def compute_summary_metrics(episodes: list) -> dict:
    successes = [ep["success"] for ep in episodes]
    rewards = [ep["total_reward"] for ep in episodes]
    lengths = [ep["episode_length"] for ep in episodes]
    distances = [ep["final_distance"] for ep in episodes]

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_final_distance": float(np.mean(distances)),
    }
