from __future__ import annotations

import base64
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.base_class import BaseAlgorithm

from roboresearch.training.env_utils import create_env


def record_episode(model: BaseAlgorithm, env_name: str, output_path: str) -> str:
    output = Path(output_path)
    video_dir = output.parent
    video_dir.mkdir(parents=True, exist_ok=True)

    env = create_env(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=output.stem,
        episode_trigger=lambda _: True,
    )

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()

    recorded_files = sorted(video_dir.glob(f"{output.stem}*.mp4"))
    if recorded_files:
        final_path = recorded_files[-1]
        if final_path != output:
            shutil.move(str(final_path), str(output))
            final_path = output
        return str(final_path)

    return str(output)


def record_best_and_worst(
    model: BaseAlgorithm,
    env_name: str,
    num_episodes: int,
    output_dir: str,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = create_env(env_name, render_mode="rgb_array")

    episode_results = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        episode_results.append({"index": i, "reward": total_reward})

    env.close()

    episode_results.sort(key=lambda x: x["reward"])
    worst = episode_results[0]
    best = episode_results[-1]

    best_path = str(out / "best_episode.mp4")
    worst_path = str(out / "worst_episode.mp4")

    best_video = record_episode(model, env_name, best_path)
    worst_video = record_episode(model, env_name, worst_path)

    return {
        "best": {
            "video_path": best_video,
            "reward": best["reward"],
            "episode_index": best["index"],
        },
        "worst": {
            "video_path": worst_video,
            "reward": worst["reward"],
            "episode_index": worst["index"],
        },
    }


def frames_to_base64(frames: list[np.ndarray]) -> list[str]:
    encoded = []
    for frame in frames:
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded.append(b64)
    return encoded
