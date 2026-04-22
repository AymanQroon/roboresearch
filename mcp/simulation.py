from __future__ import annotations

import base64
import importlib
import io
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image
from sb3_contrib import ARS, CrossQ, TQC, TRPO
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

# Make project root importable so we can pull `make_env` from train.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

mcp_server = FastMCP("RoboResearch Simulation")

ALGORITHMS = {
    "SAC": SAC, "TD3": TD3, "TQC": TQC, "CrossQ": CrossQ,
    "PPO": PPO, "DDPG": DDPG, "A2C": A2C, "TRPO": TRPO, "ARS": ARS,
}
OFF_POLICY = {"SAC", "TD3", "TQC", "CrossQ", "DDPG"}


def _train_module():
    """Import (or reload) train.py to pick up the agent's latest edits."""
    if "train" in sys.modules:
        return importlib.reload(sys.modules["train"])
    return importlib.import_module("train")


def _build_env(env_name: str, render_mode: str | None = None) -> gym.Env:
    """Construct the env. For the active task in train.py, use its make_env so
    wrappers (FlattenSingleSubtaskGoal, RewardWrapper) match the trained model."""
    train = _train_module()
    if env_name == train.ENV_NAME:
        return train.make_env(render_mode=render_mode)
    if render_mode is None:
        return gym.make(env_name)
    return gym.make(env_name, render_mode=render_mode)


class _State:
    def __init__(self) -> None:
        self.env: gym.Env | None = None
        self.env_name: str | None = None
        self.last_run: dict[str, Any] | None = None
        self.temp_dir = Path(tempfile.mkdtemp(prefix="roboresearch_"))

    def cleanup(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self.env_name = None
        for f in self.temp_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass


state = _State()


class TimeBudgetCallback(BaseCallback):
    def __init__(self, budget: int) -> None:
        super().__init__()
        self._budget = budget
        self._start: float = 0.0

    def _on_training_start(self) -> None:
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        return (time.monotonic() - self._start) < self._budget


@mcp_server.tool()
def configure_env(task: str) -> dict[str, Any]:
    """Create a Gymnasium-Robotics environment.

    Args:
        task: Environment ID. The active task defined in train.py is built with
              its wrappers (FlattenSingleSubtaskGoal + RewardWrapper); other env
              IDs are constructed bare via gym.make.
    """
    state.cleanup()
    try:
        env = _build_env(task)
    except Exception as exc:
        return {"error": f"Failed to build env '{task}': {exc}"}
    state.env = env
    state.env_name = task

    return {
        "status": "configured",
        "task": task,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    }


@mcp_server.tool()
def run_training(
    algorithm: str,
    hyperparams: dict[str, Any] | None = None,
    time_budget_seconds: int = 600,
    env_name: str | None = None,
) -> dict[str, Any]:
    """Launch an SB3 training run with time-budget stopping.

    Args:
        algorithm: RL algorithm (SAC, TD3, TQC, PPO, ...).
        hyperparams: SB3 constructor kwargs (learning_rate, batch_size, etc.).
        time_budget_seconds: Wall-clock seconds to train.
        env_name: Environment ID. Uses currently configured env if omitted.
    """
    if hyperparams is None:
        hyperparams = {}

    algo_key = algorithm.upper()
    if algo_key not in ALGORITHMS:
        return {"error": f"Unsupported algorithm. Supported: {sorted(ALGORITHMS)}"}

    task = env_name or state.env_name
    if task is None:
        return {"error": "No environment configured. Call configure_env first or pass env_name."}

    train_env = _build_env(task)
    algo_cls = ALGORITHMS[algo_key]

    kwargs = dict(hyperparams)
    if algo_key in OFF_POLICY and "replay_buffer_class" not in kwargs:
        kwargs["replay_buffer_class"] = HerReplayBuffer
        kwargs["replay_buffer_kwargs"] = kwargs.get("replay_buffer_kwargs", {
            "goal_selection_strategy": "future",
            "n_sampled_goal": 4,
        })

    model = algo_cls("MultiInputPolicy", train_env, **kwargs, verbose=0)

    train_start = time.monotonic()
    model.learn(total_timesteps=10_000_000, callback=TimeBudgetCallback(time_budget_seconds))
    training_time = time.monotonic() - train_start

    run_id = uuid.uuid4().hex[:12]
    model_path = str(state.temp_dir / f"model_{run_id}")
    model.save(model_path)

    state.last_run = {
        "run_id": run_id,
        "algorithm": algo_key,
        "env_name": task,
        "hyperparams": hyperparams,
        "total_timesteps": int(model.num_timesteps),
        "training_time_seconds": round(training_time, 2),
        "model_path": model_path,
    }

    train_env.close()

    return {
        "status": "completed",
        "run_id": run_id,
        "model_path": model_path,
        "total_timesteps": int(model.num_timesteps),
        "training_time_seconds": round(training_time, 2),
    }


@mcp_server.tool()
def capture_frames(
    env_name: str | None = None,
    model_path: str | None = None,
    num_frames: int = 10,
) -> dict[str, Any]:
    """Capture rendered frames from a policy rollout for VLM analysis.

    Args:
        env_name: Environment ID. Uses currently configured env if omitted.
        model_path: Path to a saved SB3 model. Uses last training run if omitted.
        num_frames: Number of frames to capture from the episode.
    """
    task = env_name or state.env_name
    if task is None:
        return {"error": "No environment configured."}

    resolved_path = model_path
    if resolved_path is None and state.last_run is not None:
        resolved_path = state.last_run["model_path"]
    if resolved_path is None:
        return {"error": "No model path provided and no previous training run found."}

    try:
        render_env = _build_env(task, render_mode="rgb_array")
    except Exception as exc:
        return {"error": f"Failed to build render env '{task}': {exc}"}

    algo_key = None
    if state.last_run and state.last_run.get("model_path") == resolved_path:
        algo_key = state.last_run["algorithm"]

    model = None
    if algo_key and algo_key in ALGORITHMS:
        model = ALGORITHMS[algo_key].load(resolved_path, env=render_env)
    else:
        for cls in ALGORITHMS.values():
            try:
                model = cls.load(resolved_path, env=render_env)
                break
            except Exception:
                continue
        if model is None:
            render_env.close()
            return {"error": f"Could not load model from '{resolved_path}'."}

    obs, _ = render_env.reset()
    all_frames: list[np.ndarray] = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        frame = render_env.render()
        if frame is not None:
            all_frames.append(frame)

    if not all_frames:
        render_env.close()
        return {"error": "Episode produced no frames."}

    indices = np.linspace(0, len(all_frames) - 1, min(num_frames, len(all_frames)), dtype=int)
    selected = [all_frames[i] for i in indices]

    encoded: list[str] = []
    for frame in selected:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))

    render_env.close()

    return {
        "status": "captured",
        "total_episode_steps": len(all_frames),
        "num_frames_returned": len(encoded),
        "frames_base64_png": encoded,
    }


@mcp_server.tool()
def get_training_log() -> dict[str, Any]:
    """Return training metadata from the most recent training run."""
    if state.last_run is None:
        return {"error": "No training runs recorded."}
    return {"status": "ok", **state.last_run}


@mcp_server.tool()
def reset_env() -> dict[str, Any]:
    """Tear down any active environment and clean up temporary files."""
    state.cleanup()
    state.last_run = None
    return {"status": "reset"}


if __name__ == "__main__":
    mcp_server.run()
