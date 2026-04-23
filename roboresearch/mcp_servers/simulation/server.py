from __future__ import annotations

import base64
import io
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback

mcp = FastMCP("RoboResearch Simulation Server")

SUPPORTED_ENVS = {
    "FetchReach-v4",
    "FetchPush-v4",
    "FetchSlide-v4",
    "FetchPickAndPlace-v4",
}

ALGORITHMS = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}


class _ServerState:
    def __init__(self) -> None:
        self.env: gym.Env | None = None
        self.env_name: str | None = None
        self.env_params: dict[str, Any] = {}
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
            self.env_params = {}

        for f in self.temp_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass


state = _ServerState()


class TimeBudgetCallback(BaseCallback):
    """Stops training after a wall-clock time budget is exhausted."""

    def __init__(self, time_budget_seconds: int) -> None:
        super().__init__()
        self.time_budget_seconds = time_budget_seconds
        self.start_time: float = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.monotonic()

    def _on_step(self) -> bool:
        elapsed = time.monotonic() - self.start_time
        return elapsed < self.time_budget_seconds


def _space_info(space: gym.Space) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(space).__name__}
    if hasattr(space, "shape"):
        info["shape"] = list(space.shape)
    if hasattr(space, "n"):
        info["n"] = int(space.n)
    if isinstance(space, gym.spaces.Dict):
        info["spaces"] = {k: _space_info(v) for k, v in space.spaces.items()}
    return info


def _make_env(task: str, params: dict[str, Any], render_mode: str | None = None) -> gym.Env:
    kwargs: dict[str, Any] = {}
    if render_mode:
        kwargs["render_mode"] = render_mode

    params = dict(params)
    max_episode_steps = params.pop("max_episode_steps", None)
    kwargs.update(params)

    env = gym.make(task, max_episode_steps=max_episode_steps, **kwargs)
    return env


@mcp.tool()
def configure_env(task: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a Gymnasium-Robotics environment.

    Args:
        task: Environment ID (FetchReach-v4, FetchPush-v4, FetchSlide-v4, FetchPickAndPlace-v4).
        params: Optional env kwargs like max_episode_steps, reward_type.
    """
    if params is None:
        params = {}

    if task not in SUPPORTED_ENVS:
        return {"error": f"Unsupported environment '{task}'. Supported: {sorted(SUPPORTED_ENVS)}"}

    try:
        state.cleanup()
        env = _make_env(task, params)
        state.env = env
        state.env_name = task
        state.env_params = params

        return {
            "status": "configured",
            "task": task,
            "observation_space": _space_info(env.observation_space),
            "action_space": _space_info(env.action_space),
        }
    except Exception as e:
        return {"error": f"Failed to configure environment: {e}"}


@mcp.tool()
def run_training(
    algorithm: str,
    hyperparams: dict[str, Any] | None = None,
    time_budget_seconds: int = 120,
    env_name: str | None = None,
) -> dict[str, Any]:
    """Launch an SB3 training run.

    Args:
        algorithm: RL algorithm (SAC, PPO, TD3).
        hyperparams: SB3 constructor kwargs (learning_rate, batch_size, etc.).
        time_budget_seconds: Wall-clock seconds to train.
        env_name: Environment ID. Uses currently configured env if omitted.
    """
    if hyperparams is None:
        hyperparams = {}

    algorithm_upper = algorithm.upper()
    if algorithm_upper not in ALGORITHMS:
        return {"error": f"Unsupported algorithm '{algorithm}'. Supported: {sorted(ALGORITHMS)}"}

    task = env_name or state.env_name
    if task is None:
        return {"error": "No environment configured. Call configure_env first or pass env_name."}
    if task not in SUPPORTED_ENVS:
        return {"error": f"Unsupported environment '{task}'. Supported: {sorted(SUPPORTED_ENVS)}"}

    try:
        train_env = _make_env(task, dict(state.env_params) if env_name is None else {})

        algo_cls = ALGORITHMS[algorithm_upper]
        model = algo_cls("MultiInputPolicy", train_env, **hyperparams, verbose=0)

        callback = TimeBudgetCallback(time_budget_seconds)
        # Train with a large total_timesteps; the callback enforces the time budget.
        train_start = time.monotonic()
        model.learn(total_timesteps=10_000_000, callback=callback)
        training_time = time.monotonic() - train_start

        run_id = uuid.uuid4().hex[:12]
        model_path = str(state.temp_dir / f"model_{run_id}")
        model.save(model_path)

        state.last_run = {
            "run_id": run_id,
            "algorithm": algorithm_upper,
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
    except Exception as e:
        return {"error": f"Training failed: {e}"}


@mcp.tool()
def capture_frames(
    env_name: str | None = None,
    model_path: str | None = None,
    num_frames: int = 10,
) -> dict[str, Any]:
    """Capture rendered frames from a policy rollout.

    Args:
        env_name: Environment ID. Uses currently configured env if omitted.
        model_path: Path to a saved SB3 model. Uses last training run if omitted.
        num_frames: Number of frames to capture from the episode.
    """
    task = env_name or state.env_name
    if task is None:
        return {"error": "No environment configured. Call configure_env first or pass env_name."}

    resolved_model_path = model_path
    if resolved_model_path is None and state.last_run is not None:
        resolved_model_path = state.last_run["model_path"]
    if resolved_model_path is None:
        return {"error": "No model path provided and no previous training run found."}

    try:
        render_env = _make_env(
            task,
            dict(state.env_params) if env_name is None else {},
            render_mode="rgb_array",
        )

        algo_upper = None
        if state.last_run and state.last_run.get("model_path") == resolved_model_path:
            algo_upper = state.last_run["algorithm"]

        model = None
        if algo_upper and algo_upper in ALGORITHMS:
            model = ALGORITHMS[algo_upper].load(resolved_model_path, env=render_env)
        else:
            for cls in [SAC, TD3, PPO]:
                try:
                    model = cls.load(resolved_model_path, env=render_env)
                    break
                except Exception:
                    continue
            if model is None:
                render_env.close()
                return {"error": f"Could not load model from '{resolved_model_path}'."}

        obs, _ = render_env.reset()
        all_frames: list[np.ndarray] = []

        done = False
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            step_count += 1
            all_frames.append(render_env.render())

        if len(all_frames) == 0:
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
            "total_episode_steps": step_count,
            "num_frames_returned": len(encoded),
            "frames_base64_png": encoded,
        }
    except Exception as e:
        return {"error": f"Frame capture failed: {e}"}


@mcp.tool()
def get_training_log(run_id: str | None = None) -> dict[str, Any]:
    """Return training metrics from a training run.

    Args:
        run_id: ID of the run to query. Uses the most recent run if omitted.
    """
    if state.last_run is None:
        return {"error": "No training runs recorded."}

    if run_id is not None and state.last_run["run_id"] != run_id:
        return {"error": f"Run '{run_id}' not found. Only the most recent run is stored."}

    return {
        "status": "ok",
        "run_id": state.last_run["run_id"],
        "algorithm": state.last_run["algorithm"],
        "env_name": state.last_run["env_name"],
        "hyperparams": state.last_run["hyperparams"],
        "total_timesteps": state.last_run["total_timesteps"],
        "training_time_seconds": state.last_run["training_time_seconds"],
        "model_path": state.last_run["model_path"],
    }


@mcp.tool()
def reset_env() -> dict[str, Any]:
    """Tear down any active environment and clean up temporary files."""
    try:
        state.cleanup()
        state.last_run = None
        return {"status": "reset", "message": "Environment and temporary files cleaned up."}
    except Exception as e:
        return {"error": f"Reset failed: {e}"}


if __name__ == "__main__":
    mcp.run()
