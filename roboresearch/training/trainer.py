from __future__ import annotations

import time
from pathlib import Path

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from roboresearch.training.configs import get_default_config, merge_config
from roboresearch.training.env_utils import create_env

_ALGO_MAP = {
    "SAC": SAC,
    "PPO": PPO,
    "TD3": TD3,
}

_OFF_POLICY_ALGOS = {"SAC", "TD3"}


class _TimeBudgetCallback(BaseCallback):
    def __init__(self, time_budget_seconds: float):
        super().__init__()
        self._budget = time_budget_seconds
        self._start_time: float | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self._start_time is None:
            self._start_time = time.time()
        return (time.time() - self._start_time) < self._budget


def create_model(
    algorithm: str,
    env_name: str,
    hyperparams: dict | None = None,
) -> BaseAlgorithm:
    algo = algorithm.upper()
    if algo not in _ALGO_MAP:
        raise ValueError(
            f"Unsupported algorithm: {algorithm!r}. Supported: {list(_ALGO_MAP)}"
        )

    defaults = get_default_config(algo, env_name)
    if hyperparams:
        config = merge_config(defaults, hyperparams)
    else:
        config = defaults

    env = create_env(env_name)
    algo_cls = _ALGO_MAP[algo]

    model_kwargs = {
        k: v
        for k, v in config.items()
        if k not in ("algorithm", "env_name")
    }

    if algo in _OFF_POLICY_ALGOS:
        model_kwargs["replay_buffer_class"] = HerReplayBuffer
        model_kwargs["replay_buffer_kwargs"] = {
            "goal_selection_strategy": "future",
            "n_sampled_goal": 4,
        }

    policy = model_kwargs.pop("policy")
    return algo_cls(policy, env, **model_kwargs)


def train_model(
    model: BaseAlgorithm,
    total_timesteps: int | None = None,
    time_budget_seconds: int | None = None,
) -> dict:
    if total_timesteps is None and time_budget_seconds is None:
        raise ValueError("Must specify either total_timesteps or time_budget_seconds")

    callbacks = []
    if time_budget_seconds is not None:
        callbacks.append(_TimeBudgetCallback(time_budget_seconds))

    steps = total_timesteps if total_timesteps is not None else 10_000_000

    start = time.time()
    model.learn(total_timesteps=steps, callback=callbacks if callbacks else None)
    elapsed = time.time() - start

    return {
        "total_timesteps_trained": model.num_timesteps,
        "elapsed_time": round(elapsed, 2),
        "final_mean_reward": _estimate_mean_reward(model),
    }


def _estimate_mean_reward(model: BaseAlgorithm) -> float | None:
    if hasattr(model, "logger") and model.logger is not None:
        name_map = model.logger.name_to_value
        for key in ["rollout/ep_rew_mean", "train/reward"]:
            if key in name_map:
                return float(name_map[key])
    return None


def save_model(model: BaseAlgorithm, path: str) -> str:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    return str(save_path)


def load_model(algorithm: str, path: str, env_name: str) -> BaseAlgorithm:
    algo = algorithm.upper()
    if algo not in _ALGO_MAP:
        raise ValueError(
            f"Unsupported algorithm: {algorithm!r}. Supported: {list(_ALGO_MAP)}"
        )

    env = create_env(env_name)
    algo_cls = _ALGO_MAP[algo]
    return algo_cls.load(path, env=env)
