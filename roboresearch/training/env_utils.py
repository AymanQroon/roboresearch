from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch environments

SUPPORTED_ENVS = [
    "FetchReach-v4",
    "FetchPush-v4",
    "FetchSlide-v4",
    "FetchPickAndPlace-v4",
]


def create_env(env_name: str, render_mode: str | None = None, **kwargs) -> gym.Env:
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(
            f"Unsupported environment: {env_name!r}. "
            f"Supported: {SUPPORTED_ENVS}"
        )

    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    env_kwargs.update(kwargs)

    return gym.make(env_name, **env_kwargs)


def get_env_info(env: gym.Env) -> dict:
    obs_space = env.observation_space
    action_space = env.action_space

    if hasattr(obs_space, "spaces"):
        obs_shape = {key: space.shape for key, space in obs_space.spaces.items()}
    else:
        obs_shape = obs_space.shape

    info = {
        "obs_space_shape": obs_shape,
        "action_space_shape": action_space.shape,
        "action_low": action_space.low.tolist(),
        "action_high": action_space.high.tolist(),
    }

    if hasattr(env, "spec") and env.spec is not None:
        info["max_episode_steps"] = env.spec.max_episode_steps
    else:
        info["max_episode_steps"] = None

    return info
