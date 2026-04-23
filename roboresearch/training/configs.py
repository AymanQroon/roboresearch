from __future__ import annotations

SUPPORTED_ALGORITHMS = ["SAC", "PPO", "TD3"]

_COMMON_POLICY_KWARGS = {"net_arch": [256, 256]}

_DEFAULTS = {
    "SAC": {
        "policy": "MultiInputPolicy",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 100_000,
        "tau": 0.005,
        "gamma": 0.99,
        "learning_starts": 1000,
        "policy_kwargs": dict(_COMMON_POLICY_KWARGS),
    },
    "PPO": {
        "policy": "MultiInputPolicy",
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": dict(_COMMON_POLICY_KWARGS),
    },
    "TD3": {
        "policy": "MultiInputPolicy",
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 100_000,
        "tau": 0.005,
        "gamma": 0.99,
        "learning_starts": 1000,
        "policy_kwargs": dict(_COMMON_POLICY_KWARGS),
    },
}


def get_default_config(algorithm: str, env_name: str) -> dict:
    algo = algorithm.upper()
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm: {algorithm!r}. "
            f"Supported: {SUPPORTED_ALGORITHMS}"
        )

    import copy

    config = copy.deepcopy(_DEFAULTS[algo])
    config["algorithm"] = algo
    config["env_name"] = env_name
    return config


def merge_config(defaults: dict, overrides: dict) -> dict:
    import copy

    valid_keys = set(defaults.keys())
    invalid_keys = set(overrides.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(
            f"Invalid override keys: {invalid_keys}. "
            f"Valid keys: {sorted(valid_keys)}"
        )

    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged
