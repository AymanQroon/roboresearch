"""
Robot manipulation training — reward discovery edition.

The AI agent modifies this file to improve performance. The primary search
variable is the RewardWrapper class: the agent designs custom reward functions
informed by VLM failure analysis, then trains and evaluates.

Current task: FrankaKitchen-v1 (microwave subtask) — open the microwave door
by manipulating its handle joint to a target position. The default reward is
sparse (+1 only when the task completes, 0 otherwise), so reward shaping is
the dominant lever for learning.
"""

import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

gym.register_envs(gymnasium_robotics)

# === CONFIGURATION ===
ALGORITHM = "TQC"
ENV_NAME = "FrankaKitchen-v1"
SUBTASK = "microwave"
MAX_EPISODE_STEPS = 280
TIME_BUDGET = 3600  # 1h default; FrankaKitchen episodes are ~6x longer than Fetch tasks

HYPERPARAMS = {
    "learning_rate": 1e-3,
    "batch_size": 512,
    "buffer_size": 1_000_000,
    "tau": 0.005,
    "gamma": 0.95,
    "learning_starts": 1000,
    "policy_kwargs": {"net_arch": [512, 512, 512]},
}

HER_KWARGS = {
    "goal_selection_strategy": "future",
    "n_sampled_goal": 4,
}

# === ALGORITHM MAP ===
ALGOS = {"SAC": SAC, "TD3": TD3, "TQC": TQC}
OFF_POLICY = {"SAC", "TD3", "TQC"}

# Microwave handle joint goal value (qpos index 22 in raw env, exposed as
# obs["achieved_goal"][0] after FlattenSingleSubtaskGoal — DO NOT use
# obs["observation"][22], that index is the top right burner knob, not the
# microwave). Success: |handle - GOAL| < BONUS_THRESH.
MICROWAVE_GOAL = -0.75
BONUS_THRESH = 0.3


# === GOAL FLATTEN WRAPPER ===
# FrankaKitchen returns achieved/desired goals as dict-of-dicts (one entry per
# active subtask). SB3's HerReplayBuffer expects flat arrays, so we extract the
# single subtask we're training on.
class FlattenSingleSubtaskGoal(gym.ObservationWrapper):
    def __init__(self, env, subtask):
        super().__init__(env)
        self.subtask = subtask
        sample_obs, _ = env.reset()
        goal_shape = sample_obs["achieved_goal"][subtask].shape
        self.observation_space = gym.spaces.Dict(
            {
                "observation": env.observation_space["observation"],
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, goal_shape, np.float64),
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, goal_shape, np.float64),
            }
        )

    def observation(self, obs):
        return {
            "observation": obs["observation"],
            "achieved_goal": np.asarray(obs["achieved_goal"][self.subtask], dtype=np.float64),
            "desired_goal": np.asarray(obs["desired_goal"][self.subtask], dtype=np.float64),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Sparse success signal expected by HER; matches kitchen's BONUS_THRESH semantics.
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (dist < BONUS_THRESH).astype(np.float32)


# === REWARD WRAPPER ===
# Iteration 1 (2026-04-22): Fix obs-index bug + add end-effector proximity.
#
# Why the previous version was broken: MICROWAVE_OBS_IDX=22 was a qpos index,
# but the code used it as an obs-vector index. obs[22] is the top right burner
# knob; the microwave joint is actually at obs[31] (or, cleanly,
# obs["achieved_goal"][0] after FlattenSingleSubtaskGoal). HER still got
# correct sparse signal via achieved_goal, but the dense distance term was
# regressing against the wrong joint, providing zero useful gradient toward
# opening the microwave.
#
# Two dense components, both pulled directly from MuJoCo state:
#   (1) handle_distance: |microwave joint - GOAL|, from achieved_goal
#   (2) ee_to_handle:    Euclidean distance from the Franka end-effector
#                        site to the microwave handle site (real xyz, no
#                        forward kinematics needed). This addresses the
#                        primary failure mode in long-horizon kitchen RL —
#                        the arm never gets near the handle in 280 steps
#                        of random exploration, so there is no signal for
#                        the joint-distance term to ever become useful.
class RewardWrapper(gym.Wrapper):
    """Custom reward function discovered through autonomous experimentation."""

    def __init__(self, env):
        super().__init__(env)
        unwrapped = env.unwrapped
        self._ee_site_id = unwrapped.model.site("end_effector").id
        self._handle_site_id = unwrapped.model.site("microhandle_site").id
        self._sim_data = unwrapped.data

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        handle_pos = float(obs["achieved_goal"][0])
        handle_distance = abs(handle_pos - MICROWAVE_GOAL)

        ee_xyz = self._sim_data.site_xpos[self._ee_site_id]
        handle_xyz = self._sim_data.site_xpos[self._handle_site_id]
        ee_to_handle = float(np.linalg.norm(ee_xyz - handle_xyz))

        success_bonus = 1.0 if handle_distance < BONUS_THRESH else 0.0

        shaped_reward = (
            -1.0 * handle_distance      # drive joint to goal
            - 1.0 * ee_to_handle        # bring gripper to handle
            + 5.0 * success_bonus       # large bonus on success
        )

        info["reward_components"] = {
            "handle_distance": handle_distance,
            "ee_to_handle": ee_to_handle,
            "success_bonus": success_bonus,
            "shaped_total": float(shaped_reward),
            "sparse_success": float(bool(info.get("step_task_completions"))),
        }

        return obs, shaped_reward, terminated, truncated, info


# === ENV FACTORY ===
# Exported for MCP tools and prepare.py — DO NOT change the signature without
# updating those callers. The agent may add wrappers inside this function but
# must keep `make_env` callable as `make_env()` and `make_env(render_mode=...)`.
def make_env(render_mode: str | None = None, with_reward_wrapper: bool = True):
    kwargs = {"tasks_to_complete": [SUBTASK], "max_episode_steps": MAX_EPISODE_STEPS}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(ENV_NAME, **kwargs)
    env = FlattenSingleSubtaskGoal(env, SUBTASK)
    if with_reward_wrapper:
        env = RewardWrapper(env)
    return env


# === SUCCESS DETECTION ===
# Exported helper used by evaluation/MCP tools. FrankaKitchen reports completed
# subtasks via `info["episode_task_completions"]`, not `info["is_success"]`.
def success_from_info(info: dict) -> bool:
    completions = info.get("episode_task_completions") or []
    return SUBTASK in completions


# === TIMER + PROGRESS LOG ===
class TimeBudgetCallback(BaseCallback):
    def __init__(self, budget):
        super().__init__()
        self._budget = budget
        self._start = None

    def _on_training_start(self):
        self._start = time.time()

    def _on_step(self):
        return (time.time() - (self._start or time.time())) < self._budget


PROGRESS_LOG = "progress.log"


class ProgressEvalCallback(BaseCallback):
    """Periodically run a deterministic eval and append to PROGRESS_LOG.

    Writes a line per evaluation (with explicit fsync) so progress is visible
    in real time even when stdout is buffered by `tee` or a parent harness.
    Also emits a heartbeat every `heartbeat_steps` so we can confirm the loop
    is alive before the first eval.
    """

    def __init__(
        self,
        eval_every_steps: int = 5_000,
        num_eval_episodes: int = 3,
        heartbeat_steps: int = 1_000,
    ):
        super().__init__()
        self._eval_every = eval_every_steps
        self._num_eval = num_eval_episodes
        self._heartbeat = heartbeat_steps
        self._next_eval = eval_every_steps
        self._next_heartbeat = heartbeat_steps
        self._best_sr = 0.0
        self._start_time = None

    def _write(self, line: str) -> None:
        with open(PROGRESS_LOG, "a") as f:
            f.write(line + "\n")
            f.flush()
            try:
                import os
                os.fsync(f.fileno())
            except OSError:
                pass

    def _on_training_start(self):
        self._start_time = time.time()
        self._write(f"[start] {time.strftime('%H:%M:%S')} algo={ALGORITHM}")

    def _on_step(self):
        elapsed = int(time.time() - (self._start_time or time.time()))
        if self.num_timesteps >= self._next_heartbeat:
            self._next_heartbeat += self._heartbeat
            self._write(f"[hb]   step={self.num_timesteps:>7} elapsed={elapsed:>4}s")

        if self.num_timesteps >= self._next_eval:
            self._next_eval += self._eval_every
            sr, mr = evaluate(self.model, num_episodes=self._num_eval)
            tag = " *NEW_BEST*" if sr > self._best_sr else ""
            self._best_sr = max(self._best_sr, sr)
            self._write(
                f"[eval] step={self.num_timesteps:>7} elapsed={elapsed:>4}s "
                f"success_rate={sr:.2f} mean_reward={mr:+.2f}{tag}"
            )
        return True


# === TRAINING ===
def train():
    env = make_env()

    algo_cls = ALGOS[ALGORITHM]

    kwargs = dict(HYPERPARAMS)
    if ALGORITHM in OFF_POLICY:
        kwargs["replay_buffer_class"] = HerReplayBuffer
        kwargs["replay_buffer_kwargs"] = dict(HER_KWARGS)

    model = algo_cls("MultiInputPolicy", env, verbose=0, **kwargs)
    callbacks = [TimeBudgetCallback(TIME_BUDGET), ProgressEvalCallback()]
    model.learn(total_timesteps=10_000_000, callback=callbacks)
    return model


# === EVALUATION (uses unwrapped reward for honest success measurement) ===
def evaluate(model, num_episodes=20):
    env = make_env(with_reward_wrapper=False)
    successes, rewards = 0, []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward, done, completed = 0.0, False, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if success_from_info(info):
                completed = True
            done = terminated or truncated
        if completed:
            successes += 1
        rewards.append(total_reward)

    env.close()
    return successes / num_episodes, float(np.mean(rewards))


if __name__ == "__main__":
    start = time.time()
    model = train()
    sr, mr = evaluate(model)
    elapsed = time.time() - start

    print(f"SUCCESS_RATE={sr:.4f}")
    print(f"MEAN_REWARD={mr:.2f}")
    print(f"ALGORITHM={ALGORITHM}")
    print(f"ELAPSED={elapsed:.0f}s")

    model.save("model_checkpoint")
