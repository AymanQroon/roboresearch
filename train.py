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
import torch
from sb3_contrib import TQC
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

gym.register_envs(gymnasium_robotics)


# Device selection. Apple MPS was empirically catastrophic on this workload
# (single-env MuJoCo + small MLP): step rate degrades from ~35/s to <1/s after
# ~5k env steps, likely allocator thrash. CPU is preferred. We still pick CUDA
# when available. Use ~perf cores on M-series for the BLAS path.
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _pick_device()
if DEVICE == "cpu":
    # M4 has 4 performance cores; SB3 uses single-thread inference but BLAS
    # in the gradient path benefits from a few threads.
    torch.set_num_threads(min(4, torch.get_num_threads()))

# === CONFIGURATION ===
ALGORITHM = "TQC"
ENV_NAME = "FrankaKitchen-v1"
SUBTASK = "slide cabinet"
MAX_EPISODE_STEPS = 280
TIME_BUDGET = 3600  # 1h default; FrankaKitchen episodes are ~6x longer than Fetch tasks

HYPERPARAMS = {
    "learning_rate": 1e-3,
    "batch_size": 512,
    # 200k transitions is ~3x the longest single-task run; keeping the buffer
    # small avoids macOS memory pressure after several back-to-back runs in
    # one session. Original 1M caused step-rate collapse from compressor/swap.
    "buffer_size": 200_000,
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

# Built into the env: success when ||achieved_goal - desired_goal|| < BONUS_THRESH
BONUS_THRESH = 0.3

# Per-subtask MuJoCo site name for the geometric "where to put the gripper"
# target. The end-effector → target Euclidean distance is what gives the
# policy a smooth gradient toward contact during the long random-exploration
# phase. The joint-space `achieved_goal` distance alone has no entry path
# because the arm essentially never reaches the target by chance in 280
# steps. Sites verified to exist on the loaded MuJoCo model.
SUBTASK_TARGET_SITES = {
    "microwave":      "microhandle_site",
    "slide cabinet":  "slide_site",
    "hinge cabinet":  "hinge_site2",   # right door — desired_goal moves index 1, the right hinge
    "light switch":   "light_site",
    "kettle":         "kettle_site",
    "bottom burner":  "knob2_site",
    "top burner":     "knob4_site",
}


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
        # Everything is float32 — Apple MPS doesn't support float64.
        obs_shape = sample_obs["observation"].shape
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-np.inf, np.inf, obs_shape, np.float32),
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, goal_shape, np.float32),
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, goal_shape, np.float32),
            }
        )

    def observation(self, obs):
        return {
            "observation": np.asarray(obs["observation"], dtype=np.float32),
            "achieved_goal": np.asarray(obs["achieved_goal"][self.subtask], dtype=np.float32),
            "desired_goal": np.asarray(obs["desired_goal"][self.subtask], dtype=np.float32),
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Sparse success signal expected by HER; matches kitchen's BONUS_THRESH semantics.
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return (dist < BONUS_THRESH).astype(np.float32)


# === REWARD WRAPPER ===
# The reward template that solved microwave in iter1, generalized over any
# FrankaKitchen subtask. Two dense components plus a sparse bonus:
#   (1) goal_distance: ||achieved_goal - desired_goal|| in joint space.
#       Works for 1-D (microwave/slide/burner-component) and N-D
#       (hinge cabinet 2-D, kettle 7-D pose) goals identically.
#   (2) ee_to_target:  Euclidean distance from the Franka end-effector
#                      site to the subtask's geometric target site
#                      (table at SUBTASK_TARGET_SITES). This addresses the
#                      primary failure mode in long-horizon kitchen RL —
#                      the arm essentially never reaches the target by
#                      chance in 280 steps, so the goal_distance term
#                      alone has no entry path.
#   (3) success_bonus: large positive reward inside the env's BONUS_THRESH.
class RewardWrapper(gym.Wrapper):
    """Subtask-agnostic shaped reward (joint-distance + EE-proximity + bonus)."""

    def __init__(self, env):
        super().__init__(env)
        unwrapped = env.unwrapped
        target_site_name = SUBTASK_TARGET_SITES[SUBTASK]
        self._ee_site_id = unwrapped.model.site("end_effector").id
        self._target_site_id = unwrapped.model.site(target_site_name).id
        self._sim_data = unwrapped.data

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        goal_distance = float(np.linalg.norm(
            obs["achieved_goal"] - obs["desired_goal"]
        ))

        ee_xyz = self._sim_data.site_xpos[self._ee_site_id]
        target_xyz = self._sim_data.site_xpos[self._target_site_id]
        ee_to_target = float(np.linalg.norm(ee_xyz - target_xyz))

        success_bonus = 1.0 if goal_distance < BONUS_THRESH else 0.0

        shaped_reward = (
            -1.0 * goal_distance        # drive joint(s) to goal
            - 1.0 * ee_to_target        # bring gripper to target site
            + 5.0 * success_bonus       # large bonus on success
        )

        info["reward_components"] = {
            "goal_distance": goal_distance,
            "ee_to_target": ee_to_target,
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

    Early-stops as soon as `early_stop_consecutive` evals in a row return
    SR=1.00, saving the model to `model_checkpoint.zip` before requesting stop
    so we don't lose a perfect policy when the runner exits.
    """

    def __init__(
        self,
        eval_every_steps: int = 5_000,
        num_eval_episodes: int = 3,
        heartbeat_steps: int = 1_000,
        early_stop_consecutive: int = 2,
    ):
        super().__init__()
        self._eval_every = eval_every_steps
        self._num_eval = num_eval_episodes
        self._heartbeat = heartbeat_steps
        self._next_eval = eval_every_steps
        self._next_heartbeat = heartbeat_steps
        self._best_sr = 0.0
        self._start_time = None
        self._early_stop_consecutive = early_stop_consecutive
        self._consecutive_perfect = 0

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
        self._write(
            f"[start] {time.strftime('%H:%M:%S')} algo={ALGORITHM} "
            f"subtask={SUBTASK!r} device={DEVICE} time_budget={TIME_BUDGET}s"
        )

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

            if sr >= 1.0:
                self._consecutive_perfect += 1
            else:
                self._consecutive_perfect = 0

            if self._consecutive_perfect >= self._early_stop_consecutive:
                self.model.save("model_checkpoint")
                self._write(
                    f"[stop] step={self.num_timesteps} elapsed={elapsed}s "
                    f"early-stop after {self._consecutive_perfect} perfect evals; "
                    f"saved model_checkpoint.zip"
                )
                return False
        return True


# === TRAINING ===
def train():
    env = make_env()

    algo_cls = ALGOS[ALGORITHM]

    kwargs = dict(HYPERPARAMS)
    if ALGORITHM in OFF_POLICY:
        kwargs["replay_buffer_class"] = HerReplayBuffer
        kwargs["replay_buffer_kwargs"] = dict(HER_KWARGS)

    model = algo_cls("MultiInputPolicy", env, verbose=0, device=DEVICE, **kwargs)
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
