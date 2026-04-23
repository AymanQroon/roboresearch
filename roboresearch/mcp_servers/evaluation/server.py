from __future__ import annotations

import base64
import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
from mcp.server.fastmcp import FastMCP
from PIL import Image
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

FRAME_CAPTURE_INTERVAL = 5

LOADABLE_ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
    "DDPG": DDPG,
}


@dataclass
class EpisodeResult:
    episode_index: int
    success: bool
    total_reward: float
    episode_length: int
    final_distance_to_goal: float
    frames: list[str] = field(default_factory=list)


@dataclass
class EvaluationRecord:
    evaluation_id: str
    model_path: str
    env_name: str
    timestamp: str
    episodes: list[EpisodeResult] = field(default_factory=list)


mcp = FastMCP("RoboResearch Evaluation Server")
evaluation_store: dict[str, EvaluationRecord] = {}


def _load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    last_exc = None
    for algo_cls in LOADABLE_ALGORITHMS.values():
        try:
            return algo_cls.load(model_path)
        except Exception as e:
            last_exc = e
            continue

    raise ValueError(
        f"Could not load model from {model_path} with any supported algorithm "
        f"({', '.join(LOADABLE_ALGORITHMS)}). Last error: {last_exc}"
    )


def _frame_to_base64_png(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_env(env_name: str) -> gym.Env:
    try:
        return gym.make(env_name, render_mode="rgb_array")
    except gym.error.NameNotFound:
        raise ValueError(f"Unknown environment: {env_name}")


def _compute_final_distance(obs) -> float:
    if isinstance(obs, dict):
        achieved = np.asarray(obs.get("achieved_goal", []))
        desired = np.asarray(obs.get("desired_goal", []))
        if achieved.size > 0 and desired.size > 0:
            return float(np.linalg.norm(achieved - desired))
    return -1.0


def _run_single_episode(
    model, env: gym.Env, episode_index: int, capture_frames: bool = True
) -> EpisodeResult:
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    frame_buffer: list[str] = []

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frame_buffer.append(_frame_to_base64_png(frame))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        done = terminated or truncated

        if capture_frames and step_count % FRAME_CAPTURE_INTERVAL == 0:
            frame = env.render()
            if frame is not None:
                frame_buffer.append(_frame_to_base64_png(frame))

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frame_buffer.append(_frame_to_base64_png(frame))

    success = bool(info.get("is_success", False))

    return EpisodeResult(
        episode_index=episode_index,
        success=success,
        total_reward=total_reward,
        episode_length=step_count,
        final_distance_to_goal=_compute_final_distance(obs),
        frames=frame_buffer if not success else [],
    )


@mcp.tool()
def run_evaluation(model_path: str, env_name: str, num_episodes: int = 10) -> dict:
    """Load an SB3 model and run evaluation episodes in a Gymnasium environment.

    Returns an evaluation_id and summary statistics.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    model = _load_model(model_path)
    env = _make_env(env_name)

    evaluation_id = f"eval-{uuid.uuid4().hex[:12]}"
    record = EvaluationRecord(
        evaluation_id=evaluation_id,
        model_path=model_path,
        env_name=env_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        for i in range(num_episodes):
            result = _run_single_episode(model, env, episode_index=i)
            record.episodes.append(result)
    finally:
        env.close()

    evaluation_store[evaluation_id] = record

    successes = sum(1 for ep in record.episodes if ep.success)
    rewards = [ep.total_reward for ep in record.episodes]

    return {
        "evaluation_id": evaluation_id,
        "summary": {
            "success_rate": successes / len(record.episodes),
            "mean_reward": float(np.mean(rewards)),
            "num_episodes": len(record.episodes),
        },
    }


def _get_record(evaluation_id: str) -> EvaluationRecord:
    record = evaluation_store.get(evaluation_id)
    if record is None:
        raise ValueError(f"No evaluation found with id: {evaluation_id}")
    if not record.episodes:
        raise ValueError(f"Evaluation {evaluation_id} has no episode results")
    return record


@mcp.tool()
def compute_metrics(evaluation_id: str) -> dict:
    """Compute aggregate metrics for a completed evaluation run."""
    record = _get_record(evaluation_id)
    episodes = record.episodes

    rewards = [ep.total_reward for ep in episodes]
    lengths = [ep.episode_length for ep in episodes]
    distances = [ep.final_distance_to_goal for ep in episodes]
    successes = sum(1 for ep in episodes if ep.success)

    return {
        "evaluation_id": evaluation_id,
        "success_rate": successes / len(episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_final_distance": float(np.mean(distances)),
        "num_episodes": len(episodes),
    }


@mcp.tool()
def compare_runs(eval_id_a: str, eval_id_b: str) -> dict:
    """Compare metrics from two evaluation runs side-by-side."""
    metrics_a = compute_metrics(eval_id_a)
    metrics_b = compute_metrics(eval_id_b)

    comparable_keys = [
        "success_rate",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "min_reward",
        "max_reward",
        "mean_final_distance",
    ]

    deltas: dict[str, dict] = {}
    for key in comparable_keys:
        val_a = metrics_a[key]
        val_b = metrics_b[key]
        abs_delta = val_b - val_a
        pct_delta = (abs_delta / abs(val_a) * 100) if val_a != 0 else float("inf")
        deltas[key] = {
            "absolute": abs_delta,
            "percentage": pct_delta,
        }

    better_a = metrics_a["success_rate"] > metrics_b["success_rate"]
    if metrics_a["success_rate"] == metrics_b["success_rate"]:
        better_a = metrics_a["mean_reward"] > metrics_b["mean_reward"]

    return {
        "eval_a": metrics_a,
        "eval_b": metrics_b,
        "deltas": deltas,
        "which_is_better": eval_id_a if better_a else eval_id_b,
    }


@mcp.tool()
def get_failure_episodes(evaluation_id: str) -> dict:
    """Return detailed info and frame captures for failed episodes."""
    record = _get_record(evaluation_id)

    failures = []
    for ep in record.episodes:
        if ep.success:
            continue
        failures.append({
            "episode_index": ep.episode_index,
            "frames": ep.frames,
            "reward": ep.total_reward,
            "episode_length": ep.episode_length,
            "final_distance": ep.final_distance_to_goal,
        })

    return {
        "evaluation_id": evaluation_id,
        "num_failures": len(failures),
        "total_episodes": len(record.episodes),
        "failure_rate": len(failures) / len(record.episodes),
        "episodes": failures,
    }


@mcp.tool()
def generate_report(evaluation_id: str) -> dict:
    """Generate a full markdown evaluation report with metrics and failure analysis."""
    record = _get_record(evaluation_id)
    metrics = compute_metrics(evaluation_id)
    failures = get_failure_episodes(evaluation_id)

    previous_eval = _find_previous_eval(record)
    comparison_section = ""
    if previous_eval:
        comp = compare_runs(previous_eval.evaluation_id, evaluation_id)
        deltas = comp["deltas"]
        comparison_section = f"""
## Comparison with Previous Evaluation (`{previous_eval.evaluation_id}`)

| Metric | Previous | Current | Delta |
|--------|----------|---------|-------|
| Success Rate | {comp['eval_a']['success_rate']:.2%} | {comp['eval_b']['success_rate']:.2%} | {deltas['success_rate']['absolute']:+.2%} |
| Mean Reward | {comp['eval_a']['mean_reward']:.3f} | {comp['eval_b']['mean_reward']:.3f} | {deltas['mean_reward']['absolute']:+.3f} |
| Mean Distance | {comp['eval_a']['mean_final_distance']:.4f} | {comp['eval_b']['mean_final_distance']:.4f} | {deltas['mean_final_distance']['absolute']:+.4f} |
| Mean Length | {comp['eval_a']['mean_episode_length']:.1f} | {comp['eval_b']['mean_episode_length']:.1f} | {deltas['mean_episode_length']['absolute']:+.1f} |

**Better run:** `{comp['which_is_better']}`
"""

    failure_details = ""
    if failures["num_failures"] > 0:
        rows = []
        for ep in failures["episodes"]:
            rows.append(
                f"| {ep['episode_index']} | {ep['reward']:.3f} | "
                f"{ep['episode_length']} | {ep['final_distance']:.4f} | "
                f"{len(ep['frames'])} |"
            )
        failure_table = "\n".join(rows)
        failure_details = f"""
## Failure Analysis

{failures['num_failures']} of {failures['total_episodes']} episodes failed ({failures['failure_rate']:.2%} failure rate).

| Episode | Reward | Length | Final Distance | Frames |
|---------|--------|--------|----------------|--------|
{failure_table}
"""
    else:
        failure_details = "\n## Failure Analysis\n\nAll episodes succeeded.\n"

    report = f"""# Evaluation Report

- **Evaluation ID:** `{evaluation_id}`
- **Model:** `{record.model_path}`
- **Environment:** `{record.env_name}`
- **Timestamp:** {record.timestamp}
- **Episodes:** {metrics['num_episodes']}

## Metrics Summary

| Metric | Value |
|--------|-------|
| Success Rate | {metrics['success_rate']:.2%} |
| Mean Reward | {metrics['mean_reward']:.3f} |
| Std Reward | {metrics['std_reward']:.3f} |
| Min Reward | {metrics['min_reward']:.3f} |
| Max Reward | {metrics['max_reward']:.3f} |
| Mean Episode Length | {metrics['mean_episode_length']:.1f} |
| Mean Final Distance | {metrics['mean_final_distance']:.4f} |
{comparison_section}{failure_details}"""

    return {
        "evaluation_id": evaluation_id,
        "report_markdown": report.strip(),
    }


def _find_previous_eval(current: EvaluationRecord) -> EvaluationRecord | None:
    """Find the most recent prior evaluation for the same environment."""
    candidates = [
        rec
        for eid, rec in evaluation_store.items()
        if eid != current.evaluation_id and rec.env_name == current.env_name
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda r: r.timestamp)
    return candidates[-1]


if __name__ == "__main__":
    mcp.run()
