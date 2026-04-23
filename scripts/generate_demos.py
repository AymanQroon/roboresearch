"""
Generate demo GIFs from trained models for the README.

Usage:
    python scripts/generate_demos.py
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
from PIL import Image
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
REGISTRY_DIR = PROJECT_ROOT / "registry"


def load_best_model(env_name: str) -> tuple[SAC, str] | None:
    metadata_dir = REGISTRY_DIR / "metadata"
    if not metadata_dir.exists():
        return None

    import json

    best_sr = -1.0
    best_path = None
    best_run_id = None

    for f in sorted(metadata_dir.glob("*.json")):
        meta = json.loads(f.read_text())
        cfg = meta.get("config", {})
        if cfg.get("env_name") != env_name:
            continue
        sr = float(meta.get("metrics", {}).get("success_rate", 0))
        model_path = meta.get("model_path", "")
        full_path = PROJECT_ROOT / model_path
        if not (full_path.exists() or Path(f"{full_path}.zip").exists()):
            continue
        if sr > best_sr:
            best_sr = sr
            best_path = str(full_path)
            best_run_id = meta["run_id"]

    if best_path is None:
        return None

    env = gym.make(env_name, render_mode="rgb_array")
    model = SAC.load(best_path, env=env)
    env.close()
    return model, best_run_id


def capture_episode_frames(model: SAC, env_name: str, max_steps: int = 50) -> list[np.ndarray]:
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()
    return frames


def frames_to_gif(frames: list[np.ndarray], output_path: Path, duration: int = 80) -> None:
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )


def capture_untrained_frames(env_name: str, max_steps: int = 50) -> list[np.ndarray]:
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()
    return frames


def generate_learning_curve() -> None:
    try:
        import json

        import plotly.graph_objects as go

        metadata_dir = REGISTRY_DIR / "metadata"
        if not metadata_dir.exists():
            print("  No metadata found, skipping learning curve")
            return

        runs = []
        for f in sorted(metadata_dir.glob("*.json")):
            meta = json.loads(f.read_text())
            runs.append({
                "run_id": meta["run_id"],
                "env": meta.get("config", {}).get("env_name", ""),
                "algo": meta.get("config", {}).get("algorithm", ""),
                "sr": float(meta.get("metrics", {}).get("success_rate", 0)),
            })

        if not runs:
            print("  No runs found")
            return

        fig = go.Figure()

        reach_runs = [r for r in runs if "Reach" in r["env"]]
        push_runs = [r for r in runs if "Push" in r["env"]]

        algo_colors = {"SAC": "#3498db", "TD3": "#e67e22", "PPO": "#e74c3c"}

        if reach_runs:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(reach_runs) + 1)),
                y=[r["sr"] for r in reach_runs],
                mode="lines+markers",
                name="FetchReach (SAC)",
                line={"color": "#2ecc71", "width": 3},
                marker={"size": 8},
            ))

        if push_runs:
            offset = len(reach_runs)
            fig.add_trace(go.Scatter(
                x=list(range(offset + 1, offset + len(push_runs) + 1)),
                y=[r["sr"] for r in push_runs],
                mode="lines+markers",
                name="FetchPush",
                line={"color": "#888", "width": 2, "dash": "dot"},
                marker={
                    "size": 10,
                    "color": [algo_colors.get(r["algo"], "#888") for r in push_runs],
                    "line": {"width": 1, "color": "#fff"},
                },
                text=[r["algo"] for r in push_runs],
                hovertemplate="%{text}<br>SR: %{y:.2f}<extra></extra>",
            ))

            for algo, color in algo_colors.items():
                algo_runs = [r for r in push_runs if r["algo"] == algo]
                if algo_runs:
                    fig.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode="markers",
                        name=f"FetchPush ({algo})",
                        marker={"size": 10, "color": color},
                        showlegend=True,
                    ))

        graduation_x = len(reach_runs) + 0.5
        if reach_runs and push_runs:
            fig.add_vline(
                x=graduation_x, line_dash="dash", line_color="#f39c12",
                opacity=0.7, annotation_text="Task Graduation",
                annotation_position="top", annotation_font_size=11,
                annotation_font_color="#f39c12",
            )

        fig.update_layout(
            title="RoboResearch — Autonomous Learning Curve",
            xaxis_title="Experiment #",
            yaxis_title="Success Rate",
            yaxis_range=[0, 1.05],
            template="plotly_dark",
            font={"size": 14},
            width=1000,
            height=500,
            legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(0,0,0,0.5)"},
        )

        fig.write_image(str(ASSETS_DIR / "learning_curve.png"), scale=2)
        print(f"  Learning curve saved to {ASSETS_DIR / 'learning_curve.png'}")

    except ImportError:
        print("  plotly or kaleido not available, skipping learning curve")
    except Exception as e:
        print(f"  Learning curve generation failed: {e}")


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating demo assets...\n")

    print("[1/4] FetchReach — random policy (before training)")
    random_frames = capture_untrained_frames("FetchReach-v4")
    frames_to_gif(random_frames, ASSETS_DIR / "reach_random.gif")
    print(f"  Saved {len(random_frames)} frames → assets/reach_random.gif")

    print("\n[2/4] FetchReach — trained policy")
    result = load_best_model("FetchReach-v4")
    if result:
        model, run_id = result
        print(f"  Loaded best model: {run_id}")
        trained_frames = capture_episode_frames(model, "FetchReach-v4")
        frames_to_gif(trained_frames, ASSETS_DIR / "reach_solved.gif")
        print(f"  Saved {len(trained_frames)} frames → assets/reach_solved.gif")
    else:
        print("  No trained FetchReach model found, skipping")

    print("\n[3/4] FetchPush — trained policy")
    result = load_best_model("FetchPush-v4")
    if result:
        model, run_id = result
        print(f"  Loaded best model: {run_id}")
        push_frames = capture_episode_frames(model, "FetchPush-v4")
        frames_to_gif(push_frames, ASSETS_DIR / "push_progress.gif")
        print(f"  Saved {len(push_frames)} frames → assets/push_progress.gif")
    else:
        print("  No trained FetchPush model found, skipping")

    print("\n[4/4] Learning curve chart")
    generate_learning_curve()

    print("\nDone! Assets saved to assets/")


if __name__ == "__main__":
    main()
