"""Render before/after still images for every solved subtask.

For each subtask we have a saved checkpoint for, this script:
  1. Sets train.SUBTASK to the matching task and rebuilds the env in
     rgb_array render mode (no reward wrapper; we just want frames).
  2. Captures the final frame of an untrained-policy rollout (random
     actions in the action space).
  3. Loads the trained TQC model and captures the final frame of a
     deterministic rollout.
  4. Stacks the two frames horizontally with labels and writes
     `assets/<subtask>_before_after.png`.

Run from the project root with the venv active:
    python scripts/render_demos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import TQC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402  (path-dependent)

CHECKPOINTS = [
    ("microwave", "iter1_microwave_TQC.zip"),
    ("slide cabinet", "iter2_slide_cabinet_TQC.zip"),
    ("hinge cabinet", "iter3_hinge_cabinet_TQC.zip"),
    ("light switch", "iter4_light_switch_TQC.zip"),
    ("bottom burner", "iter5_bottom_burner_TQC.zip"),
    ("top burner", "iter6_top_burner_TQC.zip"),
    ("kettle", "iter7_kettle_TQC.zip"),
]

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


def rollout_final_frame(env, policy_fn, steps: int = 280) -> np.ndarray:
    obs, _ = env.reset(seed=0)
    last_frame = env.render()
    for _ in range(steps):
        action = policy_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        last_frame = env.render()
        if terminated or truncated:
            break
    return last_frame


def label(frame: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0, 200))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


def render_subtask(subtask: str, ckpt: str | None) -> Path | None:
    train.SUBTASK = subtask
    env = train.make_env(render_mode="rgb_array", with_reward_wrapper=False)

    # Untrained policy: uniform random actions
    rng = np.random.default_rng(0)
    untrained_frame = rollout_final_frame(env, lambda _o: env.action_space.sample())
    env.close()

    if ckpt is None:
        # Just save the untrained frame
        out = ASSETS_DIR / f"{subtask.replace(' ', '_')}_untrained_only.png"
        label(untrained_frame, f"{subtask} — untrained (no checkpoint)").save(out)
        print(f"  wrote {out.name} (untrained only)")
        return out

    env = train.make_env(render_mode="rgb_array", with_reward_wrapper=False)
    model = TQC.load(str(PROJECT_ROOT / ckpt), env=env, device="cpu")
    trained_frame = rollout_final_frame(
        env, lambda o: model.predict(o, deterministic=True)[0]
    )
    env.close()

    left = label(untrained_frame, f"{subtask} — untrained")
    right = label(trained_frame, f"{subtask} — trained (TQC + HER)")

    h = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width + 8, h), color=(20, 20, 20))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 8, 0))
    out = ASSETS_DIR / f"{subtask.replace(' ', '_')}_before_after.png"
    canvas.save(out, optimize=True)
    print(f"  wrote {out.name}")
    return out


def main():
    for subtask, ckpt in CHECKPOINTS:
        print(f"[{subtask}]")
        render_subtask(subtask, ckpt)


if __name__ == "__main__":
    main()
