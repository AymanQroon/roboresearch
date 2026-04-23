"""
Long FetchPush training run to build a solid baseline policy.
Trains SAC+HER for 500K steps, evaluates every 100K, saves checkpoints.

Usage:
    python scripts/long_train.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from roboresearch.training import create_model, train_model, save_model, evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = PROJECT_ROOT / "registry"

TOTAL_STEPS = 500_000
EVAL_INTERVAL = 100_000
ENV_NAME = "FetchPush-v4"
ALGORITHM = "SAC"


def main() -> None:
    print(f"Long training: {ALGORITHM} on {ENV_NAME}, {TOTAL_STEPS:,} total steps")
    print(f"Evaluating every {EVAL_INTERVAL:,} steps\n")

    model = create_model(ALGORITHM, ENV_NAME, {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "tau": 0.005,
        "gamma": 0.95,
        "learning_starts": 1000,
        "policy_kwargs": {"net_arch": [256, 256]},
    })

    steps_trained = 0
    best_sr = 0.0
    start_time = time.time()

    while steps_trained < TOTAL_STEPS:
        chunk = min(EVAL_INTERVAL, TOTAL_STEPS - steps_trained)
        print(f"Training steps {steps_trained:,} -> {steps_trained + chunk:,}...")

        train_model(model, total_timesteps=chunk)
        steps_trained += chunk
        elapsed = time.time() - start_time

        print("  Evaluating (20 episodes)...")
        eval_result = evaluate_model(model, ENV_NAME, num_episodes=20)
        sr = eval_result["summary"]["success_rate"]
        mr = eval_result["summary"]["mean_reward"]

        print(f"  Steps: {steps_trained:,} | SR: {sr:.2f} | Reward: {mr:.1f} | Time: {elapsed:.0f}s")

        if sr > best_sr:
            best_sr = sr
            print("  New best! Saving checkpoint...")

        checkpoint_name = f"long_train_{steps_trained // 1000}k"
        save_path = str(PROJECT_ROOT / "registry" / "models" / checkpoint_name / f"roboresearch_{checkpoint_name}")
        save_model(model, save_path)

        if sr >= 1.0:
            print("  Perfect success rate — stopping early.", flush=True)
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time / 60:.1f} minutes")
    print(f"Best success rate: {best_sr:.2f}")

    print("\nFinal evaluation (50 episodes)...")
    final_eval = evaluate_model(model, ENV_NAME, num_episodes=50)
    final_sr = final_eval["summary"]["success_rate"]
    final_mr = final_eval["summary"]["mean_reward"]
    print(f"Final SR: {final_sr:.2f} | Final Reward: {final_mr:.1f}")

    # Save final model to registry
    run_id = "run_long_train"
    metadata_dir = REGISTRY / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    final_save = str(REGISTRY / "models" / run_id / f"roboresearch_{run_id}")
    save_model(model, final_save)

    metadata = {
        "run_id": run_id,
        "model_path": final_save,
        "config": {
            "algorithm": ALGORITHM,
            "env_name": ENV_NAME,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "buffer_size": 1_000_000,
            "tau": 0.005,
            "gamma": 0.95,
            "learning_starts": 1000,
            "policy_kwargs": {"net_arch": [256, 256]},
            "total_steps": TOTAL_STEPS,
        },
        "metrics": {
            "success_rate": final_sr,
            "mean_reward": final_mr,
        },
        "notes": f"Long training run: {TOTAL_STEPS:,} steps, final SR={final_sr:.2f}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (metadata_dir / f"{run_id}.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved to registry as {run_id}")


if __name__ == "__main__":
    main()
