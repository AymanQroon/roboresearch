# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

RoboResearch is a closed-loop autonomous reward-discovery system for robot manipulation. Claude Code itself is the agent: it edits `train.py` to write a new reward wrapper, runs training, watches failure-episode frames as a VLM, diagnoses *why* the policy failed spatially, and redesigns the reward. The full agent contract lives in `program.md` — read it before doing anything that resembles "running the loop".

Default task: **FrankaKitchen-v1, microwave subtask** (open the microwave door by driving the handle joint to `-0.75`). The default env reward is sparse, so reward shaping is the dominant lever.

## Hard editing rules

- **Only `train.py` is editable.** `prepare.py` and everything under `mcp/` are read-only — they are imported by the MCP servers and by evaluation, and their public surface is a contract.
- The env factory signature `make_env(render_mode: str | None = None, with_reward_wrapper: bool = True)` and the helper `success_from_info(info)` in `train.py` are imported by `prepare.py` and `mcp/*.py`. Do not rename them or change their signatures — add wrappers *inside* `make_env`, not around it.
- `FlattenSingleSubtaskGoal` assumes a single active subtask. Chaining subtasks requires updating that wrapper.
- Do not change the evaluation protocol (20 episodes, deterministic policy, success from `info["episode_task_completions"]`).
- Each new `RewardWrapper` must populate `info["reward_components"]` with per-component scalars — `prepare.evaluate_with_reward_components` and downstream analysis depend on this.
- `_archive_v1_DO_NOT_TOUCH/` is dead code from a prior iteration. Do not read it, reference it, or grep into it.

## Commands

```bash
# Setup (one-time)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Single training run (writes model_checkpoint, prints SUCCESS_RATE / MEAN_REWARD)
python train.py

# Lint / typecheck
ruff check .
pyright
```

There is no test suite. The success metric is the `SUCCESS_RATE=` line that `train.py` prints to stdout after the eval pass — that is the ground truth signal the loop reacts to.

## Architecture

Three layers, top-down:

1. **`train.py`** — the search variable. Holds `RewardWrapper` (rewritten each loop iteration), the `make_env` factory, hyperparameters (`HYPERPARAMS`, `HER_KWARGS`), the `ALGORITHM` selector, and a `TimeBudgetCallback` that caps wall-clock training at `TIME_BUDGET` seconds (default 3600). Off-policy algorithms (SAC/TD3/TQC/CrossQ/DDPG) get `HerReplayBuffer` automatically.
2. **`prepare.py`** — read-only evaluation infrastructure. Hot-reloads `train.py` (`importlib.reload`) on every call so the MCP servers always see the agent's latest edits. Provides `evaluate_model`, `evaluate_with_reward_components` (per-component reward breakdowns), `capture_failure_frames`, and the `results.tsv` log.
3. **`mcp/` servers** — three FastMCP servers wired up in `.mcp.json`:
   - `simulation.py`: `configure_env`, `run_training`, `capture_frames`, `get_training_log`, `reset_env`. Trains policies to a temp dir, returns base64 PNG frames for VLM analysis.
   - `evaluation.py`: `run_evaluation`, `compute_metrics`, `compare_runs`, `get_failure_episodes`, `generate_report`. Stores eval records in-memory keyed by `eval_id`; failure-episode frames are only retained for failed episodes.
   - `registry.py`: `save_checkpoint`, `load_checkpoint`, `list_experiments`, `get_best_model`, `diff_configs`. Persists models to `registry/models/<run_id>/` and JSON metadata to `registry/metadata/`.

Both `simulation.py` and `evaluation.py` go through `_train_module()` which reloads `train.py` — so when you edit `train.py` mid-session, the next MCP call sees your changes without restarting the server.

## Observation space (FrankaKitchen-v1)

After `FlattenSingleSubtaskGoal`, observations are `{"observation": (59,), "achieved_goal": (1,), "desired_goal": (1,)}`. Useful indices into `obs["observation"]` (full table in `program.md`):

- `[0:9]` — Franka 9-DOF joint positions
- `[22]` — microwave handle joint (the active subtask; `MICROWAVE_OBS_IDX` in `train.py`)
- `[23:30]` — kettle pose (xyz + quat)

Constants in `train.py`: `MICROWAVE_GOAL=-0.75`, `BONUS_THRESH=0.3`. Success = `subtask in info["episode_task_completions"]` (NOT `info["is_success"]`, which kitchen does not set).

## The loop

When asked to "run the loop" or similar, the canonical flow is:

1. Edit `RewardWrapper` in `train.py` based on the latest failure diagnosis.
2. Run `python train.py`, parse `SUCCESS_RATE=` from stdout.
3. Use MCP tools (`capture_frames`, `get_failure_episodes`) to render failure rollouts and diagnose visually — this is the core mechanism that distinguishes this system from hyperparameter search; it is not optional.
4. If the new reward improved success rate: `git add train.py && git commit`. If worse: `git checkout train.py`. Git history is the experiment log.
5. Repeat. Per `program.md`, do not pause to ask the human once the loop is running.
