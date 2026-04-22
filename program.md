# RoboResearch Program

## Goal
Autonomously discover reward functions that maximize success_rate on
FrankaKitchen-v1 (microwave subtask) — a 9-DOF Franka arm must manipulate
the microwave handle joint to a target position to open the door.

The default reward in FrankaKitchen is **sparse** (+1 only when the task
completes within `BONUS_THRESH=0.3` of the goal joint value, 0 otherwise).
Standard online RL barely learns from this signal — most published results
rely on offline demonstrations (D4RL). Your job is to discover dense reward
functions that make this task learnable from scratch, informed by visual
failure analysis.

## Architecture

You are the agent. You operate in a closed loop:

```
LOOP:
  1. EDIT      → Modify train.py: write a new RewardWrapper class
  2. TRAIN     → python train.py (trains with your reward, prints SUCCESS_RATE)
  3. EVALUATE  → Parse SUCCESS_RATE from stdout
  4. ANALYZE   → Use MCP tools to capture failure frames
                  Look at the frames (you are a VLM)
                  Diagnose WHY the robot fails spatially/physically
  5. REDESIGN  → Use failure diagnosis to inform the next reward wrapper
  6. DECIDE    → Improved? git add train.py && git commit
                  Worse? git checkout train.py
  7. REPEAT
```

The key insight: **your visual understanding of failure frames drives reward
redesign.** Metrics tell you THAT the robot failed. Frames tell you WHY.
"The gripper never reaches the handle" suggests an end-effector-proximity
reward. "The arm pushes the handle in the wrong direction" suggests a
directional alignment term. "The arm collides with the cabinet above"
suggests a collision penalty. This visual-to-reward feedback loop is the
core of the system.

## What You Can Modify

Edit ONLY `train.py`. You have full freedom to change:

### Reward Wrappers (Primary Search Variable)
Write `RewardWrapper` classes that override the environment's reward.
Reward components you can design include (but are not limited to):
- **Handle distance to goal** (microwave joint position vs target)
- **End-effector proximity** (Franka gripper to handle location — requires
  forward kinematics from joint angles, or a proxy distance using joint
  configuration)
- **Approach direction** (gripper moving toward the handle, not away)
- **Pull direction alignment** (handle motion in the correct rotational
  direction once contact is made)
- **Joint smoothness** (penalize jerky robot motion)
- **Contact phase rewards** (different reward shape before/after gripper
  reaches handle)
- **Collision avoidance** (penalize hitting cabinets, microwave body, table)
- **Success bonus** (large positive reward when within `BONUS_THRESH`)

Each wrapper must log per-component reward values in `info['reward_components']`
so you can analyze which components help and which hurt.

### Algorithms
Not restricted to any fixed set. Available algorithms:
- `stable_baselines3`: SAC, TD3, PPO, DDPG, A2C
- `sb3_contrib`: TQC, CrossQ, TRPO, ARS

You may also implement custom modifications inline: exploration schedules,
entropy tuning, custom noise processes, ensemble methods.

### Architecture & Hyperparameters
Network architecture, learning rate (including schedules), batch size,
buffer size, gamma, tau, HER configuration (n_sampled_goal,
goal_selection_strategy), number of critics (TQC), etc.

### Meta-Strategies
You may implement your own search logic within train.py:
- Train multiple reward variants per iteration and compare
- Use results.tsv history to guide exploration
- Curriculum approaches (start near the goal pose, then back off; or chain
  to multi-subtask goals once microwave is solved)

## Available MCP Tools

Use these to understand WHY the robot fails:
- `capture_frames` — render simulation frames from episodes
- `get_failure_episodes` — detailed data from failed episodes with frames
- `compare_runs` — side-by-side metric comparison
- `compute_metrics` — per-component reward breakdowns
- `save_checkpoint` / `list_experiments` — track experiment history

**Use capture_frames and get_failure_episodes after every experiment.**
Visual failure analysis is not optional — it is the core mechanism that
makes this system more than a hyperparameter search.

## Observation Space Reference

FrankaKitchen-v1 observation vector (59 dimensions). After the
`FlattenSingleSubtaskGoal` wrapper, the obs dict is:

```
obs['observation']        → 59-dim env state
obs['achieved_goal']      → current microwave handle joint value, shape (1,)
obs['desired_goal']       → target microwave handle joint value (-0.75), shape (1,)
```

Selected indices into `obs['observation']` (see
`gymnasium_robotics.envs.franka_kitchen.kitchen_env.OBS_ELEMENT_INDICES`):

```
obs['observation'][0:9]    → Franka 9-DOF joint positions
obs['observation'][11:13]  → bottom burner joint(s)
obs['observation'][15:17]  → top burner joint(s)
obs['observation'][17:19]  → light switch
obs['observation'][19]     → slide cabinet
obs['observation'][20:22]  → hinge cabinet
obs['observation'][22]     → microwave handle joint   ← active subtask
obs['observation'][23:30]  → kettle pose (xyz + quat)
```

`info['step_task_completions']` and `info['episode_task_completions']` from
the underlying env list which subtasks completed; these drive the ground-truth
success metric.

## Constraints
- TIME_BUDGET default is 3600 seconds (1h) — FrankaKitchen episodes are
  ~6× longer than Fetch tasks and need more samples to show signal.
  Increase for promising configs.
- Do not modify the evaluation protocol (20 episodes, deterministic policy,
  ground-truth `episode_task_completions`).
- The sparse `episode_task_completions` signal is ground truth. Your reward
  wrapper shapes training signal but SUCCESS_RATE is what matters.
- Do not modify `prepare.py` or files in `mcp/`.
- The `FlattenSingleSubtaskGoal` wrapper assumes a single active subtask;
  if you chain subtasks, you must update the wrapper accordingly.

## NEVER STOP
Once the experiment loop begins, do NOT pause to ask the human.
The human might be asleep. You are autonomous. If you run out of ideas,
try combining previous near-miss reward components, try radical redesigns,
try different algorithms with the same reward, or try the same reward
with different architectures. The loop runs until interrupted.
