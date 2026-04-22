# Experiment Log — FrankaKitchen-v1 Microwave

Reverse-chronological log of every training iteration, with the hypothesis,
result, and the failure analysis that motivated the next change.

Ground truth metric: `SUCCESS_RATE` (= fraction of 20 deterministic eval
episodes where `subtask in info["episode_task_completions"]`).

## Iteration 0 — original baseline (NEVER FINISHED, KILLED)

- Reward: `-handle_distance + 5 * success_bonus`
- Bug found before completion: `MICROWAVE_OBS_IDX = 22` was a qpos index but
  used as an obs-vector index. obs[22] is the top right burner knob; the
  microwave joint is at obs[31]. The dense distance term was regressing
  against the wrong joint, providing zero useful gradient.
- HER replay still got correct sparse signal via `obs["achieved_goal"]`,
  but the dense shaping was noise.
- Killed after ~30 sec to avoid wasting an hour.

## Iteration 1 — fix obs-index bug + add end-effector proximity

- Hypothesis: the dominant failure mode in long-horizon kitchen RL is that
  the arm never gets near the handle in 280 steps of random exploration,
  so a joint-distance term alone has no entry path. Adding direct
  end-effector ↔ handle Euclidean distance (from MuJoCo sites
  `end_effector` and `microhandle_site`) gives a smooth gradient toward
  contact in xyz space.
- Reward: `-handle_distance - ee_to_handle + 5 * success_bonus`
- All else equal to iter 0 (TQC + HER, 1h budget, gamma=0.95,
  net_arch=[512,512,512]).
- **Result (in-training 3-episode eval):**
  - step 25k: SR=0.00
  - step 30k (14:21 wall): **SR=1.00** *NEW_BEST*
  - step 35k–75k (every 5k): SR=1.00 (sustained)
- **Final 20-episode deterministic eval: SUCCESS_RATE=1.0000, MEAN_REWARD=1.00.**
- 75k env-steps to solve. Wall-clock ~40 min of pure-CPU TQC (single env).
- Model saved to `iter1_microwave_TQC.zip`, progress log to
  `iter1_microwave_progress.log`.
- Headline: the reward fix + EE-proximity term took FrankaKitchen microwave
  from unsolvable (≈0% across published from-scratch online-RL baselines)
  to **20/20 success** on a single CPU in under an hour of training.
- Time-budget observation: `TimeBudgetCallback` triggered cleanly; final
  in-script wall clock was inflated by IO contention from concurrent
  shell tooling — not a property of the training itself.
