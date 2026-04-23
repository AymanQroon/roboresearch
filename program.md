# RoboResearch Program

## Goal
Autonomously improve robot manipulation success rates in MuJoCo simulation through iterative experimentation, starting from baseline hyperparameters and converging on high-performance policies.

## Starting Point
- Environment: FetchReach-v4 (3D end-effector reaching)
- Algorithm: SAC (Soft Actor-Critic)
- Use default hyperparameters as baseline: lr=3e-4, batch_size=256, buffer_size=100000, net_arch=[256, 256]

## Research Directions

### Phase 1: Hyperparameter Optimization on FetchReach-v4
1. **Learning rate sweep** — Try rates from 1e-4 to 1e-3. SAC is sensitive to learning rate; too high causes instability, too low slows convergence.
2. **Network architecture** — Experiment with deeper networks [256, 256, 256] and wider networks [512, 512]. FetchReach is low-dimensional so smaller nets may also work well — try [128, 128].
3. **Replay buffer and batch size** — Increase buffer_size to 500K-1M for more diverse experience replay. Try larger batch sizes (512, 1024) for more stable gradient estimates.
4. **Soft update coefficient (tau)** — Try values from 0.001 to 0.02. Smaller tau means slower target network updates, which can stabilize training.
5. **Discount factor (gamma)** — Try 0.95-0.999. FetchReach episodes are short, so lower gamma may help focus on immediate rewards.

### Phase 2: Algorithm Comparison
6. **TD3 baseline** — Run TD3 with default hyperparameters on the same task for comparison.
7. **PPO baseline** — Run PPO with default hyperparameters. PPO typically underperforms off-policy methods on goal-conditioned tasks, but establish the baseline.
8. **Best hyperparams on best algorithm** — Take the winning algorithm and apply the best hyperparameters found in Phase 1.

### Phase 3: Curriculum Advancement
9. **Graduate to FetchPush-v4** — Once FetchReach reaches >80% success rate consistently, move to the pushing task.
10. **Transfer tuning** — When moving to FetchPush, start with the best hyperparameters from FetchReach but be prepared to adjust (pushing requires more training and different exploration).

## Constraints
- Training time budget: 300 seconds (5 minutes) per experiment. This is non-negotiable for iteration speed.
- Keep learning_rate between 1e-5 and 1e-2.
- Keep batch_size between 32 and 2048.
- Keep network hidden layers between 64 and 512 units each, with 1-4 layers.
- Do not modify the environment reward function — use the default Gymnasium Robotics rewards.
- Do not use custom policies or modify the policy class — stick with MultiInputPolicy.
- Maximum 50 experiments total.

## Curriculum
- **FetchReach-v4**: Target success rate > 80% sustained over 3 consecutive experiments.
- **FetchPush-v4**: Target success rate > 60% sustained over 3 consecutive experiments.
- Graduation is automatic when thresholds are met.

## Evaluation Protocol
- Each experiment is evaluated over 20 episodes with deterministic policy.
- Primary metric: success_rate (fraction of episodes where task is completed).
- Secondary metric: mean_reward (higher is better; less negative for these sparse-reward tasks).
- Keep/discard decisions are made by the QuickEvaluator agent.
- Failed episodes are analyzed by the FailureAnalyst to extract actionable feedback.

## Notes
- SAC with HER (Hindsight Experience Replay) is the expected top performer for these goal-conditioned environments.
- Early experiments will show low success rates — this is normal. The system needs time to explore the hyperparameter space.
- If stuck for 5+ experiments with no improvement, the system will automatically try switching algorithms.
