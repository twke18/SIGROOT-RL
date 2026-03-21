# Implementation Plan: RL for HalfCheetah Locomotion

## Overview

Train a HalfCheetah robot with two RL algorithms:
- **PPO** (Proximal Policy Optimization) with a Gaussian MLP policy
- **FPO** (Flow Matching Policy Gradients) with a flow matching MLP policy

---

## Phase 1: Dependencies & Package Versions

**Python version: 3.10**
Python 3.10 is the sweet spot — fully supported by `torch` 2.2, `gymnasium` 0.29, `mujoco` 3.1, and `rl_games` 1.6. `rl_games` has known issues with 3.11+ (deprecated `collections` APIs), and 3.8/3.9 lack modern typing syntax used in newer wheels.

Proposed versions for simple installation and minimal dependencies:

| Package | Version | Reason |
|---|---|---|
| `gymnasium[mujoco]` | 0.29.x | Stable Mujoco support, replaces legacy `gym` |
| `mujoco` | 3.1.x | Required by gymnasium mujoco backend |
| `torch` | 2.2.x | Stable, widely available wheels |
| `rl_games` | 1.6.x | Latest stable, supports custom algorithms |
| `tensorboard` | 2.x | Logging numerical results |
| `PyYAML` | 6.x | Config file parsing |
| `opencv-python-headless` | 4.x | Video rendering/saving (no GUI deps) |

---

## Phase 2: Project Structure

```
sigroot/
├── runner.py                     # Main entry point
├── configurations/
│   ├── PPO.yaml                  # PPO hyperparameters & env config
│   └── FPO.yaml                  # FPO hyperparameters & env config
├── algorithms/
│   ├── __init__.py
│   ├── ppo.py                    # PPO algorithm (actor-critic update)
│   └── fpo.py                    # FPO algorithm (flow matching update)
├── policies/
│   ├── __init__.py
│   ├── gaussian_policy.py        # MLP + Gaussian head for PPO
│   └── flow_policy.py            # MLP + flow matching head for FPO
├── envs/
│   ├── __init__.py
│   └── half_cheetah.py           # Vectorized HalfCheetah env wrapper
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # TensorBoard + video logging
│   └── rollout.py                # Rollout buffer / data collection
├── videos/                       # Saved test rollout videos (gitignored)
├── runs/                         # TensorBoard logs (gitignored)
├── requirements.txt
└── README.md                     # User guidance (written last)
```

---

## Phase 3: Component Implementation

### 3.1 Environment (`envs/half_cheetah.py`)
- Wrap `gymnasium.make("HalfCheetah-v4")` with `gymnasium.vector.SyncVectorEnv` for parallel envs
- Support configurable `num_envs` and random initial states (`reset_noise_scale`)
- Expose `obs_dim` and `act_dim` for policy construction

### 3.2 Robot Policies (`policies/`)

**Gaussian Policy** (`gaussian_policy.py`) — for PPO:
- MLP backbone: `[obs_dim] -> hidden_layers -> [act_dim * 2]`
- Outputs mean and log-std; samples one action via reparameterization
- Separate value head (critic) sharing the MLP backbone

**Flow Matching Policy** (`flow_policy.py`) — for FPO:
- MLP backbone conditioned on observation and time `t`
- Implements a conditional flow matching vector field `v(x, t | obs)`
- Generates multiple action samples via ODE integration (simple Euler solver)
- Separate value head (critic)

### 3.3 Algorithms (`algorithms/`)

**PPO** (`ppo.py`):
- Clipped surrogate objective with ratio clipping (epsilon)
- Value function loss (MSE) + entropy bonus
- Multiple epochs over collected rollout buffer
- GAE (Generalized Advantage Estimation) for advantage computation

**FPO** (`fpo.py`):
- Flow matching loss: MSE between predicted and target vector field
- Policy gradient update using multiple action samples from flow policy
- Value function updated with same GAE as PPO
- Follows structure from [akanazawa/fpo](https://github.com/akanazawa/fpo)

### 3.4 Rollout Buffer (`utils/rollout.py`)
- Stores `(obs, action, reward, done, value, log_prob)` tuples across all envs
- Computes returns and GAE advantages on flush
- Supports both PPO (single action) and FPO (multiple actions) storage

### 3.5 Logger (`utils/logger.py`)
- TensorBoard `SummaryWriter` with run directory named by algorithm and timestamp
  - e.g., `runs/PPO_20260321_120000/` and `runs/FPO_20260321_120000/`
- Training scalars: `train/rollout_avg_reward`, `train/actor_loss`, `train/critic_loss`
- Testing scalars: `test/accumulated_reward`, `test/success_rate`
- Video saving: `videos/PPO_epoch{N}.mp4` and `videos/FPO_epoch{N}.mp4`

---

## Phase 4: Configuration Files

### `configurations/PPO.yaml`
```yaml
env:
  name: HalfCheetah-v4
  num_envs: 16
  seed: 42

policy:
  hidden_dims: [256, 256]
  activation: tanh

algorithm:
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coef: 0.0
  value_loss_coef: 0.5
  epochs_per_update: 10
  minibatch_size: 256
  learning_rate: 3.0e-4

training:
  total_epochs: 500
  rollout_steps: 2048
  test_interval: 10
  test_episodes: 5
  video_interval: 50
```

### `configurations/FPO.yaml`
```yaml
env:
  name: HalfCheetah-v4
  num_envs: 16
  seed: 42

policy:
  hidden_dims: [256, 256]
  activation: tanh
  num_flow_samples: 8
  flow_steps: 10

algorithm:
  gamma: 0.99
  gae_lambda: 0.95
  value_loss_coef: 0.5
  flow_loss_coef: 1.0
  epochs_per_update: 10
  minibatch_size: 256
  learning_rate: 3.0e-4

training:
  total_epochs: 500
  rollout_steps: 2048
  test_interval: 10
  test_episodes: 5
  video_interval: 50
```

---

## Phase 5: Runner Script (`runner.py`)

- Parse CLI arguments: `--config configurations/PPO.yaml` (or FPO)
- Instantiate environment, policy, algorithm, and logger from config
- Run training-testing loop:
  1. Collect rollouts -> log avg reward
  2. Update policy -> log actor/critic loss
  3. Every `test_interval` epochs: test policy -> log reward/success rate
  4. Every `video_interval` epochs: render and save video
- Support `--mode train` and `--mode test` (load checkpoint for test-only)
- Save/load checkpoints as `checkpoints/PPO_epoch{N}.pt`

---

## Phase 6: Build & Verification Order

1. Install dependencies, verify `HalfCheetah-v4` renders with random actions
2. Implement env wrapper + random rollout video
3. Implement rollout buffer + logger skeleton
4. Implement Gaussian policy + PPO algorithm
5. Wire into runner, run PPO, verify training curves in TensorBoard
6. Implement flow matching policy + FPO algorithm
7. Wire into runner, run FPO, verify training curves in TensorBoard
8. Write `README.md`

---

## Key Design Decisions

- **No Isaac Gym / parallel GPU envs**: Use `gymnasium.vector` (CPU) to keep dependencies minimal
- **rl_games integration**: Use `rl_games` runner/trainer interfaces where natural; implement custom algorithm classes that conform to `rl_games` agent API
- **Single codebase, two algorithms**: Config-driven dispatch in `runner.py` — no code duplication
- **Headless video rendering**: Use `mujoco` offscreen rendering (`rgb_array` mode) + OpenCV to avoid display dependencies
