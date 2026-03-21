# Task Revision: Add Humanoid and HumanoidStandup Environments

## Overview

This revision extends the codebase from HalfCheetah-only training to support three Mujoco environments:

| Environment | Gym ID | obs_dim | act_dim |
|---|---|---|---|
| HalfCheetah | `HalfCheetah-v4` | 17 | 6 |
| Humanoid | `Humanoid-v4` | 376 | 17 |
| HumanoidStandup | `HumanoidStandup-v4` | 376 | 17 |

Both PPO and FPO algorithms should be runnable on all three environments.

---

## Changes Required

### 1. Generalize the environment wrapper

**File:** `envs/half_cheetah.py` → rename to `envs/mujoco_env.py`

Rename the `HalfCheetahEnv` class to `MujocoEnv`. The internal logic is already fully generic (it accepts `env_name` as a parameter and uses `SyncVectorEnv`), so only the class name and module name need to change. The old `envs/half_cheetah.py` can be removed.

### 2. Update the runner script

**File:** `runner.py`

Three targeted changes:

**a. Import from the renamed module:**
```python
# Before
from envs.half_cheetah import HalfCheetahEnv
# After
from envs.mujoco_env import MujocoEnv
```
Replace all `HalfCheetahEnv(...)` instantiations with `MujocoEnv(...)`.

**b. Make success threshold configurable:**

The current hardcoded `SUCCESS_THRESHOLD = 300.0` in `run_test()` is only valid for HalfCheetah. Move it to the training config and read it at runtime:

```yaml
# In each config file
training:
  success_threshold: 300.0   # tune per environment
```

```python
# In run_test() signature
def run_test(policy, env_wrapper, device, num_episodes, max_steps, success_threshold):
    ...
    success_rate = float(np.mean([r >= success_threshold for r in rewards]))
```

Call site in `train()` reads `train_cfg.get("success_threshold", 300.0)`.

**c. Include environment name in the Logger tag:**

So that runs and videos from different environments are clearly separated:

```python
# Derive a short env label from config, e.g. "HalfCheetah", "Humanoid", "HumanoidStandup"
env_label = env_cfg["name"].split("-")[0]   # strips "-v4"
run_tag = f"{algo}_{env_label}"             # e.g. "PPO_Humanoid"
logger = Logger(run_tag)
```

This makes TensorBoard run dirs and video filenames include the environment name
(e.g., `runs/PPO_Humanoid_20260321_153739/`, `videos/PPO_Humanoid_epoch0100.mp4`).

Update the runner docstring and `argparse` description to mention all three environments.

### 3. Update the `envs/__init__.py`

Export `MujocoEnv` instead of (or in addition to) the old class so imports stay clean.

### 4. Add new configuration files

Create four new config files following the existing structure. Naming convention: `<ENV>_<ALGO>.yaml` to keep config filename self-describing while keeping `detect_algo()` working (it scans for "PPO" / "FPO" in the filename).

**`configurations/Humanoid_PPO.yaml`**
```yaml
env:
  name: Humanoid-v4
  num_envs: 16
  seed: 42

policy:
  hidden_dims: [512, 512]
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
  max_grad_norm: 0.5

training:
  total_epochs: 1000
  rollout_steps: 2048
  test_interval: 10
  test_episodes: 5
  video_interval: 50
  checkpoint_interval: 50
  success_threshold: 3000.0
```

**`configurations/Humanoid_FPO.yaml`** — same env/training sections, policy includes flow fields, algorithm uses FPO hyperparameters.

**`configurations/HumanoidStandup_PPO.yaml`** — env name `HumanoidStandup-v4`, higher success threshold (~50000) since the reward accumulates differently (upright posture bonus each step).

**`configurations/HumanoidStandup_FPO.yaml`** — same env/training sections with FPO hyperparameters.

Key differences from HalfCheetah configs:
- Larger hidden dims `[512, 512]` to handle the larger obs/act space (obs=376, act=17).
- More training epochs (1000) since Humanoid is a harder task.
- Per-environment `success_threshold` values.

### 5. Rename existing HalfCheetah configs (optional, for consistency)

To match the new `<ENV>_<ALGO>.yaml` convention:

| Old name | New name |
|---|---|
| `configurations/PPO.yaml` | `configurations/HalfCheetah_PPO.yaml` |
| `configurations/FPO.yaml` | `configurations/HalfCheetah_FPO.yaml` |

If renamed, update the `runner.py` docstring examples accordingly. The existing `detect_algo()` logic (`"PPO" in name`, `"FPO" in name`) continues to work without changes.

### 6. Update CLAUDE.md

Update the description and Gym environment link to reflect that the codebase now supports HalfCheetah, Humanoid, and HumanoidStandup. Add links to the Humanoid and HumanoidStandup Gymnasium pages.

---

## Summary of file changes

| Action | File |
|---|---|
| Rename + class rename | `envs/half_cheetah.py` → `envs/mujoco_env.py` (`HalfCheetahEnv` → `MujocoEnv`) |
| Modify | `envs/__init__.py` — export `MujocoEnv` |
| Modify | `runner.py` — import, success threshold, logger tag |
| Create | `configurations/Humanoid_PPO.yaml` |
| Create | `configurations/Humanoid_FPO.yaml` |
| Create | `configurations/HumanoidStandup_PPO.yaml` |
| Create | `configurations/HumanoidStandup_FPO.yaml` |
| Rename (optional) | `configurations/PPO.yaml` → `configurations/HalfCheetah_PPO.yaml` |
| Rename (optional) | `configurations/FPO.yaml` → `configurations/HalfCheetah_FPO.yaml` |
| Modify | `CLAUDE.md` — update description and links |

No changes are required to the algorithm implementations (`algorithms/ppo.py`, `algorithms/fpo.py`), policy networks (`policies/`), rollout buffer (`utils/rollout.py`), or logger (`utils/logger.py`). The changes are localized to the environment wrapper, runner, and configuration files.
