# An RL playground for SIGROOT workshop

A minimal, from-scratch reinforcement learning codebase for training continuous-control robot policies in [MuJoCo](https://mujoco.org/). It implements and compares two on-policy policy-gradient algorithms on Gymnasium MuJoCo tasks:

- **PPO** — Proximal Policy Optimization with a Gaussian policy.
- **FPO** — [Flow Matching Policy Gradients](support_information/fpo_paper.pdf) (McAllister et al., 2025), which replaces PPO's likelihood ratio with a conditional flow-matching proxy ratio and generates actions by integrating a learned flow from noise.

Supported tasks: **HalfCheetah**, **Humanoid**, and **HumanoidStandup**.

## Repository layout

```
sigroot/
├── runner.py              # Main entry point: training & testing loop
├── algorithms/
│   ├── ppo.py             # Clipped-surrogate PPO update
│   └── fpo.py             # Flow Matching Policy Gradients (Algorithm 1)
├── policies/
│   ├── gaussian_policy.py # Actor-critic MLP with Gaussian actor (PPO)
│   └── flow_policy.py     # Conditional flow-matching actor + MLP critic (FPO)
├── envs/
│   ├── mujoco_env.py      # Generic vectorized MuJoCo env wrapper
│   └── half_cheetah.py    # HalfCheetah-specific wrapper
├── utils/
│   ├── rollout.py         # Rollout buffer + GAE
│   └── logger.py          # TensorBoard + video logging
├── configurations/        # Per-task YAML configs ({Task}_{PPO,FPO}.yaml)
├── instructions/
│   └── RobotPolicy.md     # Notes on policy architectures
├── support_information/   # FPO paper (PDF) + algorithm figure
├── checkpoints/           # Saved weights      (git-ignored)
├── runs/                  # TensorBoard logs   (git-ignored)
├── videos/                # Rendered rollouts  (git-ignored)
└── requirements.txt
```

## Installation

Requires **Python 3.10**.

```bash
git clone https://github.com/twke18/SIGROOT-RL.git
cd SIGROOT-RL/
pip install -r requirements.txt
```

Key dependencies (see [requirements.txt](requirements.txt)):

| Package | Purpose |
|---|---|
| `gymnasium[mujoco]==0.29.1` | RL environment API |
| `mujoco>=3.1.0` | Physics simulation |
| `torch>=2.9.0` | Networks & optimization |
| `tensorboard>=2.14` | Training visualization |
| `opencv-python-headless>=4.8` | MP4 video encoding |
| `PyYAML>=6.0` | Config parsing |

> Headless rendering is enabled automatically (`MUJOCO_GL=egl`) by `runner.py`.

## Usage

The single entry point is [runner.py](runner.py). The algorithm (PPO or FPO) is inferred from the config filename.

### Training

```bash
# PPO
python runner.py --config configurations/HalfCheetah_PPO.yaml
python runner.py --config configurations/Humanoid_PPO.yaml
python runner.py --config configurations/HumanoidStandup_PPO.yaml

# FPO
python runner.py --config configurations/HalfCheetah_FPO.yaml
python runner.py --config configurations/Humanoid_FPO.yaml

# Force a device (default: auto-detect CUDA, else CPU)
python runner.py --config configurations/Humanoid_FPO.yaml --device cuda
```

### Testing a checkpoint

```bash
python runner.py \
  --config configurations/HalfCheetah_PPO.yaml \
  --mode test \
  --checkpoint checkpoints/PPO_HalfCheetah_epoch0500.pt
```

### Command-line arguments

| Flag | Default | Description |
|---|---|---|
| `--config` | *(required)* | Path to a YAML config in `configurations/` |
| `--mode` | `train` | `train` or `test` |
| `--checkpoint` | `None` | `.pt` file to load (required for `test`) |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |

### Monitoring

```bash
tensorboard --logdir runs/
```

Per run, the trainer writes:
- **Scalars** → `runs/<ALGO>_<timestamp>/` (rollout reward, actor/critic loss, test reward, success rate)
- **Videos** → `videos/<ALGO>_epoch<N>.mp4`
- **Checkpoints** → `checkpoints/<ALGO>_<Task>_epoch<NNNN>.pt`

## Configuration

Each task has a `{Task}_{PPO,FPO}.yaml` file. Example ([Humanoid_FPO.yaml](configurations/Humanoid_FPO.yaml)):

```yaml
env:
  name: Humanoid-v4      # Gymnasium environment id
  num_envs: 64           # parallel environments
  seed: 42

policy:
  hidden_dims: [512, 512]
  activation: tanh
  num_flow_samples: 8    # FPO only: action samples per observation
  flow_steps: 10         # FPO only: Euler integration steps

algorithm:
  gamma: 0.99
  gae_lambda: 0.95
  value_loss_coef: 0.5
  clip_eps: 0.05         # FPO clip range (PPO uses clip_epsilon, ~0.2)
  n_mc: 8                # FPO only: MC (tau, eps) pairs for the proxy ratio
  epochs_per_update: 5
  minibatch_size: 256
  learning_rate: 1.0e-4
  max_grad_norm: 0.5

training:
  total_epochs: 1000
  rollout_steps: 2048
  test_interval: 10
  test_episodes: 5
  video_interval: 50
  checkpoint_interval: 50
  success_threshold: 3000.0   # reward defining a "success" at test time
```

PPO configs use `clip_epsilon` and an optional `entropy_coef` instead of FPO's `clip_eps` / `n_mc`.

## Algorithms

### PPO ([algorithms/ppo.py](algorithms/ppo.py))

Standard clipped-surrogate objective with a [GaussianPolicy](policies/gaussian_policy.py) (shared MLP trunk, mean head + learnable log-std, scalar critic). Advantages come from GAE, optimized with Adam and gradient clipping.

### FPO ([algorithms/fpo.py](algorithms/fpo.py))

Implements Algorithm 1 of *Flow Matching Policy Gradients*. The PPO likelihood ratio is replaced by a flow-matching proxy:

1. Sample `n_mc` pairs `(τ, ε)` and form `x_τ = (1-τ)·ε + τ·a`.
2. Evaluate the conditional flow-matching (CFM) loss `ℓ(τ,ε) = ‖v(x_τ, τ | obs) − (a − ε)‖²`.
3. Proxy ratio `r̂ = exp(−(ℓ_new − ℓ_old) / n_mc)`, plugged into the clipped objective.

The `(τ, ε)` pairs are cached during rollout so the ratio is consistent across optimization epochs. Actions are generated by [FlowPolicy](policies/flow_policy.py) by integrating the learned velocity field from Gaussian noise (`t=0`) to an action (`t=1`) with an Euler solver.

## Training loop

For each epoch, `runner.py`:

1. **Rollout** — collect `rollout_steps` transitions across `num_envs` parallel envs.
2. **Learn** — compute GAE returns/advantages and run `epochs_per_update` minibatch SGD updates.
3. **Test** — every `test_interval` epochs, evaluate over `test_episodes` and log mean reward / success rate.
4. **Record** — every `video_interval` epochs, render a rollout to MP4.
5. **Checkpoint** — every `checkpoint_interval` epochs, save weights.

## References

- McAllister et al., *Flow Matching Policy Gradients*, 2025 — included as [support_information/fpo_paper.pdf](support_information/fpo_paper.pdf).
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017).
