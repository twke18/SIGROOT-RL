# FPO Revision Plan

## Summary

The current implementation in `algorithms/fpo.py` is fundamentally incorrect. It does not implement
the FPO algorithm from the paper. Instead it uses an advantage-weighted behavior cloning loss
(flow matching toward positive-advantage rollout actions), which is a heuristic that lacks the
principled PPO-clip surrogate at the core of FPO.

The four critical issues are described below, followed by the concrete changes required.

---

## Issue 1: Missing FPO ratio

**What the paper requires (Algorithm 1, line 9; Eq. 6/16):**

The central quantity in FPO is the ratio proxy:

```
r̂_θ = exp( -1/N_mc * Σ_{i=1}^{N_mc} ( ℓ_θ(τ_i, ε_i) - ℓ_{θ_old}(τ_i, ε_i) ) )
```

This ratio stands in for the intractable likelihood ratio π_θ / π_{θ_old} used in standard PPO.
A smaller flow matching loss under θ relative to θ_old means the action is more likely under the
new policy, so r̂_θ > 1, exactly mirroring the PPO likelihood ratio.

**What the code does:**

The code never computes any ratio. It only computes a raw flow matching loss weighted by
normalized positive advantages — which is behavior cloning toward high-return actions, not FPO.

---

## Issue 2: Missing MC sample storage during rollout

**What the paper requires (Algorithm 1, lines 2-3):**

During rollout collection, for each action `a_t` taken in the environment, the algorithm must:
1. Sample N_mc (τ_i, ε_i) pairs from Uniform[0,1] × N(0,I).
2. Pre-compute and store `ℓ_{θ_old}(τ_i, ε_i)` for all N_mc pairs under the current (rollout)
   policy parameters θ_old.

These stored (τ_i, ε_i, ℓ_{θ_old}) triples are required during the update to compute the ratio.
The same noise samples must be reused across all optimization epochs so the ratio compares
θ vs θ_old on identical inputs.

**What the code does:**

The code samples fresh (τ, ε) pairs inside the update loop (`_per_sample_flow_loss`). This means
there is no stable "old policy" reference: both the numerator and denominator of the ratio use
the same freshly sampled noise, making ratio computation impossible.

**Fix:** The runner/rollout collector must call a method on the policy to pre-sample and cache N_mc
(τ_i, ε_i, ℓ_{θ_old}) triples per timestep. These are passed into `FPO.update()` alongside the
standard rollout buffers.

---

## Issue 3: Missing θ_old snapshot before optimization epochs

**What the paper requires (Algorithm 1, line 4):**

Before entering the optimization epoch loop, the algorithm saves `θ_old ← θ`. During epochs,
ℓ_{θ_old} is evaluated on the fixed (τ_i, ε_i) samples to compute the ratio denominator.

**What the code does:**

There is no θ_old snapshot. The code has no mechanism to evaluate the policy at its pre-update
parameter values during optimization.

**Fix:** At the start of `update()`, snapshot the policy parameters (e.g., via `torch.no_grad()`
evaluation or a separate frozen copy) so ℓ_{θ_old}(τ_i, ε_i) can be computed from the cached
samples during each mini-batch update.

Since ℓ_{θ_old}(τ_i, ε_i) is pre-computed and stored during rollout (Issue 2), the snapshot
is already captured implicitly. The update loop only needs to re-evaluate ℓ_θ (current parameters)
on the stored (τ_i, ε_i) samples and compare against the stored ℓ_{θ_old} values.

---

## Issue 4: Wrong actor objective — missing PPO-clip surrogate

**What the paper requires (Algorithm 1, line 10; Eq. 5):**

```
L^FPO(θ) = min( r̂_θ * Â_t,  clip(r̂_θ, 1 - ε_clip, 1 + ε_clip) * Â_t )
```

This is identical in form to the PPO-clip objective, with r̂_θ replacing the likelihood ratio.

**What the code does:**

```python
# Current (wrong):
pos_mask = b_adv > 0
flow_loss = (flow_loss_per[pos_mask] * normalized_weights).sum()
actor_loss = flow_loss_coef * flow_loss
```

This is advantage-weighted behavior cloning, not the PPO-clip surrogate. It:
- Discards all negative-advantage samples entirely (paper uses all samples via clipping).
- Has no trust-region constraint (no clip).
- Maximizes flow matching fit to rollout actions rather than the FPO ratio objective.

**Fix:** Replace the actor loss with:
```python
# Correct:
ratio = torch.exp(-(fpo_loss_current - old_fpo_loss) / N_mc)   # r̂_θ, shape (B,)
surr1 = ratio * b_adv
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
actor_loss = -torch.min(surr1, surr2).mean()
```

where `fpo_loss_current` is `Σ_i ℓ_θ(τ_i, ε_i)` computed from the stored (τ_i, ε_i) pairs,
and `old_fpo_loss` is the pre-stored `Σ_i ℓ_{θ_old}(τ_i, ε_i)`.

---

## Required Changes

### `algorithms/fpo.py`

| Location | Change |
|---|---|
| `__init__` | Add `clip_eps: float = 0.05` (paper default), `n_mc: int = 8` hyperparameters |
| `__init__` | Remove `flow_loss_coef` (no longer used as a raw loss weight) |
| `update()` signature | Add `old_fpo_losses` argument: pre-stored `Σ_i ℓ_{θ_old}(τ_i, ε_i)` per timestep, shape `(N,)` |
| `update()` signature | Add `mc_taus`, `mc_epsilons` arguments: stored (τ_i, ε_i) samples, shapes `(N, N_mc)` and `(N, N_mc, act_dim)` |
| Actor loss block | Replace advantage-weighted BC loss with PPO-clip on r̂_θ (see above) |
| `_per_sample_flow_loss` | Keep but rename to `_compute_fpo_loss`; make it accept pre-stored τ, ε rather than sampling them internally |
| Remove | Positive-only masking (`pos_mask`) logic |

### `policies/flow_policy.py` (or equivalent)

- Add `sample_mc_pairs(obs, actions, n_mc)` method: samples N_mc (τ_i, ε_i) pairs and returns
  the summed flow matching loss under current parameters. Called during rollout collection to
  produce `old_fpo_losses`, `mc_taus`, `mc_epsilons`.

### `runner.py` (rollout collection)

- After each environment step, call `policy.sample_mc_pairs(obs, action, n_mc)` and buffer the
  outputs alongside the standard (obs, action, reward, done) buffers.
- Pass `old_fpo_losses`, `mc_taus`, `mc_epsilons` to `fpo.update()`.

### `configurations/FPO.yaml`

- Add `clip_eps: 0.05` (paper's best reported value).
- Add `n_mc: 8` (paper's default; higher improves accuracy at cost of compute).

---

## What Does NOT Need to Change

- The flow matching formula in `_per_sample_flow_loss` is correct: OT interpolation
  `x_t = (1-t)*ε + t*a` with velocity target `a - ε` matches Eqs. 8-9 in the paper.
- The critic loss (MSE on value estimates) is correct per Algorithm 1 line 14.
- The GAE advantage computation in the runner is unchanged.
- The optimizer (Adam) and gradient clipping are fine.
