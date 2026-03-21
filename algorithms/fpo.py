"""Flow Matching Policy Gradients (FPO) algorithm.

Implements Algorithm 1 from "Flow Matching Policy Gradients" (McAllister et al., 2025).

The key idea is to replace PPO's intractable likelihood ratio with a proxy ratio
computed from the conditional flow matching (CFM) loss:

    r̂_θ = exp( -1/N_mc * Σ_i ( ℓ_θ(τ_i, ε_i) - ℓ_{θ_old}(τ_i, ε_i) ) )

where ℓ_θ(τ, ε) = ||v̂_θ(x_t; τ; obs) - (a - ε)||²  and  x_t = (1-τ)ε + τa.

This ratio is used in the standard PPO-clip surrogate objective:

    L^FPO(θ) = min( r̂_θ * Â,  clip(r̂_θ, 1-ε_clip, 1+ε_clip) * Â )

The value function is updated identically to standard PPO.

The (τ_i, ε_i) pairs and the old-policy loss ℓ_{θ_old}(τ_i, ε_i) are sampled
and cached during rollout collection (via FlowPolicy.sample_mc_pairs), so the
same noise inputs are reused across all optimisation epochs.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FPO:
    """
    Flow Matching Policy Gradients.

    Works with FlowPolicy (flow-based actor + MLP critic).
    """

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 3e-4,
        value_loss_coef: float = 0.5,
        clip_eps: float = 0.05,
        n_mc: int = 8,
        epochs_per_update: int = 10,
        minibatch_size: int = 256,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.value_loss_coef = value_loss_coef
        self.clip_eps = clip_eps
        self.n_mc = n_mc
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def update(
        self,
        obs,
        actions,
        old_log_probs,
        returns,
        advantages,
        mc_taus,
        mc_epsilons,
        old_fpo_losses,
    ):
        """
        Run FPO update over the collected rollout.

        Args:
            obs:           (N, obs_dim)
            actions:       (N, act_dim)   — actions taken during rollout
            old_log_probs: (N,)           — unused (kept for API compatibility)
            returns:       (N,)
            advantages:    (N,)           — normalised advantages
            mc_taus:       (N, n_mc, 1)   — cached flow timesteps from rollout
            mc_epsilons:   (N, n_mc, act_dim) — cached noise from rollout
            old_fpo_losses:(N,)           — sum of ℓ_{θ_old} over n_mc pairs

        Returns:
            (mean_actor_loss, mean_critic_loss) over all minibatches
        """
        dataset = TensorDataset(
            obs, actions, returns, advantages,
            mc_taus, mc_epsilons, old_fpo_losses,
        )
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_batches = 0

        for _ in range(self.epochs_per_update):
            for batch in loader:
                b_obs, b_actions, b_returns, b_adv, b_taus, b_eps, b_old_loss = batch
                B = b_obs.shape[0]

                # -----------------------------------------------------------
                # 1. Critic loss: MSE on value estimates (same as PPO)
                # -----------------------------------------------------------
                values = self.policy.value_head(
                    self.policy.critic_backbone(b_obs)
                ).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, b_returns)

                # -----------------------------------------------------------
                # 2. Compute current FPO loss on the stored (τ, ε) pairs
                #
                # Flatten (B, n_mc, *) → (B*n_mc, *) for vectorised eval.
                # -----------------------------------------------------------
                obs_exp = b_obs.unsqueeze(1).expand(B, self.n_mc, b_obs.shape[-1]).reshape(B * self.n_mc, -1)
                act_exp = b_actions.unsqueeze(1).expand(B, self.n_mc, b_actions.shape[-1]).reshape(B * self.n_mc, -1)
                taus_flat = b_taus.reshape(B * self.n_mc, 1)
                eps_flat = b_eps.reshape(B * self.n_mc, -1)

                x_t = (1 - taus_flat) * eps_flat + taus_flat * act_exp
                target_v = act_exp - eps_flat
                pred_v = self.policy.vector_field(obs_exp, x_t, taus_flat)

                per_sample = ((pred_v - target_v) ** 2).mean(dim=-1)       # (B*n_mc,)
                current_fpo_loss = per_sample.reshape(B, self.n_mc).sum(dim=1)  # (B,)

                # -----------------------------------------------------------
                # 3. FPO ratio and PPO-clip surrogate (Algorithm 1, lines 9-10)
                #
                #   r̂_θ = exp( -1/N_mc * (ℓ_θ - ℓ_{θ_old}) )
                #   L^FPO = min( r̂_θ * Â,  clip(r̂_θ, 1±ε) * Â )
                # -----------------------------------------------------------
                ratio = torch.exp(-(current_fpo_loss - b_old_loss) / self.n_mc)  # (B,)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                loss = actor_loss + self.value_loss_coef * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_batches += 1

        return total_actor_loss / n_batches, total_critic_loss / n_batches
