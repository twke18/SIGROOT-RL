"""Flow Matching Policy Gradients (FPO) algorithm.

FPO replaces the Gaussian policy gradient step with:
1. A policy gradient update using multiple action samples drawn from the flow policy.
2. A flow matching loss to improve the flow network itself (distillation of the
   policy-gradient signal back into the generative model).

The value function is updated with the same GAE-based critic loss as PPO.

Reference: akanazawa/fpo
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
        flow_loss_coef: float = 1.0,
        epochs_per_update: int = 10,
        minibatch_size: int = 256,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.value_loss_coef = value_loss_coef
        self.flow_loss_coef = flow_loss_coef
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def update(self, obs, actions, old_log_probs, returns, advantages):
        """
        Run FPO update over the collected rollout.

        Policy gradient step:
          - Sample K actions per obs from the current flow policy.
          - Use the advantage to weight the flow matching loss toward
            high-advantage actions (behaviour cloning toward best actions).

        Args:
            obs:           (N, obs_dim)
            actions:       (N, act_dim)   — actions actually taken during rollout
            old_log_probs: (N,)           — unused by FPO (kept for API compatibility)
            returns:       (N,)
            advantages:    (N,)           — already normalised

        Returns:
            (mean_actor_loss, mean_critic_loss) over all minibatches
        """
        dataset = TensorDataset(obs, actions, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_batches = 0

        for _ in range(self.epochs_per_update):
            for batch in loader:
                b_obs, b_actions, b_returns, b_adv = batch
                B = b_obs.shape[0]

                # ---------------------------------------------------
                # 1. Critic loss: MSE on value estimates
                # ---------------------------------------------------
                values = self.policy.value_head(
                    self.policy.critic_backbone(b_obs)
                ).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values, b_returns)

                # ---------------------------------------------------
                # 2. Actor loss: advantage-weighted flow matching loss
                #
                # We use the rollout actions as targets weighted by
                # their normalised advantages (positive-only weighting
                # to clone toward high-reward trajectories).
                # ---------------------------------------------------
                # Keep only positive-advantage samples for cloning
                pos_mask = b_adv > 0
                if pos_mask.sum() < 2:
                    # Skip actor update if no positive-advantage samples
                    flow_loss = torch.tensor(0.0, device=self.device)
                else:
                    pos_obs = b_obs[pos_mask]
                    pos_actions = b_actions[pos_mask]
                    pos_weights = b_adv[pos_mask]
                    pos_weights = pos_weights / (pos_weights.sum() + 1e-8)

                    flow_loss_per = self._per_sample_flow_loss(pos_obs, pos_actions)
                    flow_loss = (flow_loss_per * pos_weights).sum()

                actor_loss = self.flow_loss_coef * flow_loss

                loss = actor_loss + self.value_loss_coef * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_batches += 1

        return total_actor_loss / n_batches, total_critic_loss / n_batches

    def _per_sample_flow_loss(
        self, obs: torch.Tensor, actions_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample CFM loss (not yet reduced).

        Returns shape (B,).
        """
        B = obs.shape[0]
        x_0 = torch.randn_like(actions_1)
        t = torch.rand(B, 1, device=obs.device)

        x_t = (1 - t) * x_0 + t * actions_1
        target_v = actions_1 - x_0

        pred_v = self.policy.vector_field(obs, x_t, t)
        return ((pred_v - target_v) ** 2).mean(dim=-1)  # (B,)
