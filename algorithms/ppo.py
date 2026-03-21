"""Proximal Policy Optimization (PPO) algorithm."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PPO:
    """
    PPO with clipped surrogate objective and GAE.

    Works with GaussianPolicy (actor-critic with shared backbone).
    """

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0,
        epochs_per_update: int = 10,
        minibatch_size: int = 256,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def update(self, obs, actions, old_log_probs, returns, advantages):
        """
        Run multiple epochs of PPO updates over the collected rollout.

        Args:
            obs:           (N, obs_dim)
            actions:       (N, act_dim)
            old_log_probs: (N,)
            returns:       (N,)
            advantages:    (N,) — already normalised by the buffer

        Returns:
            (mean_actor_loss, mean_critic_loss) over all minibatches
        """
        dataset = TensorDataset(obs, actions, old_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_batches = 0

        for _ in range(self.epochs_per_update):
            for batch in loader:
                b_obs, b_actions, b_old_lp, b_returns, b_adv = batch

                new_log_probs, entropy, values = self.policy.evaluate(b_obs, b_actions)

                # Clipped surrogate loss
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.functional.mse_loss(values, b_returns)

                # Entropy bonus (maximise entropy -> minus sign)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_batches += 1

        return total_actor_loss / n_batches, total_critic_loss / n_batches
