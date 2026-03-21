"""Rollout buffer for collecting environment transitions."""
import numpy as np
import torch


class RolloutBuffer:
    """
    Stores (obs, action, reward, done, value, log_prob) for all envs across
    rollout_steps steps, then computes returns and GAE advantages.

    Supports both PPO (single action per step) and FPO (single sampled action
    stored; multiple samples generated externally for the policy gradient loss).
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self._obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        self._actions = np.zeros((rollout_steps, num_envs, act_dim), dtype=np.float32)
        self._rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self._dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self._values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self._log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        self._ptr = 0
        self._full = False

        # Computed by compute_returns_and_advantages()
        self.returns: np.ndarray | None = None
        self.advantages: np.ndarray | None = None

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ):
        assert self._ptr < self.rollout_steps, "Buffer is full; call reset() first."
        self._obs[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = done.astype(np.float32)
        self._values[self._ptr] = value
        self._log_probs[self._ptr] = log_prob
        self._ptr += 1
        if self._ptr == self.rollout_steps:
            self._full = True

    def compute_returns_and_advantages(self, last_values: np.ndarray):
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_values: value estimates for the state *after* the final step,
                         shape (num_envs,).
        """
        advantages = np.zeros_like(self._rewards)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - self._dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self._dones[t]
                next_values = self._values[t + 1]

            delta = (
                self._rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self._values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.returns = advantages + self._values
        self.advantages = advantages

    def get_tensors(self):
        """
        Return flattened tensors of shape (rollout_steps * num_envs, ...) for training.
        Normalizes advantages to zero mean, unit variance.
        """
        assert self._full, "Buffer not yet full."
        assert self.advantages is not None, "Call compute_returns_and_advantages() first."

        T, N = self.rollout_steps, self.num_envs

        obs = torch.tensor(self._obs.reshape(T * N, self.obs_dim), device=self.device)
        actions = torch.tensor(self._actions.reshape(T * N, self.act_dim), device=self.device)
        log_probs = torch.tensor(self._log_probs.reshape(T * N), device=self.device)
        returns = torch.tensor(self.returns.reshape(T * N), device=self.device)

        adv = self.advantages.reshape(T * N)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        advantages = torch.tensor(adv, device=self.device)

        return obs, actions, log_probs, returns, advantages

    def reset(self):
        self._ptr = 0
        self._full = False
        self.returns = None
        self.advantages = None

    @property
    def avg_reward(self) -> float:
        """Mean per-step reward across all envs during the rollout."""
        return float(self._rewards.mean())

    @property
    def total_steps(self) -> int:
        return self.rollout_steps * self.num_envs
