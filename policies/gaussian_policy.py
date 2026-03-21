"""Gaussian MLP policy for PPO (actor + critic)."""
import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_mlp(input_dim: int, hidden_dims: list[int], activation: nn.Module) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), activation]
        in_dim = h
    return nn.Sequential(*layers), in_dim


class GaussianPolicy(nn.Module):
    """
    Actor-critic with a shared MLP backbone.

    Actor head: linear -> mean; separate log_std parameter
    Critic head: linear -> scalar value
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int], activation: str = "tanh"):
        super().__init__()
        act_fn = nn.Tanh() if activation == "tanh" else nn.ReLU()

        backbone, out_dim = _build_mlp(obs_dim, hidden_dims, act_fn)
        self.backbone = backbone

        self.mean_head = nn.Linear(out_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.value_head = nn.Linear(out_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m is self.mean_head else 1.0)
                nn.init.zeros_(m.bias)
        # value head uses gain 1
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """Returns (action, log_prob, value) for a batch of observations."""
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value_head(features).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Returns (log_prob, entropy, value) for given obs-action pairs (used in PPO update)."""
        features = self.backbone(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(features).squeeze(-1)
        return log_prob, entropy, value

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.value_head(features).squeeze(-1)
