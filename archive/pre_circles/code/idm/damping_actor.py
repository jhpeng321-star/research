"""Small shared actor with a zero-initialized, bounded physical action head."""

from math import atanh, log

import torch
from torch import nn
from torch.distributions import Normal

from env.native_damping import DampingContract


class SharedDampingActor(nn.Module):
    """Actor only: critic state cannot enter this call interface.

    An affine tanh head with asymmetric bounds needs a shifted initial mean:
    raw mean zero would request -0.1 m/s², not native passthrough.
    """

    def __init__(self, contract: DampingContract = DampingContract(), hidden: int = 64):
        super().__init__()
        self.contract = contract
        self.center = (contract.residual_max + contract.residual_min) / 2
        self.radius = (contract.residual_max - contract.residual_min) / 2
        self.encoder = nn.Sequential(
            nn.Linear(contract.observation_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, atanh(-self.center / self.radius))
        self.log_std = nn.Parameter(torch.tensor([-2.0]))

    def distribution(self, local_history: torch.Tensor) -> Normal:
        if local_history.shape[-1] != self.contract.observation_dim:
            raise ValueError("actor observation schema mismatch")
        if not torch.isfinite(local_history).all():
            raise ValueError("nonfinite actor input")
        mean = self.mean_head(self.encoder(local_history))
        return Normal(mean, self.log_std.clamp(-5.0, 0.0).exp().expand_as(mean))

    def forward(self, local_history: torch.Tensor) -> torch.Tensor:
        return self.center + self.radius * self.distribution(local_history).mean.tanh()

    def sample(self, local_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(local_history)
        raw = distribution.sample()
        action = self.center + self.radius * raw.tanh()
        # Stable log(1-tanh(raw)^2), including saturated exploratory draws.
        log_jacobian = log(self.radius) + 2 * (log(2) - raw - nn.functional.softplus(-2 * raw))
        return action, (distribution.log_prob(raw) - log_jacobian).sum(dim=-1)

    def log_prob(self, local_history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(local_history)
        if actions.shape != distribution.mean.shape or not torch.isfinite(actions).all():
            raise ValueError("invalid action batch")
        scaled = (actions - self.center) / self.radius
        if (scaled.abs() > 1 + 1e-6).any():
            raise ValueError("action outside frozen physical bounds")
        raw = torch.atanh(scaled.clamp(-1 + 1e-6, 1 - 1e-6))
        log_jacobian = log(self.radius) + 2 * (log(2) - raw - nn.functional.softplus(-2 * raw))
        return (distribution.log_prob(raw) - log_jacobian).sum(dim=-1)
