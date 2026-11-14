"""Hybrid native/active policy with a causal analytical initialization.

Native has probability mass. Continuous likelihood is conditional on active;
masked cooldown choices and unused native residuals have no gradient.
"""

from math import log
import torch
from torch import nn
from torch.distributions import Bernoulli, Normal
from env.damping_v2 import SelectableContract


class SelectableActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.contract = SelectableContract()
        self.encoder = nn.Sequential(nn.Linear(89, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mode_head = nn.Linear(64, 1)
        self.mean_head = nn.Linear(64, 1)
        for head in (self.mode_head, self.mean_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.mean_head.bias, torch.atanh(torch.tensor(0.25)).item())
        self.log_std = nn.Parameter(torch.tensor([-2.0]))

    def distributions(self, x):
        if x.shape[-1] != 89 or not torch.isfinite(x).all():
            raise ValueError("invalid actor observation")
        f = x[..., 68:85]
        excited = ((f[..., 8] * 30).abs() > 0.5) | ((x[..., 85] - f[..., 0]).abs() * 30 > 0.5)
        prior = (f[..., 9] > 0.5) & (x[..., 86] > 0.5) & excited
        z = self.encoder(x)
        # This prior is an initialization, not a fixed escalation gate.
        mode = Bernoulli(logits=self.mode_head(z).squeeze(-1) + torch.where(prior, 1.0, -1.0))
        continuous = Normal(self.mean_head(z), self.log_std.clamp(-5, 0).exp())
        allowed = x[..., 88] <= 1e-7
        return mode, continuous, allowed

    def forward(self, x):
        mode, continuous, allowed = self.distributions(x)
        m = ((mode.logits > 0) & allowed).to(x.dtype)
        d = (-0.1 + 0.4 * continuous.mean.tanh()).squeeze(-1) * m
        return torch.stack((m, d), -1)

    def sample(self, x):
        mode, continuous, allowed = self.distributions(x)
        m = mode.sample() * allowed
        d = (-0.1 + 0.4 * continuous.sample().tanh()).squeeze(-1) * m
        actions = torch.stack((m, d), -1)
        return actions, self.log_prob(x, actions)

    def log_prob(self, x, actions):
        mode, continuous, allowed = self.distributions(x)
        if actions.shape != (*x.shape[:-1], 2) or not torch.isfinite(actions).all():
            raise ValueError("invalid hybrid actions")
        m, d = actions[..., 0], actions[..., 1:2]
        if not ((m == 0) | (m == 1)).all() or ((m == 1) & ~allowed).any():
            raise ValueError("invalid or masked mode")
        scaled = (d + 0.1) / 0.4
        if (scaled.abs() > 1 + 1e-6).any():
            raise ValueError("correction outside bounds")
        raw = torch.atanh(scaled.clamp(-1 + 1e-6, 1 - 1e-6))
        jac = log(0.4) + 2 * (log(2) - raw - nn.functional.softplus(-2 * raw))
        return mode.log_prob(m) * allowed + m * (continuous.log_prob(raw) - jac).sum(-1)


class FixedNative(SelectableActor):
    def forward(self, x):
        return torch.zeros((*x.shape[:-1], 2), dtype=x.dtype)


class FixedExpert(SelectableActor):
    """The initial deterministic policy, frozen for comparator evaluation."""

    pass
