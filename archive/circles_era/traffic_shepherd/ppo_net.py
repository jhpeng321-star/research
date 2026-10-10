"""Masked actor-critic MLP for the RSU PPO pilot (torch CPU).

One categorical head over Discrete(2) with a phase-dependent action meaning
(WAIT/ACTIVATE when inactive, RELEASE/CONTINUE during HOLD). Unavailable
actions are masked to -inf BEFORE the categorical is constructed, so both
sampling and the stored log-probabilities use the same mask. No second
learned feasibility network exists; the analytical admission stays
authoritative at execution.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical

from . import ppo_contract as C


def masked_logits(logits, mask):
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    if mask_t.ndim == 1:
        mask_t = mask_t.unsqueeze(0)
    if not mask_t.any(dim=-1).all():
        raise ValueError("action mask must keep at least one action per row")
    return logits.masked_fill(~mask_t, float("-inf"))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=None, hidden=None):
        super().__init__()
        obs_dim = obs_dim or C.OBS_DIM
        hidden = hidden or C.PPO_CONFIG["hidden"]
        layers, last = [], obs_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        self.body = nn.Sequential(*layers)
        self.pi_head = nn.Linear(last, 2)
        self.value_head = nn.Linear(last, 1)

    def value(self, obs):
        return self.value_head(self.body(obs)).squeeze(-1)

    def distribution(self, obs, mask):
        logits = masked_logits(self.pi_head(self.body(obs)), mask)
        return Categorical(logits=logits)

    def evaluate(self, obs, mask, actions):
        distribution = self.distribution(obs, mask)
        return (distribution.log_prob(actions), distribution.entropy(),
                self.value(obs))


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


def seed_everything(seed):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
