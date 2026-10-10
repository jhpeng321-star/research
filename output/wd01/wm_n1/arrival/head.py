"""P4 learned arrival head (Idea A step (c)).

Predicts the per-tick entrance arrivals from the entrance counters (the
``counter [B, 2, 30]`` tensor of the frozen featurizer).  Reuses the existing
``model.net.CounterEncoder`` for the counter embedding; the head itself is
distinct from ``model.net.ArrivalHead`` because it is strategy-agnostic and is
consumed directly by the P3 rollout (no plan/wall conditioning).

Outputs (float ``[B, 2]``):
``count_mu``, ``count_log_alpha`` (negative-binomial arrivals),
``speed_mu``, ``speed_log_sigma`` (insertion-speed normal).

Counter causality: the frozen featurizer (``features.counters``) bins every
departure by ``age = t - t_dep`` and restricts counts to departures ``<= t``;
bin 0 is the most recent 10 s window and bin 29 the oldest.  Every counter
value fed to :meth:`forward` is therefore already ``<= t``.  ``forward`` only
ever consumes the most recent observed bins; the optional ``valid_bins`` mask
zeroes any bin the caller marks as not yet observed, so no future count can
be represented at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ArrivalConfig:
    n_entrances: int = 2
    counter_bins: int = 30
    counter_channels: int = 128
    d_model: int = 256
    max_insert_per_entrance: int = 3

    def __post_init__(self) -> None:
        if self.n_entrances != 2:
            raise ValueError("the frozen featurizer has 2 entrance counters")


def past_bin_mask(n_observed: int, *, batch: int = 1, n_entrances: int = 2,
                  counter_bins: int = 30, device=None) -> Tensor:
    """``bool [B, 2, 30]`` selecting the observed (past) counter prefix.

    The most recent ``n_observed`` bins ``0..n_observed-1`` are ``True`` (bin 0
    is the window ending at ``t``); the remaining, older bins are treated as
    not yet observed and are zeroed before the encoder.  ``n_observed`` is
    clamped to ``[0, counter_bins]``.
    """
    n = max(0, min(int(n_observed), int(counter_bins)))
    mask = torch.zeros(batch, n_entrances, counter_bins, dtype=torch.bool,
                       device=device)
    if n:
        mask[:, :, :n] = True
    return mask


class IdeaAArrivalHead(nn.Module):
    """Counter-driven arrival head (Idea A step (c))."""

    def __init__(self, cfg: Optional[ArrivalConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ArrivalConfig()
        from model.net import CounterEncoder, WmN1Config

        self.encoder = CounterEncoder(WmN1Config(
            d_model=self.cfg.d_model,
            counter_channels=self.cfg.counter_channels))
        d = self.cfg.d_model
        self.count = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, self.cfg.n_entrances * 2))
        self.speed = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, self.cfg.n_entrances * 2))

    def forward(self, counter: Tensor, *, valid_bins: Optional[Tensor] = None,
                ) -> dict:
        """``counter [B, 2, 30]`` -> the four ``[B, 2]`` arrival parameters.

        ``counter`` holds departure counts already restricted to times
        ``<= t`` (frozen featurizer).  ``valid_bins`` is an optional
        broadcastable bool mask; bins that are ``False`` (not yet observed)
        are zeroed before the encoder, guaranteeing the head cannot use a
        count ``> t``.
        """
        if counter.dim() != 3 or counter.shape[-1] != self.cfg.counter_bins:
            raise ValueError("counter must be [B, %d, %d]; got %s"
                             % (self.cfg.n_entrances, self.cfg.counter_bins,
                                tuple(counter.shape)))
        x = counter.float()
        if valid_bins is not None:
            vb = valid_bins.to(torch.bool)
            while vb.dim() < x.dim():
                vb = vb.unsqueeze(0)
            x = x * vb.to(x.dtype)

        g = self.encoder(x)                          # [B, d_model]
        b = g.shape[0]
        c = self.count(g).view(b, self.cfg.n_entrances, 2)
        count_mu = F.softplus(c[..., 0])
        count_log_alpha = c[..., 1]
        s = self.speed(g).view(b, self.cfg.n_entrances, 2)
        speed_mu = F.softplus(s[..., 0]) + 1.0
        speed_log_sigma = s[..., 1].clamp(-6.0, 2.0)
        return dict(count_mu=count_mu, count_log_alpha=count_log_alpha,
                    speed_mu=speed_mu, speed_log_sigma=speed_log_sigma)
