"""State-only global token for the P5 correction head.

The correction head consumes ``(summary, global_token)``; the v1 pipeline fed
it a zero token.  This module builds a learned token from the t0 STATE only:
the masked mean and max over the vehicle-kinematics slots plus the entrance
counter features.  It accepts no CAV flag, action/strategy or outcome channel,
so a correction-head label-leak test can perturb those without changing the
token (and hence the head output).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

#: kinematics-only slot width (``hdv.config.KIN_F`` / ``features.idea_a``).
KIN_F = 7
#: flattened entrance-counter width (``2 * COUNTER_BINS``).
COUNTER_F = 60


class StateGlobalToken(nn.Module):
    """``veh_kin [B,K,KIN_F]`` + ``counter [B,2,30]`` -> ``[B,d_model]``.

    Masked mean + max pooling over the K slots feeds a small MLP; the entrance
    counters feed a second MLP; the two are concatenated and projected to
    ``d_model`` (the correction head's token width).  Padded/absent slots are
    excluded from the pooling and an all-masked world yields a zero vector.
    """

    def __init__(self, d_model: int = 256, *, kin_f: int = KIN_F,
                 counter_f: int = COUNTER_F, hidden: Optional[int] = None,
                 counter_bins: int = 30, counter_ch: int = 2) -> None:
        super().__init__()
        hidden = int(hidden) if hidden is not None else max(16, int(d_model))
        self.d_model = int(d_model)
        self.kin_f = int(kin_f)
        self.counter_ch = int(counter_ch)
        self.counter_bins = int(counter_bins)
        self.counter_f = int(counter_f)
        self.kin_mlp = nn.Sequential(
            nn.Linear(2 * self.kin_f, hidden), nn.GELU(),
            nn.Linear(hidden, self.d_model))
        self.counter_mlp = nn.Sequential(
            nn.Linear(self.counter_f, hidden), nn.GELU())
        self.out = nn.Sequential(
            nn.Linear(self.d_model + hidden, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, self.d_model))

    def _pool(self, kin: Tensor, kin_mask: Tensor) -> Tensor:
        mask = kin_mask.to(torch.bool)
        m = mask.unsqueeze(-1).to(kin.dtype)
        denom = m.sum(dim=1).clamp(min=1.0)
        mean = (kin * m).sum(dim=1) / denom
        neg = torch.finfo(kin.dtype).min
        filled = torch.where(mask.unsqueeze(-1), kin,
                             torch.full_like(kin, neg))
        mx = filled.amax(dim=1)
        has = mask.any(dim=1, keepdim=True)
        mx = torch.where(has, mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=-1)

    def forward(self, kin: Tensor, kin_mask: Tensor,
                counter: Optional[Tensor] = None) -> Tensor:
        if kin.dim() != 3 or kin.shape[-1] != self.kin_f:
            raise ValueError(
                f"kin must be [B,K,{self.kin_f}]; got {tuple(kin.shape)}")
        if kin_mask.shape != kin.shape[:2]:
            raise ValueError(
                f"kin_mask must be [B,K]={tuple(kin.shape[:2])}; got "
                f"{tuple(kin_mask.shape)}")
        pooled = self._pool(kin, kin_mask)
        p = self.kin_mlp(pooled)
        if counter is None:
            c = p.new_zeros(p.shape[0], self.counter_f)
        else:
            c = counter.reshape(counter.shape[0], -1)
            if c.shape[-1] != self.counter_f:
                raise ValueError(
                    f"counter must flatten to {self.counter_f}; got "
                    f"{c.shape[-1]}")
        q = self.counter_mlp(c)
        return self.out(torch.cat([p, q], dim=-1))


__all__ = ["StateGlobalToken", "KIN_F", "COUNTER_F"]
