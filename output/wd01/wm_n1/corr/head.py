"""P5 strategy-independent correction head (long-horizon outcomes).

Fed by the CAV-motion summary (P5 summary.py) and the world's global token
(from the P2/P9 encoder); produces per-tau outcome quantiles and collision
logits, trained with CRPS.  Never receives strategy labels.

The quantile output is monotone by construction: each level is a non-negative
softplus increment over the previous one, so ``q_{k+1} >= q_k`` for all worlds
and targets (an acceptance test pins this).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.losses import pinball

from .summary import CorrConfig


class CorrectionHead(nn.Module):
    """Per-tau quantile + collision head over the CAV-motion summary."""

    def __init__(self, cfg: Optional[CorrConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CorrConfig()
        d = self.cfg.d_model
        self.trunk = nn.Sequential(
            nn.Linear(self.cfg.summary_dim + d, d), nn.GELU())
        self.q = nn.ModuleList([
            nn.Linear(d, self.cfg.n_outcome_targets * self.cfg.n_quantiles)
            for _ in range(self.cfg.n_tau)])
        self.collision = nn.ModuleList([nn.Linear(d, 1)
                                        for _ in range(self.cfg.n_tau)])

    def _quantiles(self, h: Tensor, layer: nn.Linear) -> Tensor:
        raw = layer(h)
        b = raw.shape[0]
        raw = raw.view(b, self.cfg.n_outcome_targets, self.cfg.n_quantiles)
        base = F.softplus(raw[..., :1])
        inc = F.softplus(raw[..., 1:])
        return torch.cat([base, base + torch.cumsum(inc, dim=-1)], dim=-1)

    def forward(self, summary: Tensor, global_token: Tensor) -> dict:
        """``summary [B, summary_dim]`` + ``global_token [B, d_model]`` ->:

        * ``outcome_quantiles``: ``[B, n_tau, n_outcome_targets, n_quantiles]``
        * ``collision_logits``: ``[B, n_tau]``.
        """
        if summary.shape[-1] != self.cfg.summary_dim:
            raise ValueError(
                f"summary must have {self.cfg.summary_dim} dims; got "
                f"{summary.shape[-1]}")
        if global_token.shape[-1] != self.cfg.d_model:
            raise ValueError(
                f"global_token must have {self.cfg.d_model} dims; got "
                f"{global_token.shape[-1]}")
        h = self.trunk(torch.cat([summary, global_token], dim=-1))
        q = torch.stack([self._quantiles(h, layer) for layer in self.q],
                        dim=1)
        collision = torch.stack([layer(h).squeeze(-1)
                                 for layer in self.collision], dim=1)
        return {"outcome_quantiles": q, "collision_logits": collision}

    @staticmethod
    def crps(pred_quantiles: Tensor, target: Tensor,
             levels: Tensor) -> Tensor:
        """Quantile (pinball) CRPS; delegates to ``model.losses.pinball``."""
        lv = torch.as_tensor(levels, dtype=pred_quantiles.dtype,
                             device=pred_quantiles.device)
        if target.dim() == pred_quantiles.dim():
            target = target.squeeze(-1)
        loss = pinball(pred_quantiles, target, lv)
        return 2.0 * loss.mean()
