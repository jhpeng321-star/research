"""Causal per-driver parameter inference for WM-S2.

The encoder consumes only the explicitly supplied pre-t0 observation history.
Parameter deltas are bounded and reliability-shrunk toward vType defaults;
uncertainties describe the unshrunk estimate scale in physical units.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PARAMETER_NAMES = ("v0", "time_headway", "min_gap", "accel", "decel",
                   "politeness", "mobil_threshold", "merge_gap", "merge_urgency")
PARAMETER_BOUNDS = (5.0, 0.8, 2.0, 0.8, 1.5, 0.35, 0.2, 8.0, 0.4)

#: Hidden width of the shared encoder trunk.  The fused embedding handed to the
#: residual net / lane head has width ``2 * DRIVER_ENCODER_HIDDEN + F`` (pooled
#: trunk + projected interaction + last observed feature).
DRIVER_ENCODER_HIDDEN = 48


class DriverEncoder(nn.Module):
    """Attention-pool observed [B,K,H,F] motion plus neighbor interaction cues."""

    def __init__(self, history_features: int = 6, hidden: int = DRIVER_ENCODER_HIDDEN,
                 shrink_prior_steps: float = 8.0, long_history: bool = False):
        super().__init__()
        self.history_features = int(history_features)
        self.shrink_prior_steps = float(shrink_prior_steps)
        self.history_proj = nn.Sequential(nn.Linear(history_features, hidden), nn.Tanh())
        self.query = nn.Parameter(torch.zeros(hidden))
        self.interaction_proj = nn.Sequential(nn.Linear(4, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(hidden * 2 + history_features, hidden),
                                  nn.GELU(), nn.Linear(hidden, 18))
        self.register_buffer("bounds", torch.tensor(PARAMETER_BOUNDS))
        self.long_history = bool(long_history)
        if self.long_history:
            # Created only behind --history-s; legacy RNG/weights stay identical.
            self.history_gru = nn.GRUCell(9, hidden)

    def forward(self, history: Tensor, history_mask: Tensor, interaction: Tensor,
                defaults: Tensor) -> dict[str, Tensor]:
        width = 9 if self.long_history else self.history_features
        if history.ndim != 4 or history.shape[-1] != width:
            raise ValueError("history must be [B,K,H,F] with configured feature width")
        if history_mask.shape != history.shape[:-1] or interaction.shape != (*history.shape[:2], 4):
            raise ValueError("history mask or interaction shape mismatch")
        if defaults.shape != (*history.shape[:2], len(PARAMETER_NAMES)):
            raise ValueError("defaults must be [B,K,9]")
        mask = history_mask.bool()
        if self.long_history:
            safe = torch.where(mask.unsqueeze(-1), history, torch.zeros_like(history))
            h = safe.new_zeros(*safe.shape[:2], self.query.numel())
            sequence = []
            for i in range(safe.shape[2]):
                proposed = self.history_gru(safe[:, :, i].reshape(-1, width),
                                            h.reshape(-1, h.shape[-1])).reshape_as(h)
                h = torch.where(mask[:, :, i, None], proposed, h)
                sequence.append(h)
            encoded = torch.stack(sequence, -2)
            history = safe[..., :self.history_features]
        else:
            encoded = self.history_proj(history)
        score = torch.einsum("bkhf,f->bkh", encoded, self.query) / encoded.shape[-1] ** 0.5
        score = score.masked_fill(~mask, -1.0e4)
        # Avoid NaNs for vehicles with no valid history.
        attention = torch.softmax(score, -1) * mask.to(history.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1.0e-8)
        pooled = torch.einsum("bkh,bkhf->bkf", attention, encoded)
        count = mask.sum(-1, keepdim=True).to(history.dtype)
        last = (history * mask.unsqueeze(-1)).sum(-2) / count.clamp_min(1.0)
        fused = torch.cat((pooled, self.interaction_proj(interaction), last), -1)
        raw = self.head(fused).reshape(*history.shape[:2], 2, len(PARAMETER_NAMES))
        delta = torch.tanh(raw[..., 0, :]) * self.bounds.to(history)
        uncertainty = (F.softplus(raw[..., 1, :]) + 1.0e-4) * self.bounds.to(history)
        shrink = count / (count + self.shrink_prior_steps)
        values = defaults + shrink * delta
        lower = values.new_tensor((5.0, 0.2, 0.5, 0.3, 0.5, 0.0, 0.0, 2.0, 0.1))
        upper = values.new_tensor((60.0, 4.0, 10.0, 5.0, 10.0, 1.0, 2.0, 30.0, 2.0))
        values = torch.maximum(torch.minimum(values, upper), lower)
        return {"values": values, "uncertainty": uncertainty, "shrinkage": shrink.squeeze(-1),
                "attention": attention, "embedding": fused}
