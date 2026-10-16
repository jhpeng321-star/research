"""Causal per-driver parameter inference.

The encoder consumes only the explicitly supplied pre-t0 observation history.
Legacy deltas are bounded and reliability-shrunk toward the physics defaults;
physical mode shrinks logit deviations toward a learned shared centre instead.
The uncertainties describe the unshrunk estimate scale in physical units.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from wms.physics.config import PhysicsConfig

PARAMETER_NAMES = ("v0", "time_headway", "min_gap", "accel", "decel",
                   "politeness", "mobil_threshold", "merge_gap", "merge_urgency")
#: Maximum deviation of each parameter from its default (tanh-bounded delta).
PARAMETER_BOUNDS = (5.0, 0.8, 2.0, 0.8, 1.5, 0.35, 0.2, 8.0, 0.4)
#: Legacy physical clipping ranges (also used by the observation updater).
PARAMETER_LOWER = (5.0, 0.2, 0.5, 0.3, 0.5, 0.0, 0.0, 2.0, 0.1)
PARAMETER_UPPER = (60.0, 4.0, 10.0, 5.0, 10.0, 1.0, 2.0, 30.0, 2.0)
#: Physical-mode longitudinal ranges in PARAMETER_NAMES order; v0 is a ratio
#: of the road speed limit. Lane parameters retain their legacy mapping.
PHYSICAL_LOWER = (0.6, 0.4, 0.5, 0.3, 0.8)
PHYSICAL_UPPER = (1.4, 3.0, 5.0, 4.0, 6.0)

#: Hidden width of the shared encoder trunk.  The fused embedding handed to the
#: residual net / lane head has width ``2 * DRIVER_ENCODER_HIDDEN + F`` (pooled
#: trunk + projected interaction + last observed feature).
DRIVER_ENCODER_HIDDEN = 48
#: Channels of a long-history sample (see :data:`wms.model.history.LONG_HISTORY_F`).
_LONG_HISTORY_WIDTH = 9
#: Neighbour-interaction features: leader gap/speed, follower gap/speed.
_INTERACTION_F = 4


class DriverEncoder(nn.Module):
    """Attention-pool observed ``[B, K, H, F]`` motion plus neighbour interaction cues."""

    def __init__(self, history_features: int = 6, hidden: int = DRIVER_ENCODER_HIDDEN,
                 shrink_prior_steps: float = 8.0, long_history: bool = False,
                 driver_param_mode: str = "legacy", physics: PhysicsConfig | None = None):
        super().__init__()
        if driver_param_mode not in ("legacy", "physical"):
            raise ValueError("driver_param_mode must be legacy or physical")
        self.driver_param_mode = driver_param_mode
        self.history_features = int(history_features)
        self.shrink_prior_steps = float(shrink_prior_steps)
        self.history_proj = nn.Sequential(nn.Linear(history_features, hidden), nn.Tanh())
        self.query = nn.Parameter(torch.zeros(hidden))
        self.interaction_proj = nn.Sequential(nn.Linear(_INTERACTION_F, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(hidden * 2 + history_features, hidden),
                                  nn.GELU(), nn.Linear(hidden, 2 * len(PARAMETER_NAMES)))
        self.register_buffer("bounds", torch.tensor(PARAMETER_BOUNDS))
        self.long_history = bool(long_history)
        if self.long_history:
            # Created only for long history so legacy RNG/weights stay identical.
            self.history_gru = nn.GRUCell(_LONG_HISTORY_WIDTH, hidden)
        if self.driver_param_mode == "physical":
            # No new parameters/buffers or RNG draws on the legacy path. Initialise
            # from the same physics defaults as WmSModel, never per-driver labels.
            p = physics or PhysicsConfig()
            lower = torch.tensor(PHYSICAL_LOWER)
            upper = torch.tensor(PHYSICAL_UPPER)
            lower[0] *= p.speed_limit_mps
            upper[0] *= p.speed_limit_mps
            a, b, t, s0, _ = p.idm()
            initial = torch.tensor((p.v0_mps, t, s0, a, b))
            fraction = ((initial - lower) / (upper - lower)).clamp(1.0e-6, 1.0 - 1.0e-6)
            self.centre_logits = nn.Parameter(torch.logit(fraction))
            self.register_buffer("physical_lower", lower)
            self.register_buffer("physical_upper", upper)

    def _physical_values(self, raw: Tensor, shrink: Tensor) -> Tensor:
        """Smooth full-range map; reliability shrinks in logit space.

        At n=0 all drivers equal the learned centre. For any n>0 the full open
        interval remains reachable (unlike shrinking bounded physical deltas,
        which would reintroduce observation-dependent floors). The centre is
        trained through the same prediction-error graph as the encoder head.
        """
        lower, upper = self.physical_lower.to(raw), self.physical_upper.to(raw)
        return lower + (upper - lower) * torch.sigmoid(self.centre_logits + shrink * raw)

    def _encode_long(self, history: Tensor, mask: Tensor) -> Tensor:
        """Masked GRU over the long history; returns ``[B, K, H, hidden]``."""
        safe = torch.where(mask.unsqueeze(-1), history, torch.zeros_like(history))
        h = safe.new_zeros(*safe.shape[:2], self.query.numel())
        sequence = []
        for i in range(safe.shape[2]):
            proposed = self.history_gru(safe[:, :, i].reshape(-1, _LONG_HISTORY_WIDTH),
                                        h.reshape(-1, h.shape[-1])).reshape_as(h)
            h = torch.where(mask[:, :, i, None], proposed, h)
            sequence.append(h)
        return torch.stack(sequence, -2)

    def forward(self, history: Tensor, history_mask: Tensor, interaction: Tensor,
                defaults: Tensor) -> dict[str, Tensor]:
        """Infer driver parameters.

        Returns ``values`` (shrunk, clipped parameters ``[B, K, 9]``),
        ``uncertainty``, ``shrinkage``, ``attention`` and the fused ``embedding``.
        """
        width = _LONG_HISTORY_WIDTH if self.long_history else self.history_features
        if history.ndim != 4 or history.shape[-1] != width:
            raise ValueError("history must be [B,K,H,F] with configured feature width")
        if history_mask.shape != history.shape[:-1] \
                or interaction.shape != (*history.shape[:2], _INTERACTION_F):
            raise ValueError("history mask or interaction shape mismatch")
        if defaults.shape != (*history.shape[:2], len(PARAMETER_NAMES)):
            raise ValueError("defaults must be [B,K,9]")
        mask = history_mask.bool()
        if self.long_history:
            encoded = self._encode_long(history, mask)
            history = torch.where(mask.unsqueeze(-1), history,
                                  torch.zeros_like(history))[..., :self.history_features]
        else:
            encoded = self.history_proj(history)
        score = torch.einsum("bkhf,f->bkh", encoded, self.query) / encoded.shape[-1] ** 0.5
        score = score.masked_fill(~mask, -1.0e4)
        # Weights are zeroed for masked steps so vehicles without history give no NaN.
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
        lower = values.new_tensor(PARAMETER_LOWER)
        upper = values.new_tensor(PARAMETER_UPPER)
        values = torch.maximum(torch.minimum(values, upper), lower)
        if self.driver_param_mode == "physical":
            values = torch.cat((self._physical_values(raw[..., 0, :5], shrink),
                                values[..., 5:]), -1)
        return {"values": values, "uncertainty": uncertainty, "shrinkage": shrink.squeeze(-1),
                "attention": attention, "embedding": fused}
