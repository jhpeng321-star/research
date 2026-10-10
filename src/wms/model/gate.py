"""Learned monotone gate on the learned corrections."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def monotone_gate(increments: Tensor, k_steps: Tensor | int, max_steps: int) -> Tensor:
    """Gate value ``g(k)`` for ``k`` steps since the last observation.

    ``g(0) == 0`` exactly and ``g`` is non-decreasing: it is the cumulative sum of
    ``softplus(increments)``.  ``k_steps`` is clamped to ``[0, max_steps]``.
    """
    k = torch.as_tensor(k_steps, device=increments.device, dtype=torch.long)
    k = k.clamp(0, max_steps)
    cumulative = F.softplus(increments).cumsum(0)
    flat = k.reshape(-1)
    out = torch.zeros_like(flat, dtype=cumulative.dtype)
    positive = flat > 0
    if bool(positive.any()):
        out[positive] = cumulative[flat[positive] - 1]
    return out.reshape(k.shape)


def saturating_gate(logit_g_max: Tensor, raw_tau: Tensor, k_steps: Tensor | int,
                    max_steps: int) -> Tensor:
    """Learned saturating gate ``sigmoid(a) * (1 - exp(-k / softplus(b)))``.

    ``k_steps`` is measured in control steps and clamped to the trained model's
    maximum gap. ``expm1`` preserves precision near zero; the explicit zero
    branch guarantees ``g(0) == 0`` exactly.
    """
    k = torch.as_tensor(k_steps, device=logit_g_max.device, dtype=logit_g_max.dtype)
    k = k.clamp(0, max_steps)
    g_max = torch.sigmoid(logit_g_max)
    tau = F.softplus(raw_tau)
    values = g_max * -torch.expm1(-k / tau)
    return torch.where(k == 0, torch.zeros_like(values), values)
