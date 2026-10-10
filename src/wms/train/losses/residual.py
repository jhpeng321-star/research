"""Residual-size terms: raw-residual L2 penalty and the hybrid identifiability guard."""
from __future__ import annotations

from typing import Optional

from torch import Tensor

from .base import LossContext, LossTerm, TermValue


def residual_mag_loss(raw: Tensor, valid: Tensor, weight: float) -> Tensor:
    """Weighted mean-square of the raw (pre-gate) residual over valid slots."""
    selected = raw.masked_select(valid)
    return float(weight) * selected.square().sum() / max(selected.numel(), 1)


class ResidualMagnitudeLoss(LossTerm):
    """Chunk share of one unweighted, whole-batch L2 mean of the raw residual."""

    name = "residual_mag"

    def __init__(self, weight: float) -> None:
        self.weight = float(weight)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        raw = ctx.raw_residual
        if raw is None:
            raise ValueError("residual-mag loss requires model raw_residual_mps2")
        n_valid = int(ctx.valid_aux.sum())
        value = (residual_mag_loss(raw, ctx.valid_aux, self.weight) * n_valid
                 / max(ctx.targets.aux_count, 1))
        return TermValue(value, valid_count=n_valid)


class ResidualPenalty(LossTerm):
    """Hybrid-mode guard: mean gated-residual magnitude so the encoder is not absorbed."""

    name = "residual_penalty"
    shareable = False

    def __init__(self, weight: float) -> None:
        self.weight = float(weight)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        magnitude = ctx.pred.get("residual_magnitude")
        count = int(ctx.pred.get("residual_count", 0))
        if magnitude is None or count <= 0:
            return None
        return TermValue(self.weight * magnitude / count)
