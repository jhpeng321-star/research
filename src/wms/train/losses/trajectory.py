"""Per-car trajectory terms: Huber position and speed against the recorded rollout."""
from __future__ import annotations

from typing import Optional

from .base import LossContext, LossTerm, TermValue


class _HuberTerm(LossTerm):
    """Step-weighted, valid-count-normalised Huber term logged without its weight."""

    def __init__(self, weight: float) -> None:
        self.weight = float(weight)

    def _huber(self, ctx: LossContext):
        raise NotImplementedError

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        comp = (self._huber(ctx) * ctx.w * ctx.m).sum() / ctx.targets.denom
        return TermValue(self.weight * comp, log=comp)


class PositionLoss(_HuberTerm):
    name = "position"

    def _huber(self, ctx: LossContext):
        return ctx.pos_huber


class SpeedLoss(_HuberTerm):
    name = "speed"

    def _huber(self, ctx: LossContext):
        return ctx.spd_huber
