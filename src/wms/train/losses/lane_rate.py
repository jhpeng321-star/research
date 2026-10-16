"""Lane-change-rate guard: pull the predicted per-HDV-step change rate to the recorded one."""
from __future__ import annotations

from typing import Optional

from torch import Tensor

from .base import LossContext, LossTerm, TermValue


def lane_rate_loss(probabilities: Tensor, valid: Tensor, weight: float,
                   target: float = 0.0123) -> Tensor:
    """Weighted squared deviation of the mean change probability from ``target``."""
    selected = probabilities.masked_select(valid)
    if selected.numel() == 0:
        return selected.sum() * 0.0
    return float(weight) * (selected.mean() - float(target)).square()


class LaneRateLoss(LossTerm):
    """Whole-batch rate loss differentiated chunkwise through a reference rate.

    Each BPTT chunk receives the derivative of the single whole-batch rate loss
    (not a mean of chunkwise squares): the reference rate comes from a gradient-free
    replay of the same unroll.  Without a reference (forward-only evaluation) the
    term is added once per unroll from the realised rate by :meth:`final`.
    """

    name = "lane_rate"

    def __init__(self, weight: float, target: float) -> None:
        self.weight = float(weight)
        self.target = float(target)

    def __call__(self, ctx: LossContext) -> Optional[TermValue]:
        reference = ctx.rate_reference
        if reference is None:
            return None
        aux_count = max(ctx.targets.aux_count, 1)
        mass = ctx.change_probabilities.masked_select(ctx.valid_aux).sum()
        derivative = 2 * self.weight * (reference - self.target) * mass / aux_count
        value = (self.weight * (reference - self.target) ** 2
                 * int(ctx.valid_aux.sum()) / aux_count)
        return TermValue(derivative - derivative.detach() + value)

    def final(self, predicted_rate: float, aux_count: int) -> float:
        """Realised-rate term for passes without a reference rate."""
        return self.weight * (predicted_rate - self.target) ** 2 if aux_count else 0.0

