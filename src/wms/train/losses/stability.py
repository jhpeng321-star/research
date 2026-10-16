"""Once-per-unroll terms: residual-contractivity regulariser and conservation violation.

Both are OFF in the round-3 recipe (change-log #7) and kept for scoring/replaying
older configurations.  The conservation violation is a non-differentiable
bookkeeping loss (it is only added to the *logged* total), the stability
regulariser is back-propagated immediately after the chunk losses.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.model.stability import stability_regulariser

from .base import TermValue, UnrollContext, UnrollTerm


def kin_from_state(state: Mapping[str, Tensor]) -> Tensor:
    """Kinematics tensor ``[B, K, 7]`` rebuilt from the scalar state fields.

    The scalar fields are authoritative after a rollout step or a state-noise
    perturbation; ``veh_kin`` is only used when they are absent.
    """
    if not all(key in state for key in ("s_m", "v_mps", "a_mps2", "lane")) \
            and "veh_kin" in state:
        return state["veh_kin"]
    lane = state["lane"].long()
    one_hot = F.one_hot(lane.clamp_min(0), num_classes=3).to(state["s_m"].dtype)
    one_hot = one_hot * (lane >= 0).unsqueeze(-1).to(one_hot.dtype)
    length = state.get("length_m", torch.full_like(state["s_m"], 5.0))
    return torch.cat((state["s_m"].unsqueeze(-1), one_hot, state["v_mps"].unsqueeze(-1),
                      state["a_mps2"].unsqueeze(-1), length.unsqueeze(-1)), dim=-1)


class StabilityLoss(UnrollTerm):
    """Penalise residual amplification above ``margin`` (identity-safe -> 0)."""

    name = "stability"
    differentiable = True

    def __init__(self, weight: float, *, margin: float, probes: int) -> None:
        self.weight = float(weight)
        self.margin = float(margin)
        self.probes = int(probes)

    def __call__(self, ctx: UnrollContext) -> Optional[TermValue]:
        stability = stability_regulariser(
            ctx.model.residual_net, kin_from_state(ctx.initial), ctx.present0.bool(),
            margin=self.margin, n_probes=self.probes)
        return TermValue(self.weight * stability, log=stability)


class ConservationLoss(UnrollTerm):
    """Mean squared conservation violation of the rolled-out presence masks."""

    name = "conservation"
    differentiable = False

    def __init__(self, weight: float) -> None:
        self.weight = float(weight)

    def __call__(self, ctx: UnrollContext) -> Optional[TermValue]:
        if ctx.conservation is None:
            return None
        loss = ctx.conservation.loss().to(ctx.targets.states.device)
        return TermValue(self.weight * loss, log=loss)
