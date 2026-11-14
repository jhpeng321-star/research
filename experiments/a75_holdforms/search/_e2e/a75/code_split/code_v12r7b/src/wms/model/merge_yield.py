"""R13Y shared learned V6MERGE virtual-leader constraint (HDVs only).

2 m/s² and 60 m are INITIALIZATIONS from BLOWUP arm Y, not fixed behaviour.
The distance gate's width is the follower's learned IDM minimum gap. Pair and
blocked-merger selection are discrete; selected state and physics stay live.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from wms.physics.idm import IDMParams
from wms.physics.merge_v6 import mainline_merge_yield
from wms.physics.mobil import merge_end_blocked, merge_impatience_level


class LearnedMergeYield(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_decel = nn.Parameter(torch.tensor(math.log(2.0)))
        self.log_distance = nn.Parameter(torch.tensor(math.log(60.0)))

    @property
    def decel_mps2(self) -> Tensor:
        return self.log_decel.exp()

    @property
    def distance_m(self) -> Tensor:
        return self.log_distance.exp()

    def forward(self, kin, present, accel, physics, driver_params, *, is_cav=None,
                merge_wait_s=None, return_influence=False):
        if driver_params is None:
            raise ValueError("learnable mainline yield requires learned driver parameters")
        impatience = merge_impatience_level(merge_wait_s, physics)
        blocked = merge_end_blocked(kin, present, physics, impatience)
        result = mainline_merge_yield(
            kin, present, accel, physics, IDMParams.from_driver_params(driver_params),
            driver_params["decel"], blocked, impatience,
            learned_decel=self.decel_mps2, learned_distance=self.distance_m,
            return_selected=True, return_influence=return_influence)
        bound, selected = result[:2]
        if is_cav is not None:
            selected = selected & ~is_cav.bool()
        bound = torch.where(selected, bound, accel)
        return (bound, selected, result[2]) if return_influence else (bound, selected)
