"""R16 (S-RAMPSPEED): learned LC2013-like speed adaptation of ramp HDVs to the target lane.

SUMO's LC2013 adapts the longitudinal speed of a vehicle that must change lanes
(strategic change, here the ramp merge) to the target-lane leader while it drives along
the acceleration lane, so it reaches a mergeable gap while still moving.  The WM's ramp car
followed only its own lane (leader or the standing lane-end obstacle), drove to the lane end
and merged from standstill (R14_DESIGN.md D2).

For a learned (HDV) slot on the merge lane inside the merge zone (``merge_zone_s0_m <= s <=
merge_zone_s1_m``: the road geometry of the acceleration lane, where lane 1 is adjacent), with
a target-lane leader L (nearest lane-1 vehicle whose front is at or ahead of the ego front;
bumper gap ``g = s_L - len_L - s``, negative while alongside):

    a_v  = max(IDM(v, v_L, g; v0, kappa * T, s0, a, b), -b)     (driver's own learned settings)
    w    = sigmoid(omega) * sigmoid((D - d_end) / s0)             (d_end = merge_zone_s1_m - s)
    out  = a + w * min(a_v - a, 0)

Only ever a deceleration toward the target-lane leader (a virtual-leader bound like R13Y's
yield); no forced merge, timer, threshold on the merge decision or likelihood term: the
first-passage pivot, the 13E secure-gap veto, the overlap veto and the lane-end stop decide
acceptance on the resulting approach state exactly as before.  The three shared scalars
(lookahead ``D``, strength ``omega``, headway scale ``kappa``) are learned; their values in
``ramp_speed_adapt_init.json`` are INITIALIZATIONS fitted on TRAIN-role SUMO rows
(``tools/r16_checks/ramp_adapt_init_fit.py``), not fixed behaviour.  The distance gate's width
is the ego's learned IDM minimum gap (as in R13Y).  Neighbour selection is discrete; the
selected state, the driver settings and the scalars stay differentiable (pathwise adjoint).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from wms.physics.idm import IDMParams, idm_accel

INIT_PATH = Path(__file__).with_name("ramp_speed_adapt_init.json")


def load_init(path: Optional[Path] = None) -> dict[str, float]:
    doc = json.loads(Path(path or INIT_PATH).read_text(encoding="utf-8"))
    values = {k: float(doc[k]) for k in ("lookahead_m", "strength", "headway_scale")}
    if not (values["lookahead_m"] > 0.0 and 0.0 < values["strength"] < 1.0
            and values["headway_scale"] > 0.0 and all(map(math.isfinite, values.values()))):
        raise ValueError(f"invalid ramp speed-adaptation init: {values}")
    return values


def target_lane_leader(kin: Tensor, present: Tensor, physics) -> tuple[Tensor, Tensor, Tensor]:
    """``(has, gap, v_leader)`` of every slot's nearest target-lane vehicle at or ahead of its front.

    Same neighbour convention as the first-passage geometry's lane-1 leader (front positions,
    ties count as leader, bumper gap subtracts the LEADER length, may be negative).
    """
    present = present.bool()
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    onehot = kin[..., 1:4]
    lane = torch.where(onehot.sum(-1) > 0, onehot.argmax(-1), torch.full_like(s, -1, dtype=torch.long))
    target = present & (lane == int(getattr(physics, "merge_target_lane", 1))) & torch.isfinite(s)
    k = s.shape[-1]
    if k == 0:
        return torch.zeros_like(present), torch.zeros_like(s), torch.zeros_like(v)
    safe_s = torch.where(present & torch.isfinite(s), s, torch.zeros_like(s))
    ahead = safe_s.unsqueeze(-2) - safe_s.unsqueeze(-1)        # [B, i, j] = s_j - s_i
    not_self = ~torch.eye(k, dtype=torch.bool, device=kin.device)
    mask = target.unsqueeze(-2) & not_self & (ahead >= 0.0)
    dist, index = torch.where(mask, ahead, torch.full_like(ahead, torch.inf)).min(-1)
    has = torch.isfinite(dist)
    lead_len = torch.gather(torch.where(present, length, torch.zeros_like(length)), -1, index)
    lead_v = torch.gather(torch.where(present, v, torch.zeros_like(v)), -1, index)
    gap = torch.where(has, ahead.gather(-1, index.unsqueeze(-1)).squeeze(-1) - lead_len,
                      torch.zeros_like(dist))
    return has, gap, torch.where(has, lead_v, torch.zeros_like(lead_v))


class LearnedRampSpeedAdapt(nn.Module):
    """Three shared scalars: lookahead ``D`` (m), strength ``sigmoid(omega)``, headway scale."""

    def __init__(self, init: Optional[Mapping[str, float]] = None):
        super().__init__()
        values = {**load_init(), **({} if init is None else dict(init))}
        strength = float(values["strength"])
        self.log_lookahead = nn.Parameter(torch.tensor(math.log(float(values["lookahead_m"]))))
        self.strength_logit = nn.Parameter(torch.tensor(math.log(strength / (1.0 - strength))))
        self.log_headway_scale = nn.Parameter(torch.tensor(math.log(float(values["headway_scale"]))))

    @property
    def lookahead_m(self) -> Tensor:
        return self.log_lookahead.exp()

    @property
    def strength(self) -> Tensor:
        return torch.sigmoid(self.strength_logit)

    @property
    def headway_scale(self) -> Tensor:
        return self.log_headway_scale.exp()

    def candidates(self, kin: Tensor, present: Tensor, physics, learned: Tensor) -> Tensor:
        """Learned slots on the merge lane inside the merge zone (acceleration-lane geometry)."""
        s = kin[..., 0]
        onehot = kin[..., 1:4]
        lane = torch.where(onehot.sum(-1) > 0, onehot.argmax(-1), torch.full_like(s, -1, dtype=torch.long))
        start = float(getattr(physics, "merge_zone_s0_m", 0.0))
        end = float(physics.merge_zone_s1_m)
        return (present.bool() & learned.bool() & (lane == int(physics.merge_lane))
                & torch.isfinite(s) & (s >= start) & (s <= end))

    def forward(self, kin: Tensor, present: Tensor, accel: Tensor, physics,
                driver_params: Mapping[str, Tensor], *, learned: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(adapted accel, selected [B, K], weight w [B, K])``; unselected slots are unchanged."""
        if driver_params is None:
            raise ValueError("ramp speed adaptation requires learned driver parameters")
        has, gap, lead_v = target_lane_leader(kin, present, physics)
        params = IDMParams.from_driver_params(driver_params)
        s0 = torch.as_tensor(params.s0, dtype=kin.dtype, device=kin.device).expand_as(kin[..., 0])
        b_comf = torch.as_tensor(params.b_comf, dtype=kin.dtype, device=kin.device).expand_as(s0)
        v = kin[..., 4]
        selected = (self.candidates(kin, present, physics, learned) & has & torch.isfinite(v)
                    & torch.isfinite(gap) & torch.isfinite(lead_v) & torch.isfinite(s0) & (s0 > 0.0)
                    & torch.isfinite(b_comf) & (b_comf > 0.0) & torch.isfinite(accel))
        # Mask operands BEFORE arithmetic: unselected slots must not poison gradients elsewhere.
        zero = torch.zeros_like(v)
        v_s = torch.where(selected, v, zero)
        lead_s = torch.where(selected, lead_v, zero)
        gap_s = torch.where(selected, gap, torch.ones_like(gap))
        s0_s = torch.where(selected, s0, torch.ones_like(s0))
        b_s = torch.where(selected, b_comf, torch.ones_like(b_comf))
        safe = IDMParams(
            v0=torch.where(selected, torch.as_tensor(params.v0, dtype=kin.dtype).expand_as(v),
                           torch.ones_like(v)),
            time_headway=torch.where(selected, torch.as_tensor(params.time_headway, dtype=kin.dtype)
                                     .expand_as(v), torch.ones_like(v)) * self.headway_scale,
            s0=s0_s,
            a_max=torch.where(selected, torch.as_tensor(params.a_max, dtype=kin.dtype).expand_as(v),
                              torch.ones_like(v)),
            b_comf=b_s, delta=params.delta)
        a_v = idm_accel(v_s, lead_s, gap_s, safe, accel_min=physics.accel_min_mps2,
                        accel_max=physics.accel_max_mps2)
        a_v = torch.maximum(a_v, -b_s)
        d_end = float(physics.merge_zone_s1_m) - torch.where(selected, kin[..., 0], zero)
        weight = self.strength * torch.sigmoid((self.lookahead_m - d_end) / s0_s)
        accel_s = torch.where(selected, accel, zero)
        adapted = accel_s + weight * torch.minimum(a_v - accel_s, zero)
        weight = torch.where(selected, weight, zero)
        return torch.where(selected, adapted, accel), selected, weight
