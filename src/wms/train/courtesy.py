"""V10COURTESY training support: weight the courtesy-yield events.

The residual can only learn a follower's yield deceleration if the trajectory loss
sees the event.  This helper marks lane-1 followers within ``behind_m`` of a
lane-2 car whose distance to the acceleration-lane end is below ``dend_m``.  The
boost multiplies the ordinary position/speed Huber terms (``loss.courtesy_weight``,
default 1.0 = off, bit-identical).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def courtesy_follower_mask(states: Tensor, present: Tensor, physics,
                           *, behind_m: float = 40.0, dend_m: float = 40.0) -> Tensor:
    """``[B, T, K]`` bool: lane-1 follower within ``behind_m`` of a ramp car near the end.

    ``states`` is the recorded kinematics ``[B, T, K, 7]`` at the step START, i.e.
    exactly the state the residual predicts from.  "Behind" is the centre-to-centre
    longitudinal distance (0 <= s_ramp - s_follower <= behind_m); the ramp car's own
    length is small relative to the 40 m window.
    """
    kin = states.float()
    s = kin[..., 0]
    onehot = kin[..., 1:4]
    lane = onehot.argmax(dim=-1)
    lane = torch.where(onehot.sum(dim=-1) <= 0, torch.full_like(lane, -1), lane)
    present = present.bool()
    ramp = present & (lane == int(getattr(physics, "merge_lane", 2)))
    near_end = ramp & ((float(getattr(physics, "merge_zone_s1_m", 75.44)) - s) < float(dend_m))
    follower = present & (lane == 1)
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)        # [B, T, follower, ramp] = s_r - s_f
    behind = (ds >= 0.0) & (ds <= float(behind_m))
    exposed = (behind & near_end.unsqueeze(-2)).any(dim=-1)
    return follower & exposed


def courtesy_follower_boost(states: Tensor, present: Tensor, physics, weight: float,
                            *, behind_m: float = 40.0, dend_m: float = 40.0,
                            ) -> Optional[Tensor]:
    """``[B, T, K]`` multiplier (``weight`` on courtesy followers, else 1) or None.

    Returns ``None`` when the weight is off, so default training is untouched.
    """
    if physics is None or float(weight) <= 1.0:
        return None
    s = states.float()[..., 0]
    mask = courtesy_follower_mask(states, present, physics, behind_m=behind_m, dend_m=dend_m)
    return torch.where(mask, torch.as_tensor(float(weight), dtype=s.dtype, device=s.device),
                       torch.ones_like(s))
