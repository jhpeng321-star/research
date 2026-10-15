"""Physics-baseline longitudinal acceleration (IDM, accel-lane end, merge yielding)."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor

from .config import PhysicsConfig
from .idm import IDMParams, idm_accel
from .merge_cal import merge_cal_yield
from .merge_v6 import mainline_merge_yield
from .neighbours import lead_neighbour


def baseline_accel(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                   driver_params: Optional[Mapping[str, Tensor]] = None, *,
                   merge_blocked: Optional[Tensor] = None) -> Tensor:
    """IDM acceleration ``[B, K]`` of every present vehicle (zero when absent).

    With ``driver_params`` the per-driver IDM parameters replace the defaults and
    the result is additionally bounded by each driver's own ``accel``/``decel``.
    With ``emergency_decel_mps2`` set, comfortable braking is relaxed only where
    the relative stopping distance would consume the clearance above ``s0``;
    emergency IDM is bounded by the configured plant limit. The leader is assumed
    to maintain its current speed (the lane-end obstacle has speed zero).
    """
    gap, leader_v = lead_neighbour(kin, present, physics)
    params = (physics.idm_params() if driver_params is None
              else IDMParams.from_driver_params(driver_params))
    accel = idm_accel(kin[..., 4], leader_v, gap, params,
                      accel_min=physics.accel_min_mps2, accel_max=physics.accel_max_mps2)
    if driver_params is not None:
        accel = torch.maximum(accel, -driver_params["decel"])
        accel = torch.minimum(accel, driver_params["accel"])
    if physics.emergency_decel_mps2 is not None:
        # Constant-speed leader: closing-distance stopping requirement relative to s0.
        # Only an insufficient comfortable stopping distance unlocks emergency IDM.
        # This changes the physics baseline, not the learned residual or its gradients.
        closing = (kin[..., 4] - leader_v).clamp_min(0.0)
        clearance = (gap - params.s0).clamp_min(1.0e-6)
        required = closing.square() / (2.0 * clearance)
        comfortable = torch.as_tensor(params.b_comf, dtype=accel.dtype, device=accel.device)
        limit = physics.emergency_decel_mps2
        emergency = idm_accel(kin[..., 4], leader_v, gap, params,
                              accel_min=-limit, accel_max=physics.accel_max_mps2)
        emergency = torch.minimum(emergency, -required).clamp_min(-limit)
        if driver_params is not None:
            emergency = torch.minimum(emergency, driver_params["accel"])
        accel = torch.where(required > comfortable, emergency,
                            torch.maximum(accel, -comfortable)).clamp_min(-limit)
    if physics.merge_cal:
        accel = merge_cal_yield(kin, present, accel, physics)
    if physics.mainline_merge_yield:
        accel = mainline_merge_yield(kin, present, accel, physics, params,
                                     None if driver_params is None else driver_params["decel"],
                                     merge_blocked)
    if physics.emergency_decel_mps2 is not None:
        accel = accel.clamp_min(-physics.emergency_decel_mps2)
    return torch.where(present, accel, torch.zeros_like(accel))
