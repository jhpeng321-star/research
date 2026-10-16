"""Conservative same-lane stopping-distance filter for CAV accelerations."""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Union

import torch
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M

from .contract import ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, CONTROL_DT_S, PHYSICS_DT_S

_BRAKING_MPS2 = 3.0
_MIN_BUMPER_GAP_M = 2.0
_SOFTMIN_TEMPERATURE_MPS2 = 0.05


Ceiling = Union[float, Tensor]


def _upper(x: Tensor, ceiling: Ceiling, lower: float = None) -> Tensor:
    """``x.clamp(lower, ceiling)`` for a float ceiling (the exact legacy op) or a [B,K] tensor."""
    if not torch.is_tensor(ceiling):
        return x.clamp(lower, ceiling) if lower is not None else x.clamp(max=ceiling)
    x = x.clamp(min=lower) if lower is not None else x
    return torch.minimum(x, ceiling.to(device=x.device, dtype=x.dtype))


def accepts_accel_max(fn) -> bool:
    """True when the safety callable takes an ``accel_max`` keyword (or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "accel_max" or p.kind is p.VAR_KEYWORD for p in params)


def with_accel_max(fn):
    """``fn`` if it accepts ``accel_max``; else a wrapper that ignores it (custom safety layers)."""
    if accepts_accel_max(fn):
        return fn

    def call(state, accel, *, accel_max=None):
        del accel_max
        return fn(state, accel)

    return call


def safe_accel(state: Mapping[str, Tensor], accel: Tensor, *,
               accel_max: Ceiling = ACCEL_MAX_MPS2) -> Tensor:
    """Filter [B,K] actions, preserving HDVs and zeroing absent/invalid CAVs.

    ``accel_max`` is the upper actuation ceiling: a float (default: the LEGACY 1.5 cap) or a
    [B,K] tensor (per-slot: policy CAVs vs counterfactual native CAVs).  Callers bounding a
    policy's commands pass the policy's stored ``accel_max`` (ACCELCAP_NOTES.md).

    A nearest same-lane leader may immediately brake at b=3 m/s². Reserve a
    2 m bumper gap, then require follower stopping distance <= gap-2+v_lead²/2b.
    The proposed action is held for CONTROL_DT_S=1 s, followed by braking at b.
    For u=v+a*T >= 0, distance is T*(v+u)/2 + u²/(2b). Solve this quadratic for
    the largest admissible u and hence a. If stopping must occur during the
    action hold, use v²/(-2a) instead, allowing a stop before T (not a fictitious
    negative endpoint speed). No clamping of the measured bumper gap hides an
    overlap. Positions use the physics convention: s_lead-s_ego-length_lead.

    Reserve b*PHYSICS_DT_S²/8 metres for the worst overshoot of trapezoidal
    integration when speed is floored to zero part-way through a physics step.
    Exact stop-time integration also satisfies this more conservative bound.
    Unnormalised log-sum-exp soft-min is <= both proposal and stopping bound;
    the final actuator floor is -3, never a stronger emergency-braking command.

    This is conditional, not a collision guarantee: already overlapping or
    physically infeasible states may require a<-3, so the actuator saturates
    and cannot ensure avoidance. The same can happen when the conservative
    numerical reserve makes a marginal state uncertifiable. The derivation
    assumes nonnegative speeds, fixed lanes/leader identity during the hold,
    and leader braking no stronger than 3. Lane changes, unseen vehicles and
    ramp-end obstacles require separate handling. Neighbour selection, masks
    and saturation are discrete/piecewise; the active bound is differentiable.
    """
    present = state["present"].bool()
    if accel.ndim != 2 or accel.shape != present.shape:
        raise ValueError("accel and present must have matching [B, K] shapes")
    is_cav = state["is_cav"].bool()
    lane = state["lane"]
    valid = present & (lane >= 0)
    active_cav = valid & is_cav
    # Sanitise padding before arithmetic, including arbitrary NaN sentinels.
    s = torch.where(valid, state["s_m"], torch.zeros_like(state["s_m"]))
    v = torch.where(valid, state["v_mps"], torch.zeros_like(state["v_mps"])).clamp_min(0.0)
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
    length = torch.where(valid, length, torch.zeros_like(length))
    proposal = _upper(torch.where(active_cav, accel, torch.zeros_like(accel)), accel_max,
                      ACCEL_MIN_MPS2)
    if accel.shape[-1] == 0:
        return accel.clone()

    ds = s[:, None, :] - s[:, :, None]
    other = ~torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)[None, :, :]
    ahead = (valid[:, :, None] & valid[:, None, :]
             & (lane[:, :, None] == lane[:, None, :]) & (ds >= 0.0) & other)
    # Rank by position, not clamped bumper gap; equal-position overlaps count.
    distance, index = torch.where(ahead, ds, torch.full_like(ds, float("inf"))).min(-1)
    has_leader = torch.isfinite(distance)
    gap = torch.where(has_leader, distance, torch.zeros_like(distance))
    gap = gap - length.gather(-1, index)
    leader_v = v.gather(-1, index)
    b, t = _BRAKING_MPS2, CONTROL_DT_S
    integration_reserve = b * PHYSICS_DT_S**2 / 8.0
    budget = gap - _MIN_BUMPER_GAP_M + leader_v.square() / (2.0 * b)
    budget = budget - integration_reserve

    # Non-stopping branch: u² + b*T*u + b*T*v - 2*b*budget <= 0.
    remaining = (budget - 0.5 * t * v).clamp_min(0.0)
    endpoint_v = torch.sqrt((0.5 * b * t)**2 + 2.0 * b * remaining) - 0.5 * b * t
    rolling_bound = (endpoint_v - v) / t
    # Stopping branch: v²/(-2a) <= budget. A nonpositive budget with
    # positive v deliberately produces an infeasible bound, then saturates.
    stopping_bound = -v.square() / (2.0 * budget.clamp_min(torch.finfo(v.dtype).eps))
    bound = torch.where(budget >= 0.5 * t * v, rolling_bound, stopping_bound)
    bound = _upper(bound, accel_max)
    tau = _SOFTMIN_TEMPERATURE_MPS2
    filtered = -tau * torch.logsumexp(torch.stack((-proposal / tau, -bound / tau)), dim=0)
    filtered = _upper(filtered, accel_max, ACCEL_MIN_MPS2)
    # With no leader, return the bounded proposal exactly (no phantom braking).
    filtered = torch.where(has_leader, filtered, proposal)
    cav_result = torch.where(active_cav, filtered, torch.zeros_like(filtered))
    return torch.where(is_cav, cav_result, accel)
