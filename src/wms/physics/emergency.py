"""Differentiable ballistic safe-follow envelope (R13E, opt-in).

No driver types or fitted behavioural cutoffs: both braking capabilities are the
plant limit, the decision interval is the physics step, and the standstill gap
is the existing inferred IDM parameter.  This mirrors SUMO's stopping-distance
safe-speed principle, not its discrete Euler implementation.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import PhysicsConfig
from .neighbours import lane_of


def safe_accel(v: Tensor, leader_v: Tensor, gap: Tensor, nominal_accel: Tensor,
               min_gap: Tensor | float, *, dt: float, emergency_decel: float) -> Tensor:
    """Smooth upper bound on acceleration, with the existing plant lower bound.

    If u is next-step speed, require
      dt*(v+u)/2 + u**2/(2*B) + dt*u/2 <= gap-s0 + vl**2/(2*B).
    The last term is a half-step reaction distance: it bounds the excess of the
    unchanged, speed-clipped trapezoidal plant's stopping sum over the ballistic
    integral (including the final partial stopping tick).  No integrator change
    or empirically chosen safety margin is needed.
    Also bound this step's displacement using the leader's exact ballistic stop
    trajectory.  The latter matters when the leader stops within this step.
    States outside the feasible braking envelope receive best-effort full braking.
    """
    b = emergency_decel
    half_step_speed = b * dt
    budget = gap - min_gap + leader_v.square() / (2.0 * b) - v * dt / 2.0
    # At/below this discriminant no nonnegative next speed is feasible.  Clamping
    # here avoids the sqrt(0) derivative without changing any positive solution.
    root = (half_step_speed**2 + 2.0 * b * budget).clamp_min(half_step_speed**2)
    stop_speed = root.sqrt() - half_step_speed
    leader_next = (leader_v - b * dt).clamp_min(0.0)
    leader_travel = (leader_v.square() - leader_next.square()) / (2.0 * b)
    step_speed = 2.0 * (gap - min_gap + leader_travel) / dt - v
    # The only smoothing scale is numerical: sqrt(machine epsilon) times the
    # plant's one-step speed change (speed), or plant deceleration (acceleration).
    width = math.sqrt(torch.finfo(v.dtype).eps) * b
    speed_width = width * dt
    upper_speed = -speed_width * torch.logaddexp(-stop_speed / speed_width,
                                                -step_speed / speed_width)
    upper_accel = (upper_speed.clamp_min(0.0) - v) / dt
    result = -width * torch.logaddexp(-nominal_accel / width, -upper_accel / width)
    return result.clamp_min(-b)


def _upper_accel(v: Tensor, leader_v: Tensor, gap: Tensor, min_gap, *, dt: float,
                 own_decel, leader_decel, width_decel: float) -> Tensor:
    """Acceleration upper bound of :func:`safe_accel` with separate planning decelerations
    for the ego (``own_decel``) and its leader (``leader_decel``), scalars or ``[B,K]``.
    The smoothing width is the numerical scale of ``width_decel`` (the plant limit)."""
    half_step_speed = own_decel * dt
    budget = gap - min_gap + leader_v.square() / (2.0 * leader_decel) - v * dt / 2.0
    hs2 = half_step_speed * half_step_speed
    root = torch.maximum(hs2 + 2.0 * own_decel * budget, torch.as_tensor(hs2, dtype=v.dtype, device=v.device))
    stop_speed = root.sqrt() - half_step_speed
    leader_next = (leader_v - leader_decel * dt).clamp_min(0.0)
    leader_travel = (leader_v.square() - leader_next.square()) / (2.0 * leader_decel)
    step_speed = 2.0 * (gap - min_gap + leader_travel) / dt - v
    speed_width = math.sqrt(torch.finfo(v.dtype).eps) * width_decel * dt
    upper_speed = -speed_width * torch.logaddexp(-stop_speed / speed_width,
                                                -step_speed / speed_width)
    return (upper_speed.clamp_min(0.0) - v) / dt


def safe_accel_driver_b(v: Tensor, leader_v: Tensor, gap: Tensor, nominal_accel: Tensor,
                        min_gap: Tensor | float, *, dt: float, emergency_decel: float,
                        comfort_decel: Tensor, leader_comfort_decel: Tensor) -> Tensor:
    """LOOP14 ``physics.emergency_onset = "driver_b"``: SUMO-like onset of the envelope.

    SUMO's maximumSafeFollowSpeed plans with the vehicle's normal deceleration (and the
    leader's) and brakes harder only when that is unavoidable.  Here the safe speed is planned
    with each driver's LEARNED comfortable deceleration b (ego: own b, leader stop: the
    leader's b); no new constant.  When the comfortable plan would itself need more than b
    this step, the vehicle brakes with b, or harder only if the plant-limit plan (13E envelope,
    B = -accel_min) demands it.  The plant limit stays the hard floor:

        U = U_b                      if U_b >= -b
            min(-b, U_plant)         otherwise          (continuous at U_b = -b)
        a = smooth_min(nominal, U).clamp_min(-B)
    """
    b_plant = emergency_decel
    upper_b = _upper_accel(v, leader_v, gap, min_gap, dt=dt, own_decel=comfort_decel,
                           leader_decel=leader_comfort_decel, width_decel=b_plant)
    upper_plant = _upper_accel(v, leader_v, gap, min_gap, dt=dt, own_decel=b_plant,
                               leader_decel=b_plant, width_decel=b_plant)
    bound = torch.where(upper_b >= -comfort_decel, upper_b,
                        torch.minimum(-comfort_decel, upper_plant))
    width = math.sqrt(torch.finfo(v.dtype).eps) * b_plant
    result = -width * torch.logaddexp(-nominal_accel / width, -bound / width)
    return result.clamp_min(-b_plant)


def leader_inputs(kin: Tensor, present: Tensor,
                  physics: PhysicsConfig, *, return_index: bool = False):
    """True (unfloored) bumper gaps and real-leader mask, including the lane end.

    The old IDM's free-road surrogate and 0.3 m floor must not create a phantom
    braking target or hide an actual overlap in this envelope.
    """
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    lane = lane_of(kin)
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)
    valid = (present.unsqueeze(-2) & (lane.unsqueeze(-2) == lane.unsqueeze(-1))
             & (ds > 0))
    gaps = torch.where(valid, ds - length.unsqueeze(-2), torch.full_like(ds, float("inf")))
    gap, index = gaps.min(-1)
    has = torch.isfinite(gap)
    leader_v = torch.gather(v, -1, index)
    end_gap = physics.merge_zone_s1_m - s
    obstacle = (lane == physics.merge_lane) & (end_gap < gap)
    if physics.merge_cal:
        obstacle = obstacle & (end_gap <= physics.end_obstacle_range_m)
    has = present & (has | obstacle)
    gap = torch.where(obstacle, end_gap, gap)
    leader_v = torch.where(obstacle, torch.zeros_like(leader_v), leader_v)
    # Finite placeholders keep unselected branches' backward paths well-defined.
    out = (torch.where(has, gap, torch.zeros_like(gap)),
           torch.where(has, leader_v, torch.zeros_like(leader_v)), has)
    if return_index:
        # Leader slot (the lane-end obstacle has no leader slot: -1).
        return (*out, torch.where(obstacle | ~has, torch.full_like(index, -1), index))
    return out
