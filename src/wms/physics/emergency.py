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


def leader_inputs(kin: Tensor, present: Tensor,
                  physics: PhysicsConfig) -> tuple[Tensor, Tensor, Tensor]:
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
    return (torch.where(has, gap, torch.zeros_like(gap)),
            torch.where(has, leader_v, torch.zeros_like(leader_v)), has)
