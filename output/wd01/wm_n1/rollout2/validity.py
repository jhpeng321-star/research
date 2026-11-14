"""P3 validity layer + D1 position integration.

The D1 fix (branch ``zcode/wmn1d1``, model/rollout.py) integrates position
with the clipped acceleration using the trapezoidal rule on speed::

    v_new = max(0, v + a * dt)
    s_new = s + 0.5 * (v + v_new) * dt

This is exact for a constant ``a`` when the speed does not clamp, is stable at
the 0.5 s tick and never moves a vehicle backwards at rest.  It replaces the
v1 ``s_new = s.clone()`` defect (model/rollout.py:84) and the baseline
ballistic rule ``s + v_new*dt`` (partA_v2g/engine.py:661).

Integrator switch (INTERFACES_V2 Mismatch 5).  ``partA_v2g`` uses the
ballistic-on-new-speed rule ``s_new = s + v_new*dt`` (the measured SUMO
integrator form), which is NOT the trapezoid.  The two agree only at
``v_new == v``.  Per-step difference::

    s_d1 - s_v2g = 0.5 * (v - v_new) * dt

(<= 0 when accelerating, >= 0 when braking; <= 0.375 m at the a-clip limit
and dt = 0.5).  ``project_rollout_step(integrator=...)`` selects the rule:

* ``"d1"``  -- the Idea A default (trapezoidal, ``integrate_d1``);
* ``"v2g"`` -- the like-for-like G1 mode (``integrate_ballistic``) that
  reproduces ``partA_v2g.engine`` exactly.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from model.constants import A_MAX, A_MIN, N_LANES
from model.rollout import ValidityCounts  # noqa: F401  (reused, not edited)

INTEGRATORS = ("d1", "v2g")


@torch.no_grad()
def integrate_d1(s_m: Tensor, v_mps: Tensor, a_mps2: Tensor, *,
                 dt_s: float = 0.5) -> tuple[Tensor, Tensor]:
    """D1 kinematic step -> ``(s_new, v_new)`` (all ``[..., ]``, float).

    Trapezoidal on speed: ``v_new = max(0, v + a*dt)`` and
    ``s_new = s + 0.5*(v + v_new)*dt``.  The frozen Idea A integrator.
    """
    v_new = (v_mps + a_mps2 * dt_s).clamp(min=0.0)
    s_new = s_m + 0.5 * (v_mps + v_new) * dt_s
    return s_new, v_new


@torch.no_grad()
def integrate_d1_vnew(s_m: Tensor, v_mps: Tensor, v_new: Tensor, *,
                      dt_s: float = 0.5) -> Tensor:
    """Trapezoidal position step from an ALREADY-clipped ``v_new``.

    ``s_new = s + 0.5*(v + v_new)*dt`` with no speed re-clamp, so a caller
    that computed ``v_new`` exactly (e.g. the controller execution rule)
    is integrated without a division/multiplication round-trip.
    """
    return s_m + 0.5 * (v_mps + v_new) * dt_s


@torch.no_grad()
def integrate_ballistic(s_m: Tensor, v_new: Tensor, *,
                        dt_s: float = 0.5) -> Tensor:
    """The partA_v2g integrator (Mismatch 5): ``s_new = s + v_new*dt``.

    ``v_new`` must already be floored at 0.  Selected by
    ``integrator="v2g"`` so the G1 zero-residual comparison is like for
    like; the Idea A default remains ``integrate_d1``.
    """
    return s_m + v_new * dt_s


def _lane_targets(lane: Tensor, lane_class: Tensor):
    """(target lane, is-change-attempt) per vehicle from the 3-way lane
    class {0 stay, 1 left, 2 right} (same mapping as model.rollout)."""
    tgt = lane.clone()
    attempt = torch.zeros_like(lane, dtype=torch.bool)
    left = lane_class == 1
    right = lane_class == 2
    ok_left = left & (lane == 1)                  # main_right -> main_left
    tgt = torch.where(ok_left, torch.zeros_like(tgt), tgt)
    attempt |= ok_left
    r0 = right & (lane == 0)                      # main_left -> main_right
    r1 = right & (lane == 1)                      # main_right -> accel
    r2 = right & (lane == 2)                      # accel -> main_right (merge)
    tgt = torch.where(r0, torch.ones_like(tgt), tgt)
    tgt = torch.where(r1, torch.full_like(tgt, 2), tgt)
    tgt = torch.where(r2, torch.ones_like(tgt), tgt)
    attempt |= r0 | r1 | r2
    return tgt, attempt


def _overlap_blocked(s_i: float, s_j: float, len_i: float, len_j: float
                     ) -> bool:
    return abs(s_i - s_j) < 0.5 * (len_i + len_j)


@torch.no_grad()
def project_same_lane_gaps(s_m: Tensor, lane: Tensor, v_mps: Tensor,
                           length_m: Tensor, present: Tensor, *,
                           dt_s: float = 0.5,
                           n_lanes: int = N_LANES,
                           counts: Optional[ValidityCounts] = None,
                           ) -> tuple[Tensor, Tensor, ValidityCounts]:
    """Same-lane rear-bumper gap projection >= 0 (design 2.5 / v1 rule).

    Per lane, front-to-back: a follower that would overlap its leader is
    placed at the contact limit and its speed is reduced so the implied
    displacement stays feasible (exactly model.rollout.project_step).
    Returns ``(s, v, counts)``; ``counts.gap_projection`` is incremented.
    """
    counts = counts if counts is not None else ValidityCounts()
    s_out = s_m.clone()
    v_out = v_mps.clone()
    b_n, k_n = s_out.shape
    for b in range(b_n):
        for ln in range(n_lanes):
            idx = [i for i in range(k_n)
                   if bool(present[b, i]) and int(lane[b, i]) == ln]
            idx.sort(key=lambda i: float(s_out[b, i]), reverse=True)
            for front, back in zip(idx[:-1], idx[1:]):
                min_gap = 0.5 * (float(length_m[b, front])
                                 + float(length_m[b, back]))
                limit = float(s_out[b, front]) - min_gap
                if float(s_out[b, back]) > limit:
                    old_s = float(s_out[b, back])
                    s_out[b, back] = limit
                    vmax = max(0.0, (limit - old_s) / dt_s
                               + float(v_out[b, back]))
                    v_out[b, back] = min(float(v_out[b, back]), vmax)
                    counts.gap_projection += 1
    return s_out, v_out, counts


@torch.no_grad()
def project_rollout_step(s_m: Tensor, lane: Tensor, v_mps: Tensor,
                         length_m: Tensor, present: Tensor, a_mps2: Tensor,
                         lane_class: Tensor, *, dt_s: float = 0.5,
                         n_lanes: int = N_LANES,
                         integrator: str = "d1",
                         project_gaps: bool = True,
                         counts: Optional[ValidityCounts] = None,
                         obstacle: Optional[Tensor] = None,
                         ) -> tuple[Tensor, Tensor, Tensor,
                                    Tensor, Tensor,
                                    ValidityCounts]:
    """Lane-bound + non-overlap validity projection on top of D1.

    Order (design 2.5): clip acceleration to ``[-9, +3]`` (counted), apply
    the integration rule (counted speed clips; ``integrator="d1"``
    trapezoidal -- the Idea A default -- or ``integrator="v2g"`` ballistic
    for the like-for-like G1 comparison, see the module docstring), accept
    a lane change only at an in-bounds lane with a non-negative target
    bumper gap (counted rejects), then project same-lane rear bumper gaps
    >= 0 when ``project_gaps`` (counted).  ``project_gaps=False`` mirrors
    the partA_v2g "never repair" semantics for the v2g mode.

    Returns ``(s_new, lane_new, v_new, a_used, lane_class_used, counts)``.
    """
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {INTEGRATORS}, "
                         f"got {integrator!r}")
    counts = counts if counts is not None else ValidityCounts()
    a = a_mps2.clamp(A_MIN, A_MAX)
    counts.accel_clip += int((a_mps2 != a).sum().item())
    v_new = v_mps + a * dt_s
    counts.speed_clip += int((v_new < 0).sum().item())
    v_new = v_new.clamp(min=0.0)
    a_used = torch.where(a >= 0, a, (v_new - v_mps) / dt_s)
    if integrator == "d1":
        s_new = integrate_d1_vnew(s_m, v_mps, v_new, dt_s=dt_s)
    else:
        s_new = integrate_ballistic(s_m, v_new, dt_s=dt_s)

    lane_new = lane.clone()
    b_n, k_n = lane.shape
    for b in range(b_n):
        work = lane_new[b]
        tgt, attempt = _lane_targets(work, lane_class[b])
        for i in range(k_n):
            if not bool(present[b, i]) or int(lane_class[b, i]) == 0:
                continue
            if obstacle is not None and bool(obstacle[b, i]):
                continue                    # INC1: obstacle never lane-changes
            if not bool(attempt[i]):
                # a non-stay class with no valid target from this lane
                counts.lane_reject += 1
                continue
            tgt_lane = int(tgt[i])
            if tgt_lane < 0 or tgt_lane >= n_lanes:
                counts.lane_reject += 1
                continue
            blocked = False
            for j in range(k_n):
                if j == i or not bool(present[b, j]) \
                        or int(work[j]) != tgt_lane:
                    continue
                if _overlap_blocked(float(s_new[b, i]), float(s_new[b, j]),
                                    float(length_m[b, i]),
                                    float(length_m[b, j])):
                    blocked = True
                    break
            if blocked:
                counts.lane_reject += 1
            else:
                work[i] = tgt_lane
    if project_gaps:
        s_new, v_new, counts = project_same_lane_gaps(
            s_new, lane_new, v_new, length_m, present, dt_s=dt_s,
            n_lanes=n_lanes, counts=counts)
    if obstacle is not None:
        obst = obstacle.to(torch.bool) & present
        s_new = torch.where(obst, s_m, s_new)
        v_new = torch.where(obst, torch.zeros_like(v_new), v_new)
        a_used = torch.where(obst, torch.zeros_like(a_used), a_used)
        lane_new = torch.where(obst, lane, lane_new)
    return s_new, lane_new, v_new, a_used, lane_class.clone(), counts
