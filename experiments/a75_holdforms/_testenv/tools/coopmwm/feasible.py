"""Registered actuator feasibility of a planned acceleration (null-diagnosis fix).

The exact plant erases plan accelerations that cannot convert into motion
(NULL_DIAGNOSIS.md sections 3 and 6). This module mirrors the two registered
clip layers in pure arithmetic, CPU-only, with no SUMO and no hand-set numbers:

* the pre-safety inject clip to the executor acceleration bounds
  (tools/coopfree/execution.py inject, bounds from job['accel_bounds']); and
* the CommandOwnership target clip of ``traffic_shepherd.damping.direct_target``::

      lo = max(accel_min, -speed/dt)
      hi = min(accel_cap, (allowed - speed)/dt)
      command = clip(acceleration, lo, hi); target = max(0, speed + dt*command)

The allowed speed of a member comes from the CAV's subscribed
``allowed_speed_mps`` state-row field (traffic_shepherd/c4/sumo_io.py) when the
rows carry it, otherwise from the preserved network through the lane table in
``traffic_shepherd/c4/geometry.py``. Unresolvable speeds make every helper here
report "unknown": callers must degrade to the pre-fix behaviour (gate inactive,
clamp inactive), never guess a number.
"""
from __future__ import annotations

import math

CONTROL_DT_S = 1.0   # one-second MAPPO control interval (the plant's decision tick)
GATE_EPS_MPS = 1e-6  # issued-target equality tolerance (m/s); far below any physical effect


def issued_target(v_mps, raw, allowed, *, accel_min, accel_max, dt=CONTROL_DT_S):
    """Exactly ``damping.direct_target``'s issued slowDown target, or None infeasible.

    ``raw`` is the requested absolute acceleration after the pre-safety clip;
    ``allowed`` the member's allowed speed. ``None`` mirrors direct_target's
    ``lo > hi`` infeasibility (the caller must treat the second as unknown).
    """
    lo = max(float(accel_min), -float(v_mps) / float(dt))
    hi = min(float(accel_max), (float(allowed) - float(v_mps)) / float(dt))
    if lo > hi:
        return None
    return max(0.0, float(v_mps) + float(dt) * min(max(float(raw), lo), hi))


def allowed_speed_mps(row, lane_speeds=None):
    """One observable row's allowed speed: subscribed value, else the lane table.

    ``row['allowed_speed_mps']`` (CAV subscription, sumo_io.py) wins when finite
    and positive; otherwise the row's model ``lane`` indexes the preserved net's
    speeds via ``geometry.lane_allowed_speeds_mps``. None when neither resolves.
    """
    value = row.get('allowed_speed_mps') if isinstance(row, dict) else None
    if value is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value) and value > 0:
            return value
    lane = row.get('lane') if isinstance(row, dict) else None
    if lane is None:
        return None
    try:
        lane = int(lane)
    except (TypeError, ValueError):
        return None
    if lane_speeds is None:
        try:
            from traffic_shepherd.c4.geometry import lane_allowed_speeds_mps
            lane_speeds = lane_allowed_speeds_mps()
        except Exception:  # net unavailable: unknown, never a guessed number
            return None
    return lane_speeds.get(lane)


def resolve_allowed_mps(rows, members):
    """``{vid: allowed m/s}`` for every member, or None when any is unresolvable."""
    by_vid = {r['vid']: r for r in (rows or ())}
    out = {}
    for vid in members:
        row = by_vid.get(vid)
        speed = allowed_speed_mps(row) if row is not None else None
        if speed is None:
            return None
        out[vid] = speed
    return out


def member_t0_fields(rows, commands, residuals, members, bounds, *, dt=CONTROL_DT_S):
    """NULL_DIAGNOSIS 8.3 label fields per member at t0 (None when unresolvable).

    ``residuals`` maps vid -> the policy's own one-second residual at t0 (the
    sample's own_action); the policy raw is the pre-safety clip of
    ``commands[vid] + residual``. ``leader_gap_m``/``leader_v_mps`` are added
    only when the state row carries a leader (sumo_io subscription tuple
    ``(leader_id, gap)`` or explicit fields).
    """
    by_vid = {r['vid']: r for r in (rows or ())}
    if not members or any(v not in by_vid for v in members):
        return None
    out = {}
    for vid in members:
        row = by_vid[vid]
        speed = float(row['v_mps'])
        allowed = allowed_speed_mps(row)
        if allowed is None:
            return None
        fields = dict(allowed_mps=allowed, headroom_mps2=(allowed - speed) / float(dt),
                      v_mps=speed)
        if commands is not None and vid in commands and residuals is not None \
                and vid in residuals and bounds is not None:
            raw = min(max(float(commands[vid]) + float(residuals[vid]),
                          float(bounds[0])), float(bounds[1]))
            fields['policy_raw_mps2'] = raw
            fields['policy_raw_at_clip'] = bool(raw >= float(bounds[1]) - 1e-9
                                                or raw <= float(bounds[0]) + 1e-9)
        leader = row.get('leader')
        if isinstance(leader, (tuple, list)) and len(leader) >= 2 and leader[0]:
            fields['leader_gap_m'] = float(leader[1])
            if len(leader) > 2 and leader[2] is not None:   # richer subscription variant
                fields['leader_v_mps'] = float(leader[2])
        else:
            if row.get('leader_gap_m') is not None:
                fields['leader_gap_m'] = float(row['leader_gap_m'])
            if row.get('leader_v_mps') is not None:
                fields['leader_v_mps'] = float(row['leader_v_mps'])
        out[vid] = fields
    return out
