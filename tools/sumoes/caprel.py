"""Shared cap-relative algebra for SUMO live queries and WM inferred geometry.

CTRL0 is the executed zero-residual command, not the uncapped native base.
This adapter leaves the frozen runtimes and their observer wrappers untouched.
"""
import torch
import math


def scalar_executed(base, cap, speed, residual, lo=-3., hi=1.2, dt=.5, accel_max=1.2):
    if not all(math.isfinite(x) for x in (base, cap, speed, residual, lo, hi, dt, accel_max)):
        raise ValueError('nonfinite cap-relative inputs')
    floor = -speed / dt
    ctrl0 = max(floor, min(base, cap))
    issued = min(max(ctrl0 + residual, min(lo, ctrl0)), max(min(hi, accel_max), ctrl0))
    return max(floor, min(issued, cap))


def executed(base, cap, speed, residual, lo=-3., hi=1.2, dt=.5, accel_max=1.2):
    """Tensor/scalar-independent law; callers convert live scalar inputs to tensors."""
    floor = -speed / dt
    ctrl0 = torch.maximum(floor, torch.minimum(base, cap))
    lower = torch.minimum(torch.full_like(ctrl0, float(lo)), ctrl0)
    upper = torch.maximum(torch.full_like(ctrl0, min(float(hi), float(accel_max))), ctrl0)
    issued = torch.minimum(torch.maximum(ctrl0 + residual, lower), upper)
    return torch.maximum(floor, torch.minimum(issued, cap))


def wm_executed(state, residual, native, lo=-3., hi=1.2):
    from wms.control.alwayscmd import inferred_base, inferred_cap
    return executed(inferred_base(state), inferred_cap(state), state['v_mps'], residual, lo, hi)


def live_cap(api, vid, dt=.5):
    """Mode-27 prospective ceiling, using live API only; no type-assignment files."""
    from traffic_shepherd.c4.geometry import is_junction_lane, lane_and_s
    from wms.control.contract import MERGE_S1_M
    from wms.control.sumo_idm import FREE_GAP_M
    vehicle = api.vehicle
    speed = float(vehicle.getSpeed(vid))
    leader = vehicle.getLeader(vid)
    lead_id, gap = ('', FREE_GAP_M) if leader is None else leader
    lead_speed = speed if leader is None else float(vehicle.getSpeed(lead_id))
    decel = float(vehicle.getDecel(vid if leader is None else lead_id))
    safe = float(vehicle.getFollowSpeed(vid, speed, float(gap), lead_speed, decel, lead_id))
    lane_id = vehicle.getLaneID(vid)
    if not is_junction_lane(lane_id):
        lane, s = lane_and_s(lane_id, vehicle.getLanePosition(vid))
        if lane == 2:
            gap = max(MERGE_S1_M - s, 0.)
            safe = min(safe, float(vehicle.getStopSpeed(vid, speed, gap)), gap / dt)
    vmax = min(speed + float(vehicle.getAccel(vid)) * dt,
               float(vehicle.getAllowedSpeed(vid)))
    return (max(0., min(safe, vmax)) - speed) / dt
