"""Source-backed highway merge subset of SUMO 1.27.1 (no fitted thresholds).

Front-bumper coordinates, WM lanes left=0/right=1/ramp=2. The network's
acceleration lane has no outgoing link; its entry link is major, without foes.
This is a lateral merge, NOT a yield junction. See docs/rb_merge/SOURCES.md.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .merge_network import lane_desired_speed, model_lane_limit


def parameters(kin, physics, driver_params=None):
    a, b, tau, gap, _ = physics.idm()
    defaults = dict(accel=a, decel=b, time_headway=tau, min_gap=gap,
                    v0=physics.v0_mps)
    out = {name: torch.broadcast_to(torch.as_tensor(
        (driver_params or {}).get(name, value), device=kin.device, dtype=kin.dtype),
        kin.shape[:-1]) for name, value in defaults.items()}
    out['v0'] = lane_desired_speed(out['v0'], model_lane_limit(kin, physics), physics.speed_limit_mps)
    return out


def secure_gap(speed, leader_speed, accel, decel, tau):
    """IDM.cpp:218-220; minGap is added by the lane changer, not here."""
    return (speed * tau + speed * (speed - leader_speed)
            / (2 * (accel * decel).sqrt())).clamp_min(0)


def idm_next_speed(speed, gap, leader_speed, p, dt, *, stop=False):
    """IDM.cpp:_v, default stepping=.25; gap is bumper gap (stop: stop distance)."""
    iterations = max(1, int(dt / .25 + .5))
    v = speed.clone()
    g = gap.clone()
    for _ in range(iterations):
        desired = secure_gap(v, leader_speed, p['accel'], p['decel'], p['time_headway'])
        if not stop:
            desired = desired + p['min_gap']
        acceleration = p['accel'] * (1 - (v / p['v0'].clamp_min(1e-9)).pow(4)
                                    - (desired / g.clamp_min(1e-3)).square())
        v = (v + acceleration * (dt / iterations)).clamp_min(0)
        g = g - (v - leader_speed).clamp_min(0) * (dt / iterations)
    if stop:
        # IDM::stopSpeed falls back to maximumSafeStopSpeedEuler when the
        # discretized IDM stalls just short of a stop (CF.cpp:860-882).
        distance = (gap - 1e-3).clamp_min(0)
        bstep = p['decel'] * dt
        headway = dt  # actionStepLengthSecs, default equal to simulation step
        n = (.5 - (headway - .5 * (dt * dt + 4 * (
            dt * (2 * distance / bstep - headway) + headway * headway)).sqrt()) / dt).floor()
        covered = .5 * n * (n - 1) * bstep * dt + n * bstep * headway
        fallback = n * bstep + (distance - covered) / (n * dt + headway)
        v = torch.where((gap > 0) & (speed < 1e-3) & (v < 1e-3), fallback, v)
        v = torch.where(gap < .01, torch.zeros_like(v), v)
        v = torch.minimum(v, gap.clamp_min(0) / dt)
    return v


def target_gaps(kin, present, target_lane=1):
    """Nearest target leader/follower and overlap, including equal-front cars."""
    s, v, length = kin[..., 0], kin[..., 4], kin[..., 6]
    target = present & (kin[..., 1:4].argmax(-1) == target_lane)
    delta = s.unsqueeze(-2) - s.unsqueeze(-1)  # [batch, ego, other]
    valid = target.unsqueeze(-2).expand_as(delta)
    eye = torch.eye(s.shape[-1], device=s.device, dtype=torch.bool)
    valid = valid & ~eye
    inf = torch.full_like(delta, float('inf'))
    ahead = torch.where(valid & (delta >= 0), delta, inf)
    behind = torch.where(valid & (delta < 0), -delta, inf)
    da, li = ahead.min(-1)
    db, fi = behind.min(-1)
    gl = da - length.gather(-1, li)
    gf = db - length
    overlap = (valid & (delta < length.unsqueeze(-2))
               & (delta > -length.unsqueeze(-1))).any(-1)
    return dict(gap_l=gl, gap_f=gf, leader=li, follower=fi,
                has_l=da.isfinite(), has_f=db.isfinite(), overlap=overlap,
                v_l=v.gather(-1, li), v_f=v.gather(-1, fi))


def merge_geometry(kin, present, physics, driver_params=None):
    p = parameters(kin, physics, driver_params)
    g = target_gaps(kin, present, physics.merge_target_lane)
    lead_need = p['min_gap'] + secure_gap(kin[..., 4], g['v_l'], p['accel'],
                                         p['decel'], p['time_headway'])
    fp = {name: value.gather(-1, g['follower']) for name, value in p.items()}
    follow_need = fp['min_gap'] + secure_gap(g['v_f'], kin[..., 4], fp['accel'],
                                            fp['decel'], fp['time_headway'])
    ramp = present & (kin[..., 1:4].argmax(-1) == physics.merge_lane)
    legal = ramp & (kin[..., 0] >= physics.merge_zone_s0_m) & (kin[..., 0] <= physics.merge_zone_s1_m)
    g.update(ramp=ramp, legal=legal, lead_need=lead_need, follow_need=follow_need,
             blocked_l=g['has_l'] & (g['gap_l'] < lead_need),
             blocked_f=g['has_f'] & (g['gap_f'] < follow_need))
    g['safe'] = legal & ~g['overlap'] & ~g['blocked_l'] & ~g['blocked_f']
    return g


def brake_gap_euler(speed, decel, tau, dt):
    """MSCFModel.cpp:87-108; brakeGap(speed) includes the driver's tau."""
    n = (speed / (decel * dt)).floor()
    return dt * (n * speed - decel * dt * n * (n + 1) / 2) + speed * tau


def merge_speed_advice(kin, present, accel, physics, driver_params=None, *, lookahead_speed=None):
    """Lane-end stop + LC2013 informLeader/informFollower highway subset.

    Cooperates only with the paired blocking follower. No global yield zone,
    learned hazard, patience clock or fitted gap relaxation is used.
    """
    dt = physics.dt_s
    p = parameters(kin, physics, driver_params)
    g = merge_geometry(kin, present, physics, driver_params)
    v = kin[..., 4]
    wanted = (v + accel * dt).clamp_min(0)
    left = (physics.merge_zone_s1_m - kin[..., 0]).clamp_min(0)
    # LC2013.cpp:1244-1245, leftward strategic lookahead defaults 10s * 2.
    # The integrated LC2013 stream owns lookahead memory. Direct callers without
    # that state have the source's cold-start value (the current speed).
    look = v if lookahead_speed is None else torch.maximum(v, lookahead_speed.to(v))
    strategic = g['legal'] & (left < look * 10 * 2 + 2 * (kin[..., 6] + p['min_gap']))
    stop = idm_next_speed(v, left, torch.zeros_like(v), p, dt, stop=True)
    planned = torch.minimum(v, stop)
    # Lane end is always a stop constraint while the lane has no continuation.
    wanted = torch.where(g['ramp'], torch.minimum(wanted, stop), wanted)
    # LC2013.cpp:1502-1507, one required lane change, URGENCY=2.
    remaining = (left / look.clamp_min(1e-9) / 2).clamp_min(dt)
    # informLeader: do not overtake when the remaining distance cannot cover it
    # plus braking. Exact stopped-leader/counter-change exceptions are not modeled.
    next_leader = g['v_l'] - dt * torch.maximum(
        torch.ones_like(v), -kin[..., 5].gather(-1, g['leader']))
    next_gap = g['gap_l'] + (next_leader - planned) * dt
    target = torch.maximum((v - dt * p['decel']).clamp_min(0),
                           idm_next_speed(v, next_gap, next_leader, p, dt))
    lp = {name: value.gather(-1, g['leader']) for name, value in p.items()}
    overtake_dist = (g['gap_l'] - p['min_gap'] + kin[..., 6] + lp['min_gap']
        + kin[..., 6].gather(-1, g['leader']) + secure_gap(
            g['v_l'], v, lp['accel'], lp['decel'], lp['time_headway'])).clamp_min(0)
    dv = planned - g['v_l']
    # LC2013.cpp:547-557 has BOTH a space and a time test. The previous subset
    # omitted the time test and substituted ballistic stopping without tau.
    cannot_overtake = ((dv <= 0) |
                      (left - brake_gap_euler(v, p['decel'], p['time_headway'], dt) < overtake_dist) |
                      (remaining < overtake_dist / dv.clamp_min(1e-9)))
    slow = strategic & g['blocked_l'] & cannot_overtake & (target < v)
    decel = torch.minimum(p['decel'], torch.maximum(
        torch.full_like(v, 7 / 3.6), (v - target) / remaining.clamp_min(1e-9)))
    planned = torch.where(slow, torch.minimum(planned, (v - dt * decel).clamp_min(0)), planned)
    wanted = torch.where(slow, torch.minimum(wanted, planned), wanted)
    # informLeader's remainUnblocked branch still supplies a follow-speed cap
    # (LC2013.cpp:637-655); safety at current speed is not safety after accelerating.
    remain_unblocked = strategic & g['has_l'] & ~g['blocked_l']
    wanted = torch.where(remain_unblocked, torch.minimum(wanted, target), wanted)
    planned = torch.where(remain_unblocked, torch.minimum(planned, target), planned)
    # informFollower: request help only when one second's comfortable braking
    # can create the follower's own secure gap (HELP_DECEL_FACTOR=1).
    fp = {name: value.gather(-1, g['follower']) for name, value in p.items()}
    new_f = (g['v_f'] - dt * fp['decel']).clamp_min(0)
    f_one = (g['v_f'] - fp['decel']).clamp_min(0)
    decel_gap = g['gap_f'] - fp['min_gap'] + planned - f_one
    needed = secure_gap(f_one, planned, fp['accel'], fp['decel'], fp['time_headway'])
    help_ok = strategic & g['blocked_f'] & (decel_gap > 0) & (decel_gap >= needed)
    # Source gaps passed to the CF model exclude follower minGap.
    gap = g['gap_f'] - fp['min_gap']
    safe1 = torch.maximum(new_f, idm_next_speed(g['v_f'], gap + dt * planned + fp['min_gap'], planned, fp, dt))
    safe2 = torch.maximum(new_f, idm_next_speed(g['v_f'], gap + dt * (planned - safe1) + fp['min_gap'], planned, fp, dt))
    # Source informFollower branches, in their original priority order:
    # fast ego needs no braking help; otherwise one-second help, eventual help,
    # or ego slows to let the blocking follower pass (LC2013.cpp:689-993).
    closing = planned - g['v_f']
    no_help = (strategic & g['blocked_f'] & (closing >= 10 / 3.6)
               & ((secure_gap(g['v_f'], planned, fp['accel'], fp['decel'], fp['time_headway'])
                   - gap) / remaining < closing))
    eventual = (strategic & g['blocked_f'] & ~no_help & ~help_ok
                & (planned > f_one) & ((planned - f_one) * remaining > needed - decel_gap + .01))
    give_way = strategic & g['blocked_f'] & ~no_help & ~help_ok & ~eventual
    vhelp = torch.maximum(g['v_f'], v + 10 / 3.6)
    distance_to_pass = (gap + kin[..., 6] + p['min_gap']
        + kin[..., 6].gather(-1, g['follower'])
        + secure_gap(planned, vhelp, p['accel'], p['decel'], p['time_headway'])).clamp_min(0)
    own_pass_speed = torch.maximum(vhelp - distance_to_pass / remaining, v - dt * p['decel'])
    wanted = torch.where(give_way, torch.minimum(wanted, own_pass_speed.clamp_min(0)), wanted)
    message = torch.where(no_help, planned - 10 / 3.6,
                torch.where(help_ok, safe2, torch.where(eventual, new_f, vhelp)))
    valid_message = strategic & g['blocked_f'] & (message >= new_f)
    cap = torch.full_like(v, float('inf'))
    cap.scatter_reduce_(-1, g['follower'], torch.where(valid_message, message, torch.full_like(v, float('inf'))),
                        reduce='amin', include_self=True)
    wanted = torch.minimum(wanted, cap)
    result = (wanted - v) / dt
    return torch.where(present, result, torch.zeros_like(result)), g


def apply_merge(kin, present, proposed_lane, deciders, physics, driver_params=None, *, lookahead_speed=None):
    """Resolve ramp merges after motion, reserving accepted target occupancy.

    Existing target vehicles remain obstacles even if proposing to leave. Cars
    proposing to enter the target lane are reservations. Frontmost ramp first.
    """
    result = proposed_lane.clone()
    ramp = present & deciders & (kin[..., 1:4].argmax(-1) == physics.merge_lane)
    result = torch.where(ramp, torch.full_like(result, physics.merge_lane), result)
    count = int(ramp.sum(-1).max())
    if count == 0:
        return result
    occupied = kin.clone()
    reserve = present & (proposed_lane == physics.merge_target_lane) & ~ramp
    occupied[..., 1:4] = torch.where(reserve.unsqueeze(-1),
        torch.nn.functional.one_hot(torch.full_like(result, physics.merge_target_lane), 3).to(kin.dtype),
        occupied[..., 1:4])
    order = torch.where(ramp, kin[..., 0], torch.full_like(kin[..., 0], -float('inf'))).argsort(-1, descending=True)
    p = parameters(kin, physics, driver_params)
    look = kin[..., 4] if lookahead_speed is None else torch.maximum(kin[..., 4], lookahead_speed.to(kin))
    strategic = ((physics.merge_zone_s1_m - kin[..., 0]) <
                 look * 20 + 2 * (kin[..., 6] + p['min_gap']))
    for rank in range(count):
        idx = order[:, rank:rank + 1]
        g = merge_geometry(occupied, present, physics, driver_params)
        allowed = (g['safe'] & ramp & strategic).gather(-1, idx)
        result.scatter_(-1, idx, torch.where(allowed,
            torch.full_like(idx, physics.merge_target_lane), result.gather(-1, idx)))
        selected = torch.zeros_like(present).scatter(-1, idx, allowed)
        occupied[..., 1:4] = torch.where(selected.unsqueeze(-1),
            torch.nn.functional.one_hot(torch.full_like(result, physics.merge_target_lane), 3).to(kin.dtype),
            occupied[..., 1:4])
    return result
