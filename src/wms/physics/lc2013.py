"""SUMO 1.27.1 LC2013 highway rules, sublane OFF, in WM lane coordinates.

This is a source-backed highway subset, not a claim of bitwise SUMO parity.
Source locations and omissions: docs/LC2013_SOURCE_NOTES.md. No fitted constants
or weights. WM lanes are left=0/right=1/ramp=2; a ramp may only enter lane 1.
``kin`` is the lane-decision snapshot (SUMO changes lanes after moving).
The caller carries ``lc2013_*`` state; observation forcing resets wishes when
the observed lane changes. Pass numeric ``identity`` for reusable packed slots.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor

from .config import PhysicsConfig
from .mobil import physical_target_lane
from .neighbours import lane_of
from .lc2013_geometry import corridor_context, neighbour_displacements, lane_position_shift

PRECISION = 10_000_000
# SUMO v1_27_1 src/config.h.cmake:236 (general floating comparison).
NUMERICAL_EPS = 0.001
STATE_KEYS = ("lc2013_left", "lc2013_right", "lc2013_keep_right",
              "lc2013_lookahead", "lc2013_present", "lc2013_lane",
               "lc2013_identity", "lc2013_advice_accel",
               "lc2013_own_advice_accel", "lc2013_own_has_advice",
               "lc2013_has_advice", "lc2013_advice_weight")
# Diagnostic reason codes, not thresholds.
NONE, STRATEGIC, COOPERATIVE, SPEED_GAIN, KEEP_RIGHT = range(5)


def secure_gap(speed: Tensor, leader_speed: Tensor, accel: Tensor,
               decel: Tensor, tau: Tensor) -> Tensor:
    """MSCFModel_IDM.cpp:218-220; minGap is subtracted by the caller."""
    return (speed * tau + speed * (speed - leader_speed)
            / (2 * torch.sqrt(accel * decel))).clamp_min(0)


def brake_gap(speed: Tensor, decel: Tensor, dt: float, *,
              headway: Tensor | float = 0.0, ballistic: bool = False) -> Tensor:
    """MSCFModel.cpp:87-107; discrete Euler is SUMO's default."""
    if ballistic:
        return speed * (headway + 0.5 * speed / decel)
    n = torch.floor(speed / (decel * dt))
    return dt * (n * speed - decel * dt * n * (n + 1) / 2) + speed * headway


def maximum_safe_stop(gap: Tensor, decel: Tensor, tau: Tensor, dt: float,
                      *, ballistic: bool = False) -> Tensor:
    """LC anticipated speed uses onInsertion=true (MSCFModel.cpp:860-908)."""
    g = (gap - NUMERICAL_EPS).clamp_min(0)
    if ballistic:
        bt = decel * tau
        return torch.sqrt(bt.square() + 2 * decel * g) - bt
    b = decel * dt
    n = torch.floor(0.5 - (tau - 0.5 * torch.sqrt(
        (dt * dt + 4 * (dt * (2 * g / b - tau) + tau.square())).clamp_min(0))) / dt)
    h = 0.5 * n * (n - 1) * b * dt + n * b * tau
    return (n * b + (g - h) / (n * dt + tau).clamp_min(1e-9)).clamp_min(0)


def idm_follow_speed(speed: Tensor, gap: Tensor, leader_speed: Tensor,
                     vmax: Tensor, accel: Tensor, decel: Tensor, tau: Tensor,
                     min_gap: Tensor, dt: float, *, respect_min_gap: bool = True) -> Tensor:
    """IDM::_v for LC anticipation/messages; default stepping=0.25s.

    This local mathematical helper doesn't replace the shared CF workstream.
    SUMO rounds TS/stepping + .5 to an integer number of internal substeps.
    ``gap`` already has minGap subtracted, as in SUMO's CF interface.
    """
    iterations = max(1, int(dt / 0.25 + 0.5))
    v = speed.clamp_min(0)
    g = gap + min_gap if respect_min_gap else gap
    for _ in range(iterations):
        desired = secure_gap(v, leader_speed, accel, decel, tau)
        if respect_min_gap:
            desired = desired + min_gap
        g = g.clamp_min(NUMERICAL_EPS)
        a = accel * (1 - (v / vmax.clamp_min(NUMERICAL_EPS)).pow(4)
                     - (desired / g).square())
        v = (v + dt / iterations * a).clamp_min(0)
        g = g - (v - leader_speed).clamp_min(0) * dt / iterations
    return v


def _settings(v: Tensor, physics: PhysicsConfig,
              params: Optional[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    # Existing population physics settings or directly comparable motion-inferred
    # settings only. MOBIL politeness/threshold do not identify LC2013 parameters.
    a, b, tau, s0, _ = physics.idm()
    defaults = dict(accel=a, decel=b, time_headway=tau, min_gap=s0,
                    v0=physics.v0_mps, lcStrategic=1, lcCooperative=1,
                    lcSpeedGain=1, lcKeepRight=1, lcAssertive=1,
                    lcLookaheadLeft=2, lcSpeedGainRight=0.1,
                    lcSpeedGainRemainTime=20, lcKeepRightAcceptanceTime=-1,
                    lcSpeedGainUrgency=50, lcCooperativeSpeed=1,
                    actionStepLength=physics.dt_s)
    out = {key: torch.full_like(v, float(value)) for key, value in defaults.items()}
    if params is not None:
        for key in out:
            if key in params:
                out[key] = torch.as_tensor(params[key], device=v.device,
                                           dtype=v.dtype).expand_as(v)
        if "lcCooperative" in params and "lcCooperativeSpeed" not in params:
            out["lcCooperativeSpeed"] = out["lcCooperative"]
    return out


def _neighbours(kin: Tensor, present: Tensor, target: Tensor,
                context, p, dt, physics, ballistic):
    s, length = kin[..., 0], kin[..., 6]
    ds, same_edge, preceding_end = neighbour_displacements(s, lane_of(kin), context, physics)
    other = (present.unsqueeze(-2)
             & (lane_of(kin).unsqueeze(-2) == target.unsqueeze(-1)))
    other = other & ~torch.eye(s.shape[-1], device=s.device, dtype=torch.bool)
    search = brake_gap(kin[..., 4], p['decel'], dt,
                       headway=p['time_headway'], ballistic=ballistic) + p['min_gap']
    visible = same_edge | (preceding_end <= search.unsqueeze(-1))
    gl, il = torch.where(other & (ds > 0) & visible,
                         ds - length.unsqueeze(-2), float("inf")).min(-1)
    gf, iff = torch.where(other & (ds < 0), -ds - length.unsqueeze(-1), float("inf")).min(-1)
    overlap = (other & (ds < length.unsqueeze(-2)) & (-ds < length.unsqueeze(-1))).any(-1)
    return gl, il, gf, iff, overlap


def _anticipated(kin: Tensor, gap: Tensor, index: Tensor, distance: Tensor,
                 p: Mapping[str, Tensor], dt: float, accelerating: Tensor,
                 ballistic: bool) -> Tensor:
    v = kin[..., 4]
    has = torch.isfinite(gap)
    vl = v.gather(-1, index)
    bl = p["decel"].gather(-1, index)
    effective_gap = torch.where(has, gap, distance)
    leader_speed = torch.where(has, vl, torch.zeros_like(v))
    stopping_gap = effective_gap + brake_gap(
        leader_speed, torch.maximum(p["decel"], bl), dt, ballistic=ballistic)
    safe = maximum_safe_stop(stopping_gap, p["decel"], p["time_headway"], dt,
                             ballistic=ballistic)
    # LC2013.cpp:1906-1918: near-future IDM when either leader accelerates.
    speed_1s = v + p["accel"] * (1 - dt)
    idm = idm_follow_speed(speed_1s, effective_gap, leader_speed, p["v0"],
                          p["accel"], p["decel"], p["time_headway"], p["min_gap"], dt)
    return torch.minimum(p["v0"], torch.where(accelerating, idm, safe))


@torch.no_grad()
def lane_change(kin: Tensor, present: Tensor, physics: PhysicsConfig, *,
                memory: Optional[Mapping[str, Tensor]] = None,
                driver_params: Optional[Mapping[str, Tensor]] = None,
                identity: Optional[Tensor] = None,
                decision_mask: Optional[Tensor] = None,
                remaining_distance: Optional[Tensor] = None,
                target_remaining_distance: Optional[Tensor] = None,
                best_lane_offset: Optional[Tensor] = None,
                target_best_lane_offset: Optional[Tensor] = None,
                best_occupation: Optional[Tensor] = None,
                current_lane_position: Optional[Tensor] = None,
                ballistic: bool = False,
                allow_overtaking_right: bool = False) -> dict[str, Tensor]:
    """One highway LC2013 decision, including persistent fixed-point wishes.

    Optional route inputs are *distances from this snapshot* and offset in WM
    indexing (negative=left). Geometry-only defaults use the corridor's actual
    edge-local position/continuation lengths. Both through lanes continue to
    the unique exit; ramp must merge. Occupation is best-lane queued length in m.
    Override ``current_lane_position`` for another geometry: continuation LENGTH
    is remaining_distance + current_lane_position (SUMO LaneQ semantics).
    Messages are synchronous on the supplied snapshot; SUMO is downstream-first
    sequential. Speed advice is emitted for the NEXT longitudinal update.
    """
    encoded_end = torch.as_tensor(physics.merge_zone_s1_m, device=kin.device,
        dtype=torch.float32).double()
    kin = kin.detach().double()
    # Empty one-hot is a junction connector in WM observables, not lane 0.
    normal_lane = kin[..., 1:4].sum(-1) > 0
    present = present.bool() & normal_lane
    lane = torch.where(normal_lane, lane_of(kin), -1)
    # float32(75.44) is75.44000244: converting that exact represented endpoint
    # to double (including already-promoted observation tensors) must not put
    # a clamped ramp car outside its legal merge edge.
    # Snap ONLY that representation, not a hand-set extension of the zone.
    at_encoded_end = (lane == physics.merge_lane) & (kin[..., 0] == encoded_end)
    canonicalized_end = at_encoded_end & (kin[..., 0] != physics.merge_zone_s1_m)
    s = torch.where(at_encoded_end, torch.full_like(kin[..., 0], physics.merge_zone_s1_m), kin[..., 0])
    kin = torch.cat((s.unsqueeze(-1), kin[..., 1:]), -1)
    v, s, length = kin[..., 4], kin[..., 0], kin[..., 6]
    dt = float(physics.dt_s)
    p = _settings(v, physics, driver_params)
    target = physical_target_lane(lane)
    right = target > lane
    ramp = lane == physics.merge_lane
    legal = present & (lane >= 0) & (lane <= 2) & (~ramp | (
        (s >= physics.merge_zone_s0_m) & (s <= physics.merge_zone_s1_m)))
    active = present if decision_mask is None else present & decision_mask.bool()
    memory = {} if memory is None else memory
    same = present & memory.get("lc2013_present", torch.zeros_like(present)).bool()
    same = same & (lane == memory.get("lc2013_lane", lane))
    if identity is not None:
        same = same & (identity == memory.get("lc2013_identity", identity))
    def previous(key):
        return torch.where(same, memory.get(key, torch.zeros_like(v)).to(v), 0.0)
    left_wish, right_wish, keep = (previous(key) for key in STATE_KEYS[:3])
    look = previous("lc2013_lookahead")
    action_dt = p['actionStepLength']
    look_new = torch.where(v > look, v,
        ((1 - 0.1 * action_dt) * look + 0.1 * action_dt * v).clamp_min(0))
    # MSLaneChanger:359 skips wantsChange on non-action steps. In particular,
    # neither the lookahead memory nor wish counters may age on those ticks.
    # decision_mask is the caller's isActive schedule (default every sim tick).
    look = torch.where(active & legal, look_new, look)
    la = look * 10 * p["lcStrategic"] * torch.where(right, 1.0, p["lcLookaheadLeft"])
    la = la + 2 * (length + p["min_gap"])
    la = torch.where(p["lcStrategic"] < 0, -1000.0, la)
    context = corridor_context(s, lane, physics)
    position = (context["position"] if current_lane_position is None
                else current_lane_position.to(v))
    distance = (context["current_length"] - position if remaining_distance is None
                else remaining_distance.to(v))
    target_distance = (context["target_length"] - position if target_remaining_distance is None
                       else target_remaining_distance.to(v))
    current_length, target_length = distance + position, target_distance + position
    route_offset = (torch.where(ramp, -torch.ones_like(lane), torch.zeros_like(lane))
                    if best_lane_offset is None else best_lane_offset)
    target_offset = (torch.zeros_like(lane) if target_best_lane_offset is None
                     else target_best_lane_offset)
    # Source 1144-1155 resolves two equally good lanes toward the neighbor,
    # but 1330/1874 still compare against the ORIGINAL route offset.
    offset = torch.where((route_offset == 0) & (target_offset == 0),
                          target - lane, route_offset)
    to_best = (offset != 0) & ((offset > 0) == right)
    route_to_best = to_best & (offset == route_offset)
    usable = distance if best_occupation is None else distance - best_occupation.to(v)
    urgent = route_to_best & (usable / offset.abs().clamp_min(1) < la)
    stay = (~to_best & (target_distance / (offset.abs() + 2) < la))
    stay = stay | ((offset == 0) & (target_distance * 2 < la))
    gl, il, gf, iff, overlap = _neighbours(kin, present, target, context, p, dt, physics, ballistic)
    cl, ic, _, _, _ = _neighbours(kin, present, lane, context, p, dt, physics, ballistic)
    has_l, has_f, has_c = torch.isfinite(gl), torch.isfinite(gf), torch.isfinite(cl)
    vl, vf = v.gather(-1, il), v.gather(-1, iff)
    # MSLaneChanger:851-910: longer action steps extrapolate conservative
    # reaction-time speeds; with the default actionStepLength==TS this is zero.
    remainder = torch.where(action_dt == dt, 0., (p['time_headway'] - dt).clamp_min(0))
    need_l = secure_gap(v + (remainder * kin[..., 5]).clamp_min(0),
                       vl + (remainder * kin[..., 5].gather(-1, il)).clamp_max(0),
                       p["accel"], p["decel"], p["time_headway"])
    need_f = secure_gap(vf + (remainder * kin[..., 5].gather(-1, iff)).clamp_min(0),
                         v + (remainder * kin[..., 5]).clamp_max(0), p["accel"].gather(-1, iff),
                         p["decel"].gather(-1, iff), p["time_headway"].gather(-1, iff))
    valid = torch.isfinite(kin).all(-1) & (v >= 0) & (length > 0)
    for key in ("accel", "decel", "v0", "lcAssertive", "lcSpeedGainRight", "actionStepLength"):
        valid &= torch.isfinite(p[key]) & (p[key] > 0)
    for key in ("min_gap", "time_headway"):
        valid &= torch.isfinite(p[key]) & (p[key] >= 0)
    for value in p.values():
        valid &= torch.isfinite(value)
    blocked_l = has_l & ((gl - p["min_gap"] < need_l / p["lcAssertive"])
                          | ~valid.gather(-1, il))
    blocked_f = has_f & ((gf - p["min_gap"].gather(-1, iff) < need_f / p["lcAssertive"])
                          | ~valid.gather(-1, iff))
    safe = valid & ~blocked_l & ~blocked_f & ~overlap
    accelerating = ((has_l & (kin[..., 5].gather(-1, il) > 0))
                    | (has_c & (kin[..., 5].gather(-1, ic) > 0)))
    this_speed = _anticipated(kin, cl - p["min_gap"], ic, current_length,
                              p, dt, accelerating, ballistic)
    neigh_speed = _anticipated(kin, gl - p["min_gap"], il, target_length,
                               p, dt, accelerating, ballistic)
    threshold_l = torch.where(p["lcSpeedGain"] > 0,
                              torch.trunc(0.2 / p["lcSpeedGain"] * PRECISION),
                              torch.full_like(v, float(2**63 - 1)))
    threshold_r = torch.where(p["lcSpeedGain"] > 0,
                              torch.trunc(0.2 / p["lcSpeedGainRight"] / p["lcSpeedGain"] * PRECISION),
                              torch.full_like(v, float(2**63 - 1)))
    # AbstractLCM::avoidOvertakeRight and MSVehicle::congested use SUMO's
    # documented 60/60.1 km/h congestion semantics, NOT a fitted jam threshold.
    avoid_right = ((v >= 60.0 / 3.6) & (physics.speed_limit_mps >= 60.1 / 3.6)
                    & ~torch.full_like(present, allow_overtaking_right))
    dv_right = torch.maximum(p["v0"] - p["v0"].gather(-1, il), v - vl)
    follow_right = idm_follow_speed(v, gl - p["min_gap"], vl, p["v0"],
        p["accel"], p["decel"], p["time_headway"], p["min_gap"], dt)
    max_decel_speed = (v - p["decel"] * dt).clamp_min(0)
    avoid_speed = torch.where(follow_right >= max_decel_speed, follow_right,
                              torch.maximum(max_decel_speed, vl - 10.0 / 3.6))
    avoid_speed = torch.where(left_wish < threshold_l,
                              torch.maximum(avoid_speed, vl), avoid_speed)
    own_advice = (legal & active & ~urgent & ~right & has_l & avoid_right & (dv_right > 0))
    this_speed = torch.where(own_advice, torch.minimum(this_speed, avoid_speed), this_speed)
    future_follow = idm_follow_speed(v, gl - p["min_gap"] - dv_right * 8, vl,
        p["v0"], p["accel"], p["decel"], p["time_headway"], p["min_gap"], dt)
    avoid_impulse = own_advice & (future_follow < avoid_speed)
    left_wish += torch.where(avoid_impulse, torch.trunc(
        action_dt * dv_right / p["v0"].clamp_min(10) * PRECISION), 0.0)
    relative_gain = (neigh_speed - this_speed) / neigh_speed.clamp_min(10)
    # Source1643-1669 returns on cooperative clearing BEFORE1694's speed-gain
    # and keep-right accumulation. Inconvenience uses the pre-update wishes.
    request = active & legal & urgent & blocked_f
    i = torch.arange(v.shape[-1], device=v.device)
    incoming = request.unsqueeze(-1) & (iff.unsqueeze(-1) == i)
    help_left = (incoming & (~right).unsqueeze(-1)).any(-2)
    help_right = (incoming & right.unsqueeze(-1)).any(-2)
    rel_speed_diff = torch.where(this_speed == 0, 0.0,
        (this_speed - neigh_speed) / torch.maximum(this_speed, neigh_speed).clamp_min(1e-9))
    inconvenience = torch.maximum(rel_speed_diff,
        torch.where(right, left_wish / threshold_r, right_wish / threshold_l)).clamp_max(1)
    cooperative = ((help_right & right) | (help_left & ~right))
    cooperative &= inconvenience <= p["lcCooperative"]
    cooperative &= to_best | (target_length / (offset.abs() + 1) > la)
    cooperative_return = legal & active & ~urgent & ~stay & cooperative
    update = legal & active & ~urgent & ~stay & ~cooperative
    left_new = torch.where(this_speed > neigh_speed, torch.trunc(left_wish * 0.5**action_dt),
               torch.where(this_speed == neigh_speed, torch.trunc(left_wish * 0.8**action_dt),
                            left_wish + torch.trunc(action_dt * relative_gain * PRECISION)))
    right_new = torch.where(this_speed - 5 / 3.6 > neigh_speed,
                             torch.trunc(right_wish * 0.5**action_dt),
                             right_wish + torch.trunc(action_dt * relative_gain * PRECISION))
    # Source default legacy keep-right acceptance, no hand-set dwell threshold.
    road_speed_factor = p["v0"] / float(physics.speed_limit_mps)
    acceptance = road_speed_factor * torch.where(p["lcKeepRightAcceptanceTime"] == -1,
                              7 * v.clamp_min(1), p["lcKeepRightAcceptanceTime"])
    full_gap = (target_length - brake_gap(p["v0"], p["decel"], dt,
               headway=p["time_headway"], ballistic=ballistic)).clamp_min(0)
    full_seconds = torch.minimum(acceptance, full_gap / p["v0"])
    slower_target = has_l & (vl < p["v0"])
    target_full_gap = torch.minimum(full_gap, gl - p["min_gap"] - secure_gap(
        p["v0"], vl, p["accel"], p["decel"], p["time_headway"])).clamp_min(0)
    # 1742-1747 modifies BOTH fullSpeedGap and seconds only for a slower
    # target leader. A faster target leader must not limit the later own-
    # current-leader cap through an inadvertently overwritten fullSpeedGap.
    full_gap = torch.where(slower_target, target_full_gap, full_gap)
    full_seconds = torch.where(slower_target, torch.minimum(
        full_seconds, full_gap / (p["v0"] - vl)), full_seconds)
    # LC2013.cpp:1748-1752: don't return right if we cannot pass our slow
    # current-lane leader on its right. Only applies where overtaking is banned.
    current_v = v.gather(-1, ic)
    slow_current = avoid_right & has_c & (p["v0"].gather(-1, ic) < p["v0"])
    no_pass_gap = torch.minimum(full_gap, cl - p["min_gap"])
    # Source 1752 keeps the SIGN of vMax-currentLeaderSpeed. A motion-inferred
    # desired speed can be below the current observed speed; clamping that
    # denominator positive removes SUMO's negative keep-right feedback.
    full_seconds = torch.where(slow_current, torch.minimum(full_seconds,
        no_pass_gap / (p["v0"] - current_v)), full_seconds)
    keep_delta = torch.where(p["lcSpeedGain"] > 0,
        threshold_r * (full_seconds / acceptance.clamp_min(1e-9)) / 5, 0.0)
    keep_new = keep - torch.trunc(action_dt * keep_delta)
    left_wish = torch.where(update & ~right, left_new, left_wish)
    right_wish = torch.where(update & right, right_new, right_wish)
    keep = torch.where(update & right & (this_speed - 5 / 3.6 <= neigh_speed), keep_new, keep)

    # LC2013.cpp:1785 is INSIDE the non-slower-right-lane branch. A mature
    # keep-right wish persists across a slowdown but cannot request a change
    # until the anticipated speed condition is satisfied again.
    keep_request = (right & (this_speed - 5 / 3.6 <= neigh_speed)
                    & (torch.trunc(keep * p["lcKeepRight"]) < -threshold_r))
    gain_request = torch.where(right, right_wish > threshold_r,
                                (left_wish > threshold_l) & ((relative_gain > NUMERICAL_EPS) | avoid_impulse))
    gain_request &= target_length / v.clamp_min(0.1) > p["lcSpeedGainRemainTime"]
    early_strategic = route_to_best & (p["lcStrategic"] >= 0) & (relative_gain >= 0)
    early_strategic &= torch.where(right, right_wish, left_wish) > 0
    reason = torch.zeros_like(lane)
    for condition, code in ((early_strategic, STRATEGIC), (gain_request, SPEED_GAIN),
                            (keep_request, KEEP_RIGHT), (cooperative, COOPERATIVE),
                            (urgent, STRATEGIC)):
        reason = torch.where(condition, code, reason)
    wish = legal & active & ~stay & (reason != NONE)
    change = wish & safe
    choice = torch.where(change, target, lane)
    gain_probability = torch.where(right, right_wish, left_wish)
    speed_gain_urgent = (reason == SPEED_GAIN) & (gain_probability > p["lcSpeedGainUrgency"] * PRECISION)
    is_urgent = wish & (((reason == STRATEGIC) & urgent) | (reason == COOPERATIVE) | speed_gain_urgent)
    # Source1661/1861: a mature left speed-gain/cooperative request explicitly
    # asks its blocking follower to approach at max(minNextSpeed, egoSpeed-1).
    urgent_help = (wish & ~right & blocked_f
        & ((reason == SPEED_GAIN) | (reason == COOPERATIVE))
        & (left_wish > p["lcSpeedGainUrgency"] * PRECISION))
    logits = torch.full((*lane.shape, 3), -torch.inf, device=v.device, dtype=kin.dtype)
    logits.scatter_(-1, lane.clamp(0, 2).unsqueeze(-1), 0.0)
    logits.scatter_(-1, choice.clamp(0, 2).unsqueeze(-1), 1.0)

    # informFollower's Euler cut-in-with-help branch (LC2013.cpp:712-791).
    # Emit acceleration advice so delayed consumption is relative to next v.
    # Ballistic negotiation and full leader overtake/fall-behind are not ported.
    advice = torch.full_like(v, float('inf'))
    if not ballistic:
        bf = p["decel"].gather(-1, iff)
        af = p["accel"].gather(-1, iff)
        tf = p["time_headway"].gather(-1, iff)
        sf = p["min_gap"].gather(-1, iff)
        v0f = p["v0"].gather(-1, iff)
        decel_v = (vf - bf * dt).clamp_min(0)
        one_v = (vf - bf).clamp_min(0)
        decel_gap = gf - sf + v - one_v
        can_help = request & (decel_gap > 0) & (decel_gap >= secure_gap(one_v, v, af, bf, tf))
        vsafe1 = torch.maximum(decel_v, idm_follow_speed(
            vf, gf - sf + dt * v, v, v0f, af, bf, tf, sf, dt))
        vsafe = torch.maximum(decel_v, idm_follow_speed(
            vf, gf - sf + dt * (v - vsafe1), v, v0f, af, bf, tf, sf, dt))
        message = torch.where(can_help, (vsafe - vf) / dt, float('inf'))
        advice.scatter_reduce_(-1, iff, message, reduce="amin", include_self=True)
        emergency = getattr(physics, 'sumo_cf_emergency_decel_mps2', None)
        slack = min(9.0 if emergency is None else float(emergency), 1.5)
        min_follower_speed = (vf - torch.maximum(bf, torch.full_like(bf, slack)) * dt).clamp_min(0)
        help_speed = torch.maximum(min_follower_speed, v - 1.0)
        message = torch.where(urgent_help, (help_speed - vf) / dt, float('inf'))
        advice.scatter_reduce_(-1, iff, message, reduce="amin", include_self=True)
    has_advice = torch.isfinite(advice)
    advice = torch.where(has_advice, advice, 0.0)

    # Reset at the *next* observed lane change, rather than predicted changes:
    # teacher forcing must never manufacture a successful manoeuvre.
    result = dict(lane_logits=logits.to(torch.float32), next_lane=choice,
                  lc2013_reason=reason, lc2013_wish=wish, lc2013_safe=safe,
                  lc2013_blocked_leader=blocked_l, lc2013_blocked_follower=blocked_f,
                  lc2013_target_lane=target,
                  lc2013_current_length=current_length, lc2013_target_length=target_length,
                  lc2013_position=position,
                  lc2013_gap_leader=gl, lc2013_gap_follower=gf,
                  lc2013_position_shift=lane_position_shift(s, lane, choice, physics),
                  lc2013_keep_full_seconds=full_seconds,
                  lc2013_keep_signed_cap=right & slow_current & (current_v > p["v0"]),
                  lc2013_endpoint_canonicalized=canonicalized_end,
                  lc2013_cooperative_return=cooperative_return,
                  lc2013_urgent=is_urgent, lc2013_speed_gain_urgent=wish & speed_gain_urgent,
                  lc2013_this_speed=this_speed, lc2013_target_speed=neigh_speed,
                  lc2013_left=torch.where(present, left_wish, 0),
                  lc2013_right=torch.where(present, right_wish, 0),
                  lc2013_keep_right=torch.where(present, keep, 0),
                  lc2013_lookahead=torch.where(present, look, 0),
                  lc2013_present=present.clone(), lc2013_lane=lane.clone(),
                  lc2013_advice_accel=advice,
                  lc2013_has_advice=has_advice,
                  lc2013_advice_weight=p["lcCooperativeSpeed"].clamp(0, 1),
                  lc2013_own_has_advice=own_advice,
                  lc2013_own_advice_accel=torch.where(own_advice, (avoid_speed - v) / dt, 0.0))
    if identity is not None:
        result["lc2013_identity"] = identity.clone()
    return result
