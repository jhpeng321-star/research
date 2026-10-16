"""MOBIL lane-change logits with the mandatory accel-lane merge."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor

from .config import PhysicsConfig
from .idm import IDMParams, idm_accel
from .merge_cal import merge_cal_target_logit, merge_urgency
from .neighbours import _CLOSED_ROAD_FREE_GAP_M, MIN_GAP_M, lane_neighbour, lane_of, lead_neighbour

#: Logit of a forbidden lane change.
MASKED_LOGIT = -8.0
#: ``learned_discretionary``: logit of a safe adjacent change before the learned correction.
#: Matched to the BASE RATE of discretionary changes (softmax ~e^-6 ~ 0.25 % per open lane per
#: step, cf. the recorded 0.124 %/HDV-step), so the soft flip-rate hinge and the lane loss agree;
#: physics alone never changes lane (prior < stay = 0).  The learned correction needs a lane
#: logit cap above 6 to trigger a change (configs/c4_wm.toml: lane_logit_cap = 8).
LEARNED_PRIOR_LOGIT = -6.0
_DEFAULT_POLITENESS = 0.2


def _idm(physics: PhysicsConfig, v: Tensor, leader_v: Tensor, gap: Tensor,
         params: IDMParams) -> Tensor:
    return idm_accel(v, leader_v, gap, params, accel_min=physics.accel_min_mps2,
                     accel_max=physics.accel_max_mps2)


def _follower_gain(kin: Tensor, present: Tensor, lane: Tensor, target_lane: Tensor,
                   a_current: Tensor, gap_f: Tensor, follower_v: Tensor,
                   driver_params: Optional[Mapping[str, Tensor]],
                   default: IDMParams, physics: PhysicsConfig) -> tuple[Tensor, Tensor]:
    """Acceleration change of the target-lane follower caused by the lane change.

    Returns ``(follower_accel_after, follower_gain)``.
    """
    s, v = kin[..., 0], kin[..., 4]
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)
    follower = (present.unsqueeze(-2) & present.unsqueeze(-1)
                & (lane.unsqueeze(-2) == target_lane.unsqueeze(-1)) & (ds < 0))
    follower_dist = torch.where(follower, -ds, torch.full_like(ds, float("inf")))
    follower_gap, follower_index = follower_dist.min(-1)
    follower_has = torch.isfinite(follower_gap)
    follower_old = torch.gather(a_current, -1, follower_index)
    if driver_params is None:
        follower_idm = default
    else:
        follower_idm = IDMParams.from_driver_params(
            {name: torch.gather(value, -1, follower_index)
             for name, value in driver_params.items()})
    a_follow = _idm(physics, follower_v, v, gap_f, follower_idm)
    gain = torch.where(follower_has, a_follow - follower_old, torch.zeros_like(a_follow))
    return a_follow, gain


def target_overlap(kin: Tensor, present: Tensor, target_lane: Tensor,
                   margin_m: float) -> Tensor:
    """``[B, K]``: some present vehicle in ``target_lane`` overlaps the ego's footprint.

    Vehicle ``j`` occupies ``[s_j - L_j, s_j]``; it blocks ego ``i`` when the two intervals,
    each widened by ``margin_m``, intersect (side by side, including ``s_j == s_i``, which
    the ahead/behind neighbour search never sees).
    """
    s, length = kin[..., 0], kin[..., 6]
    lane = lane_of(kin)
    front_i, rear_i = s.unsqueeze(-1), (s - length).unsqueeze(-1)       # [B, i, 1]
    front_j, rear_j = s.unsqueeze(-2), (s - length).unsqueeze(-2)       # [B, 1, j]
    other = present.unsqueeze(-2) & (lane.unsqueeze(-2) == target_lane.unsqueeze(-1))
    eye = torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)
    hit = other & ~eye & (rear_j < front_i + margin_m) & (front_j > rear_i - margin_m)
    return hit.any(-1)


def anticipated_kin(kin: Tensor, horizon_s: float) -> Tensor:
    """``kin`` with every vehicle moved ``horizon_s`` ahead at its current acceleration.

    Constant-acceleration extrapolation of ``s`` and ``v`` (a braking car stops at
    ``v^2 / (2|a|)``, no reversing); lanes, accelerations and lengths are unchanged.
    """
    if horizon_s <= 0.0:
        return kin
    h = float(horizon_s)
    v, a = kin[..., 4], kin[..., 5]
    v_next = (v + a * h).clamp_min(0.0)
    stop = a.clamp_max(-1.0e-6).abs()
    dist = torch.where(v + a * h >= 0.0, v * h + 0.5 * a * h * h, v * v / (2.0 * stop))
    out = kin.clone()
    out[..., 0] = kin[..., 0] + dist.clamp_min(0.0)
    out[..., 4] = v_next
    return out


def speed_gap_safe(kin: Tensor, present: Tensor, target_lane: Tensor,
                   physics: PhysicsConfig) -> Tensor:
    """GAPRULE ``lane_gap_rule='speed'``: SUMO-LC2013-like secure gaps (``[B, K]`` bool).

    Judged on the configuration one ``gap_anticipation_s`` ahead (every car extrapolated at
    its current acceleration; :func:`anticipated_kin`): SUMO moves all cars before it
    changes lanes, and the WM applies the change together with the same step's motion, so
    the gap that matters is the one AFTER the step (a car passing side by side now is
    ahead then).  On that configuration:

    lead:   ``gap_l >= s0_lead + T_lead v + max(0, v^2 - v_lead^2) / (2 b_lead)``
    follow: ``gap_f >= s0_follow + T_follow v_f + max(0, v_f^2 - v^2) / (2 b_follow)``
    and no target-lane vehicle overlapping the ego (length-aware, ``gap_overlap_margin_m``).
    The kinematic term is the follower-braking limit: the follower must not need more than
    ``b_follow`` (beyond its headway margin) to stay behind the changing car.  Missing
    neighbours carry the free-road gap and the ego's speed, so they never block.
    """
    p = physics
    kin_a = anticipated_kin(kin, p.gap_anticipation_s)
    gap_l, lv_l = lane_neighbour(kin_a, present, target_lane, p, ahead=True)
    gap_f, fv_f = lane_neighbour(kin_a, present, target_lane, p, ahead=False)
    v = kin_a[..., 4]
    req_l = p.gap_s0_lead_m + p.gap_t_lead_s * v \
        + (v * v - lv_l * lv_l).clamp_min(0.0) / (2.0 * p.gap_b_lead_mps2)
    req_f = p.gap_s0_follow_m + p.gap_t_follow_s * fv_f \
        + (fv_f * fv_f - v * v).clamp_min(0.0) / (2.0 * p.gap_b_follow_mps2)
    clear = ~target_overlap(kin_a, present, target_lane, p.gap_overlap_margin_m)
    return (gap_l >= req_l) & (gap_f >= req_f) & clear


def stopped_merge_gap_safe(kin: Tensor, present: Tensor, target_lane: Tensor,
                           physics: PhysicsConfig) -> Tensor:
    """LANEFIX ``gap_stopped_relax``: relaxed secure gaps of a (near-)stopped merger ``[B, K]``.

    Judged on the CURRENT configuration (a stopped car does not move during the step; the
    anticipated configuration would move a creeping follower onto it): leader bumper gap
    ``>= gap_stopped_min_m``, follower gap ``>= gap_stopped_min_m + max(0, v_f^2 - v^2) /
    (2 gap_b_follow)`` (the follower can still stop behind the merger) and no target-lane
    vehicle overlapping the ego's footprint widened by ``gap_stopped_min_m``.  The caller
    applies it only to stopped ramp mergers, OR-ed with :func:`speed_gap_safe`.
    """
    p = physics
    gap_l, _ = lane_neighbour(kin, present, target_lane, p, ahead=True)
    gap_f, fv_f = lane_neighbour(kin, present, target_lane, p, ahead=False)
    v = kin[..., 4]
    req_f = p.gap_stopped_min_m + (fv_f * fv_f - v * v).clamp_min(0.0) / (2.0 * p.gap_b_follow_mps2)
    clear = ~target_overlap(kin, present, target_lane, p.gap_stopped_min_m)
    return (gap_l >= p.gap_stopped_min_m) & (gap_f >= req_f) & clear


def merge_end_waiting(kin: Tensor, present: Tensor, physics: PhysicsConfig) -> Tensor:
    """V6MERGE: lane-end mergers ``[B, K]`` - present ramp vehicles within
    ``merge_end_wait_distance_m`` of the accel-lane end, no faster than
    ``merge_end_wait_speed_mps`` (computed regardless of the switches)."""
    p = physics
    s = kin[..., 0]
    start = max(p.merge_zone_s0_m, p.merge_zone_s1_m - p.merge_end_wait_distance_m)
    return (present & (lane_of(kin) == p.merge_lane) & (s >= start)
            & (s <= p.merge_zone_s1_m) & (kin[..., 4] <= p.merge_end_wait_speed_mps))


def secure_gaps(v: Tensor, gap_l: Tensor, lv_l: Tensor, gap_f: Tensor, fv_f: Tensor,
                physics: PhysicsConfig) -> Tensor:
    """SUMO-LC2013-like secure gaps with the road's IDM constants ``[B, K]``: leader
    ``>= s0 + T v + max(0, v^2 - v_l^2) / (2 b)``, follower
    ``>= s0 + T v_f + max(0, v_f^2 - v^2) / (2 b)`` (no footprint test)."""
    _, b_comf, time_headway, s0, _ = physics.idm()
    req_l = s0 + time_headway * v + (v * v - lv_l * lv_l).clamp_min(0.0) / (2.0 * b_comf)
    req_f = s0 + time_headway * fv_f + (fv_f * fv_f - v * v).clamp_min(0.0) / (2.0 * b_comf)
    return (gap_l >= req_l) & (gap_f >= req_f)


def _secure_gaps_now(kin: Tensor, present: Tensor, target_lane: Tensor,
                     physics: PhysicsConfig) -> Tensor:
    """:func:`secure_gaps` on THIS configuration plus no footprint overlap ``[B, K]``."""
    p = physics
    gap_l, lv_l = lane_neighbour(kin, present, target_lane, p, ahead=True)
    gap_f, fv_f = lane_neighbour(kin, present, target_lane, p, ahead=False)
    clear = ~target_overlap(kin, present, target_lane, p.gap_overlap_margin_m)
    return secure_gaps(kin[..., 4], gap_l, lv_l, gap_f, fv_f, p) & clear


def merge_end_gap_safe(kin: Tensor, present: Tensor, physics: PhysicsConfig) -> Tensor:
    """V6MERGE: both lane-1 gaps secure now AND one ``gap_anticipation_s`` ahead ``[B, K]``.

    The current check stops anticipation from accepting a gap only because a follower is
    projected to drive through the standing merger; the anticipated one protects the step.
    No stopped-vehicle relaxation.
    """
    p = physics
    # Fused, op-count-lean equivalent of _secure_gaps_now(kin) & _secure_gaps_now(kin_a) for
    # target lane merge_target_lane (identical values; both configurations in one pass).
    both = torch.stack((kin, anticipated_kin(kin, p.gap_anticipation_s)))     # [2, B, K, F]
    s, v, length = both[..., 0], both[..., 4], both[..., 6]
    lane_j = (present & (lane_of(kin) == p.merge_target_lane)).unsqueeze(-2)  # [B, 1, j]
    ds = s.unsqueeze(-2) - s.unsqueeze(-1)                                    # [2, B, i, j]
    inf = float("inf")
    free = p.free_road_gap_m if p.open_road else _CLOSED_ROAD_FREE_GAP_M
    gap_l, i_l = torch.where(lane_j & (ds > 0), ds - length.unsqueeze(-2), inf).min(-1)
    gap_f, i_f = torch.where(lane_j & (ds < 0), -ds - length.unsqueeze(-1), inf).min(-1)
    has_l, has_f = torch.isfinite(gap_l), torch.isfinite(gap_f)
    gap_l = torch.where(has_l, gap_l.clamp_min(MIN_GAP_M), free)
    gap_f = torch.where(has_f, gap_f.clamp_min(MIN_GAP_M), free)
    lv_l = torch.where(has_l, v.gather(-1, i_l), v)
    fv_f = torch.where(has_f, v.gather(-1, i_f), v)
    rear = s - length
    m = p.gap_overlap_margin_m
    eye = torch.eye(s.shape[-1], dtype=torch.bool, device=s.device)
    hit = (lane_j & ~eye & (rear.unsqueeze(-2) < s.unsqueeze(-1) + m)
           & (s.unsqueeze(-2) > rear.unsqueeze(-1) - m))
    ok = secure_gaps(v, gap_l, lv_l, gap_f, fv_f, p) & ~hit.any(-1)
    return ok[0] & ok[1]


def merge_end_blocked(kin: Tensor, present: Tensor, physics: PhysicsConfig) -> Tensor:
    """V6MERGE: lane-end mergers whose lane-1 gaps are not secure ``[B, K]`` (a discrete
    mask: computed without an autograd graph)."""
    with torch.no_grad():
        kin = kin.detach()
        return merge_end_waiting(kin, present, physics) & ~merge_end_gap_safe(kin, present,
                                                                               physics)


def mobil_merge_logits(kin: Tensor, present: Tensor, physics: PhysicsConfig,
                       driver_params: Optional[Mapping[str, Tensor]] = None, *,
                       merge_blocked: Optional[Tensor] = None) -> Tensor:
    """MOBIL plus mandatory-merge baseline lane logits ``[B, K, 3]``.

    ``stay`` has logit 0; a change to an adjacent lane has
    ``3 tanh((incentive - threshold (1 - u)) / 4) - 2.5 (+ 6 u^2 on the accel lane)``
    and is masked (-8) unless the leader/follower gaps and the follower braking in
    the target lane are safe (the braking limit grows with merge urgency ``u``).
    With ``physics.lane_gap_rule == 'speed'`` that safety test is :func:`speed_gap_safe`
    instead (speed-dependent secure gaps, kinematic follower braking, no overlap, judged on
    the one-step-anticipated configuration) for discretionary changes and the ramp merge
    alike; the default ``'fixed'`` is unchanged.

    With ``physics.lane_rule == 'forced_merge_only'`` every discretionary change
    is masked as well: only the mandatory ramp merge (lane ``merge_lane`` ->
    ``merge_target_lane`` inside the merge zone, with the same gap safety and
    urgency) can keep a non-masked logit (C4).

    ``merge_blocked`` (V6MERGE, ``merge_end_wait`` only): a precomputed
    :func:`merge_end_blocked` of this configuration (computed here when ``None``).
    """
    p = physics
    s, v = kin[..., 0], kin[..., 4]
    lane = lane_of(kin)
    default = p.idm_params()
    own = default if driver_params is None else IDMParams.from_driver_params(driver_params)
    gap0, lv0 = lead_neighbour(kin, present, p)
    a_cur = _idm(p, v, lv0, gap0, own)
    urgency = merge_urgency(s, p)
    threshold = p.mobil_threshold_mps2
    politeness = torch.full_like(v, _DEFAULT_POLITENESS)
    min_gap = torch.full_like(v, p.lane_safe_gap_m)
    if driver_params is not None:
        threshold = driver_params["mobil_threshold"]
        politeness = driver_params["politeness"]
        min_gap = driver_params["merge_gap"]
        urgency = (urgency * driver_params["merge_urgency"]).clamp(0.0, 1.0)
    in_zone = (s >= p.merge_zone_s0_m) & (s <= p.merge_zone_s1_m)
    b_safe = p.b_safe_base_mps2 + (p.b_safe_end_mps2 - p.b_safe_base_mps2) * urgency
    logits = torch.full((*s.shape, 3), MASKED_LOGIT, dtype=kin.dtype, device=kin.device)
    for target in range(3):
        target_lane = torch.full_like(lane, target)
        gap_l, lv_l = lane_neighbour(kin, present, target_lane, p, ahead=True)
        gap_f, fv_f = lane_neighbour(kin, present, target_lane, p, ahead=False)
        a_new = _idm(p, v, lv_l, gap_l, own)
        a_follow, follower_gain = _follower_gain(
            kin, present, lane, target_lane, a_cur, gap_f, fv_f, driver_params,
            default, p)
        adjacent = (lane - target).abs() == 1
        ramp_ok = (lane != p.merge_lane) | (in_zone if target == p.merge_target_lane
                                            else torch.zeros_like(in_zone))
        no_to_ramp = torch.full_like(lane, target) != p.merge_lane
        forced = (lane == p.merge_lane) & (target == p.merge_target_lane)
        if p.lane_gap_rule == "speed":
            safe = speed_gap_safe(kin, present, target_lane, p)
            if p.gap_stopped_relax and target == p.merge_target_lane:
                # LANEFIX: jam merges of (near-)stopped ramp cars at the lane end.
                stopped = forced & (v < p.gap_stopped_speed_mps)
                safe = safe | (stopped & stopped_merge_gap_safe(kin, present, target_lane, p))
        else:
            safe = (gap_l >= min_gap) & (gap_f >= min_gap) & (a_follow >= -b_safe)
        allowed = adjacent & ramp_ok & no_to_ramp & safe
        discretionary = allowed & ~forced
        if p.lane_rule in ("forced_merge_only", "learned_discretionary"):
            # No physics-proposed discretionary changes: the mandatory merge alone
            # (ramp_ok already confines it to the in-zone lane 2 -> 1 change).
            allowed = allowed & forced
        u = urgency * forced.to(kin.dtype)
        if driver_params is None:
            score = (a_new - a_cur) - p.mobil_threshold_mps2 * (1.0 - u)
        else:
            score = (a_new - a_cur) + politeness * follower_gain - threshold * (1.0 - u)
        move = 3.0 * torch.tanh(score / 4.0) - 2.5 + 6.0 * u * u
        logits[..., target] = torch.where(allowed, move, logits[..., target])
        if p.lane_rule == "learned_discretionary":
            # Safe adjacent changes stay open (neutral prior) for the learned correction.
            logits[..., target] = torch.where(
                discretionary, torch.full_like(move, LEARNED_PRIOR_LOGIT), logits[..., target])
    if p.merge_cal:
        logits = merge_cal_target_logit(logits, kin, present, p, urgency, in_zone)
    if p.merge_end_wait:
        # Final veto (after the stopped relaxation and merge_cal): a masked prior keeps every
        # lane head, the forced-late window included, from merging into an insecure gap.
        logits[..., p.merge_target_lane] = torch.where(
            merge_end_blocked(kin, present, p) if merge_blocked is None else merge_blocked,
            torch.full_like(s, MASKED_LOGIT),
            logits[..., p.merge_target_lane])
    logits = logits.scatter(-1, lane.unsqueeze(-1), torch.zeros_like(s).unsqueeze(-1))
    return logits * present.unsqueeze(-1).to(kin.dtype)
