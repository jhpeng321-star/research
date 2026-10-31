"""Physics switches and IDM/MOBIL parameters of the WM-S baseline."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .idm import IDMParams

#: IDM parameters of the SUMO vType means (acceleration, comfortable decel,
#: time headway, minimum gap, exponent) used when ``sumo_fidelity`` is on.
_SUMO_IDM = (2.49, 4.2, 1.135, 2.6, 4)
#: GAPRULE speed-rule constants (TRAIN-fitted, tools/gaprule_fit.py; GAPRULE_NOTES.md).
GAP_S0_LEAD_M = 1.0
GAP_T_LEAD_S = 0.2
GAP_B_LEAD_MPS2 = 3.0
GAP_S0_FOLLOW_M = 0.5
GAP_T_FOLLOW_S = 0.6
GAP_B_FOLLOW_MPS2 = 6.0
#: Longitudinal clearance (m) every target-lane vehicle must keep from the ego's footprint.
GAP_OVERLAP_MARGIN_M = 0.5
#: The speed rule judges the gaps this far ahead (one 0.5 s WM step; SUMO moves, then changes).
GAP_ANTICIPATION_S = 0.5
#: LANEFIX ``gap_stopped_relax``: a ramp merger slower than this counts as stopped (jam merge
#: at the lane end) ...
GAP_STOPPED_SPEED_MPS = 1.0
#: ... and needs only this physical bumper clearance to its target-lane leader/follower
#: (no headway terms, no anticipation; the follower's kinematic stopping term is kept).
GAP_STOPPED_MIN_M = 0.5
#: V6MERGE ``merge_end_wait``: ramp vehicles at most this far before the accel-lane end ...
MERGE_END_WAIT_DISTANCE_M = 25.0
#: ... and no faster than this are lane-end mergers (strict two-sided secure gaps, no relaxation).
MERGE_END_WAIT_SPEED_MPS = 2.0
#: V6MERGE ``mainline_merge_yield``: lane-1 followers this far (front to front) behind a blocked
#: lane-end merger treat it as a virtual leader ...
MAINLINE_MERGE_YIELD_DISTANCE_M = 60.0
#: ... braking for it no harder than this (cooperative slowing, SUMO-LC2013-like).
MAINLINE_MERGE_YIELD_DECEL_MPS2 = 2.0
#: Default IDM parameters.
_DEFAULT_IDM = (2.6, 4.5, 1.1, 2.5, 4)
#: Car-following core selectors (S-RB-CF, 2026-10-31): ``legacy`` = the WM-S
#: textbook-IDM baseline (bit-identical default), ``sumo_idm`` = the SUMO
#: 1.27.1 ``MSCFModel_IDM`` port of :mod:`wms.physics.idm_sumo` (iterated
#: ``_v`` sub-stepping, SUMO minNextSpeed/emergencyDecel floor, maxNextSpeed
#: cap, ``stopSpeed`` at the accel-lane end).
CF_MODELS = ("legacy", "sumo_idm")
#: Lane-rule selectors (C4): full MOBIL discretionary changes or forced merges only.
LANE_RULES = ("mobil", "forced_merge_only", "learned_discretionary")
#: Gap-acceptance rule of the lane-change feasibility mask (GAPRULE): the legacy fixed
#: ``lane_safe_gap_m`` / ``merge_gap`` + IDM follower-braking check, or the speed-dependent
#: SUMO-LC2013-like secure-gap rule (see :func:`wms.physics.mobil.speed_gap_safe`).
LANE_GAP_RULES = ("fixed", "speed")
#: V8WM2 ``forced_window_mode``: the last ``FORCE_WINDOW_M`` of the accel lane forces a feasible
#: merge (``deterministic``, legacy) or keeps the learned merge hazard with a floor (``hazard``).
FORCED_WINDOW_MODES = ("deterministic", "hazard")
#: V8WM2 ``hazard`` mode: minimum per-step merge probability of a feasible forced-window row.
FORCED_WINDOW_MIN_HAZARD = 0.02
#: V8WM2 ``mainline_merge_yield_scope``: blocked lane-end mergers only (legacy) or every
#: in-zone merger blocked only from behind / beside (zipper).
MAINLINE_MERGE_YIELD_SCOPES = ("lane_end", "zone")
#: V8WM3 ``merge_impatience``: a ramp car waits while within this distance of the accel-lane
#: end and no faster than ``MERGE_IMPATIENCE_SPEED_MPS`` ...
MERGE_IMPATIENCE_DISTANCE_M = 25.0
MERGE_IMPATIENCE_SPEED_MPS = 1.0
#: ... its forced-window hazard floor rises linearly from ``forced_window_min_hazard`` to 1 over
#: this waiting time ...
MERGE_IMPATIENCE_HAZARD_TAU_S = 25.0
#: ... and its merge secure-gap margins (``s0`` / ``T``) shrink linearly to the minima below over
#: this waiting time (the kinematic braking terms and the no-overlap check are never relaxed).
MERGE_IMPATIENCE_GAP_TAU_S = 25.0
MERGE_IMPATIENCE_S0_MIN_M = 0.5
MERGE_IMPATIENCE_T_MIN_S = 0.1
#: V10UNBLOCK ``lane_learnable`` (2026-10-19): the physical lane-change envelope.  Bodies are
#: ``[s - L, s]`` (``s`` = front bumper, ``L`` = ``length_m``), judged on the configuration one
#: large step ahead (constant-acceleration ``anticipated_kin``).  A transition is inside the
#: envelope when the maneuver is road-legal (lane 2 only merges to lane 1 inside the merge zone,
#: no lane ever moves to lane 2) AND the ego's body after the step does not intersect any body of
#: the target lane.  This margin is the only clearance used (0 m: actual body non-intersection);
#: no headway / standstill / impatience / ``merge_end_wait`` margins enter it.
LANE_PHYSICAL_MARGIN_M = 0.0
#: V10LANEOPT ``mainline_lane_mode`` (2026-10-19): staged mainline lane changes under
#: ``lane_learnable``.  ``free`` = current behaviour (every learned head draws its own
#: calibrated probability); ``courtesy`` = merges stay fully learned while the ONLY mainline
#: moves allowed are the learned lane-1 -> lane-0 courtesy overtakes of HDVs inside the approach
#: zone upstream of the merge (no lane-0 -> 1 returns, no overtake elsewhere, no CAV change).
MAINLINE_LANE_MODES = ("free", "courtesy")
#: LOOP14 ``emergency_onset``: "plant" = R13E (default), "driver_b" = learned comfortable b.
EMERGENCY_ONSETS = ("plant", "driver_b")
#: V10LANEOPT ``courtesy`` zone: this many metres upstream of the acceleration-lane end
#: (``merge_zone_s1_m``, model frame) where a lane-1 HDV overtake (1 -> 0) is still a learned
#: decision.  Default 0-300 m before the merge end; 0 disables the courtesy draw (merges only).
MAINLINE_COURTESY_ZONE_M = 300.0


@dataclass
class PhysicsConfig:
    """Frozen rollout2-compatible physics switches and IDM parameters.

    Geometry is expressed in the ``s_down`` frame: the acceleration lane (2) ends
    at 75.44 m and the mandatory-merge zone is ``[0, 75.44]``.  ``merge_cal``
    enables the calibrated SUMO-like mandatory-merge gap acceptance and the
    cooperative mainline yielding; it defaults to off, in which case the legacy
    MOBIL-merge baseline is reproduced exactly.

    ``lane_rule`` selects the lane-change policy (owner decision 2026-10-12):
    ``mobil`` (default, unchanged legacy behaviour) or ``forced_merge_only``,
    which drops every discretionary MOBIL change and keeps only the mandatory
    ramp merge (lane 2 -> 1 inside the merge zone, gap safety + urgency intact),
    or ``learned_discretionary`` (owner 2026-10-12, late): the physics still proposes
    only the forced merge, but every SAFE adjacent discretionary change gets a neutral
    prior logit (below 'stay') that the learned lane correction may raise above 'stay'
    - lane changes are learned, not MOBIL; unsafe changes stay masked.

    ``lane_gap_rule`` (GAPRULE, 2026-10-13) selects the gap acceptance of that feasibility
    mask: ``fixed`` (default, legacy: both target-lane gaps >= ``lane_safe_gap_m`` or the
    driver's ``merge_gap`` and the IDM follower response >= -b_safe) or ``speed``: a
    secure-gap rule in the form of SUMO LC2013's
    ``gap_lead >= s0_lead + T_lead v + max(0, v^2 - v_lead^2) / (2 b_lead)`` and
    ``gap_follow >= s0_follow + T_follow v_follow + max(0, v_follow^2 - v^2) / (2 b_follow)``
    (the kinematic braking term replaces the IDM follower-braking check), plus a
    length-aware no-overlap check, all judged on the configuration ``gap_anticipation_s``
    ahead (constant-acceleration extrapolation: SUMO moves the cars before it changes
    lanes, and the WM applies a change together with that step's motion).  It applies to
    discretionary changes and to the ramp merge alike.  The ``gap_*`` constants were chosen on the C4 TRAIN windows only
    (tools/gaprule_fit.py, GAPRULE_NOTES.md) and are unused in ``fixed`` mode.

    ``gap_stopped_relax`` (LANEFIX, 2026-10-14; ``speed`` rule only, default off): a ramp
    merger (lane 2 -> 1) slower than ``gap_stopped_speed_mps`` is judged on the CURRENT
    configuration (no anticipation) with only ``gap_stopped_min_m`` of bumper clearance to
    the target-lane leader and follower plus the follower's kinematic stopping term
    ``max(0, v_f^2 - v^2) / (2 gap_b_follow)`` and the same no-overlap test with margin
    ``gap_stopped_min_m`` (jam merges at the lane end).  It only ever adds admissions.

    ``merge_end_wait`` (V6MERGE, 2026-10-16, default off): a ramp vehicle within
    ``merge_end_wait_distance_m`` of the accel-lane end and no faster than
    ``merge_end_wait_speed_mps`` may enter lane 1 only if BOTH lane-1 gaps are SUMO-secure with
    the road's IDM constants (``s0 + T v + max(0, dv^2) / (2 b)``) on the current AND the
    one-step-anticipated configuration; the stopped-merger relaxation never applies to it and
    no lane head (forced-late window included) can override the veto.  A lane-2 vehicle never
    moves past the lane end (it stops there).  ``mainline_merge_yield`` (V6MERGE, default off):
    the nearest lane-1 follower within ``mainline_merge_yield_distance_m`` behind such a blocked
    merger (merger's rear ahead of the follower's front) brakes for it as a virtual leader with
    its own IDM, by at most ``mainline_merge_yield_decel_mps2``.

    V8WM2 (2026-10-17, all defaults = legacy):

    * ``forced_window_mode``: ``deterministic`` (default) forces a feasible merge in the last
      ``wms.model.lane_v5.FORCE_WINDOW_M`` (logit ``max(prior, 8)``); ``hazard`` lets the
      learned merge head act there too, with the logit floored at
      ``logit(forced_window_min_hazard)`` (a feasible car eventually leaves); the feasibility
      mask is unchanged, forced-window rows count as learnable rows (lane-v5 rows / AP, merge
      timing loss) and they always draw their lane (sampled, whatever ``lane_decision``).
      Requires ``model.lane_model = 'hazard_v5'``.
    * ``merge_end_wait_s0_follow_m`` / ``merge_end_wait_t_follow_s`` (``None`` = road IDM
      ``s0`` / ``T``): the follower constants of the ``merge_end_wait`` secure-gap test.
    * ``mainline_merge_yield_scope``: ``lane_end`` (default) or ``zone`` (see
      :func:`wms.physics.merge_v6.mainline_merge_yield`; requires ``lane_gap_rule='speed'``).

    V8WM3 ``merge_impatience`` (2026-10-17, default off = legacy, bit-identical): LC2013-like
    impatience of lane-end mergers.  Every ramp vehicle carries a waiting time ``w``
    (``state['merge_wait_s']``, carried by the rollout across truncated-BPTT chunks, initialised
    from the observed pre-t0 long history; :func:`wms.physics.mobil.update_merge_wait`): it grows
    by ``dt`` per step while the car is in the accel lane within ``merge_impatience_distance_m``
    of its end and no faster than ``merge_impatience_speed_mps``, is held while the car stays in
    the accel lane, and resets to 0 off it.  (a) The forced-window hazard floor
    (``forced_window_mode='hazard'``) is ``min(1, h0 + (1 - h0) w / merge_impatience_hazard_tau_s)``
    with ``h0 = forced_window_min_hazard`` (logit capped at the forced-merge logit 8).
    (b) With level ``q = min(1, w / merge_impatience_gap_tau_s)`` every ``s0`` / ``T`` margin of the
    merge feasibility (speed gap rule leader/follower, the ``merge_end_wait`` secure-gap leader
    and follower constants) shrinks linearly towards ``merge_impatience_s0_min_m`` /
    ``merge_impatience_t_min_s`` (never raised): ``c(q) = c - q max(0, c - c_min)``.  The kinematic
    braking terms and the hard no-overlap footprint check are never relaxed.  Approximates SUMO
    LC2013's ``lcImpatience`` / ``lcTimeToImpatience`` (impatience grows while a lane change is
    blocked and scales ``lcAssertive``, i.e. divides the required secure gaps); here the
    relaxation is on the margins only, so the follower can still always stop within ``b``.

    V10UNBLOCK ``lane_learnable`` (2026-10-19, default off = bit-identical): replaces every
    BEHAVIOURAL lane-change veto by learned acceptance over a strictly PHYSICAL envelope, so a
    closed Dreamer loop can learn merges / lane changes / yield from data.  With it on:

    * ``wms.physics.mobil.physical_merge_envelope`` is the only feasibility test of the lane
      heads and of the physics lane logits: road-legal transitions (lane 2 -> 1 only inside the
      merge zone; no transition into lane 2) AND actual body non-intersection of the ego's body
      ``[s - L, s]`` with the target lane one step ahead (``lane_physical_margin_m`` clearance).
      The speed-gap headway margins, the ``gap_stopped_relax`` exception, the ``merge_impatience``
      relaxation and the ``merge_end_wait`` ``merge_end_waiting & ~secure`` veto no longer mask
      anything here.
    * The lane heads of ``model.lane_model='hazard_v5'`` always sample their own calibrated
      probability (no ``lane_decision='gated'`` AP gate, no forced last-5 m window, no
      impatience / hazard floor); the ``history`` lane head and a ``lane_teacher`` never overwrite
      a free rollout.
    * ``mainline_merge_yield`` braking is not applied (courtesy must be learned instead).
    * Training rows of ``wms.model.lane_v5.recorded_rows`` cover ALL at-risk HDV steps (stay is a
      survival negative), the envelope-infeasible rows are excluded and counted.

    V10LANEOPT ``mainline_lane_mode`` (2026-10-19, ``lane_learnable`` only, default ``free`` =
    bit-identical): ``free`` samples every learned head (the current V10UNBLOCK behaviour);
    ``courtesy`` keeps the merge head fully learned but restricts the ROLLOUT decision of the
    mainline heads to the lane-1 -> lane-0 overtake of HDVs inside the ``courtesy`` approach zone
    ``[merge_zone_s1_m - mainline_courtesy_zone_m, merge_zone_s1_m]`` (returns 0 -> 1 and any
    overtake outside the zone stay at the physics prior, i.e. no lane change).  The lane-head LOSS
    still trains all heads on all at-risk rows in both modes, so the free stage starts calibrated;
    only the decision is restricted.
    """

    dt_s: float = 0.5
    open_road: bool = True
    #: Remove a vehicle once its predicted position passes the mainline road end
    #: (model frame 553.33 m; ``wms.physics.batched.MODEL_FRAME_ROAD_END_M``).  Plain-bug
    #: fix: every rollout (evaluation predictor, policy imagination, WM training unroll)
    #: supplies ``exit_s_m``.  ``False`` reproduces pre-fix results bit-identically.
    exit_at_road_end: bool = True
    sumo_fidelity: bool = False
    sumo_idm: bool = True
    speed_factor: float = 1.0
    speed_limit_mps: float = 31.29
    free_road_gap_m: float = 250.0
    max_gap_steps: int = 240
    accel_min_mps2: float = -9.0
    accel_max_mps2: float = 3.0
    lane_gap_m: float = 5.0
    lane_incentive_mps: float = 0.4
    lane_safe_gap_m: float = 6.0
    merge_zone_s0_m: float = 0.0
    merge_zone_s1_m: float = 75.44
    merge_lane: int = 2
    merge_target_lane: int = 1
    # S-RB-MERGE: absent = legacy checkpoint behavior.
    merge_model: str = "legacy"
    mobil_threshold_mps2: float = 0.1
    b_safe_base_mps2: float = 4.0
    b_safe_end_mps2: float = 6.0
    merge_cal: bool = False
    t_lead_base_s: float = 1.0
    t_lead_end_s: float = 0.10
    b_accept_base_mps2: float = 4.0
    b_accept_end_mps2: float = 6.0
    gap_floor_m: float = 0.3
    u_yield: float = 0.35
    yield_lookahead_m: float = 120.0
    b_yield_max_mps2: float = 1.75
    yield_min_speed_mps: float = 2.5
    yield_min_merger_speed_mps: float = 3.5
    end_obstacle_range_m: float = 25.0
    lane_rule: str = "mobil"
    lane_gap_rule: str = "fixed"
    gap_s0_lead_m: float = GAP_S0_LEAD_M
    gap_t_lead_s: float = GAP_T_LEAD_S
    gap_b_lead_mps2: float = GAP_B_LEAD_MPS2
    gap_s0_follow_m: float = GAP_S0_FOLLOW_M
    gap_t_follow_s: float = GAP_T_FOLLOW_S
    gap_b_follow_mps2: float = GAP_B_FOLLOW_MPS2
    gap_overlap_margin_m: float = GAP_OVERLAP_MARGIN_M
    gap_anticipation_s: float = GAP_ANTICIPATION_S
    gap_stopped_relax: bool = False
    gap_stopped_speed_mps: float = GAP_STOPPED_SPEED_MPS
    gap_stopped_min_m: float = GAP_STOPPED_MIN_M
    # V6 integration: absent from old physics.json files = legacy execution.
    stop_time_integration: bool = False
    emergency_decel_mps2: float | None = None
    # R13E: ballistic safe-speed envelope; no new network/checkpoint tensors.
    emergency_braking: bool = False
    # R13E: rollout-only two-sided secure gaps from learned per-slot driver settings.
    # No new network tensors; absent/off keeps the original pivot bit-identical.
    secure_gap_veto: bool = False
    # R14 D1: HDV mainline lane changes also need the R13E gs secure gaps (learned T, b) to the
    # new leader and follower (lane logit mask, lane_learnable only).  Off = bit-identical.
    lane_change_secure_gap: bool = False
    # LOOP14 (S-TTC): onset of the R13E envelope.  "plant" (default) = 13E (plan with the plant
    # limit B = -accel_min); "driver_b" = plan with each driver's learned comfortable decel b and
    # brake harder only when the plant-limit plan demands it (SUMO maximumSafeFollowSpeed).
    emergency_onset: str = "plant"
    # V6MERGE (2026-10-16): opt-in lane-end interaction rules; absent = legacy execution.
    merge_end_wait: bool = False
    merge_end_wait_distance_m: float = MERGE_END_WAIT_DISTANCE_M
    merge_end_wait_speed_mps: float = MERGE_END_WAIT_SPEED_MPS
    mainline_merge_yield: bool = False
    mainline_merge_yield_distance_m: float = MAINLINE_MERGE_YIELD_DISTANCE_M
    mainline_merge_yield_decel_mps2: float = MAINLINE_MERGE_YIELD_DECEL_MPS2
    # V8WM2 (2026-10-17): opt-in lane-end hazard / zipper yield / merge_end_wait follower.
    forced_window_mode: str = "deterministic"
    forced_window_min_hazard: float = FORCED_WINDOW_MIN_HAZARD
    mainline_merge_yield_scope: str = "lane_end"
    merge_end_wait_s0_follow_m: float | None = None
    merge_end_wait_t_follow_s: float | None = None
    # V8WM3 (2026-10-17): opt-in waiting-time impatience of lane-end mergers.
    merge_impatience: bool = False
    merge_impatience_distance_m: float = MERGE_IMPATIENCE_DISTANCE_M
    merge_impatience_speed_mps: float = MERGE_IMPATIENCE_SPEED_MPS
    merge_impatience_hazard_tau_s: float = MERGE_IMPATIENCE_HAZARD_TAU_S
    merge_impatience_gap_tau_s: float = MERGE_IMPATIENCE_GAP_TAU_S
    merge_impatience_s0_min_m: float = MERGE_IMPATIENCE_S0_MIN_M
    merge_impatience_t_min_s: float = MERGE_IMPATIENCE_T_MIN_S
    # V10UNBLOCK (2026-10-19): learn merge / lane-change / yield from data (default off).
    lane_learnable: bool = False
    # R9LANE (2026-10-22, default off = bit-identical): a ramp-lane car cannot be moved past
    # the real lane end ``merge_zone_s1_m`` (SUMO never lets a vehicle pass its lane end);
    # it is held there at speed 0 unless it changes lanes in that step.
    ramp_lane_end_stop: bool = False
    # R9LANE2 (2026-10-22, default off = bit-identical): a started merge-risk chain is not
    # terminated by a tick on which the WM-predicted next position of a car that is still on the
    # ramp lane (before its end) falls just outside the zone-start bound (crawling queue head at
    # route x = merge_zone_s0_m); real exits still are (lane change, past the end, absence).
    ramp_risk_retain: bool = False
    lane_physical_margin_m: float = LANE_PHYSICAL_MARGIN_M
    # V10LANEOPT (2026-10-19): staged mainline lane changes (default 'free' = current behaviour).
    mainline_lane_mode: str = "free"
    mainline_courtesy_zone_m: float = MAINLINE_COURTESY_ZONE_M
    # V10INSERT (2026-10-20): use the learned insertion speed / acceptance hazard
    # (``wms.model.entry_learned``) instead of the hand-set sumo_like_v2 rule.  Default off
    # keeps every pre-V10INSERT result bit-identical.
    entry_learned: bool = False
    # S-RB-CF (2026-10-31, default "legacy" = bit-identical): the longitudinal
    # car-following core.  ``sumo_idm`` replaces the textbook-IDM core (and its
    # legacy emergency_decel / emergency-braking envelope and the hand merge-yield
    # rules) by the faithful SUMO 1.27.1 MSCFModel_IDM step; every other physics
    # switch (lane/merge models) is unchanged and composes on top.
    cf_model: str = "legacy"
    #: SUMO vType ``cf.idmStepping`` default (s); 0.5 s steps -> 2 sub-iterations.
    sumo_cf_stepping_s: float = 0.25
    #: SUMO vType ``emergencyDecel`` (None = decel, the SUMO default).
    sumo_cf_emergency_decel_mps2: float | None = None
    #: SUMO vType ``maxSpeed`` cap of ``maxNextSpeed`` (None = unbounded).
    sumo_cf_vtype_max_mps: float | None = None
    #: SUMO vType ``actionStepLength`` (None = every step, the SUMO default; the
    #: rollout does not carry the held-acceleration state for larger values yet).
    sumo_cf_action_step_s: float | None = None

    def __post_init__(self) -> None:
        if self.merge_model not in ("legacy", "sumo"):
            raise ValueError("merge_model must be 'legacy' or 'sumo'")
        if not isinstance(self.secure_gap_veto, bool):
            raise ValueError("secure_gap_veto must be a boolean")
        if not isinstance(self.emergency_braking, bool):
            raise ValueError("emergency_braking must be a boolean")
        if not isinstance(self.lane_change_secure_gap, bool):
            raise ValueError("lane_change_secure_gap must be a boolean")
        if self.lane_change_secure_gap and not self.lane_learnable:
            raise ValueError("lane_change_secure_gap requires lane_learnable")
        if self.emergency_onset not in EMERGENCY_ONSETS:
            raise ValueError(f"emergency_onset must be one of {EMERGENCY_ONSETS}: {self.emergency_onset!r}")
        if self.emergency_onset != "plant" and not self.emergency_braking:
            raise ValueError("emergency_onset='driver_b' requires emergency_braking")
        if self.emergency_braking:
            if not (math.isfinite(self.dt_s) and self.dt_s > 0.0
                    and math.isfinite(self.accel_min_mps2) and self.accel_min_mps2 < 0.0):
                raise ValueError("emergency_braking requires positive dt_s and a negative plant limit")
            if self.emergency_decel_mps2 is not None:
                raise ValueError("emergency_braking cannot combine with legacy emergency_decel_mps2")
        if not isinstance(self.exit_at_road_end, bool):
            raise ValueError("exit_at_road_end must be a boolean")
        for name in ("merge_end_wait_distance_m", "merge_end_wait_speed_mps",
                     "mainline_merge_yield_distance_m", "mainline_merge_yield_decel_mps2"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and > 0")
        emergency = self.emergency_decel_mps2
        if emergency is not None and (
                isinstance(emergency, bool) or not isinstance(emergency, (int, float))
                or not math.isfinite(emergency) or not 0.0 < emergency <= -self.accel_min_mps2):
            raise ValueError("emergency_decel_mps2 must be finite, positive and no greater "
                             "than -accel_min_mps2")
        if self.lane_rule not in LANE_RULES:
            raise ValueError(f"lane_rule must be one of {LANE_RULES}: {self.lane_rule!r}")
        if self.lane_gap_rule not in LANE_GAP_RULES:
            raise ValueError(f"lane_gap_rule must be one of {LANE_GAP_RULES}: "
                             f"{self.lane_gap_rule!r}")
        if not (self.gap_s0_lead_m > 0.0 and self.gap_s0_follow_m > 0.0
                and self.gap_t_lead_s >= 0.0 and self.gap_t_follow_s >= 0.0
                and self.gap_b_lead_mps2 > 0.0 and self.gap_b_follow_mps2 > 0.0
                and self.gap_overlap_margin_m >= 0.0 and self.gap_anticipation_s >= 0.0):
            raise ValueError("gap_s0_* and gap_b_* must be > 0, gap_t_*, "
                             "gap_overlap_margin_m and gap_anticipation_s >= 0")
        if self.gap_stopped_relax and self.lane_gap_rule != "speed":
            raise ValueError("gap_stopped_relax requires lane_gap_rule='speed'")
        if not (self.gap_stopped_speed_mps > 0.0 and self.gap_stopped_min_m > 0.0):
            raise ValueError("gap_stopped_speed_mps and gap_stopped_min_m must be > 0")
        if self.forced_window_mode not in FORCED_WINDOW_MODES:
            raise ValueError(f"forced_window_mode must be one of {FORCED_WINDOW_MODES}: "
                             f"{self.forced_window_mode!r}")
        if not (isinstance(self.forced_window_min_hazard, (int, float))
                and 0.0 < float(self.forced_window_min_hazard) < 1.0):
            raise ValueError("forced_window_min_hazard must be in (0, 1)")
        if self.mainline_merge_yield_scope not in MAINLINE_MERGE_YIELD_SCOPES:
            raise ValueError(f"mainline_merge_yield_scope must be one of "
                             f"{MAINLINE_MERGE_YIELD_SCOPES}: {self.mainline_merge_yield_scope!r}")
        if (self.mainline_merge_yield and self.mainline_merge_yield_scope == "zone"
                and (self.lane_gap_rule != "speed" or self.merge_cal)):
            raise ValueError("mainline_merge_yield_scope='zone' requires lane_gap_rule='speed' "
                             "and merge_cal off")
        for name in ("merge_end_wait_s0_follow_m", "merge_end_wait_t_follow_s"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be None or finite and >= 0")
        for name in ("merge_impatience_distance_m", "merge_impatience_speed_mps",
                     "merge_impatience_hazard_tau_s", "merge_impatience_gap_tau_s",
                     "merge_impatience_s0_min_m", "merge_impatience_t_min_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0.0
                    or (value == 0.0 and not name.endswith("_min_m")
                        and not name.endswith("_min_s"))):
                raise ValueError(f"{name} must be finite and > 0 (minima >= 0)")
        if self.merge_impatience and (self.lane_gap_rule != "speed" or self.merge_cal):
            raise ValueError("merge_impatience requires lane_gap_rule='speed' and merge_cal off")
        if not isinstance(self.lane_learnable, bool):
            raise ValueError("lane_learnable must be a boolean")
        if not isinstance(self.ramp_lane_end_stop, bool):
            raise ValueError("ramp_lane_end_stop must be a boolean")
        if not isinstance(self.ramp_risk_retain, bool):
            raise ValueError("ramp_risk_retain must be a boolean")
        if (isinstance(self.lane_physical_margin_m, bool)
                or not isinstance(self.lane_physical_margin_m, (int, float))
                or not math.isfinite(self.lane_physical_margin_m)
                or self.lane_physical_margin_m < 0.0):
            raise ValueError("lane_physical_margin_m must be finite and >= 0")
        if self.mainline_lane_mode not in MAINLINE_LANE_MODES:
            raise ValueError(f"mainline_lane_mode must be one of {MAINLINE_LANE_MODES}: "
                             f"{self.mainline_lane_mode!r}")
        if (isinstance(self.mainline_courtesy_zone_m, bool)
                or not isinstance(self.mainline_courtesy_zone_m, (int, float))
                or not math.isfinite(self.mainline_courtesy_zone_m)
                or self.mainline_courtesy_zone_m < 0.0):
            raise ValueError("mainline_courtesy_zone_m must be finite and >= 0")
        if not isinstance(self.entry_learned, bool):
            raise ValueError("entry_learned must be a boolean")
        if self.cf_model not in CF_MODELS:
            raise ValueError(f"cf_model must be one of {CF_MODELS}: {self.cf_model!r}")
        for name in ("sumo_cf_stepping_s",):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("sumo_cf_emergency_decel_mps2", "sumo_cf_vtype_max_mps",
                     "sumo_cf_action_step_s"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be None or finite and > 0")
        if self.sumo_cf_action_step_s is not None \
                and self.sumo_cf_action_step_s < self.dt_s:
            raise ValueError("sumo_cf_action_step_s must be >= dt_s")

    def merge_end_follow_constants(self) -> tuple[float, float]:
        """``(s0, T)`` of the ``merge_end_wait`` follower secure gap (default: road IDM)."""
        _, _, time_headway, s0, _ = self.idm()
        return (s0 if self.merge_end_wait_s0_follow_m is None
                else float(self.merge_end_wait_s0_follow_m),
                time_headway if self.merge_end_wait_t_follow_s is None
                else float(self.merge_end_wait_t_follow_s))

    def idm(self) -> tuple[float, float, float, float, int]:
        """``(a_max, b_comf, time_headway, s0, delta)`` of the default driver."""
        if self.sumo_fidelity and self.sumo_idm:
            return _SUMO_IDM
        return _DEFAULT_IDM

    @property
    def v0_mps(self) -> float:
        """Desired free speed of the default driver."""
        return self.speed_limit_mps * (0.992 if self.sumo_fidelity else 1.0) \
            * float(self.speed_factor)

    def idm_params(self) -> IDMParams:
        """Default-driver IDM parameters as an :class:`IDMParams`."""
        a_max, b_comf, time_headway, s0, delta = self.idm()
        return IDMParams(v0=max(self.v0_mps, 1.0e-3), time_headway=time_headway,
                         s0=s0, a_max=a_max, b_comf=b_comf, delta=delta)

    @classmethod
    def from_params_file(cls, path: str | Path, **overrides) -> "PhysicsConfig":
        """Build a config from a fitted merge-cal params JSON (unknown keys ignored).

        ``merge_cal``, ``lane_rule``, ``lane_gap_rule``, ``gap_stopped_relax`` and the V6
        integration switches are never
        read from the file; the caller enables them explicitly.  ``overrides`` win over the file.
        """
        known = set(cls.__dataclass_fields__)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"merge-cal params must be a JSON object: {path}")
        data = {k: v for k, v in raw.items()
                if k in known and k not in ("merge_cal", "lane_rule", "lane_gap_rule",
                                            "gap_stopped_relax", "stop_time_integration",
                                            "emergency_decel_mps2", "emergency_braking",
                                            "secure_gap_veto", "lane_change_secure_gap",
                                            "merge_end_wait",
                                            "mainline_merge_yield", "forced_window_mode",
                                            "mainline_merge_yield_scope", "merge_impatience",
                                            "lane_learnable", "ramp_lane_end_stop", "ramp_risk_retain")
                and v is not None}
        data.update(overrides)
        return cls(**data)
