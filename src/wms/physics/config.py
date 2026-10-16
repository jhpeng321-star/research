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
#: Lane-rule selectors (C4): full MOBIL discretionary changes or forced merges only.
LANE_RULES = ("mobil", "forced_merge_only", "learned_discretionary")
#: Gap-acceptance rule of the lane-change feasibility mask (GAPRULE): the legacy fixed
#: ``lane_safe_gap_m`` / ``merge_gap`` + IDM follower-braking check, or the speed-dependent
#: SUMO-LC2013-like secure-gap rule (see :func:`wms.physics.mobil.speed_gap_safe`).
LANE_GAP_RULES = ("fixed", "speed")


@dataclass
class PhysicsConfig:
    """Frozen rollout2-compatible physics switches and IDM parameters.

    Geometry is expressed in the ``s_down`` frame: the acceleration lane (2) ends
    at 75.44 m and the mandatory-merge zone is ``[0, 75.44]``.  ``merge_cal``
    enables the calibrated SUMO-like mandatory-merge gap acceptance and the
    cooperative mainline yielding; it defaults to off, in which case the legacy
    MOBIL-merge baseline is reproduced exactly.

    ``lane_rule`` selects the lane-change policy:
    ``mobil`` (default, unchanged legacy behaviour) or ``forced_merge_only``,
    which drops every discretionary MOBIL change and keeps only the mandatory
    ramp merge (lane 2 -> 1 inside the merge zone, gap safety + urgency intact),
    or ``learned_discretionary``: the physics still proposes
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
    """

    dt_s: float = 0.5
    open_road: bool = True
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
    # V6MERGE (2026-10-16): opt-in lane-end interaction rules; absent = legacy execution.
    merge_end_wait: bool = False
    merge_end_wait_distance_m: float = MERGE_END_WAIT_DISTANCE_M
    merge_end_wait_speed_mps: float = MERGE_END_WAIT_SPEED_MPS
    mainline_merge_yield: bool = False
    mainline_merge_yield_distance_m: float = MAINLINE_MERGE_YIELD_DISTANCE_M
    mainline_merge_yield_decel_mps2: float = MAINLINE_MERGE_YIELD_DECEL_MPS2

    def __post_init__(self) -> None:
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
                                            "emergency_decel_mps2", "merge_end_wait",
                                            "mainline_merge_yield")
                and v is not None}
        data.update(overrides)
        return cls(**data)
