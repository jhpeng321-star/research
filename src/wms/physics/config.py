"""Physics switches and IDM/MOBIL parameters of the WM-S baseline."""
from __future__ import annotations

import json
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

    def __post_init__(self) -> None:
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

        ``merge_cal``, ``lane_rule`` and ``lane_gap_rule`` are never read from the file;
        the caller enables them explicitly.  ``overrides`` win over the file.
        """
        known = set(cls.__dataclass_fields__)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"merge-cal params must be a JSON object: {path}")
        data = {k: v for k, v in raw.items()
                if k in known and k not in ("merge_cal", "lane_rule", "lane_gap_rule")
                and v is not None}
        data.update(overrides)
        return cls(**data)
