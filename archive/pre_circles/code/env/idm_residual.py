"""IDM-native selected-absorber analytical controller and residual interface.

The selector and projection helpers are pure and unit-testable.  The engine is
stateful only where the scientific contract requires it: downstream-reference
filtering, member latching/handoffs, jerk memory, and smooth release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from statistics import median
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import traci

from config.constants import IDM_SAFE_SPEED_MODE, PRE_MERGE_EDGE
from env.features import distance_to_merge
from env.snapshot import get_active_snapshot
from idm.decentralized_damping import (
    ABSTAIN,
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
    ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
    ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
    ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
    Bid,
    Candidate,
    LEGACY_ACTION_SET,
    action_mask,
    contend,
    decode_mode,
    enumerate_candidates,
    terminal_reference_percentile,
)
from idm.follower_stopper import (
    MODIFIED_FOLLOWER_STOPPER_P30_V1,
    modified_follower_stopper_speed,
)
from idm.projected_release import (
    PROJECTED_RELEASE_V1,
    PROJECTED_RELEASE_V2,
    PROJECTED_RELEASE_V3,
    ProjectedReleaseContract,
    RearFloorInput,
    handoff_converged,
    release_blend_fraction,
    solve_rear_speed_floor,
)
from idm.projected_blend import (
    IDM_P30_PROJECTED_BLEND_V1,
    native_proxy_acceleration,
    projected_blend_acceleration,
)
from idm.rear_aware_feedback import (
    FeedbackResponse,
    REAR_AWARE_FEEDBACK_V1,
    REAR_AWARE_FEEDBACK_V2,
    feedback_release_reason,
    treatment_strength,
)
from idm.structured_actions import (
    ActionFamily,
    StructuredActionBounds,
    adjusted_reference_speed,
    clip_action,
    gain_desired_acceleration,
    low_pass_action,
    nominal_action,
)


@dataclass(frozen=True)
class IDMResidualConfig:
    dt_s: float = 1.0
    kv_s_inv: float = 0.15
    ks_s2_inv: float = 0.02
    kd_s_inv: float = 0.20
    desired_headway_s: float = 1.4
    gap_error_cap_m: float = 20.0
    residual_min_mps2: float = -0.5
    residual_max_mps2: float = 0.3
    filter_tau_s: float = 0.0
    comfortable_decel_mps2: float = 1.0
    comfortable_accel_mps2: float = 0.5
    jerk_limit_mps3: float = 0.5
    front_ttc_floor_s: float = 3.0
    rear_ttc_floor_s: float = 4.0
    minimum_runway_m: float = 150.0
    preview_min_m: float = 100.0
    preview_max_m: float = 800.0
    disturbance_horizon_m: float = 1000.0
    disturbance_min_vehicles: int = 4
    disturbance_speed_deficit_mps: float = 3.0
    disturbance_std_min_mps: float = 2.0
    reference_percentile: float = 30.0
    # Phase-3 passed with the detected disturbance percentile. Keep that
    # analytical checkpoint as the default foundation for residual learning.
    reference_mode: str = "disturbance_p30"
    reference_filter_tau_s: float = 20.0
    reference_rate_limit_mps_per_s: float = 0.3
    reference_floor_mps: float = 8.0
    minimum_hold_steps: int = 10
    minimum_absorber_spacing_m: float = 150.0
    release_tolerance_mps: float = 0.5
    release_accel_tolerance_mps2: float = 0.25
    release_max_steps: int = 10
    action_family: ActionFamily = "residual"
    structured_action_filter_tau_s: float = 5.0
    gain_min: float = 0.50
    gain_max: float = 1.50
    reference_offset_min_mps: float = -2.0
    reference_offset_max_mps: float = 2.0
    decentralized_damping_enabled: bool = False
    decentralized_option_steps: int = 20
    decentralized_minimum_predicted_dose_s: float = 16.0
    decentralized_require_secure_gap: bool = True
    decentralized_terminal_rear_headroom_m: float = 5.0
    decentralized_action_set: str = LEGACY_ACTION_SET


@dataclass(frozen=True)
class VehicleSample:
    vehicle_id: str
    road_id: str
    lane_id: str
    lane_index: int
    lane_pos_m: float
    distance_to_merge_m: float
    speed_mps: float
    acceleration_mps2: float


@dataclass(frozen=True)
class Disturbance:
    lane_id: str
    distance_to_merge_m: float
    speed_p30_mps: float
    speed_p50_mps: float
    speed_p70_mps: float
    speed_std_mps: float
    vehicle_count: int


@dataclass(frozen=True)
class ProjectionInput:
    current_speed_mps: float
    desired_acceleration_mps2: float
    prior_acceleration_mps2: float
    allowed_speed_mps: float
    leader_gap_m: Optional[float] = None
    leader_speed_mps: Optional[float] = None
    follow_speed_mps: Optional[float] = None
    follower_gap_m: Optional[float] = None
    follower_speed_mps: Optional[float] = None


@dataclass(frozen=True)
class ProjectionResult:
    acceleration_mps2: float
    target_speed_mps: float
    lower_speed_mps: float
    upper_speed_mps: float
    reasons: tuple[str, ...]
    constraint_conflict: bool


def analytical_acceleration(
    *,
    speed_mps: float,
    reference_speed_mps: float,
    gap_m: Optional[float],
    min_gap_m: float,
    leader_speed_mps: Optional[float],
    cfg: IDMResidualConfig,
) -> float:
    """Contract equation before residual and deterministic projection."""
    speed_term = cfg.kv_s_inv * (reference_speed_mps - speed_mps)
    gap_term = 0.0
    relative_term = 0.0
    if gap_m is not None and leader_speed_mps is not None:
        desired_gap = min_gap_m + cfg.desired_headway_s * speed_mps
        gap_error = float(np.clip(gap_m - desired_gap, -cfg.gap_error_cap_m, cfg.gap_error_cap_m))
        gap_term = cfg.ks_s2_inv * gap_error
        relative_term = cfg.kd_s_inv * (leader_speed_mps - speed_mps)
    return float(speed_term + gap_term + relative_term)


def project_command(
    inp: ProjectionInput,
    cfg: IDMResidualConfig,
    *,
    minimum_target_speed_mps: Optional[float] = None,
    minimum_speed_reason: str = "external_speed_floor",
    minimum_acceleration_mps2: Optional[float] = None,
    maximum_acceleration_mps2: Optional[float] = None,
) -> ProjectionResult:
    """Project a desired command into jerk, front, follower, and speed bounds."""
    dt = max(cfg.dt_s, 1e-6)
    reasons: list[str] = []
    acceleration_minimum = (
        -cfg.comfortable_decel_mps2
        if minimum_acceleration_mps2 is None
        else float(minimum_acceleration_mps2)
    )
    acceleration_maximum = (
        cfg.comfortable_accel_mps2
        if maximum_acceleration_mps2 is None
        else float(maximum_acceleration_mps2)
    )
    if acceleration_minimum > acceleration_maximum:
        acceleration_minimum = acceleration_maximum
    desired = float(
        np.clip(
            inp.desired_acceleration_mps2,
            acceleration_minimum,
            acceleration_maximum,
        )
    )
    if desired != inp.desired_acceleration_mps2:
        reasons.append("comfortable_acceleration")

    jerk_delta = cfg.jerk_limit_mps3 * dt
    jerked = float(
        np.clip(
            desired,
            inp.prior_acceleration_mps2 - jerk_delta,
            inp.prior_acceleration_mps2 + jerk_delta,
        )
    )
    if abs(jerked - desired) > 1e-12:
        reasons.append("jerk")

    lower = max(0.0, inp.current_speed_mps - cfg.comfortable_decel_mps2 * dt)
    upper = min(
        max(0.0, inp.allowed_speed_mps),
        inp.current_speed_mps + cfg.comfortable_accel_mps2 * dt,
    )
    if inp.follow_speed_mps is not None and isfinite(inp.follow_speed_mps):
        follow_cap = max(0.0, inp.follow_speed_mps)
        if follow_cap < upper:
            upper = follow_cap
            reasons.append("idm_follow_speed")
    if (
        inp.leader_gap_m is not None
        and inp.leader_speed_mps is not None
        and inp.leader_gap_m >= 0.0
    ):
        front_cap = inp.leader_speed_mps + inp.leader_gap_m / cfg.front_ttc_floor_s
        if front_cap < upper:
            upper = max(0.0, front_cap)
            reasons.append("front_ttc")
    if (
        inp.follower_gap_m is not None
        and inp.follower_speed_mps is not None
        and inp.follower_gap_m >= 0.0
    ):
        rear_floor = inp.follower_speed_mps - inp.follower_gap_m / cfg.rear_ttc_floor_s
        if rear_floor > lower:
            lower = max(0.0, rear_floor)
            reasons.append("follower_ttc")
    if minimum_target_speed_mps is not None and isfinite(minimum_target_speed_mps):
        external_floor = max(0.0, float(minimum_target_speed_mps))
        if external_floor > lower:
            lower = external_floor
            reasons.append(str(minimum_speed_reason))

    conflict = lower > upper
    if conflict:
        # Front collision avoidance dominates. SUMO speedMode 31 supplies the
        # final native car-following clamp if a physically inconsistent state
        # leaves no comfortable solution.
        lower = upper
        reasons.append("front_follower_conflict")

    unconstrained_speed = max(0.0, inp.current_speed_mps + jerked * dt)
    target = float(np.clip(unconstrained_speed, lower, upper))
    acceleration = (target - inp.current_speed_mps) / dt
    return ProjectionResult(
        acceleration_mps2=float(acceleration),
        target_speed_mps=target,
        lower_speed_mps=float(lower),
        upper_speed_mps=float(upper),
        reasons=tuple(dict.fromkeys(reasons)),
        constraint_conflict=conflict,
    )


def detect_disturbances(
    samples: Iterable[VehicleSample],
    cfg: IDMResidualConfig,
) -> dict[str, Disturbance]:
    """Detect lane-local downstream speed deficits from pre-action state."""
    by_lane: dict[str, list[VehicleSample]] = {}
    for sample in samples:
        if sample.road_id != PRE_MERGE_EDGE:
            continue
        if 0.0 <= sample.distance_to_merge_m <= cfg.disturbance_horizon_m:
            by_lane.setdefault(sample.lane_id, []).append(sample)
    result: dict[str, Disturbance] = {}
    for lane_id, rows in by_lane.items():
        if len(rows) < cfg.disturbance_min_vehicles:
            continue
        speeds = np.asarray([row.speed_mps for row in rows], dtype=float)
        p30 = float(np.percentile(speeds, 30.0))
        p50 = float(np.percentile(speeds, 50.0))
        p70 = float(np.percentile(speeds, 70.0))
        std = float(np.std(speeds))
        if p70 - p30 < cfg.disturbance_speed_deficit_mps and std < cfg.disturbance_std_min_mps:
            continue
        slow_rows = [row for row in rows if row.speed_mps <= p30 + 1e-9]
        location = float(median(row.distance_to_merge_m for row in slow_rows))
        result[lane_id] = Disturbance(
            lane_id=lane_id,
            distance_to_merge_m=location,
            speed_p30_mps=p30,
            speed_p50_mps=p50,
            speed_p70_mps=p70,
            speed_std_mps=std,
            vehicle_count=len(rows),
        )
    return result


def select_absorbers(
    samples: Iterable[VehicleSample],
    cav_ids: Iterable[str],
    disturbances: Mapping[str, Disturbance],
    cfg: IDMResidualConfig,
) -> dict[str, str]:
    """Select the nearest feasible upstream CAV, at most one per lane."""
    cav_set = set(cav_ids)
    by_lane: dict[str, list[tuple[float, str]]] = {}
    for sample in samples:
        disturbance = disturbances.get(sample.lane_id)
        if sample.vehicle_id not in cav_set or disturbance is None:
            continue
        runway = sample.distance_to_merge_m - disturbance.distance_to_merge_m
        if runway < cfg.minimum_runway_m or runway > cfg.preview_max_m:
            continue
        by_lane.setdefault(sample.lane_id, []).append((runway, sample.vehicle_id))
    return {
        lane: min(candidates, key=lambda item: (item[0], item[1]))[1]
        for lane, candidates in sorted(by_lane.items())
    }


@dataclass
class _MemberState:
    vehicle_id: str
    selected_step: int
    action_family: Optional[str] = None
    action_value: Optional[float] = None
    mode_index: Optional[int] = None
    feedback_baseline: Optional[FeedbackResponse] = None
    feedback_projection_survival_count: int = 0
    feedback_meaningful_command_count: int = 0


@dataclass
class _ReleaseState:
    prior_acceleration_mps2: float
    age_steps: int = 0


@dataclass
class _ProjectedReleaseState:
    lane_id: str
    selected_step: int
    initial_acceleration_mps2: float
    prior_acceleration_mps2: float
    age_steps: int = 0
    converged_steps: int = 0
    last_target_speed_mps: Optional[float] = None


class IDMResidualActionEngine:
    """Apply analytical smoothing plus bounded residual to selected absorbers."""

    def __init__(self, config: Optional[IDMResidualConfig] = None):
        self.cfg = config or IDMResidualConfig()
        self.reset()

    def reset(self) -> None:
        self.step_index = 0
        self.members: dict[str, _MemberState] = {}
        self.reference_by_lane: dict[str, float] = {}
        self.prior_acceleration: dict[str, float] = {}
        self.filtered_residual: dict[str, float] = {}
        self.filtered_structured_action_by_lane: dict[tuple[str, str], float] = {}
        self.releasing: dict[str, _ReleaseState] = {}
        self.projected_releasing: dict[str, _ProjectedReleaseState] = {}
        self.projected_release_cooldown_until: dict[str, int] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def _samples(self) -> list[VehicleSample]:
        snap = get_active_snapshot()
        if snap is None:
            return []
        rows: list[VehicleSample] = []
        for vehicle_id in snap.ordered_ids():
            road = snap.road(vehicle_id)
            lane_id = snap.lane_id(vehicle_id)
            lane_index = snap.lane_index(vehicle_id)
            lane_pos = snap.lane_pos(vehicle_id)
            speed = snap.speed(vehicle_id)
            accel = snap.accel(vehicle_id)
            if None in (road, lane_id, lane_index, lane_pos, speed, accel):
                continue
            dist = distance_to_merge(vehicle_id)
            if not isfinite(dist):
                continue
            rows.append(
                VehicleSample(
                    vehicle_id=str(vehicle_id),
                    road_id=str(road),
                    lane_id=str(lane_id),
                    lane_index=int(lane_index),
                    lane_pos_m=float(lane_pos),
                    distance_to_merge_m=float(dist),
                    speed_mps=float(speed),
                    acceleration_mps2=float(accel),
                )
            )
        return rows

    def _leader_follower(self, vehicle_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "leader_id": None,
            "leader_gap_m": None,
            "leader_speed_mps": None,
            "follower_id": None,
            "follower_gap_m": None,
            "follower_speed_mps": None,
        }
        try:
            leader = traci.vehicle.getLeader(vehicle_id, 250.0)
            if leader and leader[0]:
                result.update(
                    leader_id=str(leader[0]),
                    leader_gap_m=float(leader[1]),
                    leader_speed_mps=float(traci.vehicle.getSpeed(leader[0])),
                )
        except Exception:
            pass
        try:
            follower = traci.vehicle.getFollower(vehicle_id, 250.0)
            if follower and follower[0]:
                result.update(
                    follower_id=str(follower[0]),
                    follower_gap_m=float(follower[1]),
                    follower_speed_mps=float(traci.vehicle.getSpeed(follower[0])),
                )
        except Exception:
            pass
        return result

    def _candidate_admission(
        self,
        vehicle_id: str,
        candidate: Optional[Candidate] = None,
    ) -> dict[str, Any]:
        state = self._leader_follower(vehicle_id)
        reasons: list[str] = []
        try:
            speed = float(traci.vehicle.getSpeed(vehicle_id))
        except Exception:
            return {"admissible": False, "reasons": ("missing_speed",)}
        gap = state["leader_gap_m"]
        leader_speed = state["leader_speed_mps"]
        front_ttc = None
        if gap is not None and leader_speed is not None and speed > leader_speed:
            front_ttc = gap / max(speed - leader_speed, 1e-6)
            if front_ttc < self.cfg.front_ttc_floor_s:
                reasons.append("front_ttc")
        fgap = state["follower_gap_m"]
        follower_speed = state["follower_speed_mps"]
        rear_ttc = None
        if fgap is not None and follower_speed is not None and follower_speed > speed:
            rear_ttc = fgap / max(follower_speed - speed, 1e-6)
            if rear_ttc < self.cfg.rear_ttc_floor_s:
                reasons.append("rear_ttc")

        front_secure_gap = None
        rear_secure_gap = None
        if self.cfg.decentralized_require_secure_gap and state["leader_id"] is not None:
            try:
                leader_decel = float(traci.vehicle.getDecel(state["leader_id"]))
                front_secure_gap = float(
                    traci.vehicle.getSecureGap(
                        vehicle_id,
                        speed,
                        float(leader_speed),
                        leader_decel,
                        state["leader_id"],
                    )
                )
                if gap is not None and gap + 1e-9 < front_secure_gap:
                    reasons.append("front_secure_gap")
            except Exception:
                front_secure_gap = None
                reasons.append("front_secure_gap_unavailable")
        if self.cfg.decentralized_require_secure_gap and state["follower_id"] is not None:
            try:
                ego_decel = float(traci.vehicle.getDecel(vehicle_id))
                rear_secure_gap = float(
                    traci.vehicle.getSecureGap(
                        state["follower_id"],
                        float(follower_speed),
                        speed,
                        ego_decel,
                        vehicle_id,
                    )
                )
                if fgap is not None and fgap + 1e-9 < rear_secure_gap:
                    reasons.append("rear_secure_gap")
            except Exception:
                rear_secure_gap = None
                reasons.append("rear_secure_gap_unavailable")

        predicted_dose_s = None
        if candidate is not None:
            predicted_dose_s = candidate.runway_m / max(speed, 1.0)
            if predicted_dose_s + 1e-9 < self.cfg.decentralized_minimum_predicted_dose_s:
                reasons.append("insufficient_predicted_dose")
        return {
            "admissible": not reasons,
            "reasons": tuple(dict.fromkeys(reasons)),
            "speed_mps": speed,
            "front_ttc_s": front_ttc,
            "rear_ttc_s": rear_ttc,
            "front_secure_gap_m": front_secure_gap,
            "rear_secure_gap_m": rear_secure_gap,
            "front_gap_m": gap,
            "rear_gap_m": fgap,
            "predicted_dose_s": predicted_dose_s,
        }

    def _candidate_safe(self, vehicle_id: str) -> bool:
        return bool(self._candidate_admission(vehicle_id)["admissible"])

    def _terminal_rear_admission(self, vehicle_id: str, target_speed_mps: float) -> dict[str, Any]:
        """Conservative rear-gap feasibility at a proposed terminal speed."""
        state = self._leader_follower(vehicle_id)
        if state["follower_id"] is None or state["follower_gap_m"] is None:
            return {
                "terminal_rear_admissible": True,
                "terminal_rear_secure_gap_m": None,
                "terminal_rear_headroom_m": None,
                "terminal_rear_reason": None,
            }
        try:
            ego_decel = float(traci.vehicle.getDecel(vehicle_id))
            secure = float(
                traci.vehicle.getSecureGap(
                    state["follower_id"],
                    float(state["follower_speed_mps"]),
                    float(target_speed_mps),
                    ego_decel,
                    vehicle_id,
                )
            )
            headroom = float(state["follower_gap_m"]) - secure
            required_headroom = self.cfg.decentralized_terminal_rear_headroom_m
            admissible = headroom + 1e-9 >= required_headroom
            return {
                "terminal_rear_admissible": admissible,
                "terminal_rear_secure_gap_m": secure,
                "terminal_rear_headroom_m": headroom,
                "terminal_rear_required_headroom_m": required_headroom,
                "terminal_rear_reason": None if admissible else "terminal_rear_secure_gap",
            }
        except Exception:
            return {
                "terminal_rear_admissible": False,
                "terminal_rear_secure_gap_m": None,
                "terminal_rear_headroom_m": None,
                "terminal_rear_reason": "terminal_rear_secure_gap_unavailable",
            }

    def _decentralized_candidate_state(
        self,
        samples: Iterable[VehicleSample],
        cav_ids: Iterable[str],
        disturbances: Mapping[str, Disturbance],
    ) -> tuple[dict[str, tuple[Candidate, ...]], dict[str, dict[str, Any]]]:
        candidates = enumerate_candidates(
            (sample for sample in samples if sample.road_id == PRE_MERGE_EDGE),
            cav_ids,
            disturbances,
            minimum_runway_m=self.cfg.minimum_runway_m,
            maximum_runway_m=self.cfg.preview_max_m,
        )
        admissions: dict[str, dict[str, Any]] = {}
        for lane_candidates in candidates.values():
            for candidate in lane_candidates:
                admission = self._candidate_admission(candidate.vehicle_id, candidate)
                if terminal_reference_percentile(self.cfg.decentralized_action_set) is not None:
                    disturbance = disturbances[candidate.lane_id]
                    percentile = terminal_reference_percentile(self.cfg.decentralized_action_set)
                    terminal_speed = float(getattr(disturbance, f"speed_{percentile}_mps"))
                    terminal = self._terminal_rear_admission(
                        candidate.vehicle_id,
                        max(self.cfg.reference_floor_mps, terminal_speed),
                    )
                    reasons = list(admission.get("reasons", ()))
                    if terminal["terminal_rear_reason"] is not None:
                        reasons.append(str(terminal["terminal_rear_reason"]))
                    admission = {
                        **admission,
                        **terminal,
                        "reasons": tuple(dict.fromkeys(reasons)),
                        "admissible": not reasons,
                    }
                admissions[candidate.vehicle_id] = admission
        return candidates, admissions

    def _uses_projected_release(self) -> bool:
        return self.cfg.decentralized_damping_enabled and self.cfg.decentralized_action_set in {
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
            ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        }

    def _projected_release_contract(self) -> ProjectedReleaseContract:
        if self.cfg.decentralized_action_set in {
            ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        }:
            return PROJECTED_RELEASE_V2
        if (
            self.cfg.decentralized_action_set
            == ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET
        ):
            return PROJECTED_RELEASE_V2
        if (
            self.cfg.decentralized_action_set
            == ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET
        ):
            return PROJECTED_RELEASE_V3
        return PROJECTED_RELEASE_V1

    def _rear_aware_feedback_contract(self):
        if (
            self.cfg.decentralized_action_set
            == ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET
        ):
            return REAR_AWARE_FEEDBACK_V2
        return REAR_AWARE_FEEDBACK_V1

    def _projected_release_blocked_lanes(self, step: int) -> set[str]:
        blocked = {state.lane_id for state in self.projected_releasing.values()}
        blocked.update(
            lane_id
            for lane_id, until in self.projected_release_cooldown_until.items()
            if int(step) <= int(until)
        )
        return blocked

    def _uses_release_lane_blocking(self) -> bool:
        """Prevent a second option while a treatment member is handing off."""
        return self._uses_projected_release() or (
            self.cfg.decentralized_damping_enabled
            and self.cfg.decentralized_action_set
            in {
                ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
                ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
            }
        )

    def _start_feedback_release_cooldown(self, lane_id: str) -> None:
        self.projected_release_cooldown_until[str(lane_id)] = int(self.step_index) + int(
            self.cfg.release_max_steps
        )

    def _begin_release(
        self,
        state: _MemberState,
        lane_id: str,
        *,
        use_projected: Optional[bool] = None,
    ) -> None:
        vehicle_id = state.vehicle_id
        prior = self.prior_acceleration.get(vehicle_id, 0.0)
        projected = self._uses_projected_release() if use_projected is None else bool(use_projected)
        if projected:
            self.releasing.pop(vehicle_id, None)
            self.projected_releasing[vehicle_id] = _ProjectedReleaseState(
                lane_id=str(lane_id),
                selected_step=int(state.selected_step),
                initial_acceleration_mps2=float(prior),
                prior_acceleration_mps2=float(prior),
            )
        else:
            self.projected_releasing.pop(vehicle_id, None)
            self.releasing[vehicle_id] = _ReleaseState(float(prior))

    def _start_projected_release_cooldown(self, lane_id: str) -> None:
        contract = self._projected_release_contract()
        self.projected_release_cooldown_until[str(lane_id)] = int(self.step_index) + int(
            contract.cooldown_steps
        )

    def _update_members(
        self,
        proposed: Mapping[str, str],
        samples: Mapping[str, VehicleSample],
    ) -> tuple[list[dict[str, Any]], set[str]]:
        changes: list[dict[str, Any]] = []
        retained: dict[str, _MemberState] = {}
        former = {state.vehicle_id: (lane, state) for lane, state in self.members.items()}
        for lane, state in sorted(self.members.items()):
            sample = samples.get(state.vehicle_id)
            age = self.step_index - state.selected_step
            valid = (
                sample is not None
                and sample.lane_id == lane
                and sample.road_id == PRE_MERGE_EDGE
                and self._candidate_safe(state.vehicle_id)
            )
            proposed_id = proposed.get(lane)
            if valid and (age < self.cfg.minimum_hold_steps or proposed_id == state.vehicle_id):
                retained[lane] = state
            else:
                reason = (
                    "unsafe_or_exited"
                    if not valid
                    else "handoff_requested"
                    if proposed_id is not None
                    else "disturbance_cleared"
                )
                changes.append(
                    {
                        "lane_id": lane,
                        "old_member": state.vehicle_id,
                        "new_member": None,
                        "reason": reason,
                        "hold_steps": age,
                    }
                )
        for lane, vehicle_id in sorted(proposed.items()):
            if lane in retained:
                continue
            if not self._candidate_safe(vehicle_id):
                continue
            retained[lane] = _MemberState(vehicle_id, self.step_index)
            old = self.members.get(lane)
            changes.append(
                {
                    "lane_id": lane,
                    "old_member": None if old is None else old.vehicle_id,
                    "new_member": vehicle_id,
                    "reason": "selected" if old is None else "handoff",
                    "hold_steps": 0 if old is None else self.step_index - old.selected_step,
                }
            )
        self.members = retained
        current = {state.vehicle_id for state in retained.values()}
        for vehicle_id in set(former) - current:
            lane_id, state = former[vehicle_id]
            # The analytical (non-decentralized) path always resolves to the
            # legacy release.  Keep this helper shared so a former member's
            # state is never reconstructed from mutable TraCI data.
            self._begin_release(state, lane_id, use_projected=False)
        return changes, current

    def _update_decentralized_members(
        self,
        candidates: Mapping[str, tuple[Candidate, ...]],
        admissions: Mapping[str, Mapping[str, Any]],
        actions: Mapping[str, object],
        samples: Mapping[str, VehicleSample],
    ) -> tuple[list[dict[str, Any]], set[str], dict[str, str], list[dict[str, Any]]]:
        """Latch safe local bids and resolve conflicts by V2V priority."""
        changes: list[dict[str, Any]] = []
        former = {state.vehicle_id: (lane, state) for lane, state in self.members.items()}
        retained: dict[str, _MemberState] = {}
        reserved: list[Candidate] = []
        projected_release_ids: set[str] = set()
        blocked_release_lanes: set[str] = set()
        if self._uses_release_lane_blocking():
            self.projected_release_cooldown_until = {
                lane_id: until
                for lane_id, until in self.projected_release_cooldown_until.items()
                if self.step_index <= until
            }
            blocked_release_lanes = self._projected_release_blocked_lanes(self.step_index)

        for lane, state in sorted(self.members.items()):
            sample = samples.get(state.vehicle_id)
            age = self.step_index - state.selected_step
            runtime_admission = (
                self._candidate_admission(state.vehicle_id)
                if sample is not None
                else {"admissible": False, "reasons": ("missing_sample",)}
            )
            safe = sample is not None and bool(runtime_admission.get("admissible", False))
            valid = (
                sample is not None
                and sample.lane_id == lane
                and sample.road_id == PRE_MERGE_EDGE
                and safe
            )
            if valid and age < self.cfg.decentralized_option_steps:
                retained[lane] = state
                disturbance_distance = next(
                    (
                        row.disturbance_distance_to_merge_m
                        for row in candidates.get(lane, ())
                        if row.vehicle_id == state.vehicle_id
                    ),
                    sample.distance_to_merge_m,
                )
                reserved.append(
                    Candidate(
                        vehicle_id=state.vehicle_id,
                        lane_id=lane,
                        distance_to_merge_m=sample.distance_to_merge_m,
                        disturbance_distance_to_merge_m=disturbance_distance,
                        runway_m=sample.distance_to_merge_m - disturbance_distance,
                        rank=0,
                    )
                )
                continue
            release_reason = "unsafe_or_exited" if not valid else "option_completed"
            changes.append(
                {
                    "lane_id": lane,
                    "old_member": state.vehicle_id,
                    "new_member": None,
                    "reason": release_reason,
                    "reason_details": (
                        tuple(runtime_admission.get("reasons", ()))
                        if sample is not None
                        and sample.lane_id == lane
                        and sample.road_id == PRE_MERGE_EDGE
                        else ("missing_sample",)
                        if sample is None
                        else (f"lane_changed:{sample.lane_id}",)
                        if sample.lane_id != lane
                        else (f"road_changed:{sample.road_id}",)
                    ),
                    "hold_steps": age,
                    "mode_index": state.mode_index,
                }
            )
            if self._uses_projected_release() and (
                release_reason == "option_completed"
                or self.cfg.decentralized_action_set
                in {
                    ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
                    ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
                }
            ):
                projected_release_ids.add(state.vehicle_id)
                blocked_release_lanes.add(lane)
            elif self.cfg.decentralized_action_set in {
                ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
                ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
            }:
                # A feedback member must finish its controlled handoff before
                # another same-lane option can start.  This applies to safety
                # exits as well as normal completion; otherwise an unsafe old
                # member can release while a newly selected backup is active.
                self._start_feedback_release_cooldown(lane)
                blocked_release_lanes.add(lane)

        bids: list[Bid] = []
        bid_rows: list[dict[str, Any]] = []
        for lane, lane_candidates in sorted(candidates.items()):
            if lane in retained:
                continue
            for candidate in lane_candidates:
                admission = admissions.get(candidate.vehicle_id, {})
                mode = decode_mode(
                    actions.get(candidate.vehicle_id, ABSTAIN),
                    self.cfg.decentralized_action_set,
                )
                row = {
                    "vehicle_id": candidate.vehicle_id,
                    "lane_id": lane,
                    "rank": candidate.rank,
                    "runway_m": candidate.runway_m,
                    "mode_index": mode.index,
                    "mode": mode.name,
                    "admissible": bool(admission.get("admissible", False)),
                }
                if lane in blocked_release_lanes:
                    row["outcome"] = "projected_release_or_cooldown"
                elif not row["admissible"]:
                    row["outcome"] = "safety_rejected"
                elif mode.index == ABSTAIN:
                    row["outcome"] = "abstained"
                else:
                    row["outcome"] = "contending"
                    bids.append(Bid(candidate, mode))
                bid_rows.append(row)

        winners, rejected = contend(
            bids,
            reserved=reserved,
            minimum_spacing_m=self.cfg.minimum_absorber_spacing_m,
        )
        for lane, bid in sorted(winners.items()):
            retained[lane] = _MemberState(
                vehicle_id=bid.candidate.vehicle_id,
                selected_step=self.step_index,
                action_family=bid.mode.family,
                action_value=bid.mode.value,
                mode_index=bid.mode.index,
            )
            old = self.members.get(lane)
            changes.append(
                {
                    "lane_id": lane,
                    "old_member": None if old is None else old.vehicle_id,
                    "new_member": bid.candidate.vehicle_id,
                    "reason": "selected" if old is None else "handoff",
                    "hold_steps": 0 if old is None else self.step_index - old.selected_step,
                    "mode_index": bid.mode.index,
                    "mode": bid.mode.name,
                }
            )
        for row in bid_rows:
            vehicle_id = row["vehicle_id"]
            if any(bid.candidate.vehicle_id == vehicle_id for bid in winners.values()):
                row["outcome"] = "selected"
            elif vehicle_id in rejected:
                row["outcome"] = rejected[vehicle_id]

        self.members = retained
        current = {state.vehicle_id for state in retained.values()}
        for vehicle_id in set(former) - current:
            lane_id, state = former[vehicle_id]
            self._begin_release(
                state,
                lane_id,
                use_projected=vehicle_id in projected_release_ids,
            )
        return changes, current, rejected, bid_rows

    def preview_decentralized_state(
        self, cav_ids: Iterable[str]
    ) -> tuple[set[str], dict[str, tuple[int, ...]], dict[str, dict[str, Any]]]:
        """Preview local eligibility and masks without advancing controller state."""
        cav_set = set(cav_ids)
        samples = self._samples()
        by_id = {sample.vehicle_id: sample for sample in samples}
        disturbances = detect_disturbances(samples, self.cfg)
        candidates, admissions = self._decentralized_candidate_state(samples, cav_set, disturbances)
        next_step = self.step_index + 1
        latched: dict[str, int] = {}
        retained_lanes: set[str] = set()
        blocked_release_lanes = (
            self._projected_release_blocked_lanes(next_step)
            if self._uses_release_lane_blocking()
            else set()
        )
        for lane, state in self.members.items():
            sample = by_id.get(state.vehicle_id)
            valid = (
                sample is not None
                and sample.lane_id == lane
                and sample.road_id == PRE_MERGE_EDGE
                and self._candidate_admission(state.vehicle_id).get("admissible", False)
            )
            if (
                valid
                and state.mode_index is not None
                and next_step - state.selected_step < self.cfg.decentralized_option_steps
            ):
                latched[state.vehicle_id] = state.mode_index
                retained_lanes.add(lane)
            elif (
                self._uses_release_lane_blocking()
                and valid
                and next_step - state.selected_step >= self.cfg.decentralized_option_steps
            ):
                # The next apply call moves this member into projected release;
                # preview must already hide every replacement bid in its lane.
                blocked_release_lanes.add(lane)
        # A retained option reserves its lane.  Backup candidates in that lane
        # cannot affect _update_decentralized_members until the option expires,
        # so exposing both actions to them would create phantom PPO decisions.
        eligible = {
            candidate.vehicle_id
            for lane, lane_candidates in candidates.items()
            if lane not in retained_lanes and lane not in blocked_release_lanes
            for candidate in lane_candidates
            if admissions.get(candidate.vehicle_id, {}).get("admissible", False)
        }
        masks = {
            vehicle_id: action_mask(
                eligible=vehicle_id in eligible,
                latched_mode=latched.get(vehicle_id),
                action_set=self.cfg.decentralized_action_set,
            )
            for vehicle_id in cav_set
        }
        return eligible | set(latched), masks, admissions

    def preview_active_members(self, cav_ids: Iterable[str]) -> set[str]:
        """Return the members that the next :meth:`apply_all` will control.

        This read-only observation hook mirrors membership retention, safety
        admission, and deterministic handoff at ``step_index + 1`` without
        advancing filters or latches.
        """
        if self.cfg.decentralized_damping_enabled:
            eligible, _, _ = self.preview_decentralized_state(cav_ids)
            return eligible
        samples = self._samples()
        by_id = {sample.vehicle_id: sample for sample in samples}
        proposed = select_absorbers(
            samples, cav_ids, detect_disturbances(samples, self.cfg), self.cfg
        )
        next_step = self.step_index + 1
        retained: dict[str, str] = {}
        for lane, state in sorted(self.members.items()):
            sample = by_id.get(state.vehicle_id)
            age = next_step - state.selected_step
            valid = (
                sample is not None
                and sample.lane_id == lane
                and sample.road_id == PRE_MERGE_EDGE
                and self._candidate_safe(state.vehicle_id)
            )
            proposed_id = proposed.get(lane)
            if valid and (age < self.cfg.minimum_hold_steps or proposed_id == state.vehicle_id):
                retained[lane] = state.vehicle_id
        for lane, vehicle_id in sorted(proposed.items()):
            if lane not in retained and self._candidate_safe(vehicle_id):
                retained[lane] = vehicle_id
        return set(retained.values())

    def _reference(
        self,
        member: VehicleSample,
        samples: Iterable[VehicleSample],
        disturbance: Optional[Disturbance] = None,
        disturbance_percentile: str = "p30",
    ) -> tuple[float, dict[str, Any]]:
        downstream = [
            sample
            for sample in samples
            if sample.lane_id == member.lane_id
            and self.cfg.preview_min_m
            <= member.distance_to_merge_m - sample.distance_to_merge_m
            <= self.cfg.preview_max_m
        ]
        source = downstream if downstream else [member]
        if self.cfg.reference_mode == "disturbance_p30":
            percentile_value = {
                "p30": self.cfg.reference_percentile,
                "p50": 50.0,
                "p70": 70.0,
            }.get(disturbance_percentile)
            if percentile_value is None:
                raise ValueError(f"unknown disturbance percentile {disturbance_percentile!r}")
            if disturbance is None:
                raw = float(
                    np.percentile(
                        [sample.speed_mps for sample in source],
                        percentile_value,
                    )
                )
                source_mode = f"preview_{disturbance_percentile}_fallback"
            else:
                raw = float(getattr(disturbance, f"speed_{disturbance_percentile}_mps"))
                source_mode = f"disturbance_{disturbance_percentile}"
        elif self.cfg.reference_mode == "preview_p30":
            raw = float(
                np.percentile(
                    [sample.speed_mps for sample in source],
                    self.cfg.reference_percentile,
                )
            )
            source_mode = "preview_p30"
        else:
            raise ValueError(f"unknown reference_mode {self.cfg.reference_mode!r}")
        try:
            allowed = float(traci.vehicle.getAllowedSpeed(member.vehicle_id))
        except Exception:
            allowed = 35.0
        shadow_p50_raw = float(
            disturbance.speed_p50_mps
            if disturbance is not None
            else np.percentile([sample.speed_mps for sample in source], 50.0)
        )
        shadow_p50_raw = float(np.clip(shadow_p50_raw, self.cfg.reference_floor_mps, allowed))
        raw = float(np.clip(raw, self.cfg.reference_floor_mps, allowed))
        previous = self.reference_by_lane.get(member.lane_id, raw)
        alpha = min(1.0, self.cfg.dt_s / max(self.cfg.reference_filter_tau_s, self.cfg.dt_s))
        filtered = previous + alpha * (raw - previous)
        rate = self.cfg.reference_rate_limit_mps_per_s * self.cfg.dt_s
        filtered = float(np.clip(filtered, previous - rate, previous + rate))
        self.reference_by_lane[member.lane_id] = filtered
        return filtered, {
            "raw_reference_mps": raw,
            "shadow_p50_raw_reference_mps": shadow_p50_raw,
            "source_mode": source_mode,
            "filtered_reference_mps": filtered,
            "source_vehicle_count": len(source),
            "source_min_distance_m": min(
                (member.distance_to_merge_m - s.distance_to_merge_m for s in source),
                default=0.0,
            ),
            "source_max_distance_m": max(
                (member.distance_to_merge_m - s.distance_to_merge_m for s in source),
                default=0.0,
            ),
            "filter_tau_s": self.cfg.reference_filter_tau_s,
        }

    def _command_member(
        self,
        sample: VehicleSample,
        action_value: float,
        samples: Iterable[VehicleSample],
        disturbance: Optional[Disturbance] = None,
        action_family: Optional[str] = None,
        mode_index: Optional[int] = None,
        active_age_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        context = self._leader_follower(sample.vehicle_id)
        min_gap = 2.5
        try:
            min_gap = float(traci.vehicle.getMinGap(sample.vehicle_id))
        except Exception:
            pass
        family = action_family or self.cfg.action_family
        disturbance_percentile = {
            "idm_p30_projected_blend": "p30",
            "modified_followerstopper_p30_ceiling": "p30",
            "rear_aware_feedback_pulse": "p50",
            "speed_ceiling": "p30",
            "speed_ceiling_p50": "p50",
            "speed_ceiling_p70": "p70",
        }.get(family, "p30")
        reference, ref_diag = self._reference(
            sample,
            samples,
            disturbance=disturbance,
            disturbance_percentile=disturbance_percentile,
        )
        nominal = analytical_acceleration(
            speed_mps=sample.speed_mps,
            reference_speed_mps=reference,
            gap_m=context["leader_gap_m"],
            min_gap_m=min_gap,
            leader_speed_mps=context["leader_speed_mps"],
            cfg=self.cfg,
        )
        follower_stopper_diag = None
        projected_blend_diag = None
        rear_aware_diag = None
        rear_floor = None
        rear_floor_feasible = True
        follower_acceleration = None
        if family in {
            "acceleration_cap",
            "idm_p30_projected_blend",
            "modified_followerstopper_p30_ceiling",
            "rear_aware_feedback_pulse",
            "speed_ceiling",
            "speed_ceiling_p50",
            "speed_ceiling_p70",
        }:
            requested_action = float(action_value)
        else:
            requested_action = clip_action(
                family,
                action_value,
                residual_min_mps2=self.cfg.residual_min_mps2,
                residual_max_mps2=self.cfg.residual_max_mps2,
                bounds=StructuredActionBounds(
                    gain_min=self.cfg.gain_min,
                    gain_max=self.cfg.gain_max,
                    reference_offset_min_mps=self.cfg.reference_offset_min_mps,
                    reference_offset_max_mps=self.cfg.reference_offset_max_mps,
                ),
            )
        effective_reference = reference
        if family == "residual" and self.cfg.filter_tau_s > 0.0:
            alpha = self.cfg.dt_s / (self.cfg.filter_tau_s + self.cfg.dt_s)
            previous = self.filtered_residual.get(sample.vehicle_id, 0.0)
            filtered_action = float(previous + alpha * (requested_action - previous))
            self.filtered_residual[sample.vehicle_id] = filtered_action
        elif family == "residual":
            filtered_action = requested_action
        elif family in {"gain", "reference_offset"}:
            filter_key = (sample.lane_id, family)
            previous = self.filtered_structured_action_by_lane.get(
                filter_key, nominal_action(family)
            )
            filtered_action = low_pass_action(
                previous,
                requested_action,
                dt_s=self.cfg.dt_s,
                tau_s=self.cfg.structured_action_filter_tau_s,
            )
            self.filtered_structured_action_by_lane[filter_key] = filtered_action
        if family == "residual":
            residual = filtered_action
            self.filtered_residual[sample.vehicle_id] = residual
            desired_acceleration = nominal + residual
        elif family == "gain":
            desired_acceleration = gain_desired_acceleration(nominal, filtered_action)
            residual = desired_acceleration - nominal
        elif family == "reference_offset":
            try:
                allowed_for_reference = float(traci.vehicle.getAllowedSpeed(sample.vehicle_id))
            except Exception:
                allowed_for_reference = 35.0
            effective_reference = adjusted_reference_speed(
                reference,
                filtered_action,
                floor_mps=self.cfg.reference_floor_mps,
                allowed_speed_mps=allowed_for_reference,
            )
            desired_acceleration = analytical_acceleration(
                speed_mps=sample.speed_mps,
                reference_speed_mps=effective_reference,
                gap_m=context["leader_gap_m"],
                min_gap_m=min_gap,
                leader_speed_mps=context["leader_speed_mps"],
                cfg=self.cfg,
            )
            residual = desired_acceleration - nominal
        elif family == "acceleration_cap":
            filtered_action = requested_action
            desired_acceleration = min(nominal, filtered_action)
            residual = desired_acceleration - nominal
        elif family == "idm_p30_projected_blend":
            filtered_action = requested_action
            try:
                allowed_for_blend = float(traci.vehicle.getAllowedSpeed(sample.vehicle_id))
            except Exception:
                allowed_for_blend = 35.0
            try:
                native_acceleration_limit = max(
                    0.0, float(traci.vehicle.getAccel(sample.vehicle_id))
                )
            except Exception:
                native_acceleration_limit = self.cfg.comfortable_accel_mps2
            native_follow_speed = None
            if context["leader_id"] is not None:
                try:
                    leader_decel = float(traci.vehicle.getDecel(context["leader_id"]))
                    native_follow_speed = float(
                        traci.vehicle.getFollowSpeed(
                            sample.vehicle_id,
                            sample.speed_mps,
                            context["leader_gap_m"],
                            context["leader_speed_mps"],
                            leader_decel,
                            context["leader_id"],
                        )
                    )
                except Exception:
                    native_follow_speed = None
            native_target_speed, native_raw_acceleration = native_proxy_acceleration(
                current_speed_mps=sample.speed_mps,
                allowed_speed_mps=allowed_for_blend,
                maximum_acceleration_mps2=native_acceleration_limit,
                follow_speed_mps=native_follow_speed,
                dt_s=self.cfg.dt_s,
            )
            p30_ceiling_acceleration = (reference - sample.speed_mps) / max(self.cfg.dt_s, 1e-6)
            p30_raw_acceleration = min(nominal, p30_ceiling_acceleration)
            prior_acceleration = self.prior_acceleration.get(
                sample.vehicle_id, sample.acceleration_mps2
            )

            def project_endpoint(desired: float):
                return project_command(
                    ProjectionInput(
                        current_speed_mps=sample.speed_mps,
                        desired_acceleration_mps2=desired,
                        prior_acceleration_mps2=prior_acceleration,
                        allowed_speed_mps=allowed_for_blend,
                        leader_gap_m=context["leader_gap_m"],
                        leader_speed_mps=context["leader_speed_mps"],
                        follow_speed_mps=native_follow_speed,
                        follower_gap_m=context["follower_gap_m"],
                        follower_speed_mps=context["follower_speed_mps"],
                    ),
                    self.cfg,
                )

            native_endpoint = project_endpoint(native_raw_acceleration)
            p30_endpoint = project_endpoint(p30_raw_acceleration)
            desired_acceleration = projected_blend_acceleration(
                projected_native_acceleration_mps2=(native_endpoint.acceleration_mps2),
                projected_p30_acceleration_mps2=(p30_endpoint.acceleration_mps2),
                strength=filtered_action,
            )
            residual = desired_acceleration - nominal
            try:
                reported_native_speed = float(traci.vehicle.getSpeedWithoutTraCI(sample.vehicle_id))
                if not isfinite(reported_native_speed):
                    reported_native_speed = None
            except Exception:
                reported_native_speed = None
            projected_blend_diag = {
                "schema": IDM_P30_PROJECTED_BLEND_V1.schema,
                "contract": IDM_P30_PROJECTED_BLEND_V1.to_dict(),
                "strength": filtered_action,
                "native_proxy": {
                    "allowed_speed_mps": allowed_for_blend,
                    "maximum_acceleration_mps2": native_acceleration_limit,
                    "follow_speed_mps": native_follow_speed,
                    "follow_speed_available": native_follow_speed is not None,
                    "target_speed_mps": native_target_speed,
                    "raw_acceleration_mps2": native_raw_acceleration,
                    "reported_speed_without_traci_mps": reported_native_speed,
                },
                "p30": {
                    "reference_speed_mps": reference,
                    "analytical_acceleration_mps2": nominal,
                    "ceiling_acceleration_mps2": p30_ceiling_acceleration,
                    "raw_acceleration_mps2": p30_raw_acceleration,
                },
                "projected_native_endpoint": asdict(native_endpoint),
                "projected_p30_endpoint": asdict(p30_endpoint),
                "endpoint_span_mps2": abs(
                    native_endpoint.acceleration_mps2 - p30_endpoint.acceleration_mps2
                ),
                "mixed_projected_endpoint_acceleration_mps2": (desired_acceleration),
            }
        elif family == "rear_aware_feedback_pulse":
            filtered_action = requested_action
            age = int(active_age_steps or 1)
            feedback_contract = self._rear_aware_feedback_contract()
            strength = treatment_strength(age, feedback_contract)
            try:
                allowed_for_reference = float(traci.vehicle.getAllowedSpeed(sample.vehicle_id))
            except Exception:
                allowed_for_reference = 35.0
            effective_reference = adjusted_reference_speed(
                reference,
                0.0,
                floor_mps=self.cfg.reference_floor_mps,
                allowed_speed_mps=allowed_for_reference,
            )
            feedback = modified_follower_stopper_speed(
                ego_speed_mps=sample.speed_mps,
                reference_speed_mps=effective_reference,
                leader_gap_m=context["leader_gap_m"],
                leader_speed_mps=context["leader_speed_mps"],
                contract=MODIFIED_FOLLOWER_STOPPER_P30_V1,
            )
            feedback_raw_acceleration = (feedback.desired_speed_mps - sample.speed_mps) / max(
                self.cfg.dt_s, 1e-6
            )
            try:
                native_acceleration_limit = max(
                    0.0, float(traci.vehicle.getAccel(sample.vehicle_id))
                )
            except Exception:
                native_acceleration_limit = self.cfg.comfortable_accel_mps2
            try:
                reported_native_speed = float(traci.vehicle.getSpeedWithoutTraCI(sample.vehicle_id))
                if not isfinite(reported_native_speed):
                    reported_native_speed = None
            except Exception:
                reported_native_speed = None
            native_follow_speed = None
            if context["leader_id"] is not None:
                try:
                    leader_decel = float(traci.vehicle.getDecel(context["leader_id"]))
                    native_follow_speed = float(
                        traci.vehicle.getFollowSpeed(
                            sample.vehicle_id,
                            sample.speed_mps,
                            context["leader_gap_m"],
                            context["leader_speed_mps"],
                            leader_decel,
                            context["leader_id"],
                        )
                    )
                except Exception:
                    native_follow_speed = None
            native_proxy_speed, native_proxy_raw_acceleration = native_proxy_acceleration(
                current_speed_mps=sample.speed_mps,
                allowed_speed_mps=allowed_for_reference,
                maximum_acceleration_mps2=native_acceleration_limit,
                follow_speed_mps=native_follow_speed,
                dt_s=self.cfg.dt_s,
            )
            native_target_speed = (
                min(max(reported_native_speed, 0.0), allowed_for_reference)
                if reported_native_speed is not None
                else native_proxy_speed
            )
            native_raw_acceleration = (native_target_speed - sample.speed_mps) / max(
                self.cfg.dt_s, 1e-6
            )
            prior_acceleration = self.prior_acceleration.get(
                sample.vehicle_id, sample.acceleration_mps2
            )

            rear_parameter_available = True
            if (
                context["follower_id"] is not None
                and context["follower_gap_m"] is not None
                and context["follower_speed_mps"] is not None
            ):
                follower_id = str(context["follower_id"])
                try:
                    ego_decel = float(traci.vehicle.getDecel(sample.vehicle_id))
                    follower_decel = float(traci.vehicle.getDecel(follower_id))
                    follower_acceleration = float(traci.vehicle.getAcceleration(follower_id))

                    def follower_response(next_speed_mps: float) -> tuple[float, float]:
                        predicted_speed = float(
                            traci.vehicle.getFollowSpeed(
                                follower_id,
                                float(context["follower_speed_mps"]),
                                float(context["follower_gap_m"]),
                                float(next_speed_mps),
                                ego_decel,
                                sample.vehicle_id,
                            )
                        )
                        secure_gap = float(
                            traci.vehicle.getSecureGap(
                                follower_id,
                                predicted_speed,
                                float(next_speed_mps),
                                ego_decel,
                                sample.vehicle_id,
                            )
                        )
                        return predicted_speed, secure_gap

                    rear_floor = solve_rear_speed_floor(
                        RearFloorInput(
                            current_ego_speed_mps=sample.speed_mps,
                            follower_speed_mps=float(context["follower_speed_mps"]),
                            follower_gap_m=float(context["follower_gap_m"]),
                            minimum_next_speed_mps=max(
                                0.0,
                                sample.speed_mps - self.cfg.comfortable_decel_mps2 * self.cfg.dt_s,
                            ),
                            maximum_next_speed_mps=min(
                                allowed_for_reference,
                                sample.speed_mps + self.cfg.comfortable_accel_mps2 * self.cfg.dt_s,
                            ),
                            dt_s=self.cfg.dt_s,
                            rear_ttc_floor_s=self.cfg.rear_ttc_floor_s,
                            follower_decel_mps2=follower_decel,
                        ),
                        follower_response_at_speed=follower_response,
                    )
                except Exception:
                    rear_parameter_available = False
            rear_floor_feasible = rear_floor is None or (
                rear_parameter_available
                and rear_floor.follower_response_available
                and rear_floor.feasible_within_upper_bound
                and (
                    rear_floor.secure_headroom_m is None
                    # The solver's headroom already subtracts the frozen
                    # five-metre margin. Requiring another five metres here
                    # would silently turn the contract into a ten-metre guard.
                    or rear_floor.secure_headroom_m >= -1e-9
                )
            )

            def project_endpoint(desired: float, *, protect_rear: bool):
                return project_command(
                    ProjectionInput(
                        current_speed_mps=sample.speed_mps,
                        desired_acceleration_mps2=desired,
                        prior_acceleration_mps2=prior_acceleration,
                        allowed_speed_mps=allowed_for_reference,
                        leader_gap_m=context["leader_gap_m"],
                        leader_speed_mps=context["leader_speed_mps"],
                        follow_speed_mps=native_follow_speed,
                        follower_gap_m=context["follower_gap_m"],
                        follower_speed_mps=context["follower_speed_mps"],
                    ),
                    self.cfg,
                    minimum_target_speed_mps=(
                        rear_floor.minimum_next_speed_mps
                        if protect_rear and rear_floor is not None
                        else None
                    ),
                    minimum_speed_reason="rear_secure_gap",
                )

            native_endpoint = project_endpoint(native_raw_acceleration, protect_rear=False)
            feedback_endpoint = project_endpoint(feedback_raw_acceleration, protect_rear=True)
            desired_acceleration = projected_blend_acceleration(
                projected_native_acceleration_mps2=native_endpoint.acceleration_mps2,
                projected_p30_acceleration_mps2=feedback_endpoint.acceleration_mps2,
                strength=strength,
            )
            residual = desired_acceleration - nominal
            rear_aware_diag = {
                "schema": feedback_contract.schema,
                "contract": feedback_contract.to_dict(),
                "active_age_steps": age,
                "phase": ("probe" if age <= feedback_contract.probe_steps else "hold"),
                "follower_id": context["follower_id"],
                "strength": strength,
                "native_target_speed_mps": native_target_speed,
                "reported_native_speed_without_traci_mps": reported_native_speed,
                "native_proxy_target_speed_mps": native_proxy_speed,
                "native_proxy_raw_acceleration_mps2": native_proxy_raw_acceleration,
                "native_raw_acceleration_mps2": native_raw_acceleration,
                "front_feedback": feedback.to_dict(),
                "feedback_raw_acceleration_mps2": feedback_raw_acceleration,
                "projected_native_endpoint": asdict(native_endpoint),
                "projected_feedback_endpoint": asdict(feedback_endpoint),
                "rear_parameter_available": rear_parameter_available,
                "rear_floor": None if rear_floor is None else asdict(rear_floor),
                "rear_floor_feasible": rear_floor_feasible,
            }
        elif family == "modified_followerstopper_p30_ceiling":
            filtered_action = requested_action
            try:
                allowed_for_reference = float(traci.vehicle.getAllowedSpeed(sample.vehicle_id))
            except Exception:
                allowed_for_reference = 35.0
            effective_reference = adjusted_reference_speed(
                reference,
                filtered_action,
                floor_mps=self.cfg.reference_floor_mps,
                allowed_speed_mps=allowed_for_reference,
            )
            feedback = modified_follower_stopper_speed(
                ego_speed_mps=sample.speed_mps,
                reference_speed_mps=effective_reference,
                leader_gap_m=context["leader_gap_m"],
                leader_speed_mps=context["leader_speed_mps"],
                contract=MODIFIED_FOLLOWER_STOPPER_P30_V1,
            )
            ceiling_acceleration = (feedback.desired_speed_mps - sample.speed_mps) / max(
                self.cfg.dt_s, 1e-6
            )
            desired_acceleration = min(nominal, ceiling_acceleration)
            residual = desired_acceleration - nominal
            follower_stopper_diag = {
                **feedback.to_dict(),
                "schema": MODIFIED_FOLLOWER_STOPPER_P30_V1.schema,
                "ceiling_acceleration_mps2": ceiling_acceleration,
                "analytical_acceleration_mps2": nominal,
                "desired_acceleration_mps2": desired_acceleration,
            }
        elif family in {"speed_ceiling", "speed_ceiling_p50", "speed_ceiling_p70"}:
            filtered_action = requested_action
            try:
                allowed_for_reference = float(traci.vehicle.getAllowedSpeed(sample.vehicle_id))
            except Exception:
                allowed_for_reference = 35.0
            effective_reference = adjusted_reference_speed(
                reference,
                filtered_action,
                floor_mps=self.cfg.reference_floor_mps,
                allowed_speed_mps=allowed_for_reference,
            )
            ceiling_acceleration = (effective_reference - sample.speed_mps) / max(
                self.cfg.dt_s, 1e-6
            )
            desired_acceleration = min(nominal, ceiling_acceleration)
            residual = desired_acceleration - nominal
        else:
            raise ValueError(f"unknown structured action family {family!r}")
        follow_speed = None
        secure_gap = None
        if context["leader_id"] is not None:
            try:
                leader_decel = float(traci.vehicle.getDecel(context["leader_id"]))
                follow_speed = float(
                    traci.vehicle.getFollowSpeed(
                        sample.vehicle_id,
                        sample.speed_mps,
                        context["leader_gap_m"],
                        context["leader_speed_mps"],
                        leader_decel,
                        context["leader_id"],
                    )
                )
                secure_gap = float(
                    traci.vehicle.getSecureGap(
                        sample.vehicle_id,
                        sample.speed_mps,
                        context["leader_speed_mps"],
                        leader_decel,
                        context["leader_id"],
                    )
                )
            except Exception:
                pass
        try:
            allowed = float(traci.vehicle.getAllowedSpeed(sample.vehicle_id))
        except Exception:
            allowed = 35.0
        projection_input = ProjectionInput(
            current_speed_mps=sample.speed_mps,
            desired_acceleration_mps2=desired_acceleration,
            prior_acceleration_mps2=self.prior_acceleration.get(
                sample.vehicle_id, sample.acceleration_mps2
            ),
            allowed_speed_mps=allowed,
            leader_gap_m=context["leader_gap_m"],
            leader_speed_mps=context["leader_speed_mps"],
            follow_speed_mps=follow_speed,
            follower_gap_m=context["follower_gap_m"],
            follower_speed_mps=context["follower_speed_mps"],
        )
        projected = project_command(
            projection_input,
            self.cfg,
            minimum_target_speed_mps=(
                rear_floor.minimum_next_speed_mps
                if family == "rear_aware_feedback_pulse" and rear_floor is not None
                else None
            ),
            minimum_speed_reason="rear_secure_gap",
        )
        traci.vehicle.setSpeedMode(sample.vehicle_id, IDM_SAFE_SPEED_MODE)
        traci.vehicle.setSpeed(sample.vehicle_id, projected.target_speed_mps)
        self.prior_acceleration[sample.vehicle_id] = projected.acceleration_mps2
        result = {
            "vehicle_id": sample.vehicle_id,
            "lane_id": sample.lane_id,
            "distance_to_merge_m": sample.distance_to_merge_m,
            "nominal_acceleration_mps2": nominal,
            "action_family": family,
            "mode_index": mode_index,
            "requested_action": requested_action,
            "filtered_action": filtered_action,
            "nominal_action": (
                None
                if family
                in {
                    "acceleration_cap",
                    "idm_p30_projected_blend",
                    "modified_followerstopper_p30_ceiling",
                    "rear_aware_feedback_pulse",
                    "speed_ceiling",
                    "speed_ceiling_p50",
                    "speed_ceiling_p70",
                }
                else nominal_action(family)
            ),
            "requested_residual_mps2": requested_action if family == "residual" else 0.0,
            "residual_mps2": residual,
            "learned_acceleration_delta_mps2": residual,
            "desired_acceleration_mps2": desired_acceleration,
            "current_speed_mps": sample.speed_mps,
            "minimum_gap_m": min_gap,
            "projection_input": asdict(projection_input),
            "projected": asdict(projected),
            "secure_gap_m": secure_gap,
            "leader": context,
            "reference": {
                **ref_diag,
                "effective_reference_mps": effective_reference,
                "requested_offset_mps": requested_action if family == "reference_offset" else 0.0,
                "filtered_offset_mps": filtered_action if family == "reference_offset" else 0.0,
            },
        }
        if follower_stopper_diag is not None:
            result["modified_follower_stopper"] = follower_stopper_diag
            result["treatment_infeasible"] = bool(projected.constraint_conflict)
            result["treatment_abort_reason"] = (
                "projection_interval_conflict" if projected.constraint_conflict else None
            )
            result["rear_bound_active_in_projection_conflict"] = bool(
                projected.constraint_conflict and "follower_ttc" in projected.reasons
            )
        if projected_blend_diag is not None:
            result["projected_blend"] = {
                **projected_blend_diag,
                "final_projected_acceleration_mps2": (projected.acceleration_mps2),
                "final_projection_idempotence_error_mps2": abs(
                    projected.acceleration_mps2 - desired_acceleration
                ),
            }
            endpoint_conflict = any(
                bool(row.constraint_conflict) for row in (native_endpoint, p30_endpoint, projected)
            )
            result["treatment_infeasible"] = endpoint_conflict
            result["treatment_abort_reason"] = (
                "projection_interval_conflict" if endpoint_conflict else None
            )
            result["rear_bound_active_in_projection_conflict"] = bool(
                endpoint_conflict and "follower_ttc" in projected.reasons
            )
        if rear_aware_diag is not None:
            rear_closing_speed = (
                None
                if context["follower_speed_mps"] is None
                else float(context["follower_speed_mps"]) - sample.speed_mps
            )
            response = FeedbackResponse(
                disturbance_present=disturbance is not None,
                disturbance_std_mps=(None if disturbance is None else disturbance.speed_std_mps),
                rear_gap_m=context["follower_gap_m"],
                rear_closing_speed_mps=rear_closing_speed,
                follower_acceleration_mps2=follower_acceleration,
                local_acceleration_energy_m2ps4=(
                    sample.acceleration_mps2**2
                    + (0.0 if follower_acceleration is None else follower_acceleration**2)
                ),
                rear_floor_feasible=rear_floor_feasible,
                projection_survived=(rear_floor_feasible and not projected.constraint_conflict),
                command_delta_from_native_mps2=(
                    projected.acceleration_mps2
                    - float(rear_aware_diag["projected_native_endpoint"]["acceleration_mps2"])
                ),
                follower_id=context["follower_id"],
            )
            result["rear_aware_feedback"] = {
                **rear_aware_diag,
                "current_response": response.to_dict(),
                "final_projected_acceleration_mps2": projected.acceleration_mps2,
            }
            result["treatment_infeasible"] = not response.projection_survived
            result["treatment_abort_reason"] = (
                "rear_floor_infeasible"
                if not rear_floor_feasible
                else "projection_interval_conflict"
                if projected.constraint_conflict
                else None
            )
            result["rear_bound_active_in_projection_conflict"] = bool(
                projected.constraint_conflict
                and any(
                    reason in projected.reasons for reason in ("follower_ttc", "rear_secure_gap")
                )
            )
        return result

    def _release(
        self,
        active_members: set[str],
        *,
        skip_vehicle_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        live = set(traci.vehicle.getIDList())
        skipped = set(skip_vehicle_ids)
        for vehicle_id in sorted(list(self.releasing)):
            if vehicle_id in skipped:
                continue
            if vehicle_id in active_members:
                self.releasing.pop(vehicle_id, None)
                continue
            state = self.releasing[vehicle_id]
            if vehicle_id not in live:
                self.releasing.pop(vehicle_id, None)
                self.prior_acceleration.pop(vehicle_id, None)
                continue
            try:
                speed = float(traci.vehicle.getSpeed(vehicle_id))
                native = float(traci.vehicle.getSpeedWithoutTraCI(vehicle_id))
                desired = (native - speed) / max(self.cfg.dt_s, 1e-6)
                prior = state.prior_acceleration_mps2
                delta = self.cfg.jerk_limit_mps3 * self.cfg.dt_s
                acceleration = float(np.clip(desired, prior - delta, prior + delta))
                acceleration = float(
                    np.clip(
                        acceleration,
                        -self.cfg.comfortable_decel_mps2,
                        self.cfg.comfortable_accel_mps2,
                    )
                )
                target = max(0.0, speed + acceleration * self.cfg.dt_s)
                state.age_steps += 1
                converged = (
                    abs(native - speed) <= self.cfg.release_tolerance_mps
                    and abs(acceleration) <= self.cfg.release_accel_tolerance_mps2
                )
                if converged or state.age_steps >= self.cfg.release_max_steps:
                    traci.vehicle.setSpeed(vehicle_id, -1)
                    self.releasing.pop(vehicle_id, None)
                    self.prior_acceleration.pop(vehicle_id, None)
                    diagnostics.append(
                        {
                            "vehicle_id": vehicle_id,
                            "released_to_idm": True,
                            "age_steps": state.age_steps,
                            "native_speed_mps": native,
                        }
                    )
                else:
                    traci.vehicle.setSpeedMode(vehicle_id, IDM_SAFE_SPEED_MODE)
                    traci.vehicle.setSpeed(vehicle_id, target)
                    state.prior_acceleration_mps2 = acceleration
                    self.prior_acceleration[vehicle_id] = acceleration
                    diagnostics.append(
                        {
                            "vehicle_id": vehicle_id,
                            "released_to_idm": False,
                            "age_steps": state.age_steps,
                            "target_speed_mps": target,
                            "native_speed_mps": native,
                            "acceleration_mps2": acceleration,
                        }
                    )
            except Exception as exc:
                diagnostics.append(
                    {
                        "vehicle_id": vehicle_id,
                        "release_error": type(exc).__name__,
                    }
                )
        return diagnostics

    def _release_projected_rear_aware(
        self,
        active_members: set[str],
        *,
        skip_vehicle_ids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Return the new P30 arm to native IDM through the full projection.

        Frozen P30 continues to use :meth:`_release` unchanged.  This path is
        selected only by the separately named projected-release action set.
        """
        diagnostics: list[dict[str, Any]] = []
        live = set(traci.vehicle.getIDList())
        dt = max(float(self.cfg.dt_s), 1e-6)
        contract = self._projected_release_contract()
        release_schema = contract.schema
        skipped = {str(vehicle_id) for vehicle_id in skip_vehicle_ids}
        for vehicle_id in sorted(list(self.projected_releasing)):
            state = self.projected_releasing[vehicle_id]
            treatment_instance = f"{vehicle_id}.selected{state.selected_step}.lane{state.lane_id}"
            if vehicle_id in skipped:
                diagnostics.append(
                    {
                        "schema": release_schema,
                        "release_contract": release_schema,
                        "treatment_instance_id": treatment_instance,
                        "phase": "projected_release_pending",
                        "vehicle_id": vehicle_id,
                        "lane_id": state.lane_id,
                        "release_deferred_until_next_step": True,
                        "released_to_idm": False,
                        "set_speed_minus_one": False,
                    }
                )
                continue
            if vehicle_id in active_members:
                self.projected_releasing.pop(vehicle_id, None)
                diagnostics.append(
                    {
                        "schema": release_schema,
                        "release_contract": release_schema,
                        "treatment_instance_id": treatment_instance,
                        "phase": "projected_release",
                        "vehicle_id": vehicle_id,
                        "lane_id": state.lane_id,
                        "release_invariant_violation": "reactivated_while_releasing",
                        "released_to_idm": False,
                        "set_speed_minus_one": False,
                    }
                )
                continue
            if vehicle_id not in live:
                self.projected_releasing.pop(vehicle_id, None)
                self.prior_acceleration.pop(vehicle_id, None)
                self._start_projected_release_cooldown(state.lane_id)
                diagnostics.append(
                    {
                        "schema": release_schema,
                        "release_contract": release_schema,
                        "treatment_instance_id": treatment_instance,
                        "phase": "projected_release",
                        "vehicle_id": vehicle_id,
                        "lane_id": state.lane_id,
                        "vehicle_exited": True,
                        "released_to_idm": False,
                        "set_speed_minus_one": False,
                        "age_steps": state.age_steps,
                    }
                )
                continue
            if state.age_steps >= contract.nominal_budget_steps:
                handoff_contract_pass = state.converged_steps >= contract.convergence_steps
                traci.vehicle.setSpeedMode(vehicle_id, IDM_SAFE_SPEED_MODE)
                traci.vehicle.setSpeed(vehicle_id, -1)
                self.projected_releasing.pop(vehicle_id, None)
                self.prior_acceleration.pop(vehicle_id, None)
                self._start_projected_release_cooldown(state.lane_id)
                diagnostics.append(
                    {
                        "schema": release_schema,
                        "release_contract": release_schema,
                        "treatment_instance_id": treatment_instance,
                        "phase": "native_handoff",
                        "vehicle_id": vehicle_id,
                        "lane_id": state.lane_id,
                        "age_steps": state.age_steps,
                        "release_budget_steps": (contract.nominal_budget_steps),
                        "convergence_streak": state.converged_steps,
                        "handoff_contract_pass": handoff_contract_pass,
                        "unmatched_handoff": not handoff_contract_pass,
                        "released_to_idm": True,
                        "set_speed_minus_one": True,
                        "controlled_release_seconds": 0.0,
                    }
                )
                continue
            try:
                speed = float(traci.vehicle.getSpeed(vehicle_id))
                reported_native = float(traci.vehicle.getSpeedWithoutTraCI(vehicle_id))
                allowed = float(traci.vehicle.getAllowedSpeed(vehicle_id))
                native_acceleration_limit = max(0.0, float(traci.vehicle.getAccel(vehicle_id)))
                if not isfinite(speed) or speed < 0.0:
                    raise ValueError("invalid_release_speed")
                if not isfinite(allowed) or allowed <= 0.0:
                    raise ValueError("invalid_release_allowed_speed")
                if not isfinite(native_acceleration_limit):
                    raise ValueError("invalid_release_acceleration_limit")
                if not isfinite(reported_native) or reported_native < 0.0:
                    reported_native = None
                context = self._leader_follower(vehicle_id)

                follow_speed = None
                front_secure_gap = None
                if context["leader_id"] is not None:
                    try:
                        leader_decel = float(traci.vehicle.getDecel(context["leader_id"]))
                        follow_speed = float(
                            traci.vehicle.getFollowSpeed(
                                vehicle_id,
                                speed,
                                float(context["leader_gap_m"]),
                                float(context["leader_speed_mps"]),
                                leader_decel,
                                context["leader_id"],
                            )
                        )
                        front_secure_gap = float(
                            traci.vehicle.getSecureGap(
                                vehicle_id,
                                speed,
                                float(context["leader_speed_mps"]),
                                leader_decel,
                                context["leader_id"],
                            )
                        )
                    except Exception:
                        follow_speed = None
                        front_secure_gap = None

                native = min(
                    max(0.0, allowed),
                    speed + native_acceleration_limit * dt,
                )
                if follow_speed is not None and isfinite(follow_speed):
                    native = min(native, max(0.0, float(follow_speed)))
                native_acceleration = (native - speed) / dt
                release_fraction = release_blend_fraction(
                    state.age_steps + 1,
                    contract,
                )
                blend_origin_acceleration = (
                    0.0
                    if contract.blend_origin == "zero_acceleration"
                    else state.initial_acceleration_mps2
                )
                blended_release_acceleration = (
                    1.0 - release_fraction
                ) * blend_origin_acceleration + release_fraction * native_acceleration

                rear_floor = None
                rear_parameter_available = True
                follower_decel = follower_tau = ego_decel = None
                if (
                    context["follower_id"] is not None
                    and context["follower_gap_m"] is not None
                    and context["follower_speed_mps"] is not None
                ):
                    follower_id = str(context["follower_id"])
                    try:
                        ego_decel = float(traci.vehicle.getDecel(vehicle_id))
                        follower_decel = float(traci.vehicle.getDecel(follower_id))
                        follower_tau = float(traci.vehicle.getTau(follower_id))
                    except Exception:
                        rear_parameter_available = False
                        ego_decel = float(self.cfg.comfortable_decel_mps2)
                        follower_decel = float(self.cfg.comfortable_decel_mps2)
                        follower_tau = None

                    def follower_response(next_speed_mps: float) -> tuple[float, float]:
                        predicted_follower_speed = float(
                            traci.vehicle.getFollowSpeed(
                                follower_id,
                                float(context["follower_speed_mps"]),
                                float(context["follower_gap_m"]),
                                float(next_speed_mps),
                                float(ego_decel),
                                vehicle_id,
                            )
                        )
                        secure_gap = float(
                            traci.vehicle.getSecureGap(
                                follower_id,
                                predicted_follower_speed,
                                float(next_speed_mps),
                                float(ego_decel),
                                vehicle_id,
                            )
                        )
                        return predicted_follower_speed, secure_gap

                    rear_floor = solve_rear_speed_floor(
                        RearFloorInput(
                            current_ego_speed_mps=speed,
                            follower_speed_mps=float(context["follower_speed_mps"]),
                            follower_gap_m=float(context["follower_gap_m"]),
                            minimum_next_speed_mps=max(
                                0.0,
                                speed - self.cfg.comfortable_decel_mps2 * dt,
                            ),
                            maximum_next_speed_mps=min(
                                max(0.0, allowed),
                                speed + self.cfg.comfortable_accel_mps2 * dt,
                            ),
                            dt_s=dt,
                            rear_ttc_floor_s=self.cfg.rear_ttc_floor_s,
                            follower_decel_mps2=float(follower_decel),
                        ),
                        follower_response_at_speed=(
                            follower_response if rear_parameter_available else None
                        ),
                    )

                projection_input = ProjectionInput(
                    current_speed_mps=speed,
                    desired_acceleration_mps2=blended_release_acceleration,
                    prior_acceleration_mps2=state.prior_acceleration_mps2,
                    allowed_speed_mps=allowed,
                    leader_gap_m=context["leader_gap_m"],
                    leader_speed_mps=context["leader_speed_mps"],
                    follow_speed_mps=follow_speed,
                    follower_gap_m=context["follower_gap_m"],
                    follower_speed_mps=context["follower_speed_mps"],
                )
                projected = project_command(
                    projection_input,
                    self.cfg,
                    minimum_target_speed_mps=(
                        None if rear_floor is None else rear_floor.minimum_next_speed_mps
                    ),
                    minimum_speed_reason="rear_secure_gap",
                    minimum_acceleration_mps2=-self.cfg.comfortable_decel_mps2,
                    maximum_acceleration_mps2=native_acceleration_limit,
                )
                rear_feasible = rear_floor is None or (
                    rear_parameter_available
                    and rear_floor.follower_response_available
                    and rear_floor.feasible_within_upper_bound
                )
                converged = handoff_converged(
                    projected_target_speed_mps=projected.target_speed_mps,
                    native_target_speed_mps=native,
                    projected_acceleration_mps2=projected.acceleration_mps2,
                    native_acceleration_mps2=native_acceleration,
                    constraint_conflict=projected.constraint_conflict,
                    rear_feasible=rear_feasible,
                )
                state.converged_steps = state.converged_steps + 1 if converged else 0
                state.age_steps += 1
                budget_complete = state.age_steps >= contract.nominal_budget_steps
                previous_target = state.last_target_speed_mps
                previous_realization_error = (
                    None if previous_target is None else speed - previous_target
                )

                traci.vehicle.setSpeedMode(vehicle_id, IDM_SAFE_SPEED_MODE)
                traci.vehicle.setSpeed(vehicle_id, projected.target_speed_mps)
                state.prior_acceleration_mps2 = projected.acceleration_mps2
                state.last_target_speed_mps = projected.target_speed_mps
                self.prior_acceleration[vehicle_id] = projected.acceleration_mps2

                diagnostics.append(
                    {
                        "schema": release_schema,
                        "release_contract": release_schema,
                        "treatment_instance_id": treatment_instance,
                        "phase": "projected_release",
                        "vehicle_id": vehicle_id,
                        "lane_id": state.lane_id,
                        "age_steps": state.age_steps,
                        "selected_step": state.selected_step,
                        "current_speed_mps": speed,
                        "native_speed_mps": native,
                        "reported_native_speed_without_traci_mps": reported_native,
                        "native_acceleration_limit_mps2": native_acceleration_limit,
                        "native_acceleration_mps2": native_acceleration,
                        "release_fraction": release_fraction,
                        "initial_release_acceleration_mps2": (state.initial_acceleration_mps2),
                        "blend_origin_acceleration_mps2": (blend_origin_acceleration),
                        "blended_release_acceleration_mps2": (blended_release_acceleration),
                        "target_speed_mps": projected.target_speed_mps,
                        "acceleration_mps2": projected.acceleration_mps2,
                        "previous_target_speed_mps": previous_target,
                        "previous_realization_error_mps": previous_realization_error,
                        "projection_input": asdict(projection_input),
                        "projected": asdict(projected),
                        "leader": context,
                        "front_secure_gap_m": front_secure_gap,
                        "follower_decel_mps2": follower_decel,
                        "follower_tau_s": follower_tau,
                        "ego_decel_mps2": ego_decel,
                        "rear_parameter_available": rear_parameter_available,
                        "rear_floor": (None if rear_floor is None else asdict(rear_floor)),
                        "rear_feasible": rear_feasible,
                        "converged_this_step": converged,
                        "convergence_streak": state.converged_steps,
                        "release_budget_complete": budget_complete,
                        "handoff_ready_next_step": (
                            budget_complete and state.converged_steps >= contract.convergence_steps
                        ),
                        "released_to_idm": False,
                        "set_speed_minus_one": False,
                        "controlled_release_seconds": dt,
                    }
                )
            except Exception as exc:
                state.converged_steps = 0
                state.age_steps += 1
                budget_complete = state.age_steps >= contract.nominal_budget_steps
                diagnostics.append(
                    {
                        "schema": release_schema,
                        "release_contract": release_schema,
                        "treatment_instance_id": treatment_instance,
                        "phase": "projected_release",
                        "vehicle_id": vehicle_id,
                        "lane_id": state.lane_id,
                        "age_steps": state.age_steps,
                        "release_error": type(exc).__name__,
                        "release_budget_complete": budget_complete,
                        "released_to_idm": False,
                        "set_speed_minus_one": False,
                    }
                )
        return diagnostics

    def apply_all(
        self,
        residual_actions: Mapping[str, object],
        cav_ids: Iterable[str],
    ) -> dict[str, Any]:
        """Apply one analytical/residual action step and return diagnostics."""
        self.step_index += 1
        samples = self._samples()
        by_id = {sample.vehicle_id: sample for sample in samples}
        disturbances = detect_disturbances(samples, self.cfg)
        candidate_rows: list[dict[str, Any]] = []
        bid_rows: list[dict[str, Any]] = []
        contention_rejections: dict[str, str] = {}
        if self.cfg.decentralized_damping_enabled:
            candidates, admissions = self._decentralized_candidate_state(
                samples, cav_ids, disturbances
            )
            for lane_candidates in candidates.values():
                for candidate in lane_candidates:
                    admission = dict(admissions[candidate.vehicle_id])
                    candidate_rows.append(
                        {
                            "vehicle_id": candidate.vehicle_id,
                            "lane_id": candidate.lane_id,
                            "rank": candidate.rank,
                            "runway_m": candidate.runway_m,
                            "distance_to_merge_m": candidate.distance_to_merge_m,
                            **admission,
                        }
                    )
            changes, active, contention_rejections, bid_rows = self._update_decentralized_members(
                candidates,
                admissions,
                residual_actions,
                by_id,
            )
            proposed = {lane: state.vehicle_id for lane, state in self.members.items()}
        else:
            proposed = select_absorbers(samples, cav_ids, disturbances, self.cfg)
            changes, active = self._update_members(proposed, by_id)
        commands: list[dict[str, Any]] = []
        for lane, state in sorted(self.members.items()):
            sample = by_id.get(state.vehicle_id)
            if sample is None:
                continue
            if self.cfg.decentralized_damping_enabled:
                command_value = float(state.action_value or 0.0)
                command_family = state.action_family
            else:
                raw = residual_actions.get(state.vehicle_id, 0.0)
                try:
                    command_value = float(np.asarray(raw, dtype=np.float32).reshape(-1)[0])
                except Exception:
                    command_value = 0.0
                command_family = None
            if self.cfg.decentralized_damping_enabled:
                command = self._command_member(
                    sample,
                    command_value,
                    samples,
                    disturbance=disturbances.get(lane),
                    action_family=command_family,
                    mode_index=state.mode_index,
                    active_age_steps=self.step_index - state.selected_step + 1,
                )
            else:
                # Keep the frozen residual/gain/reference call contract exact.
                command = self._command_member(
                    sample,
                    command_value,
                    samples,
                    disturbance=disturbances.get(lane),
                )
            commands.append(command)
        treatment_safety_aborts: list[dict[str, Any]] = []
        feedback_releases: list[dict[str, Any]] = []
        feedback_release_ids: set[str] = set()
        if (
            self.cfg.decentralized_damping_enabled
            and self.cfg.decentralized_action_set
            in {
                ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
                ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
            }
        ):
            feedback_contract = self._rear_aware_feedback_contract()
            safety_release_reasons = {
                "rear_floor_infeasible",
                "projection_conflict",
                "severe_follower_braking",
                "rear_gap_compression",
                "rear_closing_deterioration",
            }
            for command in commands:
                feedback = command.get("rear_aware_feedback")
                if not feedback:
                    continue
                vehicle_id = str(command["vehicle_id"])
                lane = str(command["lane_id"])
                state = self.members.get(lane)
                if state is None or state.vehicle_id != vehicle_id:
                    continue
                response = FeedbackResponse(**feedback["current_response"])
                if state.feedback_baseline is None:
                    state.feedback_baseline = response
                state.feedback_projection_survival_count += int(response.projection_survived)
                state.feedback_meaningful_command_count += int(
                    abs(response.command_delta_from_native_mps2)
                    >= feedback_contract.meaningful_command_delta_mps2
                )
                age = int(feedback["active_age_steps"])
                reason = feedback_release_reason(
                    active_age_steps=age,
                    baseline=state.feedback_baseline,
                    current=response,
                    projection_survival_count=(state.feedback_projection_survival_count),
                    meaningful_command_count=(state.feedback_meaningful_command_count),
                    contract=feedback_contract,
                )
                feedback.update(
                    {
                        "projection_survival_count": (state.feedback_projection_survival_count),
                        "meaningful_command_count": (state.feedback_meaningful_command_count),
                        "release_reason": reason,
                        "hold_authorized": (
                            age >= feedback_contract.probe_steps and reason is None
                        ),
                    }
                )
                if reason is None:
                    continue
                self.members.pop(lane, None)
                active.discard(vehicle_id)
                self._begin_release(state, lane, use_projected=True)
                feedback_release_ids.add(vehicle_id)
                release_row = {
                    "vehicle_id": vehicle_id,
                    "lane_id": lane,
                    "reason": reason,
                    "selected_step": state.selected_step,
                    "active_steps": age,
                    "probe_completed": age >= feedback_contract.probe_steps,
                    "release_deferred_until_next_step": True,
                }
                feedback_releases.append(release_row)
                changes.append(
                    {
                        "lane_id": lane,
                        "old_member": vehicle_id,
                        "new_member": None,
                        "reason": "feedback_early_release",
                        "reason_details": (reason,),
                        "hold_steps": age,
                        "mode_index": state.mode_index,
                    }
                )
                if reason in safety_release_reasons:
                    treatment_safety_aborts.append(release_row)
        if self.cfg.decentralized_damping_enabled and self.cfg.decentralized_action_set in {
            ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
            ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
        }:
            abort_ids = {
                str(command["vehicle_id"])
                for command in commands
                if bool(command.get("treatment_infeasible", False))
            }
            for lane, state in sorted(list(self.members.items())):
                if state.vehicle_id not in abort_ids:
                    continue
                self.members.pop(lane, None)
                active.discard(state.vehicle_id)
                self._begin_release(state, lane, use_projected=False)
                self._start_feedback_release_cooldown(lane)
                abort = {
                    "vehicle_id": state.vehicle_id,
                    "lane_id": lane,
                    "reason": "projection_interval_conflict",
                    "selected_step": state.selected_step,
                    "hold_steps": self.step_index - state.selected_step,
                    "release_deferred_until_next_step": True,
                }
                treatment_safety_aborts.append(abort)
                changes.append(
                    {
                        "lane_id": lane,
                        "old_member": state.vehicle_id,
                        "new_member": None,
                        "reason": "runtime_projection_conflict",
                        "reason_details": ("projection_interval_conflict",),
                        "hold_steps": self.step_index - state.selected_step,
                        "mode_index": state.mode_index,
                    }
                )
        else:
            abort_ids = set()
        legacy_release = (
            self._release(active, skip_vehicle_ids=abort_ids)
            if abort_ids
            else self._release(active)
        )
        if self._uses_projected_release():
            # Unsafe/exited members retain the frozen native fail-safe in
            # ``releasing``; only normally completed P30 options enter the
            # separately controlled projected-release state.
            projected_release = (
                self._release_projected_rear_aware(
                    active,
                    skip_vehicle_ids=feedback_release_ids,
                )
                if feedback_release_ids
                else self._release_projected_rear_aware(active)
            )
            release = legacy_release + projected_release
        else:
            release = legacy_release
        diag = {
            "step": self.step_index,
            "disturbances": {lane: asdict(value) for lane, value in disturbances.items()},
            "proposed_members": dict(proposed),
            "active_members": {lane: state.vehicle_id for lane, state in self.members.items()},
            "member_changes": changes,
            "decentralized_damping_enabled": self.cfg.decentralized_damping_enabled,
            "candidate_admissions": candidate_rows,
            "local_bids": bid_rows,
            "contention_rejections": contention_rejections,
            "commands": commands,
            "release": release,
            "feedback_releases": feedback_releases,
            "treatment_safety_aborts": treatment_safety_aborts,
            "simultaneous_absorber_count": len(commands),
            "controlled_cav_seconds": len(commands) * self.cfg.dt_s,
        }
        self.last_diagnostics = diag
        self.history.append(diag)
        return diag
