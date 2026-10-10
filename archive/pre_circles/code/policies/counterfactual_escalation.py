"""Damping-preserving temporary moving-wall escalation.

The causal contract is deliberately narrow:

``DAMPING_ONLY``
    Return the frozen decentralized damping actions unchanged.

``DAMPING_PLUS_WALL``
    Compute the same damping actions for every active CAV, directly override
    only formation-feasible CAVs inside the frozen wall band, and return the
    original damping actions for every other CAV.

The wall has no health trigger.  An external replay/deployment schedule issues
the explicit command.  Ending or aborting a pulse never calls
``release_to_sumo``; the next environment step applies the already-computed
damping actions again, preserving the damping controller and its state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import traci

from config.constants import PRE_MERGE_EDGE
from env.features import distance_to_merge
from policies import Policy
from policies.inflow_control import WallHoldPolicy
from policies.rllib_policy import RLlibPolicy


class EscalationCommand(IntEnum):
    DAMPING_ONLY = 0
    DAMPING_PLUS_WALL = 1


DAMPING_ONLY = EscalationCommand.DAMPING_ONLY
DAMPING_PLUS_WALL = EscalationCommand.DAMPING_PLUS_WALL


class TreatmentState(str, Enum):
    """Explicit lifecycle for one externally requested wall pulse."""

    DAMPING = "DAMPING"
    ASSEMBLING = "ASSEMBLING"
    SETTLING = "SETTLING_TO_LOW_AUTHORITY"
    HOLDING = "HOLDING"
    TAPERING = "TAPERING"
    RELEASING = "RELEASING"
    RELEASED = "RELEASED"
    COOLDOWN = "COOLDOWN"


class WallGeometry(str, Enum):
    """Predeclared geometry used to recruit and retain wall members."""

    STATIC_ALL = "static_all"
    COMPACT_MOVING = "compact_moving"


class CompactControlMode(str, Enum):
    """Longitudinal controller used after compact members are selected."""

    FIXED_TARGET = "fixed_target"
    ADAPTIVE_SERVO = "adaptive_servo"


DEFAULT_DAMPING_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "rllib"
    / "shepherd_pruned_v1_dense_mpen"
    / "checkpoints"
    / "best"
)

DEFAULT_WALL_PARAMS: Dict[str, float] = {
    "d_lo": 200.0,
    "d_hi": 400.0,
    "wall_speed": 10.0,
    "wall_tau": 3.0,
    "a_max_dec": 0.3,
    "tau_ramp_step": 0.05,
}

DEFAULT_ACTIVATION_SAFETY_PARAMS: Dict[str, Any] = {
    "activation_acceleration_floor_mps2": -3.0,
    "activation_min_front_gap_m": None,
    "activation_min_rear_gap_m": None,
    "activation_min_front_ttc_s": 2.0,
    "activation_min_rear_ttc_s": 3.0,
    "monitor_controlled_followers": True,
}


def default_protocol_metadata() -> Dict[str, Any]:
    return {
        "profile": "counterfactual_assemble_hold_release_v2",
        "wall_params": dict(DEFAULT_WALL_PARAMS),
        "membership": {
            "activation_lane_set": "frozen",
            "initial_members": "all_active_cavs_in_frozen_band",
            "member_slots": "frozen_count_per_activation_lane",
            "handoff": "deterministic_same_lane_activation_safe",
            "newly_occupied_lanes": "diagnostic_only",
            "runtime_formation_loss": "dose_diagnostic_not_abort",
        },
        "release": {
            "controller": "identical_frozen_damping",
            "release_to_sumo": False,
        },
        "activation_safety": dict(DEFAULT_ACTIVATION_SAFETY_PARAMS),
        "runtime_abort": {
            "hard_brake_threshold_mps2": -4.5,
            "abort_on_collision": True,
            "abort_on_teleport": True,
            "cooldown_steps": 30,
        },
    }


_FORBIDDEN_AUTONOMOUS_WALL_OPTIONS = (
    "trigger_h",
    "trigger_v",
    "trigger_k",
    "arm_time",
    "release_time",
    "assembly_hi",
    "ramp_pace",
)


@dataclass(frozen=True)
class FormationStatus:
    feasible: bool
    reason: str
    controlled_cav_ids: tuple[str, ...]
    cav_lane_ids: tuple[str, ...]
    occupied_lane_ids: tuple[str, ...]
    required_lane_ids: tuple[str, ...]
    longitudinal_span_m: float
    predicted_max_span_m: Optional[float] = None
    predicted_terminal_span_m: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivationSafetyStatus:
    safe: bool
    reason: str
    checked_cav_ids: tuple[str, ...]
    violations: tuple[str, ...]
    min_acceleration_mps2: Optional[float]
    min_front_gap_m: Optional[float]
    min_rear_gap_m: Optional[float]
    min_front_ttc_s: Optional[float]
    min_rear_ttc_s: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CounterfactualEscalationPolicy(Policy):
    """Compose a frozen damping policy with an explicit temporary wall pulse."""

    direct_control = False

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_DAMPING_CHECKPOINT,
        *,
        base_policy: Optional[Policy] = None,
        wall_params: Optional[Mapping[str, Any]] = None,
        # A separately versioned scenario may add an upstream approach edge.
        # Offsets map each eligible approach edge to its downstream distance
        # from the merge.  The historical default is deliberately unchanged.
        approach_edge_distance_offsets_m: Optional[Mapping[str, float]] = None,
        wall_factory: Callable[..., WallHoldPolicy] = WallHoldPolicy,
        min_wall_cavs: int = 1,
        min_wall_lanes: int = 1,
        require_all_occupied_lanes: bool = True,
        required_lane_ids: Optional[Sequence[str]] = None,
        max_longitudinal_span_m: Optional[float] = None,
        geometry_mode: str | WallGeometry = WallGeometry.STATIC_ALL,
        planned_treatment_duration_s: Optional[float] = None,
        compact_release_boundary_m: Optional[float] = None,
        compact_selection_mode: str = "initial_compact",
        compact_assembly_duration_s: float = 10.0,
        compact_min_assembly_steps: int = 0,
        compact_requested_hold_duration_s: Optional[float] = None,
        compact_assembly_speed_tolerance_mps: float = 2.0,
        compact_assembly_a_max_dec_mps2: float = 1.0,
        compact_control_mode: str | CompactControlMode = CompactControlMode.FIXED_TARGET,
        compact_servo_a_max_dec_mps2: float = 1.0,
        # v5.6.2 opt-in: adaptive-servo historically used the same cap in
        # every lifecycle phase.  Keep that exact default for all stored
        # profiles; a phase-split profile must name its initial cap explicitly.
        compact_initial_assembly_servo_a_max_dec_mps2: Optional[float] = None,
        compact_high_authority_assembly_steps: int = 0,
        # v5.6.2 opt-in: this is deliberately separate from the residual
        # controller's jerk projection.  A phase-split wall must name the
        # bound and its finite in-window command budget.
        compact_settling_jerk_limit_mps3: Optional[float] = None,
        compact_settling_max_steps: int = 0,
        compact_max_activation_speed_spread_mps: float = 4.0,
        compact_max_speed_reduction_fraction: float = 0.20,
        compact_shock_gap_lo_mps: float = 5.0,
        compact_shock_gap_hi_mps: float = 15.0,
        compact_position_kp_s_inv: float = 0.05,
        compact_position_kd: float = 0.5,
        compact_position_speed_offset_cap_mps: float = 1.0,
        compact_safety_min_front_ttc_s: float = 3.0,
        compact_safety_min_rear_ttc_s: float = 4.0,
        compact_influence_upstream_m: float = 100.0,
        compact_influence_max_followers: int = 3,
        compact_influence_min_ttc_s: float = 1.5,
        cooldown_steps: int = 30,
        hard_brake_abort_threshold_mps2: Optional[float] = -4.5,
        abort_on_collision: bool = True,
        abort_on_teleport: bool = True,
        activation_acceleration_floor_mps2: Optional[float] = -3.0,
        # Gap alone is not a hazard in dense car-following.  Use closing TTC
        # and current braking for the veto; keep gap thresholds optional for
        # protocol variants that explicitly need them.
        activation_min_front_gap_m: Optional[float] = None,
        activation_min_rear_gap_m: Optional[float] = None,
        activation_min_front_ttc_s: Optional[float] = 2.0,
        activation_min_rear_ttc_s: Optional[float] = 3.0,
        monitor_controlled_followers: bool = True,
        diagnostic_formation_preview: bool = False,
        treatment_integrity_version: str = "v5.2",
        compact_handoff_enabled: bool = False,
        max_handoffs_per_lane: int = 0,
        compactness_enforcement_samples: int = 0,
        taper_duration_s: float = 0.0,
        taper_function: str = "none",
        # v5.5.1 is opt-in so older frozen profiles retain their exact
        # behaviour.  Its thresholds are deliberately drawn from the existing
        # activation/projection/compactness contract rather than tuned to an
        # outcome: 20 m activation gap, 4 s rear-TTC projection, -3 m/s2
        # activation braking floor, three followers / 100 m influence zone,
        # and the existing two-sample compactness persistence rule.
        adaptive_release_enabled: bool = False,
        adaptive_release_soft_rear_gap_m: Optional[float] = None,
        adaptive_release_soft_rear_ttc_s: Optional[float] = None,
        adaptive_release_soft_follower_accel_mps2: Optional[float] = None,
        adaptive_release_soft_gap_loss_m: Optional[float] = None,
        adaptive_release_soft_closing_increase_mps: Optional[float] = None,
        adaptive_release_persistence_samples: int = 2,
        adaptive_release_missing_follower_behavior: str = "release",
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path))
        self._base = base_policy
        self._wall_factory = wall_factory
        self.wall_params = dict(DEFAULT_WALL_PARAMS)
        self.wall_params.update(dict(wall_params or {}))
        self._validate_wall_params()
        self.approach_edge_distance_offsets_m = {
            PRE_MERGE_EDGE: 0.0,
            **{
                str(edge): max(0.0, float(offset))
                for edge, offset in dict(approach_edge_distance_offsets_m or {}).items()
            },
        }
        self.min_wall_cavs = max(1, int(min_wall_cavs))
        self.min_wall_lanes = max(1, int(min_wall_lanes))
        self.require_all_occupied_lanes = bool(require_all_occupied_lanes)
        self.required_lane_ids = tuple(sorted(str(x) for x in (required_lane_ids or ())))
        geometry_value = (
            geometry_mode.value if isinstance(geometry_mode, WallGeometry) else str(geometry_mode)
        )
        try:
            self.geometry_mode = WallGeometry(geometry_value)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in WallGeometry)
            raise ValueError(
                f"unknown wall geometry {geometry_mode!r}; expected one of {choices}"
            ) from exc
        self.planned_treatment_duration_s = (
            None if planned_treatment_duration_s is None else float(planned_treatment_duration_s)
        )
        if self.geometry_mode == WallGeometry.COMPACT_MOVING and (
            self.planned_treatment_duration_s is None or self.planned_treatment_duration_s <= 0.0
        ):
            raise ValueError("compact_moving geometry requires planned_treatment_duration_s > 0")
        self.compact_release_boundary_m = float(
            self.wall_params["d_lo"]
            if compact_release_boundary_m is None
            else compact_release_boundary_m
        )
        if not (0.0 <= self.compact_release_boundary_m <= float(self.wall_params["d_lo"])):
            raise ValueError(
                "compact_release_boundary_m must satisfy 0 <= boundary <= recruitment d_lo"
            )
        if compact_selection_mode not in ("initial_compact", "trajectory_coherent"):
            raise ValueError(
                "compact_selection_mode must be initial_compact or trajectory_coherent"
            )
        self.compact_selection_mode = str(compact_selection_mode)
        self.compact_assembly_duration_s = max(0.0, float(compact_assembly_duration_s))
        self.compact_min_assembly_steps = max(0, int(compact_min_assembly_steps))
        self.compact_requested_hold_duration_s = (
            None
            if compact_requested_hold_duration_s is None
            else max(0.0, float(compact_requested_hold_duration_s))
        )
        self.compact_assembly_speed_tolerance_mps = max(
            0.0, float(compact_assembly_speed_tolerance_mps)
        )
        self.compact_assembly_a_max_dec_mps2 = max(
            float(self.wall_params["a_max_dec"]),
            float(compact_assembly_a_max_dec_mps2),
        )
        control_mode_value = (
            compact_control_mode.value
            if isinstance(compact_control_mode, CompactControlMode)
            else str(compact_control_mode)
        )
        try:
            self.compact_control_mode = CompactControlMode(control_mode_value)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in CompactControlMode)
            raise ValueError(
                f"unknown compact control mode {compact_control_mode!r}; expected one of {choices}"
            ) from exc
        self.compact_servo_a_max_dec_mps2 = max(
            float(self.wall_params["a_max_dec"]),
            float(compact_servo_a_max_dec_mps2),
        )
        self.compact_initial_assembly_servo_a_max_dec_mps2 = max(
            float(self.wall_params["a_max_dec"]),
            float(
                self.compact_servo_a_max_dec_mps2
                if compact_initial_assembly_servo_a_max_dec_mps2 is None
                else compact_initial_assembly_servo_a_max_dec_mps2
            ),
        )
        self.compact_high_authority_assembly_steps = max(
            0, int(compact_high_authority_assembly_steps)
        )
        self.compact_settling_jerk_limit_mps3 = (
            None
            if compact_settling_jerk_limit_mps3 is None
            else max(0.0, float(compact_settling_jerk_limit_mps3))
        )
        self.compact_settling_max_steps = max(0, int(compact_settling_max_steps))
        if self.compact_settling_max_steps > 0 and (
            self.compact_settling_jerk_limit_mps3 is None
            or self.compact_settling_jerk_limit_mps3 <= 0.0
        ):
            raise ValueError("positive compact settling steps require a positive jerk limit")
        self.compact_max_activation_speed_spread_mps = max(
            0.0, float(compact_max_activation_speed_spread_mps)
        )
        self.compact_max_speed_reduction_fraction = min(
            1.0, max(0.0, float(compact_max_speed_reduction_fraction))
        )
        self.compact_shock_gap_lo_mps = float(compact_shock_gap_lo_mps)
        self.compact_shock_gap_hi_mps = float(compact_shock_gap_hi_mps)
        if self.compact_shock_gap_hi_mps <= self.compact_shock_gap_lo_mps:
            raise ValueError("compact shock gap hi must exceed lo")
        self.compact_position_kp_s_inv = max(0.0, float(compact_position_kp_s_inv))
        self.compact_position_kd = max(0.0, float(compact_position_kd))
        self.compact_position_speed_offset_cap_mps = max(
            0.0, float(compact_position_speed_offset_cap_mps)
        )
        self.compact_safety_min_front_ttc_s = max(0.0, float(compact_safety_min_front_ttc_s))
        self.compact_safety_min_rear_ttc_s = max(0.0, float(compact_safety_min_rear_ttc_s))
        self.compact_influence_upstream_m = max(0.0, float(compact_influence_upstream_m))
        self.compact_influence_max_followers = max(0, int(compact_influence_max_followers))
        self.compact_influence_min_ttc_s = max(0.0, float(compact_influence_min_ttc_s))
        if (
            self.geometry_mode == WallGeometry.COMPACT_MOVING
            and self.compact_assembly_duration_s >= float(self.planned_treatment_duration_s or 0.0)
        ):
            raise ValueError("compact assembly duration must be shorter than the treatment")
        if self.compact_requested_hold_duration_s is not None and (
            self.compact_assembly_duration_s
            + self.compact_requested_hold_duration_s
            + max(0.0, float(taper_duration_s))
            > float(self.planned_treatment_duration_s or 0.0) + 1.0e-9
        ):
            raise ValueError("assembly + requested hold + taper exceeds treatment horizon")
        default_span = float(self.wall_params["d_hi"]) - float(self.wall_params["d_lo"])
        self.max_longitudinal_span_m = float(
            default_span if max_longitudinal_span_m is None else max_longitudinal_span_m
        )
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.hard_brake_abort_threshold_mps2 = (
            None
            if hard_brake_abort_threshold_mps2 is None
            else float(hard_brake_abort_threshold_mps2)
        )
        self.abort_on_collision = bool(abort_on_collision)
        self.abort_on_teleport = bool(abort_on_teleport)
        self.activation_acceleration_floor_mps2 = activation_acceleration_floor_mps2
        self.activation_min_front_gap_m = activation_min_front_gap_m
        self.activation_min_rear_gap_m = activation_min_rear_gap_m
        self.activation_min_front_ttc_s = activation_min_front_ttc_s
        self.activation_min_rear_ttc_s = activation_min_rear_ttc_s
        self.monitor_controlled_followers = bool(monitor_controlled_followers)
        self.diagnostic_formation_preview = bool(diagnostic_formation_preview)
        self.treatment_integrity_version = str(treatment_integrity_version)
        self.compact_handoff_enabled = bool(compact_handoff_enabled)
        self.max_handoffs_per_lane = max(0, int(max_handoffs_per_lane))
        self.compactness_enforcement_samples = max(
            0, int(compactness_enforcement_samples)
        )
        self.taper_duration_s = max(0.0, float(taper_duration_s))
        self.taper_function = str(taper_function)
        if self.taper_function not in ("none", "linear"):
            raise ValueError("taper_function must be none or linear")
        if self.taper_duration_s > 0.0 and self.taper_function != "linear":
            raise ValueError("a positive taper duration requires the frozen linear taper")
        if self.taper_duration_s >= float(self.planned_treatment_duration_s or float("inf")):
            raise ValueError("taper duration must be shorter than the treatment horizon")
        self.adaptive_release_enabled = bool(adaptive_release_enabled)
        self.adaptive_release_soft_rear_gap_m = (
            None
            if adaptive_release_soft_rear_gap_m is None
            else max(0.0, float(adaptive_release_soft_rear_gap_m))
        )
        self.adaptive_release_soft_rear_ttc_s = (
            None
            if adaptive_release_soft_rear_ttc_s is None
            else max(0.0, float(adaptive_release_soft_rear_ttc_s))
        )
        self.adaptive_release_soft_follower_accel_mps2 = (
            None
            if adaptive_release_soft_follower_accel_mps2 is None
            else float(adaptive_release_soft_follower_accel_mps2)
        )
        self.adaptive_release_soft_gap_loss_m = (
            None
            if adaptive_release_soft_gap_loss_m is None
            else max(0.0, float(adaptive_release_soft_gap_loss_m))
        )
        self.adaptive_release_soft_closing_increase_mps = (
            None
            if adaptive_release_soft_closing_increase_mps is None
            else max(0.0, float(adaptive_release_soft_closing_increase_mps))
        )
        self.adaptive_release_persistence_samples = max(
            1, int(adaptive_release_persistence_samples)
        )
        if adaptive_release_missing_follower_behavior not in {"release", "ignore"}:
            raise ValueError("adaptive_release_missing_follower_behavior must be release or ignore")
        self.adaptive_release_missing_follower_behavior = str(
            adaptive_release_missing_follower_behavior
        )
        if self.adaptive_release_enabled and (
            self.taper_duration_s <= 0.0 or self.taper_function != "linear"
        ):
            raise ValueError("adaptive release requires the existing positive linear taper")
        self._wall: Optional[WallHoldPolicy] = None
        self._env: Any = None
        self._command = DAMPING_ONLY
        self._previous_command = DAMPING_ONLY
        self._wall_active = False
        self._cooldown_remaining = 0
        self._last_controlled_ids: tuple[str, ...] = ()
        self._treatment_state = TreatmentState.DAMPING
        self._hold_elapsed_steps = 0
        self._frozen_lane_ids: tuple[str, ...] = ()
        self._member_slots: Dict[str, list[Optional[str]]] = {}
        self._reserve_ids_by_lane: Dict[str, list[str]] = {}
        self._handoff_counts_by_lane: Dict[str, int] = {}
        self._pulse_member_ids_seen: set[str] = set()
        self._assembly_elapsed_steps = 0
        self._assembly_target_speed_mps: Optional[float] = None
        self._settling_elapsed_steps = 0
        self._settling_prior_accelerations_mps2: Dict[str, float] = {}
        self._settling_target_accelerations_mps2: Dict[str, float] = {}
        self._settling_max_jerk_mps3 = 0.0
        self._settling_tau_frozen = False
        self._adaptive_common_target_mps: Optional[float] = None
        self._adaptive_reference_speed_mps: Optional[float] = None
        self._adaptive_downstream_speed_mps: Optional[float] = None
        self._adaptive_speed_reduction_fraction = 0.0
        self._last_member_speed_targets_mps: Dict[str, float] = {}
        self._last_member_raw_targets_mps: Dict[str, float] = {}
        self._last_safety_projection: Dict[str, Dict[str, Any]] = {}
        self._current_wall_speed_target_mps = self.wall_params.get("wall_speed")
        self._last_wall_a_max_dec_mps2: Optional[float] = None
        self._active_treatment_steps = 0
        self._compactness_excess_samples = 0
        self._taper_elapsed_steps = 0
        self._taper_reason: Optional[str] = None
        self._taper_origin_targets_mps: Dict[str, float] = {}
        self._last_taper_blend_fraction: Optional[float] = None
        self._last_taper_base_targets_mps: Dict[str, float] = {}
        self._last_tau_taper: tuple[Dict[str, Any], ...] = ()
        self._adaptive_release_latched = False
        self._adaptive_release_soft_samples = 0
        self._adaptive_release_first_trigger: Optional[Dict[str, Any]] = None
        self._adaptive_release_follower_ids: Dict[str, Optional[str]] = {}
        self._adaptive_release_follower_baselines: Dict[str, Dict[str, Optional[float]]] = {}
        self._adaptive_release_previous_gaps_m: Dict[str, Optional[float]] = {}
        self._adaptive_release_last_compression: Dict[str, Any] = {}
        self.last_audit: Dict[str, Any] = {}

    @property
    def base_policy(self) -> Policy:
        if self._base is None:
            self._base = RLlibPolicy(self.checkpoint_path, policy_id="cav_policy")
        return self._base

    @property
    def command(self) -> EscalationCommand:
        return self._command

    @property
    def wall_active(self) -> bool:
        return self._wall_active

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    @property
    def treatment_state(self) -> TreatmentState:
        return self._treatment_state

    def _validate_wall_params(self) -> None:
        for name in _FORBIDDEN_AUTONOMOUS_WALL_OPTIONS:
            value = self.wall_params.get(name)
            if value not in (None, False, 0, 0.0):
                raise ValueError(
                    f"counterfactual wall cannot use autonomous option {name}={value!r}"
                )
        d_lo = float(self.wall_params["d_lo"])
        d_hi = float(self.wall_params["d_hi"])
        if not (0.0 <= d_lo < d_hi):
            raise ValueError("wall band must satisfy 0 <= d_lo < d_hi")

    def reset(self, env: Any, seed: int = None) -> None:
        self._env = env
        self._command = DAMPING_ONLY
        self._previous_command = DAMPING_ONLY
        self._wall_active = False
        self._cooldown_remaining = 0
        self._last_controlled_ids = ()
        self._treatment_state = TreatmentState.DAMPING
        self._frozen_lane_ids = ()
        self._member_slots = {}
        self._reserve_ids_by_lane = {}
        self._handoff_counts_by_lane = {}
        self._pulse_member_ids_seen = set()
        self._assembly_elapsed_steps = 0
        self._hold_elapsed_steps = 0
        self._assembly_target_speed_mps = None
        self._settling_elapsed_steps = 0
        self._settling_prior_accelerations_mps2 = {}
        self._settling_target_accelerations_mps2 = {}
        self._settling_max_jerk_mps3 = 0.0
        self._settling_tau_frozen = False
        self._adaptive_common_target_mps = None
        self._adaptive_reference_speed_mps = None
        self._adaptive_downstream_speed_mps = None
        self._adaptive_speed_reduction_fraction = 0.0
        self._last_member_speed_targets_mps = {}
        self._last_member_raw_targets_mps = {}
        self._last_safety_projection = {}
        self._current_wall_speed_target_mps = self.wall_params.get("wall_speed")
        self._last_wall_a_max_dec_mps2 = None
        self._active_treatment_steps = 0
        self._compactness_excess_samples = 0
        self._taper_elapsed_steps = 0
        self._taper_reason = None
        self._taper_origin_targets_mps = {}
        self._last_taper_blend_fraction = None
        self._last_taper_base_targets_mps = {}
        self._last_tau_taper = ()
        self._adaptive_release_latched = False
        self._adaptive_release_soft_samples = 0
        self._adaptive_release_first_trigger = None
        self._adaptive_release_follower_ids = {}
        self._adaptive_release_follower_baselines = {}
        self._adaptive_release_previous_gaps_m = {}
        self._adaptive_release_last_compression = {}
        self.last_audit = {}
        self.base_policy.reset(env, seed=seed)
        actuator_params = dict(self.wall_params)
        if self.geometry_mode == WallGeometry.COMPACT_MOVING:
            # Recruitment remains confined to wall_params[d_lo, d_hi]. Once
            # selected, compact members are followed as a moving actuator.
            # The outer overlay remains authoritative and never passes a
            # member after it leaves PRE_MERGE_EDGE.
            actuator_params.update({"d_lo": 0.0, "d_hi": 1.0e9})
        self._wall = self._wall_factory(**actuator_params)
        self._wall.reset(env, seed=seed)

    def set_command(self, command: EscalationCommand | int) -> None:
        try:
            self._command = EscalationCommand(int(command))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown escalation command: {command!r}") from exc

    @staticmethod
    def _live_ids() -> set[str]:
        try:
            return {str(vehicle_id) for vehicle_id in traci.vehicle.getIDList()}
        except Exception:
            return set()

    @staticmethod
    def _road_id(vehicle_id: str) -> Optional[str]:
        try:
            return str(traci.vehicle.getRoadID(vehicle_id))
        except Exception:
            return None

    def _is_approach_edge(self, vehicle_id: str) -> bool:
        return self._road_id(vehicle_id) in self.approach_edge_distance_offsets_m

    def _approach_distance_to_merge(self, vehicle_id: str) -> float:
        """Route-aware distance for the opt-in extended approach geometry."""
        edge = self._road_id(vehicle_id)
        if edge is None or edge not in self.approach_edge_distance_offsets_m:
            raise RuntimeError("vehicle is outside the configured approach edges")
        if edge == PRE_MERGE_EDGE:
            return float(distance_to_merge(vehicle_id))
        lane_id = self._lane_id(vehicle_id)
        if lane_id is None:
            raise RuntimeError("lane unavailable")
        lane_pos = self._vehicle_float("getLanePosition", vehicle_id)
        if lane_pos is None:
            raise RuntimeError("lane position unavailable")
        try:
            lane_length = float(traci.lane.getLength(lane_id))
        except Exception as exc:
            raise RuntimeError("lane length unavailable") from exc
        return max(0.0, float(self.approach_edge_distance_offsets_m[edge]) + lane_length - lane_pos)

    @staticmethod
    def _lane_id(vehicle_id: str) -> Optional[str]:
        try:
            return str(traci.vehicle.getLaneID(vehicle_id))
        except Exception:
            return None

    def _distance_in_band(self, vehicle_id: str) -> Optional[float]:
        if not self._is_approach_edge(vehicle_id):
            return None
        try:
            distance = self._approach_distance_to_merge(vehicle_id)
        except Exception:
            return None
        if float(self.wall_params["d_lo"]) <= distance <= float(self.wall_params["d_hi"]):
            return distance
        return None

    def _projected_travel_from_speed(self, speed_mps: float, duration_s: float) -> float:
        """Travel under the same rate-limited speed command as the wall."""
        duration = max(0.0, float(duration_s))
        speed = max(0.0, float(speed_mps))
        target_value = self.wall_params.get("wall_speed")
        if target_value is None:
            return speed * duration
        target = max(0.0, float(target_value))
        if speed > target:
            acceleration = max(1.0e-9, float(self.wall_params["a_max_dec"]))
            ramp_time = min(duration, (speed - target) / acceleration)
            end_speed = max(target, speed - acceleration * ramp_time)
        else:
            acceleration = 1.0
            ramp_time = min(duration, (target - speed) / acceleration)
            end_speed = min(target, speed + acceleration * ramp_time)
        return 0.5 * (speed + end_speed) * ramp_time + target * (duration - ramp_time)

    def _projected_wall_travel_m(self, vehicle_id: str) -> Optional[float]:
        """Conservative travel under the wall command over the full pulse."""
        duration = self.planned_treatment_duration_s
        if duration is None:
            return None
        speed = self._vehicle_float("getSpeed", vehicle_id)
        if speed is None:
            return None
        return self._projected_travel_from_speed(speed, duration)

    def _has_full_pulse_headroom(self, vehicle_id: str, distance: float) -> bool:
        projected = self._projected_wall_travel_m(vehicle_id)
        if projected is None:
            return False
        return distance - projected >= self.compact_release_boundary_m

    @staticmethod
    def _rate_limited_speed(
        speed_mps: float,
        target_mps: float,
        dt_s: float,
        *,
        a_max_dec_mps2: float,
        a_max_acc_mps2: float = 1.0,
    ) -> float:
        speed = max(0.0, float(speed_mps))
        target = max(0.0, float(target_mps))
        if speed > target:
            return max(target, speed - float(a_max_dec_mps2) * float(dt_s))
        return min(target, speed + float(a_max_acc_mps2) * float(dt_s))

    def _project_compact_formation(
        self,
        candidate_ids: tuple[str, ...],
        cav_distances: Mapping[str, float],
        candidate_speeds: Mapping[str, float],
    ) -> tuple[float, float, float, float]:
        """Project the declared assemble-then-hold member trajectories."""
        distances = {cav_id: float(cav_distances[cav_id]) for cav_id in candidate_ids}
        speeds = {cav_id: max(0.0, float(candidate_speeds[cav_id])) for cav_id in candidate_ids}
        initial_span = max(distances.values()) - min(distances.values())
        maximum_span = initial_span
        duration = float(self.planned_treatment_duration_s or 0.0)
        assembly_duration = min(self.compact_assembly_duration_s, duration)
        if self.compact_control_mode == CompactControlMode.ADAPTIVE_SERVO:
            downstream_speed = self._mean_speed_in_distance_band(0.0, 200.0)
            (
                adaptive_target,
                _reference_speed,
                _speed_gap,
                _reduction_fraction,
            ) = self._adaptive_target_from_values(
                tuple(speeds.values()),
                downstream_speed,
            )
            assembly_target = max(adaptive_target, min(speeds.values()))
            controller_step_s = self._simulation_step_s()
            assembly_steps = 0
            assembling = True
            elapsed = 0.0
            while elapsed < duration - 1.0e-9:
                dt_s = min(controller_step_s, duration - elapsed)
                speed_spread = max(speeds.values()) - min(speeds.values())
                if (
                    assembling
                    and assembly_steps > 0
                    and speed_spread <= self.compact_assembly_speed_tolerance_mps
                ):
                    assembling = False
                common_target = assembly_target if assembling else adaptive_target
                center = sum(distances.values()) / len(distances)
                mean_speed = sum(speeds.values()) / len(speeds)
                targets = {}
                for cav_id in candidate_ids:
                    position_error = distances[cav_id] - center
                    relative_speed = speeds[cav_id] - mean_speed
                    offset = min(
                        self.compact_position_speed_offset_cap_mps,
                        max(
                            -self.compact_position_speed_offset_cap_mps,
                            self.compact_position_kp_s_inv * position_error
                            - self.compact_position_kd * relative_speed,
                        ),
                    )
                    targets[cav_id] = max(0.0, common_target + offset)
                for cav_id in candidate_ids:
                    old_speed = speeds[cav_id]
                    # v5.6.2 has one explicitly bounded formation command.
                    # Historical profiles leave high_authority_steps at zero,
                    # preserving their original all-phase servo behaviour.
                    authority_cap = (
                        self.compact_initial_assembly_servo_a_max_dec_mps2
                        if (
                            assembling
                            and self.compact_high_authority_assembly_steps > 0
                            and assembly_steps < self.compact_high_authority_assembly_steps
                        )
                        else self.compact_servo_a_max_dec_mps2
                    )
                    new_speed = self._rate_limited_speed(
                        old_speed,
                        targets[cav_id],
                        dt_s,
                        a_max_dec_mps2=authority_cap,
                        # Formation feasibility must not assume that a slower
                        # cross-lane member can accelerate through surrounding
                        # traffic. Runtime may accelerate when admissible, but
                        # the predecision gate is deliberately conservative.
                        a_max_acc_mps2=0.0,
                    )
                    distances[cav_id] -= 0.5 * (old_speed + new_speed) * dt_s
                    speeds[cav_id] = new_speed
                if assembling:
                    assembly_steps += 1
                elapsed += dt_s
                maximum_span = max(
                    maximum_span,
                    max(distances.values()) - min(distances.values()),
                )
            terminal_span = max(distances.values()) - min(distances.values())
            return (
                initial_span,
                maximum_span,
                terminal_span,
                min(distances.values()),
            )
        wall_target_value = self.wall_params.get("wall_speed")
        wall_target = (
            min(speeds.values())
            if wall_target_value is None
            else max(0.0, float(wall_target_value))
        )
        assembly_target = max(wall_target, min(speeds.values()))
        controller_step_s = self._simulation_step_s()
        elapsed = 0.0
        while elapsed < duration - 1.0e-9:
            dt_s = min(controller_step_s, duration - elapsed)
            assembling = elapsed < assembly_duration - 1.0e-9
            target = assembly_target if assembling else wall_target
            deceleration = (
                self.compact_assembly_a_max_dec_mps2
                if assembling
                else float(self.wall_params["a_max_dec"])
            )
            for cav_id in candidate_ids:
                old_speed = speeds[cav_id]
                new_speed = self._rate_limited_speed(
                    old_speed,
                    target,
                    dt_s,
                    a_max_dec_mps2=deceleration,
                )
                distances[cav_id] -= 0.5 * (old_speed + new_speed) * dt_s
                speeds[cav_id] = new_speed
            elapsed += dt_s
            maximum_span = max(
                maximum_span,
                max(distances.values()) - min(distances.values()),
            )
        terminal_span = max(distances.values()) - min(distances.values())
        return initial_span, maximum_span, terminal_span, min(distances.values())

    def _one_step_assembly_reaches_compactness(
        self,
        candidate_ids: tuple[str, ...],
        cav_distances: Mapping[str, float],
        candidate_speeds: Mapping[str, float],
    ) -> bool:
        """Apply the declared single assembly command in the feasibility gate.

        This prevents a candidate from being admitted by a long-horizon
        projection and then failing at runtime because v5.6.2 permits only one
        high-authority assembly command.  Like the main trajectory projection,
        it deliberately assumes no acceleration for a slower member.
        """
        if not (
            self.compact_control_mode == CompactControlMode.ADAPTIVE_SERVO
            and self.compact_high_authority_assembly_steps > 0
        ):
            return True
        distances = {cav_id: float(cav_distances[cav_id]) for cav_id in candidate_ids}
        speeds = {cav_id: max(0.0, float(candidate_speeds[cav_id])) for cav_id in candidate_ids}
        downstream_speed = self._mean_speed_in_distance_band(0.0, 200.0)
        adaptive_target, _reference_speed, _gap, _fraction = self._adaptive_target_from_values(
            tuple(speeds.values()), downstream_speed
        )
        assembly_target = max(adaptive_target, min(speeds.values()))
        center = sum(distances.values()) / len(distances)
        mean_speed = sum(speeds.values()) / len(speeds)
        next_speeds: dict[str, float] = {}
        for cav_id in candidate_ids:
            position_error = distances[cav_id] - center
            relative_speed = speeds[cav_id] - mean_speed
            offset = min(
                self.compact_position_speed_offset_cap_mps,
                max(
                    -self.compact_position_speed_offset_cap_mps,
                    self.compact_position_kp_s_inv * position_error
                    - self.compact_position_kd * relative_speed,
                ),
            )
            next_speeds[cav_id] = self._rate_limited_speed(
                speeds[cav_id],
                max(0.0, assembly_target + offset),
                self._simulation_step_s(),
                a_max_dec_mps2=self.compact_initial_assembly_servo_a_max_dec_mps2,
                a_max_acc_mps2=0.0,
            )
        return (
            max(next_speeds.values()) - min(next_speeds.values())
            <= self.compact_assembly_speed_tolerance_mps
        )

    def _compact_selection(
        self,
        cav_distances: Mapping[str, float],
        cav_lane_by_id: Mapping[str, str],
        required_lanes: set[str],
    ) -> tuple[tuple[str, ...], str, Optional[float], Optional[float]]:
        selection_lanes = tuple(sorted(required_lanes or set(cav_lane_by_id.values())))
        candidates_by_lane: list[list[str]] = []
        candidate_speeds: dict[str, float] = {}
        candidate_profiles: dict[str, tuple[float, ...]] = {}
        duration = float(self.planned_treatment_duration_s or 0.0)
        profile_times = [min(duration, 0.5 * index) for index in range(int(duration / 0.5) + 1)]
        if not profile_times or profile_times[-1] < duration:
            profile_times.append(duration)
        for lane_id in selection_lanes:
            candidates = []
            for cav_id in sorted(cav_distances):
                if cav_lane_by_id.get(cav_id) != lane_id:
                    continue
                speed = self._vehicle_float("getSpeed", cav_id)
                if speed is None:
                    continue
                if (
                    self.compact_selection_mode == "initial_compact"
                    and not self._has_full_pulse_headroom(cav_id, cav_distances[cav_id])
                ):
                    continue
                candidates.append(cav_id)
                candidate_speeds[cav_id] = speed
                candidate_profiles[cav_id] = tuple(
                    cav_distances[cav_id] - self._projected_travel_from_speed(speed, time_s)
                    for time_s in profile_times
                )
            if not candidates:
                return (), "insufficient_kinematic_headroom", None, None
            candidates_by_lane.append(candidates)
        if not candidates_by_lane:
            return (), "insufficient_cav_lane_coverage", None, None

        def spans(
            candidate_ids: tuple[str, ...],
        ) -> tuple[float, float, float, float]:
            if self.compact_selection_mode == "trajectory_coherent":
                return self._project_compact_formation(
                    candidate_ids,
                    cav_distances,
                    candidate_speeds,
                )
            initial = [cav_distances[cav_id] for cav_id in candidate_ids]
            time_spans = [
                max(candidate_profiles[cav_id][index] for cav_id in candidate_ids)
                - min(candidate_profiles[cav_id][index] for cav_id in candidate_ids)
                for index in range(len(profile_times))
            ]
            terminal_distances = [candidate_profiles[cav_id][-1] for cav_id in candidate_ids]
            return (
                max(initial) - min(initial),
                max(time_spans),
                time_spans[-1],
                min(terminal_distances),
            )

        def score(candidate_ids: tuple[str, ...]) -> tuple[Any, ...]:
            distances = [cav_distances[cav_id] for cav_id in candidate_ids]
            (
                initial_span,
                predicted_max_span,
                terminal_span,
                minimum_terminal_distance,
            ) = spans(candidate_ids)
            speed_span = max(candidate_speeds[x] for x in candidate_ids) - min(
                candidate_speeds[x] for x in candidate_ids
            )
            if self.compact_selection_mode == "trajectory_coherent":
                adaptive_speed_veto = (
                    self.compact_control_mode == CompactControlMode.ADAPTIVE_SERVO
                    and speed_span > self.compact_max_activation_speed_spread_mps
                )
                return (
                    minimum_terminal_distance < self.compact_release_boundary_m,
                    adaptive_speed_veto,
                    predicted_max_span,
                    terminal_span,
                    speed_span,
                    initial_span,
                    -min(distances),
                    candidate_ids,
                )
            return (
                initial_span,
                predicted_max_span,
                speed_span,
                -min(distances),
                candidate_ids,
            )

        selected = min(product(*candidates_by_lane), key=score)
        (
            selected_span,
            predicted_max_span,
            terminal_span,
            minimum_terminal_distance,
        ) = spans(selected)
        selected_speed_span = max(candidate_speeds[x] for x in selected) - min(
            candidate_speeds[x] for x in selected
        )
        if minimum_terminal_distance < self.compact_release_boundary_m:
            return (), "insufficient_kinematic_headroom", predicted_max_span, terminal_span
        if (
            self.compact_control_mode == CompactControlMode.ADAPTIVE_SERVO
            and selected_speed_span > self.compact_max_activation_speed_spread_mps
        ):
            return (
                (),
                "activation_speed_spread_exceeded",
                predicted_max_span,
                terminal_span,
            )
        if not self._one_step_assembly_reaches_compactness(
            selected, cav_distances, candidate_speeds
        ):
            return (
                (),
                "one_step_assembly_not_compact",
                predicted_max_span,
                terminal_span,
            )
        if selected_span > self.max_longitudinal_span_m:
            return (), "formation_span_exceeded", predicted_max_span, terminal_span
        if (
            self.compact_selection_mode == "trajectory_coherent"
            and predicted_max_span > self.max_longitudinal_span_m
        ):
            return (
                (),
                "predicted_trajectory_span_exceeded",
                predicted_max_span,
                terminal_span,
            )
        return (
            tuple(sorted(selected)),
            "feasible",
            predicted_max_span,
            terminal_span,
        )

    def _formation_status(
        self, active_cav_ids: Iterable[str], live_ids: set[str]
    ) -> FormationStatus:
        occupied_lanes: set[str] = set()
        for vehicle_id in sorted(live_ids):
            if self._distance_in_band(vehicle_id) is None:
                continue
            lane_id = self._lane_id(vehicle_id)
            if lane_id is not None:
                occupied_lanes.add(lane_id)

        cav_distances: Dict[str, float] = {}
        cav_lanes: set[str] = set()
        cav_lane_by_id: Dict[str, str] = {}
        for cav_id in sorted(str(x) for x in active_cav_ids):
            if cav_id not in live_ids:
                continue
            distance = self._distance_in_band(cav_id)
            if distance is None:
                continue
            lane_id = self._lane_id(cav_id)
            if lane_id is None:
                continue
            cav_distances[cav_id] = distance
            cav_lanes.add(lane_id)
            cav_lane_by_id[cav_id] = lane_id

        if self.required_lane_ids:
            required_lanes = set(self.required_lane_ids)
        elif self.require_all_occupied_lanes:
            required_lanes = set(occupied_lanes)
        else:
            required_lanes = set()
        selected_ids = tuple(sorted(cav_distances))
        selection_reason = "feasible"
        predicted_max_span = None
        predicted_terminal_span = None
        if self.geometry_mode == WallGeometry.COMPACT_MOVING:
            (
                selected_ids,
                selection_reason,
                predicted_max_span,
                predicted_terminal_span,
            ) = self._compact_selection(cav_distances, cav_lane_by_id, required_lanes)
        selected_distances = [cav_distances[x] for x in selected_ids]
        span = max(selected_distances) - min(selected_distances) if selected_distances else 0.0

        reason = "feasible"
        feasible = True
        if len(cav_distances) < self.min_wall_cavs:
            feasible, reason = False, "insufficient_inband_cavs"
        elif len(cav_lanes) < self.min_wall_lanes:
            feasible, reason = False, "insufficient_cav_lane_coverage"
        elif required_lanes and not required_lanes.issubset(cav_lanes):
            feasible, reason = False, "uncovered_occupied_lane"
        elif selection_reason != "feasible":
            feasible, reason = False, selection_reason
        elif len(selected_ids) < self.min_wall_cavs:
            feasible, reason = False, "insufficient_selected_cavs"
        elif span > self.max_longitudinal_span_m:
            feasible, reason = False, "formation_span_exceeded"

        return FormationStatus(
            feasible=feasible,
            reason=reason,
            controlled_cav_ids=selected_ids,
            cav_lane_ids=tuple(sorted(cav_lanes)),
            occupied_lane_ids=tuple(sorted(occupied_lanes)),
            required_lane_ids=tuple(sorted(required_lanes)),
            longitudinal_span_m=float(span),
            predicted_max_span_m=predicted_max_span,
            predicted_terminal_span_m=predicted_terminal_span,
        )

    @staticmethod
    def _event_ids(getter_name: str) -> tuple[str, ...]:
        try:
            getter = getattr(traci.simulation, getter_name)
            return tuple(sorted(str(x) for x in getter()))
        except (AttributeError, RuntimeError):
            return ()

    @staticmethod
    def _vehicle_float(getter_name: str, vehicle_id: str) -> Optional[float]:
        try:
            return float(getattr(traci.vehicle, getter_name)(vehicle_id))
        except Exception:
            return None

    @staticmethod
    def _neighbor(getter_name: str, vehicle_id: str) -> tuple[Optional[str], Optional[float]]:
        try:
            neighbor = getattr(traci.vehicle, getter_name)(vehicle_id, 200.0)
            if not neighbor or not neighbor[0]:
                return None, None
            gap = float(neighbor[1])
            return str(neighbor[0]), gap if gap >= 0.0 else None
        except Exception:
            return None, None

    @staticmethod
    def _closing_ttc(gap_m: Optional[float], closing_speed_mps: float) -> Optional[float]:
        if gap_m is None or closing_speed_mps <= 1e-6:
            return None
        return gap_m / closing_speed_mps

    def _activation_safety_status(
        self, controlled_ids: Sequence[str], live_ids: set[str]
    ) -> ActivationSafetyStatus:
        violations: list[str] = []
        accelerations: list[float] = []
        front_gaps: list[float] = []
        rear_gaps: list[float] = []
        front_ttcs: list[float] = []
        rear_ttcs: list[float] = []

        for cav_id in controlled_ids:
            if cav_id not in live_ids:
                continue
            acceleration = self._vehicle_float("getAcceleration", cav_id)
            speed = self._vehicle_float("getSpeed", cav_id)
            if acceleration is not None:
                accelerations.append(acceleration)
                floor = self.activation_acceleration_floor_mps2
                if floor is not None and acceleration <= float(floor):
                    violations.append(f"{cav_id}:already_braking")

            leader_id, front_gap = self._neighbor("getLeader", cav_id)
            if front_gap is not None:
                front_gaps.append(front_gap)
                threshold = self.activation_min_front_gap_m
                if threshold is not None and front_gap < float(threshold):
                    violations.append(f"{cav_id}:front_gap")
            if leader_id is not None and speed is not None:
                leader_speed = self._vehicle_float("getSpeed", leader_id)
                if leader_speed is not None:
                    front_ttc = self._closing_ttc(front_gap, speed - leader_speed)
                    if front_ttc is not None:
                        front_ttcs.append(front_ttc)
                        threshold = self.activation_min_front_ttc_s
                        if threshold is not None and front_ttc < float(threshold):
                            violations.append(f"{cav_id}:front_ttc")

            follower_id, rear_gap = self._neighbor("getFollower", cav_id)
            if rear_gap is not None:
                rear_gaps.append(rear_gap)
                threshold = self.activation_min_rear_gap_m
                if threshold is not None and rear_gap < float(threshold):
                    violations.append(f"{cav_id}:rear_gap")
            if follower_id is not None and speed is not None:
                follower_speed = self._vehicle_float("getSpeed", follower_id)
                if follower_speed is not None:
                    rear_ttc = self._closing_ttc(rear_gap, follower_speed - speed)
                    if rear_ttc is not None:
                        rear_ttcs.append(rear_ttc)
                        threshold = self.activation_min_rear_ttc_s
                        if threshold is not None and rear_ttc < float(threshold):
                            violations.append(f"{cav_id}:rear_ttc")

        unique_violations = tuple(sorted(set(violations)))
        return ActivationSafetyStatus(
            safe=not unique_violations,
            reason="safe" if not unique_violations else unique_violations[0].split(":")[-1],
            checked_cav_ids=tuple(sorted(str(x) for x in controlled_ids)),
            violations=unique_violations,
            min_acceleration_mps2=min(accelerations) if accelerations else None,
            min_front_gap_m=min(front_gaps) if front_gaps else None,
            min_rear_gap_m=min(rear_gaps) if rear_gaps else None,
            min_front_ttc_s=min(front_ttcs) if front_ttcs else None,
            min_rear_ttc_s=min(rear_ttcs) if rear_ttcs else None,
        )

    def _safety_abort_reason(self, live_ids: set[str]) -> Optional[str]:
        if self.abort_on_collision and self._event_ids("getCollidingVehiclesIDList"):
            return "collision"
        if self.abort_on_teleport and self._event_ids("getStartingTeleportIDList"):
            return "teleport"
        threshold = self.hard_brake_abort_threshold_mps2
        if threshold is not None:
            for cav_id in self._last_controlled_ids:
                if cav_id not in live_ids:
                    continue
                try:
                    if float(traci.vehicle.getAcceleration(cav_id)) <= threshold:
                        return "controlled_cav_hard_brake"
                except Exception:
                    continue
                if self.monitor_controlled_followers:
                    follower_id, _gap = self._neighbor("getFollower", cav_id)
                    if follower_id is not None and follower_id in live_ids:
                        follower_acceleration = self._vehicle_float("getAcceleration", follower_id)
                        if follower_acceleration is not None and follower_acceleration <= threshold:
                            return "controlled_follower_hard_brake"
            if self.compact_control_mode == CompactControlMode.ADAPTIVE_SERVO:
                for member_id in self._last_controlled_ids:
                    if member_id not in live_ids:
                        continue
                    leader_id = member_id
                    leader_speed = self._vehicle_float("getSpeed", leader_id)
                    accumulated_distance = 0.0
                    for follower_index in range(self.compact_influence_max_followers):
                        follower_id, gap = self._neighbor("getFollower", leader_id)
                        if follower_id is None or follower_id not in live_ids:
                            break
                        accumulated_distance += float(gap or 0.0) + 5.0
                        if accumulated_distance > self.compact_influence_upstream_m:
                            break
                        follower_acceleration = self._vehicle_float("getAcceleration", follower_id)
                        if follower_acceleration is not None and follower_acceleration <= threshold:
                            return f"influence_zone_hard_brake:{follower_index + 1}"
                        follower_speed = self._vehicle_float("getSpeed", follower_id)
                        if (
                            leader_speed is not None
                            and follower_speed is not None
                            and gap is not None
                            and self.compact_influence_min_ttc_s > 0.0
                        ):
                            ttc = self._closing_ttc(gap, follower_speed - leader_speed)
                            if ttc is not None and ttc < self.compact_influence_min_ttc_s:
                                return f"influence_zone_low_ttc:{follower_index + 1}"
                        leader_id = follower_id
                        leader_speed = follower_speed
        return None

    def _adaptive_follower_compression(
        self,
        member_ids: Sequence[str],
        live_ids: set[str],
    ) -> Dict[str, Any]:
        """Read the existing follower-chain safety state without changing observations.

        The direct follower is identity-latched when HOLD begins.  A replacement
        is never treated as the same trajectory: v5.5.1 releases safely instead
        of silently rebasing.  A member with no initial follower has no rear
        compression constraint; losing a previously latched follower is an
        explicit missing-observation condition.
        """

        rows: list[Dict[str, Any]] = []
        reasons: list[str] = []
        min_gap: Optional[float] = None
        min_ttc: Optional[float] = None
        max_closing = 0.0
        braking_count = 0
        missing_or_changed: list[str] = []
        for member_id in sorted(str(value) for value in member_ids):
            if member_id not in live_ids:
                missing_or_changed.append(f"member_missing:{member_id}")
                continue
            leader_id = member_id
            leader_speed = self._vehicle_float("getSpeed", leader_id)
            accumulated_distance = 0.0
            for follower_index in range(self.compact_influence_max_followers):
                follower_id, gap_m = self._neighbor("getFollower", leader_id)
                if follower_index == 0:
                    expected = self._adaptive_release_follower_ids.get(member_id)
                    if member_id not in self._adaptive_release_follower_ids:
                        self._adaptive_release_follower_ids[member_id] = follower_id
                        expected = follower_id
                    elif expected is not None and follower_id != expected:
                        missing_or_changed.append(
                            f"follower_identity_changed:{member_id}:{expected}->{follower_id}"
                        )
                if follower_id is None or follower_id not in live_ids:
                    if follower_index == 0 and self._adaptive_release_follower_ids.get(member_id):
                        missing_or_changed.append(f"follower_missing:{member_id}")
                    break
                accumulated_distance += float(gap_m or 0.0) + 5.0
                if accumulated_distance > self.compact_influence_upstream_m:
                    break
                follower_speed = self._vehicle_float("getSpeed", follower_id)
                follower_acceleration = self._vehicle_float("getAcceleration", follower_id)
                closing_speed = (
                    max(0.0, float(follower_speed) - float(leader_speed))
                    if follower_speed is not None and leader_speed is not None
                    else 0.0
                )
                rear_ttc = self._closing_ttc(gap_m, closing_speed)
                previous_gap = self._adaptive_release_previous_gaps_m.get(follower_id)
                gap_closing = bool(
                    previous_gap is not None
                    and gap_m is not None
                    and float(gap_m) < float(previous_gap)
                )
                if gap_m is not None:
                    self._adaptive_release_previous_gaps_m[follower_id] = float(gap_m)
                    min_gap = float(gap_m) if min_gap is None else min(min_gap, float(gap_m))
                if rear_ttc is not None:
                    min_ttc = rear_ttc if min_ttc is None else min(min_ttc, rear_ttc)
                max_closing = max(max_closing, closing_speed)
                braking = bool(
                    follower_acceleration is not None
                    and self.adaptive_release_soft_follower_accel_mps2 is not None
                    and follower_acceleration <= self.adaptive_release_soft_follower_accel_mps2
                )
                braking_count += int(braking)
                baseline = self._adaptive_release_follower_baselines.get(member_id)
                if follower_index == 0 and baseline is None:
                    baseline = {
                        "rear_gap_m": gap_m,
                        "closing_speed_mps": closing_speed,
                        "follower_acceleration_mps2": follower_acceleration,
                    }
                    self._adaptive_release_follower_baselines[member_id] = baseline
                baseline_gap = None if baseline is None else baseline.get("rear_gap_m")
                baseline_closing = (
                    None if baseline is None else baseline.get("closing_speed_mps")
                )
                gap_loss = (
                    max(0.0, float(baseline_gap) - float(gap_m))
                    if baseline_gap is not None and gap_m is not None
                    else None
                )
                closing_increase = (
                    max(0.0, closing_speed - float(baseline_closing))
                    if baseline_closing is not None
                    else None
                )
                row = {
                    "member_id": member_id,
                    "follower_index": follower_index + 1,
                    "follower_id": follower_id,
                    "gap_m": gap_m,
                    "follower_speed_mps": follower_speed,
                    "leader_speed_mps": leader_speed,
                    "closing_speed_mps": closing_speed,
                    "rear_ttc_s": rear_ttc,
                    "follower_acceleration_mps2": follower_acceleration,
                    "gap_closing": gap_closing,
                    "baseline_rear_gap_m": baseline_gap,
                    "baseline_closing_speed_mps": baseline_closing,
                    "gap_loss_m": gap_loss,
                    "closing_increase_mps": closing_increase,
                    "distance_from_member_m": accumulated_distance,
                }
                rows.append(row)
                risk_gate = bool(
                    self.adaptive_release_soft_rear_gap_m is not None
                    and gap_m is not None
                    and float(gap_m) <= self.adaptive_release_soft_rear_gap_m
                ) or bool(
                    self.adaptive_release_soft_rear_ttc_s is not None
                    and rear_ttc is not None
                    and rear_ttc <= self.adaptive_release_soft_rear_ttc_s
                )
                if follower_index == 0:
                    projection_reasons = tuple(
                        (self._last_safety_projection.get(member_id) or {}).get(
                            "clip_reasons", ()
                        )
                    )
                    projected_rear_risk = any(
                        reason in {"rear_ttc_clip", "front_rear_constraint_conflict"}
                        for reason in projection_reasons
                    )
                    trend = bool(
                        self.adaptive_release_soft_gap_loss_m is not None
                        and gap_loss is not None
                        and gap_loss > self.adaptive_release_soft_gap_loss_m
                    ) or bool(
                        self.adaptive_release_soft_closing_increase_mps is not None
                        and closing_increase is not None
                        and closing_increase > self.adaptive_release_soft_closing_increase_mps
                    )
                    if braking:
                        reasons.append(f"follower_braking:{member_id}")
                    if projected_rear_risk:
                        reasons.append(f"rear_projection_clip:{member_id}")
                    if risk_gate and trend:
                        reasons.append(f"risk_gated_compression:{member_id}")
                leader_id = follower_id
                leader_speed = follower_speed
        if missing_or_changed and self.adaptive_release_missing_follower_behavior == "release":
            reasons.extend(missing_or_changed)
        return {
            "enabled": self.adaptive_release_enabled,
            "member_count": len(tuple(member_ids)),
            "follower_rows": tuple(rows),
            "soft_reasons": tuple(sorted(set(reasons))),
            "identity_or_missing_reasons": tuple(sorted(set(missing_or_changed))),
            "min_rear_gap_m": min_gap,
            "min_rear_ttc_s": min_ttc,
            "max_closing_speed_mps": max_closing,
            "braking_follower_count": braking_count,
        }

    def _adaptive_release_status(
        self, member_ids: Sequence[str], live_ids: set[str]
    ) -> Dict[str, Any]:
        """Update deterministic soft-trigger persistence for v5.5.1."""

        if not self.adaptive_release_enabled:
            return {"enabled": False, "should_taper": False, "freeze_tightening": False}
        compression = self._adaptive_follower_compression(member_ids, live_ids)
        reasons = tuple(compression["soft_reasons"])
        if reasons:
            self._adaptive_release_soft_samples += 1
            if self._adaptive_release_first_trigger is None:
                self._adaptive_release_first_trigger = {
                    "reasons": reasons,
                    "sample_index": self._adaptive_release_soft_samples,
                }
        else:
            self._adaptive_release_soft_samples = 0
        should_taper = bool(compression["identity_or_missing_reasons"]) or (
            bool(reasons)
            and self._adaptive_release_soft_samples >= self.adaptive_release_persistence_samples
        )
        result = {
            **compression,
            "soft_samples": self._adaptive_release_soft_samples,
            "persistence_samples_required": self.adaptive_release_persistence_samples,
            "freeze_tightening": bool(reasons),
            "should_taper": should_taper,
            "latched": self._adaptive_release_latched,
        }
        self._adaptive_release_last_compression = result
        return result

    def _latch_members(
        self,
        formation: FormationStatus,
        active_cav_ids: Iterable[str] = (),
        live_ids: Optional[set[str]] = None,
    ) -> list[Dict[str, Any]]:
        """Freeze members and the v5.3 ranked same-lane reserve order."""
        by_lane: Dict[str, list[str]] = {}
        for cav_id in formation.controlled_cav_ids:
            lane_id = self._lane_id(cav_id)
            if lane_id is not None:
                by_lane.setdefault(lane_id, []).append(cav_id)
        for members in by_lane.values():
            members.sort()

        frozen_lanes = (
            tuple(formation.required_lane_ids)
            if formation.required_lane_ids
            else tuple(sorted(by_lane))
        )
        # Preserve the current treatment intensity: every initially in-band
        # CAV is a member. Required lanes with no slot cannot occur after a
        # feasible activation, but keeping the empty lane makes dose loss
        # explicit if a protocol variant supplies required_lane_ids.
        self._frozen_lane_ids = tuple(sorted(set(frozen_lanes) | set(by_lane)))
        self._member_slots = {
            lane_id: list(by_lane.get(lane_id, ())) for lane_id in self._frozen_lane_ids
        }
        self._reserve_ids_by_lane = {lane_id: [] for lane_id in self._frozen_lane_ids}
        self._handoff_counts_by_lane = {lane_id: 0 for lane_id in self._frozen_lane_ids}
        if self.compact_handoff_enabled:
            live = self._live_ids() if live_ids is None else set(live_ids)
            primary_ids = {
                member_id for members in self._member_slots.values() for member_id in members
            }
            primary_distance_by_lane: Dict[str, float] = {}
            for lane_id, members in self._member_slots.items():
                distances = []
                for member_id in members:
                    try:
                        distances.append(self._approach_distance_to_merge(member_id))
                    except Exception:
                        continue
                if distances:
                    primary_distance_by_lane[lane_id] = sum(distances) / len(distances)
            ranked: Dict[str, list[tuple[float, str]]] = {
                lane_id: [] for lane_id in self._frozen_lane_ids
            }
            midpoint = 0.5 * (
                float(self.wall_params["d_lo"]) + float(self.wall_params["d_hi"])
            )
            for cav_id in sorted({str(value) for value in active_cav_ids} - primary_ids):
                if cav_id not in live:
                    continue
                lane_id = self._lane_id(cav_id)
                distance = self._distance_in_band(cav_id)
                if lane_id not in ranked or distance is None:
                    continue
                reference = primary_distance_by_lane.get(lane_id, midpoint)
                ranked[lane_id].append((abs(float(distance) - reference), cav_id))
            for lane_id, candidates in ranked.items():
                candidates.sort()
                self._reserve_ids_by_lane[lane_id] = [item[1] for item in candidates]
        self._pulse_member_ids_seen = {
            member_id
            for members in self._member_slots.values()
            for member_id in members
            if member_id is not None
        }
        events = [
            {
                "event": "latched",
                "lane_id": lane_id,
                "slot_index": slot_index,
                "member_id": member_id,
            }
            for lane_id in self._frozen_lane_ids
            for slot_index, member_id in enumerate(self._member_slots[lane_id])
        ]
        events.extend(
            {
                "event": "reserve_latched",
                "lane_id": lane_id,
                "reserve_rank": rank,
                "member_id": member_id,
            }
            for lane_id in self._frozen_lane_ids
            for rank, member_id in enumerate(self._reserve_ids_by_lane.get(lane_id, ()), 1)
        )
        return events

    def _member_exit_reason(
        self,
        member_id: str,
        lane_id: str,
        active_cav_ids: set[str],
        live_ids: set[str],
    ) -> Optional[str]:
        if member_id not in live_ids:
            return "vehicle_not_live"
        if member_id not in active_cav_ids:
            return "cav_not_active"
        if self.geometry_mode == WallGeometry.COMPACT_MOVING:
            if not self._is_approach_edge(member_id):
                return "left_approach_edge"
            try:
                distance = self._approach_distance_to_merge(member_id)
            except Exception:
                return "distance_unavailable"
            if distance < self.compact_release_boundary_m:
                return "crossed_release_boundary"
        elif self._distance_in_band(member_id) is None:
            return "left_wall_band"
        if self._lane_id(member_id) != lane_id:
            return "lane_changed"
        return None

    def _refresh_members(
        self,
        active_cav_ids: Iterable[str],
        live_ids: set[str],
    ) -> tuple[tuple[str, ...], list[Dict[str, Any]], tuple[str, ...]]:
        """Keep latched slots filled with deterministic, safe same-lane handoffs."""
        active_ids = {str(value) for value in active_cav_ids}
        assigned: set[str] = set()
        events: list[Dict[str, Any]] = []
        rejected_candidates: set[str] = set()
        vacated_by: Dict[tuple[str, int], Optional[str]] = {}

        for lane_id in self._frozen_lane_ids:
            slots = self._member_slots.setdefault(lane_id, [])
            for slot_index, member_id in enumerate(slots):
                if member_id is None:
                    vacated_by[(lane_id, slot_index)] = None
                    continue
                reason = self._member_exit_reason(member_id, lane_id, active_ids, live_ids)
                if reason is None and member_id not in assigned:
                    assigned.add(member_id)
                    continue
                slots[slot_index] = None
                vacated_by[(lane_id, slot_index)] = member_id
                events.append(
                    {
                        "event": "removed",
                        "lane_id": lane_id,
                        "slot_index": slot_index,
                        "member_id": member_id,
                        "reason": reason or "duplicate_assignment",
                    }
                )

        removed_member_ids = tuple(
            sorted(
                str(event["member_id"])
                for event in events
                if event.get("event") == "removed" and event.get("member_id") is not None
            )
        )
        if removed_member_ids:
            self._restore_wall_tau(removed_member_ids)

        if self.geometry_mode == WallGeometry.COMPACT_MOVING and not self.compact_handoff_enabled:
            member_ids = tuple(sorted(assigned))
            return member_ids, events, ()

        candidates: Dict[str, list[tuple[float, str]]] = {
            lane_id: [] for lane_id in self._frozen_lane_ids
        }
        if self.geometry_mode == WallGeometry.COMPACT_MOVING:
            for lane_id in self._frozen_lane_ids:
                for rank, cav_id in enumerate(self._reserve_ids_by_lane.get(lane_id, ())):
                    if cav_id in assigned or cav_id not in active_ids or cav_id not in live_ids:
                        continue
                    if self._lane_id(cav_id) != lane_id or not self._is_approach_edge(cav_id):
                        continue
                    try:
                        if self._approach_distance_to_merge(cav_id) < self.compact_release_boundary_m:
                            continue
                    except Exception:
                        continue
                    candidates[lane_id].append((float(rank), cav_id))
        else:
            midpoint = 0.5 * (float(self.wall_params["d_lo"]) + float(self.wall_params["d_hi"]))
            for cav_id in sorted(active_ids - assigned):
                if cav_id not in live_ids:
                    continue
                distance = self._distance_in_band(cav_id)
                lane_id = self._lane_id(cav_id)
                if distance is None or lane_id not in candidates:
                    continue
                candidates[lane_id].append((abs(distance - midpoint), cav_id))
        for lane_candidates in candidates.values():
            lane_candidates.sort()

        for lane_id in self._frozen_lane_ids:
            slots = self._member_slots[lane_id]
            for slot_index, member_id in enumerate(slots):
                if member_id is not None:
                    continue
                if self._treatment_state == TreatmentState.TAPERING:
                    continue
                if (
                    self.geometry_mode == WallGeometry.COMPACT_MOVING
                    and self._handoff_counts_by_lane.get(lane_id, 0)
                    >= self.max_handoffs_per_lane
                ):
                    continue
                replacement = None
                while candidates[lane_id]:
                    _offset, candidate_id = candidates[lane_id].pop(0)
                    safety = self._activation_safety_status((candidate_id,), live_ids)
                    if safety.safe:
                        replacement = candidate_id
                        break
                    rejected_candidates.add(candidate_id)
                if replacement is None:
                    continue
                slots[slot_index] = replacement
                assigned.add(replacement)
                self._pulse_member_ids_seen.add(replacement)
                if self.geometry_mode == WallGeometry.COMPACT_MOVING:
                    self._handoff_counts_by_lane[lane_id] = (
                        self._handoff_counts_by_lane.get(lane_id, 0) + 1
                    )
                events.append(
                    {
                        "event": "handoff",
                        "lane_id": lane_id,
                        "slot_index": slot_index,
                        "previous_member_id": vacated_by.get((lane_id, slot_index)),
                        "member_id": replacement,
                        "reason": "same_lane_replacement",
                        "handoff_count_for_lane": self._handoff_counts_by_lane.get(lane_id, 0),
                    }
                )

        member_ids = tuple(
            sorted(
                member_id
                for lane_id in self._frozen_lane_ids
                for member_id in self._member_slots[lane_id]
                if member_id is not None
            )
        )
        return member_ids, events, tuple(sorted(rejected_candidates))

    def _restore_wall_tau(self, member_ids: Iterable[str]) -> tuple[Dict[str, Any], ...]:
        """Restore wall-only headway state while preserving damping speed control."""

        if self._wall is None:
            return ()
        restore = getattr(self._wall, "restore_tau", None)
        if not callable(restore):
            return ()
        result = restore(tuple(sorted({str(member_id) for member_id in member_ids})))
        if not isinstance(result, (list, tuple)):
            return ()
        return tuple(dict(row) for row in result if isinstance(row, Mapping))

    def _coverage_audit(self, member_ids: Sequence[str]) -> Dict[str, Any]:
        member_set = set(member_ids)
        covered_lanes = tuple(
            sorted(
                {
                    lane_id
                    for lane_id in self._frozen_lane_ids
                    if any(
                        member_id is not None and member_id in member_set
                        for member_id in self._member_slots.get(lane_id, ())
                    )
                }
            )
        )
        required_count = len(self._frozen_lane_ids)
        return {
            "frozen_lane_ids": self._frozen_lane_ids,
            "covered_lane_ids": covered_lanes,
            "required_lane_count": required_count,
            "covered_lane_count": len(covered_lanes),
            "coverage_fraction": (len(covered_lanes) / required_count if required_count else 0.0),
            "controlled_cav_count": len(member_ids),
        }

    def _control_delta_audit(
        self,
        member_ids: Sequence[str],
        base_actions: Mapping[str, object],
    ) -> Dict[str, Any]:
        records: Dict[str, Any] = {}
        for member_id in member_ids:
            records[member_id] = {
                "base_action_suppressed": base_actions.get(member_id),
                "wall_speed_target_mps": self._current_wall_speed_target_mps,
                "member_raw_speed_target_mps": (self._last_member_raw_targets_mps.get(member_id)),
                "member_projected_speed_target_mps": (
                    self._last_member_speed_targets_mps.get(member_id)
                ),
                "member_applied_speed_command_mps": (
                    getattr(self._wall, "last_speed_commands_mps", {}).get(member_id)
                    if self._wall is not None
                    else None
                ),
                "wall_a_max_dec_mps2": self._last_wall_a_max_dec_mps2,
                "safety_projection": self._last_safety_projection.get(member_id),
                "wall_tau_offset_target_s": self.wall_params.get("wall_tau"),
                "treatment_phase": self._treatment_state.value,
                "pre_override_speed_mps": self._vehicle_float("getSpeed", member_id),
                "pre_override_acceleration_mps2": self._vehicle_float("getAcceleration", member_id),
                "pre_override_tau_s": self._vehicle_float("getTau", member_id),
            }
        return records

    def _simulation_step_s(self) -> float:
        return max(1.0e-6, float(getattr(self._env, "simulation_step", 1.0)))

    def _mean_speed_in_distance_band(
        self,
        d_lo_m: float,
        d_hi_m: float,
    ) -> Optional[float]:
        speeds: list[float] = []
        for vehicle_id in sorted(self._live_ids()):
            if not self._is_approach_edge(vehicle_id):
                continue
            try:
                distance = self._approach_distance_to_merge(vehicle_id)
                speed = float(traci.vehicle.getSpeed(vehicle_id))
            except Exception:
                continue
            if float(d_lo_m) <= distance < float(d_hi_m):
                speeds.append(speed)
        return sum(speeds) / len(speeds) if speeds else None

    def _freeze_adaptive_common_target(
        self,
        member_ids: Sequence[str],
    ) -> float:
        speeds = [
            speed
            for member_id in member_ids
            if (speed := self._vehicle_float("getSpeed", member_id)) is not None
        ]
        floor_value = self.wall_params.get("wall_speed")
        floor_speed = max(0.0, float(floor_value or 0.0))
        if not speeds:
            self._adaptive_reference_speed_mps = None
            self._adaptive_downstream_speed_mps = None
            self._adaptive_speed_reduction_fraction = 0.0
            return floor_speed
        downstream_speed = self._mean_speed_in_distance_band(0.0, 200.0)
        (
            target,
            reference_speed,
            _speed_gap,
            reduction_fraction,
        ) = self._adaptive_target_from_values(
            tuple(speeds),
            downstream_speed,
        )
        self._adaptive_reference_speed_mps = reference_speed
        self._adaptive_downstream_speed_mps = downstream_speed
        self._adaptive_speed_reduction_fraction = reduction_fraction
        return target

    def _adaptive_target_from_values(
        self,
        speeds: Sequence[float],
        downstream_speed_mps: Optional[float],
    ) -> tuple[float, Optional[float], float, float]:
        floor_value = self.wall_params.get("wall_speed")
        floor_speed = max(0.0, float(floor_value or 0.0))
        if not speeds:
            return floor_speed, None, 0.0, 0.0
        reference_speed = min(float(speed) for speed in speeds)
        speed_gap = (
            0.0
            if downstream_speed_mps is None
            else max(0.0, reference_speed - float(downstream_speed_mps))
        )
        urgency = min(
            1.0,
            max(
                0.0,
                (speed_gap - self.compact_shock_gap_lo_mps)
                / (self.compact_shock_gap_hi_mps - self.compact_shock_gap_lo_mps),
            ),
        )
        reduction_fraction = self.compact_max_speed_reduction_fraction * urgency
        target = max(
            floor_speed,
            reference_speed * (1.0 - reduction_fraction),
        )
        return target, reference_speed, speed_gap, reduction_fraction

    def _safety_project_target(
        self,
        member_id: str,
        raw_target_mps: float,
    ) -> tuple[float, Dict[str, Any]]:
        lower_bound = 0.0
        upper_bound = float("inf")
        reasons: list[str] = []
        speed = self._vehicle_float("getSpeed", member_id)

        follower_id, rear_gap = self._neighbor("getFollower", member_id)
        follower_speed = (
            self._vehicle_float("getSpeed", follower_id) if follower_id is not None else None
        )
        if (
            follower_speed is not None
            and rear_gap is not None
            and self.compact_safety_min_rear_ttc_s > 0.0
        ):
            rear_bound = follower_speed - (rear_gap / self.compact_safety_min_rear_ttc_s)
            lower_bound = max(lower_bound, rear_bound)

        leader_id, front_gap = self._neighbor("getLeader", member_id)
        leader_speed = self._vehicle_float("getSpeed", leader_id) if leader_id is not None else None
        if (
            leader_speed is not None
            and front_gap is not None
            and self.compact_safety_min_front_ttc_s > 0.0
        ):
            front_bound = leader_speed + (front_gap / self.compact_safety_min_front_ttc_s)
            upper_bound = min(upper_bound, front_bound)

        raw_target = max(0.0, float(raw_target_mps))
        if lower_bound > upper_bound:
            projected = max(0.0, float(speed or raw_target))
            reasons.append("front_rear_constraint_conflict")
        else:
            projected = min(max(raw_target, lower_bound), upper_bound)
            if projected > raw_target + 1.0e-9:
                reasons.append("rear_ttc_clip")
            if projected < raw_target - 1.0e-9:
                reasons.append("front_ttc_clip")
        return projected, {
            "raw_target_mps": raw_target,
            "projected_target_mps": projected,
            "lower_bound_mps": lower_bound,
            "upper_bound_mps": (None if upper_bound == float("inf") else upper_bound),
            "current_speed_mps": speed,
            "leader_id": leader_id,
            "follower_id": follower_id,
            "front_gap_m": front_gap,
            "rear_gap_m": rear_gap,
            "clip_reasons": tuple(reasons),
        }

    def _adaptive_member_targets(
        self,
        member_ids: Sequence[str],
        common_target_mps: float,
    ) -> Dict[str, float]:
        distances: Dict[str, float] = {}
        speeds: Dict[str, float] = {}
        for member_id in member_ids:
            try:
                distances[member_id] = self._approach_distance_to_merge(member_id)
            except Exception:
                pass
            speed = self._vehicle_float("getSpeed", member_id)
            if speed is not None:
                speeds[member_id] = speed
        center = sum(distances.values()) / len(distances) if distances else 0.0
        mean_speed = sum(speeds.values()) / len(speeds) if speeds else float(common_target_mps)
        raw_targets: Dict[str, float] = {}
        projected_targets: Dict[str, float] = {}
        projection: Dict[str, Dict[str, Any]] = {}
        for member_id in member_ids:
            position_error = distances.get(member_id, center) - center
            relative_speed = speeds.get(member_id, mean_speed) - mean_speed
            offset = min(
                self.compact_position_speed_offset_cap_mps,
                max(
                    -self.compact_position_speed_offset_cap_mps,
                    self.compact_position_kp_s_inv * position_error
                    - self.compact_position_kd * relative_speed,
                ),
            )
            raw_target = max(0.0, float(common_target_mps) + offset)
            target, detail = self._safety_project_target(member_id, raw_target)
            detail.update(
                {
                    "distance_to_merge_m": distances.get(member_id),
                    "formation_center_distance_m": center,
                    "position_error_m": position_error,
                    "relative_speed_mps": relative_speed,
                    "servo_offset_mps": offset,
                }
            )
            raw_targets[member_id] = raw_target
            projected_targets[member_id] = target
            projection[member_id] = detail
        self._last_member_raw_targets_mps = raw_targets
        self._last_member_speed_targets_mps = projected_targets
        self._last_safety_projection = projection
        return projected_targets

    def _member_speed_spread(self, member_ids: Sequence[str]) -> Optional[float]:
        speeds = [
            speed
            for member_id in member_ids
            if (speed := self._vehicle_float("getSpeed", member_id)) is not None
        ]
        if len(speeds) != len(member_ids) or not speeds:
            return None
        return max(speeds) - min(speeds)

    def _begin_compact_assembly(self, member_ids: Sequence[str]) -> None:
        speeds = [
            speed
            for member_id in member_ids
            if (speed := self._vehicle_float("getSpeed", member_id)) is not None
        ]
        if self.compact_control_mode == CompactControlMode.ADAPTIVE_SERVO:
            wall_target = self._freeze_adaptive_common_target(member_ids)
            self._adaptive_common_target_mps = wall_target
        else:
            wall_target_value = self.wall_params.get("wall_speed")
            wall_target = (
                min(speeds)
                if wall_target_value is None and speeds
                else max(0.0, float(wall_target_value or 0.0))
            )
            self._adaptive_common_target_mps = None
        self._assembly_target_speed_mps = max(wall_target, min(speeds)) if speeds else wall_target
        self._assembly_elapsed_steps = 0
        self._treatment_state = TreatmentState.ASSEMBLING

    def _advance_compact_assembly(self, member_ids: Sequence[str]) -> Optional[str]:
        if self._treatment_state != TreatmentState.ASSEMBLING:
            return None
        elapsed_s = self._assembly_elapsed_steps * self._simulation_step_s()
        speed_spread = self._member_speed_spread(member_ids)
        if (
            self._assembly_elapsed_steps >= self.compact_min_assembly_steps
            and speed_spread is not None
            and speed_spread <= self.compact_assembly_speed_tolerance_mps
        ):
            if self.compact_settling_max_steps > 0:
                return self._begin_settling(member_ids)
            self._treatment_state = TreatmentState.HOLDING
            return "wall_assembly_complete"
        if (
            self.compact_high_authority_assembly_steps > 0
            and self._assembly_elapsed_steps >= self.compact_high_authority_assembly_steps
        ):
            self._stop_wall("assembly_abort:high_authority_step_not_compact")
            self.last_audit["abort_reason"] = "high_authority_step_not_compact"
            return "assembly_abort:high_authority_step_not_compact"
        if elapsed_s >= self.compact_assembly_duration_s:
            self._stop_wall("assembly_abort:speed_mismatch_timeout")
            self.last_audit["abort_reason"] = "assembly_speed_mismatch_timeout"
            return "assembly_abort:speed_mismatch_timeout"
        return "wall_assembling"

    def _begin_settling(self, member_ids: Sequence[str]) -> str:
        """Latch observed assembly acceleration before relaxing wall authority."""
        actual = {
            member_id: self._vehicle_float("getAcceleration", member_id)
            for member_id in member_ids
        }
        if any(value is None for value in actual.values()):
            self._stop_wall("settling_abort:missing_realized_assembly_acceleration")
            self.last_audit["abort_reason"] = "missing_realized_assembly_acceleration"
            return "settling_abort:missing_realized_assembly_acceleration"
        self._settling_elapsed_steps = 0
        self._settling_prior_accelerations_mps2 = {
            member_id: float(value) for member_id, value in actual.items() if value is not None
        }
        self._settling_target_accelerations_mps2 = {}
        self._settling_max_jerk_mps3 = 0.0
        self._settling_tau_frozen = False
        low_floor = -self.compact_servo_a_max_dec_mps2
        if all(value >= low_floor - 1.0e-9 for value in self._settling_prior_accelerations_mps2.values()):
            self._treatment_state = TreatmentState.HOLDING
            return "wall_assembly_complete_no_settling"
        self._treatment_state = TreatmentState.SETTLING
        return "wall_assembly_complete_settling_started"

    def _advance_settling(self, member_ids: Sequence[str]) -> Optional[str]:
        """Finish only after observed acceleration reaches the low envelope."""
        if self._treatment_state != TreatmentState.SETTLING:
            return None
        actual = {
            member_id: self._vehicle_float("getAcceleration", member_id)
            for member_id in member_ids
        }
        if any(value is None for value in actual.values()):
            self._stop_wall("settling_abort:missing_realized_acceleration")
            self.last_audit["abort_reason"] = "missing_realized_acceleration"
            return "settling_abort:missing_realized_acceleration"
        self._settling_prior_accelerations_mps2 = {
            member_id: float(value) for member_id, value in actual.items() if value is not None
        }
        low_floor = -self.compact_servo_a_max_dec_mps2
        if all(value >= low_floor - 1.0e-9 for value in self._settling_prior_accelerations_mps2.values()):
            self._treatment_state = TreatmentState.HOLDING
            return "settling_complete_low_authority_envelope"
        if self._settling_elapsed_steps >= self.compact_settling_max_steps:
            self._stop_wall("settling_abort:max_duration_not_low_authority")
            self.last_audit["abort_reason"] = "settling_max_duration_not_low_authority"
            return "settling_abort:max_duration_not_low_authority"
        return "wall_settling"

    def _apply_settling_actuator(self, inband_obs: Mapping[str, Any]) -> None:
        """Issue per-member, realized-acceleration-based jerk-limited targets."""
        if self._wall is None:
            raise RuntimeError("policy has not been reset")
        dt_s = self._simulation_step_s()
        jerk_delta = float(self.compact_settling_jerk_limit_mps3 or 0.0) * dt_s
        low_floor = -self.compact_servo_a_max_dec_mps2
        speeds: Dict[str, float] = {}
        targets: Dict[str, float] = {}
        target_accelerations: Dict[str, float] = {}
        for member_id in inband_obs:
            prior = self._settling_prior_accelerations_mps2.get(member_id)
            speed = self._vehicle_float("getSpeed", member_id)
            if prior is None or speed is None:
                self._stop_wall("settling_abort:missing_member_state")
                self.last_audit["abort_reason"] = "settling_missing_member_state"
                return
            # Only relax a restrictive actual assembly response.  A member
            # already inside the low-authority envelope receives no new
            # tightening during this transition.
            target_acceleration = (
                min(low_floor, float(prior) + jerk_delta)
                if float(prior) < low_floor
                else float(prior)
            )
            speeds[member_id] = float(speed)
            target_accelerations[member_id] = target_acceleration
            targets[member_id] = max(0.0, float(speed) + target_acceleration * dt_s)
        authority_cap = max(
            self.compact_servo_a_max_dec_mps2,
            max((-value for value in target_accelerations.values() if value < 0.0), default=0.0),
        )
        tau_before = dict(getattr(self._wall, "_tau_offset_cache", {}))
        original_deceleration = getattr(self._wall, "a_max_dec", None)
        try:
            self._last_member_raw_targets_mps = dict(targets)
            self._last_member_speed_targets_mps = dict(targets)
            self._current_wall_speed_target_mps = None
            self._last_wall_a_max_dec_mps2 = authority_cap
            self._wall.a_max_dec = authority_cap
            self._wall.act_with_targets(dict(inband_obs), targets)
            # SETTLING may never increase tau/headway authority.  Restore the
            # exact pre-call fraction after WallHoldPolicy's normal ramp.
            tau_blend = getattr(self._wall, "set_tau_blend", None)
            tau_for = getattr(self._wall, "_tau_for", None)
            if callable(tau_blend) and callable(tau_for):
                fractions = []
                for member_id in inband_obs:
                    full_offset = max(0.0, float(tau_for(member_id)))
                    fractions.append(
                        0.0
                        if full_offset <= 0.0
                        else max(0.0, min(1.0, float(tau_before.get(member_id, 0.0)) / full_offset))
                    )
                tau_blend(tuple(inband_obs), min(fractions, default=0.0))
            self._settling_tau_frozen = True
        finally:
            self._wall.a_max_dec = original_deceleration
        self._settling_target_accelerations_mps2 = target_accelerations
        self._settling_max_jerk_mps3 = max(
            self._settling_max_jerk_mps3,
            max(
                abs(target_accelerations[member_id] - self._settling_prior_accelerations_mps2[member_id]) / dt_s
                for member_id in target_accelerations
            ),
        )
        self._settling_elapsed_steps += 1
        self.last_audit["transition_reason"] = "wall_settling"

    def _apply_wall_actuator(
        self,
        inband_obs: Mapping[str, Any],
        *,
        freeze_tightening: bool = False,
    ) -> None:
        if self._wall is None:
            raise RuntimeError("policy has not been reset")
        if self.compact_control_mode == CompactControlMode.ADAPTIVE_SERVO:
            common_target = (
                self._assembly_target_speed_mps
                if self._treatment_state == TreatmentState.ASSEMBLING
                else self._adaptive_common_target_mps
            )
            if common_target is None:
                common_target = max(0.0, float(self.wall_params.get("wall_speed") or 0.0))
            prior_targets = dict(self._last_member_speed_targets_mps)
            targets = self._adaptive_member_targets(
                tuple(sorted(str(x) for x in inband_obs)),
                float(common_target),
            )
            tau_before = dict(getattr(self._wall, "_tau_offset_cache", {}))
            if freeze_tightening:
                # A lower speed target is stronger restriction.  Once a soft
                # follower-compression sample is observed, preserve the least
                # restrictive target already commanded and restore the previous
                # tau fraction after the actuator call so neither lever tightens.
                for member_id, target in tuple(targets.items()):
                    previous = prior_targets.get(member_id)
                    if previous is not None and target < previous:
                        targets[member_id] = float(previous)
                        self._last_safety_projection.setdefault(member_id, {})[
                            "adaptive_release_speed_freeze"
                        ] = True
            actuator = getattr(self._wall, "act_with_targets", None)
            if not callable(actuator):
                raise RuntimeError("adaptive compact control requires act_with_targets")
            original_deceleration = getattr(self._wall, "a_max_dec", None)
            try:
                self._current_wall_speed_target_mps = float(common_target)
                authority_cap = (
                    self.compact_initial_assembly_servo_a_max_dec_mps2
                    if self._treatment_state == TreatmentState.ASSEMBLING
                    else self.compact_servo_a_max_dec_mps2
                )
                self._last_wall_a_max_dec_mps2 = authority_cap
                self._wall.a_max_dec = authority_cap
                actuator(dict(inband_obs), targets)
                if freeze_tightening:
                    tau_blend = getattr(self._wall, "set_tau_blend", None)
                    tau_for = getattr(self._wall, "_tau_for", None)
                    if callable(tau_blend) and callable(tau_for):
                        fractions = []
                        for member_id in inband_obs:
                            full_offset = max(0.0, float(tau_for(member_id)))
                            fractions.append(
                                0.0
                                if full_offset <= 0.0
                                else max(
                                    0.0,
                                    min(1.0, float(tau_before.get(member_id, 0.0)) / full_offset),
                                )
                            )
                        # The per-member offsets normally agree for this wall;
                        # use their minimum so a member is never tightened.
                        tau_blend(tuple(inband_obs), min(fractions, default=0.0))
                if self._treatment_state == TreatmentState.ASSEMBLING:
                    self._assembly_elapsed_steps += 1
            finally:
                self._wall.a_max_dec = original_deceleration
            return
        self._current_wall_speed_target_mps = self.wall_params.get("wall_speed")
        self._last_wall_a_max_dec_mps2 = float(self.wall_params["a_max_dec"])
        if self._treatment_state != TreatmentState.ASSEMBLING:
            self._wall.act(dict(inband_obs))
            return
        target = self._assembly_target_speed_mps
        original_target = getattr(self._wall, "wall_speed", None)
        original_deceleration = getattr(self._wall, "a_max_dec", None)
        try:
            self._current_wall_speed_target_mps = target
            self._last_wall_a_max_dec_mps2 = self.compact_assembly_a_max_dec_mps2
            self._wall.wall_speed = target
            self._wall.a_max_dec = self.compact_assembly_a_max_dec_mps2
            self._wall.act(dict(inband_obs))
            self._assembly_elapsed_steps += 1
        finally:
            self._wall.wall_speed = original_target
            self._wall.a_max_dec = original_deceleration

    def _current_member_span_m(self, member_ids: Sequence[str]) -> Optional[float]:
        distances: list[float] = []
        for member_id in member_ids:
            try:
                distances.append(self._approach_distance_to_merge(member_id))
            except Exception:
                return None
        return max(distances) - min(distances) if distances else None

    def _compactness_failure_reason(self, member_ids: Sequence[str]) -> Optional[str]:
        """Apply the frozen v5.3 two-sample compactness fail-safe."""

        if self.compactness_enforcement_samples <= 0:
            return None
        span = self._current_member_span_m(member_ids)
        if span is None or span <= self.max_longitudinal_span_m:
            self._compactness_excess_samples = 0
            return None
        self._compactness_excess_samples += 1
        if self._compactness_excess_samples >= self.compactness_enforcement_samples:
            return "compactness_lost"
        return None

    def _base_target_speed_mps(self, member_id: str) -> float:
        """Recover the command already applied by the identical damping arm."""

        diagnostics = getattr(self.base_policy, "last_diagnostics", {}) or {}
        for collection in (diagnostics.get("commands", ()), diagnostics.get("release", ())):
            for row in collection or ():
                if not isinstance(row, Mapping) or str(row.get("vehicle_id")) != member_id:
                    continue
                projected = row.get("projected") or {}
                value = projected.get("target_speed_mps", row.get("target_speed_mps"))
                if value is not None:
                    try:
                        return max(0.0, float(value))
                    except (TypeError, ValueError):
                        pass
        try:
            getter = getattr(traci.vehicle, "getSpeedWithoutTraCI")
            return max(0.0, float(getter(member_id)))
        except Exception:
            return max(0.0, float(self._vehicle_float("getSpeed", member_id) or 0.0))

    def _begin_taper(self, reason: str, member_ids: Sequence[str]) -> None:
        if self.taper_duration_s <= 0.0:
            self._stop_wall(reason)
            return
        if self._treatment_state == TreatmentState.TAPERING:
            return
        self._treatment_state = TreatmentState.TAPERING
        self._taper_elapsed_steps = 0
        self._taper_reason = str(reason)
        self._taper_origin_targets_mps = {
            member_id: float(
                self._last_member_speed_targets_mps.get(
                    member_id,
                    self._vehicle_float("getSpeed", member_id) or 0.0,
                )
            )
            for member_id in member_ids
        }
        self.last_audit["transition_reason"] = f"taper_started:{reason}"

    def _taper_step_count(self) -> int:
        return max(1, round(self.taper_duration_s / self._simulation_step_s()))

    def _apply_taper_actuator(self, inband_obs: Mapping[str, Any]) -> None:
        if self._wall is None:
            raise RuntimeError("policy has not been reset")
        self._taper_elapsed_steps += 1
        total = self._taper_step_count()
        blend = max(0.0, 1.0 - self._taper_elapsed_steps / total)
        base_targets = {
            member_id: self._base_target_speed_mps(member_id) for member_id in inband_obs
        }
        targets = {
            member_id: base_targets[member_id]
            + blend
            * (
                self._taper_origin_targets_mps.get(member_id, base_targets[member_id])
                - base_targets[member_id]
            )
            for member_id in inband_obs
        }
        projected_targets: Dict[str, float] = {}
        projection: Dict[str, Dict[str, Any]] = {}
        for member_id, raw_target in targets.items():
            target, detail = self._safety_project_target(member_id, raw_target)
            detail.update(
                {
                    "taper_blend_fraction": blend,
                    "taper_base_target_mps": base_targets[member_id],
                    "taper_origin_target_mps": self._taper_origin_targets_mps.get(member_id),
                }
            )
            projected_targets[member_id] = target
            projection[member_id] = detail
        self._last_member_raw_targets_mps = dict(targets)
        self._last_member_speed_targets_mps = projected_targets
        self._last_safety_projection = projection
        original_deceleration = getattr(self._wall, "a_max_dec", None)
        try:
            self._last_wall_a_max_dec_mps2 = self.compact_servo_a_max_dec_mps2
            self._wall.a_max_dec = self.compact_servo_a_max_dec_mps2
            self._wall.act_with_targets(dict(inband_obs), projected_targets)
        finally:
            self._wall.a_max_dec = original_deceleration
        tau_blend = getattr(self._wall, "set_tau_blend", None)
        self._last_tau_taper = (
            tuple(tau_blend(tuple(inband_obs), blend)) if callable(tau_blend) else ()
        )
        self._last_taper_blend_fraction = blend
        self._last_taper_base_targets_mps = base_targets
        self._current_wall_speed_target_mps = None
        self.last_audit["transition_reason"] = (
            f"taper_complete:{self._taper_reason}"
            if blend <= 0.0
            else f"wall_tapering:{self._taper_reason}"
        )

    def _complete_taper(self) -> None:
        tau_restorations = self._restore_wall_tau(self._pulse_member_ids_seen)
        self._wall_active = False
        self._last_controlled_ids = ()
        self._cooldown_remaining = self.cooldown_steps
        self._treatment_state = TreatmentState.RELEASED
        self.last_audit["transition_reason"] = f"taper_released:{self._taper_reason}"
        self.last_audit["tau_restorations"] = tau_restorations

    def _stop_wall(self, reason: str) -> None:
        tau_restorations = self._restore_wall_tau(self._pulse_member_ids_seen)
        self._wall_active = False
        self._last_controlled_ids = ()
        self._assembly_elapsed_steps = 0
        self._assembly_target_speed_mps = None
        self._settling_elapsed_steps = 0
        self._settling_prior_accelerations_mps2 = {}
        self._settling_target_accelerations_mps2 = {}
        self._settling_max_jerk_mps3 = 0.0
        self._settling_tau_frozen = False
        self._adaptive_common_target_mps = None
        self._adaptive_reference_speed_mps = None
        self._adaptive_downstream_speed_mps = None
        self._adaptive_speed_reduction_fraction = 0.0
        self._last_member_speed_targets_mps = {}
        self._last_member_raw_targets_mps = {}
        self._last_safety_projection = {}
        self._current_wall_speed_target_mps = self.wall_params.get("wall_speed")
        self._last_wall_a_max_dec_mps2 = None
        self._cooldown_remaining = self.cooldown_steps
        self._treatment_state = (
            TreatmentState.RELEASED
            if self.treatment_integrity_version in ("v5.3", "v5.4", "v5.5.1")
            else TreatmentState.RELEASING
        )
        if self._wall is not None:
            reset_transition = getattr(self._wall, "reset_transition", None)
            if callable(reset_transition):
                reset_transition()
        self.last_audit["transition_reason"] = reason
        self.last_audit["tau_restorations"] = tau_restorations

    def _base_audit(self, base_actions: Mapping[str, object]) -> Dict[str, Any]:
        base_diagnostics = getattr(self.base_policy, "last_diagnostics", {}) or {}
        direct_control_ids = tuple(
            sorted(
                str(row["vehicle_id"])
                for row in base_diagnostics.get("commands", ())
                if isinstance(row, Mapping) and row.get("vehicle_id") is not None
            )
        )
        base_control_ids = tuple(
            sorted(set(str(x) for x in base_actions) | set(direct_control_ids))
        )
        return {
            "geometry_mode": self.geometry_mode.value,
            "recruitment_band_m": (
                float(self.wall_params["d_lo"]),
                float(self.wall_params["d_hi"]),
            ),
            "max_longitudinal_span_m": self.max_longitudinal_span_m,
            "planned_treatment_duration_s": self.planned_treatment_duration_s,
            "compact_release_boundary_m": self.compact_release_boundary_m,
            "compact_selection_mode": self.compact_selection_mode,
            "compact_control_mode": self.compact_control_mode.value,
            "compact_servo_a_max_dec_mps2": (self.compact_servo_a_max_dec_mps2),
            "compact_initial_assembly_servo_a_max_dec_mps2": (
                self.compact_initial_assembly_servo_a_max_dec_mps2
            ),
            "compact_high_authority_assembly_steps": self.compact_high_authority_assembly_steps,
            "compact_settling_jerk_limit_mps3": self.compact_settling_jerk_limit_mps3,
            "compact_settling_max_steps": self.compact_settling_max_steps,
            "compact_max_activation_speed_spread_mps": (
                self.compact_max_activation_speed_spread_mps
            ),
            "compact_max_speed_reduction_fraction": (self.compact_max_speed_reduction_fraction),
            "compact_shock_gap_lo_mps": self.compact_shock_gap_lo_mps,
            "compact_shock_gap_hi_mps": self.compact_shock_gap_hi_mps,
            "compact_position_kp_s_inv": self.compact_position_kp_s_inv,
            "compact_position_kd": self.compact_position_kd,
            "compact_position_speed_offset_cap_mps": (self.compact_position_speed_offset_cap_mps),
            "compact_safety_min_front_ttc_s": (self.compact_safety_min_front_ttc_s),
            "compact_safety_min_rear_ttc_s": (self.compact_safety_min_rear_ttc_s),
            "compact_influence_upstream_m": self.compact_influence_upstream_m,
            "compact_influence_max_followers": (self.compact_influence_max_followers),
            "compact_influence_min_ttc_s": self.compact_influence_min_ttc_s,
            "compact_assembly_duration_s": self.compact_assembly_duration_s,
            "compact_assembly_speed_tolerance_mps": (self.compact_assembly_speed_tolerance_mps),
            "compact_assembly_a_max_dec_mps2": (self.compact_assembly_a_max_dec_mps2),
            "treatment_integrity_version": self.treatment_integrity_version,
            "compact_handoff_enabled": self.compact_handoff_enabled,
            "max_handoffs_per_lane": self.max_handoffs_per_lane,
            "compactness_enforcement_samples": self.compactness_enforcement_samples,
            "taper_duration_s": self.taper_duration_s,
            "taper_function": self.taper_function,
            "adaptive_release_enabled": self.adaptive_release_enabled,
            "adaptive_release_soft_rear_gap_m": self.adaptive_release_soft_rear_gap_m,
            "adaptive_release_soft_rear_ttc_s": self.adaptive_release_soft_rear_ttc_s,
            "adaptive_release_soft_follower_accel_mps2": (
                self.adaptive_release_soft_follower_accel_mps2
            ),
            "adaptive_release_persistence_samples": self.adaptive_release_persistence_samples,
            "adaptive_release_missing_follower_behavior": (
                self.adaptive_release_missing_follower_behavior
            ),
            "assembly_elapsed_steps": self._assembly_elapsed_steps,
            "assembly_target_speed_mps": self._assembly_target_speed_mps,
            "settling_elapsed_steps": self._settling_elapsed_steps,
            "settling_prior_accelerations_mps2": dict(self._settling_prior_accelerations_mps2),
            "settling_target_accelerations_mps2": dict(self._settling_target_accelerations_mps2),
            "settling_max_jerk_mps3": self._settling_max_jerk_mps3,
            "settling_tau_frozen": self._settling_tau_frozen,
            "adaptive_common_target_mps": self._adaptive_common_target_mps,
            "adaptive_reference_speed_mps": self._adaptive_reference_speed_mps,
            "adaptive_downstream_speed_mps": (self._adaptive_downstream_speed_mps),
            "adaptive_speed_reduction_fraction": (self._adaptive_speed_reduction_fraction),
            "command": int(self._command),
            "command_name": self._command.name,
            "base_action_ids": tuple(sorted(str(x) for x in base_actions)),
            "base_actions": dict(base_actions),
            "base_direct_control_ids": direct_control_ids,
            "base_control_ids": base_control_ids,
            "base_diagnostics": base_diagnostics,
            "applied_env_actions": dict(base_actions),
            "wall_active": False,
            "wall_override_ids": (),
            "out_of_band_damping_ids": tuple(sorted(base_control_ids)),
            "cooldown_remaining": self._cooldown_remaining,
            "transition_reason": "damping_only",
            "formation": None,
            "activation_safety": None,
            "treatment_state": self._treatment_state.value,
            "frozen_lane_ids": self._frozen_lane_ids,
            "member_slot_count": sum(len(slots) for slots in self._member_slots.values()),
            "wall_member_ids": (),
            "member_changes": (),
            "handoff_rejected_ids": (),
            "reserve_ids_by_lane": {
                lane_id: tuple(values) for lane_id, values in self._reserve_ids_by_lane.items()
            },
            "handoff_counts_by_lane": dict(self._handoff_counts_by_lane),
            "active_treatment_steps": self._active_treatment_steps,
            "compactness_excess_samples": self._compactness_excess_samples,
            "taper_elapsed_steps": self._taper_elapsed_steps,
            "taper_reason": self._taper_reason,
            "taper_blend_fraction": self._last_taper_blend_fraction,
            "taper_base_targets_mps": dict(self._last_taper_base_targets_mps),
            "tau_taper": self._last_tau_taper,
            "lane_coverage": self._coverage_audit(()),
            "control_deltas": {},
            "abort_reason": None,
            "emergency_taper_zero_handoff": False,
            "adaptive_release": dict(self._adaptive_release_last_compression),
            "adaptive_release_latched": self._adaptive_release_latched,
            "adaptive_release_first_trigger": self._adaptive_release_first_trigger,
        }

    def act(self, obs_dict: Dict[str, Any]) -> Dict[str, object]:
        if self._treatment_state == TreatmentState.RELEASING and not self._wall_active:
            self._treatment_state = (
                TreatmentState.COOLDOWN if self._cooldown_remaining > 0 else TreatmentState.DAMPING
            )
        # This call is unconditional: both causal arms evaluate the identical
        # damping policy for every active CAV at every decision step.
        base_actions = self.base_policy.act(obs_dict)
        self.last_audit = self._base_audit(base_actions)
        base_control_ids = set(self.last_audit["base_control_ids"])
        live_ids = self._live_ids()

        if (
            self._wall_active
            and self._treatment_state == TreatmentState.TAPERING
            and self._last_taper_blend_fraction == 0.0
        ):
            self._complete_taper()

        aborted_this_step = False
        if self._wall_active:
            abort_reason = self._safety_abort_reason(live_ids)
            if abort_reason is not None:
                if (
                    self.treatment_integrity_version in ("v5.4", "v5.5.1", "v5.7", "v5.7.1")
                    and self._treatment_state == TreatmentState.TAPERING
                ):
                    # A hard safety abort must stop wall authority immediately.
                    # Record that physical handoff as the zero endpoint of the
                    # taper: this step returns the already-computed identical
                    # damping action and performs no further wall actuation.
                    self._last_taper_blend_fraction = 0.0
                    self._last_taper_base_targets_mps = {
                        member_id: self._base_target_speed_mps(member_id)
                        for member_id in self._last_controlled_ids
                    }
                    self._complete_taper()
                    self.last_audit["transition_reason"] = f"safety_abort:{abort_reason}"
                    self.last_audit["emergency_taper_zero_handoff"] = True
                else:
                    self._stop_wall(f"safety_abort:{abort_reason}")
                self.last_audit["abort_reason"] = abort_reason
                aborted_this_step = True

        if self._command == DAMPING_ONLY:
            if self._wall_active:
                self._stop_wall("pulse_released_to_damping")
            elif self._cooldown_remaining > 0 and not aborted_this_step:
                self._cooldown_remaining -= 1
                self._treatment_state = (
                    TreatmentState.COOLDOWN
                    if self._cooldown_remaining > 0
                    else TreatmentState.DAMPING
                )
            elif self._treatment_state != TreatmentState.RELEASING:
                self._treatment_state = TreatmentState.DAMPING
            self._previous_command = self._command
            self.last_audit.update(
                {
                    "wall_active": False,
                    "cooldown_remaining": self._cooldown_remaining,
                    "out_of_band_damping_ids": tuple(sorted(base_control_ids)),
                    "treatment_state": self._treatment_state.value,
                    "frozen_lane_ids": self._frozen_lane_ids,
                    "member_slot_count": sum(len(slots) for slots in self._member_slots.values()),
                    "wall_member_ids": (),
                    "lane_coverage": self._coverage_audit(()),
                }
            )
            if self.diagnostic_formation_preview:
                live_ids = self._live_ids()
                preview = self._formation_status(obs_dict.keys(), live_ids)
                self.last_audit["formation_preview"] = preview.to_dict()
                if preview.feasible:
                    self.last_audit["activation_safety_preview"] = (
                        self._activation_safety_status(
                            preview.controlled_cav_ids, live_ids
                        ).to_dict()
                    )
            # Return the exact object from the base policy: the off arm is
            # action-for-action identical and performs no direct TraCI writes.
            return base_actions

        rising_edge = self._previous_command != DAMPING_PLUS_WALL
        formation = self._formation_status(obs_dict.keys(), live_ids)
        self.last_audit["formation"] = formation.to_dict()
        activation_safety: Optional[ActivationSafetyStatus] = None
        if rising_edge and formation.feasible:
            activation_safety = self._activation_safety_status(
                formation.controlled_cav_ids, live_ids
            )
            self.last_audit["activation_safety"] = activation_safety.to_dict()

        started_this_step = False
        member_changes: list[Dict[str, Any]] = []
        rejected_handoffs: tuple[str, ...] = ()
        if not self._wall_active:
            if aborted_this_step:
                pass
            elif not rising_edge:
                self.last_audit["transition_reason"] = "wall_request_latched"
            elif self._cooldown_remaining > 0:
                self.last_audit["transition_reason"] = "cooldown_veto"
            elif not formation.feasible:
                self.last_audit["transition_reason"] = f"formation_veto:{formation.reason}"
            elif activation_safety is not None and not activation_safety.safe:
                self.last_audit["transition_reason"] = (
                    f"activation_safety_veto:{activation_safety.reason}"
                )
            else:
                member_changes = self._latch_members(formation, obs_dict.keys(), live_ids)
                self._wall_active = True
                started_this_step = True
                self._active_treatment_steps = 0
                self._hold_elapsed_steps = 0
                self._compactness_excess_samples = 0
                self._taper_elapsed_steps = 0
                self._taper_reason = None
                self._taper_origin_targets_mps = {}
                self._last_taper_blend_fraction = None
                self._last_taper_base_targets_mps = {}
                self._last_tau_taper = ()
                self._adaptive_release_latched = False
                self._adaptive_release_soft_samples = 0
                self._adaptive_release_first_trigger = None
                self._adaptive_release_follower_ids = {}
                self._adaptive_release_follower_baselines = {}
                self._adaptive_release_previous_gaps_m = {}
                self._adaptive_release_last_compression = {}
                if self._wall is not None:
                    reset_transition = getattr(self._wall, "reset_transition", None)
                    if callable(reset_transition):
                        reset_transition()
                initial_member_ids = tuple(
                    sorted(
                        member_id
                        for slots in self._member_slots.values()
                        for member_id in slots
                        if member_id is not None
                    )
                )
                if (
                    self.geometry_mode == WallGeometry.COMPACT_MOVING
                    and self.compact_assembly_duration_s > 0.0
                ):
                    self._begin_compact_assembly(initial_member_ids)
                    self.last_audit["transition_reason"] = "wall_assembly_started"
                else:
                    self._treatment_state = TreatmentState.HOLDING
                    self.last_audit["transition_reason"] = "wall_started"

        wall_ids: tuple[str, ...] = ()
        if self._wall_active:
            self._active_treatment_steps += 1
            if started_this_step:
                wall_ids = tuple(
                    sorted(
                        member_id
                        for slots in self._member_slots.values()
                        for member_id in slots
                        if member_id is not None
                    )
                )
            else:
                wall_ids, member_changes, rejected_handoffs = self._refresh_members(
                    obs_dict.keys(), live_ids
                )
                self.last_audit["transition_reason"] = (
                    "wall_holding"
                    if formation.feasible
                    else f"wall_holding_with_formation_gap:{formation.reason}"
                )
                if self._treatment_state == TreatmentState.SETTLING and member_changes:
                    self._stop_wall("settling_abort:member_identity_changed")
                    self.last_audit["abort_reason"] = "settling_member_identity_changed"
                    wall_ids = ()
            coverage_now = self._coverage_audit(wall_ids)
            if (
                self._wall_active
                and self.compact_handoff_enabled
                and coverage_now["coverage_fraction"] < 1.0
                and self._treatment_state != TreatmentState.TAPERING
            ):
                if self._treatment_state == TreatmentState.SETTLING:
                    self._stop_wall("settling_abort:occupied_lane_coverage_lost")
                    self.last_audit["abort_reason"] = "settling_occupied_lane_coverage_lost"
                    wall_ids = ()
                else:
                    self._begin_taper("occupied_lane_coverage_lost", wall_ids)
            compactness_reason = (
                self._compactness_failure_reason(wall_ids)
                if self._wall_active and self._treatment_state != TreatmentState.TAPERING
                else None
            )
            if compactness_reason is not None:
                if self._treatment_state == TreatmentState.SETTLING:
                    self._stop_wall(f"settling_abort:{compactness_reason}")
                    self.last_audit["abort_reason"] = f"settling_{compactness_reason}"
                    wall_ids = ()
                else:
                    self._begin_taper(compactness_reason, wall_ids)
            if (
                self._wall_active
                and self.taper_duration_s > 0.0
                and self._treatment_state == TreatmentState.HOLDING
                and self.compact_requested_hold_duration_s is not None
                and self._hold_elapsed_steps >= round(
                    self.compact_requested_hold_duration_s / self._simulation_step_s()
                )
            ):
                self._begin_taper("requested_hold_complete", wall_ids)
            elif (
                self._wall_active
                and self.taper_duration_s > 0.0
                and self._treatment_state != TreatmentState.TAPERING
            ):
                planned_steps = max(
                    1,
                    round(
                        float(self.planned_treatment_duration_s or 0.0)
                        / self._simulation_step_s()
                    ),
                )
                if self._active_treatment_steps > planned_steps - self._taper_step_count():
                    self._begin_taper("planned_in_horizon_release", wall_ids)
            if (
                self._wall_active
                and self.geometry_mode == WallGeometry.COMPACT_MOVING
                and self._treatment_state == TreatmentState.ASSEMBLING
                and not started_this_step
            ):
                assembly_transition = self._advance_compact_assembly(wall_ids)
                if assembly_transition is not None:
                    self.last_audit["transition_reason"] = assembly_transition
                if not self._wall_active:
                    wall_ids = ()
            if (
                self._wall_active
                and self.geometry_mode == WallGeometry.COMPACT_MOVING
                and self._treatment_state == TreatmentState.SETTLING
                and not started_this_step
            ):
                settling_transition = self._advance_settling(wall_ids)
                if settling_transition is not None:
                    self.last_audit["transition_reason"] = settling_transition
                if not self._wall_active:
                    wall_ids = ()
            adaptive_status: Dict[str, Any] = {
                "enabled": False,
                "should_taper": False,
                "freeze_tightening": False,
            }
            if self._wall_active and self._treatment_state == TreatmentState.HOLDING:
                adaptive_status = self._adaptive_release_status(wall_ids, live_ids)
                if bool(adaptive_status.get("should_taper")):
                    self._adaptive_release_latched = True
                    reasons = ",".join(adaptive_status.get("soft_reasons") or ("soft_compression",))
                    self._begin_taper(f"adaptive_soft_compression:{reasons}", wall_ids)
                    adaptive_status["latched"] = True
            inband_obs = {cav_id: obs_dict[cav_id] for cav_id in wall_ids if cav_id in obs_dict}
            if inband_obs:
                # The wall has no trigger and receives only in-band CAVs, so
                # WallHoldPolicy cannot release or alter out-of-band CAVs.
                if self._treatment_state == TreatmentState.TAPERING:
                    self._apply_taper_actuator(inband_obs)
                elif self._treatment_state == TreatmentState.SETTLING:
                    self._apply_settling_actuator(inband_obs)
                else:
                    self._apply_wall_actuator(
                        inband_obs,
                        freeze_tightening=bool(adaptive_status.get("freeze_tightening")),
                    )
            elif self._treatment_state == TreatmentState.TAPERING:
                self._complete_taper()
            self._last_controlled_ids = wall_ids
            if self._wall_active and self._treatment_state == TreatmentState.HOLDING:
                self._hold_elapsed_steps += 1

        self._previous_command = self._command
        if not self._wall_active and self._cooldown_remaining > 0 and not aborted_this_step:
            self._cooldown_remaining -= 1
            self._treatment_state = (
                TreatmentState.COOLDOWN if self._cooldown_remaining > 0 else TreatmentState.DAMPING
            )
        wall_set = set(wall_ids)
        coverage = self._coverage_audit(wall_ids)
        common_audit = {
            "wall_active": self._wall_active,
            "wall_override_ids": wall_ids,
            "cooldown_remaining": self._cooldown_remaining,
            "treatment_state": self._treatment_state.value,
            "frozen_lane_ids": self._frozen_lane_ids,
            "member_slot_count": sum(len(slots) for slots in self._member_slots.values()),
            "wall_member_ids": wall_ids,
            "member_changes": tuple(member_changes),
            "handoff_rejected_ids": rejected_handoffs,
            "lane_coverage": coverage,
            "control_deltas": self._control_delta_audit(wall_ids, base_actions),
            "assembly_elapsed_steps": self._assembly_elapsed_steps,
            "assembly_elapsed_s": (self._assembly_elapsed_steps * self._simulation_step_s()),
            "hold_elapsed_steps": self._hold_elapsed_steps,
            "hold_elapsed_s": self._hold_elapsed_steps * self._simulation_step_s(),
            "requested_hold_duration_s": self.compact_requested_hold_duration_s,
            "assembly_target_speed_mps": self._assembly_target_speed_mps,
            "member_speed_spread_mps": self._member_speed_spread(wall_ids),
            "current_member_span_m": self._current_member_span_m(wall_ids),
            "treatment_integrity_version": self.treatment_integrity_version,
            "active_treatment_steps": self._active_treatment_steps,
            "compactness_excess_samples": self._compactness_excess_samples,
            "reserve_ids_by_lane": {
                lane_id: tuple(values) for lane_id, values in self._reserve_ids_by_lane.items()
            },
            "handoff_counts_by_lane": dict(self._handoff_counts_by_lane),
            "taper_duration_s": self.taper_duration_s,
            "taper_function": self.taper_function,
            "taper_elapsed_steps": self._taper_elapsed_steps,
            "taper_reason": self._taper_reason,
            "taper_blend_fraction": self._last_taper_blend_fraction,
            "taper_base_targets_mps": dict(self._last_taper_base_targets_mps),
            "tau_taper": self._last_tau_taper,
            "adaptive_common_target_mps": self._adaptive_common_target_mps,
            "adaptive_reference_speed_mps": self._adaptive_reference_speed_mps,
            "adaptive_downstream_speed_mps": (self._adaptive_downstream_speed_mps),
            "adaptive_speed_reduction_fraction": (self._adaptive_speed_reduction_fraction),
            "member_raw_speed_targets_mps": dict(self._last_member_raw_targets_mps),
            "member_projected_speed_targets_mps": dict(self._last_member_speed_targets_mps),
            "safety_projection": dict(self._last_safety_projection),
            "adaptive_release": dict(self._adaptive_release_last_compression),
            "adaptive_release_latched": self._adaptive_release_latched,
            "adaptive_release_first_trigger": self._adaptive_release_first_trigger,
        }
        if not wall_set:
            self.last_audit.update(
                {
                    **common_audit,
                    "wall_override_ids": (),
                    "out_of_band_damping_ids": tuple(sorted(base_control_ids)),
                    "applied_env_actions": dict(base_actions),
                }
            )
            return base_actions
        out_actions = {
            cav_id: action for cav_id, action in base_actions.items() if cav_id not in wall_set
        }
        self.last_audit.update(
            {
                **common_audit,
                "out_of_band_damping_ids": tuple(
                    sorted((base_control_ids - wall_set) | set(str(x) for x in out_actions))
                ),
                "applied_env_actions": dict(out_actions),
            }
        )
        return out_actions

    def close(self) -> None:
        if self._base is not None:
            self._base.close()
