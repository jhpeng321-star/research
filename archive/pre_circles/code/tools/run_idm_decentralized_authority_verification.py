"""Resumable paired pretraining verification for decentralized IDM damping."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import traci

from env.core import TrafficShepherdMultiAgentEnv
from env.features import (
    RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
    RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1,
    RSU_DIAGNOSTIC_RANGE_M,
    RSU_OBSERVATION_CONTRACTS,
)
from env.idm_residual import (
    ProjectionInput,
    analytical_acceleration,
    detect_disturbances,
    project_command,
)
from env.snapshot import get_active_snapshot
from experiments import REPO_ROOT
from experiments.idm.run_idm_residual_checkpoint_eval import _env_config
from experiments.io import atomic_write_json
from idm.action_local import (
    LandmarkContract,
    classify_effect,
    landmark_signal,
    replay_deadband,
    safety_composite,
)
from idm.decentralized_damping import (
    ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
    ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
    ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
    ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
    ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
    mode_by_index,
    mode_count,
    modes_for,
    nominal_mode_index,
    terminal_reference_percentile,
)
from idm.follower_stopper import MODIFIED_FOLLOWER_STOPPER_P30_V1
from idm.projected_release import (
    PROJECTED_RELEASE_V1,
    PROJECTED_RELEASE_V2,
    PROJECTED_RELEASE_V3,
)
from idm.projected_blend import IDM_P30_PROJECTED_BLEND_V1
from idm.rear_aware_feedback import REAR_AWARE_FEEDBACK_V1, REAR_AWARE_FEEDBACK_V2
from idm.contract import (
    CONTROLLER_FEATURE_VALIDATION_SEEDS,
    DAMPING_ACCEPTANCE_SEEDS,
)
from idm.phase_diagnostics import (
    PHASE_DIAGNOSTIC_SCHEMA,
    TreatmentPhaseTracker,
    cumulative_paired_deltas,
    phase_step_diagnostic,
)
from idm.spatial_history import (
    SPATIAL_HISTORY_SCHEMA,
    SpatialHistoryContract,
    SpatialRSUHistory,
    capture_corridor_frame,
)
from idm.spatial_response import capture_spatial_response
from tools.audit_idm_phase5_decision_state_sufficiency import _canonical_hash
from tools.run_idm_structured_authority_screen import (
    _merge_discharge_step,
    _snapshot_acceleration,
)


ROOT = REPO_ROOT
SCHEMA = "idm_decentralized_authority/1"
RAW_LOCAL_DIM = 134
CLASSIC_RSU_SLICE = slice(9, 17)

_PHYSICAL_CANDIDATE_FIELDS = (
    "slot",
    "vehicle_id",
    "lane_id",
    "geometric_rank",
    "distance_to_merge_m",
    "disturbance_distance_to_merge_m",
    "runway_m",
    "predicted_dose_s",
    "front_ttc_s",
    "rear_ttc_s",
    "front_gap_m",
    "rear_gap_m",
    "front_secure_gap_m",
    "rear_secure_gap_m",
    "prospective_rear_secure_gap_m",
    "prospective_rear_headroom_m",
    "lane_change_state",
)

_RAW_DECISION_CONTEXT_FIELDS = tuple(
    field
    for field in _PHYSICAL_CANDIDATE_FIELDS
    if field
    not in {
        "prospective_rear_secure_gap_m",
        "prospective_rear_headroom_m",
    }
)


def _terminal_speed_for_action_set(action_set: str, disturbance: Any) -> float:
    percentile = terminal_reference_percentile(action_set)
    if percentile is None:
        percentile = "p30"
    return float(getattr(disturbance, f"speed_{percentile}_mps"))


def _projected_release_contract_for_action_set(action_set: str):
    if action_set == ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET:
        return PROJECTED_RELEASE_V1
    if action_set == ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET:
        return PROJECTED_RELEASE_V2
    if action_set == ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET:
        return PROJECTED_RELEASE_V3
    if action_set == ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET:
        return PROJECTED_RELEASE_V2
    if action_set == ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET:
        return PROJECTED_RELEASE_V2
    return None


def _feedback_contract_for_action_set(action_set: str):
    if action_set == ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET:
        return MODIFIED_FOLLOWER_STOPPER_P30_V1
    if action_set == ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET:
        return REAR_AWARE_FEEDBACK_V1
    if action_set == ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET:
        return REAR_AWARE_FEEDBACK_V2
    return None


def _blend_contract_for_action_set(action_set: str):
    if action_set == ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET:
        return IDM_P30_PROJECTED_BLEND_V1
    return None


def _validate_feedback_option_steps(action_set: str, option_steps: int) -> None:
    """Reject a run whose duration disagrees with its embedded treatment contract."""
    contract = _feedback_contract_for_action_set(
        action_set
    ) or _blend_contract_for_action_set(action_set)
    frozen_steps = (
        None
        if contract is None
        else int(
            getattr(contract, "option_steps", None)
            or getattr(contract, "maximum_active_steps")
        )
    )
    if frozen_steps is not None and int(option_steps) != frozen_steps:
        raise ValueError(
            f"{action_set} freezes option_steps={frozen_steps}; "
            f"received {int(option_steps)}"
        )


def _runner_training_authorized(
    action_set: str, pilot_gates: Mapping[str, bool]
) -> bool:
    """Keep development-only mechanism probes from authorizing RL."""
    return (
        action_set
        not in {
            ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
            ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        }
        and all(bool(value) for value in pilot_gates.values())
    )


def _treatment_identity(action_set: str) -> dict[str, Any]:
    """Fields that make treatment-specific fingerprints and artifacts distinct."""
    identity: dict[str, Any] = {
        "action_set": action_set,
        "terminal_reference_percentile": terminal_reference_percentile(action_set),
    }
    release_contract = _projected_release_contract_for_action_set(action_set)
    if release_contract is not None:
        identity["release_contract"] = release_contract.to_dict()
    feedback_contract = _feedback_contract_for_action_set(action_set)
    if feedback_contract is not None:
        identity["feedback_contract"] = feedback_contract.to_dict()
    blend_contract = _blend_contract_for_action_set(action_set)
    if blend_contract is not None:
        identity["blend_contract"] = blend_contract.to_dict()
    return identity


def _matched_decision_source_purpose(
    *,
    action_set: str,
    discover_only: bool,
    rsu_observation_contract: str,
    response_trace_contract: str,
    diagnostic_spatial_history: bool,
) -> str | None:
    """Return the one approved purpose for a matched P30 decision source."""
    if (
        action_set == "activation_speed_ceiling_p50_v1"
        and not discover_only
        and rsu_observation_contract == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1
    ):
        return "p50_treatment_match"
    if (
        action_set == "activation_speed_ceiling_v1"
        and discover_only
        and rsu_observation_contract == RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1
    ):
        return "rsu_observation_replay"
    if (
        action_set == "activation_speed_ceiling_v1"
        and not discover_only
        and rsu_observation_contract == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1
        and response_trace_contract == PHASE_DIAGNOSTIC_SCHEMA
        and diagnostic_spatial_history
    ):
        return "p30_phase_spatial_replay"
    if (
        action_set == ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET
        and not discover_only
        and rsu_observation_contract == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1
        and response_trace_contract == PHASE_DIAGNOSTIC_SCHEMA
        and diagnostic_spatial_history
    ):
        return "p30_projected_blend_match"
    if (
        action_set == ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET
        and not discover_only
        and rsu_observation_contract == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1
        and response_trace_contract == PHASE_DIAGNOSTIC_SCHEMA
        and diagnostic_spatial_history
    ):
        return "p30_modified_followerstopper_match"
    if (
        action_set
        in {
            ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        }
        and not discover_only
        and rsu_observation_contract == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1
        and response_trace_contract == PHASE_DIAGNOSTIC_SCHEMA
        and diagnostic_spatial_history
    ):
        return "rear_aware_feedback_match"
    if (
        action_set
        in {
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
            ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
        }
        and not discover_only
        and rsu_observation_contract == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1
        and response_trace_contract == PHASE_DIAGNOSTIC_SCHEMA
        and diagnostic_spatial_history
    ):
        return "p30_projected_release_match"
    return None


def _matched_seed_scope_is_valid(
    *, source_seeds: Iterable[int], requested_seeds: Iterable[int], purpose: str
) -> bool:
    source = {int(value) for value in source_seeds}
    requested = {int(value) for value in requested_seeds}
    if purpose in {
        "p30_phase_spatial_replay",
        "p30_projected_release_match",
        "p30_modified_followerstopper_match",
        "p30_projected_blend_match",
        "rear_aware_feedback_match",
    }:
        return bool(requested) and requested <= source
    return requested == source


def _service_horizon_steps(safety_horizon_steps: int) -> int:
    """Retain the frozen 60-step service window or extend it for safety.

    Development release comparisons may request a longer safety horizon to
    keep post-release observation time aligned.  The simulation must actually
    run for that full interval; otherwise late outcomes are silently censored.
    """
    return max(60, int(safety_horizon_steps))


def _physical_state_fingerprint(decision: Mapping[str, Any]) -> str:
    """Hash traffic/geometry state without actor observation or treatment target."""
    payload = {
        "simulation_time_s": decision.get("simulation_time_s"),
        "lane_id": decision.get("lane_id"),
        "candidates": [
            {key: row.get(key) for key in _PHYSICAL_CANDIDATE_FIELDS}
            for row in decision.get("candidates", [])
        ],
        "pattern_context": decision.get("pattern_context") or {},
    }
    return _canonical_hash(payload)


def _raw_decision_context_fingerprint(decision: Mapping[str, Any]) -> str:
    """Treatment-agnostic pre-action traffic and candidate context."""
    payload = {
        "schema": "idm_raw_decision_context/1",
        "simulation_time_s": decision.get("simulation_time_s"),
        "lane_id": decision.get("lane_id"),
        "candidates": [
            {key: row.get(key) for key in _RAW_DECISION_CONTEXT_FIELDS}
            for row in decision.get("candidates", [])
        ],
        "pattern_context": decision.get("pattern_context") or {},
    }
    return _canonical_hash(payload)


def _treatment_fingerprint(decision: Mapping[str, Any]) -> str:
    """Hash the frozen treatment definition and all causal command previews."""
    payload = {
        "action_set": decision.get("action_set"),
        "terminal_reference_percentile": decision.get(
            "terminal_reference_percentile"
        ),
        "simulation_time_s": decision.get("simulation_time_s"),
        "lane_id": decision.get("lane_id"),
        "candidates": decision.get("candidates", []),
        "pattern_context": decision.get("pattern_context") or {},
    }
    if "release_contract" in decision:
        payload["release_contract"] = decision.get("release_contract")
    if "feedback_contract" in decision:
        payload["feedback_contract"] = decision.get("feedback_contract")
    if "blend_contract" in decision:
        payload["blend_contract"] = decision.get("blend_contract")
    return _canonical_hash(payload)


def _observation_fingerprint(decision: Mapping[str, Any]) -> str:
    payload = {
        "candidate_observations": decision.get("candidate_observations") or {},
        "candidate_observation_metadata": decision.get(
            "candidate_observation_metadata"
        )
        or {},
    }
    return _canonical_hash(payload)


def _nonclassic_observation_hash(observation: Any) -> str:
    values = np.asarray(observation, dtype=np.float32).reshape(-1)[:RAW_LOCAL_DIM]
    if values.size != RAW_LOCAL_DIM:
        raise ValueError(
            f"expected {RAW_LOCAL_DIM} physical observation fields, got {values.size}"
        )
    retained = np.concatenate(
        [values[: CLASSIC_RSU_SLICE.start], values[CLASSIC_RSU_SLICE.stop :]]
    )
    return hashlib.sha256(retained.tobytes()).hexdigest()


def _p50_shadow_command(
    command: Mapping[str, Any],
    disturbance: Mapping[str, Any],
    cfg: Any,
    reference_by_lane: dict[str, float],
) -> dict[str, Any] | None:
    """Project p50 on the observed p30 state without mutating SUMO or control."""
    if command.get("action_family") != "speed_ceiling":
        return None
    projection = command.get("projection_input") or {}
    required = {
        "current_speed_mps",
        "prior_acceleration_mps2",
        "allowed_speed_mps",
    }
    reference_diag = command.get("reference") or {}
    p50_raw = disturbance.get("speed_p50_mps")
    if p50_raw is None:
        p50_raw = reference_diag.get("shadow_p50_raw_reference_mps")
    if not required.issubset(projection) or p50_raw is None:
        return None
    lane_id = str(command["lane_id"])
    allowed = float(projection["allowed_speed_mps"])
    raw = float(
        np.clip(
            float(p50_raw),
            float(cfg.reference_floor_mps),
            allowed,
        )
    )
    previous = reference_by_lane.get(lane_id, raw)
    alpha = min(1.0, float(cfg.dt_s) / max(float(cfg.reference_filter_tau_s), float(cfg.dt_s)))
    filtered = previous + alpha * (raw - previous)
    rate = float(cfg.reference_rate_limit_mps_per_s) * float(cfg.dt_s)
    filtered = float(np.clip(filtered, previous - rate, previous + rate))
    reference_by_lane[lane_id] = filtered
    current_speed = float(projection["current_speed_mps"])
    nominal = analytical_acceleration(
        speed_mps=current_speed,
        reference_speed_mps=filtered,
        gap_m=projection.get("leader_gap_m"),
        min_gap_m=float(command.get("minimum_gap_m", 2.5)),
        leader_speed_mps=projection.get("leader_speed_mps"),
        cfg=cfg,
    )
    desired = min(
        nominal,
        (filtered - current_speed) / max(float(cfg.dt_s), 1e-6),
    )
    shadow_input = ProjectionInput(
        current_speed_mps=current_speed,
        desired_acceleration_mps2=desired,
        prior_acceleration_mps2=float(projection["prior_acceleration_mps2"]),
        allowed_speed_mps=allowed,
        leader_gap_m=projection.get("leader_gap_m"),
        leader_speed_mps=projection.get("leader_speed_mps"),
        follow_speed_mps=projection.get("follow_speed_mps"),
        follower_gap_m=projection.get("follower_gap_m"),
        follower_speed_mps=projection.get("follower_speed_mps"),
    )
    projected = project_command(shadow_input, cfg)
    return {
        "diagnostic_only": True,
        "raw_reference_speed_mps": raw,
        "filtered_reference_speed_mps": filtered,
        "nominal_acceleration_mps2": nominal,
        "desired_acceleration_mps2": desired,
        "projected_acceleration_mps2": projected.acceleration_mps2,
        "projected_target_speed_mps": projected.target_speed_mps,
        "projection_reasons": list(projected.reasons),
        "constraint_conflict": projected.constraint_conflict,
    }


def _env(
    demand: int,
    seed: int,
    output: Path,
    tag: str,
    max_steps: int,
    action_set: str,
    option_steps: int = 20,
    minimum_dose_steps: int = 16,
    cav_penetration: float = 0.10,
    rsu_observation_contract: str = RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
):
    cfg = _env_config(demand, seed, output / "artifacts", tag, max_steps, None)
    cfg.update(
        eval_mode=False,
        eval_fcd_output=None,
        eval_tripinfo_output=None,
        eval_stats_output=None,
        centralized_critic=False,
        action_masking=True,
        local_obs_dim=RAW_LOCAL_DIM + mode_count(action_set),
        idm_residual_decentralized_damping_enabled=True,
        idm_residual_decentralized_option_steps=option_steps,
        idm_residual_decentralized_minimum_predicted_dose_s=float(minimum_dose_steps),
        idm_residual_decentralized_require_secure_gap=True,
        idm_residual_decentralized_action_set=action_set,
        cav_penetration=cav_penetration,
        rsu_observation_contract=rsu_observation_contract,
        zone_comm_range=(
            RSU_DIAGNOSTIC_RANGE_M
            if rsu_observation_contract == RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1
            else 500.0
        ),
    )
    return TrafficShepherdMultiAgentEnv(cfg)


def _actions(
    obs: Mapping[str, np.ndarray], *, target_id: str | None = None, mode: int = 0
) -> dict[str, int]:
    return {
        str(vehicle_id): int(mode if str(vehicle_id) == target_id else 0)
        for vehicle_id in obs
    }


def _local(
    obs: Mapping[str, np.ndarray], local_dim: int
) -> dict[str, list[float]]:
    return {
        str(vehicle_id): np.asarray(values, dtype=np.float32).reshape(-1)[:local_dim].tolist()
        for vehicle_id, values in sorted(obs.items())
        if np.asarray(values).size >= local_dim
    }


def _candidate_snapshot(
    env: TrafficShepherdMultiAgentEnv,
    obs: Mapping[str, np.ndarray],
    candidate_ranking: str = "runway",
    candidate_count: int = 2,
    forced_candidate_ids: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    samples = env._idm_residual_engine._samples()
    disturbances = detect_disturbances(samples, env._idm_residual_engine.cfg)
    candidates, admissions = env._idm_residual_engine._decentralized_candidate_state(
        samples, env._active_cavs, disturbances
    )
    by_id = {sample.vehicle_id: sample for sample in samples}

    def prospective_rear(row: Any) -> tuple[float, float | None]:
        context = env._idm_residual_engine._leader_follower(row.vehicle_id)
        rear_gap = context.get("follower_gap_m")
        follower_id = context.get("follower_id")
        follower_speed = context.get("follower_speed_mps")
        if rear_gap is None or follower_id is None or follower_speed is None:
            return 250.0, None
        target_speed = max(
            env._idm_residual_engine.cfg.reference_floor_mps,
            _terminal_speed_for_action_set(
                str(env._idm_residual_engine.cfg.decentralized_action_set),
                disturbances[row.lane_id],
            ),
        )
        try:
            ego_decel = float(traci.vehicle.getDecel(row.vehicle_id))
            secure = float(
                traci.vehicle.getSecureGap(
                    follower_id,
                    float(follower_speed),
                    target_speed,
                    ego_decel,
                    row.vehicle_id,
                )
            )
        except Exception:
            return -math.inf, None
        return float(rear_gap) - secure, secure

    if forced_candidate_ids is not None:
        candidate_by_id = {
            row.vehicle_id: row for rows in candidates.values() for row in rows
        }
        if len(forced_candidate_ids) != candidate_count:
            raise ValueError("forced candidate count differs from frozen candidate count")
        if any(
            vehicle_id not in candidate_by_id
            or not admissions[vehicle_id].get("admissible", False)
            for vehicle_id in forced_candidate_ids
        ):
            return None
        selected = tuple(candidate_by_id[vehicle_id] for vehicle_id in forced_candidate_ids)
        lane_ids = {row.lane_id for row in selected}
        if len(lane_ids) != 1:
            return None
        lane_id = selected[0].lane_id
    else:
        choices: list[tuple[float, str, tuple[Any, ...]]] = []
        for lane_id, rows in candidates.items():
            admitted_rows = tuple(
                row for row in rows if admissions[row.vehicle_id].get("admissible", False)
            )
            if candidate_ranking in {"rear_headroom", "prospective_rear_headroom"}:
                def headroom(row: Any) -> float:
                    if candidate_ranking == "prospective_rear_headroom":
                        return prospective_rear(row)[0]
                    admission = admissions[row.vehicle_id]
                    rear_gap = admission.get("rear_gap_m")
                    rear_secure = admission.get("rear_secure_gap_m")
                    if rear_gap is None:
                        return 250.0
                    return float(rear_gap) - float(rear_secure or 0.0)

                admitted = tuple(
                    sorted(
                        admitted_rows,
                        key=lambda row: (-headroom(row), row.runway_m, row.vehicle_id),
                    )
                )
                lane_priority = -headroom(admitted[0]) if admitted else math.inf
            else:
                admitted = admitted_rows
                lane_priority = admitted[0].runway_m if admitted else math.inf
            if len(admitted) >= candidate_count:
                choices.append((lane_priority, lane_id, admitted))
        if not choices:
            return None
        _, lane_id, admitted = min(choices, key=lambda item: (item[0], item[1]))
        selected = admitted[:candidate_count]
    local = _local(obs, int(env._local_obs_dim))
    if any(row.vehicle_id not in local for row in selected):
        return None
    candidate_rows = []
    for slot, row in enumerate(selected, start=1):
        admission = admissions[row.vehicle_id]
        sample = by_id[row.vehicle_id]
        context = env._idm_residual_engine._leader_follower(row.vehicle_id)
        disturbance = disturbances[row.lane_id]
        try:
            allowed_speed = float(traci.vehicle.getAllowedSpeed(row.vehicle_id))
            minimum_gap = float(traci.vehicle.getMinGap(row.vehicle_id))
        except Exception:
            allowed_speed, minimum_gap = 35.0, 2.5
        raw_reference = float(
            np.clip(
                _terminal_speed_for_action_set(
                    str(env._idm_residual_engine.cfg.decentralized_action_set),
                    disturbance,
                ),
                env._idm_residual_engine.cfg.reference_floor_mps,
                allowed_speed,
            )
        )
        nominal_acceleration = analytical_acceleration(
            speed_mps=sample.speed_mps,
            reference_speed_mps=raw_reference,
            gap_m=context["leader_gap_m"],
            min_gap_m=minimum_gap,
            leader_speed_mps=context["leader_speed_mps"],
            cfg=env._idm_residual_engine.cfg,
        )
        follow_speed = None
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
            except Exception:
                pass
        terminal_speed_previews: dict[str, dict[str, Any]] = {}
        for percentile, speed in (
            ("p30", disturbance.speed_p30_mps),
            ("p50", disturbance.speed_p50_mps),
            ("p70", disturbance.speed_p70_mps),
        ):
            reference = float(
                np.clip(
                    speed,
                    env._idm_residual_engine.cfg.reference_floor_mps,
                    allowed_speed,
                )
            )
            analytical = analytical_acceleration(
                speed_mps=sample.speed_mps,
                reference_speed_mps=reference,
                gap_m=context["leader_gap_m"],
                min_gap_m=minimum_gap,
                leader_speed_mps=context["leader_speed_mps"],
                cfg=env._idm_residual_engine.cfg,
            )
            desired = min(
                analytical,
                (reference - sample.speed_mps)
                / max(env._idm_residual_engine.cfg.dt_s, 1e-6),
            )
            projected = project_command(
                ProjectionInput(
                    current_speed_mps=sample.speed_mps,
                    desired_acceleration_mps2=desired,
                    prior_acceleration_mps2=env._idm_residual_engine.prior_acceleration.get(
                        sample.vehicle_id, sample.acceleration_mps2
                    ),
                    allowed_speed_mps=allowed_speed,
                    leader_gap_m=context["leader_gap_m"],
                    leader_speed_mps=context["leader_speed_mps"],
                    follow_speed_mps=follow_speed,
                    follower_gap_m=context["follower_gap_m"],
                    follower_speed_mps=context["follower_speed_mps"],
                ),
                env._idm_residual_engine.cfg,
            )
            terminal_speed_previews[percentile] = {
                "reference_speed_mps": reference,
                "analytical_acceleration_mps2": analytical,
                "desired_acceleration_mps2": desired,
                "projected_acceleration_mps2": projected.acceleration_mps2,
                "projected_target_speed_mps": projected.target_speed_mps,
                "projection_reasons": list(projected.reasons),
                "constraint_conflict": projected.constraint_conflict,
            }
        prospective_margin, prospective_secure = prospective_rear(row)
        lane_change_state = {}
        for direction, name in ((-1, "right"), (1, "left")):
            try:
                original, influenced = traci.vehicle.getLaneChangeState(
                    row.vehicle_id, direction
                )
                lane_change_state[name] = {
                    "original": int(original),
                    "influenced": int(influenced),
                    "pretty": list(
                        traci.vehicle.getLaneChangeStatePretty(row.vehicle_id, direction)
                    ),
                    "could_change": bool(
                        traci.vehicle.couldChangeLane(row.vehicle_id, direction)
                    ),
                    "wants_and_could_change": bool(
                        traci.vehicle.wantsAndCouldChangeLane(row.vehicle_id, direction)
                    ),
                }
            except Exception as exc:
                lane_change_state[name] = {"error": type(exc).__name__}
        candidate_rows.append(
            {
                "slot": slot,
                "vehicle_id": row.vehicle_id,
                "lane_id": row.lane_id,
                "geometric_rank": row.rank,
                "distance_to_merge_m": row.distance_to_merge_m,
                "disturbance_distance_to_merge_m": row.disturbance_distance_to_merge_m,
                "runway_m": row.runway_m,
                "predicted_dose_s": admission.get("predicted_dose_s"),
                "front_ttc_s": admission.get("front_ttc_s"),
                "rear_ttc_s": admission.get("rear_ttc_s"),
                "front_gap_m": admission.get("front_gap_m"),
                "rear_gap_m": admission.get("rear_gap_m"),
                "front_secure_gap_m": admission.get("front_secure_gap_m"),
                "rear_secure_gap_m": admission.get("rear_secure_gap_m"),
                "preview_reference_speed_mps": raw_reference,
                "preview_nominal_acceleration_mps2": nominal_acceleration,
                "terminal_speed_previews": terminal_speed_previews,
                "prospective_rear_secure_gap_m": prospective_secure,
                "prospective_rear_headroom_m": prospective_margin,
                "lane_change_state": lane_change_state,
            }
        )
    pattern_context = {
        key: float(value)
        for key, value in env._idm_pattern_context.items()
    }
    observation_timestamp_s = float(traci.simulation.getTime())
    candidate_observation_metadata = {
        row.vehicle_id: env._obs_assembler.rsu_metadata(
            row.vehicle_id,
            observation_timestamp_s=observation_timestamp_s,
            merge_edge=env._merge_edge,
            pre_edge=env._pre_edge,
        )
        for row in selected
    }
    fingerprint_payload = {
        **_treatment_identity(
            str(env._idm_residual_engine.cfg.decentralized_action_set)
        ),
        "simulation_time_s": observation_timestamp_s,
        "lane_id": lane_id,
        "candidates": candidate_rows,
        "candidate_observations": {
            row.vehicle_id: local[row.vehicle_id] for row in selected
        },
        "pattern_context": pattern_context,
    }
    result = {
        **fingerprint_payload,
        "candidate_observation_metadata": candidate_observation_metadata,
        "fingerprint": _canonical_hash(fingerprint_payload),
    }
    result["physical_state_fingerprint"] = _physical_state_fingerprint(result)
    result["treatment_fingerprint"] = _treatment_fingerprint(result)
    result["observation_fingerprint"] = _observation_fingerprint(result)
    return result


def _landmark_at_state(
    state: Mapping[str, Any], landmark: str, contract: LandmarkContract
) -> str | None:
    """Return the frozen causal landmark represented by a candidate state."""
    if landmark == "two_candidate":
        return "two_candidate"
    context = state["pattern_context"]
    choices = contract.priority if landmark == "risk_onset" else (landmark,)
    return next(
        (name for name in choices if landmark_signal(name, context, contract)),
        None,
    )


def _has_command_opportunity(
    state: Mapping[str, Any], maximum_nominal_acceleration_mps2: float | None
) -> bool:
    if maximum_nominal_acceleration_mps2 is None:
        return True
    return any(
        float(row["preview_nominal_acceleration_mps2"])
        <= maximum_nominal_acceleration_mps2
        for row in state["candidates"]
    )


def discover(spec: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    demand, seed = int(spec["demand"]), int(spec["seed"])
    frozen_source = spec.get("frozen_decision_source")
    if frozen_source is not None and bool(frozen_source.get("decision_missing")):
        return {
            "schema": SCHEMA,
            "protocol_hash": spec["protocol_hash"],
            "demand": demand,
            "seed": seed,
            "decision": None,
            "steps_completed": 0,
            "runtime_s": time.perf_counter() - started,
            "matched_source_decision_missing": True,
        }
    env = _env(
        demand,
        seed,
        Path(spec["output"]),
        f"discover.m{demand}.seed{seed}",
        int(spec["max_predecision_steps"]) + 5,
        str(spec["action_set"]),
        int(spec["option_steps"]),
        int(spec["minimum_dose_steps"]),
        float(spec["cav_penetration"]),
        str(
            spec.get(
                "rsu_observation_contract",
                RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            )
        ),
    )
    decision = None
    completed = 0
    contract = LandmarkContract(
        maximum_landmarks_per_cell=int(spec.get("maximum_landmarks_per_cell", 2))
    )
    persistence = 0
    persistent_key = None
    landmark_occurrences = 0
    previous_landmark_step: int | None = None
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(int(spec["max_predecision_steps"]) + 1):
            state = _candidate_snapshot(
                env,
                obs,
                str(spec.get("candidate_ranking", "runway")),
                int(spec.get("candidate_count", 2)),
                (
                    (str(frozen_source["target_vehicle_id"]),)
                    if frozen_source is not None
                    else None
                ),
            )
            if frozen_source is not None and step == int(frozen_source["decision_step"]):
                if state is None:
                    raise RuntimeError(
                        f"matched decision source has no p50-admissible state at step {step}"
                    )
                expected_target = str(frozen_source["target_vehicle_id"])
                actual_targets = [str(row["vehicle_id"]) for row in state["candidates"]]
                if not actual_targets or actual_targets[0] != expected_target:
                    raise RuntimeError(
                        "matched decision target drifted: "
                        f"expected {expected_target!r}, received {actual_targets}"
                    )
                matched_identity = {}
                source_purpose = str(spec.get("decision_source_purpose"))
                if source_purpose in {
                    "rsu_observation_replay",
                    "p30_phase_spatial_replay",
                    "p30_projected_release_match",
                    "p30_modified_followerstopper_match",
                    "p30_projected_blend_match",
                    "rear_aware_feedback_match",
                }:
                    source_physical = str(frozen_source["physical_state_fingerprint"])
                    source_treatment = str(frozen_source["treatment_fingerprint"])
                    source_raw_context = str(
                        frozen_source["raw_decision_context_fingerprint"]
                    )
                    source_nonclassic = str(
                        frozen_source["nonclassic_observation_hash"]
                    )
                    actual_observation = state["candidate_observations"][expected_target]
                    actual_nonclassic = _nonclassic_observation_hash(actual_observation)
                    matched_identity = {
                        "source_physical_state_fingerprint": source_physical,
                        "physical_state_identity": (
                            state["physical_state_fingerprint"] == source_physical
                        ),
                        "source_raw_decision_context_fingerprint": source_raw_context,
                        "raw_decision_context_identity": (
                            _raw_decision_context_fingerprint(state)
                            == source_raw_context
                        ),
                        "source_treatment_fingerprint": source_treatment,
                        "treatment_identity": (
                            state["treatment_fingerprint"] == source_treatment
                        ),
                        "source_nonclassic_observation_hash": source_nonclassic,
                        "nonclassic_observation_identity": (
                            actual_nonclassic == source_nonclassic
                        ),
                    }
                    if source_purpose == "p30_phase_spatial_replay":
                        matched_identity.update(
                            {
                                "source_decision_fingerprint": str(
                                    frozen_source["decision_fingerprint"]
                                ),
                                "decision_fingerprint_identity": (
                                    state["fingerprint"]
                                    == str(frozen_source["decision_fingerprint"])
                                ),
                            }
                        )
                    required_identity_keys = [
                        "nonclassic_observation_identity",
                    ]
                    if source_purpose in {
                        "p30_projected_release_match",
                        "p30_modified_followerstopper_match",
                        "p30_projected_blend_match",
                        "rear_aware_feedback_match",
                    }:
                        required_identity_keys.append(
                            "raw_decision_context_identity"
                        )
                        matched_identity["treatment_difference_expected"] = (
                            not matched_identity["treatment_identity"]
                            and state.get("terminal_reference_percentile")
                            == terminal_reference_percentile(str(spec["action_set"]))
                        )
                        required_identity_keys.append("treatment_difference_expected")
                    else:
                        required_identity_keys.extend(
                            ["physical_state_identity", "treatment_identity"]
                        )
                    if source_purpose == "p30_phase_spatial_replay":
                        required_identity_keys.append("decision_fingerprint_identity")
                    if not all(
                        matched_identity[key]
                        for key in required_identity_keys
                    ):
                        raise RuntimeError(
                            "matched P30 replay changed a frozen decision component at "
                            f"step {step}: {matched_identity}"
                        )
                decision = {
                    "decision_step": step,
                    "decision_index": int(frozen_source["decision_index"]),
                    "landmark": str(frozen_source["landmark"]),
                    "landmark_persistence_steps": int(
                        frozen_source["landmark_persistence_steps"]
                    ),
                    "matched_decision_source": {
                        "protocol_hash": str(frozen_source["protocol_hash"]),
                        "decision_fingerprint": str(frozen_source["decision_fingerprint"]),
                        "target_vehicle_id": expected_target,
                        "purpose": str(spec.get("decision_source_purpose")),
                        **matched_identity,
                    },
                    **state,
                }
                break
            if frozen_source is not None:
                if step >= int(spec["max_predecision_steps"]):
                    break
                obs, _, terminated, truncated, _ = env.step(_actions(obs))
                completed = step + 1
                if terminated.get("__all__") or truncated.get("__all__"):
                    break
                continue
            landmark = (
                None
                if state is None
                else _landmark_at_state(state, str(spec["landmark"]), contract)
            )
            current_key = (
                landmark,
                None if state is None else state["lane_id"],
                tuple(
                    row["vehicle_id"] for row in state["candidates"]
                ) if state is not None else (),
            )
            if (
                state is not None
                and landmark is not None
                and _has_command_opportunity(
                    state, spec.get("maximum_nominal_acceleration_mps2")
                )
                and step >= int(spec.get("minimum_decision_step", 0))
            ):
                persistence = persistence + 1 if current_key == persistent_key else 1
                persistent_key = current_key
                required = (
                    1
                    if str(spec["landmark"]) == "two_candidate"
                    else contract.persistence_steps
                )
                if persistence >= required:
                    separation = contract.minimum_landmark_separation_steps
                    if (
                        previous_landmark_step is None
                        or step - previous_landmark_step >= separation
                    ):
                        landmark_occurrences += 1
                        previous_landmark_step = step
                        if landmark_occurrences >= int(spec.get("decision_index", 1)):
                            decision = {
                                "decision_step": step,
                                "decision_index": landmark_occurrences,
                                "landmark": landmark,
                                "landmark_persistence_steps": persistence,
                                **state,
                            }
                            break
                    persistence = 0
                    persistent_key = None
            else:
                persistence = 0
                persistent_key = None
            if step >= int(spec["max_predecision_steps"]):
                break
            obs, _, terminated, truncated, _ = env.step(_actions(obs))
            completed = step + 1
            if terminated.get("__all__") or truncated.get("__all__"):
                break
    finally:
        env.close()
    return {
        "schema": SCHEMA,
        "protocol_hash": spec["protocol_hash"],
        "demand": demand,
        "seed": seed,
        "decision": decision,
        "steps_completed": completed,
        "runtime_s": time.perf_counter() - started,
    }


def _spatial_totals() -> dict[str, dict[str, float]]:
    return {
        level: {
            "acceleration_energy_m2ps4": 0.0,
            "severe_braking_count": 0.0,
            "integrated_negative_acceleration_mps": 0.0,
            "low_ttc_count": 0.0,
        }
        for level in ("immediate_follower", "influence_region", "lane", "corridor")
    }


def run_branch(spec: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    decision = spec["decision"]
    target = None
    if not spec["is_control"]:
        target = decision["candidates"][int(spec["candidate_slot"]) - 1]["vehicle_id"]
    env = _env(
        int(spec["demand"]),
        int(spec["seed"]),
        Path(spec["output"]),
        spec["branch_id"],
        int(decision["decision_step"]) + int(spec["service_horizon_steps"]) + 5,
        str(spec["action_set"]),
        int(spec["option_steps"]),
        int(spec["minimum_dose_steps"]),
        float(spec["cav_penetration"]),
        str(
            spec.get(
                "rsu_observation_contract",
                RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            )
        ),
    )
    severe_total = wave_total = ttc_total = 0.0
    low_ttc_steps = arrivals = discharge = collisions = teleports = 0
    delay_rates: list[float] = []
    collapse_run = collapse_max = completed = 0
    target_command_steps: set[int] = set()
    command_rows: list[dict[str, Any]] = []
    release_rows: list[dict[str, Any]] = []
    feedback_release_rows: list[dict[str, Any]] = []
    treatment_safety_abort_rows: list[dict[str, Any]] = []
    member_change_rows: list[dict[str, Any]] = []
    target_admission_rows: list[dict[str, Any]] = []
    target_reward_sum = 0.0
    role_reward_sum = 0.0
    role_reward_safety_horizon_sum = 0.0
    all_agent_reward_sum = 0.0
    role_reward_component_sums: dict[str, float] = {}
    spatial = _spatial_totals()
    unexpected_command_count = 0
    control_command_count = 0
    branch_fingerprint = None
    p50_shadow_reference_by_lane: dict[str, float] = {}
    response_trace_contract = str(spec.get("response_trace_contract", "none"))
    projected_release_contract = _projected_release_contract_for_action_set(
        str(spec["action_set"])
    )
    feedback_contract = _feedback_contract_for_action_set(str(spec["action_set"]))
    blend_contract = _blend_contract_for_action_set(str(spec["action_set"]))
    records_release_rows = (
        projected_release_contract is not None
        or feedback_contract is not None
        or blend_contract is not None
    )
    phase_tracker = (
        TreatmentPhaseTracker(
            probe_steps=int(getattr(feedback_contract, "probe_steps", 0)),
            release_steps=(
                projected_release_contract.nominal_budget_steps
                if projected_release_contract is not None
                else 10
                if feedback_contract is not None
                else 5
            ),
        )
        if response_trace_contract == PHASE_DIAGNOSTIC_SCHEMA
        else None
    )
    phase_response_steps: list[dict[str, Any]] = []
    spatial_history = (
        SpatialRSUHistory(SpatialHistoryContract())
        if bool(spec.get("diagnostic_spatial_history", False))
        else None
    )
    predecision_spatial_history: list[dict[str, Any]] = []
    postdecision_spatial_history: list[dict[str, Any]] = []
    try:
        obs, _ = env.reset(seed=int(spec["seed"]))
        for _ in range(int(decision["decision_step"])):
            obs, _, terminated, truncated, _ = env.step(_actions(obs))
            if spatial_history is not None:
                frame = capture_corridor_frame(env, spatial_history.contract)
                spatial_history.observe(frame)
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        if spatial_history is not None:
            predecision_spatial_history = spatial_history.window(
                spatial_history.contract.retained_history_s
            )
        state = _candidate_snapshot(
            env,
            obs,
            str(spec.get("candidate_ranking", "runway")),
            int(spec.get("candidate_count", 2)),
            tuple(str(row["vehicle_id"]) for row in decision["candidates"]),
        )
        branch_fingerprint = None if state is None else state["fingerprint"]
        snapshot = get_active_snapshot()
        previous_road = (
            {
                vehicle_id: str(road)
                for vehicle_id in snapshot.ordered_ids()
                for road in [snapshot.road(vehicle_id)]
                if road is not None
            }
            if snapshot is not None
            else {}
        )
        role_id = target or decision["candidates"][0]["vehicle_id"]
        for relative_step in range(1, int(spec["service_horizon_steps"]) + 1):
            fire = relative_step == 1 and not spec["is_control"]
            actions = _actions(
                obs,
                target_id=target if fire else None,
                mode=int(spec["mode_index"]) if fire else 0,
            )
            previous_acceleration = _snapshot_acceleration()
            obs, rewards, terminated, truncated, info = env.step(actions)
            all_agent_reward_sum += sum(float(value) for value in rewards.values())
            role_reward_sum += float(rewards.get(role_id, 0.0))
            if relative_step <= int(spec["safety_horizon_steps"]):
                role_reward_safety_horizon_sum += float(rewards.get(role_id, 0.0))
            if target is not None:
                target_reward_sum += float(rewards.get(target, 0.0))
            for key, value in (
                info.get("idm_reward_components", {}).get(role_id, {}).items()
            ):
                if key in {"wave_risk", "executed_residual_mps2", "total"}:
                    continue
                role_reward_component_sums[key] = (
                    role_reward_component_sums.get(key, 0.0) + float(value)
                )
            commands = list(info.get("idm_residual", {}).get("commands", []))
            idm_diag = info.get("idm_residual", {})
            raw_release_rows = [
                dict(row) for row in (idm_diag.get("release") or [])
            ]
            step_release_rows = [
                {"relative_step": relative_step, **row} for row in raw_release_rows
            ]
            if records_release_rows:
                release_rows.extend(step_release_rows)
            member_changes = list(idm_diag.get("member_changes", []))
            treatment_safety_abort_rows.extend(
                {"relative_step": relative_step, **dict(row)}
                for row in idm_diag.get("treatment_safety_aborts", [])
            )
            feedback_release_rows.extend(
                {"relative_step": relative_step, **dict(row)}
                for row in idm_diag.get("feedback_releases", [])
            )
            for change in member_changes:
                member_change_rows.append(
                    {"relative_step": relative_step, **dict(change)}
                )
            if target is not None:
                target_admission_rows.extend(
                    {
                        "relative_step": relative_step,
                        **dict(row),
                    }
                    for row in idm_diag.get("candidate_admissions", [])
                    if str(row.get("vehicle_id")) == target
                )
            if spec["is_control"]:
                control_command_count += len(commands)
            for command in commands:
                vehicle_id = str(command["vehicle_id"])
                if target is not None and vehicle_id == target:
                    target_command_steps.add(relative_step)
                elif not spec["is_control"]:
                    unexpected_command_count += 1
                projected = command.get("projected") or {}
                disturbance = (idm_diag.get("disturbances") or {}).get(
                    str(command.get("lane_id")), {}
                )
                p50_shadow = _p50_shadow_command(
                    command,
                    disturbance,
                    env._idm_residual_engine.cfg,
                    p50_shadow_reference_by_lane,
                )
                command_rows.append(
                    {
                        "relative_step": relative_step,
                        "vehicle_id": vehicle_id,
                        "mode_index": command.get("mode_index"),
                        "action_family": command.get("action_family"),
                        "lane_id": command.get("lane_id"),
                        "current_speed_mps": command.get("current_speed_mps"),
                        "nominal_acceleration_mps2": command.get(
                            "nominal_acceleration_mps2"
                        ),
                        "desired_acceleration_mps2": command.get(
                            "desired_acceleration_mps2"
                        ),
                        "projected_acceleration_mps2": projected.get(
                            "acceleration_mps2"
                        ),
                        "projection_reasons": list(projected.get("reasons") or []),
                        "reference": command.get("reference"),
                        "projection_input": command.get("projection_input"),
                        "p50_shadow": p50_shadow,
                        "modified_follower_stopper": command.get(
                            "modified_follower_stopper"
                        ),
                        "projected_blend": command.get("projected_blend"),
                        "rear_aware_feedback": command.get("rear_aware_feedback"),
                        "treatment_infeasible": bool(
                            command.get("treatment_infeasible", False)
                        ),
                        "treatment_abort_reason": command.get(
                            "treatment_abort_reason"
                        ),
                        "rear_bound_active_in_projection_conflict": bool(
                            command.get(
                                "rear_bound_active_in_projection_conflict",
                                False,
                            )
                        ),
                    }
                )
            trace = capture_spatial_response(
                env,
                relative_step=relative_step,
                role_ids={str(role_id)},
                frozen_lane_id=str(decision["lane_id"]),
                previous_acceleration=previous_acceleration,
                commands=commands,
            )
            step_discharge = _merge_discharge_step(env, previous_road)
            if phase_tracker is not None:
                phase = phase_tracker.observe(
                    relative_step=relative_step,
                    active_vehicle_ids=(
                        str(command["vehicle_id"]) for command in commands
                    ),
                    member_changes=member_changes,
                )
                phase_step = phase_step_diagnostic(
                    phase,
                    trace,
                    merge_discharge_this_step=step_discharge,
                )
                phase_step["member_changes"] = [dict(row) for row in member_changes]
                phase_step["release_rows"] = (
                    [dict(row) for row in step_release_rows]
                    if projected_release_contract is not None
                    else raw_release_rows
                )
                phase_step["commanded_vehicle_ids"] = [
                    str(command["vehicle_id"]) for command in commands
                ]
                phase_response_steps.append(phase_step)
            if spatial_history is not None:
                frame = capture_corridor_frame(env, spatial_history.contract)
                enriched = spatial_history.observe(frame)
                if enriched is not None:
                    postdecision_spatial_history.append(enriched)
            for level, metrics in spatial.items():
                values = trace.get(level, {})
                for metric in metrics:
                    value = values.get(metric)
                    if value is not None and math.isfinite(float(value)):
                        metrics[metric] += float(value)
            if relative_step <= int(spec["safety_horizon_steps"]):
                severe_total += float(env._corridor_severe_braking_count() or 0.0)
                wave_total += float(
                    env._corridor_acceleration_energy_sum_m2ps4() or 0.0
                )
                current_ttc = int(env._corridor_ttc_count() or 0)
                ttc_total += current_ttc
                low_ttc_steps += int(current_ttc > 0)
            delay = env._corridor_delay_rate_live()
            if delay is not None and math.isfinite(float(delay)):
                delay_rates.append(float(delay))
            arrivals += int(traci.simulation.getArrivedNumber())
            discharge += step_discharge
            collisions += int(traci.simulation.getCollidingVehiclesNumber())
            teleports += int(traci.simulation.getStartingTeleportNumber())
            collapse_step = bool(trace.get("merge_service", {}).get("collapse_condition"))
            collapse_run = collapse_run + 1 if collapse_step else 0
            collapse_max = max(collapse_max, collapse_run)
            completed += 1
            if terminated.get("__all__") or truncated.get("__all__"):
                break
    finally:
        env.close()
    projection_survival = []
    for row in command_rows:
        rear_aware = row.get("rear_aware_feedback")
        if rear_aware:
            native = rear_aware["projected_native_endpoint"]["acceleration_mps2"]
            feedback = rear_aware["projected_feedback_endpoint"]["acceleration_mps2"]
            requested_delta = float(rear_aware["strength"]) * (
                float(feedback) - float(native)
            )
            realized_delta = (
                float(rear_aware["final_projected_acceleration_mps2"])
                - float(native)
            )
            if abs(requested_delta) > 1e-9:
                projection_survival.append(
                    min(1.0, abs(realized_delta) / abs(requested_delta))
                )
            continue
        nominal = row["nominal_acceleration_mps2"]
        desired = row["desired_acceleration_mps2"]
        projected = row["projected_acceleration_mps2"]
        if None in (nominal, desired, projected):
            continue
        requested_delta = float(desired) - float(nominal)
        if abs(requested_delta) > 1e-9:
            projection_survival.append(
                min(1.0, abs(float(projected) - float(nominal)) / abs(requested_delta))
            )
    branch = {
        "schema": SCHEMA,
        "protocol_hash": spec["protocol_hash"],
        "case_id": spec["case_id"],
        "branch_id": spec["branch_id"],
        "arm": spec["arm"],
        "is_control": bool(spec["is_control"]),
        "candidate_slot": spec["candidate_slot"],
        "mode_index": int(spec["mode_index"]),
        "mode": mode_by_index(str(spec["action_set"]))[int(spec["mode_index"])].name,
        **_treatment_identity(str(spec["action_set"])),
        "demand": int(spec["demand"]),
        "seed": int(spec["seed"]),
        "decision_step": int(decision["decision_step"]),
        "target_vehicle_id": target,
        "decision_fingerprint": branch_fingerprint,
        "discovery_fingerprint": decision["fingerprint"],
        "decision_identity_with_discovery": branch_fingerprint == decision["fingerprint"],
        "completed_horizon_steps": completed,
        "target_command_step_count": len(target_command_steps),
        "unexpected_command_count": unexpected_command_count,
        "control_command_count": control_command_count,
        "command_rows": command_rows,
        "member_change_rows": member_change_rows,
        "treatment_safety_abort_rows": treatment_safety_abort_rows,
        "feedback_release_rows": feedback_release_rows,
        "target_admission_rows": target_admission_rows,
        "target_reward_sum": target_reward_sum,
        "role_reward_sum": role_reward_sum,
        "role_reward_safety_horizon_sum": role_reward_safety_horizon_sum,
        "all_agent_reward_sum": all_agent_reward_sum,
        "role_reward_component_sums": role_reward_component_sums,
        "projection_survival_values": projection_survival,
        "spatial_totals": spatial,
        "severe_braking_count_sum": severe_total,
        "acceleration_energy_sum": wave_total,
        "ttc_count_sum": ttc_total,
        "low_ttc_exposure_steps": low_ttc_steps,
        "mean_corridor_delay_rate": float(np.mean(delay_rates)) if delay_rates else None,
        "arrived_vehicles": arrivals,
        "merge_discharge_count": discharge,
        "collision_observations": collisions,
        "teleport_observations": teleports,
        "collapse_max_consecutive_steps": collapse_max,
        "runtime_s": time.perf_counter() - started,
    }
    if records_release_rows:
        branch["release_rows"] = release_rows
    if (
        feedback_contract is not None
        and hasattr(feedback_contract, "probe_steps")
    ):
        realized_steps = len(target_command_steps)
        probe_steps = int(feedback_contract.probe_steps)
        maximum_steps = int(feedback_contract.maximum_active_steps)
        target_feedback_rows = [
            row["rear_aware_feedback"]
            for row in command_rows
            if row["vehicle_id"] == target and row.get("rear_aware_feedback")
        ]
        branch["treatment_dose"] = {
            "requested_probe_steps": probe_steps,
            "requested_maximum_active_steps": maximum_steps,
            "realized_active_steps": realized_steps,
            "realized_fraction_of_probe": min(1.0, realized_steps / probe_steps),
            "realized_fraction_of_maximum": min(1.0, realized_steps / maximum_steps),
            "conditional_hold_authorized": any(
                bool(row.get("hold_authorized", False))
                for row in target_feedback_rows
            ),
            "release_reasons": [
                str(row["reason"])
                for row in feedback_release_rows
                if row.get("vehicle_id") == target
            ],
        }
    if phase_tracker is not None:
        branch["phase_response"] = {
            "schema": PHASE_DIAGNOSTIC_SCHEMA,
            "phase_definition": {
                "probe_supported": bool(
                    feedback_contract is not None
                    and hasattr(feedback_contract, "probe_steps")
                ),
                "probe_window_steps": int(
                    getattr(feedback_contract, "probe_steps", 0)
                ),
                "active_treatment_phase": (
                    "probe_then_conditional_hold"
                    if feedback_contract is not None
                    and hasattr(feedback_contract, "probe_steps")
                    else "hold"
                ),
                "release_window_steps": phase_tracker.release_steps,
                "note": (
                    "The rear-aware action uses the frozen causal probe/hold/release "
                    "contract. Other fixed latched actions have no probe."
                    if feedback_contract is not None
                    and hasattr(feedback_contract, "probe_steps")
                    else "The fixed latched option has no probe; diagnostics do not "
                    "alter treatment timing."
                ),
            },
            "steps": phase_response_steps,
        }
    if spatial_history is not None:
        branch["diagnostic_spatial_history"] = {
            "schema": SPATIAL_HISTORY_SCHEMA,
            "contract": spatial_history.contract.to_dict(),
            "predecision_window": predecision_spatial_history,
            "postdecision_frames": postdecision_spatial_history,
        }
    return branch


def _arms(
    action_set: str = "gain_reference_v1", candidate_slots: tuple[int, ...] = (1, 2)
) -> list[dict[str, Any]]:
    arms = [
        {"arm": "abstain", "is_control": True, "candidate_slot": None, "mode_index": 0},
        {
            "arm": "abstain_replay",
            "is_control": True,
            "candidate_slot": None,
            "mode_index": 0,
        },
    ]
    for slot in candidate_slots:
        for definition in modes_for(action_set):
            if definition.index == 0:
                continue
            mode = definition.index
            arms.append(
                {
                    "arm": f"candidate{slot}_{definition.name}",
                    "is_control": False,
                    "candidate_slot": slot,
                    "mode_index": mode,
                }
            )
    return arms


def _effect(
    control: Mapping[str, Any],
    action: Mapping[str, Any],
    refs: Mapping[str, float],
    deadband: float,
    minimum_dose_steps: int = 16,
    safety_horizon_steps: int = 40,
    service_horizon_steps: int = 60,
) -> dict[str, Any]:
    physical = safety_composite(
        zero_severe=control["severe_braking_count_sum"],
        action_severe=action["severe_braking_count_sum"],
        severe_reference=refs["severe"],
        zero_wave=control["acceleration_energy_sum"],
        action_wave=action["acceleration_energy_sum"],
        wave_reference=refs["wave"],
        horizon_steps=safety_horizon_steps,
    )
    delay_delta = action["mean_corridor_delay_rate"] - control["mean_corridor_delay_rate"]
    arrival_delta = action["arrived_vehicles"] - control["arrived_vehicles"]
    discharge_delta = action["merge_discharge_count"] - control["merge_discharge_count"]
    constraints = {
        "complete_horizon": (
            action["completed_horizon_steps"] == int(service_horizon_steps)
        ),
        "dose_at_least_minimum": (
            action["target_command_step_count"] >= minimum_dose_steps
        ),
        "zero_unexpected_commands": action["unexpected_command_count"] == 0,
        "zero_added_collisions": action["collision_observations"]
        <= control["collision_observations"],
        "zero_added_teleports": action["teleport_observations"]
        <= control["teleport_observations"],
        "low_ttc_within_budget": action["low_ttc_exposure_steps"]
        - control["low_ttc_exposure_steps"]
        <= 2,
        "delay_within_budget": delay_delta <= 1.0,
        "arrival_within_budget": arrival_delta >= -1,
        "discharge_within_budget": discharge_delta >= -1,
    }
    if "treatment_safety_abort_rows" in action:
        constraints["zero_treatment_safety_aborts"] = not bool(
            action["treatment_safety_abort_rows"]
        )
    constraints_pass = all(constraints.values())
    control_influence = control["spatial_totals"]["influence_region"][
        "acceleration_energy_m2ps4"
    ]
    action_influence = action["spatial_totals"]["influence_region"][
        "acceleration_energy_m2ps4"
    ]
    effect = {
        "arm": action["arm"],
        "candidate_slot": action["candidate_slot"],
        "mode_index": action["mode_index"],
        "mode": action["mode"],
        "target_command_step_count": action["target_command_step_count"],
        **physical,
        "constraints": constraints,
        "constraints_pass": constraints_pass,
        "classification": classify_effect(
            physical["safety_composite_reference_units"],
            deadband=deadband,
            constraints_pass=constraints_pass,
        ),
        "delay_rate_delta": delay_delta,
        "arrival_delta": arrival_delta,
        "merge_discharge_count_delta": discharge_delta,
        "low_ttc_exposure_step_delta": action["low_ttc_exposure_steps"]
        - control["low_ttc_exposure_steps"],
        "influence_acceleration_energy_delta": action_influence - control_influence,
        "projection_survival_values": action["projection_survival_values"],
        "treatment_safety_abort_count": len(
            action.get("treatment_safety_abort_rows") or []
        ),
        "treatment_dose": action.get("treatment_dose"),
    }
    control_phase = control.get("phase_response") or {}
    action_phase = action.get("phase_response") or {}
    if control_phase.get("schema") == action_phase.get("schema") == PHASE_DIAGNOSTIC_SCHEMA:
        effect["paired_phase_response"] = {
            "schema": PHASE_DIAGNOSTIC_SCHEMA,
            "sign_convention": "treatment_minus_control",
            "steps": cumulative_paired_deltas(
                control_phase.get("steps") or [],
                action_phase.get("steps") or [],
            ),
        }
    return effect


def _paired_projection_survival(
    nominal: Mapping[str, Any], action: Mapping[str, Any]
) -> float | None:
    """Measure mode authority against the paired nominal command trajectory."""
    nominal_by_step = {
        int(row["relative_step"]): row
        for row in nominal["command_rows"]
        if row["vehicle_id"] == nominal["target_vehicle_id"]
    }
    action_by_step = {
        int(row["relative_step"]): row
        for row in action["command_rows"]
        if row["vehicle_id"] == action["target_vehicle_id"]
    }
    requested = executed = 0.0
    for step in sorted(set(nominal_by_step) & set(action_by_step)):
        base = nominal_by_step[step]
        treatment = action_by_step[step]
        values = (
            base.get("desired_acceleration_mps2"),
            treatment.get("desired_acceleration_mps2"),
            base.get("projected_acceleration_mps2"),
            treatment.get("projected_acceleration_mps2"),
        )
        if any(value is None for value in values):
            continue
        requested += abs(float(values[1]) - float(values[0]))
        executed += abs(float(values[3]) - float(values[2]))
    if requested <= 1e-9:
        return None
    return min(1.0, executed / requested)


def _analyze(
    protocol: Mapping[str, Any], discoveries: list[dict[str, Any]], branches: list[dict[str, Any]]
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in branches:
        by_case.setdefault(row["case_id"], []).append(row)
    candidate_slots = tuple(int(value) for value in protocol["candidate_slots"])
    arms = _arms(str(protocol["action_set"]), candidate_slots)
    active_modes = tuple(int(value) for value in protocol["active_modes"])
    nominal_index = int(protocol["nominal_mode_index"])
    complete = [rows for rows in by_case.values() if len(rows) == len(arms)]
    refs_payload = json.loads(
        Path(str(protocol["reward_refs_path"])).read_text(encoding="utf-8")
    )["refs"]
    replay_effects = []
    for rows in complete:
        indexed = {row["arm"]: row for row in rows}
        lhs, rhs = indexed["abstain"], indexed["abstain_replay"]
        refs = refs_payload[f"osm.dense.idm_m{lhs['demand']}r400.sumocfg"]
        replay_effects.append(
            abs(
                safety_composite(
                    zero_severe=lhs["severe_braking_count_sum"],
                    action_severe=rhs["severe_braking_count_sum"],
                    severe_reference=refs["corridor_severe_braking_count_ref_per_s"],
                    zero_wave=lhs["acceleration_energy_sum"],
                    action_wave=rhs["acceleration_energy_sum"],
                    wave_reference=refs["corridor_acceleration_energy_sum_ref_m2ps4"],
                    horizon_steps=int(protocol["safety_horizon_steps"]),
                )["safety_composite_reference_units"]
            )
        )
    deadband = replay_deadband(0.02, replay_effects)
    cases = []
    for rows in complete:
        indexed = {row["arm"]: row for row in rows}
        control = indexed["abstain"]
        refs = refs_payload[f"osm.dense.idm_m{control['demand']}r400.sumocfg"]
        effects = []
        for row in rows:
            if row["is_control"]:
                continue
            effect = _effect(
                control,
                row,
                {
                    "severe": refs["corridor_severe_braking_count_ref_per_s"],
                    "wave": refs["corridor_acceleration_energy_sum_ref_m2ps4"],
                },
                deadband,
                minimum_dose_steps=int(protocol["minimum_dose_steps"]),
                safety_horizon_steps=int(protocol["safety_horizon_steps"]),
                service_horizon_steps=int(protocol["service_horizon_steps"]),
            )
            nominal = next(
                candidate
                for candidate in rows
                if not candidate["is_control"]
                and candidate["candidate_slot"] == row["candidate_slot"]
                and candidate["mode_index"] == nominal_index
            )
            effect["paired_projection_survival"] = (
                None
                if row["mode_index"] == nominal_index
                else _paired_projection_survival(nominal, row)
            )
            effects.append(effect)
        cases.append(
            {
                "case_id": control["case_id"],
                "seed": control["seed"],
                "decision_step": control["decision_step"],
                "all_branch_decision_identity": len(
                    {row["decision_fingerprint"] for row in rows}
                )
                == 1,
                "all_discovery_identity": all(
                    row["decision_identity_with_discovery"] for row in rows
                ),
                "effects": sorted(
                    effects, key=lambda row: (row["candidate_slot"], row["mode_index"])
                ),
            }
        )
    effects = [effect for case in cases for effect in case["effects"]]
    dose_qualified = [
        row
        for row in effects
        if row["target_command_step_count"] >= int(protocol["minimum_dose_steps"])
    ]
    safety_service_admissible = [
        row
        for row in effects
        if all(
            value
            for key, value in row["constraints"].items()
            if key != "dose_at_least_minimum"
        )
    ]
    response = [
        row
        for row in dose_qualified
        if abs(row["safety_composite_reference_units"]) > deadband
    ]
    helpful_seeds = {
        case["seed"]
        for case in cases
        if any(row["classification"] == "helpful" for row in case["effects"])
    }
    nonbeneficial_seeds = {
        case["seed"]
        for case in cases
        if any(row["classification"] != "helpful" for row in case["effects"])
    }
    mode_winners = Counter()
    placement_contrasts = []
    slot_wins = Counter()
    for case in cases:
        admissible = [row for row in case["effects"] if row["constraints_pass"]]
        if admissible:
            winner = max(
                admissible, key=lambda row: row["safety_composite_reference_units"]
            )
            mode_winners[winner["mode"]] += 1
        indexed = {
            (row["candidate_slot"], row["mode_index"]): row for row in case["effects"]
        }
        if len(candidate_slots) >= 2:
            for mode in active_modes:
                first, second = indexed[(1, mode)], indexed[(2, mode)]
                contrast = second["safety_composite_reference_units"] - first[
                    "safety_composite_reference_units"
                ]
                placement_contrasts.append(contrast)
                if contrast > deadband:
                    slot_wins[2] += 1
                elif contrast < -deadband:
                    slot_wins[1] += 1
    projection_values = [
        float(row["paired_projection_survival"])
        for row in dose_qualified
        if row["mode_index"] != nominal_index
        and row["paired_projection_survival"] is not None
    ]
    integrity = {
        "decisions_found_for_all_requested_seeds": len(discoveries) > 0
        and all(row["decision"] is not None for row in discoveries),
        "all_branch_decision_identity": bool(cases)
        and all(case["all_branch_decision_identity"] for case in cases),
        "all_discovery_identity": bool(cases)
        and all(case["all_discovery_identity"] for case in cases),
        "abstain_replay_exact": all(value == 0.0 for value in replay_effects),
        "all_horizons_complete": all(
            row["completed_horizon_steps"]
            == int(protocol["service_horizon_steps"])
            for row in branches
        ),
        "control_issues_zero_commands": all(
            row["control_command_count"] == 0 for row in branches if row["is_control"]
        ),
        "treatments_target_only_declared_cav": all(
            row["unexpected_command_count"] == 0
            for row in branches
            if not row["is_control"]
        ),
        "zero_collisions": all(row["collision_observations"] == 0 for row in branches),
        "zero_teleports": all(row["teleport_observations"] == 0 for row in branches),
    }
    dose_fraction = len(dose_qualified) / len(effects) if effects else 0.0
    safe_fraction = (
        len(safety_service_admissible) / len(effects) if effects else 0.0
    )
    response_fraction = len(response) / len(dose_qualified) if dose_qualified else 0.0
    influence_fraction = (
        sum(abs(row["influence_acceleration_energy_delta"]) > 1e-6 for row in dose_qualified)
        / len(dose_qualified)
        if dose_qualified
        else 0.0
    )
    contrast_fraction = (
        sum(abs(value) > deadband for value in placement_contrasts)
        / len(placement_contrasts)
        if placement_contrasts
        else 0.0
    )
    smoke_gates = {
        **integrity,
        "dose_qualified_fraction_at_least_80pct": dose_fraction >= 0.80,
        "safe_fraction_at_least_90pct": safe_fraction >= 0.90,
    }
    activation_only = len(active_modes) == 1
    placement_screen = len(candidate_slots) >= 2
    pilot_integrity = {
        key: value
        for key, value in smoke_gates.items()
        if key != "decisions_found_for_all_requested_seeds"
    }
    pilot_gates = {
        **pilot_integrity,
        "at_least_five_seed_decisions": len(cases) >= 5,
        "response_fraction_at_least_30pct": response_fraction >= 0.30,
        "helpful_in_at_least_three_seeds": len(helpful_seeds) >= 3,
        "nonbeneficial_in_at_least_three_seeds": len(nonbeneficial_seeds) >= 3,
        "at_least_two_winning_modes": activation_only or len(mode_winners) >= 2,
        "placement_contrast_fraction_at_least_20pct": (
            not placement_screen or contrast_fraction >= 0.20
        ),
        "both_candidate_slots_win_twice": (
            not placement_screen or (slot_wins[1] >= 2 and slot_wins[2] >= 2)
        ),
        "influence_response_fraction_at_least_30pct": influence_fraction >= 0.30,
        "median_projection_survival_at_least_50pct": activation_only
        or (bool(projection_values) and float(np.median(projection_values)) >= 0.50),
    }
    return {
        "schema": SCHEMA,
        "protocol_hash": _canonical_hash(protocol),
        "protocol": protocol,
        "requested_seed_count": len(protocol["seeds"]),
        "decision_count": sum(row["decision"] is not None for row in discoveries),
        "branch_count": len(branches),
        "case_count": len(cases),
        "frozen_deadband": deadband,
        "replay_absolute_effects": replay_effects,
        "classification_counts": dict(
            Counter(row["classification"] for row in effects)
        ),
        "dose_qualified_fraction": dose_fraction,
        "safe_fraction": safe_fraction,
        "response_fraction": response_fraction,
        "helpful_seeds": sorted(helpful_seeds),
        "nonbeneficial_seeds": sorted(nonbeneficial_seeds),
        "mode_winner_counts": dict(mode_winners),
        "placement_contrast_fraction": contrast_fraction,
        "candidate_slot_win_counts": {str(key): value for key, value in slot_wins.items()},
        "influence_response_fraction": influence_fraction,
        "median_projection_survival": (
            float(np.median(projection_values)) if projection_values else None
        ),
        "smoke_gates": smoke_gates,
        "smoke_pass": all(smoke_gates.values()),
        "pilot_gates": pilot_gates,
        "training_authorized": _runner_training_authorized(
            str(protocol["action_set"]), pilot_gates
        ),
        "training_authorization_blockers": (
            ["development_only_feedback_treatment"]
            if protocol["action_set"]
            in {
                ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
                ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
                ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
                ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
            }
            else []
        ),
        "cases": sorted(cases, key=lambda row: row["seed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--evidence-role",
        choices=("development", "damping_acceptance"),
        default="development",
        help=(
            "Seed/provenance role. The prospective damping-acceptance role is "
            "restricted exactly to frozen seeds 4201--4220."
        ),
    )
    parser.add_argument(
        "--demand",
        type=int,
        default=2200,
        choices=(1900, 2000, 2100, 2200, 2700),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-predecision-steps", type=int, default=900)
    parser.add_argument(
        "--action-set",
        choices=(
            "gain_reference_v1",
            "acceleration_caps_v1",
            "activation_only_v1",
            "activation_speed_ceiling_v1",
            "activation_speed_ceiling_p50_v1",
            "activation_speed_ceiling_p70_v1",
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_ACTION_SET,
            ACTIVATION_SPEED_CEILING_P30_PROJECTED_RELEASE_V2_ACTION_SET,
            ACTIVATION_SPEED_CEILING_P30_SETTLE_RELEASE_ACTION_SET,
            ACTIVATION_MODIFIED_FOLLOWER_STOPPER_P30_ACTION_SET,
            ACTIVATION_IDM_P30_PROJECTED_BLEND_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_ACTION_SET,
            ACTIVATION_REAR_AWARE_FEEDBACK_V2_ACTION_SET,
        ),
        default="gain_reference_v1",
    )
    parser.add_argument(
        "--maximum-nominal-acceleration-mps2",
        type=float,
        default=None,
        help=(
            "Require at least one candidate's frozen analytical preview to be "
            "at or below this value (for example -0.2)."
        ),
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Write deterministic decision/lane-change diagnostics without treatment branches.",
    )
    parser.add_argument("--minimum-decision-step", type=int, default=0)
    parser.add_argument(
        "--option-steps",
        type=int,
        default=20,
        choices=(20, 25, 27, 29, 30),
        help=(
            "Frozen analytical-option duration. Values other than 20 are "
            "reserved for the P30 release-mechanism development screen; dose "
            "admission remains ceil(0.8 * option_steps)."
        ),
    )
    parser.add_argument(
        "--safety-horizon-steps",
        type=int,
        default=40,
        choices=(40, 60, 65, 67, 69, 70, 80),
        help=(
            "Paired outcome horizon. Values 65--70 are reserved "
            "for release-mechanism development comparisons that retain the "
            "frozen 40-step post-active observation window."
        ),
    )
    parser.add_argument(
        "--cav-penetration",
        type=float,
        default=0.10,
        choices=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    )
    parser.add_argument("--decision-index", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument(
        "--maximum-landmarks-per-cell", type=int, default=3, choices=(1, 2, 3)
    )
    parser.add_argument(
        "--reward-refs",
        type=Path,
        default=ROOT / "runs/eval/idm_phase5_reward_refs_v1.json",
        help="Frozen reward/reference-scale JSON covering the selected demand.",
    )
    parser.add_argument(
        "--candidate-ranking",
        choices=("runway", "rear_headroom", "prospective_rear_headroom"),
        default="runway",
    )
    parser.add_argument("--candidate-count", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--landmark",
        choices=("two_candidate", "wave_onset", "severe_onset", "risk_onset"),
        default="two_candidate",
        help=(
            "Freeze the decision at a persistent causal onset; risk_onset uses "
            "the existing wave-then-severe priority."
        ),
    )
    parser.add_argument(
        "--decision-source",
        type=Path,
        default=None,
        help=(
            "Optional frozen P30 authority-verification directory whose decision "
            "steps and candidate-1 targets define an approved matched P50, RSU, "
            "or phase/spatial diagnostic replay."
        ),
    )
    parser.add_argument(
        "--rsu-observation-contract",
        choices=RSU_OBSERVATION_CONTRACTS,
        default=RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
        help=(
            "Versioned RSU observation contract. The 2000 m contract is "
            "restricted to discover-only replay of a frozen P30 decision source."
        ),
    )
    parser.add_argument(
        "--response-trace-contract",
        choices=("none", PHASE_DIAGNOSTIC_SCHEMA),
        default="none",
        help=(
            "Opt-in phase-resolved response trace. This is diagnostic only and "
            "does not change the fixed P30/P50 treatment."
        ),
    )
    parser.add_argument(
        "--diagnostic-spatial-history",
        action="store_true",
        help=(
            "Record a causal 30-second spatial RSU history without adding it "
            "to actor observations, rewards, or candidate ranking."
        ),
    )
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    allowed_seeds = (
        set(DAMPING_ACCEPTANCE_SEEDS)
        if args.evidence_role == "damping_acceptance"
        else set(range(4107, 4121)) | set(CONTROLLER_FEATURE_VALIDATION_SEEDS)
    )
    unexpected_seeds = sorted(set(seeds) - allowed_seeds)
    if unexpected_seeds:
        raise ValueError(
            f"verification seed role {args.evidence_role!r} rejects "
            f"{unexpected_seeds}"
        )
    if args.evidence_role == "damping_acceptance" and tuple(seeds) != tuple(
        DAMPING_ACCEPTANCE_SEEDS
    ):
        raise ValueError(
            "prospective damping acceptance must use the complete frozen seed set "
            "4201--4220 in one resumable protocol"
        )
    if args.evidence_role == "damping_acceptance":
        prospective_contract = {
            "demand": args.demand in {1900, 2000, 2100},
            "action_set": args.action_set == "activation_speed_ceiling_v1",
            "option_steps": args.option_steps == 20,
            "safety_horizon_steps": args.safety_horizon_steps == 60,
            "cav_penetration": args.cav_penetration in {0.05, 0.10},
            "decision_index": args.decision_index == 3,
            "maximum_landmarks_per_cell": args.maximum_landmarks_per_cell == 3,
            "candidate_ranking": args.candidate_ranking == "runway",
            "candidate_count": args.candidate_count == 1,
            "landmark": args.landmark == "risk_onset",
            "fresh_discovery": args.decision_source is None,
            "not_discover_only": not args.discover_only,
            "legacy_rsu_contract": args.rsu_observation_contract
            == RSU_CONTRACT_LEGACY_CONFIGURED_RANGE_V1,
            "no_response_trace": args.response_trace_contract == "none",
            "no_spatial_diagnostic": not args.diagnostic_spatial_history,
            "default_predecision_horizon": args.max_predecision_steps == 900,
            "no_minimum_step_override": args.minimum_decision_step == 0,
            "no_nominal_acceleration_filter": args.maximum_nominal_acceleration_mps2
            is None,
        }
        failed = sorted(
            name for name, valid in prospective_contract.items() if not valid
        )
        if failed:
            raise ValueError(
                "prospective damping-acceptance contract mismatch: "
                + ", ".join(failed)
            )
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must remain in [1, 4]")
    _validate_feedback_option_steps(args.action_set, args.option_steps)
    definitions = modes_for(args.action_set)
    feedback_contract = _feedback_contract_for_action_set(args.action_set)
    minimum_dose_steps = (
        int(feedback_contract.probe_steps)
        if feedback_contract is not None
        and hasattr(feedback_contract, "probe_steps")
        else int(math.ceil(0.8 * args.option_steps))
    )
    reward_refs_path = args.reward_refs.resolve()
    reward_refs_bytes = reward_refs_path.read_bytes()
    if (
        args.evidence_role == "damping_acceptance"
        and hashlib.sha256(reward_refs_bytes).hexdigest()
        != "737b9ce4bb1945f7681b807941c86509ae3143128bd0729d2cb2aee5237c5b58"
    ):
        raise ValueError("prospective damping acceptance requires the frozen reward refs")
    reward_refs_payload = json.loads(reward_refs_bytes.decode("utf-8"))
    reward_refs_key = f"osm.dense.idm_m{args.demand}r400.sumocfg"
    if reward_refs_key not in reward_refs_payload.get("refs", {}):
        raise ValueError(
            f"reward reference {reward_refs_key!r} is absent from {reward_refs_path}"
        )
    if args.decision_index > args.maximum_landmarks_per_cell:
        raise ValueError("decision index exceeds the frozen maximum landmark count")
    landmark_contract = LandmarkContract(
        maximum_landmarks_per_cell=args.maximum_landmarks_per_cell
    )
    decision_source_path = (
        args.decision_source.resolve() if args.decision_source is not None else None
    )
    decision_sources: dict[int, dict[str, Any]] = {}
    decision_source_manifest = None
    decision_source_manifest_sha256 = None
    decision_source_purpose = None
    if decision_source_path is not None:
        source_manifest_path = decision_source_path / "manifest.json"
        source_manifest_bytes = source_manifest_path.read_bytes()
        decision_source_manifest = json.loads(source_manifest_bytes.decode("utf-8"))
        decision_source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
        if decision_source_manifest.get("action_set") != "activation_speed_ceiling_v1":
            raise ValueError("matched decision source must use frozen P30")
        if int(decision_source_manifest.get("demand", -1)) != args.demand:
            raise ValueError("matched decision source demand differs from requested demand")
        decision_source_purpose = _matched_decision_source_purpose(
            action_set=args.action_set,
            discover_only=bool(args.discover_only),
            rsu_observation_contract=args.rsu_observation_contract,
            response_trace_contract=args.response_trace_contract,
            diagnostic_spatial_history=bool(args.diagnostic_spatial_history),
        )
        if decision_source_purpose is None:
            raise ValueError(
                "--decision-source is restricted to an approved P50 match, "
                "discover-only P30 RSU replay, or full P30 phase/spatial replay"
                ", including the frozen projected-release and projected-blend "
                "mechanism matches"
            )
        if not _matched_seed_scope_is_valid(
            source_seeds=decision_source_manifest.get("seeds", ()),
            requested_seeds=seeds,
            purpose=decision_source_purpose,
        ):
            raise ValueError(
                "matched decision-source seed scope is incompatible with the "
                f"approved purpose {decision_source_purpose!r}"
            )
        for path in sorted((decision_source_path / "discoveries").glob("*.json")):
            source = json.loads(path.read_text(encoding="utf-8"))
            if int(source["seed"]) not in set(seeds):
                continue
            source_decision = source.get("decision")
            if source_decision is None:
                decision_sources[int(source["seed"])] = {"decision_missing": True}
                continue
            candidates = source_decision.get("candidates") or []
            if not candidates:
                continue
            decision_sources[int(source["seed"])] = {
                "decision_step": int(source_decision["decision_step"]),
                "decision_index": int(source_decision["decision_index"]),
                "landmark": str(source_decision["landmark"]),
                "landmark_persistence_steps": int(
                    source_decision["landmark_persistence_steps"]
                ),
                "protocol_hash": str(source["protocol_hash"]),
                "decision_fingerprint": str(source_decision["fingerprint"]),
                "target_vehicle_id": str(candidates[0]["vehicle_id"]),
                "physical_state_fingerprint": _physical_state_fingerprint(
                    source_decision
                ),
                "raw_decision_context_fingerprint": (
                    _raw_decision_context_fingerprint(source_decision)
                ),
                "treatment_fingerprint": _treatment_fingerprint(source_decision),
                "nonclassic_observation_hash": _nonclassic_observation_hash(
                    source_decision["candidate_observations"][
                        str(candidates[0]["vehicle_id"])
                    ]
                ),
            }
        if set(decision_sources) != set(seeds):
            raise ValueError("matched decision source lacks a discovery row for every requested seed")
    if (
        args.rsu_observation_contract
        == RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1
        and decision_source_purpose != "rsu_observation_replay"
    ):
        raise ValueError(
            "the 2000 m diagnostic RSU contract requires --discover-only and a "
            "frozen P30 --decision-source"
        )
    protocol = {
        "schema": SCHEMA,
        "development_only": args.evidence_role == "development",
        "demand": args.demand,
        "seeds": seeds,
        "candidate_slots": list(range(1, args.candidate_count + 1)),
        "action_set": args.action_set,
        "terminal_reference_percentile": terminal_reference_percentile(args.action_set),
        "active_modes": [mode.index for mode in definitions if mode.index != 0],
        "nominal_mode_index": nominal_mode_index(args.action_set),
        "option_steps": args.option_steps,
        "minimum_dose_steps": minimum_dose_steps,
        "cav_penetration": args.cav_penetration,
        "decision_index": args.decision_index,
        "candidate_ranking": args.candidate_ranking,
        "candidate_count": args.candidate_count,
        "safety_horizon_steps": args.safety_horizon_steps,
        "service_horizon_steps": _service_horizon_steps(
            args.safety_horizon_steps
        ),
        "max_predecision_steps": args.max_predecision_steps,
        "minimum_decision_step": args.minimum_decision_step,
        "landmark": args.landmark,
        "landmark_contract": landmark_contract.to_dict(),
        "maximum_landmarks_per_cell": args.maximum_landmarks_per_cell,
        "reward_refs_path": str(reward_refs_path),
        "reward_refs_sha256": hashlib.sha256(reward_refs_bytes).hexdigest(),
        "maximum_nominal_acceleration_mps2": args.maximum_nominal_acceleration_mps2,
        "workers": args.workers,
        "replay_deadband_base": 0.02,
        "discover_only": bool(args.discover_only),
        "decision_source_path": (
            str(decision_source_path) if decision_source_path is not None else None
        ),
        "decision_source_manifest_sha256": decision_source_manifest_sha256,
        "decision_source_protocol_hash": (
            decision_source_manifest.get("protocol_hash")
            if decision_source_manifest is not None
            else None
        ),
    }
    # Preserve every existing development manifest hash. The role field is
    # emitted only for the new prospective protocol.
    if args.evidence_role != "development":
        protocol["evidence_role"] = args.evidence_role
    # Keep the legacy/P50 protocol byte-for-byte compatible with existing
    # manifests. Observation-contract identity is added only for the new
    # diagnostic contract, never retrofitted into frozen legacy hashes.
    if args.rsu_observation_contract == RSU_CONTRACT_LIVE_2000_DIAGNOSTIC_V1:
        protocol.update(
            {
                "rsu_observation_contract": args.rsu_observation_contract,
                "rsu_communication_range_m": RSU_DIAGNOSTIC_RANGE_M,
                "decision_source_purpose": decision_source_purpose,
            }
        )
    elif decision_source_purpose in {
        "p30_phase_spatial_replay",
        "p30_projected_release_match",
        "p30_modified_followerstopper_match",
        "p30_projected_blend_match",
        "rear_aware_feedback_match",
    }:
        protocol.update(
            {
                "decision_source_purpose": decision_source_purpose,
                "decision_source_seed_scope": "requested_subset_of_frozen_manifest",
            }
        )
        if decision_source_purpose == "p30_projected_release_match":
            protocol["projected_release_contract"] = (
                _projected_release_contract_for_action_set(args.action_set).to_dict()
            )
        elif decision_source_purpose == "p30_modified_followerstopper_match":
            protocol["feedback_contract"] = (
                _feedback_contract_for_action_set(args.action_set).to_dict()
            )
        elif decision_source_purpose == "p30_projected_blend_match":
            protocol["blend_contract"] = (
                _blend_contract_for_action_set(args.action_set).to_dict()
            )
        elif decision_source_purpose == "rear_aware_feedback_match":
            protocol["feedback_contract"] = (
                _feedback_contract_for_action_set(args.action_set).to_dict()
            )
            protocol["projected_release_contract"] = (
                _projected_release_contract_for_action_set(args.action_set).to_dict()
            )
    if args.response_trace_contract != "none":
        protocol["response_trace_contract"] = args.response_trace_contract
    if args.diagnostic_spatial_history:
        protocol["diagnostic_spatial_history_contract"] = (
            SpatialHistoryContract().to_dict()
        )
    protocol_hash = _canonical_hash(protocol)
    output = args.output.resolve()
    for child in ("discoveries", "branches"):
        (output / child).mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol_hash:
            raise RuntimeError("output contains a different frozen protocol")
    else:
        atomic_write_json(manifest, {**protocol, "protocol_hash": protocol_hash})

    discoveries = []
    pending = []
    for seed in seeds:
        path = output / "discoveries" / f"m{args.demand}.seed{seed}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("protocol_hash") == protocol_hash:
                discoveries.append(row)
                continue
        pending.append(
            {
                "demand": args.demand,
                "seed": seed,
                "output": str(output),
                "max_predecision_steps": args.max_predecision_steps,
                "minimum_decision_step": args.minimum_decision_step,
                "landmark": args.landmark,
                "maximum_nominal_acceleration_mps2": (
                    args.maximum_nominal_acceleration_mps2
                ),
                "protocol_hash": protocol_hash,
                "action_set": args.action_set,
                "option_steps": args.option_steps,
                "minimum_dose_steps": minimum_dose_steps,
                "cav_penetration": args.cav_penetration,
                "decision_index": args.decision_index,
                "maximum_landmarks_per_cell": args.maximum_landmarks_per_cell,
                "candidate_ranking": args.candidate_ranking,
                "candidate_count": args.candidate_count,
                "frozen_decision_source": decision_sources.get(seed),
                "decision_source_purpose": decision_source_purpose,
                "rsu_observation_contract": args.rsu_observation_contract,
                "path": str(path),
            }
        )
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(discover, spec): spec for spec in pending}
            for future in as_completed(futures):
                row = future.result()
                atomic_write_json(Path(futures[future]["path"]), row)
                discoveries.append(row)
                print(
                    f"[discover] seed{row['seed']} found={row['decision'] is not None} "
                    f"runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )

    if args.discover_only:
        payload = {
            "schema": "idm_decentralized_discovery_diagnostic/1",
            "protocol": protocol,
            "protocol_hash": protocol_hash,
            "requested_seed_count": len(seeds),
            "decision_count": sum(row.get("decision") is not None for row in discoveries),
            "discoveries": sorted(discoveries, key=lambda row: row["seed"]),
        }
        atomic_write_json(output / "discovery_summary.json", payload)
        print(
            json.dumps(
                {
                    "requested_seed_count": payload["requested_seed_count"],
                    "decision_count": payload["decision_count"],
                },
                indent=2,
            )
        )
        return

    arms = _arms(args.action_set, tuple(protocol["candidate_slots"]))
    branches = []
    pending_branches = []
    for discovery in sorted(discoveries, key=lambda row: row["seed"]):
        decision = discovery["decision"]
        if decision is None:
            continue
        case_id = f"m{args.demand}.seed{discovery['seed']}.t{decision['decision_step']}"
        for arm in arms:
            branch_id = f"{case_id}.{arm['arm']}"
            path = output / "branches" / f"{branch_id}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("protocol_hash") == protocol_hash:
                    branches.append(row)
                    continue
            pending_branches.append(
                {
                    **arm,
                    "demand": args.demand,
                    "seed": discovery["seed"],
                    "decision": decision,
                    "case_id": case_id,
                    "branch_id": branch_id,
                    "output": str(output),
                    "protocol_hash": protocol_hash,
                    "action_set": args.action_set,
                    "option_steps": args.option_steps,
                    "minimum_dose_steps": minimum_dose_steps,
                    "cav_penetration": args.cav_penetration,
                    "candidate_ranking": args.candidate_ranking,
                    "candidate_count": args.candidate_count,
                    "safety_horizon_steps": args.safety_horizon_steps,
                    "service_horizon_steps": _service_horizon_steps(
                        args.safety_horizon_steps
                    ),
                    "rsu_observation_contract": args.rsu_observation_contract,
                    "response_trace_contract": args.response_trace_contract,
                    "diagnostic_spatial_history": bool(
                        args.diagnostic_spatial_history
                    ),
                    "path": str(path),
                }
            )
    atomic_write_json(
        output / "progress.json",
        {
            "protocol_hash": protocol_hash,
            "complete": len(branches),
            "pending": len(pending_branches),
        },
    )
    if pending_branches:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_branch, spec): spec for spec in pending_branches}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                atomic_write_json(Path(futures[future]["path"]), row)
                branches.append(row)
                atomic_write_json(
                    output / "progress.json",
                    {
                        "protocol_hash": protocol_hash,
                        "complete": len(branches),
                        "pending": len(pending_branches) - index,
                    },
                )
                print(
                    f"[branch {index}/{len(pending_branches)}] {row['branch_id']} "
                    f"dose={row['target_command_step_count']} runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    summary = _analyze(protocol, discoveries, branches)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        output / "progress.json",
        {"protocol_hash": protocol_hash, "complete": len(branches), "pending": 0},
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "decision_count",
                    "branch_count",
                    "case_count",
                    "classification_counts",
                    "dose_qualified_fraction",
                    "safe_fraction",
                    "response_fraction",
                    "smoke_gates",
                    "smoke_pass",
                    "training_authorized",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
