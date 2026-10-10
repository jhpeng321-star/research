"""Resumable action-local residual-pulse diagnostic for IDM Phase 5.

Landmark and branch artifacts are written atomically and tied to a frozen
manifest. Rerunning an identical command skips completed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import traci

from env.core import TrafficShepherdMultiAgentEnv
from env.idm_residual import detect_disturbances
from env.snapshot import get_active_snapshot
from experiments import REPO_ROOT
from experiments.idm.run_idm_residual_checkpoint_eval import REWARD_COMPONENT_KEYS, _env_config
from experiments.io import atomic_write_json
from idm.action_local import (
    LandmarkContract,
    OutcomeContract,
    classify_effect,
    collapse_condition,
    landmark_signal,
    replay_deadband,
    safety_composite,
)

ROOT = REPO_ROOT
SCHEMA = "idm_phase5_action_local_pulse/5"
LOCAL_DIM = 134
GATE_INDEX = 123
CONTEXT_INDICES = {
    "idm_nominal_acceleration": 124,
    "idm_filtered_reference": 125,
    "idm_reference_error": 126,
    "idm_projection_delta": 127,
    "idm_severe_level": 128,
    "idm_wave_energy_level": 129,
    "idm_severe_short_ema": 130,
    "idm_wave_short_ema": 131,
    "idm_severe_trend": 132,
    "idm_wave_energy_trend": 133,
}


def _slug_residual(value: float) -> str:
    if abs(value) < 1e-12:
        return "zero"
    return ("p" if value > 0 else "n") + f"{abs(value):.3f}".replace(".", "")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _zero_actions(obs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {agent_id: np.asarray([0.0], dtype=np.float32) for agent_id in obs}


def _treatment_actions(
    obs: Mapping[str, np.ndarray], *, treatment_ids: set[str], residual: float
) -> dict[str, np.ndarray]:
    return {
        agent_id: np.asarray([residual if agent_id in treatment_ids else 0.0], dtype=np.float32)
        for agent_id in obs
    }


def _local_observations(obs: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    result = {}
    for agent_id, packed in sorted(obs.items()):
        local = np.asarray(packed, dtype=np.float32).reshape(-1)[:LOCAL_DIM]
        if local.size == LOCAL_DIM:
            result[str(agent_id)] = local.tolist()
    return result


def _selected_ids(local: Mapping[str, list[float]]) -> list[str]:
    return sorted(agent_id for agent_id, row in local.items() if row[GATE_INDEX] > 0.5)


def _context(row: list[float]) -> dict[str, float]:
    return {name: float(row[index]) for name, index in CONTEXT_INDICES.items()}


def _premerge_speeds(env: TrafficShepherdMultiAgentEnv) -> list[float]:
    snap = get_active_snapshot()
    if snap is None:
        return []
    values = []
    for vehicle_id in snap.ordered_ids():
        if snap.road(vehicle_id) == env._pre_edge:
            speed = snap.speed(vehicle_id)
            if speed is not None:
                values.append(float(speed))
    return values


def _selected_ids_in_lane(local: Mapping[str, list[float]], lane_id: str) -> set[str]:
    snap = get_active_snapshot()
    if snap is None:
        return set()
    return {
        vehicle_id for vehicle_id in _selected_ids(local) if snap.lane_id(vehicle_id) == lane_id
    }


def _absorber_geometry(env: TrafficShepherdMultiAgentEnv, absorber_id: str) -> dict[str, Any]:
    engine = env._idm_residual_engine
    samples = engine._samples()
    by_id = {sample.vehicle_id: sample for sample in samples}
    sample = by_id.get(absorber_id)
    disturbances = detect_disturbances(samples, engine.cfg)
    disturbance = disturbances.get(sample.lane_id) if sample is not None else None
    runway = (
        sample.distance_to_merge_m - disturbance.distance_to_merge_m
        if sample is not None and disturbance is not None
        else None
    )
    follower = engine._leader_follower(absorber_id) if sample is not None else {}
    return {
        "lane_id": None if sample is None else sample.lane_id,
        "distance_to_merge_m": None if sample is None else sample.distance_to_merge_m,
        "disturbance_distance_to_merge_m": (
            None if disturbance is None else disturbance.distance_to_merge_m
        ),
        "runway_m": runway,
        "follower_id": follower.get("follower_id"),
        "follower_gap_m": follower.get("follower_gap_m"),
        "follower_speed_mps": follower.get("follower_speed_mps"),
        "leader_id": follower.get("leader_id"),
        "leader_gap_m": follower.get("leader_gap_m"),
        "leader_speed_mps": follower.get("leader_speed_mps"),
    }


def _population_metrics(
    snap: Any,
    vehicle_ids: set[str],
    *,
    previous_acceleration: Mapping[str, float],
    severe_threshold_mps2: float,
) -> dict[str, Any]:
    """Summarize one spatial population from the current causal snapshot."""
    speeds: list[float] = []
    accelerations: list[float] = []
    jerks: list[float] = []
    ttc_values: list[float] = []
    integrated_negative_acceleration = 0.0
    severe = 0
    for vehicle_id in sorted(vehicle_ids):
        speed = snap.speed(vehicle_id)
        acceleration = snap.accel(vehicle_id)
        if speed is None or acceleration is None:
            continue
        acceleration = float(acceleration)
        speeds.append(float(speed))
        accelerations.append(acceleration)
        integrated_negative_acceleration += max(-acceleration, 0.0)
        severe += int(acceleration < severe_threshold_mps2)
        if vehicle_id in previous_acceleration:
            jerks.append(acceleration - float(previous_acceleration[vehicle_id]))
        leader = snap.leader(vehicle_id)
        if leader and leader[0] and snap.speed(str(leader[0])) is not None:
            closing_speed = float(speed) - float(snap.speed(str(leader[0])))
            if closing_speed > 1e-6:
                ttc_values.append(max(float(leader[1]), 0.0) / closing_speed)
    return {
        "vehicle_count": len(accelerations),
        "mean_speed_mps": float(np.mean(speeds)) if speeds else None,
        "speed_variance_m2ps2": float(np.var(speeds)) if speeds else None,
        "mean_acceleration_mps2": (
            float(np.mean(accelerations)) if accelerations else None
        ),
        "acceleration_energy_m2ps4": sum(value * value for value in accelerations),
        "integrated_negative_acceleration_mps": integrated_negative_acceleration,
        "severe_braking_count": severe,
        "mean_abs_jerk_mps3": float(np.mean(np.abs(jerks))) if jerks else None,
        "minimum_ttc_s": min(ttc_values) if ttc_values else None,
    }


def _spatial_diagnostics(
    env: TrafficShepherdMultiAgentEnv,
    *,
    role_ids: set[str],
    frozen_lane_id: str,
    previous_acceleration: Mapping[str, float],
    influence_distance_m: float = 300.0,
) -> dict[str, Any]:
    """Measure the absorber-to-corridor response chain without changing control."""
    snap = get_active_snapshot()
    if snap is None:
        return {}
    valid_roles = {vehicle_id for vehicle_id in role_ids if snap.has(vehicle_id)}
    followers: set[str] = set()
    for vehicle_id in valid_roles:
        follower = env._idm_residual_engine._leader_follower(vehicle_id).get("follower_id")
        if follower and snap.has(str(follower)):
            followers.add(str(follower))
    lane_ids = {
        vehicle_id
        for vehicle_id in snap.ordered_ids()
        if snap.lane_id(vehicle_id) == frozen_lane_id
    }
    influence_ids: set[str] = set()
    role_positions = [
        float(position)
        for vehicle_id in valid_roles
        for position in [snap.lane_pos(vehicle_id)]
        if position is not None
    ]
    if role_positions:
        front_position = max(role_positions)
        influence_ids = {
            vehicle_id
            for vehicle_id in lane_ids
            for position in [snap.lane_pos(vehicle_id)]
            if position is not None and 0.0 <= front_position - float(position) <= influence_distance_m
        }
    corridor_edges = {
        env._pre_edge,
        env._merge_edge,
        env._ramp_edge,
        env._shepherd_engine.down_edge,
    }
    corridor_ids = {
        vehicle_id
        for vehicle_id in snap.ordered_ids()
        if snap.road(vehicle_id) in corridor_edges
    }
    threshold = float(env.config.get("idm_severe_brake_threshold_mps2", -3.0))
    return {
        "role_vehicle_ids": sorted(valid_roles),
        "immediate_follower_ids": sorted(followers),
        "absorber": _population_metrics(
            snap,
            valid_roles,
            previous_acceleration=previous_acceleration,
            severe_threshold_mps2=threshold,
        ),
        "immediate_follower": _population_metrics(
            snap,
            followers,
            previous_acceleration=previous_acceleration,
            severe_threshold_mps2=threshold,
        ),
        "influence_region": _population_metrics(
            snap,
            influence_ids,
            previous_acceleration=previous_acceleration,
            severe_threshold_mps2=threshold,
        ),
        "lane": _population_metrics(
            snap,
            lane_ids,
            previous_acceleration=previous_acceleration,
            severe_threshold_mps2=threshold,
        ),
        "corridor": _population_metrics(
            snap,
            corridor_ids,
            previous_acceleration=previous_acceleration,
            severe_threshold_mps2=threshold,
        ),
    }


def _decision_fingerprint(
    *,
    step: int,
    landmark: str,
    absorber_id: str,
    observations: Mapping[str, list[float]],
    selected_ids: list[str],
    controller_diag: Mapping[str, Any],
) -> str:
    return _canonical_hash(
        {
            "step": step,
            "landmark": landmark,
            "absorber_id": absorber_id,
            "observations": observations,
            "selected_ids": selected_ids,
            "controller_diag": controller_diag,
        }
    )


def _env(demand: int, seed: int, output: Path, tag: str, max_steps: int):
    cfg = _env_config(demand, seed, output / "artifacts", tag, max_steps, None)
    cfg["eval_mode"] = False
    return TrafficShepherdMultiAgentEnv(cfg)


def discover_landmarks(spec: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    demand, seed = int(spec["demand"]), int(spec["seed"])
    contract = LandmarkContract(**spec["landmark_contract"])
    output = Path(spec["output"])
    env = _env(demand, seed, output, f"discover.m{demand}.seed{seed}", spec["max_steps"] + 5)
    persistence = {name: 0 for name in contract.priority}
    persistence_absorber: dict[str, str | None] = {name: None for name in contract.priority}
    accepted_types: set[str] = set()
    last_accepted_step: int | None = None
    collapse_run = 0
    landmarks: list[dict[str, Any]] = []
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(int(spec["max_steps"]) + 1):
            local = _local_observations(obs)
            selected = _selected_ids(local)
            collapse_run = (
                collapse_run + 1 if collapse_condition(_premerge_speeds(env), contract) else 0
            )
            eligible = len(selected) == 1 and collapse_run < contract.collapse_sustain_steps
            absorber_id = selected[0] if eligible else None
            geometry = _absorber_geometry(env, absorber_id) if absorber_id else {}
            runway = geometry.get("runway_m")
            eligible = bool(
                eligible and runway is not None and float(runway) >= contract.minimum_runway_m
            )
            current_context = _context(local[absorber_id]) if eligible and absorber_id else None
            matured: list[str] = []
            for landmark in contract.priority:
                active = bool(
                    eligible
                    and landmark not in accepted_types
                    and current_context is not None
                    and landmark_signal(landmark, current_context, contract)
                )
                if active and persistence_absorber[landmark] == absorber_id:
                    persistence[landmark] += 1
                elif active:
                    persistence[landmark] = 1
                    persistence_absorber[landmark] = absorber_id
                else:
                    persistence[landmark] = 0
                    persistence_absorber[landmark] = None
                if persistence[landmark] >= contract.persistence_steps:
                    matured.append(landmark)
            separated = (
                last_accepted_step is None
                or step - last_accepted_step >= contract.minimum_landmark_separation_steps
            )
            chosen = next((name for name in contract.priority if name in matured), None)
            if chosen is not None and separated and absorber_id is not None:
                controller_diag = dict(env._last_idm_residual_diag)
                landmarks.append(
                    {
                        "landmark_index": len(landmarks) + 1,
                        "landmark": chosen,
                        "decision_step": step,
                        "absorber_id": absorber_id,
                        "selected_ids": selected,
                        "geometry": geometry,
                        "context": current_context,
                        "observation": local[absorber_id],
                        "observation_fingerprint": _decision_fingerprint(
                            step=step,
                            landmark=chosen,
                            absorber_id=absorber_id,
                            observations=local,
                            selected_ids=selected,
                            controller_diag=controller_diag,
                        ),
                        "collapse_run_steps": collapse_run,
                    }
                )
                accepted_types.add(chosen)
                last_accepted_step = step
                persistence[chosen] = 0
                persistence_absorber[chosen] = None
                if len(landmarks) >= contract.maximum_landmarks_per_cell:
                    break
            if step >= int(spec["max_steps"]):
                break
            obs, _, terminated, truncated, _ = env.step(_zero_actions(obs))
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
    finally:
        env.close()
    return {
        "schema": SCHEMA,
        "protocol_hash": spec["protocol_hash"],
        "demand": demand,
        "seed": seed,
        "landmarks": landmarks,
        "runtime_s": time.perf_counter() - started,
    }


def _projection_details(command: Mapping[str, Any]) -> tuple[bool, list[str], float | None]:
    projected = command.get("projected") or {}
    reasons = list(projected.get("reasons") or projected.get("constraints") or [])
    acceleration = projected.get("acceleration_mps2")
    return (
        bool(reasons),
        [str(reason) for reason in reasons],
        (None if acceleration is None else float(acceleration)),
    )


def run_branch(spec: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    demand, seed = int(spec["demand"]), int(spec["seed"])
    residual = float(spec["residual"])
    decision = spec["decision"]
    outcome = OutcomeContract(**spec["outcome_contract"])
    output = Path(spec["output"])
    decision_step = int(decision["decision_step"])
    absorber_id = str(decision["absorber_id"])
    env = _env(
        demand,
        seed,
        output,
        f"{spec['branch_id']}.m{demand}.seed{seed}",
        decision_step + outcome.service_horizon_steps + 5,
    )
    severe_total = wave_total = ttc_count_total = 0.0
    low_ttc_exposure_steps = 0
    delay_rates: list[float] = []
    merge_discharge_rates: list[float] = []
    reward_total = 0.0
    reward_by_agent: dict[str, float] = {}
    component_totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    collisions = teleports = arrivals = 0
    completed = 0
    command_trace: list[dict[str, Any]] = []
    spatial_trace: list[dict[str, Any]] = []
    treatment_request_violations = 0
    treated_role_ids: set[str] = set()
    observed_role_ids: set[str] = set()
    local: dict[str, list[float]] = {}
    selected: list[str] = []
    fingerprint = ""
    geometry: dict[str, Any] = {}
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(decision_step):
            obs, _, terminated, truncated, _ = env.step(_zero_actions(obs))
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        local = _local_observations(obs)
        selected = _selected_ids(local)
        fingerprint = _decision_fingerprint(
            step=decision_step,
            landmark=str(decision["landmark"]),
            absorber_id=absorber_id,
            observations=local,
            selected_ids=selected,
            controller_diag=dict(env._last_idm_residual_diag),
        )
        geometry = _absorber_geometry(env, absorber_id)
        frozen_lane_id = str(decision["geometry"]["lane_id"])
        for relative_step in range(1, outcome.service_horizon_steps + 1):
            requested = residual if relative_step <= outcome.pulse_steps else 0.0
            current_local = _local_observations(obs)
            role_ids = _selected_ids_in_lane(current_local, frozen_lane_id)
            treatment_ids = role_ids if relative_step <= outcome.pulse_steps else set()
            observed_role_ids.update(role_ids)
            treated_role_ids.update(treatment_ids)
            pre_step_snapshot = get_active_snapshot()
            previous_acceleration = (
                {
                    vehicle_id: float(acceleration)
                    for vehicle_id in pre_step_snapshot.ordered_ids()
                    for acceleration in [pre_step_snapshot.accel(vehicle_id)]
                    if acceleration is not None
                }
                if pre_step_snapshot is not None
                else {}
            )
            actions = _treatment_actions(obs, treatment_ids=treatment_ids, residual=requested)
            treatment_request_violations += sum(
                abs(float(action[0])) > 1e-12 and agent_id not in treatment_ids
                for agent_id, action in actions.items()
            )
            obs, rewards, terminated, truncated, info = env.step(actions)
            if relative_step <= outcome.safety_horizon_steps:
                severe_total += float(env._corridor_severe_braking_count() or 0.0)
                wave_total += float(env._corridor_acceleration_energy_sum_m2ps4() or 0.0)
                current_ttc = float(env._corridor_ttc_count() or 0.0)
                ttc_count_total += current_ttc
                low_ttc_exposure_steps += int(current_ttc > 0)
            delay = env._corridor_delay_rate_live()
            if delay is not None and math.isfinite(float(delay)):
                delay_rates.append(float(delay))
            merge_discharge_rates.append(float(env._flow_tracker.rate()))
            collisions += int(traci.simulation.getCollidingVehiclesNumber())
            teleports += int(traci.simulation.getStartingTeleportNumber())
            arrivals += int(traci.simulation.getArrivedNumber())
            reward_total += sum(float(value) for value in rewards.values())
            for agent_id, value in rewards.items():
                reward_by_agent[str(agent_id)] = reward_by_agent.get(str(agent_id), 0.0) + float(
                    value
                )
            for component_row in info.get("idm_reward_components", {}).values():
                for key in REWARD_COMPONENT_KEYS:
                    component_totals[key] += float(component_row.get(key, 0.0))
            step_commands = []
            for command in info.get("idm_residual", {}).get("commands", []):
                projected, reasons, executed_acceleration = _projection_details(command)
                vehicle_id = str(command["vehicle_id"])
                executed_residual = float(command.get("residual_mps2", 0.0))
                step_commands.append(
                    {
                        "vehicle_id": vehicle_id,
                        "requested_residual_mps2": float(
                            command.get("requested_residual_mps2", 0.0)
                        ),
                        "executed_residual_mps2": executed_residual,
                        "nominal_acceleration_mps2": float(
                            command.get("nominal_acceleration_mps2", 0.0)
                        ),
                        "desired_acceleration_mps2": float(
                            command.get("desired_acceleration_mps2", 0.0)
                        ),
                        "executed_acceleration_mps2": executed_acceleration,
                        "projected": projected,
                        "projection_reasons": reasons,
                        "treatment_role": vehicle_id in role_ids,
                        "requested_treatment": vehicle_id in treatment_ids,
                        "zero_request_release_tail": (
                            vehicle_id in treated_role_ids
                            and vehicle_id not in treatment_ids
                            and abs(executed_residual) > 1e-12
                        ),
                    }
                )
            command_trace.append({"relative_step": relative_step, "commands": step_commands})
            spatial_trace.append(
                {
                    "relative_step": relative_step,
                    **_spatial_diagnostics(
                        env,
                        role_ids=role_ids,
                        frozen_lane_id=frozen_lane_id,
                        previous_acceleration=previous_acceleration,
                    ),
                }
            )
            completed += 1
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
    finally:
        env.close()
    treatment_role_commands = [
        command
        for step in command_trace
        for command in step["commands"]
        if command["treatment_role"]
    ]
    requested_treatment_commands = [
        command
        for step in command_trace
        for command in step["commands"]
        if command["requested_treatment"]
    ]
    release_tail_commands = [
        command
        for step in command_trace
        for command in step["commands"]
        if command["zero_request_release_tail"]
    ]
    nonrole_commands = [
        command
        for step in command_trace
        for command in step["commands"]
        if not command["treatment_role"]
    ]
    return {
        "schema": SCHEMA,
        "protocol_hash": spec["protocol_hash"],
        "branch_id": spec["branch_id"],
        "case_id": spec["case_id"],
        "arm": spec["arm"],
        "requested_residual_mps2": residual,
        "demand": demand,
        "seed": seed,
        "landmark": decision["landmark"],
        "decision_step": decision_step,
        "absorber_id": absorber_id,
        "decision_selected_ids": selected,
        "decision_observation": local.get(absorber_id),
        "decision_fingerprint": fingerprint,
        "discovery_fingerprint": decision["observation_fingerprint"],
        "decision_identity_with_discovery": fingerprint == decision["observation_fingerprint"],
        "decision_geometry": geometry,
        "completed_horizon_steps": completed,
        "pulse_steps": outcome.pulse_steps,
        "safety_horizon_steps": outcome.safety_horizon_steps,
        "service_horizon_steps": outcome.service_horizon_steps,
        "severe_braking_count_sum": severe_total,
        "acceleration_energy_sum": wave_total,
        "ttc_count_sum": ttc_count_total,
        "low_ttc_exposure_steps": low_ttc_exposure_steps,
        "mean_corridor_delay_rate": float(np.mean(delay_rates)) if delay_rates else None,
        "mean_merge_discharge_rate": (
            float(np.mean(merge_discharge_rates)) if merge_discharge_rates else None
        ),
        "arrived_vehicles": arrivals,
        "collision_observations": collisions,
        "teleport_observations": teleports,
        "reward_sum": reward_total,
        "reward_by_agent": reward_by_agent,
        "reward_component_totals": component_totals,
        "command_trace": command_trace,
        "spatial_trace": spatial_trace,
        "treated_role_vehicle_ids": sorted(treated_role_ids),
        "observed_role_vehicle_ids": sorted(observed_role_ids),
        "treated_role_command_count": len(treatment_role_commands),
        "requested_treatment_command_count": len(requested_treatment_commands),
        "requested_treatment_step_count": sum(
            any(command["requested_treatment"] for command in step["commands"])
            for step in command_trace
        ),
        "zero_request_release_tail_command_count": len(release_tail_commands),
        "zero_request_release_tail_abs_residual_seconds": sum(
            abs(row["executed_residual_mps2"]) for row in release_tail_commands
        ),
        "treated_role_mean_abs_executed_residual_mps2": (
            float(np.mean([abs(row["executed_residual_mps2"]) for row in treatment_role_commands]))
            if treatment_role_commands
            else 0.0
        ),
        "treated_role_projection_count": sum(row["projected"] for row in treatment_role_commands),
        "nonrole_nonzero_executed_residual_count": sum(
            abs(row["executed_residual_mps2"]) > 1e-12 for row in nonrole_commands
        ),
        "unexpected_nonrole_nonzero_executed_residual_count": sum(
            abs(row["executed_residual_mps2"]) > 1e-12
            and not row["zero_request_release_tail"]
            for row in nonrole_commands
        ),
        "treatment_request_violations": treatment_request_violations,
        "runtime_s": time.perf_counter() - started,
    }


def _load_valid(path: Path, protocol_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("protocol_hash") == protocol_hash else None


def _write_progress(output: Path, protocol_hash: str, **counts: Any) -> None:
    atomic_write_json(
        output / "progress.json",
        {
            "schema": SCHEMA,
            "protocol_hash": protocol_hash,
            "updated_unix_s": time.time(),
            **counts,
        },
    )


def _paired_case(
    branches: list[dict[str, Any]], refs: Mapping[str, float], deadband: float
) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in branches}
    zero = by_arm["zero"]
    effects = []
    fingerprints = {row["decision_fingerprint"] for row in branches}
    for arm, action in sorted(by_arm.items()):
        if arm.startswith("zero"):
            continue
        physical = safety_composite(
            zero_severe=zero["severe_braking_count_sum"],
            action_severe=action["severe_braking_count_sum"],
            severe_reference=refs["severe"],
            zero_wave=zero["acceleration_energy_sum"],
            action_wave=action["acceleration_energy_sum"],
            wave_reference=refs["wave"],
            horizon_steps=zero["safety_horizon_steps"],
        )
        delay_delta = (
            None
            if zero["mean_corridor_delay_rate"] is None
            or action["mean_corridor_delay_rate"] is None
            else action["mean_corridor_delay_rate"] - zero["mean_corridor_delay_rate"]
        )
        arrival_delta = int(action["arrived_vehicles"]) - int(zero["arrived_vehicles"])
        low_ttc_excess = int(action["low_ttc_exposure_steps"]) - int(zero["low_ttc_exposure_steps"])
        constraints = {
            "zero_added_collisions": action["collision_observations"]
            <= zero["collision_observations"],
            "zero_added_teleports": action["teleport_observations"]
            <= zero["teleport_observations"],
            "low_ttc_within_budget": low_ttc_excess <= 2,
            "arrival_within_budget": arrival_delta >= -1,
            "delay_within_budget": delay_delta is not None and delay_delta <= 1.0,
            "complete_horizon": action["completed_horizon_steps"]
            == action["service_horizon_steps"],
            "only_anchor_requested": action["treatment_request_violations"] == 0,
            "nonrole_zero_residual": action[
                "unexpected_nonrole_nonzero_executed_residual_count"
            ]
            == 0,
            "pulse_role_dose_at_least_80pct": action["requested_treatment_step_count"]
            >= math.ceil(0.8 * action["pulse_steps"]),
        }
        constraints_pass = all(constraints.values())
        effects.append(
            {
                "arm": arm,
                "requested_residual_mps2": action["requested_residual_mps2"],
                **physical,
                "classification": classify_effect(
                    physical["safety_composite_reference_units"],
                    deadband=deadband,
                    constraints_pass=constraints_pass,
                ),
                "constraints": constraints,
                "constraints_pass": constraints_pass,
                "low_ttc_excess_steps": low_ttc_excess,
                "delay_rate_delta": delay_delta,
                "arrival_delta": arrival_delta,
                "merge_discharge_rate_delta": (
                    action["mean_merge_discharge_rate"] - zero["mean_merge_discharge_rate"]
                ),
                "reward_delta": float(action["reward_sum"]) - float(zero["reward_sum"]),
                "executed_residual_magnitude": action[
                    "treated_role_mean_abs_executed_residual_mps2"
                ],
            }
        )
    admissible = [row for row in effects if row["constraints_pass"]]
    oracle = max(
        [{"arm": "zero", "safety_composite_reference_units": 0.0}, *admissible],
        key=lambda row: (row["safety_composite_reference_units"], row["arm"] == "zero"),
    )
    return {
        "case_id": zero["case_id"],
        "demand": zero["demand"],
        "seed": zero["seed"],
        "landmark": zero["landmark"],
        "decision_step": zero["decision_step"],
        "absorber_id": zero["absorber_id"],
        "all_branch_decision_identity": len(fingerprints) == 1,
        "all_discovery_identity": all(row["decision_identity_with_discovery"] for row in branches),
        "effects": effects,
        "oracle_action": oracle["arm"],
    }


def _zero_replay_effect(branches: list[dict[str, Any]], refs: Mapping[str, float]) -> float | None:
    by_arm = {row["arm"]: row for row in branches}
    if "zero" not in by_arm or "zero_replay" not in by_arm:
        return None
    lhs, rhs = by_arm["zero"], by_arm["zero_replay"]
    return abs(
        safety_composite(
            zero_severe=lhs["severe_braking_count_sum"],
            action_severe=rhs["severe_braking_count_sum"],
            severe_reference=refs["severe"],
            zero_wave=lhs["acceleration_energy_sum"],
            action_wave=rhs["acceleration_energy_sum"],
            wave_reference=refs["wave"],
            horizon_steps=lhs["safety_horizon_steps"],
        )["safety_composite_reference_units"]
    )


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    landmark = LandmarkContract(maximum_landmarks_per_cell=args.max_landmarks_per_cell)
    outcome = OutcomeContract(
        pulse_steps=args.pulse_steps,
        safety_horizon_steps=args.safety_horizon_steps,
        service_horizon_steps=args.service_horizon_steps,
        base_deadband=args.deadband,
    )
    return {
        "schema": SCHEMA,
        "development_only": True,
        "demands": sorted(set(args.demands)),
        "seeds": sorted(set(args.seeds)),
        "residuals": sorted(set(float(value) for value in args.residuals)),
        "landmarks": list(args.landmarks),
        "landmark_contract": landmark.to_dict(),
        "outcome_contract": outcome.to_dict(),
        "max_predecision_steps": args.max_predecision_steps,
        "duplicate_zero_replay": bool(args.duplicate_zero_replay),
        "replay_noise_source": getattr(args, "replay_noise_source_record", None),
        "actor_feature_subsets": {"snapshot": [0, 129], "snapshot_temporal": [0, 133]},
        "forbidden_actor_inputs": ["demand", "seed", "arm", "filename", "future_outcome"],
        "grouping": "complete_seed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--demands", type=int, nargs="+", default=[2200, 2700])
    parser.add_argument(
        "--landmarks", nargs="+", choices=["wave_onset", "severe_onset"], required=True
    )
    parser.add_argument("--residuals", type=float, nargs="+", default=[-0.10, 0.10])
    parser.add_argument("--pulse-steps", type=int, default=10)
    parser.add_argument("--safety-horizon-steps", type=int, default=20)
    parser.add_argument("--service-horizon-steps", type=int, default=60)
    parser.add_argument("--max-landmarks-per-cell", type=int, default=2)
    parser.add_argument("--max-predecision-steps", type=int, default=1200)
    parser.add_argument("--deadband", type=float, default=0.02)
    parser.add_argument("--duplicate-zero-replay", action="store_true")
    parser.add_argument("--replay-noise-source", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if any(seed < 4107 or seed > 4120 for seed in args.seeds):
        raise ValueError("action-local diagnostic is restricted to development seeds 4107--4120")
    if args.service_horizon_steps < args.safety_horizon_steps:
        raise ValueError("service horizon must be at least the safety horizon")
    args.replay_noise_source_record = None
    if args.replay_noise_source is not None:
        source_path = args.replay_noise_source.resolve()
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes.decode("utf-8"))
        if not source.get("smoke_contract_pass"):
            raise ValueError("replay-noise source did not pass its smoke contract")
        args.deadband = float(source["frozen_deadband"])
        args.replay_noise_source_record = {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "frozen_deadband": args.deadband,
        }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "landmarks").mkdir(exist_ok=True)
    (output / "branches").mkdir(exist_ok=True)
    protocol = _protocol(args)
    protocol_hash = _canonical_hash(protocol)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol_hash:
            raise RuntimeError(
                "output contains a different frozen protocol; choose a new output directory"
            )
    else:
        atomic_write_json(
            manifest_path,
            {**protocol, "protocol_hash": protocol_hash, "created_unix_s": time.time()},
        )
    refs_payload = json.loads(
        (ROOT / "runs/eval/idm_phase5_reward_refs_v1.json").read_text(encoding="utf-8")
    )["refs"]
    landmark_rows = []
    discovery_specs = []
    for demand in protocol["demands"]:
        for seed in protocol["seeds"]:
            path = output / "landmarks" / f"m{demand}.seed{seed}.json"
            cached = _load_valid(path, protocol_hash)
            if cached is not None:
                landmark_rows.append(cached)
            else:
                discovery_specs.append(
                    {
                        "demand": demand,
                        "seed": seed,
                        "landmark_contract": protocol["landmark_contract"],
                        "max_steps": protocol["max_predecision_steps"],
                        "output": str(output),
                        "protocol_hash": protocol_hash,
                        "path": str(path),
                    }
                )
    if discovery_specs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(discover_landmarks, spec): spec for spec in discovery_specs}
            try:
                for future in as_completed(futures):
                    row = future.result()
                    spec = futures[future]
                    atomic_write_json(Path(spec["path"]), row)
                    landmark_rows.append(row)
                    print(
                        f"[landmark] m{row['demand']} seed{row['seed']} "
                        f"found={len(row['landmarks'])} runtime={row['runtime_s']:.1f}s",
                        flush=True,
                    )
            except KeyboardInterrupt:
                print("interrupted; completed landmark files are resumable", flush=True)
                raise
    landmark_rows.sort(key=lambda row: (row["demand"], row["seed"]))
    specs = []
    branch_rows = []
    arms = [("zero", 0.0)]
    if args.duplicate_zero_replay:
        arms.append(("zero_replay", 0.0))
    arms.extend((_slug_residual(value), value) for value in protocol["residuals"])
    allowed_landmarks = set(protocol["landmarks"])
    for cell in landmark_rows:
        for decision in cell["landmarks"]:
            if decision["landmark"] not in allowed_landmarks:
                continue
            case_id = (
                f"m{cell['demand']}.seed{cell['seed']}."
                f"{decision['landmark']}.t{decision['decision_step']}"
            )
            for arm, residual in arms:
                branch_id = f"{case_id}.{arm}"
                path = output / "branches" / f"{branch_id}.json"
                cached = _load_valid(path, protocol_hash)
                if cached is not None:
                    branch_rows.append(cached)
                else:
                    specs.append(
                        {
                            "output": str(output),
                            "protocol_hash": protocol_hash,
                            "case_id": case_id,
                            "branch_id": branch_id,
                            "arm": arm,
                            "residual": residual,
                            "demand": cell["demand"],
                            "seed": cell["seed"],
                            "decision": decision,
                            "outcome_contract": protocol["outcome_contract"],
                            "path": str(path),
                        }
                    )
    _write_progress(
        output,
        protocol_hash,
        landmark_cells_complete=len(landmark_rows),
        branch_count_complete=len(branch_rows),
        branch_count_pending=len(specs),
    )
    if specs:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(run_branch, spec): spec for spec in specs}
            try:
                for index, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    spec = futures[future]
                    atomic_write_json(Path(spec["path"]), row)
                    branch_rows.append(row)
                    _write_progress(
                        output,
                        protocol_hash,
                        landmark_cells_complete=len(landmark_rows),
                        branch_count_complete=len(branch_rows),
                        branch_count_pending=len(specs) - index,
                    )
                    print(
                        f"[branch {index}/{len(specs)}] {row['branch_id']} "
                        f"runtime={row['runtime_s']:.1f}s",
                        flush=True,
                    )
            except KeyboardInterrupt:
                print("interrupted; completed branch files are resumable", flush=True)
                raise
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in branch_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    expected_arms = len(arms)
    complete_cases = [rows for rows in by_case.values() if len(rows) == expected_arms]
    replay_effects = []
    for rows in complete_cases:
        demand = int(rows[0]["demand"])
        ref = refs_payload[f"osm.dense.idm_m{demand}r400.sumocfg"]
        noise = _zero_replay_effect(
            rows,
            {
                "severe": ref["corridor_severe_braking_count_ref_per_s"],
                "wave": ref["corridor_acceleration_energy_sum_ref_m2ps4"],
            },
        )
        if noise is not None:
            replay_effects.append(noise)
    frozen_deadband = replay_deadband(args.deadband, replay_effects)
    cases = []
    for rows in complete_cases:
        demand = int(rows[0]["demand"])
        ref = refs_payload[f"osm.dense.idm_m{demand}r400.sumocfg"]
        cases.append(
            _paired_case(
                rows,
                {
                    "severe": ref["corridor_severe_braking_count_ref_per_s"],
                    "wave": ref["corridor_acceleration_energy_sum_ref_m2ps4"],
                },
                frozen_deadband,
            )
        )
    cases.sort(key=lambda row: (row["demand"], row["seed"], row["decision_step"]))
    effects = [effect for case in cases for effect in case["effects"]]
    gates = {
        "all_branch_decision_identity": all(case["all_branch_decision_identity"] for case in cases),
        "all_discovery_identity": all(case["all_discovery_identity"] for case in cases),
        "all_horizons_complete": all(
            row["completed_horizon_steps"] == row["service_horizon_steps"] for row in branch_rows
        ),
        "zero_collision_observations": all(
            row["collision_observations"] == 0 for row in branch_rows
        ),
        "zero_teleport_observations": all(row["teleport_observations"] == 0 for row in branch_rows),
        "only_anchor_requested": all(
            row["treatment_request_violations"] == 0 for row in branch_rows
        ),
        "nonrole_zero_residual": all(
            row["unexpected_nonrole_nonzero_executed_residual_count"] == 0
            for row in branch_rows
        ),
        "treatment_role_dose_at_least_80pct": all(
            row["requested_residual_mps2"] == 0.0
            or row["requested_treatment_step_count"] >= math.ceil(0.8 * row["pulse_steps"])
            for row in branch_rows
        ),
        "treatments_executed": all(
            effect["executed_residual_magnitude"] > 1e-6 for effect in effects
        ),
        "replay_noise_within_base_deadband": all(
            value <= args.deadband for value in replay_effects
        ),
    }
    summary = {
        "schema": SCHEMA,
        "protocol_hash": protocol_hash,
        "development_only": True,
        "resumable": True,
        "resume_instruction": "rerun the identical command; completed atomic artifacts are skipped",
        "protocol": protocol,
        "landmark_cells": landmark_rows,
        "branch_count": len(branch_rows),
        "case_count": len(cases),
        "replay_absolute_safety_composite_effects": replay_effects,
        "frozen_deadband": frozen_deadband,
        "cases": cases,
        "classification_counts": {
            name: sum(effect["classification"] == name for effect in effects)
            for name in ("helpful", "neutral", "harmful")
        },
        "oracle_action_counts": {
            arm: sum(case["oracle_action"] == arm for case in cases)
            for arm, _ in arms
            if arm != "zero_replay"
        },
        "gates": gates,
        "smoke_contract_pass": bool(cases) and all(gates.values()),
        "runtime_s_total_branches": sum(float(row["runtime_s"]) for row in branch_rows),
        "generated_unix_s": time.time(),
    }
    atomic_write_json(output / "summary.json", summary)
    _write_progress(
        output,
        protocol_hash,
        landmark_cells_complete=len(landmark_rows),
        branch_count_complete=len(branch_rows),
        branch_count_pending=0,
        case_count_complete=len(cases),
        complete=True,
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "branch_count": len(branch_rows),
                "frozen_deadband": frozen_deadband,
                "classification_counts": summary["classification_counts"],
                "oracle_action_counts": summary["oracle_action_counts"],
                "gates": gates,
                "smoke_contract_pass": summary["smoke_contract_pass"],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
