"""Capture the frozen v5.4 shadow-observation-v2 development partition.

This driver is deliberately development-only.  It captures one causal control
prefix for each seed in 4401--4520 and launches an exact ten-second treatment
twin only for the treatment-blind Gate-F admissions.  Existing paired outcome
rows are never parsed; their manifests and every development seed file are
bound by byte hash only.

Artifacts are atomic per seed.  Re-running the identical command skips only a
fully validated seed artifact with the frozen contract and source hashes.
Malformed or mismatched artifacts fail closed instead of being overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from config.defaults import DEFAULT_CONFIG
from counterfactual.idm_shadow_observation_v2_contract import (
    CAPTURE_ROOT,
    CAPTURE_SCHEMA,
    MAX_WORKERS,
    PROGRESS_EVERY_SEEDS,
    SEED_SCHEMA,
    SHADOW_HORIZON_S,
    SOURCE_FEASIBILITY_ROOT,
    SOURCE_LABEL_ROOT,
    SOURCE_SHA256,
    contract_manifest,
)
from counterfactual.twin_replay import (
    FrozenPulseSchedule,
    ReplayTrace,
    capture_step,
    run_replay,
)
from env.features import distance_to_merge
from env.wall_supervisor import corridor_state
from experiments import REPO_ROOT
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.wall_v5_4_p30_support_contract import (
    CAV_PENETRATION,
    DECISION_STEPS,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    TREATMENT_STEPS,
    overlay_overrides,
    profile_manifest,
    require_accepted_damping_factory,
)
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from policies.counterfactual_escalation import CounterfactualEscalationPolicy
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


CASE_SCHEMA = "idm_v5_4_shadow_observation_v2_case/1"
PROGRESS_SCHEMA = "idm_v5_4_shadow_observation_v2_progress/1"
SCIENTIFIC_ROLE = "development_only_shadow_observation_repair"
CAPTURE_PHASE = "after_identical_damping_before_wall_override"
DEVELOPMENT_SEEDS = tuple(range(4401, 4521))
FROZEN_DECISIONS = (620, 660, 680, 700)
EXPECTED_DECLARED_CASES = 480
EXPECTED_ADMITTED_SHADOWS = 199

_SOURCE_FILES = {
    "label_manifest": SOURCE_LABEL_ROOT / "manifest.json",
    "label_summary": SOURCE_LABEL_ROOT / "summary.json",
    "feasibility_analysis": SOURCE_FEASIBILITY_ROOT / "analysis.json",
    "v1_diagnosis": Path(
        "runs/eval/idm_wall_v5_4_observation_sufficiency_diagnosis/report.json"
    ),
}


class CaptureIntegrityError(RuntimeError):
    """Raised when a frozen source or resumable artifact fails validation."""


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-compatible deep copy."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    try:
        return _json_safe(value.tolist())
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        return _json_safe(value.item())
    except (AttributeError, TypeError, ValueError):
        return repr(value)


def _sha256_file(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _resolved_source_files(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    return {name: (root / relative).resolve() for name, relative in _SOURCE_FILES.items()}


def verify_frozen_sources(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Verify source bytes without parsing any paired label or outcome row."""
    observed: dict[str, str] = {}
    for name, path in _resolved_source_files(repo_root).items():
        digest = _sha256_file(path)
        expected = str(SOURCE_SHA256[name]).lower()
        if digest.lower() != expected:
            raise CaptureIntegrityError(
                f"frozen source hash mismatch for {name}: {digest} != {expected}"
            )
        observed[name] = digest.lower()
    return observed


def hash_frozen_seed_inventories(
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[str, str]]:
    """Bind every development seed byte without parsing any outcome row.

    Paths are constructed from the frozen partition rather than discovered by
    globbing, so this function cannot wander into either sealed partition.
    """
    root = Path(repo_root).resolve()
    inventories: dict[str, dict[str, str]] = {}
    for role, source_root in (
        ("paired_label_seed_files", SOURCE_LABEL_ROOT),
        ("gate_f_feasibility_seed_files", SOURCE_FEASIBILITY_ROOT),
    ):
        rows: dict[str, str] = {}
        for seed in DEVELOPMENT_SEEDS:
            relative = source_root / "seeds" / f"seed{seed}.json"
            path = (root / relative).resolve()
            if not path.is_file():
                raise CaptureIntegrityError(
                    f"frozen {role} inventory is missing {relative.as_posix()}"
                )
            rows[relative.as_posix()] = _sha256_file(path)
        if len(rows) != len(DEVELOPMENT_SEEDS):
            raise CaptureIntegrityError(f"frozen {role} inventory is incomplete")
        inventories[role] = rows
    return inventories


def _case_id(seed: int, decision_step: int) -> str:
    return f"idm-v54-m2700-p30-seed{int(seed)}-step{int(decision_step)}"


def _feasibility_seed_path(seed: int, repo_root: Path = REPO_ROOT) -> Path:
    return (
        Path(repo_root).resolve()
        / SOURCE_FEASIBILITY_ROOT
        / "seeds"
        / f"seed{int(seed)}.json"
    )


def load_feasibility_rows(seed: int, repo_root: Path = REPO_ROOT) -> dict[int, dict[str, Any]]:
    """Load one treatment-blind Gate-F seed and reject any changed grid/role."""
    if int(seed) not in DEVELOPMENT_SEEDS:
        raise CaptureIntegrityError(f"seed {seed} is outside frozen development partition")
    path = _feasibility_seed_path(seed, repo_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "idm_wall_v5_4_feasibility_seed/1":
        raise CaptureIntegrityError(f"seed {seed} has wrong feasibility schema")
    if int(payload.get("seed", -1)) != int(seed):
        raise CaptureIntegrityError(f"seed {seed} feasibility identity mismatch")
    if payload.get("wall_was_applied") is not False:
        raise CaptureIntegrityError(f"seed {seed} feasibility source applied the wall")
    if payload.get("treatment_outcomes_read") is not False:
        raise CaptureIntegrityError(f"seed {seed} feasibility source is not treatment blind")
    if tuple(int(step) for step in payload.get("decision_steps", ())) != FROZEN_DECISIONS:
        raise CaptureIntegrityError(f"seed {seed} feasibility decision grid changed")
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != len(FROZEN_DECISIONS):
        raise CaptureIntegrityError(f"seed {seed} feasibility rows are incomplete")
    mapped: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CaptureIntegrityError(f"seed {seed} has a non-object feasibility row")
        step = int(row.get("decision_step", -1))
        if step in mapped:
            raise CaptureIntegrityError(f"seed {seed} duplicates decision {step}")
        mapped[step] = deepcopy(row)
    if tuple(sorted(mapped)) != FROZEN_DECISIONS:
        raise CaptureIntegrityError(f"seed {seed} feasibility rows changed order/grid")
    return mapped


def build_capture_manifest(
    source_hashes: Mapping[str, str],
    source_seed_sha256: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build the fully deterministic manifest used for new and resumed runs."""
    contract = contract_manifest()
    if tuple(int(seed) for seed in contract["source_seeds"]) != DEVELOPMENT_SEEDS:
        raise CaptureIntegrityError("contract development partition changed")
    if tuple(int(step) for step in contract["decision_steps"]) != FROZEN_DECISIONS:
        raise CaptureIntegrityError("contract decision grid changed")
    expected_inventory_roles = {
        "paired_label_seed_files",
        "gate_f_feasibility_seed_files",
    }
    if set(source_seed_sha256) != expected_inventory_roles or any(
        len(rows) != len(DEVELOPMENT_SEEDS) for rows in source_seed_sha256.values()
    ):
        raise CaptureIntegrityError("source seed-file inventory is incomplete")
    payload = {
        "schema": CAPTURE_SCHEMA,
        "scientific_role": SCIENTIFIC_ROLE,
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "source_sha256": dict(sorted(source_hashes.items())),
        "source_seed_sha256": {
            role: dict(sorted(rows.items()))
            for role, rows in sorted(source_seed_sha256.items())
        },
        "source_paths": {
            name: relative.as_posix() for name, relative in sorted(_SOURCE_FILES.items())
        },
        "scenario": {
            "hdv_model": "IDM",
            "demand_mainline_vph": DEMAND_MAINLINE_VPH,
            "demand_ramp_vph": DEMAND_RAMP_VPH,
            "cav_penetration": CAV_PENETRATION,
            "decision_steps": list(FROZEN_DECISIONS),
            "shadow_horizon_s": SHADOW_HORIZON_S,
        },
        "base_control_arm": frozen_idm_analytical_manifest(),
        "wall_profile": profile_manifest(),
        "overlay_overrides": _json_safe(overlay_overrides()),
        "capture": {
            "phase": CAPTURE_PHASE,
            "all_decisions": EXPECTED_DECLARED_CASES,
            "treatment_twins_gate_f_only": EXPECTED_ADMITTED_SHADOWS,
            "labels_joined_during_capture": False,
            "paired_outcome_rows_opened": False,
            "new_realized_outcomes": False,
            "atomic_per_seed": True,
            "validated_resume": True,
            "maximum_workers": MAX_WORKERS,
            "progress_every_new_seeds": PROGRESS_EVERY_SEEDS,
        },
    }
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**payload, "manifest_sha256": sha256(encoded).hexdigest()}


def ensure_immutable_manifest(output: Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(output) / "manifest.json"
    expected = _json_safe(manifest)
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CaptureIntegrityError(f"existing manifest is unreadable: {path}") from exc
        if observed != expected:
            raise CaptureIntegrityError("existing capture manifest differs from frozen manifest")
    else:
        atomic_write_json(path, expected)
    return path


def _safe_vehicle_call(call: Callable[..., Any], *args: Any) -> tuple[Any, bool]:
    try:
        return call(*args), True
    except Exception:
        return None, False


def _value_with_mask(call: Callable[..., Any], *args: Any) -> dict[str, Any]:
    value, available = _safe_vehicle_call(call, *args)
    return {"value": _json_safe(value) if available else None, "available": bool(available)}


def _link_record(traci_module: Any, vehicle_id: str, *, follower: bool) -> dict[str, Any]:
    getter = traci_module.vehicle.getFollower if follower else traci_module.vehicle.getLeader
    value, available = _safe_vehicle_call(getter, vehicle_id, 10000.0)
    if not available or not value or not str(value[0]):
        return {
            "internal_vehicle_id": None,
            "net_gap_m": None,
            "relative_speed_mps": None,
            "available": False,
        }
    other_id, gap = str(value[0]), float(value[1])
    ego_speed, ego_ok = _safe_vehicle_call(traci_module.vehicle.getSpeed, vehicle_id)
    other_speed, other_ok = _safe_vehicle_call(traci_module.vehicle.getSpeed, other_id)
    relative = None
    if ego_ok and other_ok:
        relative = (
            float(other_speed) - float(ego_speed)
            if follower
            else float(ego_speed) - float(other_speed)
        )
    return {
        "internal_vehicle_id": other_id,
        "net_gap_m": gap,
        "relative_speed_mps": relative,
        "available": True,
    }


def _vehicle_record(traci_module: Any, vehicle_id: str) -> dict[str, Any]:
    vehicle = traci_module.vehicle
    road = _value_with_mask(vehicle.getRoadID, vehicle_id)
    lane = _value_with_mask(vehicle.getLaneID, vehicle_id)
    lane_index = _value_with_mask(vehicle.getLaneIndex, vehicle_id)
    lane_position = _value_with_mask(vehicle.getLanePosition, vehicle_id)
    distance, distance_ok = _safe_vehicle_call(distance_to_merge, vehicle_id)
    return {
        "internal_vehicle_id": str(vehicle_id),
        "road": road,
        "lane": lane,
        "lane_index": lane_index,
        "lane_position_m": lane_position,
        "distance_to_merge_m": {
            "value": float(distance) if distance_ok else None,
            "available": bool(distance_ok),
        },
        "speed_mps": _value_with_mask(vehicle.getSpeed, vehicle_id),
        "acceleration_mps2": _value_with_mask(vehicle.getAcceleration, vehicle_id),
        "length_m": _value_with_mask(vehicle.getLength, vehicle_id),
        "type": _value_with_mask(vehicle.getTypeID, vehicle_id),
        "route_index": _value_with_mask(vehicle.getRouteIndex, vehicle_id),
        "leader": _link_record(traci_module, vehicle_id, follower=False),
        "follower": _link_record(traci_module, vehicle_id, follower=True),
        "parameters": {
            "tau_s": _value_with_mask(vehicle.getTau, vehicle_id),
            "min_gap_m": _value_with_mask(vehicle.getMinGap, vehicle_id),
            "accel_mps2": _value_with_mask(vehicle.getAccel, vehicle_id),
            "decel_mps2": _value_with_mask(vehicle.getDecel, vehicle_id),
            "emergency_decel_mps2": _value_with_mask(vehicle.getEmergencyDecel, vehicle_id),
            "apparent_decel_mps2": _value_with_mask(vehicle.getApparentDecel, vehicle_id),
            "speed_factor": _value_with_mask(vehicle.getSpeedFactor, vehicle_id),
            "allowed_speed_mps": _value_with_mask(vehicle.getAllowedSpeed, vehicle_id),
            "max_speed_mps": _value_with_mask(vehicle.getMaxSpeed, vehicle_id),
            "action_step_length_s": _value_with_mask(vehicle.getActionStepLength, vehicle_id),
            "speed_mode": _value_with_mask(vehicle.getSpeedMode, vehicle_id),
            "lane_change_mode": _value_with_mask(vehicle.getLaneChangeMode, vehicle_id),
        },
    }


def _rank_reserves(
    policy: CounterfactualEscalationPolicy,
    active_cav_ids: Sequence[str],
    feasibility_row: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    formation = feasibility_row.get("formation") or {}
    primary_ids = {str(value) for value in formation.get("controlled_cav_ids", ())}
    lanes = tuple(str(value) for value in formation.get("required_lane_ids", ()))
    primary_distance: dict[str, list[float]] = {lane: [] for lane in lanes}
    for member_id in sorted(primary_ids):
        lane_id, lane_ok = _safe_vehicle_call(policy._lane_id, member_id)
        distance, distance_ok = _safe_vehicle_call(policy._distance_in_band, member_id)
        if lane_ok and distance_ok and lane_id in primary_distance and distance is not None:
            primary_distance[str(lane_id)].append(float(distance))
    midpoint = 0.5 * (
        float(policy.wall_params["d_lo"]) + float(policy.wall_params["d_hi"])
    )
    references = {
        lane: (sum(values) / len(values) if values else midpoint)
        for lane, values in primary_distance.items()
    }
    ranked: dict[str, list[tuple[float, str, float]]] = {lane: [] for lane in lanes}
    live = policy._live_ids()
    for cav_id in sorted({str(value) for value in active_cav_ids} - primary_ids):
        if cav_id not in live:
            continue
        lane_id, lane_ok = _safe_vehicle_call(policy._lane_id, cav_id)
        distance, distance_ok = _safe_vehicle_call(policy._distance_in_band, cav_id)
        if (
            not lane_ok
            or not distance_ok
            or lane_id not in ranked
            or distance is None
        ):
            continue
        distance_value = float(distance)
        ranked[str(lane_id)].append(
            (abs(distance_value - references[str(lane_id)]), cav_id, distance_value)
        )
    return {
        lane: [
            {
                "internal_vehicle_id": cav_id,
                "reserve_rank": rank,
                "distance_to_merge_m": distance,
                "distance_from_primary_reference_m": delta,
            }
            for rank, (delta, cav_id, distance) in enumerate(sorted(values), 1)
        ]
        for lane, values in sorted(ranked.items())
    }


def capture_live_predecision_record(
    traci_module: Any,
    policy: CounterfactualEscalationPolicy,
    *,
    decision_step: int,
    obs: Mapping[str, Any],
    action_dict: Mapping[str, Any],
    feasibility_row: Mapping[str, Any],
    inflow_tracker: Mapping[str, Any],
) -> dict[str, Any]:
    """Deeply capture the exact state after damping and before wall override."""
    live_ids = tuple(sorted(str(value) for value in traci_module.vehicle.getIDList()))
    audit = deepcopy(policy.last_audit or {})
    if int(audit.get("command", -1)) != 0 or bool(audit.get("wall_active")):
        raise CaptureIntegrityError("predecision capture was not damping-only")
    formation = deepcopy(feasibility_row.get("formation") or {})
    return _json_safe(
        {
            "schema": "idm_v5_4_shadow_observation_v2_predecision_live/1",
            "capture_phase": CAPTURE_PHASE,
            "decision_step": int(decision_step),
            "vehicle_selection": "all_live_corridor_vehicles_internal_linkage_only",
            "vehicles": [_vehicle_record(traci_module, vehicle_id) for vehicle_id in live_ids],
            "active_cav_internal_ids": sorted(str(value) for value in obs),
            "controlled_cav_internal_ids": sorted(
                str(value) for value in audit.get("base_control_ids", ())
            ),
            "damping": {
                "action_dict_by_internal_id": dict(action_dict),
                "base_actions_by_internal_id": deepcopy(audit.get("base_actions", {})),
                "projection_diagnostics": deepcopy(audit.get("base_diagnostics", {})),
            },
            "proposed_wall": {
                "member_internal_ids": sorted(
                    str(value) for value in formation.get("controlled_cav_ids", ())
                ),
                "required_lane_ids": sorted(
                    str(value) for value in formation.get("required_lane_ids", ())
                ),
                "ranked_same_lane_reserves": _rank_reserves(
                    policy, tuple(str(value) for value in obs), feasibility_row
                ),
                "release_boundary_m": float(policy.compact_release_boundary_m),
                "formation_preflight": formation,
                "authority_safety_preflight": {
                    "lane_coverage": deepcopy(feasibility_row.get("lane_coverage")),
                    "projected_dose": deepcopy(feasibility_row.get("projected_dose")),
                    "activation_safety": deepcopy(feasibility_row.get("activation_safety")),
                },
            },
            # This is the already-mutated causal tracker from the real control
            # loop.  Capture never calls corridor_state a second time.
            "causal_detector_history": deepcopy(inflow_tracker),
            "internal_identifiers_are_linkage_metadata_only": True,
        }
    )


def _control_env_config(seed: int, sumo_binary: str) -> dict[str, Any]:
    total_steps = max(FROZEN_DECISIONS) + SHADOW_HORIZON_S
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "project_root": SCENARIO_ROOT,
            "sumo_binary": str(sumo_binary),
            "sumo_config": transition_scenario_config(DEMAND_MAINLINE_VPH).name,
            "sumo_seed": int(seed),
            "cav_penetration": CAV_PENETRATION,
            "sumo_step_length": 0.5,
            "max_episode_steps": total_steps + 5,
            "eval_mode": False,
            "control_mode": "shepherd_hybrid",
            "reward_mode": "shepherd",
            "centralized_critic": True,
            "use_v2_obs": True,
            "shepherd_modes": ["coast", "recover", "guard"],
            "shepherd_disabled_modes": [],
            "max_neighbors": 20,
            "cav_warmup_steps": 400,
            "max_cavs_critic": 32,
            "local_obs_dim": 124,
            "action_masking": True,
            "use_meter_load": False,
            "use_cav_count": False,
            "use_health_obs": False,
            "use_dv_obs": False,
            "use_margin_obs": False,
            "feasibility_mask": False,
            "feasibility_lmin_a": 0.0,
            "feasibility_lmin_b": 0.0,
            "fairness_r_ref": 0.45,
            "tail_r_ref": 0.53,
            "flow_q_ref": 0.95,
            "flow_v_ref": 17.5,
        }
    )
    cfg.update(supervisor_overrides())
    return cfg


def run_control_capture(
    *,
    seed: int,
    sumo_binary: str,
    feasibility_rows: Mapping[int, Mapping[str, Any]],
) -> tuple[ReplayTrace, dict[int, dict[str, Any]]]:
    """Run one causal damping prefix and capture live pre-wall decision records."""
    import traci
    from env.core import TrafficShepherdMultiAgentEnv

    factory = require_accepted_damping_factory(build_frozen_idm_analytical_policy)
    env = TrafficShepherdMultiAgentEnv(_control_env_config(seed, sumo_binary))
    policy = CounterfactualEscalationPolicy(
        base_policy=factory(), **dict(overlay_overrides())
    )
    snapshots = []
    audits = []
    records: dict[int, dict[str, Any]] = {}
    previous_health: float | None = None
    health_history: list[float] = []
    inflow_tracker: dict[str, Any] = {}
    total_steps = max(FROZEN_DECISIONS) + SHADOW_HORIZON_S
    terminated_early = False
    try:
        obs, reset_info = env.reset(seed=seed)
        policy.reset(env, seed=seed)
        supervisor_observation = corridor_state(
            env, None, 0, 0, 0, inflow_tracker, health_history
        )
        snapshots.append(
            capture_step(
                traci,
                relative_step=0,
                action=0,
                supervisor_observation=supervisor_observation,
                active_cav_ids=obs.keys(),
                digits=6,
            )
        )
        audits.append({"relative_step": 0, "command": 0, "command_name": "DAMPING_ONLY"})
        for relative_step in range(1, total_steps + 1):
            policy.set_command(0)
            action_dict = policy.act(obs)
            audit = _json_safe(policy.last_audit)
            audit["relative_step"] = relative_step
            audits.append(audit)
            declared_decision = relative_step - 1
            if declared_decision in feasibility_rows:
                records[declared_decision] = capture_live_predecision_record(
                    traci,
                    policy,
                    decision_step=declared_decision,
                    obs=obs,
                    action_dict=action_dict,
                    feasibility_row=feasibility_rows[declared_decision],
                    inflow_tracker=inflow_tracker,
                )
            obs, _rewards, terminated, truncated, _infos = env.step(action_dict)
            supervisor_observation = corridor_state(
                env,
                previous_health,
                0,
                0,
                relative_step,
                inflow_tracker,
                health_history,
            )
            previous_health = float(supervisor_observation[0])
            health_history.append(previous_health)
            snapshots.append(
                capture_step(
                    traci,
                    relative_step=relative_step,
                    action=0,
                    supervisor_observation=supervisor_observation,
                    active_cav_ids=obs.keys(),
                    digits=6,
                )
            )
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                terminated_early = relative_step < total_steps
                break
        if tuple(sorted(records)) != FROZEN_DECISIONS:
            raise CaptureIntegrityError(f"seed {seed} did not capture all decision records")
        clean_reset = _json_safe(reset_info)
        if not isinstance(clean_reset, dict):
            clean_reset = {"value": clean_reset}
        clean_reset.setdefault("sumo_config", transition_scenario_config(DEMAND_MAINLINE_VPH).name)
        clean_reset.setdefault("paired_seed", int(seed))
        trace = ReplayTrace(
            seed=int(seed),
            decision_step=max(FROZEN_DECISIONS),
            horizon_steps=SHADOW_HORIZON_S,
            prefix_action=0,
            post_action=0,
            treatment_steps=None,
            reset_info=clean_reset,
            snapshots=tuple(snapshots),
            terminated_early=terminated_early,
            controller_metadata={
                "scientific_role": SCIENTIFIC_ROLE,
                "capture_phase": CAPTURE_PHASE,
                "base_control_arm": frozen_idm_analytical_manifest(),
            },
            controller_audits=tuple(audits),
        )
        if trace.completed_steps < total_steps or trace.terminated_early:
            raise CaptureIntegrityError(f"seed {seed} control trace ended before shadow horizon")
        return trace, records
    finally:
        try:
            policy.close()
        finally:
            env.close()


def run_treatment_twin(*, seed: int, decision_step: int, sumo_binary: str) -> ReplayTrace:
    factory = require_accepted_damping_factory(build_frozen_idm_analytical_policy)

    def fixed_horizon_environment_factory(config: Mapping[str, Any]):
        """Keep episode-progress observations identical to the shared control.

        ``run_replay`` normally derives ``max_episode_steps`` from each
        decision-specific trace length.  That makes an otherwise identical
        prefix expose a different episode-progress observation at decisions
        620/660/680.  The shadow contract instead freezes the common 710-step
        planning episode and merely stops recording each treatment twin after
        ten postdecision seconds.
        """
        from env.core import TrafficShepherdMultiAgentEnv

        frozen = dict(config)
        frozen["max_episode_steps"] = max(FROZEN_DECISIONS) + SHADOW_HORIZON_S + 5
        return TrafficShepherdMultiAgentEnv(frozen)

    trace = run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=str(sumo_binary),
        sumo_config=transition_scenario_config(DEMAND_MAINLINE_VPH).name,
        seed=int(seed),
        decision_step=int(decision_step),
        horizon_steps=SHADOW_HORIZON_S,
        treatment_steps=TREATMENT_STEPS,
        cav_penetration=CAV_PENETRATION,
        sumo_step_length=0.5,
        supervisor_overrides=supervisor_overrides(),
        digits=6,
        base_policy_factory=factory,
        intervention_schedule=FrozenPulseSchedule(decision_step, TREATMENT_STEPS),
        overlay_overrides=overlay_overrides(),
        environment_factory=fixed_horizon_environment_factory,
        controller_metadata={
            "scientific_role": SCIENTIFIC_ROLE,
            "capture_phase": CAPTURE_PHASE,
            "wall_profile": profile_manifest(),
        },
    )
    if trace.completed_steps < decision_step + SHADOW_HORIZON_S or trace.terminated_early:
        raise CaptureIntegrityError(
            f"seed {seed} decision {decision_step} treatment twin ended early"
        )
    return trace


def _prefix_identity(
    control: ReplayTrace, treatment: ReplayTrace, decision_step: int
) -> dict[str, Any]:
    if len(control.snapshots) <= decision_step or len(treatment.snapshots) <= decision_step:
        return {"passed": False, "reason": "missing_decision_snapshot"}
    first_mismatch = None
    for left, right in zip(
        control.snapshots[: decision_step + 1],
        treatment.snapshots[: decision_step + 1],
    ):
        if left.digest != right.digest:
            first_mismatch = int(left.relative_step)
            break
    return {
        "passed": first_mismatch is None,
        "prefix_matches_through_step": decision_step if first_mismatch is None else first_mismatch - 1,
        "first_mismatch_step": first_mismatch,
        "control_decision_digest": control.snapshots[decision_step].digest,
        "treatment_decision_digest": treatment.snapshots[decision_step].digest,
    }


def _load_shadow_helpers():
    from counterfactual.idm_shadow_observation_v2 import (
        capture_predecision_record,
        shadow_observation_features,
        validate_model_feature_names,
    )

    return capture_predecision_record, shadow_observation_features, validate_model_feature_names


def control_trace_view(control: ReplayTrace, decision_step: int) -> ReplayTrace:
    """Return the exact shared-control slice with the requested decision metadata."""
    terminal_step = int(decision_step) + SHADOW_HORIZON_S
    snapshots = tuple(
        row for row in control.snapshots if int(row.relative_step) <= terminal_step
    )
    audits = tuple(
        row
        for row in control.controller_audits
        if int(row.get("relative_step", -1)) <= terminal_step
    )
    if not snapshots or int(snapshots[-1].relative_step) != terminal_step:
        raise CaptureIntegrityError(
            f"shared control trace does not reach decision {decision_step} shadow horizon"
        )
    return ReplayTrace(
        seed=control.seed,
        decision_step=int(decision_step),
        horizon_steps=SHADOW_HORIZON_S,
        prefix_action=0,
        post_action=0,
        treatment_steps=None,
        reset_info=control.reset_info,
        snapshots=snapshots,
        terminated_early=False,
        controller_metadata={
            **dict(control.controller_metadata),
            "decision_step": int(decision_step),
            "shared_control_trace_view": True,
        },
        controller_audits=audits,
    )


def run_seed(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Capture one complete atomic seed artifact."""
    seed = int(spec["seed"])
    output = Path(str(spec["output"]))
    manifest = dict(spec["manifest"])
    result_path = output / "seeds" / f"seed{seed}.json"
    if load_valid_seed(result_path, seed, manifest) is not None:
        return {"status": "skipped", "seed": seed, "path": str(result_path)}

    started = time.time()
    feasibility = load_feasibility_rows(seed)
    capture_predecision_record, shadow_observation_features, validate_names = (
        _load_shadow_helpers()
    )
    control, live_records = run_control_capture(
        seed=seed,
        sumo_binary=str(spec["sumo_binary"]),
        feasibility_rows=feasibility,
    )
    cases = []
    for decision_step in FROZEN_DECISIONS:
        scan = feasibility[decision_step]
        admitted = bool(scan.get("admitted"))
        decision_control = control_trace_view(control, decision_step)
        baseline = capture_predecision_record(
            decision_control,
            decision_step=decision_step,
            feasibility_row=deepcopy(scan),
            live_predecision_record=deepcopy(live_records[decision_step]),
        )
        validate_names(baseline.get("model_features") or {})
        shadow = None
        treatment_digest = None
        prefix = {
            "passed": True,
            "not_replayed_reason": "deterministic_gate_f_veto",
            "control_decision_digest": control.snapshots[decision_step].digest,
            "treatment_decision_digest": None,
        }
        if admitted:
            treatment = run_treatment_twin(
                seed=seed,
                decision_step=decision_step,
                sumo_binary=str(spec["sumo_binary"]),
            )
            prefix = _prefix_identity(control, treatment, decision_step)
            if prefix.get("passed") is not True:
                raise CaptureIntegrityError(
                    f"seed {seed} decision {decision_step} twin prefix mismatch"
                )
            treatment_digest = treatment.trace_digest
            shadow = shadow_observation_features(
                decision_control,
                treatment,
                decision_step=decision_step,
                feasibility_row=deepcopy(scan),
            )
            validate_names(shadow.get("model_features") or {})
        cases.append(
            {
                "schema": CASE_SCHEMA,
                "case_id": _case_id(seed, decision_step),
                "seed": seed,
                "decision_step": decision_step,
                "gate_f_admitted": admitted,
                "capture_phase": CAPTURE_PHASE,
                "predecision_record": live_records[decision_step],
                "baseline_observation": baseline,
                "shadow_observation": shadow,
                "integrity": {
                    "feasibility_source_treatment_blind": True,
                    "control_decision_digest": control.snapshots[decision_step].digest,
                    "treatment_trace_digest": treatment_digest,
                    "prefix_identity": prefix,
                    "prefix_identity_passed": bool(prefix.get("passed")),
                    "shadow_generated": shadow is not None,
                },
            }
        )
    payload = {
        "schema": SEED_SCHEMA,
        "scientific_role": SCIENTIFIC_ROLE,
        "manifest_sha256": manifest["manifest_sha256"],
        "contract_sha256": manifest["contract_sha256"],
        "source_sha256": manifest["source_sha256"],
        "seed": seed,
        "decision_steps": list(FROZEN_DECISIONS),
        "cases": cases,
        "admitted_shadow_count": sum(int(case["gate_f_admitted"]) for case in cases),
        "control_trace": {
            "decision_step": control.decision_step,
            "horizon_steps": control.horizon_steps,
            "trace_digest": control.trace_digest,
            "terminated_early": control.terminated_early,
        },
        "labels_joined_during_capture": False,
        "paired_outcome_rows_opened": False,
        "new_realized_outcomes": False,
        "wall_clock_seconds": time.time() - started,
    }
    validate_seed_artifact(payload, seed, manifest)
    atomic_write_json(result_path, payload)
    return {
        "status": "completed",
        "seed": seed,
        "path": str(result_path),
        "admitted_shadows": payload["admitted_shadow_count"],
        "wall_clock_seconds": payload["wall_clock_seconds"],
    }


def validate_seed_artifact(
    payload: Mapping[str, Any], seed: int, manifest: Mapping[str, Any]
) -> None:
    if payload.get("schema") != SEED_SCHEMA:
        raise CaptureIntegrityError(f"seed {seed} artifact schema mismatch")
    if payload.get("scientific_role") != SCIENTIFIC_ROLE:
        raise CaptureIntegrityError(f"seed {seed} artifact scientific role mismatch")
    if int(payload.get("seed", -1)) != int(seed):
        raise CaptureIntegrityError(f"seed {seed} artifact identity mismatch")
    if payload.get("contract_sha256") != manifest.get("contract_sha256"):
        raise CaptureIntegrityError(f"seed {seed} artifact contract hash mismatch")
    if payload.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise CaptureIntegrityError(f"seed {seed} artifact manifest hash mismatch")
    if payload.get("source_sha256") != manifest.get("source_sha256"):
        raise CaptureIntegrityError(f"seed {seed} artifact source hashes mismatch")
    if tuple(int(value) for value in payload.get("decision_steps", ())) != FROZEN_DECISIONS:
        raise CaptureIntegrityError(f"seed {seed} artifact decision grid mismatch")
    if payload.get("labels_joined_during_capture") is not False:
        raise CaptureIntegrityError(f"seed {seed} capture contains joined labels")
    if payload.get("paired_outcome_rows_opened") is not False:
        raise CaptureIntegrityError(f"seed {seed} capture opened outcome rows")
    if payload.get("new_realized_outcomes") is not False:
        raise CaptureIntegrityError(f"seed {seed} capture contains new realized outcomes")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != len(FROZEN_DECISIONS):
        raise CaptureIntegrityError(f"seed {seed} artifact must contain four cases")
    expected_ids = [_case_id(seed, step) for step in FROZEN_DECISIONS]
    if [str(case.get("case_id")) for case in cases] != expected_ids:
        raise CaptureIntegrityError(f"seed {seed} artifact case IDs mismatch")
    _, _, validate_names = _load_shadow_helpers()
    admitted = 0
    for case, decision in zip(cases, FROZEN_DECISIONS):
        if case.get("schema") != CASE_SCHEMA or int(case.get("decision_step", -1)) != decision:
            raise CaptureIntegrityError(f"seed {seed} case schema/decision mismatch")
        if int(case.get("seed", -1)) != seed or case.get("capture_phase") != CAPTURE_PHASE:
            raise CaptureIntegrityError(f"seed {seed} case metadata mismatch")
        predecision = case.get("predecision_record")
        if not isinstance(predecision, Mapping):
            raise CaptureIntegrityError(f"seed {seed} decision {decision} lacks live capture")
        if (
            predecision.get("schema")
            != "idm_v5_4_shadow_observation_v2_predecision_live/1"
            or predecision.get("capture_phase") != CAPTURE_PHASE
            or int(predecision.get("decision_step", -1)) != decision
        ):
            raise CaptureIntegrityError(
                f"seed {seed} decision {decision} live capture metadata mismatch"
            )
        baseline = case.get("baseline_observation")
        if not isinstance(baseline, Mapping):
            raise CaptureIntegrityError(f"seed {seed} decision {decision} lacks baseline")
        if not isinstance(case.get("gate_f_admitted"), bool):
            raise CaptureIntegrityError(
                f"seed {seed} decision {decision} Gate-F flag is not Boolean"
            )
        gate_f = bool(case["gate_f_admitted"])
        shadow = case.get("shadow_observation")
        if gate_f != (shadow is not None):
            raise CaptureIntegrityError(
                f"seed {seed} decision {decision} shadow/admission mismatch"
            )
        integrity = case.get("integrity") or {}
        if integrity.get("prefix_identity_passed") is not True:
            raise CaptureIntegrityError(f"seed {seed} decision {decision} prefix failed")
        validate_names(baseline.get("model_features") or {})
        if isinstance(shadow, Mapping):
            validate_names(shadow.get("model_features") or {})
        admitted += int(gate_f)
    if int(payload.get("admitted_shadow_count", -1)) != admitted:
        raise CaptureIntegrityError(f"seed {seed} admitted shadow count mismatch")


def load_valid_seed(path: Path, seed: int, manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureIntegrityError(f"existing seed artifact is unreadable: {path}") from exc
    validate_seed_artifact(payload, seed, manifest)
    return payload


def write_progress(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    valid_complete: int,
    newly_completed: int,
    pending: int,
    failed: int = 0,
) -> None:
    atomic_write_json(
        Path(output) / "progress.json",
        {
            "schema": PROGRESS_SCHEMA,
            "manifest_sha256": manifest["manifest_sha256"],
            "contract_sha256": manifest["contract_sha256"],
            "source_sha256": manifest["source_sha256"],
            "updated_unix_s": time.time(),
            "total_seed_count": len(DEVELOPMENT_SEEDS),
            "valid_complete_seed_count": int(valid_complete),
            "newly_completed_seed_count": int(newly_completed),
            "pending_seed_count": int(pending),
            "failed_seed_count": int(failed),
        },
    )


def _run_specs(
    specs: Sequence[Mapping[str, Any]], workers: int
) -> list[dict[str, Any]]:
    if not specs:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_seed, spec): spec for spec in specs}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / CAPTURE_ROOT)
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=4)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if tuple(int(value) for value in DECISION_STEPS) != FROZEN_DECISIONS:
        raise CaptureIntegrityError("runtime treatment decision grid changed")
    source_hashes = verify_frozen_sources(REPO_ROOT)
    source_seed_sha256 = hash_frozen_seed_inventories(REPO_ROOT)
    manifest = build_capture_manifest(source_hashes, source_seed_sha256)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "seeds").mkdir(exist_ok=True)
    ensure_immutable_manifest(output, manifest)

    valid: dict[int, dict[str, Any]] = {}
    pending: list[int] = []
    for seed in DEVELOPMENT_SEEDS:
        path = output / "seeds" / f"seed{seed}.json"
        payload = load_valid_seed(path, seed, manifest)
        if payload is None:
            pending.append(seed)
        else:
            valid[seed] = payload
    write_progress(
        output,
        manifest,
        valid_complete=len(valid),
        newly_completed=0,
        pending=len(pending),
    )
    print(
        json.dumps(
            {"status": "resume_audit", "valid": len(valid), "pending": len(pending)},
            sort_keys=True,
        ),
        flush=True,
    )

    common = {
        "output": str(output),
        "sumo_binary": args.sumo_binary,
        "manifest": manifest,
    }
    new_count = 0

    # The frozen protocol requires a four-seed timing benchmark before scale-up.
    benchmark_seeds = [seed for seed in pending if seed in DEVELOPMENT_SEEDS[:4]]
    if benchmark_seeds:
        benchmark_results = _run_specs(
            [{**common, "seed": seed} for seed in benchmark_seeds], args.workers
        )
        for result in sorted(benchmark_results, key=lambda row: int(row["seed"])):
            seed = int(result["seed"])
            valid[seed] = load_valid_seed(
                output / "seeds" / f"seed{seed}.json", seed, manifest
            ) or {}
            new_count += int(result["status"] == "completed")
        pending = [seed for seed in pending if seed not in benchmark_seeds]
        write_progress(
            output,
            manifest,
            valid_complete=len(valid),
            newly_completed=new_count,
            pending=len(pending),
        )
        print(
            json.dumps(
                {
                    "status": "four_seed_benchmark_complete",
                    "seeds": benchmark_seeds,
                    "runtime_s": [row.get("wall_clock_seconds") for row in benchmark_results],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if pending:
        specs = [{**common, "seed": seed} for seed in pending]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_seed, spec): spec for spec in specs}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    seed = int(result["seed"])
                    valid[seed] = load_valid_seed(
                        output / "seeds" / f"seed{seed}.json", seed, manifest
                    ) or {}
                    if result["status"] == "completed":
                        new_count += 1
                    remaining = len(DEVELOPMENT_SEEDS) - len(valid)
                    if new_count % PROGRESS_EVERY_SEEDS == 0 or remaining == 0:
                        write_progress(
                            output,
                            manifest,
                            valid_complete=len(valid),
                            newly_completed=new_count,
                            pending=remaining,
                        )
                        print(
                            json.dumps(
                                {
                                    "status": "progress",
                                    "valid": len(valid),
                                    "new": new_count,
                                    "pending": remaining,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
            except KeyboardInterrupt:
                write_progress(
                    output,
                    manifest,
                    valid_complete=len(valid),
                    newly_completed=new_count,
                    pending=len(DEVELOPMENT_SEEDS) - len(valid),
                )
                print("interrupted; completed atomic seed files are resumable", flush=True)
                raise

    if set(valid) != set(DEVELOPMENT_SEEDS):
        raise CaptureIntegrityError("capture ended without all development seed artifacts")
    declared = sum(len(payload["cases"]) for payload in valid.values())
    admitted = sum(int(payload["admitted_shadow_count"]) for payload in valid.values())
    if declared != EXPECTED_DECLARED_CASES or admitted != EXPECTED_ADMITTED_SHADOWS:
        raise CaptureIntegrityError(
            f"frozen capture counts changed: declared={declared}, admitted={admitted}"
        )
    summary = {
        "schema": CAPTURE_SCHEMA,
        "scientific_role": SCIENTIFIC_ROLE,
        "manifest_sha256": manifest["manifest_sha256"],
        "contract_sha256": manifest["contract_sha256"],
        "source_sha256": manifest["source_sha256"],
        "seed_count": len(valid),
        "declared_case_count": declared,
        "gate_f_admitted_shadow_count": admitted,
        "all_seed_artifacts_valid": True,
        "labels_joined_during_capture": False,
        "paired_outcome_rows_opened": False,
        "new_realized_outcomes": False,
        "analysis_authorized": True,
        "validation_or_final_outcomes_read": False,
    }
    atomic_write_json(output / "summary.json", summary)
    write_progress(
        output,
        manifest,
        valid_complete=len(valid),
        newly_completed=new_count,
        pending=0,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
