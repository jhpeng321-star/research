"""Resumable parallel authority screen for structured IDM damping actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import traci

from env.core import TrafficShepherdMultiAgentEnv
from env.snapshot import get_active_snapshot
from experiments import REPO_ROOT
from experiments.idm.run_idm_residual_checkpoint_eval import _env_config
from experiments.io import atomic_write_json
from idm.action_local import LandmarkContract, classify_effect, replay_deadband, safety_composite
from idm.spatial_response import capture_spatial_response
from idm.structured_actions import MODE_TO_GAIN, nominal_action
from tools.audit_idm_phase5_decision_state_sufficiency import (
    _canonical_hash,
    _local_observations,
    _selected_ids,
    _selected_ids_in_lane,
    discover_landmarks,
)

ROOT = REPO_ROOT
SCHEMA = "idm_structured_authority_screen/1"


def _parse_cell(value: str) -> tuple[int, tuple[int, ...]]:
    demand_text, separator, seed_text = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("cell must be DEMAND:SEED,SEED,...")
    try:
        demand = int(demand_text)
        seeds = tuple(int(seed) for seed in seed_text.split(",") if seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid cell {value!r}") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("cell requires at least one seed")
    return demand, seeds


def _arm_slug(family: str, value: float) -> str:
    sign = "p" if value >= 0 else "n"
    return f"{family}.{sign}{abs(value):.3f}".replace(".", "_")


def _arms(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms = [
        {"arm": "nominal", "family": "gain", "value": 1.0, "is_control": True},
        {
            "arm": "nominal_replay",
            "family": "gain",
            "value": 1.0,
            "is_control": True,
        },
    ]
    arms.extend(
        {
            "arm": _arm_slug("gain", float(value)),
            "family": "gain",
            "value": float(value),
            "is_control": False,
        }
        for value in protocol["gain_values"]
        if abs(float(value) - 1.0) > 1e-12
    )
    arms.extend(
        {
            "arm": _arm_slug("reference_offset", float(value)),
            "family": "reference_offset",
            "value": float(value),
            "is_control": False,
        }
        for value in protocol["reference_offsets_mps"]
        if abs(float(value)) > 1e-12
    )
    return arms


def _state_fingerprint(decision: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {
            "step": int(decision["decision_step"]),
            "landmark": decision["landmark"],
            "absorber_id": decision["absorber_id"],
            "selected_ids": decision["selected_ids"],
            "observation": decision["observation"],
        }
    )


def _branch_fingerprint(
    decision: Mapping[str, Any], local: Mapping[str, list[float]], selected: list[str]
) -> str:
    return _canonical_hash(
        {
            "step": int(decision["decision_step"]),
            "landmark": decision["landmark"],
            "absorber_id": decision["absorber_id"],
            "selected_ids": selected,
            "observation": local.get(str(decision["absorber_id"])),
        }
    )


def _structured_env(spec: Mapping[str, Any]) -> TrafficShepherdMultiAgentEnv:
    output = Path(spec["output"])
    max_steps = int(spec["decision"]["decision_step"]) + int(spec["service_horizon_steps"]) + 5
    cfg = _env_config(
        int(spec["demand"]),
        int(spec["seed"]),
        output / "artifacts",
        spec["branch_id"],
        max_steps,
        None,
    )
    cfg.update(
        eval_mode=False,
        eval_fcd_output=None,
        eval_tripinfo_output=None,
        eval_stats_output=None,
        idm_residual_action_family=spec["family"],
        idm_residual_structured_action_filter_tau_s=float(spec["action_filter_tau_s"]),
    )
    return TrafficShepherdMultiAgentEnv(cfg)


def _actions(
    obs: Mapping[str, np.ndarray],
    *,
    family: str,
    treatment_ids: set[str],
    treatment_value: float,
) -> dict[str, np.ndarray]:
    baseline = nominal_action(family)
    return {
        agent_id: np.asarray(
            [treatment_value if agent_id in treatment_ids else baseline], dtype=np.float32
        )
        for agent_id in obs
    }


def _snapshot_acceleration() -> dict[str, float]:
    snapshot = get_active_snapshot()
    if snapshot is None:
        return {}
    return {
        vehicle_id: float(acceleration)
        for vehicle_id in snapshot.ordered_ids()
        for acceleration in [snapshot.accel(vehicle_id)]
        if acceleration is not None
    }


def _merge_discharge_step(env: TrafficShepherdMultiAgentEnv, previous_road: dict[str, str]) -> int:
    snapshot = get_active_snapshot()
    if snapshot is None:
        return 0
    current = {
        vehicle_id: str(road)
        for vehicle_id in snapshot.ordered_ids()
        for road in [snapshot.road(vehicle_id)]
        if road is not None
    }
    count = sum(
        old_road == env._merge_edge and current.get(vehicle_id) != env._merge_edge
        for vehicle_id, old_road in previous_road.items()
    )
    previous_road.clear()
    previous_road.update(current)
    return int(count)


def run_branch(spec: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    env = _structured_env(spec)
    decision = spec["decision"]
    family = str(spec["family"])
    value = float(spec["value"])
    baseline = nominal_action(family)
    decision_step = int(decision["decision_step"])
    pulse_steps = int(spec["pulse_steps"])
    safety_horizon = int(spec["safety_horizon_steps"])
    service_horizon = int(spec["service_horizon_steps"])
    lane_id = str(decision["geometry"]["lane_id"])
    severe_total = wave_total = ttc_total = 0.0
    low_ttc_steps = arrivals = collisions = teleports = discharge = 0
    collapse_run = collapse_max = 0
    delay_rates: list[float] = []
    command_rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    action_request_violations = 0
    realized_action_steps: set[int] = set()
    fingerprint = ""
    discovery_fingerprint = _state_fingerprint(decision)
    completed = 0
    try:
        obs, _ = env.reset(seed=int(spec["seed"]))
        for _ in range(decision_step):
            obs, _, terminated, truncated, _ = env.step(
                _actions(
                    obs,
                    family=family,
                    treatment_ids=set(),
                    treatment_value=baseline,
                )
            )
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        local = _local_observations(obs)
        selected = _selected_ids(local)
        fingerprint = _branch_fingerprint(decision, local, selected)
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
        for relative_step in range(1, service_horizon + 1):
            local = _local_observations(obs)
            role_ids = _selected_ids_in_lane(local, lane_id)
            treatment_ids = role_ids if relative_step <= pulse_steps and not spec["is_control"] else set()
            actions = _actions(
                obs,
                family=family,
                treatment_ids=treatment_ids,
                treatment_value=value,
            )
            action_request_violations += sum(
                abs(float(action[0]) - baseline) > 1e-12 and agent_id not in treatment_ids
                for agent_id, action in actions.items()
            )
            previous_acceleration = _snapshot_acceleration()
            obs, _, terminated, truncated, info = env.step(actions)
            raw_commands = list(info.get("idm_residual", {}).get("commands", []))
            step_rows = []
            for command in raw_commands:
                vehicle_id = str(command["vehicle_id"])
                projected = command.get("projected") or {}
                treatment_role = vehicle_id in treatment_ids
                if treatment_role:
                    realized_action_steps.add(relative_step)
                step_rows.append(
                    {
                        "relative_step": relative_step,
                        "simulation_time_s": float(traci.simulation.getTime()),
                        "vehicle_id": vehicle_id,
                        "lane_id": command.get("lane_id"),
                        "treatment_role": treatment_role,
                        "requested_action": float(command.get("requested_action", baseline)),
                        "filtered_action": float(command.get("filtered_action", baseline)),
                        "nominal_acceleration_mps2": float(
                            command.get("nominal_acceleration_mps2", 0.0)
                        ),
                        "desired_acceleration_mps2": float(
                            command.get("desired_acceleration_mps2", 0.0)
                        ),
                        "projected_acceleration_mps2": projected.get("acceleration_mps2"),
                        "projection_reasons": list(projected.get("reasons") or []),
                    }
                )
            command_rows.extend(step_rows)
            trace_records.append(
                capture_spatial_response(
                    env,
                    relative_step=relative_step,
                    role_ids=role_ids,
                    frozen_lane_id=lane_id,
                    previous_acceleration=previous_acceleration,
                    commands=raw_commands,
                )
            )
            if relative_step <= safety_horizon:
                severe_total += float(env._corridor_severe_braking_count() or 0.0)
                wave_total += float(env._corridor_acceleration_energy_sum_m2ps4() or 0.0)
                current_ttc = int(env._corridor_ttc_count() or 0)
                ttc_total += current_ttc
                low_ttc_steps += int(current_ttc > 0)
            delay = env._corridor_delay_rate_live()
            if delay is not None and math.isfinite(float(delay)):
                delay_rates.append(float(delay))
            arrivals += int(traci.simulation.getArrivedNumber())
            discharge += _merge_discharge_step(env, previous_road)
            collisions += int(traci.simulation.getCollidingVehiclesNumber())
            teleports += int(traci.simulation.getStartingTeleportNumber())
            collapse_step = bool(
                trace_records[-1].get("merge_service", {}).get("collapse_condition", False)
            )
            collapse_run = collapse_run + 1 if collapse_step else 0
            collapse_max = max(collapse_max, collapse_run)
            completed += 1
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
    finally:
        env.close()
    treatment_commands = [row for row in command_rows if row["treatment_role"]]
    return {
        "schema": SCHEMA,
        "protocol_hash": spec["protocol_hash"],
        "branch_id": spec["branch_id"],
        "case_id": spec["case_id"],
        "arm": spec["arm"],
        "family": family,
        "value": value,
        "is_control": bool(spec["is_control"]),
        "demand": int(spec["demand"]),
        "seed": int(spec["seed"]),
        "landmark": decision["landmark"],
        "decision_step": decision_step,
        "absorber_id": decision["absorber_id"],
        "decision_fingerprint": fingerprint,
        "discovery_fingerprint": discovery_fingerprint,
        "decision_identity_with_discovery": fingerprint == discovery_fingerprint,
        "pulse_steps": pulse_steps,
        "safety_horizon_steps": safety_horizon,
        "service_horizon_steps": service_horizon,
        "completed_horizon_steps": completed,
        "realized_action_step_count": len(realized_action_steps),
        "action_request_violations": action_request_violations,
        "command_rows": command_rows,
        "projection_count": sum(bool(row["projection_reasons"]) for row in treatment_commands),
        "projection_reason_counts": dict(
            Counter(reason for row in treatment_commands for reason in row["projection_reasons"])
        ),
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
        "trace_records": trace_records,
    }


def _atomic_trace(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    levels = ("immediate_follower", "influence_region", "lane", "corridor")
    metrics = (
        "vehicle_count",
        "mean_speed_mps",
        "speed_variance_m2ps2",
        "acceleration_energy_m2ps4",
        "integrated_negative_acceleration_mps",
        "severe_braking_count",
        "hard_braking_count",
        "minimum_ttc_s",
        "low_ttc_count",
    )
    payload: dict[str, Any] = {
        "relative_step": np.asarray([row["relative_step"] for row in records], dtype=np.int16),
        "simulation_time_s": np.asarray(
            [row.get("simulation_time_s", np.nan) for row in records], dtype=np.float64
        ),
        "event_json": np.asarray(
            [json.dumps(row, separators=(",", ":"), allow_nan=False) for row in records]
        ),
    }
    for level in levels:
        for metric in metrics:
            payload[f"{level}.{metric}"] = np.asarray(
                [
                    np.nan if row.get(level, {}).get(metric) is None else row[level][metric]
                    for row in records
                ],
                dtype=np.float64,
            )
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_valid(path: Path, protocol_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    trace = path.parent.parent / row.get("trace_path", "")
    return row if row.get("protocol_hash") == protocol_hash and trace.exists() else None


def _write_progress(output: Path, protocol_hash: str, **values: Any) -> None:
    atomic_write_json(
        output / "progress.json",
        {"schema": SCHEMA, "protocol_hash": protocol_hash, "updated_unix_s": time.time(), **values},
    )


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    cells = [
        {"demand": demand, "seeds": sorted(set(seeds))}
        for demand, seeds in sorted(args.cells)
    ]
    return {
        "schema": SCHEMA,
        "development_only": True,
        "cells": cells,
        "landmarks": list(args.landmarks),
        "landmark_contract": LandmarkContract(maximum_landmarks_per_cell=2).to_dict(),
        "gain_values": sorted(set(float(value) for value in args.gain_values)),
        "reference_offsets_mps": sorted(
            set(float(value) for value in args.reference_offsets_mps)
        ),
        "mode_aliases": MODE_TO_GAIN,
        "pulse_steps": int(args.pulse_steps),
        "safety_horizon_steps": int(args.safety_horizon_steps),
        "service_horizon_steps": int(args.service_horizon_steps),
        "max_predecision_steps": int(args.max_predecision_steps),
        "action_filter_tau_s": 5.0,
        "replay_deadband_base": 0.02,
        "minimum_dose_fraction": 0.8,
        "workers": int(args.workers),
        "reserved_seeds_unopened": True,
    }


def _effect_case(rows: list[dict[str, Any]], refs: Mapping[str, float], deadband: float) -> dict:
    by_arm = {row["arm"]: row for row in rows}
    control = by_arm["nominal"]
    fingerprints = {row["decision_fingerprint"] for row in rows}
    effects = []
    for arm, action in sorted(by_arm.items()):
        if action["is_control"]:
            continue
        physical = safety_composite(
            zero_severe=control["severe_braking_count_sum"],
            action_severe=action["severe_braking_count_sum"],
            severe_reference=refs["severe"],
            zero_wave=control["acceleration_energy_sum"],
            action_wave=action["acceleration_energy_sum"],
            wave_reference=refs["wave"],
            horizon_steps=control["safety_horizon_steps"],
        )
        delay_delta = action["mean_corridor_delay_rate"] - control["mean_corridor_delay_rate"]
        arrival_delta = action["arrived_vehicles"] - control["arrived_vehicles"]
        discharge_delta = action["merge_discharge_count"] - control["merge_discharge_count"]
        constraints = {
            "complete_horizon": action["completed_horizon_steps"] == action["service_horizon_steps"],
            "dose_at_least_80pct": action["realized_action_step_count"]
            >= math.ceil(0.8 * action["pulse_steps"]),
            "zero_action_leakage": action["action_request_violations"] == 0,
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
        effects.append(
            {
                "arm": arm,
                "family": action["family"],
                "value": action["value"],
                **physical,
                "constraints": constraints,
                "constraints_pass": all(constraints.values()),
                "classification": classify_effect(
                    physical["safety_composite_reference_units"],
                    deadband=deadband,
                    constraints_pass=all(constraints.values()),
                ),
                "delay_rate_delta": delay_delta,
                "arrival_delta": arrival_delta,
                "merge_discharge_count_delta": discharge_delta,
            }
        )
    return {
        "case_id": control["case_id"],
        "demand": control["demand"],
        "seed": control["seed"],
        "landmark": control["landmark"],
        "decision_step": control["decision_step"],
        "all_branch_decision_identity": len(fingerprints) == 1,
        "all_discovery_identity": all(row["decision_identity_with_discovery"] for row in rows),
        "effects": effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", dest="cells", type=_parse_cell, action="append", required=True)
    parser.add_argument(
        "--landmarks", nargs="+", choices=["wave_onset", "severe_onset"], required=True
    )
    parser.add_argument("--gain-values", type=float, nargs="+", required=True)
    parser.add_argument("--reference-offsets-mps", type=float, nargs="+", required=True)
    parser.add_argument("--pulse-steps", type=int, default=20)
    parser.add_argument("--safety-horizon-steps", type=int, default=40)
    parser.add_argument("--service-horizon-steps", type=int, default=60)
    parser.add_argument("--max-predecision-steps", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    requested_seeds = {seed for _, seeds in args.cells for seed in seeds}
    if any(seed < 4107 or seed > 4120 for seed in requested_seeds):
        raise ValueError("authority screen is restricted to development seeds 4107--4120")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("authority screen workers must remain in [1, 4]")
    if args.service_horizon_steps < args.safety_horizon_steps:
        raise ValueError("service horizon must be at least the safety horizon")
    output = args.output.resolve()
    for child in ("landmarks", "branches", "traces"):
        (output / child).mkdir(parents=True, exist_ok=True)
    protocol = _protocol(args)
    protocol_hash = _canonical_hash(protocol)
    manifest = output / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol_hash:
            raise RuntimeError("output contains a different frozen protocol")
    else:
        atomic_write_json(
            manifest, {**protocol, "protocol_hash": protocol_hash, "created_unix_s": time.time()}
        )
    landmark_rows = []
    discovery_specs = []
    for cell in protocol["cells"]:
        for seed in cell["seeds"]:
            path = output / "landmarks" / f"m{cell['demand']}.seed{seed}.json"
            cached = None
            if path.exists():
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if candidate.get("protocol_hash") == protocol_hash:
                    cached = candidate
            if cached is not None:
                landmark_rows.append(cached)
            else:
                discovery_specs.append(
                    {
                        "demand": cell["demand"],
                        "seed": seed,
                        "landmark_contract": protocol["landmark_contract"],
                        "max_steps": protocol["max_predecision_steps"],
                        "output": str(output),
                        "protocol_hash": protocol_hash,
                        "path": str(path),
                    }
                )
    if discovery_specs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(discover_landmarks, spec): spec for spec in discovery_specs}
            for future in as_completed(futures):
                row = future.result()
                atomic_write_json(Path(futures[future]["path"]), row)
                landmark_rows.append(row)
                print(
                    f"[landmark] m{row['demand']} seed{row['seed']} found={len(row['landmarks'])} "
                    f"runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    arm_rows = _arms(protocol)
    branch_rows = []
    branch_specs = []
    allowed_landmarks = set(protocol["landmarks"])
    for cell in sorted(landmark_rows, key=lambda row: (row["demand"], row["seed"])):
        for decision in cell["landmarks"]:
            if decision["landmark"] not in allowed_landmarks:
                continue
            case_id = (
                f"m{cell['demand']}.seed{cell['seed']}."
                f"{decision['landmark']}.t{decision['decision_step']}"
            )
            for arm in arm_rows:
                branch_id = f"{case_id}.{arm['arm']}"
                path = output / "branches" / f"{branch_id}.json"
                cached = _load_valid(path, protocol_hash)
                if cached is not None:
                    branch_rows.append(cached)
                    continue
                branch_specs.append(
                    {
                        **arm,
                        "output": str(output),
                        "protocol_hash": protocol_hash,
                        "case_id": case_id,
                        "branch_id": branch_id,
                        "demand": cell["demand"],
                        "seed": cell["seed"],
                        "decision": decision,
                        "pulse_steps": protocol["pulse_steps"],
                        "safety_horizon_steps": protocol["safety_horizon_steps"],
                        "service_horizon_steps": protocol["service_horizon_steps"],
                        "action_filter_tau_s": protocol["action_filter_tau_s"],
                        "path": str(path),
                    }
                )
    _write_progress(
        output,
        protocol_hash,
        branch_count_complete=len(branch_rows),
        branch_count_pending=len(branch_specs),
    )
    if branch_specs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_branch, spec): spec for spec in branch_specs}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                spec = futures[future]
                trace_records = row.pop("trace_records")
                trace_path = output / "traces" / f"{row['branch_id']}.npz"
                _atomic_trace(trace_path, trace_records)
                row["trace_path"] = str(trace_path.relative_to(output))
                row["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                atomic_write_json(Path(spec["path"]), row)
                branch_rows.append(row)
                _write_progress(
                    output,
                    protocol_hash,
                    branch_count_complete=len(branch_rows),
                    branch_count_pending=len(branch_specs) - index,
                )
                print(
                    f"[branch {index}/{len(branch_specs)}] {row['branch_id']} "
                    f"runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in branch_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    complete = [rows for rows in by_case.values() if len(rows) == len(arm_rows)]
    refs_payload = json.loads(
        (ROOT / "runs/eval/idm_phase5_reward_refs_v1.json").read_text(encoding="utf-8")
    )["refs"]
    replay_effects = []
    for rows in complete:
        by_arm = {row["arm"]: row for row in rows}
        lhs, rhs = by_arm["nominal"], by_arm["nominal_replay"]
        ref = refs_payload[f"osm.dense.idm_m{lhs['demand']}r400.sumocfg"]
        replay_effects.append(
            abs(
                safety_composite(
                    zero_severe=lhs["severe_braking_count_sum"],
                    action_severe=rhs["severe_braking_count_sum"],
                    severe_reference=ref["corridor_severe_braking_count_ref_per_s"],
                    zero_wave=lhs["acceleration_energy_sum"],
                    action_wave=rhs["acceleration_energy_sum"],
                    wave_reference=ref["corridor_acceleration_energy_sum_ref_m2ps4"],
                    horizon_steps=lhs["safety_horizon_steps"],
                )["safety_composite_reference_units"]
            )
        )
    deadband = replay_deadband(0.02, replay_effects)
    cases = []
    for rows in complete:
        demand = int(rows[0]["demand"])
        ref = refs_payload[f"osm.dense.idm_m{demand}r400.sumocfg"]
        cases.append(
            _effect_case(
                rows,
                {
                    "severe": ref["corridor_severe_braking_count_ref_per_s"],
                    "wave": ref["corridor_acceleration_energy_sum_ref_m2ps4"],
                },
                deadband,
            )
        )
    effects = [effect for case in cases for effect in case["effects"]]
    gates = {
        "all_branch_decision_identity": all(case["all_branch_decision_identity"] for case in cases),
        "all_discovery_identity": all(case["all_discovery_identity"] for case in cases),
        "nominal_replay_exact": all(value == 0.0 for value in replay_effects),
        "all_horizons_complete": all(
            row["completed_horizon_steps"] == row["service_horizon_steps"] for row in branch_rows
        ),
        "zero_action_leakage": all(row["action_request_violations"] == 0 for row in branch_rows),
        "zero_collisions": all(row["collision_observations"] == 0 for row in branch_rows),
        "zero_teleports": all(row["teleport_observations"] == 0 for row in branch_rows),
        "causal_timestamps": all(
            Path(output / row["trace_path"]).exists() for row in branch_rows
        ),
    }
    summary = {
        "schema": SCHEMA,
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "branch_count": len(branch_rows),
        "case_count": len(cases),
        "frozen_deadband": deadband,
        "replay_absolute_safety_composite_effects": replay_effects,
        "classification_counts": dict(Counter(row["classification"] for row in effects)),
        "gates": gates,
        "smoke_integrity_pass": bool(cases) and all(gates.values()),
        "cases": sorted(cases, key=lambda row: (row["demand"], row["seed"], row["decision_step"])),
        "generated_unix_s": time.time(),
    }
    atomic_write_json(output / "summary.json", summary)
    _write_progress(
        output,
        protocol_hash,
        branch_count_complete=len(branch_rows),
        branch_count_pending=0,
        complete=True,
    )
    print(
        json.dumps(
            {
                "branch_count": summary["branch_count"],
                "case_count": summary["case_count"],
                "classification_counts": summary["classification_counts"],
                "gates": gates,
                "smoke_integrity_pass": summary["smoke_integrity_pass"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
