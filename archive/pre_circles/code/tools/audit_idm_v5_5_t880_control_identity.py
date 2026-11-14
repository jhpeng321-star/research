"""Reconcile density-mapper and t=880 pilot damping-control identity.

This audit opens only the already-used development seeds and never requests a
wall. It reruns the density-mapper and pilot command-off wrappers, then
separates microscopic traffic/control identity from the supervisor's
episode-progress observation, whose denominator differs between the runners.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from counterfactual.twin_replay import ReplayTrace, StepSnapshot, run_replay
from experiments import REPO_ROOT
from experiments.idm.idm_v5_5_t880_contract import (
    CAV_PENETRATION,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    DEVELOPMENT_SEEDS,
    contract_manifest,
)
from experiments.idm.run_idm_v5_5_damping_density_mapper import _always_damping
from experiments.idm.run_idm_v5_5_t880_wall_pilot import (
    DEFAULT_DENSITY,
    DEFAULT_GATE_F,
    DEFAULT_OUTPUT as DEFAULT_PILOT,
    DEFAULT_SUMO_EXE,
    _gate_f_row,
    _relative_decision,
    _replay as run_pilot_control,
)
from experiments.idm.run_idm_wall_pilot import supervisor_overrides
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from policies.idm_analytical import build_frozen_idm_analytical_policy
from policies.idm_analytical import frozen_idm_analytical_manifest


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "idm_v5_5_t880_control_identity_audit_p30_s4401_4406"
)
CASE_SCHEMA = "idm_v5_5_t880_control_identity_case/1"
SUMMARY_SCHEMA = "idm_v5_5_t880_control_identity_summary/1"
SUPERVISOR_PROGRESS_INDEX = 10
DYNAMIC_CONTROL_KEYS = (
    "command",
    "command_name",
    "base_action_ids",
    "base_actions",
    "base_direct_control_ids",
    "base_control_ids",
    "base_diagnostics",
    "applied_env_actions",
    "wall_active",
    "wall_override_ids",
    "out_of_band_damping_ids",
    "transition_reason",
)
AUDIT_DEFINITION = {
    "case_schema": CASE_SCHEMA,
    "supervisor_progress_index_excluded": SUPERVISOR_PROGRESS_INDEX,
    "dynamic_control_keys": DYNAMIC_CONTROL_KEYS,
    "physical_state_fields": (
        "relative_step",
        "sim_time",
        "vehicles",
        "active_cavs",
        "collision_vehicle_ids",
        "starting_teleport_vehicle_ids",
    ),
    "reset_info_exact": True,
}


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_snapshot(snapshot: StepSnapshot) -> dict[str, Any]:
    return {
        "relative_step": int(snapshot.relative_step),
        "sim_time": snapshot.sim_time,
        "vehicles": snapshot.vehicles,
        "active_cavs": snapshot.active_cavs,
        "collision_vehicle_ids": snapshot.collision_vehicle_ids,
        "starting_teleport_vehicle_ids": snapshot.starting_teleport_vehicle_ids,
    }


def physical_state_digest(snapshots: Sequence[StepSnapshot]) -> str:
    """Hash full microscopic traffic state without supervisor bookkeeping."""

    return _sha256([_physical_snapshot(snapshot) for snapshot in snapshots])


def normalized_supervisor_digest(snapshots: Sequence[StepSnapshot]) -> str:
    """Hash observations after removing only the episode-progress component."""

    rows = []
    for snapshot in snapshots:
        observation = tuple(snapshot.supervisor_observation)
        rows.append(
            {
                "relative_step": int(snapshot.relative_step),
                "sim_time": snapshot.sim_time,
                "observation_without_episode_progress": (
                    observation[:SUPERVISOR_PROGRESS_INDEX]
                    + observation[SUPERVISOR_PROGRESS_INDEX + 1 :]
                ),
            }
        )
    return _sha256(rows)


def dynamic_control_digest(trace: ReplayTrace) -> str:
    """Hash causal damping outputs while excluding dormant overlay metadata."""

    rows = [
        {
            "relative_step": int(row["relative_step"]),
            **{key: row.get(key) for key in DYNAMIC_CONTROL_KEYS},
        }
        for row in trace.controller_audits
    ]
    return _sha256(rows)


def _component_difference(left: StepSnapshot, right: StepSnapshot) -> dict[str, bool]:
    return {
        "relative_step": left.relative_step != right.relative_step,
        "sim_time": left.sim_time != right.sim_time,
        "action": left.action != right.action,
        "supervisor_observation": left.supervisor_observation != right.supervisor_observation,
        "vehicles": left.vehicles != right.vehicles,
        "active_cavs": left.active_cavs != right.active_cavs,
        "collisions": left.collision_vehicle_ids != right.collision_vehicle_ids,
        "teleports": (
            left.starting_teleport_vehicle_ids != right.starting_teleport_vehicle_ids
        ),
    }


def _observation_difference_indices(left: Sequence[Any], right: Sequence[Any]) -> list[int]:
    if len(left) != len(right):
        return list(range(max(len(left), len(right))))
    return [index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]]


def compare_control_traces(density: ReplayTrace, pilot: ReplayTrace) -> dict[str, Any]:
    shared = min(len(density.snapshots), len(pilot.snapshots))
    component_differences: set[str] = set()
    observation_indices: set[int] = set()
    first_difference = None
    for index in range(shared):
        left = density.snapshots[index]
        right = pilot.snapshots[index]
        different = _component_difference(left, right)
        component_differences.update(name for name, value in different.items() if value)
        observation_indices.update(
            _observation_difference_indices(
                left.supervisor_observation, right.supervisor_observation
            )
        )
        if first_difference is None and any(different.values()):
            first_difference = {
                "relative_step": int(left.relative_step),
                "sim_time_s": float(left.sim_time),
                "different_components": sorted(
                    name for name, value in different.items() if value
                ),
                "supervisor_difference_indices": sorted(
                    _observation_difference_indices(
                        left.supervisor_observation, right.supervisor_observation
                    )
                ),
            }
    density_physical = physical_state_digest(density.snapshots)
    pilot_physical = physical_state_digest(pilot.snapshots)
    density_observation = normalized_supervisor_digest(density.snapshots)
    pilot_observation = normalized_supervisor_digest(pilot.snapshots)
    density_control = dynamic_control_digest(density)
    pilot_control = dynamic_control_digest(pilot)
    checks = {
        "same_snapshot_count": len(density.snapshots) == len(pilot.snapshots),
        "reset_info_exact": dict(density.reset_info) == dict(pilot.reset_info),
        "microscopic_physical_state_exact": density_physical == pilot_physical,
        "dynamic_damping_control_exact": density_control == pilot_control,
        "normalized_supervisor_observation_exact": density_observation == pilot_observation,
        "raw_difference_confined_to_supervisor_observation": (
            component_differences == {"supervisor_observation"}
        ),
        "supervisor_difference_confined_to_episode_progress_index_10": (
            observation_indices == {SUPERVISOR_PROGRESS_INDEX}
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "raw_trace_digest_exact": density.trace_digest == pilot.trace_digest,
        "raw_different_components": sorted(component_differences),
        "supervisor_difference_indices": sorted(observation_indices),
        "first_difference": first_difference,
        "snapshot_counts": [len(density.snapshots), len(pilot.snapshots)],
        "reset_info": {
            "density": dict(density.reset_info),
            "pilot": dict(pilot.reset_info),
        },
        "digests": {
            "density_raw_trace": density.trace_digest,
            "pilot_raw_trace": pilot.trace_digest,
            "density_microscopic_physical_state": density_physical,
            "pilot_microscopic_physical_state": pilot_physical,
            "density_dynamic_damping_control": density_control,
            "pilot_dynamic_damping_control": pilot_control,
            "density_normalized_supervisor_observation": density_observation,
            "pilot_normalized_supervisor_observation": pilot_observation,
        },
    }


def _density_control(seed: int, sumo_binary: str) -> ReplayTrace:
    return run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=sumo_binary,
        sumo_config=transition_scenario_config(DEMAND_MAINLINE_VPH).name,
        seed=seed,
        decision_step=0,
        horizon_steps=1080,
        treatment_steps=None,
        cav_penetration=CAV_PENETRATION,
        sumo_step_length=0.5,
        supervisor_overrides={
            **supervisor_overrides(),
            "use_health_obs": False,
            "use_dv_obs": False,
            "feasibility_mask": False,
            "action_masking": False,
        },
        digits=6,
        base_policy_factory=build_frozen_idm_analytical_policy,
        intervention_schedule=_always_damping,
        overlay_overrides=None,
        controller_metadata={"audit_role": "density_mapper_control_replay"},
    )


def _source_paths(case: Mapping[str, Any], seed: int) -> tuple[Path, Path]:
    density_path = (
        Path(case["density"])
        / "cases"
        / f"m{DEMAND_MAINLINE_VPH}.seed{seed}.primary.json"
    )
    pilot_path = Path(case["pilot"]) / "cases" / f"seed{seed}.primary.json"
    return density_path, pilot_path


def _source_provenance(case: Mapping[str, Any], seed: int) -> dict[str, Any]:
    density_path, pilot_path = _source_paths(case, seed)
    return {
        "density_case_sha256": _sha256_file(density_path),
        "pilot_case_sha256": _sha256_file(pilot_path),
        "audit_definition_sha256": _sha256(AUDIT_DEFINITION),
        "t880_contract_fingerprint_sha256": contract_manifest()[
            "contract_fingerprint_sha256"
        ],
    }


def _validate_source_artifacts(
    density: Mapping[str, Any], pilot: Mapping[str, Any], *, seed: int
) -> dict[str, bool]:
    return {
        "density_schema": density.get("schema") == "idm_v5_5_density_mapper_branch/1",
        "density_identity": int(density.get("seed", -1)) == seed
        and density.get("repeat") is False,
        "density_scenario": int(density.get("demand_mainline_vph", -1))
        == DEMAND_MAINLINE_VPH
        and int(density.get("demand_ramp_vph", -1)) == DEMAND_RAMP_VPH
        and abs(float(density.get("cav_penetration", -1.0)) - CAV_PENETRATION)
        <= 1e-12,
        "density_base_control": density.get("base_control_arm")
        == frozen_idm_analytical_manifest(),
        "pilot_schema": pilot.get("schema") == "idm_v5_5_t880_wall_pilot_case/1",
        "pilot_identity": int(pilot.get("seed", -1)) == seed
        and pilot.get("repeat") is False,
        "pilot_scenario": int(pilot.get("demand_mainline_vph", -1))
        == DEMAND_MAINLINE_VPH
        and int(pilot.get("demand_ramp_vph", -1)) == DEMAND_RAMP_VPH
        and abs(float(pilot.get("cav_penetration", -1.0)) - CAV_PENETRATION)
        <= 1e-12,
        "pilot_contract": pilot.get("contract_fingerprint_sha256")
        == contract_manifest()["contract_fingerprint_sha256"],
        "pilot_base_control": pilot.get("base_control_arm")
        == frozen_idm_analytical_manifest(),
        "pilot_exact_arms": set(pilot.get("arms") or {}) == {"0", "1"},
    }


def _run_seed(case: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    path = output / "seeds" / f"seed{seed}.json"
    expected_provenance = _source_provenance(case, seed)
    if path.is_file() and not bool(case.get("force")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != CASE_SCHEMA
            or int(payload.get("seed", -1)) != seed
            or payload.get("source_provenance") != expected_provenance
            or not bool(payload.get("passed"))
        ):
            raise RuntimeError(f"invalid resumable identity artifact for seed {seed}")
        return {"status": "skipped", "seed": seed, "path": str(path)}
    started = time.time()
    gate_f = _gate_f_row(Path(case["gate_f"]), seed)
    decision = _relative_decision(gate_f)
    density = _density_control(seed, str(case["sumo_binary"]))
    pilot = run_pilot_control(
        seed=seed,
        decision_step=decision,
        treatment=False,
        sumo_binary=str(case["sumo_binary"]),
    )
    comparison = compare_control_traces(density, pilot)
    density_path, pilot_path = _source_paths(case, seed)
    density_artifact = json.loads(density_path.read_text(encoding="utf-8"))
    pilot_artifact = json.loads(pilot_path.read_text(encoding="utf-8"))
    source_checks = _validate_source_artifacts(
        density_artifact, pilot_artifact, seed=seed
    )
    density_audit_digest = _sha256(density.controller_audits)
    pilot_audit_digest = _sha256(pilot.controller_audits)
    provenance_checks = {
        **source_checks,
        "rerun_density_raw_trace_matches_opened_density_artifact": (
            density.trace_digest == density_artifact["trace_digest"]
        ),
        "rerun_density_controller_audit_matches_opened_density_artifact": (
            density_audit_digest == density_artifact["controller_audit_digest"]
        ),
        "rerun_density_reset_matches_opened_density_artifact": (
            dict(density.reset_info) == density_artifact["reset_info"]
        ),
        "rerun_pilot_raw_trace_matches_completed_pilot_control": (
            pilot.trace_digest == pilot_artifact["arms"]["0"]["trace_digest"]
        ),
        "rerun_pilot_controller_audit_matches_completed_pilot_control": (
            pilot_audit_digest
            == pilot_artifact["arms"]["0"]["controller_audit_digest"]
        ),
        "rerun_pilot_reset_matches_completed_pilot_control": (
            dict(pilot.reset_info)
            == pilot_artifact["arms"]["0"]["trace_summary"]["reset_info"]
        ),
    }
    payload = {
        "schema": CASE_SCHEMA,
        "seed": seed,
        "wall_requested": False,
        "new_seed_opened": False,
        "source_provenance": expected_provenance,
        "comparison": comparison,
        "provenance_checks": provenance_checks,
        "passed": comparison["passed"] and all(provenance_checks.values()),
        "wall_clock_seconds": time.time() - started,
    }
    atomic_write_json(path, payload)
    return {
        "status": "completed",
        "seed": seed,
        "passed": payload["passed"],
        "path": str(path),
        "wall_clock_seconds": payload["wall_clock_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--gate-f", type=Path, default=DEFAULT_GATE_F)
    parser.add_argument("--density", type=Path, default=DEFAULT_DENSITY)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    (output / "seeds").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output / "manifest.json",
        {
            "schema": "idm_v5_5_t880_control_identity_manifest/1",
            "seeds": list(DEVELOPMENT_SEEDS),
            "workers": min(max(1, args.workers), len(DEVELOPMENT_SEEDS)),
            "wall_requested": False,
            "supervisor_progress_index_excluded": SUPERVISOR_PROGRESS_INDEX,
            "scientific_outcomes_used": False,
            "audit_definition": AUDIT_DEFINITION,
            "audit_definition_sha256": _sha256(AUDIT_DEFINITION),
        },
    )
    cases = [
        {
            "seed": seed,
            "sumo_binary": str(args.sumo_binary),
            "gate_f": str(args.gate_f.resolve()),
            "density": str(args.density.resolve()),
            "pilot": str(args.pilot.resolve()),
            "output": str(output),
            "force": bool(args.force),
        }
        for seed in DEVELOPMENT_SEEDS
    ]
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 6)) as pool:
        futures = [pool.submit(_run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    rows = [
        json.loads((output / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
        for seed in DEVELOPMENT_SEEDS
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "seed_count": len(rows),
        "all_microscopic_control_identity_passed": all(row["passed"] for row in rows),
        "all_raw_trace_digests_differ": all(
            not row["comparison"]["raw_trace_digest_exact"] for row in rows
        ),
        "all_differences_confined_to_supervisor_progress": all(
            row["comparison"]["checks"][
                "supervisor_difference_confined_to_episode_progress_index_10"
            ]
            for row in rows
        ),
        "per_seed": [
            {
                "seed": row["seed"],
                "passed": row["passed"],
                "checks": row["comparison"]["checks"],
                "provenance_checks": row["provenance_checks"],
            }
            for row in rows
        ],
        "interpretation": (
            "The original raw trace hash was entangled with the runner-specific "
            "episode-progress observation; full microscopic traffic state and "
            "dynamic damping control are the corrected physical identity layers."
        ),
        "wall_requested": False,
        "new_seed_opened": False,
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["all_microscopic_control_identity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
