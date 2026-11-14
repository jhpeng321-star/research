"""Treatment-blind Gate-F scan at absolute SUMO time 880 for v5.5."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import traci

from env.core import TrafficShepherdMultiAgentEnv
from experiments import REPO_ROOT
from experiments.idm.idm_v5_5_t880_contract import (
    ACTIVATION_TIME_S,
    CAV_PENETRATION,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    DEVELOPMENT_SEEDS,
    MINIMUM_DURATION_FRACTION,
    MINIMUM_FEASIBLE_SEEDS,
    SCIENTIFIC_ROLE,
    TREATMENT_DURATION_S,
    contract_manifest,
)
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.scan_idm_wall_timing import _env_config as _base_env_config
from experiments.idm.wall_v5_4_contract import overlay_overrides, profile_manifest
from experiments.io import atomic_write_json
from idm.contract import transition_scenario_config
from idm.wall_t880 import gate_f_assessment
from policies.counterfactual_escalation import CounterfactualEscalationPolicy
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "probe" / "idm_v5_5_t880_gate_f_p30_s4401_4406"
)
MANIFEST_SCHEMA = "idm_v5_5_t880_gate_f_manifest/1"
SEED_SCHEMA = "idm_v5_5_t880_gate_f_seed/1"
SUMMARY_SCHEMA = "idm_v5_5_t880_gate_f_summary/1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_provenance() -> dict[str, Any]:
    config = transition_scenario_config(DEMAND_MAINLINE_VPH)
    return {
        "scenario_config": str(config),
        "scenario_config_sha256": _sha256_file(config),
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "cav_penetration": CAV_PENETRATION,
    }


def _validate_manifest_payload(payload: dict[str, Any]) -> None:
    expected = contract_manifest()
    checks = {
        "schema": payload.get("schema") == MANIFEST_SCHEMA,
        "contract": payload.get("contract") == expected,
        "development_seeds": payload.get("already_opened_development_seed_reuse")
        == list(DEVELOPMENT_SEEDS),
        "wall_off": payload.get("wall_was_applied") is False,
        "outcomes_unread": payload.get("treatment_outcomes_read") is False,
        "scenario": payload.get("scenario_provenance") == _scenario_provenance(),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gate-F manifest provenance mismatch: {failed}")


def _validate_seed_payload(payload: dict[str, Any], seed: int) -> dict[str, bool]:
    expected_contract = contract_manifest()
    formation = payload.get("formation") or {}
    selected_members = payload.get("selected_members") or []
    selected_ids = sorted(str(row.get("vehicle_id")) for row in selected_members)
    formation_ids = sorted(str(value) for value in formation.get("controlled_cav_ids", ()))
    member_lanes = sorted({str(row.get("lane_id")) for row in selected_members})
    selected_lanes = sorted(str(value) for value in payload.get("selected_lane_ids", ()))
    safety = payload.get("activation_safety")
    safety_ids = sorted(str(value) for value in (safety or {}).get("checked_cav_ids", ()))
    recomputed = gate_f_assessment(
        formation=formation,
        selected_lane_ids=selected_lanes,
        activation_safety=safety,
        terminal_runway_margin_m=payload.get("terminal_runway_margin_m"),
        requested_duration_s=TREATMENT_DURATION_S,
    )
    checks = {
        "schema": payload.get("schema") == SEED_SCHEMA,
        "seed": int(payload.get("seed", -1)) == int(seed)
        and int(seed) in DEVELOPMENT_SEEDS,
        "scientific_role": payload.get("scientific_role") == SCIENTIFIC_ROLE,
        "contract_fingerprint": payload.get("contract_fingerprint_sha256")
        == expected_contract["contract_fingerprint_sha256"],
        "scenario": payload.get("scenario_provenance") == _scenario_provenance(),
        "absolute_time": abs(float(payload.get("decision_sim_time_s", -1.0)) - ACTIVATION_TIME_S)
        <= 1e-6,
        "controller": payload.get("base_control_arm") == frozen_idm_analytical_manifest(),
        "wall_profile": payload.get("wall_profile") == profile_manifest(),
        "treatment_blind": payload.get("wall_was_applied") is False
        and payload.get("treatment_outcomes_read") is False,
        "selected_member_identity": selected_ids == formation_ids,
        "selected_lane_identity": selected_lanes == member_lanes,
        "safety_member_identity": (safety is None and not selected_ids)
        or safety_ids == selected_ids,
        "raw_gate_f_recomputed": payload.get("gate_f") == recomputed,
        "wall_duration_consistency": float(
            profile_manifest()["overlay_overrides"]["planned_treatment_duration_s"]
        )
        == float(TREATMENT_DURATION_S),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gate-F seed {seed} provenance mismatch: {failed}")
    return checks


def _env_config(seed: int, sumo_binary: str) -> dict[str, Any]:
    cfg = _base_env_config(seed, ACTIVATION_TIME_S + 5, sumo_binary)
    cfg.update(supervisor_overrides())
    cfg.update(
        {
            "sumo_config": transition_scenario_config(DEMAND_MAINLINE_VPH).name,
            "cav_penetration": CAV_PENETRATION,
            "use_health_obs": False,
            "use_dv_obs": False,
            "feasibility_mask": False,
            "action_masking": False,
        }
    )
    return cfg


def _selected_member_projection(
    overlay: CounterfactualEscalationPolicy,
    selected_ids: tuple[str, ...],
) -> tuple[list[dict[str, Any]], float | None]:
    if not selected_ids:
        return [], None
    distances: dict[str, float] = {}
    speeds: dict[str, float] = {}
    rows = []
    for vehicle_id in selected_ids:
        distance = overlay._distance_in_band(vehicle_id)
        speed = overlay._vehicle_float("getSpeed", vehicle_id)
        lane_id = overlay._lane_id(vehicle_id)
        if distance is None or speed is None:
            return [], None
        distances[vehicle_id] = float(distance)
        speeds[vehicle_id] = float(speed)
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "lane_id": lane_id,
                "distance_to_merge_m": float(distance),
                "speed_mps": float(speed),
            }
        )
    _initial, _maximum, _terminal, minimum_terminal_distance = (
        overlay._project_compact_formation(selected_ids, distances, speeds)
    )
    margin = float(minimum_terminal_distance) - float(overlay.compact_release_boundary_m)
    return rows, margin


def _probe(env: Any, obs: dict[str, Any], seed: int) -> dict[str, Any]:
    overlay = CounterfactualEscalationPolicy(
        base_policy=build_frozen_idm_analytical_policy(),
        **overlay_overrides(),
    )
    try:
        overlay.reset(env, seed=seed)
        live_ids = overlay._live_ids()
        formation = overlay._formation_status(obs.keys(), live_ids)
        selected_ids = tuple(str(value) for value in formation.controlled_cav_ids)
        selected_lane_ids = tuple(
            sorted(
                {
                    lane_id
                    for vehicle_id in selected_ids
                    if (lane_id := overlay._lane_id(vehicle_id)) is not None
                }
            )
        )
        member_rows, runway_margin = _selected_member_projection(overlay, selected_ids)
        safety = None
        if formation.feasible:
            safety = overlay._activation_safety_status(selected_ids, live_ids).to_dict()
        assessment = gate_f_assessment(
            formation=formation.to_dict(),
            selected_lane_ids=selected_lane_ids,
            activation_safety=safety,
            terminal_runway_margin_m=runway_margin,
            requested_duration_s=TREATMENT_DURATION_S,
        )
        return {
            "formation": formation.to_dict(),
            "selected_lane_ids": list(selected_lane_ids),
            "selected_members": member_rows,
            "activation_safety": safety,
            "terminal_runway_margin_m": runway_margin,
            "gate_f": assessment,
            "active_cav_count": len(obs),
            "live_vehicle_count": len(live_ids),
            "wall_was_applied": False,
            "treatment_outcomes_read": False,
        }
    finally:
        overlay.close()


def run_seed(case: dict[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    output = Path(case["output"])
    path = output / "seeds" / f"seed{seed}.json"
    if path.is_file() and not bool(case.get("force")):
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_seed_payload(existing, seed)
        return {"status": "skipped", "seed": seed, "path": str(path)}
    env = TrafficShepherdMultiAgentEnv(_env_config(seed, str(case["sumo_binary"])))
    base = build_frozen_idm_analytical_policy()
    started = time.time()
    try:
        obs, reset_info = env.reset(seed=seed)
        base.reset(env, seed=seed)
        while float(traci.simulation.getTime()) < ACTIVATION_TIME_S:
            base.act(obs)
            obs, _rewards, terminated, truncated, info = env.step({})
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                raise RuntimeError(
                    f"seed {seed} ended before Gate-F at {info.get('episode_time')}"
                )
        decision_time = float(traci.simulation.getTime())
        if abs(decision_time - ACTIVATION_TIME_S) > 1e-6:
            raise RuntimeError(
                f"seed {seed} missed exact absolute Gate-F time: {decision_time}"
            )
        # This is the identical damping call that the treatment branch will
        # execute at its decision. The probe reads state but never commands a wall.
        base.act(obs)
        result = _probe(env, obs, seed)
        result.update(
            {
                "schema": SEED_SCHEMA,
                "scientific_role": SCIENTIFIC_ROLE,
                "seed": seed,
                "decision_sim_time_s": decision_time,
                "contract_fingerprint_sha256": contract_manifest()[
                    "contract_fingerprint_sha256"
                ],
                "scenario_provenance": _scenario_provenance(),
                "reset_info": reset_info,
                "damping_command_count": len(base.last_diagnostics.get("commands", ())),
                "base_control_arm": frozen_idm_analytical_manifest(),
                "wall_profile": profile_manifest(),
                "wall_clock_seconds": time.time() - started,
            }
        )
        atomic_write_json(path, result)
    finally:
        base.close()
        env.close()
    return {
        "status": "completed",
        "seed": seed,
        "path": str(path),
        "admitted": bool(result["gate_f"]["admitted"]),
        "reason": str(result["formation"]["reason"]),
    }


def build_summary(output: Path) -> dict[str, Any]:
    rows = [
        json.loads((output / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
        for seed in DEVELOPMENT_SEEDS
    ]
    row_checks = {
        int(row["seed"]): _validate_seed_payload(row, int(row["seed"])) for row in rows
    }
    admitted = [row for row in rows if bool(row["gate_f"]["admitted"])]
    observed_seeds = sorted(int(row["seed"]) for row in rows)
    gates = {
        "exact_declared_seed_artifacts": observed_seeds == list(DEVELOPMENT_SEEDS),
        "all_seed_provenance_checks": all(
            all(checks.values()) for checks in row_checks.values()
        ),
        "exact_absolute_880_decision": all(
            abs(float(row["decision_sim_time_s"]) - ACTIVATION_TIME_S) <= 1e-6
            for row in rows
        ),
        "treatment_blind": all(
            row["wall_was_applied"] is False
            and row["treatment_outcomes_read"] is False
            for row in rows
        ),
        "frozen_controller_and_wall": all(
            row["base_control_arm"] == frozen_idm_analytical_manifest()
            and row["wall_profile"] == profile_manifest()
            for row in rows
        ),
        "at_least_four_feasible_seeds": len(admitted) >= MINIMUM_FEASIBLE_SEEDS,
        "all_admitted_project_full_duration": all(
            float(row["gate_f"]["projected_duration_fraction"])
            >= MINIMUM_DURATION_FRACTION
            for row in admitted
        ),
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "contract": contract_manifest(),
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "admitted": bool(row["gate_f"]["admitted"]),
                "formation_reason": str(row["formation"]["reason"]),
                "occupied_lane_ids": row["formation"]["occupied_lane_ids"],
                "selected_lane_ids": row["selected_lane_ids"],
                "selected_member_count": len(row["selected_members"]),
                "initial_span_m": row["formation"]["longitudinal_span_m"],
                "predicted_max_span_m": row["formation"]["predicted_max_span_m"],
                "predicted_terminal_span_m": row["formation"]["predicted_terminal_span_m"],
                "terminal_runway_margin_m": row["terminal_runway_margin_m"],
                "activation_safety": row["activation_safety"],
                "gate_f": row["gate_f"],
            }
            for row in rows
        ],
        "admitted_seed_count": len(admitted),
        "admitted_seeds": [int(row["seed"]) for row in admitted],
        "seed_provenance_checks": row_checks,
        "gates": gates,
        "selection_passed": all(gates.values()),
        "wall_pilot_authorized": all(gates.values()),
        "treatment_outcomes_read": False,
        "validation_or_final_seeds_opened": False,
        "predictor_training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "seeds").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_unix_s": time.time(),
        "contract": contract_manifest(),
        "command": {
            "workers": args.workers,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
        },
        "wall_was_applied": False,
        "treatment_outcomes_read": False,
        "already_opened_development_seed_reuse": list(DEVELOPMENT_SEEDS),
    }
    manifest_path = output / "manifest.json"
    manifest["scenario_provenance"] = _scenario_provenance()
    if manifest_path.is_file() and not args.force:
        _validate_manifest_payload(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    else:
        atomic_write_json(manifest_path, manifest)
    cases = [
        {
            "seed": seed,
            "sumo_binary": args.sumo_binary,
            "output": str(output),
            "force": args.force,
        }
        for seed in DEVELOPMENT_SEEDS
    ]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(cases)))) as pool:
        futures = [pool.submit(run_seed, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    summary = build_summary(output)
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["selection_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
