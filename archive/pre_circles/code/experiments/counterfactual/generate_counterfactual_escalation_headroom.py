"""Generate the corrected v2 damping-plus-wall headroom pilot.

Each independent seed reuses one frozen damping trace for three predeclared
decision times and replays one damping-plus-wall branch per decision.  Results
are saved atomically per seed and may be resumed without duplicating work.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from counterfactual.headroom import (
    HeadroomMargins,
    TreatmentRepairCriteria,
    classify_case,
    summarize_headroom,
    summarize_treatment_repair,
)
from counterfactual.safety_service_oracle import classify_safety_service_case
from counterfactual.dataset_contract import (
    CONTRACT_VERSION as DATASET_CONTRACT_VERSION,
    contract_manifest,
    validate_assigned_seeds,
)
from counterfactual.outcomes import (
    OutcomeDefinition,
    summarize_branch,
    summarize_predecision_congestion,
)
from counterfactual.pilot import features_from_trace
from counterfactual.supervisor import (
    NO_SUPERVISOR,
    SUPERVISOR_PROFILES,
    evaluate_predecision_supervisor,
)
from counterfactual.twin_replay import FrozenPulseSchedule, run_replay
from config.constants import MERGE_EDGE, PRE_MERGE_EDGE
from config.defaults import DEFAULT_CONFIG
from policies.counterfactual_escalation import (
    DEFAULT_DAMPING_CHECKPOINT,
    DEFAULT_WALL_PARAMS,
    default_protocol_metadata,
)
from policies.rllib_policy import RLlibPolicy
from experiments.counterfactual.verify_counterfactual_escalation import (
    _audit_contract,
    _prefix_identity,
)


from experiments import REPO_ROOT
from experiments.io import atomic_write_json as _atomic_json

PROJECT = REPO_ROOT
DEFAULT_SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
SCHEMA_VERSION = "counterfactual_escalation_v2_headroom/8"
CONTROLLER_STEP_LENGTH_S = float(DEFAULT_CONFIG["simulation_step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected v2 headroom pilot.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(111, 123)))
    parser.add_argument("--decision-steps", type=int, nargs="+", default=[120, 180, 240])
    parser.add_argument("--outcome-steps", type=int, default=180)
    parser.add_argument("--treatment-steps", type=int, default=60)
    parser.add_argument("--cav-penetration", type=float, default=0.2)
    parser.add_argument("--demand", choices=["dense", "hyper"], default="hyper")
    parser.add_argument("--sumo-step-length", type=float, default=0.5)
    parser.add_argument("--sumo-project-root", type=Path, default=DEFAULT_SUMO_ROOT)
    parser.add_argument("--sumo-config", default="osm.hyper.sumocfg")
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DAMPING_CHECKPOINT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--development",
        action="store_true",
        help="allow a bounded diagnostic seed set; never treat it as confirmation",
    )
    parser.add_argument(
        "--dataset-role",
        choices=[
            "headroom_gate",
            "treatment_development",
            "model_development",
            "locked_test",
            "transfer_test",
        ],
        default=None,
        help=(
            "Frozen experimental role. model_development and locked_test seeds "
            "are checked against counterfactual.dataset_contract."
        ),
    )
    parser.add_argument(
        "--primary-estimand",
        choices=["safety_service", "collapse_prevention"],
        default="collapse_prevention",
        help=(
            "Predeclared pilot target. safety_service keeps collapse outcomes "
            "as diagnostics but does not require collapse-prevention diversity."
        ),
    )
    parser.add_argument(
        "--predecision-supervisor",
        choices=SUPERVISOR_PROFILES,
        default=NO_SUPERVISOR,
        help=(
            "Versioned observable safety envelope used to evaluate which "
            "geometrically feasible wall commands are admissible."
        ),
    )
    parser.add_argument(
        "--overlay-profile",
        default="counterfactual_assemble_hold_release_v2",
    )
    parser.add_argument(
        "--wall-speed", type=float, default=float(DEFAULT_WALL_PARAMS["wall_speed"])
    )
    parser.add_argument("--wall-d-lo", type=float, default=float(DEFAULT_WALL_PARAMS["d_lo"]))
    parser.add_argument("--wall-d-hi", type=float, default=float(DEFAULT_WALL_PARAMS["d_hi"]))
    parser.add_argument(
        "--geometry-mode",
        choices=["static_all", "compact_moving"],
        default="static_all",
    )
    parser.add_argument(
        "--max-longitudinal-span-m",
        type=float,
        default=None,
        help="formation span cap; compact moving-wall experiments use 50 m",
    )
    parser.add_argument(
        "--compact-release-boundary-m",
        type=float,
        default=None,
        help=(
            "downstream release boundary for tracked compact members; "
            "defaults to wall-d-lo for backward compatibility"
        ),
    )
    parser.add_argument(
        "--compact-selection-mode",
        choices=["initial_compact", "trajectory_coherent"],
        default="initial_compact",
    )
    parser.add_argument("--compact-assembly-duration-s", type=float, default=10.0)
    parser.add_argument("--compact-assembly-speed-tolerance-mps", type=float, default=2.0)
    parser.add_argument("--compact-assembly-a-max-dec-mps2", type=float, default=1.0)
    parser.add_argument(
        "--compact-control-mode",
        choices=["fixed_target", "adaptive_servo"],
        default="fixed_target",
    )
    parser.add_argument("--compact-servo-a-max-dec-mps2", type=float, default=1.0)
    parser.add_argument("--compact-max-activation-speed-spread-mps", type=float, default=4.0)
    parser.add_argument("--compact-max-speed-reduction-fraction", type=float, default=0.20)
    parser.add_argument("--compact-shock-gap-lo-mps", type=float, default=5.0)
    parser.add_argument("--compact-shock-gap-hi-mps", type=float, default=15.0)
    parser.add_argument("--compact-position-kp-s-inv", type=float, default=0.05)
    parser.add_argument("--compact-position-kd", type=float, default=0.5)
    parser.add_argument("--compact-position-speed-offset-cap-mps", type=float, default=1.0)
    parser.add_argument("--compact-safety-min-front-ttc-s", type=float, default=3.0)
    parser.add_argument("--compact-safety-min-rear-ttc-s", type=float, default=4.0)
    parser.add_argument("--compact-influence-upstream-m", type=float, default=100.0)
    parser.add_argument("--compact-influence-max-followers", type=int, default=3)
    parser.add_argument("--compact-influence-min-ttc-s", type=float, default=1.5)
    parser.add_argument("--wall-tau", type=float, default=float(DEFAULT_WALL_PARAMS["wall_tau"]))
    parser.add_argument(
        "--wall-a-max-dec",
        type=float,
        default=float(DEFAULT_WALL_PARAMS["a_max_dec"]),
    )
    parser.add_argument(
        "--tau-ramp-step",
        type=float,
        default=float(DEFAULT_WALL_PARAMS["tau_ramp_step"]),
    )
    parser.add_argument(
        "--allow-partial-lane-wall",
        action="store_true",
        help=(
            "Development-only supervisor-consistent variant: do not abort an "
            "active wall merely because an occupied lane temporarily lacks an "
            "in-band CAV. Safety aborts remain active."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "runs" / "probe" / "counterfactual_escalation_v2_headroom_v2",
    )
    parser.add_argument(
        "--summarize-completed-only",
        action="store_true",
        help=(
            "Do not launch pending seeds; atomically summarize valid completed "
            "seed artifacts and mark the requested cohort incomplete."
        ),
    )
    parser.add_argument(
        "--early-stop-reason",
        default=None,
        help="Auditable reason recorded with --summarize-completed-only.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _base_policy_factory(checkpoint: str):
    return lambda: RLlibPolicy(checkpoint, explore=False)


def _merge_entries(trace, start_step: int, end_step: int) -> int:
    """Count PRE_MERGE_EDGE -> MERGE_EDGE crossings in an inclusive window."""
    snapshots = {int(row.relative_step): row for row in trace.snapshots}
    entries: set[str] = set()
    for step in range(int(start_step), int(end_step) + 1):
        previous = snapshots.get(step - 1)
        current = snapshots.get(step)
        if previous is None or current is None:
            continue
        previous_map = previous.vehicle_map()
        current_map = current.vehicle_map()
        for vehicle_id, previous_row in previous_map.items():
            current_row = current_map.get(vehicle_id)
            if (
                str(previous_row[1]) == PRE_MERGE_EDGE
                and current_row is not None
                and str(current_row[1]) == MERGE_EDGE
            ):
                entries.add(vehicle_id)
    return len(entries)


def _first_stage_inflow(control, treatment, decision_step: int, treatment_steps: int):
    during_start = decision_step + 1
    during_end = decision_step + treatment_steps
    post_start = during_end + 1
    post_end = during_end + treatment_steps

    def window(start: int, end: int) -> dict[str, int]:
        control_count = _merge_entries(control, start, end)
        treatment_count = _merge_entries(treatment, start, end)
        return {
            "start_step": start,
            "end_step": end,
            "control_merge_entries": control_count,
            "treatment_merge_entries": treatment_count,
            "merge_entry_reduction": control_count - treatment_count,
        }

    during = window(during_start, during_end)
    post = window(post_start, post_end)
    return {
        "during_treatment": during,
        "equal_post_window": post,
        "combined_merge_entry_reduction": (
            during["merge_entry_reduction"] + post["merge_entry_reduction"]
        ),
    }


def _run_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    seed = int(payload["seed"])
    decision_steps = [int(step) for step in payload["decision_steps"]]
    outcome_steps = int(payload["outcome_steps"])
    treatment_steps = int(payload["treatment_steps"])
    checkpoint = str(payload["checkpoint"])
    metadata = dict(payload["controller_metadata"])
    common = {
        "project_root": Path(payload["sumo_project_root"]),
        "sumo_binary": str(payload["sumo_binary"]),
        "sumo_config": str(payload["sumo_config"]),
        "seed": seed,
        "cav_penetration": float(payload["cav_penetration"]),
        "sumo_step_length": float(payload["sumo_step_length"]),
        "treatment_steps": treatment_steps,
        "base_policy_factory": _base_policy_factory(checkpoint),
        "controller_metadata": metadata,
        "overlay_overrides": dict(payload.get("overlay_overrides") or {}),
    }
    definition = OutcomeDefinition()
    started = time.perf_counter()
    max_step = max(decision_steps) + outcome_steps
    control = run_replay(
        **common,
        decision_step=0,
        horizon_steps=max_step,
        intervention_schedule=lambda _step: 0,
    )

    cases = []
    for decision_step in decision_steps:
        treatment = run_replay(
            **common,
            decision_step=decision_step,
            # Every branch must have the same frozen episode horizon because
            # episode progress is part of the damping policy observation.
            horizon_steps=max_step - decision_step,
            intervention_schedule=FrozenPulseSchedule(
                decision_step=decision_step,
                treatment_steps=treatment_steps,
            ),
        )
        prefix = _prefix_identity(control, treatment, decision_step)
        contract = _audit_contract(
            control,
            treatment,
            decision_step=decision_step,
            treatment_steps=treatment_steps,
            step_length_s=float(payload["controller_step_length_s"]),
        )
        contract_path_checks = {
            key: bool(value)
            for key, value in contract["checks"].items()
            if key != "wall_engaged_at_least_once"
        }
        # A deterministic formation veto is a valid DAMPING_ONLY decision, not
        # a broken causal pair.  It contributes to action burden/availability
        # but is never labeled as a treated wall case.
        contract_path_passed = all(contract_path_checks.values())
        eligible = bool(prefix["passed"] and contract_path_passed)
        decision_snapshot = control.snapshots[decision_step]
        features = features_from_trace(
            control,
            decision_step=decision_step,
            definition=definition,
        )
        predecision_supervisor = evaluate_predecision_supervisor(
            features,
            str(payload.get("predecision_supervisor", NO_SUPERVISOR)),
        )
        row = {
            "schema_version": SCHEMA_VERSION,
            "case_id": (
                f"{payload['demand']}-p{int(round(float(payload['cav_penetration']) * 100))}"
                f"-seed-{seed}-step-{decision_step}"
            ),
            "seed": seed,
            "decision_step": decision_step,
            "decision_sim_time_s": float(decision_snapshot.sim_time),
            "decision_state_digest": decision_snapshot.digest,
            "eligible": eligible,
            "reason": None if eligible else "live_replay_contract_failed",
            "features": features,
            "predecision_supervisor": predecision_supervisor,
            "predecision_congestion": summarize_predecision_congestion(
                control,
                decision_step=decision_step,
                definition=definition,
            ),
            "prefix_identity": prefix,
            "overlay_action_contract": contract,
            "treatment_dose": contract["treatment_dose"],
            "first_stage_inflow": _first_stage_inflow(
                control, treatment, decision_step, treatment_steps
            ),
            "overlay_path_checks_excluding_formation_veto": contract_path_checks,
            "overlay_path_passed": contract_path_passed,
            "treatment_steps": treatment_steps,
            "outcome_steps": outcome_steps,
            "outcome_definition": definition.to_dict(),
            "scenario": {
                "demand": str(payload["demand"]),
                "hdv_model": "route-file Krauss mixture",
                "sumo_config": str(payload["sumo_config"]),
                "cav_penetration": float(payload["cav_penetration"]),
                "sumo_step_length": float(payload["sumo_step_length"]),
            },
            "arms": {
                "0": {
                    "name": "DAMPING_ONLY",
                    "trace_digest": control.trace_digest,
                    "outcomes": summarize_branch(
                        control,
                        decision_step=decision_step,
                        outcome_steps=outcome_steps,
                        definition=definition,
                    ),
                },
                "1": {
                    "name": "DAMPING_PLUS_WALL",
                    "trace_digest": treatment.trace_digest,
                    "wall_engaged": bool(contract["wall_override_steps"]),
                    "wall_override_step_count": len(contract["wall_override_steps"]),
                    "wall_override_ids": contract["wall_override_ids"],
                    "outcomes": summarize_branch(
                        treatment,
                        decision_step=decision_step,
                        outcome_steps=outcome_steps,
                        definition=definition,
                    ),
                },
            },
        }
        row["oracle"] = classify_case(row)
        row["safety_service_oracle"] = classify_safety_service_case(row)
        cases.append(row)

    return {
        "schema_version": SCHEMA_VERSION,
        "configuration_sha256": payload.get("configuration_sha256"),
        "seed": seed,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "control_trace_digest": control.trace_digest,
        "cases": cases,
    }


def _load_completed(
    path: Path,
    expected_seed: int,
    expected_configuration_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if int(payload.get("seed", -1)) != expected_seed:
        return None
    if payload.get("configuration_sha256") != expected_configuration_sha256:
        return None
    return payload


def main() -> int:
    args = parse_args()
    role = args.dataset_role or ("treatment_development" if args.development else "headroom_gate")
    if args.development and args.dataset_role not in (None, "treatment_development"):
        raise ValueError("--development is only compatible with treatment_development")
    seeds = validate_assigned_seeds(
        role=role,
        demand=str(args.demand),
        cav_penetration=float(args.cav_penetration),
        seeds=args.seeds,
    )
    if role == "headroom_gate" and (len(seeds) < 12 or len(seeds) > 20):
        raise ValueError("the v2 headroom gate requires 12--20 independent seeds")
    if role == "treatment_development" and (len(seeds) < 4 or len(seeds) > 20):
        raise ValueError("development pilots require 4--20 inspected seeds")
    if role in ("model_development", "locked_test", "transfer_test") and not (
        1 <= len(seeds) <= 20
    ):
        raise ValueError("dataset chunks require 1--20 assigned independent seeds")
    if role != "treatment_development" and len(args.decision_steps) not in (3, 4):
        raise ValueError("the v2 headroom gate requires three or four decision windows")
    if role == "treatment_development" and not (1 <= len(args.decision_steps) <= 4):
        raise ValueError("development pilots require one to four decision windows")
    if args.allow_partial_lane_wall and role != "treatment_development":
        raise ValueError(
            "--allow-partial-lane-wall is diagnostic-only and cannot define "
            "the corrected headroom treatment"
        )
    if not (0.0 <= float(args.wall_d_lo) < float(args.wall_d_hi)):
        raise ValueError("wall band must satisfy 0 <= wall-d-lo < wall-d-hi")
    if args.geometry_mode == "compact_moving" and role not in (
        "treatment_development",
        "headroom_gate",
    ):
        raise ValueError(
            "compact_moving is allowed only for treatment development or the "
            "formal headroom gate until treatment verification passes"
        )
    if args.geometry_mode == "compact_moving" and args.max_longitudinal_span_m is None:
        args.max_longitudinal_span_m = 50.0
    if args.geometry_mode == "compact_moving":
        if args.compact_release_boundary_m is None:
            args.compact_release_boundary_m = float(args.wall_d_lo)
        if not (0.0 <= float(args.compact_release_boundary_m) <= float(args.wall_d_lo)):
            raise ValueError("compact release boundary must satisfy 0 <= boundary <= wall-d-lo")
        planned_duration_s = float(args.treatment_steps) * CONTROLLER_STEP_LENGTH_S
        if not (0.0 <= float(args.compact_assembly_duration_s) < planned_duration_s):
            raise ValueError(
                "compact assembly duration must be nonnegative and shorter than the treatment"
            )
        if float(args.compact_assembly_speed_tolerance_mps) < 0.0:
            raise ValueError("compact assembly speed tolerance must be nonnegative")
        if float(args.compact_assembly_a_max_dec_mps2) < float(args.wall_a_max_dec):
            raise ValueError(
                "compact assembly deceleration bound cannot be gentler than "
                "the hold deceleration bound"
            )
        if args.compact_control_mode == "adaptive_servo" and float(
            args.compact_servo_a_max_dec_mps2
        ) < float(args.wall_a_max_dec):
            raise ValueError(
                "compact servo deceleration bound cannot be gentler than "
                "the base wall deceleration bound"
            )
        if not (0.0 <= float(args.compact_max_speed_reduction_fraction) <= 1.0):
            raise ValueError("compact max speed reduction fraction must lie in [0, 1]")
        if float(args.compact_shock_gap_hi_mps) <= float(args.compact_shock_gap_lo_mps):
            raise ValueError("compact shock gap hi must exceed lo")
        for value, name in (
            (
                args.compact_max_activation_speed_spread_mps,
                "compact max activation speed spread",
            ),
            (args.compact_position_kp_s_inv, "compact position gain"),
            (args.compact_position_kd, "compact position derivative gain"),
            (
                args.compact_position_speed_offset_cap_mps,
                "compact position speed offset cap",
            ),
            (args.compact_safety_min_front_ttc_s, "compact front TTC"),
            (args.compact_safety_min_rear_ttc_s, "compact rear TTC"),
            (args.compact_influence_upstream_m, "compact influence zone"),
            (args.compact_influence_min_ttc_s, "compact influence TTC"),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if int(args.compact_influence_max_followers) < 0:
            raise ValueError("compact influence follower count must be nonnegative")
    elif args.compact_control_mode != "fixed_target":
        raise ValueError("adaptive compact control requires compact_moving geometry")

    output_dir = args.output_dir.resolve()
    seed_dir = output_dir / "seeds"
    checkpoint = args.checkpoint.resolve()
    wall_params = {
        "d_lo": float(args.wall_d_lo),
        "d_hi": float(args.wall_d_hi),
        "wall_speed": float(args.wall_speed),
        "wall_tau": float(args.wall_tau),
        "a_max_dec": float(args.wall_a_max_dec),
        "tau_ramp_step": float(args.tau_ramp_step),
    }
    overlay_protocol = default_protocol_metadata()
    overlay_protocol["profile"] = str(args.overlay_profile)
    overlay_protocol["wall_params"] = dict(wall_params)
    overlay_protocol["geometry_mode"] = str(args.geometry_mode)
    overlay_protocol["compact_release_boundary_m"] = (
        float(args.compact_release_boundary_m) if args.geometry_mode == "compact_moving" else None
    )
    overlay_protocol["compact_selection_mode"] = str(args.compact_selection_mode)
    overlay_protocol["compact_assembly"] = {
        "duration_s": float(args.compact_assembly_duration_s),
        "speed_tolerance_mps": float(args.compact_assembly_speed_tolerance_mps),
        "a_max_dec_mps2": float(args.compact_assembly_a_max_dec_mps2),
        "target": "max(wall_speed, minimum_selected_member_speed)",
        "timeout": "explicit_treatment_failure",
    }
    overlay_protocol["compact_control"] = {
        "mode": str(args.compact_control_mode),
        "a_max_dec_mps2": float(args.compact_servo_a_max_dec_mps2),
        "max_activation_speed_spread_mps": float(args.compact_max_activation_speed_spread_mps),
        "max_speed_reduction_fraction": float(args.compact_max_speed_reduction_fraction),
        "shock_gap_lo_mps": float(args.compact_shock_gap_lo_mps),
        "shock_gap_hi_mps": float(args.compact_shock_gap_hi_mps),
        "position_kp_s_inv": float(args.compact_position_kp_s_inv),
        "position_kd": float(args.compact_position_kd),
        "position_speed_offset_cap_mps": float(args.compact_position_speed_offset_cap_mps),
        "safety_min_front_ttc_s": float(args.compact_safety_min_front_ttc_s),
        "safety_min_rear_ttc_s": float(args.compact_safety_min_rear_ttc_s),
        "influence_upstream_m": float(args.compact_influence_upstream_m),
        "influence_max_followers": int(args.compact_influence_max_followers),
        "influence_min_ttc_s": float(args.compact_influence_min_ttc_s),
        "common_target": (
            "predecision_speed_gap_bounded_fraction"
            if args.compact_control_mode == "adaptive_servo"
            else "fixed_wall_speed"
        ),
        "member_targets": (
            "bounded_cross_lane_position_servo_with_front_rear_projection"
            if args.compact_control_mode == "adaptive_servo"
            else "identical_common_target"
        ),
    }
    overlay_protocol["membership"] = (
        {
            "activation_lane_set": "frozen",
            "initial_members": "one_kinematically_feasible_cav_per_lane",
            "member_slots": "one_persistent_slot_per_activation_lane",
            "handoff": "none",
            "moving_members": True,
        }
        if args.geometry_mode == "compact_moving"
        else overlay_protocol["membership"]
    )
    metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_tree(checkpoint),
        "checkpoint_policy_sha256": _sha256_file(
            checkpoint / "policies" / "cav_policy" / "policy_state.pkl"
        ),
        "checkpoint_ray_version": "2.55.1",
        "checkpoint_python_runtime": "3.10",
        "overlay_protocol": overlay_protocol,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "seeds": [int(seed) for seed in seeds],
        "decision_steps": [int(step) for step in args.decision_steps],
        "outcome_steps": int(args.outcome_steps),
        "treatment_steps": int(args.treatment_steps),
        "cav_penetration": float(args.cav_penetration),
        "sumo_step_length": float(args.sumo_step_length),
        "controller_step_length_s": CONTROLLER_STEP_LENGTH_S,
        "sumo_project_root": str(args.sumo_project_root.resolve()),
        "sumo_binary": str(args.sumo_binary),
        "sumo_config": str(args.sumo_config),
        "demand": str(args.demand),
        "checkpoint": str(checkpoint),
        "controller_metadata": metadata,
        "outcome_definition": OutcomeDefinition().to_dict(),
        "headroom_margins": HeadroomMargins().to_dict(),
        "primary_estimand": str(args.primary_estimand),
        "predecision_supervisor": str(args.predecision_supervisor),
        "pilot_role": role,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "dataset_contract": contract_manifest(),
        "overlay_overrides": {
            "wall_params": wall_params,
            "require_all_occupied_lanes": not bool(args.allow_partial_lane_wall),
            "geometry_mode": str(args.geometry_mode),
            "max_longitudinal_span_m": args.max_longitudinal_span_m,
            "planned_treatment_duration_s": (
                float(args.treatment_steps) * CONTROLLER_STEP_LENGTH_S
                if args.geometry_mode == "compact_moving"
                else None
            ),
            "compact_release_boundary_m": (
                float(args.compact_release_boundary_m)
                if args.geometry_mode == "compact_moving"
                else None
            ),
            "compact_selection_mode": str(args.compact_selection_mode),
            "compact_assembly_duration_s": float(args.compact_assembly_duration_s),
            "compact_assembly_speed_tolerance_mps": float(
                args.compact_assembly_speed_tolerance_mps
            ),
            "compact_assembly_a_max_dec_mps2": float(args.compact_assembly_a_max_dec_mps2),
            "compact_control_mode": str(args.compact_control_mode),
            "compact_servo_a_max_dec_mps2": float(args.compact_servo_a_max_dec_mps2),
            "compact_max_activation_speed_spread_mps": float(
                args.compact_max_activation_speed_spread_mps
            ),
            "compact_max_speed_reduction_fraction": float(
                args.compact_max_speed_reduction_fraction
            ),
            "compact_shock_gap_lo_mps": float(args.compact_shock_gap_lo_mps),
            "compact_shock_gap_hi_mps": float(args.compact_shock_gap_hi_mps),
            "compact_position_kp_s_inv": float(args.compact_position_kp_s_inv),
            "compact_position_kd": float(args.compact_position_kd),
            "compact_position_speed_offset_cap_mps": float(
                args.compact_position_speed_offset_cap_mps
            ),
            "compact_safety_min_front_ttc_s": float(args.compact_safety_min_front_ttc_s),
            "compact_safety_min_rear_ttc_s": float(args.compact_safety_min_rear_ttc_s),
            "compact_influence_upstream_m": float(args.compact_influence_upstream_m),
            "compact_influence_max_followers": int(args.compact_influence_max_followers),
            "compact_influence_min_ttc_s": float(args.compact_influence_min_ttc_s),
        },
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config["configuration_sha256"] = config_hash
    _atomic_json(output_dir / "manifest.json", config)

    completed: dict[int, dict[str, Any]] = {}
    pending = []
    for seed in seeds:
        seed_path = seed_dir / f"seed_{int(seed)}.json"
        existing = _load_completed(seed_path, int(seed), config_hash)
        if existing is not None:
            completed[int(seed)] = existing
            print(f"seed={seed}: resume existing", flush=True)
            continue
        pending.append(
            {
                **config,
                "seed": int(seed),
                "checkpoint": str(checkpoint),
            }
        )

    started = time.perf_counter()
    if pending and not args.summarize_completed_only:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {executor.submit(_run_seed, item): int(item["seed"]) for item in pending}
            for future in as_completed(futures):
                seed = futures[future]
                result = future.result()
                _atomic_json(seed_dir / f"seed_{seed}.json", result)
                completed[seed] = result
                print(
                    f"seed={seed}: saved {len(result['cases'])} cases "
                    f"in {result['elapsed_wall_seconds']:.1f}s "
                    f"({len(completed)}/{len(seeds)})",
                    flush=True,
                )
    elif pending:
        if role == "headroom_gate" and len(completed) < 12:
            raise ValueError("a stopped headroom pilot requires at least 12 completed seeds")
        print(
            f"summarizing {len(completed)}/{len(seeds)} completed seeds; "
            f"skipping {len(pending)} pending seeds",
            flush=True,
        )

    remaining_seed_ids = sorted(
        int(item["seed"]) for item in pending if int(item["seed"]) not in completed
    )
    rows = [
        row
        for seed in sorted(completed)
        for row in sorted(completed[seed]["cases"], key=lambda item: item["decision_step"])
    ]
    summary = summarize_headroom(rows)
    repair_criteria = TreatmentRepairCriteria(
        require_collapse_prevention_diversity=(args.primary_estimand == "collapse_prevention")
    )
    treatment_repair = summarize_treatment_repair(rows, repair_criteria)
    incomplete_requested_cohort = bool(remaining_seed_ids and args.summarize_completed_only)
    primary_headroom_passed = bool(
        treatment_repair["passed"]
        and (args.primary_estimand == "safety_service" or summary["passed"])
    )
    combined_passed = bool(primary_headroom_passed and not incomplete_requested_cohort)
    summary["legacy_headroom_passed"] = bool(summary["passed"])
    summary["primary_estimand"] = str(args.primary_estimand)
    summary["primary_headroom_passed"] = primary_headroom_passed
    supervisor_records = [
        row.get("predecision_supervisor")
        or {
            "profile": NO_SUPERVISOR,
            "admitted": True,
        }
        for row in rows
    ]
    summary["predecision_supervisor"] = {
        "profile": str(args.predecision_supervisor),
        "admitted_case_count": sum(
            bool(record.get("admitted", True)) for record in supervisor_records
        ),
        "refused_case_count": sum(
            not bool(record.get("admitted", True)) for record in supervisor_records
        ),
    }
    summary["treatment_repair"] = treatment_repair
    summary["passed"] = combined_passed
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "configuration_sha256": config_hash,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_this_invocation_seconds": time.perf_counter() - started,
            "dataset_path": str((output_dir / "cases.jsonl").resolve()),
            "dataset_role": role,
            "pilot_completion": {
                "intended_seed_count": len(seeds),
                "completed_seed_count": len(completed),
                "completed_seeds": sorted(completed),
                "incomplete_seeds": remaining_seed_ids,
                "early_stopped": incomplete_requested_cohort,
                "early_stop_reason": (
                    str(args.early_stop_reason)
                    if incomplete_requested_cohort and args.early_stop_reason
                    else None
                ),
            },
            "verdict": (
                (
                    (
                        "headroom_failed_early_stop_do_not_train_predictor"
                        if incomplete_requested_cohort
                        else "headroom_failed_do_not_train_predictor"
                    )
                    if not summary["passed"]
                    else "headroom_passed_predictor_training_permitted"
                )
                if role == "headroom_gate"
                else "paired_dataset_chunk_completed"
            ),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    temporary_cases = cases_path.with_suffix(".jsonl.tmp")
    temporary_cases.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary_cases.replace(cases_path)
    _atomic_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "passed": summary["passed"],
                "verdict": summary["verdict"],
                "case_count": summary["case_count"],
                "wall_selected_case_count": summary["wall_selected_case_count"],
                "damping_selected_case_count": summary["damping_selected_case_count"],
                "safe_beneficial_seed_count": summary["safe_beneficial_seed_count"],
                "dose_qualified_fraction": summary["treatment_repair"]["dose_qualified_fraction"],
                "safety_admissible_fraction": summary["treatment_repair"][
                    "safety_admissible_fraction"
                ],
                "beneficial_completed_seed_count": summary["treatment_repair"][
                    "beneficial_completed_seed_count"
                ],
                "nonbeneficial_completed_seed_count": summary["treatment_repair"][
                    "nonbeneficial_completed_seed_count"
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {output_dir}")
    return 0 if role != "headroom_gate" or summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
