"""Control-only feasibility sweep for compact moving-wall recruitment.

This diagnostic replays frozen decentralized damping once per seed and audits
predeclared compact-wall geometries at the saved decision states.  It never
applies a wall, labels treatment benefit, or generates predictor training data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from config.constants import PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from config.defaults import DEFAULT_CONFIG
from counterfactual.twin_replay import StepSnapshot, run_replay
from policies.counterfactual_escalation import DEFAULT_DAMPING_CHECKPOINT
from policies.rllib_policy import RLlibPolicy


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
DEFAULT_SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit compact moving-wall formation without applying treatment."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(501, 507)))
    parser.add_argument("--decision-steps", type=int, nargs="+", default=[120, 180, 240])
    parser.add_argument("--release-boundaries-m", type=float, nargs="+", default=[0, 50, 100, 200])
    parser.add_argument("--durations-s", type=float, nargs="+", default=[20, 30])
    parser.add_argument("--span-caps-m", type=float, nargs="+", default=[50, 100])
    parser.add_argument(
        "--selection-modes",
        choices=["initial_compact", "trajectory_coherent"],
        nargs="+",
        default=["initial_compact", "trajectory_coherent"],
    )
    parser.add_argument("--d-lo", type=float, default=200.0)
    parser.add_argument("--d-hi", type=float, default=800.0)
    parser.add_argument("--wall-speed", type=float, default=10.0)
    parser.add_argument("--a-max-dec", type=float, default=0.3)
    parser.add_argument("--assembly-duration-s", type=float, default=10.0)
    parser.add_argument("--assembly-a-max-dec", type=float, default=1.0)
    parser.add_argument("--cav-penetration", type=float, default=0.2)
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
        "--output",
        type=Path,
        default=(PROJECT / "runs" / "probe" / "counterfactual_compact_feasibility_s501_506.json"),
    )
    return parser.parse_args()


def projected_wall_travel_m(
    speed_mps: float,
    duration_s: float,
    wall_speed_mps: float = 10.0,
    a_max_dec_mps2: float = 0.3,
    a_max_acc_mps2: float = 1.0,
) -> float:
    speed = max(0.0, float(speed_mps))
    duration = max(0.0, float(duration_s))
    target = max(0.0, float(wall_speed_mps))
    if speed > target:
        acceleration = max(1.0e-9, float(a_max_dec_mps2))
        ramp_time = min(duration, (speed - target) / acceleration)
        end_speed = max(target, speed - acceleration * ramp_time)
    else:
        acceleration = max(1.0e-9, float(a_max_acc_mps2))
        ramp_time = min(duration, (target - speed) / acceleration)
        end_speed = min(target, speed + acceleration * ramp_time)
    return 0.5 * (speed + end_speed) * ramp_time + target * (duration - ramp_time)


def audit_snapshot_scenario(
    snapshot: StepSnapshot,
    *,
    d_lo: float,
    d_hi: float,
    release_boundary_m: float,
    duration_s: float,
    span_cap_m: float,
    wall_speed_mps: float,
    a_max_dec_mps2: float,
    selection_mode: str = "initial_compact",
    assembly_duration_s: float = 10.0,
    assembly_a_max_dec_mps2: float = 1.0,
    controller_step_length_s: float = float(DEFAULT_CONFIG["simulation_step"]),
) -> dict[str, Any]:
    if selection_mode not in ("initial_compact", "trajectory_coherent"):
        raise ValueError("unknown compact selection mode")
    active_cavs = {str(vehicle_id) for vehicle_id in snapshot.active_cavs}
    occupied_lanes: set[str] = set()
    candidates_by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in snapshot.vehicles:
        if str(row[1]) != PRE_MERGE_EDGE:
            continue
        distance = PRE_MERGE_LANE_LENGTH - float(row[3])
        if not (float(d_lo) <= distance <= float(d_hi)):
            continue
        vehicle_id = str(row[0])
        lane_id = str(row[2])
        occupied_lanes.add(lane_id)
        if vehicle_id not in active_cavs:
            continue
        candidates_by_lane.setdefault(lane_id, []).append(
            {
                "vehicle_id": vehicle_id,
                "distance_to_merge_m": distance,
                "speed_mps": float(row[4]),
            }
        )

    required_lanes = tuple(sorted(occupied_lanes))
    if not required_lanes:
        return {
            "feasible": False,
            "reason": "no_occupied_recruitment_lane",
            "required_lane_ids": required_lanes,
            "selected_cav_ids": (),
            "longitudinal_span_m": None,
        }
    if any(not candidates_by_lane.get(lane_id) for lane_id in required_lanes):
        return {
            "feasible": False,
            "reason": "uncovered_occupied_lane",
            "required_lane_ids": required_lanes,
            "selected_cav_ids": (),
            "longitudinal_span_m": None,
        }

    headroom_candidates: list[list[dict[str, Any]]] = []
    for lane_id in required_lanes:
        candidates = []
        for candidate in candidates_by_lane[lane_id]:
            projected = projected_wall_travel_m(
                candidate["speed_mps"],
                duration_s,
                wall_speed_mps,
                a_max_dec_mps2,
            )
            if (
                selection_mode == "trajectory_coherent"
                or candidate["distance_to_merge_m"] - projected >= release_boundary_m
            ):
                profile_times = [
                    min(float(duration_s), 0.5 * index)
                    for index in range(int(float(duration_s) / 0.5) + 1)
                ]
                if not profile_times or profile_times[-1] < float(duration_s):
                    profile_times.append(float(duration_s))
                candidates.append(
                    {
                        **candidate,
                        "projected_travel_m": projected,
                        "profile_times_s": tuple(profile_times),
                        "projected_distances_m": tuple(
                            candidate["distance_to_merge_m"]
                            - projected_wall_travel_m(
                                candidate["speed_mps"],
                                time_s,
                                wall_speed_mps,
                                a_max_dec_mps2,
                            )
                            for time_s in profile_times
                        ),
                    }
                )
        if not candidates:
            return {
                "feasible": False,
                "reason": "insufficient_kinematic_headroom",
                "required_lane_ids": required_lanes,
                "selected_cav_ids": (),
                "longitudinal_span_m": None,
            }
        headroom_candidates.append(candidates)

    def spans(selected: tuple[Mapping[str, Any], ...]) -> tuple[float, float, float, float]:
        if selection_mode == "trajectory_coherent":
            distances_by_id = {
                str(item["vehicle_id"]): float(item["distance_to_merge_m"]) for item in selected
            }
            speeds_by_id = {str(item["vehicle_id"]): float(item["speed_mps"]) for item in selected}
            ids = tuple(sorted(distances_by_id))
            initial_span = max(distances_by_id.values()) - min(distances_by_id.values())
            maximum_span = initial_span
            duration = float(duration_s)
            assembly_duration = min(float(assembly_duration_s), duration)
            assembly_target = max(float(wall_speed_mps), min(speeds_by_id.values()))
            elapsed = 0.0
            while elapsed < duration - 1.0e-9:
                dt_s = min(float(controller_step_length_s), duration - elapsed)
                assembling = elapsed < assembly_duration - 1.0e-9
                target = assembly_target if assembling else float(wall_speed_mps)
                deceleration = (
                    float(assembly_a_max_dec_mps2) if assembling else float(a_max_dec_mps2)
                )
                for vehicle_id in ids:
                    old_speed = speeds_by_id[vehicle_id]
                    if old_speed > target:
                        new_speed = max(target, old_speed - deceleration * dt_s)
                    else:
                        new_speed = min(target, old_speed + 1.0 * dt_s)
                    distances_by_id[vehicle_id] -= 0.5 * (old_speed + new_speed) * dt_s
                    speeds_by_id[vehicle_id] = new_speed
                elapsed += dt_s
                maximum_span = max(
                    maximum_span,
                    max(distances_by_id.values()) - min(distances_by_id.values()),
                )
            terminal_span = max(distances_by_id.values()) - min(distances_by_id.values())
            return (
                initial_span,
                maximum_span,
                terminal_span,
                min(distances_by_id.values()),
            )
        distances = [float(item["distance_to_merge_m"]) for item in selected]
        time_count = len(selected[0]["projected_distances_m"])
        time_spans = [
            max(float(item["projected_distances_m"][index]) for item in selected)
            - min(float(item["projected_distances_m"][index]) for item in selected)
            for index in range(time_count)
        ]
        terminal_distances = [float(item["projected_distances_m"][-1]) for item in selected]
        return (
            max(distances) - min(distances),
            max(time_spans),
            time_spans[-1],
            min(terminal_distances),
        )

    def score(selected: tuple[Mapping[str, Any], ...]) -> tuple[Any, ...]:
        distances = [float(item["distance_to_merge_m"]) for item in selected]
        (
            initial_span,
            predicted_max_span,
            terminal_span,
            minimum_terminal_distance,
        ) = spans(selected)
        speeds = [float(item["speed_mps"]) for item in selected]
        ids = tuple(str(item["vehicle_id"]) for item in selected)
        if selection_mode == "trajectory_coherent":
            return (
                minimum_terminal_distance < float(release_boundary_m),
                predicted_max_span,
                terminal_span,
                max(speeds) - min(speeds),
                initial_span,
                -min(distances),
                ids,
            )
        return (
            initial_span,
            predicted_max_span,
            max(speeds) - min(speeds),
            -min(distances),
            ids,
        )

    selected = min(product(*headroom_candidates), key=score)
    distances = [float(item["distance_to_merge_m"]) for item in selected]
    (
        span,
        predicted_max_span,
        terminal_span,
        minimum_terminal_distance,
    ) = spans(selected)
    if minimum_terminal_distance < float(release_boundary_m):
        return {
            "feasible": False,
            "reason": "insufficient_kinematic_headroom",
            "selection_mode": selection_mode,
            "required_lane_ids": required_lanes,
            "selected_cav_ids": tuple(sorted(str(item["vehicle_id"]) for item in selected)),
            "longitudinal_span_m": span,
            "predicted_max_span_m": predicted_max_span,
            "predicted_terminal_span_m": terminal_span,
        }
    feasible = span <= float(span_cap_m)
    reason = "feasible" if feasible else "formation_span_exceeded"
    if (
        feasible
        and selection_mode == "trajectory_coherent"
        and predicted_max_span > float(span_cap_m)
    ):
        feasible = False
        reason = "predicted_trajectory_span_exceeded"
    return {
        "feasible": feasible,
        "reason": reason,
        "selection_mode": selection_mode,
        "required_lane_ids": required_lanes,
        "selected_cav_ids": tuple(sorted(str(item["vehicle_id"]) for item in selected)),
        "selected_distances_to_merge_m": tuple(sorted(distances)),
        "selected_speeds_mps": tuple(float(item["speed_mps"]) for item in selected),
        "longitudinal_span_m": span,
        "predicted_max_span_m": predicted_max_span,
        "predicted_terminal_span_m": terminal_span,
    }


def _run_seed(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    checkpoint = str(payload["checkpoint"])
    trace = run_replay(
        project_root=Path(payload["sumo_project_root"]),
        sumo_binary=str(payload["sumo_binary"]),
        sumo_config=str(payload["sumo_config"]),
        seed=int(payload["seed"]),
        decision_step=0,
        horizon_steps=max(int(x) for x in payload["decision_steps"]),
        treatment_steps=1,
        cav_penetration=float(payload["cav_penetration"]),
        sumo_step_length=float(payload["sumo_step_length"]),
        base_policy_factory=lambda: RLlibPolicy(checkpoint, explore=False),
        intervention_schedule=lambda _step: 0,
        controller_metadata={
            "purpose": "control_only_compact_formation_feasibility",
            "checkpoint_path": checkpoint,
        },
    )
    snapshots = {int(row.relative_step): row for row in trace.snapshots}
    decisions = []
    for decision_step in payload["decision_steps"]:
        snapshot = snapshots[int(decision_step)]
        scenarios = []
        for boundary, duration, span, selection_mode in product(
            payload["release_boundaries_m"],
            payload["durations_s"],
            payload["span_caps_m"],
            payload["selection_modes"],
        ):
            audit = audit_snapshot_scenario(
                snapshot,
                d_lo=float(payload["d_lo"]),
                d_hi=float(payload["d_hi"]),
                release_boundary_m=float(boundary),
                duration_s=float(duration),
                span_cap_m=float(span),
                wall_speed_mps=float(payload["wall_speed"]),
                a_max_dec_mps2=float(payload["a_max_dec"]),
                selection_mode=str(selection_mode),
                assembly_duration_s=float(payload["assembly_duration_s"]),
                assembly_a_max_dec_mps2=float(payload["assembly_a_max_dec"]),
                controller_step_length_s=float(payload["controller_step_length_s"]),
            )
            scenarios.append(
                {
                    "release_boundary_m": float(boundary),
                    "duration_s": float(duration),
                    "span_cap_m": float(span),
                    "selection_mode": str(selection_mode),
                    **audit,
                }
            )
        decisions.append(
            {
                "seed": int(payload["seed"]),
                "decision_step": int(decision_step),
                "decision_state_digest": snapshot.digest,
                "scenarios": scenarios,
            }
        )
    return {
        "seed": int(payload["seed"]),
        "control_trace_digest": trace.trace_digest,
        "terminated_early": trace.terminated_early,
        "decisions": decisions,
    }


def _scenario_key(row: Mapping[str, Any]) -> tuple[str, float, float, float]:
    return (
        str(row["selection_mode"]),
        float(row["release_boundary_m"]),
        float(row["duration_s"]),
        float(row["span_cap_m"]),
    )


def summarize(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, float, float], list[Mapping[str, Any]]] = {}
    for seed_result in results:
        for decision in seed_result["decisions"]:
            for scenario in decision["scenarios"]:
                grouped.setdefault(_scenario_key(scenario), []).append(
                    {
                        **scenario,
                        "seed": int(decision["seed"]),
                        "decision_step": int(decision["decision_step"]),
                    }
                )
    summaries = []
    for key, rows in sorted(grouped.items()):
        feasible = [row for row in rows if bool(row["feasible"])]
        reason_counts: dict[str, int] = {}
        for row in rows:
            reason = str(row["reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        spans = [
            float(row["longitudinal_span_m"])
            for row in feasible
            if row.get("longitudinal_span_m") is not None
        ]
        feasible_seeds = sorted({int(row["seed"]) for row in feasible})
        summaries.append(
            {
                "selection_mode": key[0],
                "release_boundary_m": key[1],
                "duration_s": key[2],
                "span_cap_m": key[3],
                "case_count": len(rows),
                "feasible_case_count": len(feasible),
                "feasible_fraction": len(feasible) / len(rows) if rows else 0.0,
                "feasible_seed_count": len(feasible_seeds),
                "feasible_seeds": feasible_seeds,
                "mean_selected_span_m": fmean(spans) if spans else None,
                "reason_counts": reason_counts,
                "passes_pilot_availability_gate": (len(feasible) >= 6 and len(feasible_seeds) >= 3),
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.d_lo < args.d_hi):
        raise ValueError("recruitment band must satisfy 0 <= d_lo < d_hi")
    if any(not (0.0 <= value <= args.d_lo) for value in args.release_boundaries_m):
        raise ValueError("release boundaries must lie between 0 and d_lo")
    common = {
        "decision_steps": [int(x) for x in args.decision_steps],
        "release_boundaries_m": [float(x) for x in args.release_boundaries_m],
        "durations_s": [float(x) for x in args.durations_s],
        "span_caps_m": [float(x) for x in args.span_caps_m],
        "selection_modes": [str(x) for x in args.selection_modes],
        "d_lo": float(args.d_lo),
        "d_hi": float(args.d_hi),
        "wall_speed": float(args.wall_speed),
        "a_max_dec": float(args.a_max_dec),
        "assembly_duration_s": float(args.assembly_duration_s),
        "assembly_a_max_dec": float(args.assembly_a_max_dec),
        "cav_penetration": float(args.cav_penetration),
        "sumo_step_length": float(args.sumo_step_length),
        "controller_step_length_s": float(DEFAULT_CONFIG["simulation_step"]),
        "sumo_project_root": str(args.sumo_project_root.resolve()),
        "sumo_binary": str(args.sumo_binary),
        "sumo_config": str(args.sumo_config),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    results = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(_run_seed, {**common, "seed": int(seed)}): int(seed)
            for seed in args.seeds
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"seed={result['seed']}: control feasibility saved", flush=True)
    results.sort(key=lambda row: int(row["seed"]))
    summary = summarize(results)
    eligible = [
        row
        for row in summary
        if row["passes_pilot_availability_gate"]
        and row["selection_mode"] == "trajectory_coherent"
        and row["span_cap_m"] == 50.0
    ]
    recommended = (
        max(
            eligible,
            key=lambda row: (row["duration_s"], row["release_boundary_m"]),
        )
        if eligible
        else None
    )
    report = {
        "schema_version": "counterfactual_compact_formation_feasibility/1",
        "purpose": "control_only_treatment_definition_audit_not_predictor_data",
        "configuration": common,
        "selection_rule": {
            "minimum_feasible_cases": 6,
            "minimum_independent_seeds": 3,
            "selection_mode": "trajectory_coherent",
            "preserve_span_cap_m": 50.0,
            "choose_longest_duration_then_highest_release_boundary": True,
        },
        "recommended_scenario": recommended,
        "scenario_summary": summary,
        "seeds": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "recommended_scenario": recommended,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
