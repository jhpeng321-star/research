"""Verify the v2 damping-preserving counterfactual replay contract.

This is the mandatory go/no-go test before generating a corrected headroom
pilot or training an escalation predictor.  It deliberately runs four fresh
branches per seed: two damping-only repeats and two damping-plus-wall repeats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from config.constants import PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from config.defaults import DEFAULT_CONFIG
from counterfactual.outcomes import OutcomeDefinition, summarize_branch
from counterfactual.twin_replay import (
    FrozenPulseSchedule,
    ReplayTrace,
    compare_traces,
    run_replay,
)
from policies.counterfactual_escalation import (
    DEFAULT_DAMPING_CHECKPOINT,
    default_protocol_metadata,
)
from policies.rllib_policy import RLlibPolicy


from experiments import REPO_ROOT

PROJECT = REPO_ROOT
DEFAULT_SUMO_ROOT = Path(r"D:\research\highway_merge_shepherd")
DEFAULT_SUMO_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
CONTROLLER_STEP_LENGTH_S = float(DEFAULT_CONFIG["simulation_step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-seed v2 damping/overlay acceptance test."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102])
    parser.add_argument("--decision-step", type=int, default=180)
    parser.add_argument("--horizon-steps", type=int, default=180)
    parser.add_argument("--treatment-steps", type=int, default=60)
    parser.add_argument("--cav-penetration", type=float, default=0.2)
    parser.add_argument("--sumo-step-length", type=float, default=0.5)
    parser.add_argument("--sumo-project-root", type=Path, default=DEFAULT_SUMO_ROOT)
    parser.add_argument("--sumo-config", default="osm.hyper.sumocfg")
    parser.add_argument(
        "--sumo-binary",
        default=str(DEFAULT_SUMO_EXE if DEFAULT_SUMO_EXE.exists() else "sumo"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DAMPING_CHECKPOINT)
    parser.add_argument(
        "--geometry-mode",
        choices=["static_all", "compact_moving"],
        default="static_all",
    )
    parser.add_argument("--wall-d-lo", type=float, default=200.0)
    parser.add_argument("--wall-d-hi", type=float, default=400.0)
    parser.add_argument("--wall-speed", type=float, default=10.0)
    parser.add_argument("--wall-tau", type=float, default=3.0)
    parser.add_argument("--wall-a-max-dec", type=float, default=0.3)
    parser.add_argument("--tau-ramp-step", type=float, default=0.05)
    parser.add_argument("--max-longitudinal-span-m", type=float, default=None)
    parser.add_argument("--compact-release-boundary-m", type=float, default=None)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT / "runs" / "probe" / "counterfactual_escalation_v2_twin_check.json"),
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


def _physical_state(snapshot: Any) -> tuple[Any, ...]:
    """State fingerprint without the branch command itself."""
    return (
        snapshot.relative_step,
        snapshot.sim_time,
        snapshot.supervisor_observation,
        snapshot.vehicles,
        snapshot.active_cavs,
        snapshot.collision_vehicle_ids,
        snapshot.starting_teleport_vehicle_ids,
    )


def _first_physical_divergence(left: ReplayTrace, right: ReplayTrace) -> int | None:
    for left_step, right_step in zip(left.snapshots, right.snapshots):
        if _physical_state(left_step) != _physical_state(right_step):
            return int(left_step.relative_step)
    if len(left.snapshots) != len(right.snapshots):
        return min(len(left.snapshots), len(right.snapshots))
    return None


def _actions(audit: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = audit.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _audit_contract(
    control: ReplayTrace,
    treatment: ReplayTrace,
    *,
    decision_step: int,
    treatment_steps: int,
    step_length_s: float = 0.5,
) -> dict[str, Any]:
    control_audits = {int(row["relative_step"]): row for row in control.controller_audits}
    treatment_audits = {int(row["relative_step"]): row for row in treatment.controller_audits}
    common_steps = sorted(set(control_audits) & set(treatment_audits))

    prefix_base_action_identity = all(
        _actions(control_audits[step], "base_actions")
        == _actions(treatment_audits[step], "base_actions")
        and control_audits[step].get("base_diagnostics", {})
        == treatment_audits[step].get("base_diagnostics", {})
        and int(control_audits[step].get("command", 0)) == 0
        and int(treatment_audits[step].get("command", 0)) == 0
        for step in common_steps
        if 1 <= step <= decision_step
    )

    treatment_window = range(decision_step + 1, decision_step + treatment_steps + 1)
    treatment_rows = [
        treatment_audits[step] for step in treatment_window if step in treatment_audits
    ]
    wall_override_steps = [
        int(row["relative_step"]) for row in treatment_rows if row.get("wall_override_ids")
    ]
    transition_reason_counts: dict[str, int] = {}
    for row in treatment_rows:
        reason = str(row.get("transition_reason") or "unspecified")
        transition_reason_counts[reason] = transition_reason_counts.get(reason, 0) + 1
    wall_active_steps = [
        int(row["relative_step"]) for row in treatment_rows if bool(row.get("wall_active"))
    ]
    assembly_steps = [
        int(row["relative_step"])
        for row in treatment_rows
        if str(row.get("treatment_state")) == "ASSEMBLING" and row.get("wall_override_ids")
    ]
    holding_steps = [
        int(row["relative_step"])
        for row in treatment_rows
        if str(row.get("treatment_state")) == "HOLDING" and row.get("wall_override_ids")
    ]
    taper_steps = [
        int(row["relative_step"])
        for row in treatment_rows
        if (
            str(row.get("treatment_state")) == "TAPERING" and row.get("wall_override_ids")
        )
        or bool(row.get("emergency_taper_zero_handoff"))
    ]
    taper_rows = [row for row in treatment_rows if int(row["relative_step"]) in taper_steps]
    taper_blends = [
        float(row["taper_blend_fraction"])
        for row in taper_rows
        if row.get("taper_blend_fraction") is not None
    ]
    controlled_cav_steps = sum(len(row.get("wall_override_ids", ())) for row in treatment_rows)
    controlled_lane_steps = sum(
        int((row.get("lane_coverage") or {}).get("covered_lane_count", 0)) for row in treatment_rows
    )
    requested_lane_steps = sum(
        int((row.get("lane_coverage") or {}).get("required_lane_count", 0))
        for row in treatment_rows
    )
    lane_coverage_fraction = (
        controlled_lane_steps / requested_lane_steps if requested_lane_steps else 0.0
    )
    member_changes = [
        {
            "relative_step": int(row["relative_step"]),
            **dict(change),
        }
        for row in treatment_rows
        for change in row.get("member_changes", ())
    ]
    member_change_counts: dict[str, int] = {}
    for change in member_changes:
        event = str(change.get("event") or "unspecified")
        member_change_counts[event] = member_change_counts.get(event, 0) + 1
    abort_events = [
        {
            "relative_step": int(row["relative_step"]),
            "reason": row.get("abort_reason")
            or str(row.get("transition_reason", "")).partition(":")[2],
            "transition_reason": row.get("transition_reason"),
        }
        for row in treatment_rows
        if (
            row.get("abort_reason")
            or str(row.get("transition_reason", "")).startswith(
                ("safety_abort:", "assembly_abort:")
            )
        )
    ]
    safety_abort_events = [
        event
        for event in abort_events
        if str(event.get("transition_reason", "")).startswith("safety_abort:")
    ]
    assembly_abort_events = [
        event
        for event in abort_events
        if str(event.get("transition_reason", "")).startswith("assembly_abort:")
    ]
    control_deltas = [
        {
            "relative_step": int(row["relative_step"]),
            "vehicle_id": str(vehicle_id),
            **dict(delta),
        }
        for row in treatment_rows
        for vehicle_id, delta in _actions(row, "control_deltas").items()
    ]
    safety_projection_clip_reason_counts: dict[str, int] = {}
    member_target_sample_count = 0
    for delta in control_deltas:
        if delta.get("member_projected_speed_target_mps") is not None:
            member_target_sample_count += 1
        projection = delta.get("safety_projection") or {}
        for reason in projection.get("clip_reasons", ()):
            reason = str(reason)
            safety_projection_clip_reason_counts[reason] = (
                safety_projection_clip_reason_counts.get(reason, 0) + 1
            )
    adaptive_reduction_samples = [
        float(row["adaptive_speed_reduction_fraction"])
        for row in treatment_rows
        if row.get("wall_override_ids") and row.get("adaptive_speed_reduction_fraction") is not None
    ]
    adaptive_common_target_samples = [
        float(row["adaptive_common_target_mps"])
        for row in treatment_rows
        if row.get("wall_override_ids") and row.get("adaptive_common_target_mps") is not None
    ]
    wall_deceleration_bounds = [
        float(delta["wall_a_max_dec_mps2"])
        for delta in control_deltas
        if delta.get("wall_a_max_dec_mps2") is not None
    ]
    commanded_deceleration_samples = [
        max(
            0.0,
            (
                float(delta["pre_override_speed_mps"])
                - float(delta["member_applied_speed_command_mps"])
            )
            / float(step_length_s),
        )
        for delta in control_deltas
        if delta.get("pre_override_speed_mps") is not None
        and delta.get("member_applied_speed_command_mps") is not None
        and float(step_length_s) > 0.0
    ]
    holding_control_deltas = [
        delta
        for delta in control_deltas
        if str(delta.get("treatment_phase", "")).upper() == "HOLDING"
    ]

    def _speed_pair(delta: Mapping[str, Any]) -> tuple[float, float] | None:
        actual = delta.get("pre_override_speed_mps")
        target = delta.get("member_projected_speed_target_mps")
        if actual is None or target is None:
            return None
        return float(actual), float(target)

    holding_speed_pairs = [
        pair for delta in holding_control_deltas if (pair := _speed_pair(delta)) is not None
    ]
    holding_steps_with_targets = sorted(
        {
            int(delta["relative_step"])
            for delta in holding_control_deltas
            if _speed_pair(delta) is not None
        }
    )
    terminal_tail_steps = set(holding_steps_with_targets[-5:])
    terminal_tail_speed_pairs = [
        pair
        for delta in holding_control_deltas
        if int(delta["relative_step"]) in terminal_tail_steps
        and (pair := _speed_pair(delta)) is not None
    ]

    def _speed_delivery_summary(
        pairs: Sequence[tuple[float, float]],
    ) -> dict[str, Any]:
        errors = [actual - target for actual, target in pairs]
        return {
            "sample_count": len(pairs),
            "mean_actual_speed_mps": (
                fmean(actual for actual, _target in pairs) if pairs else None
            ),
            "mean_projected_target_mps": (
                fmean(target for _actual, target in pairs) if pairs else None
            ),
            "mean_speed_above_target_mps": fmean(errors) if errors else None,
            "mean_absolute_target_error_mps": (
                fmean(abs(error) for error in errors) if errors else None
            ),
            "fraction_within_1_mps": (
                sum(abs(error) <= 1.0 for error in errors) / len(errors) if errors else None
            ),
        }

    holding_speed_delivery = _speed_delivery_summary(holding_speed_pairs)
    terminal_tail_speed_delivery = _speed_delivery_summary(terminal_tail_speed_pairs)
    control_snapshots = {int(snapshot.relative_step): snapshot for snapshot in control.snapshots}
    treatment_snapshots = {
        int(snapshot.relative_step): snapshot for snapshot in treatment.snapshots
    }
    geometry_samples = []
    for row in treatment_rows:
        member_ids = tuple(str(x) for x in row.get("wall_override_ids", ()))
        if not member_ids:
            continue
        snapshot = treatment_snapshots.get(int(row["relative_step"]))
        if snapshot is None:
            continue
        vehicle_map = snapshot.vehicle_map()
        distances = []
        speeds = []
        for member_id in member_ids:
            vehicle = vehicle_map.get(member_id)
            if vehicle is None or str(vehicle[1]) != PRE_MERGE_EDGE:
                continue
            try:
                distances.append(PRE_MERGE_LANE_LENGTH - float(vehicle[3]))
                speeds.append(float(vehicle[4]))
            except (TypeError, ValueError):
                continue
        if not distances:
            continue
        geometry_samples.append(
            {
                "relative_step": int(row["relative_step"]),
                "treatment_state": row.get("treatment_state"),
                "member_count": len(distances),
                "longitudinal_span_m": max(distances) - min(distances),
                "member_speed_std_mps": pstdev(speeds) if len(speeds) > 1 else 0.0,
            }
        )

    def physical_delta(
        left_row: Sequence[Any] | None,
        right_row: Sequence[Any] | None,
        index: int,
    ) -> float | None:
        if left_row is None or right_row is None:
            return None
        if len(left_row) <= index or len(right_row) <= index:
            return None
        try:
            return float(right_row[index]) - float(left_row[index])
        except (TypeError, ValueError):
            return None

    realized_control_deltas = []
    for row in treatment_rows:
        step = int(row["relative_step"])
        control_snapshot = control_snapshots.get(step)
        treatment_snapshot = treatment_snapshots.get(step)
        if control_snapshot is None or treatment_snapshot is None:
            continue
        control_vehicles = control_snapshot.vehicle_map()
        treatment_vehicles = treatment_snapshot.vehicle_map()
        for vehicle_id in row.get("wall_override_ids", ()):
            vehicle_id = str(vehicle_id)
            left = control_vehicles.get(vehicle_id)
            right = treatment_vehicles.get(vehicle_id)
            realized_control_deltas.append(
                {
                    "relative_step": step,
                    "vehicle_id": vehicle_id,
                    "speed_delta_vs_damping_mps": physical_delta(left, right, 4),
                    "acceleration_delta_vs_damping_mps2": physical_delta(left, right, 5),
                    "time_loss_delta_vs_damping_s": physical_delta(left, right, 6),
                    "tau_delta_vs_damping_s": physical_delta(left, right, 9),
                }
            )
    override_runs: list[int] = []
    current_run = 0
    for row in treatment_rows:
        if row.get("wall_override_ids"):
            current_run += 1
        elif current_run:
            override_runs.append(current_run)
            current_run = 0
    if current_run:
        override_runs.append(current_run)
    first_post_override_transition = None
    if wall_override_steps:
        for row in treatment_rows:
            if int(row["relative_step"]) > wall_override_steps[0] and not row.get(
                "wall_override_ids"
            ):
                first_post_override_transition = {
                    "relative_step": int(row["relative_step"]),
                    "transition_reason": row.get("transition_reason"),
                    "formation": row.get("formation"),
                    "activation_safety": row.get("activation_safety"),
                }
                break
    override_ids = sorted(
        {
            str(vehicle_id)
            for row in treatment_rows
            for vehicle_id in row.get("wall_override_ids", [])
        }
    )

    out_of_band_preserved = True
    wall_ids_removed_from_env_actions = True
    for row in treatment_rows:
        base = _actions(row, "base_actions")
        applied = _actions(row, "applied_env_actions")
        direct_ids = set(str(x) for x in row.get("base_direct_control_ids", ()))
        base_control_ids = set(
            str(x) for x in row.get("base_control_ids", tuple(base) + tuple(direct_ids))
        )
        wall_ids = set(str(x) for x in row.get("wall_override_ids", ()))
        for vehicle_id in row.get("out_of_band_damping_ids", []):
            vehicle_id = str(vehicle_id)
            if vehicle_id in direct_ids:
                if vehicle_id not in base_control_ids or vehicle_id in wall_ids:
                    out_of_band_preserved = False
            elif vehicle_id not in base or applied.get(vehicle_id) != base[vehicle_id]:
                out_of_band_preserved = False
        if any(vehicle_id in applied for vehicle_id in row.get("wall_override_ids", [])):
            wall_ids_removed_from_env_actions = False

    outside_rows = [
        row
        for step, row in treatment_audits.items()
        if step <= decision_step or step > decision_step + treatment_steps
    ]
    override_only_during_treatment = all(
        not row.get("wall_override_ids") and not bool(row.get("wall_active"))
        for row in outside_rows
    )
    post_release_rows = [
        row for step, row in treatment_audits.items() if step > decision_step + treatment_steps
    ]
    release_returns_to_damping = bool(post_release_rows) and all(
        int(row.get("command", -1)) == 0
        and not row.get("wall_override_ids")
        and _actions(row, "applied_env_actions") == _actions(row, "base_actions")
        and set(str(x) for x in row.get("base_direct_control_ids", ()))
        <= set(str(x) for x in row.get("base_control_ids", ()))
        for row in post_release_rows
    )

    first_treatment_step = decision_step + 1
    first_control = control_audits.get(first_treatment_step, {})
    first_treatment = treatment_audits.get(first_treatment_step, {})
    first_request_same_base_actions = (
        bool(first_control)
        and bool(first_treatment)
        and _actions(first_control, "base_actions") == _actions(first_treatment, "base_actions")
        and first_control.get("base_diagnostics", {}) == first_treatment.get("base_diagnostics", {})
    )
    first_treatment_reason = first_treatment.get("transition_reason")
    activation_feasible = first_treatment_reason in (
        "wall_started",
        "wall_assembly_started",
    )
    compact_requested = first_treatment.get("geometry_mode") == "compact_moving"

    checks = {
        "prefix_base_action_identity": prefix_base_action_identity,
        "first_request_same_base_actions": first_request_same_base_actions,
        "wall_engaged_at_least_once": bool(wall_override_steps),
        "override_only_during_treatment": override_only_during_treatment,
        "out_of_band_damping_preserved": out_of_band_preserved,
        "wall_ids_removed_from_env_action_path": wall_ids_removed_from_env_actions,
        "release_returns_to_damping": release_returns_to_damping,
        "runtime_formation_loss_is_not_an_abort": not any(
            reason.startswith("formation_abort:") for reason in transition_reason_counts
        ),
        "compact_assembly_reaches_holding": (
            not compact_requested or not activation_feasible or bool(holding_steps)
        ),
        "taper_is_monotone_and_reaches_zero": (
            not taper_rows
            or (
                len(taper_blends) == len(taper_rows)
                and all(left >= right for left, right in zip(taper_blends, taper_blends[1:]))
                and taper_blends[-1] == 0.0
            )
        ),
    }
    requested_duration_steps = max(0, int(treatment_steps))
    realized_duration_steps = len(wall_override_steps)
    realized_duration_fraction = (
        realized_duration_steps / requested_duration_steps if requested_duration_steps else 0.0
    )
    treatment_dose = {
        "activation_feasible": activation_feasible,
        "requested_duration_steps": requested_duration_steps,
        "requested_duration_s": requested_duration_steps * float(step_length_s),
        "wall_active_duration_steps": len(wall_active_steps),
        "wall_active_duration_s": len(wall_active_steps) * float(step_length_s),
        "realized_override_duration_steps": realized_duration_steps,
        "realized_override_duration_s": realized_duration_steps * float(step_length_s),
        "realized_duration_fraction": realized_duration_fraction,
        "assembly_duration_steps": len(assembly_steps),
        "assembly_duration_s": len(assembly_steps) * float(step_length_s),
        "holding_duration_steps": len(holding_steps),
        "holding_duration_s": len(holding_steps) * float(step_length_s),
        "taper_duration_steps": len(taper_steps),
        "taper_duration_s": len(taper_steps) * float(step_length_s),
        "taper_blend_fractions": taper_blends,
        "emergency_taper_zero_handoff_steps": [
            int(row["relative_step"])
            for row in taper_rows
            if bool(row.get("emergency_taper_zero_handoff"))
        ],
        "taper_reasons": sorted(
            {
                str(row.get("taper_reason"))
                for row in taper_rows
                if row.get("taper_reason") is not None
            }
        ),
        "taper_base_target_sample_count": sum(
            len(row.get("taper_base_targets_mps") or {}) for row in taper_rows
        ),
        "tau_taper_sample_count": sum(len(row.get("tau_taper") or ()) for row in taper_rows),
        "holding_fraction_of_override": (
            len(holding_steps) / realized_duration_steps if realized_duration_steps else 0.0
        ),
        "assembly_completed": bool(holding_steps),
        "controlled_cav_steps": controlled_cav_steps,
        "controlled_cav_seconds": controlled_cav_steps * float(step_length_s),
        "requested_lane_steps": requested_lane_steps,
        "controlled_lane_steps": controlled_lane_steps,
        "controlled_lane_seconds": controlled_lane_steps * float(step_length_s),
        "lane_coverage_fraction": lane_coverage_fraction,
        "member_change_count": len(member_changes),
        "member_change_counts": member_change_counts,
        "abort_count": len(abort_events),
        "safety_abort_count": len(safety_abort_events),
        "assembly_abort_count": len(assembly_abort_events),
        "member_target_sample_count": member_target_sample_count,
        "safety_projection_clip_count": sum(safety_projection_clip_reason_counts.values()),
        "safety_projection_clip_reason_counts": (safety_projection_clip_reason_counts),
        "mean_adaptive_speed_reduction_fraction": (
            fmean(adaptive_reduction_samples) if adaptive_reduction_samples else None
        ),
        "mean_adaptive_common_target_mps": (
            fmean(adaptive_common_target_samples) if adaptive_common_target_samples else None
        ),
        "configured_wall_deceleration_bounds_mps2": sorted(set(wall_deceleration_bounds)),
        "member_applied_command_sample_count": len(commanded_deceleration_samples),
        "mean_commanded_deceleration_mps2": (
            fmean(commanded_deceleration_samples) if commanded_deceleration_samples else None
        ),
        "max_commanded_deceleration_mps2": (
            max(commanded_deceleration_samples) if commanded_deceleration_samples else None
        ),
        "holding_speed_delivery": holding_speed_delivery,
        "terminal_tail_speed_delivery": {
            "holding_step_count": len(terminal_tail_steps),
            **terminal_tail_speed_delivery,
        },
        # Legacy field retained for artifact compatibility.  It covers only
        # temporal dose and lane coverage, not compact-wall geometry.
        "dose_at_least_80_percent": (
            realized_duration_fraction >= 0.8 and lane_coverage_fraction >= 0.8
        ),
        "dose_at_least_80_percent_scope": "duration_and_lane_coverage_only",
    }
    span_cap = first_treatment.get("max_longitudinal_span_m")
    spans = [float(row["longitudinal_span_m"]) for row in geometry_samples]
    speed_stds = [float(row["member_speed_std_mps"]) for row in geometry_samples]
    treatment_dose["wall_geometry"] = {
        "geometry_mode": first_treatment.get("geometry_mode"),
        "sample_count": len(geometry_samples),
        "mean_longitudinal_span_m": fmean(spans) if spans else None,
        "max_longitudinal_span_m": max(spans) if spans else None,
        "mean_member_speed_std_mps": fmean(speed_stds) if speed_stds else None,
        "max_member_speed_std_mps": max(speed_stds) if speed_stds else None,
        "configured_span_cap_m": span_cap,
        "fraction_within_configured_span": (
            sum(span <= float(span_cap) for span in spans) / len(spans)
            if spans and span_cap is not None
            else None
        ),
        "samples": geometry_samples,
    }
    duration_lane_qualified = bool(treatment_dose["dose_at_least_80_percent"])
    geometry_fraction = treatment_dose["wall_geometry"][
        "fraction_within_configured_span"
    ]
    compact_composite_qualified = duration_lane_qualified and (
        not compact_requested
        or (
            bool(treatment_dose["assembly_completed"])
            and float(treatment_dose["holding_fraction_of_override"]) >= 0.5
            and geometry_fraction is not None
            and float(geometry_fraction) >= 0.8
        )
    )
    treatment_dose.update(
        {
            "duration_and_lane_dose_qualified": duration_lane_qualified,
            "compact_composite_dose_qualified": compact_composite_qualified,
            "compact_composite_thresholds": {
                "minimum_duration_fraction": 0.8,
                "minimum_lane_coverage_fraction": 0.8,
                "minimum_holding_fraction_of_override": 0.5,
                "minimum_geometry_within_span_fraction": 0.8,
            },
            "planned_release_step": decision_step + requested_duration_steps + 1,
            "completion_reason": (
                "not_activated"
                if not activation_feasible
                else (
                    "runtime_abort"
                    if abort_events
                    else (
                        "planned_duration_complete"
                        if realized_duration_steps >= requested_duration_steps
                        else "incomplete_override"
                    )
                )
            ),
        }
    )
    release_step = decision_step + treatment_steps + 1
    release_control = control_snapshots.get(release_step)
    release_treatment = treatment_snapshots.get(release_step)
    release_diagnostics = []
    if release_control is not None and release_treatment is not None:
        control_vehicles = release_control.vehicle_map()
        treatment_vehicles = release_treatment.vehicle_map()
        for vehicle_id in override_ids:
            left = control_vehicles.get(vehicle_id)
            right = treatment_vehicles.get(vehicle_id)
            release_diagnostics.append(
                {
                    "relative_step": release_step,
                    "vehicle_id": vehicle_id,
                    "present_in_control": left is not None,
                    "present_in_treatment": right is not None,
                    "speed_delta_vs_damping_mps": physical_delta(left, right, 4),
                    "acceleration_delta_vs_damping_mps2": physical_delta(left, right, 5),
                    "tau_delta_vs_damping_s": physical_delta(left, right, 9),
                }
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "wall_override_steps": wall_override_steps,
        "wall_override_ids": override_ids,
        "wall_override_run_lengths": override_runs,
        "transition_reason_counts": transition_reason_counts,
        "first_post_override_transition": first_post_override_transition,
        "first_treatment_transition_reason": first_treatment_reason,
        "first_treatment_formation": first_treatment.get("formation"),
        "first_treatment_activation_safety": first_treatment.get("activation_safety"),
        "treatment_dose": treatment_dose,
        "member_changes": member_changes,
        "abort_events": abort_events,
        "emergency_taper_zero_handoff_steps": [
            int(row["relative_step"])
            for row in taper_rows
            if bool(row.get("emergency_taper_zero_handoff"))
        ],
        "control_deltas": control_deltas,
        "realized_control_deltas": realized_control_deltas,
        "release_physical_diagnostics": release_diagnostics,
    }


def _prefix_identity(
    control: ReplayTrace, treatment: ReplayTrace, decision_step: int
) -> dict[str, Any]:
    prefix_matches = all(
        _physical_state(control.snapshots[step]) == _physical_state(treatment.snapshots[step])
        for step in range(decision_step + 1)
    )
    first_divergence = _first_physical_divergence(control, treatment)
    return {
        "passed": prefix_matches and (first_divergence is None or first_divergence > decision_step),
        "prefix_matches_through_step": decision_step if prefix_matches else None,
        "first_physical_divergence_step": first_divergence,
    }


def _run_branch(
    *,
    args: argparse.Namespace,
    seed: int,
    schedule: Any,
    metadata: Mapping[str, Any],
) -> ReplayTrace:
    checkpoint = str(args.checkpoint.resolve())
    return run_replay(
        project_root=args.sumo_project_root,
        sumo_binary=args.sumo_binary,
        sumo_config=args.sumo_config,
        seed=seed,
        decision_step=args.decision_step,
        horizon_steps=args.horizon_steps,
        treatment_steps=args.treatment_steps,
        cav_penetration=args.cav_penetration,
        sumo_step_length=args.sumo_step_length,
        base_policy_factory=lambda: RLlibPolicy(checkpoint, explore=False),
        intervention_schedule=schedule,
        controller_metadata=metadata,
        overlay_overrides=dict(args.overlay_overrides),
    )


def _run_seed(
    args: argparse.Namespace,
    seed: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    def off_schedule(_step: int) -> int:
        return 0

    pulse_schedule = FrozenPulseSchedule(
        decision_step=args.decision_step,
        treatment_steps=args.treatment_steps,
    )
    print(f"seed={seed}: damping repeat A", flush=True)
    control_a = _run_branch(args=args, seed=seed, schedule=off_schedule, metadata=metadata)
    print(f"seed={seed}: damping repeat B", flush=True)
    control_b = _run_branch(args=args, seed=seed, schedule=off_schedule, metadata=metadata)
    print(f"seed={seed}: treatment repeat A", flush=True)
    treatment_a = _run_branch(args=args, seed=seed, schedule=pulse_schedule, metadata=metadata)
    print(f"seed={seed}: treatment repeat B", flush=True)
    treatment_b = _run_branch(args=args, seed=seed, schedule=pulse_schedule, metadata=metadata)

    damping_identity = compare_traces(control_a, control_b)
    treatment_repeat = compare_traces(treatment_a, treatment_b)
    prefix_identity = _prefix_identity(control_a, treatment_a, args.decision_step)
    audit_contract = _audit_contract(
        control_a,
        treatment_a,
        decision_step=args.decision_step,
        treatment_steps=args.treatment_steps,
        step_length_s=CONTROLLER_STEP_LENGTH_S,
    )
    checks = {
        "damping_vs_damping_replay_identity": bool(damping_identity["passed"]),
        "treatment_repeat_identity": bool(treatment_repeat["passed"]),
        "pre_treatment_prefix_identity": bool(prefix_identity["passed"]),
        "overlay_action_contract": bool(audit_contract["passed"]),
    }
    definition = OutcomeDefinition()
    return {
        "seed": seed,
        "passed": all(checks.values()),
        "checks": checks,
        "damping_identity": damping_identity,
        "treatment_repeat": treatment_repeat,
        "prefix_identity": prefix_identity,
        "overlay_action_contract": audit_contract,
        "damping_outcomes": summarize_branch(
            control_a,
            decision_step=args.decision_step,
            outcome_steps=args.horizon_steps,
            definition=definition,
        ),
        "treatment_outcomes": summarize_branch(
            treatment_a,
            decision_step=args.decision_step,
            outcome_steps=args.horizon_steps,
            definition=definition,
        ),
    }


def main() -> int:
    args = parse_args()
    if not (0.0 <= float(args.wall_d_lo) < float(args.wall_d_hi)):
        raise ValueError("wall band must satisfy 0 <= wall-d-lo < wall-d-hi")
    if args.geometry_mode == "compact_moving":
        if args.max_longitudinal_span_m is None:
            args.max_longitudinal_span_m = 50.0
        if args.compact_release_boundary_m is None:
            args.compact_release_boundary_m = float(args.wall_d_lo)
        planned_duration_s = float(args.treatment_steps) * CONTROLLER_STEP_LENGTH_S
        if not (0.0 <= float(args.compact_assembly_duration_s) < planned_duration_s):
            raise ValueError(
                "compact assembly duration must be nonnegative and shorter than the treatment"
            )
        if not (0.0 <= float(args.compact_max_speed_reduction_fraction) <= 1.0):
            raise ValueError("compact max speed reduction fraction must lie in [0, 1]")
        if float(args.compact_shock_gap_hi_mps) <= float(args.compact_shock_gap_lo_mps):
            raise ValueError("compact shock gap hi must exceed lo")
        if args.compact_control_mode == "adaptive_servo" and float(
            args.compact_servo_a_max_dec_mps2
        ) < float(args.wall_a_max_dec):
            raise ValueError(
                "compact servo deceleration bound cannot be gentler than "
                "the base wall deceleration bound"
            )
    elif args.compact_control_mode != "fixed_target":
        raise ValueError("adaptive compact control requires compact_moving geometry")
    wall_params = {
        "d_lo": float(args.wall_d_lo),
        "d_hi": float(args.wall_d_hi),
        "wall_speed": float(args.wall_speed),
        "wall_tau": float(args.wall_tau),
        "a_max_dec": float(args.wall_a_max_dec),
        "tau_ramp_step": float(args.tau_ramp_step),
    }
    args.overlay_overrides = {
        "wall_params": wall_params,
        "require_all_occupied_lanes": True,
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
        "compact_assembly_speed_tolerance_mps": float(args.compact_assembly_speed_tolerance_mps),
        "compact_assembly_a_max_dec_mps2": float(args.compact_assembly_a_max_dec_mps2),
        "compact_control_mode": str(args.compact_control_mode),
        "compact_servo_a_max_dec_mps2": float(args.compact_servo_a_max_dec_mps2),
        "compact_max_activation_speed_spread_mps": float(
            args.compact_max_activation_speed_spread_mps
        ),
        "compact_max_speed_reduction_fraction": float(args.compact_max_speed_reduction_fraction),
        "compact_shock_gap_lo_mps": float(args.compact_shock_gap_lo_mps),
        "compact_shock_gap_hi_mps": float(args.compact_shock_gap_hi_mps),
        "compact_position_kp_s_inv": float(args.compact_position_kp_s_inv),
        "compact_position_kd": float(args.compact_position_kd),
        "compact_position_speed_offset_cap_mps": float(args.compact_position_speed_offset_cap_mps),
        "compact_safety_min_front_ttc_s": float(args.compact_safety_min_front_ttc_s),
        "compact_safety_min_rear_ttc_s": float(args.compact_safety_min_rear_ttc_s),
        "compact_influence_upstream_m": float(args.compact_influence_upstream_m),
        "compact_influence_max_followers": int(args.compact_influence_max_followers),
        "compact_influence_min_ttc_s": float(args.compact_influence_min_ttc_s),
    }
    checkpoint = args.checkpoint.resolve()
    overlay_protocol = default_protocol_metadata()
    overlay_protocol["wall_params"] = dict(wall_params)
    overlay_protocol["geometry_mode"] = str(args.geometry_mode)
    overlay_protocol["exact_overlay_overrides"] = dict(args.overlay_overrides)
    metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_tree(checkpoint),
        "checkpoint_policy_sha256": _sha256_file(
            checkpoint / "policies" / "cav_policy" / "policy_state.pkl"
        ),
        "hdv_model": "route-file Krauss mixture",
        "sumo_config": args.sumo_config,
        "cav_penetration": args.cav_penetration,
        "overlay_protocol": overlay_protocol,
    }
    seed_results = []
    for seed in args.seeds:
        result = _run_seed(args, int(seed), metadata)
        seed_results.append(result)
        print(f"seed={seed}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)

    passed = bool(seed_results) and all(result["passed"] for result in seed_results)
    report = {
        "schema_version": "counterfactual_escalation_v2_twin_check/2",
        "purpose": "mandatory causal-contract gate before headroom/predictor testing",
        "passed": passed,
        "verdict": (
            "causal_contract_passed_predictor_not_yet_tested"
            if passed
            else "causal_contract_failed_do_not_test_predictor"
        ),
        "configuration": {
            "seeds": args.seeds,
            "decision_step": args.decision_step,
            "horizon_steps": args.horizon_steps,
            "treatment_steps": args.treatment_steps,
            "cav_penetration": args.cav_penetration,
            "sumo_step_length": args.sumo_step_length,
            "controller_step_length_s": CONTROLLER_STEP_LENGTH_S,
            "sumo_project_root": str(args.sumo_project_root.resolve()),
            "sumo_binary": args.sumo_binary,
            "sumo_config": args.sumo_config,
            "overlay_overrides": dict(args.overlay_overrides),
            **metadata,
        },
        "seeds": seed_results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": passed, "verdict": report["verdict"]}, indent=2))
    print(f"wrote {output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
