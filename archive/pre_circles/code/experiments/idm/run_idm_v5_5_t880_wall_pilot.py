"""Run the frozen absolute-time v5.5 Gate-F wall headroom pilot."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from counterfactual.idm_gate_labels import IDMGateBMargins, classify_idm_gate_b_case
from counterfactual.outcomes import OutcomeDefinition, summarize_branch
from counterfactual.twin_replay import FrozenPulseSchedule, ReplayTrace, run_replay
from experiments import REPO_ROOT
from experiments.counterfactual.verify_counterfactual_escalation import (
    _audit_contract,
    _prefix_identity,
)
from experiments.idm.idm_v5_5_t880_contract import (
    ACTIVATION_TIME_S,
    ANALYSIS_END_S,
    CAV_PENETRATION,
    DEMAND_MAINLINE_VPH,
    DEMAND_RAMP_VPH,
    DENSITY_MAPPER_SUMMARY_SHA256,
    DEVELOPMENT_SEEDS,
    MINIMUM_REALIZED_DURATION_FRACTION,
    MINIMUM_REALIZED_LANE_COVERAGE_FRACTION,
    MINIMUM_SHORTENING_S,
    MINIMUM_FEASIBLE_SEEDS,
    PRIMARY_QUEUE_COUNT,
    PRIMARY_SPEED_THRESHOLD_MPS,
    PRIMARY_SUSTAIN_S,
    RELEASE_TIME_S,
    REPEAT_SEED,
    SCIENTIFIC_ROLE,
    SECONDARY_HEALTH_SUSTAIN_S,
    SECONDARY_HEALTH_THRESHOLD,
    TREATMENT_DURATION_S,
    contract_manifest,
)
from experiments.idm.run_idm_wall_pilot import DEFAULT_SUMO_EXE, supervisor_overrides
from experiments.idm.run_idm_wall_v5_3_stage1 import _mechanics
from experiments.idm.scan_idm_v5_5_t880_gate_f import _validate_seed_payload
from experiments.idm.wall_v5_4_contract import overlay_overrides, profile_manifest
from experiments.io import atomic_write_json
from idm.contract import SCENARIO_ROOT, transition_scenario_config
from idm.density_mapper import summarize_density_trace
from idm.wall_t880 import congestion_run_summary, upstream_storage_summary
from policies.idm_analytical import (
    build_frozen_idm_analytical_policy,
    frozen_idm_analytical_manifest,
)


DEFAULT_GATE_F = REPO_ROOT / "runs" / "probe" / "idm_v5_5_t880_gate_f_p30_s4401_4406"
DEFAULT_DENSITY = (
    REPO_ROOT / "runs" / "probe" / "idm_v5_5_damping_density_mapper_p30_s4401_4406"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "probe" / "idm_v5_5_t880_wall_pilot_p30_s4401_4406"
)
DEFAULT_IDENTITY_AUDIT = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "idm_v5_5_t880_control_identity_audit_p30_s4401_4406"
)
MANIFEST_SCHEMA = "idm_v5_5_t880_wall_pilot_manifest/1"
CASE_SCHEMA = "idm_v5_5_t880_wall_pilot_case/1"
SUMMARY_SCHEMA = "idm_v5_5_t880_wall_pilot_summary/1"
OUTCOME_STEPS = ANALYSIS_END_S - ACTIVATION_TIME_S
CAUSAL_CHECK_NAMES = (
    "prefix_base_action_identity",
    "first_request_same_base_actions",
    "override_only_during_treatment",
    "out_of_band_damping_preserved",
    "wall_ids_removed_from_env_action_path",
    "release_returns_to_damping",
)


def _sha256_json(payload: Any) -> str:
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


def _always_damping(_step: int) -> int:
    return 0


def _gate_f_row(root: Path, seed: int) -> dict[str, Any]:
    row = json.loads((root / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8"))
    _validate_seed_payload(row, seed)
    return row


def _gate_f_bundle_provenance(root: Path) -> dict[str, Any]:
    files = {
        "manifest.json": _sha256_file(root / "manifest.json"),
        "summary.json": _sha256_file(root / "summary.json"),
        **{
            f"seeds/seed{seed}.json": _sha256_file(root / "seeds" / f"seed{seed}.json")
            for seed in DEVELOPMENT_SEEDS
        },
    }
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def _density_bundle_provenance(root: Path) -> dict[str, Any]:
    files = {
        "summary.json": _sha256_file(root / "summary.json"),
        **{
            f"cases/m{DEMAND_MAINLINE_VPH}.seed{seed}.primary.json": _sha256_file(
                root / "cases" / f"m{DEMAND_MAINLINE_VPH}.seed{seed}.primary.json"
            )
            for seed in DEVELOPMENT_SEEDS
        },
    }
    return {"files": files, "bundle_sha256": _sha256_json(files)}


def _relative_decision(gate_f: Mapping[str, Any]) -> int:
    reset_time = float((gate_f.get("reset_info") or {}).get("episode_time"))
    decision = int(round(ACTIVATION_TIME_S - reset_time))
    if decision < 0 or abs(reset_time + decision - ACTIVATION_TIME_S) > 1e-6:
        raise RuntimeError(
            f"cannot map reset time {reset_time} to absolute {ACTIVATION_TIME_S}"
        )
    return decision


def _replay(
    *,
    seed: int,
    decision_step: int,
    treatment: bool,
    sumo_binary: str,
    wall_overlay_overrides: Callable[[], dict[str, Any]] = overlay_overrides,
    wall_profile_manifest: Callable[[], dict[str, Any]] = profile_manifest,
    treatment_arm_name: str = "damping_plus_v5_4_wall",
) -> ReplayTrace:
    schedule = (
        FrozenPulseSchedule(decision_step, TREATMENT_DURATION_S)
        if treatment
        else _always_damping
    )
    return run_replay(
        project_root=SCENARIO_ROOT,
        sumo_binary=sumo_binary,
        sumo_config=transition_scenario_config(DEMAND_MAINLINE_VPH).name,
        seed=seed,
        decision_step=decision_step,
        horizon_steps=OUTCOME_STEPS,
        treatment_steps=TREATMENT_DURATION_S,
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
        intervention_schedule=schedule,
        overlay_overrides=wall_overlay_overrides(),
        controller_metadata={
            "scientific_role": SCIENTIFIC_ROLE,
            "arm": treatment_arm_name if treatment else "damping_control",
            "activation_absolute_s": ACTIVATION_TIME_S,
            "release_absolute_s": RELEASE_TIME_S,
            "common_endpoint_absolute_s": ANALYSIS_END_S,
            "wall_profile": wall_profile_manifest(),
        },
    )


def _terminal_integrity(trace: ReplayTrace) -> dict[str, Any]:
    final = trace.snapshots[-1]
    final_time = float(final.sim_time)
    exact_endpoint = abs(final_time - ANALYSIS_END_S) <= 1e-6
    expected = trace.controller_metadata.get("terminal_expected_vehicle_count")
    natural_clearance = bool(
        final_time < ANALYSIS_END_S
        and str(trace.controller_metadata.get("termination_reason")) == "terminated"
        and int(final.vehicle_count) == 0
        and expected is not None
        and int(expected) == 0
    )
    return {
        "final_sim_time_s": final_time,
        "final_vehicle_count": int(final.vehicle_count),
        "terminal_expected_vehicle_count": expected,
        "runner_termination_reason": trace.controller_metadata.get("termination_reason"),
        "exact_1080_endpoint": exact_endpoint,
        "proven_natural_clearance": natural_clearance,
        "valid_common_horizon": exact_endpoint or natural_clearance,
        "empty_remainder_zero_padded_conceptually": natural_clearance,
    }


def _absolute_timing_integrity(
    trace: ReplayTrace, *, decision_step: int, treatment: bool
) -> dict[str, Any]:
    audits = {int(row["relative_step"]): row for row in trace.controller_audits}
    snapshots = {int(row.relative_step): row for row in trace.snapshots}
    decision_snapshot = snapshots.get(decision_step)
    pulse_steps = list(range(decision_step + 1, decision_step + TREATMENT_DURATION_S + 1))
    pulse_rows = [audits.get(step, {}) for step in pulse_steps]
    action_times = [
        float(snapshots[step - 1].sim_time)
        for step in pulse_steps
        if step - 1 in snapshots
    ]
    result_times = [
        float(snapshots[step].sim_time) for step in pulse_steps if step in snapshots
    ]
    release_step = decision_step + TREATMENT_DURATION_S + 1
    release_row = audits.get(release_step, {})
    release_action_time = (
        float(snapshots[release_step - 1].sim_time)
        if release_step - 1 in snapshots
        else None
    )
    expected_action_times = [float(value) for value in range(880, 900)]
    expected_result_times = [float(value) for value in range(881, 901)]
    command = 1 if treatment else 0
    checks = {
        "decision_state_is_absolute_880": bool(
            decision_snapshot
            and abs(float(decision_snapshot.sim_time) - ACTIVATION_TIME_S) <= 1e-6
        ),
        "twenty_exact_command_rows": len(pulse_rows) == TREATMENT_DURATION_S
        and all(int(row.get("command", -1)) == command for row in pulse_rows),
        "action_times_are_880_through_899": action_times == expected_action_times,
        "result_times_are_881_through_900": result_times == expected_result_times,
        "release_command_begins_at_900": bool(
            release_row
            and release_action_time == float(RELEASE_TIME_S)
            and int(release_row.get("command", -1)) == 0
            and not release_row.get("wall_override_ids")
            and not bool(release_row.get("wall_active"))
        ),
        "valid_common_horizon": _terminal_integrity(trace)["valid_common_horizon"],
    }
    return {
        "decision_relative_step": decision_step,
        "pulse_relative_steps": pulse_steps,
        "pulse_action_times_s": action_times,
        "pulse_result_times_s": result_times,
        "release_relative_step": release_step,
        "release_action_time_s": release_action_time,
        "checks": checks,
        "passed": all(checks.values()),
        "terminal": _terminal_integrity(trace),
    }


def _flow_window(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in telemetry
        if ACTIVATION_TIME_S < float(row["sim_time_s"]) <= ANALYSIS_END_S
    ]
    keys = (
        "mainline_arrivals",
        "ramp_arrivals",
        "mainline_merge_entries",
        "ramp_merge_entries",
        "downstream_discharges",
    )
    return {key: sum(int(row[key]) for row in rows) for key in keys}


def _telemetry_integrity(density: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = density.get("telemetry") or []
    terminal = density.get("terminal") or {}
    terminal_time = float(terminal.get("sim_time_s", -1.0))
    coverage_end = min(float(ANALYSIS_END_S), terminal_time)
    observed = {
        round(float(row["sim_time_s"]), 6)
        for row in telemetry
        if ACTIVATION_TIME_S <= float(row["sim_time_s"]) <= coverage_end
    }
    required = {
        round(float(value), 6)
        for value in range(ACTIVATION_TIME_S, int(coverage_end) + 1)
    }
    clock = density.get("clock_integrity") or {}
    checks = {
        "one_hz": bool(clock.get("one_hz")),
        "absolute_900_anchor": bool(clock.get("exact_anchor_observed")),
        "complete_880_to_terminal_coverage": bool(required)
        and required.issubset(observed),
        "valid_terminal": str(terminal.get("reason"))
        in {"horizon_reached", "natural_clearance"},
    }
    return {"checks": checks, "passed": all(checks.values())}


def _arm_record(trace: ReplayTrace, *, decision_step: int) -> dict[str, Any]:
    definition = OutcomeDefinition(
        queue_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        congestion_queue_count=PRIMARY_QUEUE_COUNT,
        congestion_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        sustain_steps=PRIMARY_SUSTAIN_S,
    )
    outcome = summarize_branch(
        trace,
        decision_step=decision_step,
        outcome_steps=OUTCOME_STEPS,
        definition=definition,
    )
    density = summarize_density_trace(
        trace,
        analysis_start_s=RELEASE_TIME_S,
        analysis_end_s=ANALYSIS_END_S,
        prevalence_lookback_s=PRIMARY_SUSTAIN_S,
        primary_queue_count=PRIMARY_QUEUE_COUNT,
        primary_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        primary_sustain_s=PRIMARY_SUSTAIN_S,
        secondary_health_threshold=SECONDARY_HEALTH_THRESHOLD,
        secondary_health_sustain_s=SECONDARY_HEALTH_SUSTAIN_S,
    )
    telemetry = density["telemetry"]
    return {
        "trace_summary": trace.summary(),
        "trace_digest": trace.trace_digest,
        "controller_audit_digest": _sha256_json(trace.controller_audits),
        "outcomes_880_to_1080": outcome,
        "density_telemetry": density,
        "telemetry_integrity": _telemetry_integrity(density),
        "physical_congestion_880_to_1080": congestion_run_summary(
            telemetry,
            start_s=ACTIVATION_TIME_S,
            end_s=ANALYSIS_END_S,
            sustain_s=PRIMARY_SUSTAIN_S,
        ),
        "primary_physical_congestion_900_to_1080": congestion_run_summary(
            telemetry,
            start_s=RELEASE_TIME_S,
            end_s=ANALYSIS_END_S,
            sustain_s=PRIMARY_SUSTAIN_S,
        ),
        "upstream_storage_880_to_1080": upstream_storage_summary(
            telemetry, start_s=ACTIVATION_TIME_S, end_s=ANALYSIS_END_S
        ),
        "flow_880_to_1080": _flow_window(telemetry),
        "timing_integrity": _absolute_timing_integrity(
            trace, decision_step=decision_step, treatment=trace.post_action != 0
        ),
    }


def _lifecycle_rows(trace: ReplayTrace, decision_step: int) -> list[dict[str, Any]]:
    snapshots = {int(row.relative_step): row for row in trace.snapshots}
    result = []
    for audit in trace.controller_audits:
        step = int(audit.get("relative_step", -1))
        if not (decision_step + 1 <= step <= decision_step + TREATMENT_DURATION_S + 1):
            continue
        result.append(
            {
                "relative_step": step,
                "action_time_s": (
                    None if step - 1 not in snapshots else float(snapshots[step - 1].sim_time)
                ),
                "result_state_time_s": (
                    None if step not in snapshots else float(snapshots[step].sim_time)
                ),
                "command": int(audit.get("command", 0)),
                "treatment_state": audit.get("treatment_state"),
                "transition_reason": audit.get("transition_reason"),
                "wall_active": bool(audit.get("wall_active")),
                "wall_override_ids": list(audit.get("wall_override_ids", ())),
                "out_of_band_damping_ids": list(audit.get("out_of_band_damping_ids", ())),
                "lane_coverage": audit.get("lane_coverage"),
                "current_member_span_m": audit.get("current_member_span_m"),
                "member_changes": list(audit.get("member_changes", ())),
                "abort_reason": audit.get("abort_reason"),
                "taper_blend_fraction": audit.get("taper_blend_fraction"),
                "taper_reason": audit.get("taper_reason"),
                "emergency_taper_zero_handoff": bool(
                    audit.get("emergency_taper_zero_handoff")
                ),
            }
        )
    return result


def _physical_effect(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    control_primary = control["primary_physical_congestion_900_to_1080"]
    treatment_primary = treatment["primary_physical_congestion_900_to_1080"]
    duration_reduction = int(control_primary["sustained_duration_s"]) - int(
        treatment_primary["sustained_duration_s"]
    )
    longest_reduction = int(control_primary["longest_run_s"]) - int(
        treatment_primary["longest_run_s"]
    )
    predicate_reduction = int(control_primary["predicate_duration_s"]) - int(
        treatment_primary["predicate_duration_s"]
    )
    materially_reduced = bool(
        predicate_reduction >= MINIMUM_SHORTENING_S
        and longest_reduction >= MINIMUM_SHORTENING_S
    )
    prevention = bool(
        control_primary["sustained_event"]
        and not treatment_primary["sustained_event"]
        and materially_reduced
    )
    shortening = bool(
        control_primary["sustained_event"]
        and treatment_primary["sustained_event"]
        and duration_reduction >= MINIMUM_SHORTENING_S
        and materially_reduced
    )
    return {
        "prevents_sustained_premerge_congestion": prevention,
        "materially_shortens_sustained_congestion": shortening,
        "primary_physical_benefit": prevention or shortening,
        "sustained_duration_reduction_s": duration_reduction,
        "longest_run_reduction_s": longest_reduction,
        "predicate_duration_reduction_s": predicate_reduction,
        "materiality_pass": materially_reduced,
        "minimum_material_shortening_s": MINIMUM_SHORTENING_S,
        "induces_sustained_congestion": bool(
            not control_primary["sustained_event"] and treatment_primary["sustained_event"]
        ),
    }


def _dose_complete(audit: Mapping[str, Any]) -> bool:
    dose = audit.get("treatment_dose") or {}
    checks = audit.get("checks") or {}
    return bool(
        float(dose.get("realized_duration_fraction", 0.0))
        >= MINIMUM_REALIZED_DURATION_FRACTION
        and float(dose.get("lane_coverage_fraction", 0.0))
        >= MINIMUM_REALIZED_LANE_COVERAGE_FRACTION
        and bool(dose.get("assembly_completed"))
        and int(dose.get("abort_count", 0)) == 0
        and dose.get("completion_reason") == "planned_duration_complete"
        and bool(checks.get("taper_is_monotone_and_reaches_zero"))
        and bool(checks.get("release_returns_to_damping"))
    )


def _causal_integrity(audit: Mapping[str, Any]) -> bool:
    checks = audit.get("checks") or {}
    return all(bool(checks.get(name)) for name in CAUSAL_CHECK_NAMES)


def _case_classification(
    *,
    gate_f_admitted: bool,
    integrity_valid: bool,
    dose_complete: bool,
    safety_service: Mapping[str, Any],
    physical_effect: Mapping[str, Any],
) -> dict[str, Any]:
    if not integrity_valid:
        label = "invalid_experiment_integrity"
    elif not gate_f_admitted:
        label = "infeasible_formation"
    elif not dose_complete:
        label = "failed_realized_dose"
    elif not bool(safety_service.get("safety_admissible")):
        label = "safety_failure"
    elif not bool(safety_service.get("traffic_noninferior")):
        label = "service_failure"
    elif bool(physical_effect.get("primary_physical_benefit")) and bool(
        safety_service.get("traffic_noninferior")
    ):
        label = "safe_beneficial"
    else:
        label = "safe_nonbeneficial"
    return {
        "first_stage_label": label,
        "integrity_valid": integrity_valid,
        "gate_f_admitted": gate_f_admitted,
        "dose_complete": dose_complete,
        "safety_admissible": bool(safety_service.get("safety_admissible")),
        "service_noninferior": bool(safety_service.get("traffic_noninferior")),
        "primary_physical_benefit": bool(
            physical_effect.get("primary_physical_benefit")
        ),
        "safe_beneficial": label == "safe_beneficial",
        "safe_nonbeneficial": label == "safe_nonbeneficial",
    }


def _validate_case_record(
    payload: Mapping[str, Any],
    *,
    seed: int,
    repeat: bool,
    gate_f_bundle_sha256: str | None = None,
    density_bundle_sha256: str | None = None,
) -> None:
    arms = payload.get("arms") or {}
    control = arms.get("0") or {}
    treatment = arms.get("1") or {}
    recomputed_effect = _physical_effect(control, treatment) if control and treatment else None
    recomputed_gate_b = None
    if control and treatment:
        recomputed_gate_b = classify_idm_gate_b_case(
            {
                "arms": {
                    "0": {"outcomes": control["outcomes_880_to_1080"], "wall_engaged": False},
                    "1": {
                        "outcomes": treatment["outcomes_880_to_1080"],
                        "wall_engaged": bool(
                            (payload.get("treatment_contract") or {}).get(
                                "wall_override_steps"
                            )
                        ),
                    },
                }
            },
            IDMGateBMargins(),
        )
    stored_classification = payload.get("classification") or {}
    expected_decision = _relative_decision(payload.get("gate_f") or {}) if payload.get("gate_f") else None
    recomputed_integrity = bool(
        (payload.get("prefix_identity") or {}).get("passed")
        and control
        and treatment
        and (control.get("timing_integrity") or {}).get("passed")
        and (treatment.get("timing_integrity") or {}).get("passed")
        and (control.get("telemetry_integrity") or {}).get("passed")
        and (treatment.get("telemetry_integrity") or {}).get("passed")
        and _causal_integrity(payload.get("treatment_contract") or {})
    )
    recomputed_classification = None
    if recomputed_effect is not None and recomputed_gate_b is not None:
        recomputed_classification = _case_classification(
            gate_f_admitted=bool(stored_classification.get("gate_f_admitted")),
            integrity_valid=recomputed_integrity,
            dose_complete=_dose_complete(payload.get("treatment_contract") or {}),
            safety_service=recomputed_gate_b,
            physical_effect=recomputed_effect,
        )
    checks = {
        "schema": payload.get("schema") == CASE_SCHEMA,
        "seed": int(payload.get("seed", -1)) == seed,
        "repeat": bool(payload.get("repeat")) is repeat,
        "contract": payload.get("contract_fingerprint_sha256")
        == contract_manifest()["contract_fingerprint_sha256"],
        "base": payload.get("base_control_arm") == frozen_idm_analytical_manifest(),
        "wall": payload.get("wall_profile") == profile_manifest(),
        "scientific_role": payload.get("scientific_role") == SCIENTIFIC_ROLE,
        "scenario": int(payload.get("demand_mainline_vph", -1)) == DEMAND_MAINLINE_VPH
        and int(payload.get("demand_ramp_vph", -1)) == DEMAND_RAMP_VPH
        and abs(float(payload.get("cav_penetration", -1.0)) - CAV_PENETRATION) <= 1e-12,
        "decision": expected_decision is not None
        and int(payload.get("decision_relative_step", -1)) == expected_decision,
        "arms": set(arms) == {"0", "1"},
        "gate_f_bundle": gate_f_bundle_sha256 is None
        or payload.get("gate_f_bundle_sha256") == gate_f_bundle_sha256,
        "density_bundle": density_bundle_sha256 is None
        or payload.get("density_bundle_sha256") == density_bundle_sha256,
        "physical_effect": recomputed_effect == payload.get("primary_physical_effect"),
        "safety_service": recomputed_gate_b == payload.get("safety_service_guardrail"),
        "classification": recomputed_classification == stored_classification,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"wall pilot case {seed}/{repeat} mismatch: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    seed = int(case["seed"])
    repeat = bool(case["repeat"])
    output = Path(case["output"])
    suffix = "repeat" if repeat else "primary"
    path = output / "cases" / f"seed{seed}.{suffix}.json"
    gate_f = _gate_f_row(Path(case["gate_f"]), seed)
    decision = _relative_decision(gate_f)
    gate_f_bundle_sha256 = str(case["gate_f_bundle_sha256"])
    density_bundle_sha256 = str(case["density_bundle_sha256"])
    if path.is_file() and not bool(case.get("force")):
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_case_record(
            existing,
            seed=seed,
            repeat=repeat,
            gate_f_bundle_sha256=gate_f_bundle_sha256,
            density_bundle_sha256=density_bundle_sha256,
        )
        return {"status": "skipped", "seed": seed, "repeat": repeat, "path": str(path)}
    started = time.time()
    control_trace = _replay(
        seed=seed,
        decision_step=decision,
        treatment=False,
        sumo_binary=str(case["sumo_binary"]),
    )
    treatment_trace = _replay(
        seed=seed,
        decision_step=decision,
        treatment=True,
        sumo_binary=str(case["sumo_binary"]),
    )
    control = _arm_record(control_trace, decision_step=decision)
    treatment = _arm_record(treatment_trace, decision_step=decision)
    audit = _audit_contract(
        control_trace,
        treatment_trace,
        decision_step=decision,
        treatment_steps=TREATMENT_DURATION_S,
        step_length_s=1.0,
    )
    mechanics_audit = dict(audit)
    mechanics_audit["controller_audits"] = [
        dict(row) for row in treatment_trace.controller_audits
    ]
    prefix = _prefix_identity(control_trace, treatment_trace, decision)
    row_for_gate_b = {
        "arms": {
            "0": {"outcomes": control["outcomes_880_to_1080"], "wall_engaged": False},
            "1": {
                "outcomes": treatment["outcomes_880_to_1080"],
                "wall_engaged": bool(audit["wall_override_steps"]),
            },
        }
    }
    safety_service = classify_idm_gate_b_case(row_for_gate_b, IDMGateBMargins())
    effect = _physical_effect(control, treatment)
    admitted = bool(gate_f["gate_f"]["admitted"])
    integrity_valid = bool(
        prefix["passed"]
        and control["timing_integrity"]["passed"]
        and treatment["timing_integrity"]["passed"]
        and control["telemetry_integrity"]["passed"]
        and treatment["telemetry_integrity"]["passed"]
        and _causal_integrity(audit)
    )
    dose_complete = _dose_complete(audit)
    classification = _case_classification(
        gate_f_admitted=admitted,
        integrity_valid=integrity_valid,
        dose_complete=dose_complete,
        safety_service=safety_service,
        physical_effect=effect,
    )
    record = {
        "schema": CASE_SCHEMA,
        "scientific_role": SCIENTIFIC_ROLE,
        "contract_fingerprint_sha256": contract_manifest()[
            "contract_fingerprint_sha256"
        ],
        "gate_f_bundle_sha256": gate_f_bundle_sha256,
        "density_bundle_sha256": density_bundle_sha256,
        "seed": seed,
        "repeat": repeat,
        "demand_mainline_vph": DEMAND_MAINLINE_VPH,
        "demand_ramp_vph": DEMAND_RAMP_VPH,
        "cav_penetration": CAV_PENETRATION,
        "decision_relative_step": decision,
        "gate_f": gate_f,
        "base_control_arm": frozen_idm_analytical_manifest(),
        "wall_profile": profile_manifest(),
        "prefix_identity": prefix,
        "treatment_contract": audit,
        "treatment_mechanics": _mechanics(mechanics_audit),
        "treatment_lifecycle_telemetry": _lifecycle_rows(treatment_trace, decision),
        "arms": {"0": control, "1": treatment},
        "safety_service_guardrail": safety_service,
        "primary_physical_effect": effect,
        "classification": classification,
        "wall_clock_seconds": time.time() - started,
    }
    atomic_write_json(path, record)
    return {
        "status": "completed",
        "seed": seed,
        "repeat": repeat,
        "path": str(path),
        "classification": classification["first_stage_label"],
        "dose_fraction": audit["treatment_dose"]["realized_duration_fraction"],
        "seconds": record["wall_clock_seconds"],
        "safety_admissible": classification["safety_admissible"],
        "service_noninferior": classification["service_noninferior"],
        "physical_benefit": classification["primary_physical_benefit"],
        "safety_service_effects": safety_service["effects"],
    }


def _density_control(root: Path, seed: int) -> dict[str, Any]:
    path = root / "cases" / f"m{DEMAND_MAINLINE_VPH}.seed{seed}.primary.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema": row.get("schema") == "idm_v5_5_density_mapper_branch/1",
        "seed": int(row.get("seed", -1)) == seed,
        "primary": row.get("repeat") is False,
        "demand": int(row.get("demand_mainline_vph", -1)) == DEMAND_MAINLINE_VPH
        and int(row.get("demand_ramp_vph", -1)) == DEMAND_RAMP_VPH,
        "p30": abs(float(row.get("cav_penetration", -1.0)) - CAV_PENETRATION) <= 1e-12,
        "base": row.get("base_control_arm") == frozen_idm_analytical_manifest(),
        "integrity": all(bool(value) for value in (row.get("integrity") or {}).values()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"density control {seed} provenance mismatch: "
            f"{[name for name, passed in checks.items() if not passed]}"
        )
    return row


def _external_control_identity(record: Mapping[str, Any], density: Mapping[str, Any]) -> dict:
    control = record["arms"]["0"]
    checks = {
        "physical_trace_digest_exact": control["trace_digest"] == density["trace_digest"],
        "anonymous_telemetry_digest_exact": (
            control["density_telemetry"]["telemetry_digest"]
            == density["telemetry_digest"]
        ),
        "terminal_exact": control["density_telemetry"]["terminal"] == density["terminal"],
        "flow_population_exact": (
            control["density_telemetry"]["flow_and_population"]
            == density["flow_and_population"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _identity_audit_bundle(
    root: Path, *, pilot_root: Path, density_root: Path
) -> dict[str, Any]:
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != "idm_v5_5_t880_control_identity_summary/1"
        or int(summary.get("seed_count", -1)) != len(DEVELOPMENT_SEEDS)
        or not bool(summary.get("all_microscopic_control_identity_passed"))
        or not bool(summary.get("all_differences_confined_to_supervisor_progress"))
    ):
        raise RuntimeError("control-identity reconciliation summary did not pass")
    files = {"summary.json": _sha256_file(summary_path)}
    rows: dict[int, dict[str, Any]] = {}
    for seed in DEVELOPMENT_SEEDS:
        path = root / "seeds" / f"seed{seed}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        provenance = row.get("source_provenance") or {}
        comparison = row.get("comparison") or {}
        checks = comparison.get("checks") or {}
        provenance_checks = row.get("provenance_checks") or {}
        expected_density = (
            density_root / "cases" / f"m{DEMAND_MAINLINE_VPH}.seed{seed}.primary.json"
        )
        expected_pilot = pilot_root / "cases" / f"seed{seed}.primary.json"
        validations = {
            "schema": row.get("schema") == "idm_v5_5_t880_control_identity_case/1",
            "seed": int(row.get("seed", -1)) == seed,
            "passed": bool(row.get("passed")),
            "no_wall": row.get("wall_requested") is False,
            "no_new_seed": row.get("new_seed_opened") is False,
            "density_source_hash": provenance.get("density_case_sha256")
            == _sha256_file(expected_density),
            "pilot_source_hash": provenance.get("pilot_case_sha256")
            == _sha256_file(expected_pilot),
            "contract": provenance.get("t880_contract_fingerprint_sha256")
            == contract_manifest()["contract_fingerprint_sha256"],
            "all_microscopic_checks": bool(checks) and all(checks.values()),
            "all_provenance_checks": bool(provenance_checks)
            and all(provenance_checks.values()),
            "raw_hash_remains_diagnostic_mismatch": comparison.get(
                "raw_trace_digest_exact"
            )
            is False,
            "only_progress_index_differs": comparison.get(
                "supervisor_difference_indices"
            )
            == [10],
        }
        if not all(validations.values()):
            raise RuntimeError(
                f"control-identity audit seed {seed} mismatch: "
                f"{[name for name, passed in validations.items() if not passed]}"
            )
        row["summary_validation"] = validations
        rows[seed] = row
        files[f"seeds/seed{seed}.json"] = _sha256_file(path)
    return {
        "root": str(root),
        "files": files,
        "bundle_sha256": _sha256_json(files),
        "summary": summary,
        "rows": rows,
    }


def _repeat_check(primary: Mapping[str, Any], repeat: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for arm in ("0", "1"):
        left = primary["arms"][arm]
        right = repeat["arms"][arm]
        prefix = "control" if arm == "0" else "treatment"
        checks[f"{prefix}_trace_digest_exact"] = left["trace_digest"] == right["trace_digest"]
        checks[f"{prefix}_controller_audit_digest_exact"] = (
            left["controller_audit_digest"] == right["controller_audit_digest"]
        )
        checks[f"{prefix}_telemetry_digest_exact"] = (
            left["density_telemetry"]["telemetry_digest"]
            == right["density_telemetry"]["telemetry_digest"]
        )
        checks[f"{prefix}_outcomes_exact"] = (
            left["outcomes_880_to_1080"] == right["outcomes_880_to_1080"]
        )
        checks[f"{prefix}_physical_endpoint_exact"] = (
            left["primary_physical_congestion_900_to_1080"]
            == right["primary_physical_congestion_900_to_1080"]
        )
    checks["treatment_contract_exact"] = (
        primary["treatment_contract"] == repeat["treatment_contract"]
    )
    checks["classification_exact"] = primary["classification"] == repeat["classification"]
    return {"checks": checks, "passed": all(checks.values()), "repeat_noise_s": 0 if all(checks.values()) else None}


def build_summary(
    output: Path,
    gate_f_root: Path,
    density_root: Path,
    identity_root: Path = DEFAULT_IDENTITY_AUDIT,
) -> dict[str, Any]:
    gate_bundle = _gate_f_bundle_provenance(gate_f_root)
    density_bundle = _density_bundle_provenance(density_root)
    expected_case_names = {
        *(f"seed{seed}.primary.json" for seed in DEVELOPMENT_SEEDS),
        f"seed{REPEAT_SEED}.repeat.json",
    }
    observed_case_names = {
        path.name for path in (output / "cases").glob("seed*.json") if path.is_file()
    }
    primary = {
        seed: json.loads(
            (output / "cases" / f"seed{seed}.primary.json").read_text(encoding="utf-8")
        )
        for seed in DEVELOPMENT_SEEDS
    }
    repeat = json.loads(
        (output / "cases" / f"seed{REPEAT_SEED}.repeat.json").read_text(encoding="utf-8")
    )
    for seed, row in primary.items():
        _validate_case_record(
            row,
            seed=seed,
            repeat=False,
            gate_f_bundle_sha256=gate_bundle["bundle_sha256"],
            density_bundle_sha256=density_bundle["bundle_sha256"],
        )
    _validate_case_record(
        repeat,
        seed=REPEAT_SEED,
        repeat=True,
        gate_f_bundle_sha256=gate_bundle["bundle_sha256"],
        density_bundle_sha256=density_bundle["bundle_sha256"],
    )
    gate_summary = json.loads((gate_f_root / "summary.json").read_text(encoding="utf-8"))
    external = {
        seed: _external_control_identity(row, _density_control(density_root, seed))
        for seed, row in primary.items()
    }
    identity_audit = _identity_audit_bundle(
        identity_root, pilot_root=output, density_root=density_root
    )
    normalized_external = {
        seed: {
            "passed": bool(identity_audit["rows"][seed]["passed"]),
            "checks": identity_audit["rows"][seed]["comparison"]["checks"],
            "provenance_checks": identity_audit["rows"][seed][
                "provenance_checks"
            ],
            "raw_trace_digest_exact": identity_audit["rows"][seed][
                "comparison"
            ]["raw_trace_digest_exact"],
            "supervisor_difference_indices": identity_audit["rows"][seed][
                "comparison"
            ]["supervisor_difference_indices"],
        }
        for seed in DEVELOPMENT_SEEDS
    }
    repeat_result = _repeat_check(primary[REPEAT_SEED], repeat)
    admitted = [row for row in primary.values() if row["classification"]["gate_f_admitted"]]
    labels = [row["classification"]["first_stage_label"] for row in primary.values()]
    beneficial = sum(label == "safe_beneficial" for label in labels)
    safe_nonbeneficial = sum(label == "safe_nonbeneficial" for label in labels)
    gates = {
        "gate_f_selection_passed": bool(gate_summary.get("selection_passed")),
        "at_least_four_treatment_blind_admissions": (
            len(admitted) >= MINIMUM_FEASIBLE_SEEDS
        ),
        "exact_seven_case_artifacts_fourteen_branches": (
            observed_case_names == expected_case_names and len(primary) * 2 + 2 == 14
        ),
        "repeat_seed_is_lowest_admitted": bool(
            admitted
            and REPEAT_SEED
            == min(int(row["seed"]) for row in admitted)
        ),
        "all_control_external_raw_omnibus_trace_identity": all(
            value["checks"]["physical_trace_digest_exact"]
            for value in external.values()
        ),
        "all_control_external_aggregate_identity": all(
            value["checks"]["anonymous_telemetry_digest_exact"]
            and value["checks"]["terminal_exact"]
            and value["checks"]["flow_population_exact"]
            for value in external.values()
        ),
        "all_control_normalized_microscopic_and_dynamic_identity": all(
            value["passed"] for value in normalized_external.values()
        ),
        "raw_omnibus_difference_confined_to_supervisor_progress_index_10": all(
            value["supervisor_difference_indices"] == [10]
            for value in normalized_external.values()
        ),
        "self_contained_repeat_identity": bool(repeat_result["passed"]),
        "all_predecision_identity": all(
            row["prefix_identity"]["passed"] for row in primary.values()
        ),
        "all_case_causal_integrity": all(
            row["classification"]["integrity_valid"]
            and _causal_integrity(row["treatment_contract"])
            for row in primary.values()
        ),
        "all_absolute_timing_and_terminal_integrity": all(
            row["arms"][arm]["timing_integrity"]["passed"]
            for row in primary.values()
            for arm in ("0", "1")
        ),
        "all_one_hz_telemetry_complete": all(
            row["arms"][arm]["telemetry_integrity"]["passed"]
            for row in primary.values()
            for arm in ("0", "1")
        ),
        "all_feasible_realize_full_dose": all(
            row["classification"]["dose_complete"] for row in admitted
        ),
        "all_feasible_safety_admissible": all(
            row["classification"]["safety_admissible"] for row in admitted
        ),
        "all_feasible_service_noninferior": all(
            row["classification"]["service_noninferior"] for row in admitted
        ),
        "infeasible_requests_apply_no_override": all(
            not row["treatment_contract"]["wall_override_steps"]
            for row in primary.values()
            if not row["classification"]["gate_f_admitted"]
        ),
        "multiple_independent_safe_benefits": beneficial >= 2,
        "multiple_independent_safe_nonbenefits": safe_nonbeneficial >= 2,
    }
    execution_integrity = all(
        gates[name]
        for name in (
            "gate_f_selection_passed",
            "at_least_four_treatment_blind_admissions",
            "exact_seven_case_artifacts_fourteen_branches",
            "repeat_seed_is_lowest_admitted",
            "all_control_external_aggregate_identity",
            "all_control_normalized_microscopic_and_dynamic_identity",
            "raw_omnibus_difference_confined_to_supervisor_progress_index_10",
            "self_contained_repeat_identity",
            "all_predecision_identity",
            "all_case_causal_integrity",
            "all_absolute_timing_and_terminal_integrity",
            "all_one_hz_telemetry_complete",
            "infeasible_requests_apply_no_override",
        )
    )
    expansion = bool(
        execution_integrity
        and gates["all_feasible_realize_full_dose"]
        and gates["all_feasible_safety_admissible"]
        and gates["all_feasible_service_noninferior"]
        and gates["multiple_independent_safe_benefits"]
        and gates["multiple_independent_safe_nonbenefits"]
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "contract": contract_manifest(),
        "gate_f_summary_fingerprint": gate_summary["contract"][
            "contract_fingerprint_sha256"
        ],
        "gate_f_bundle_provenance": gate_bundle,
        "density_bundle_provenance": density_bundle,
        "density_mapper_summary_sha256": _sha256_file(density_root / "summary.json"),
        "control_identity_reconciliation": {
            "post_execution_diagnostic_amendment": True,
            "original_raw_omnibus_trace_gate_passed": gates[
                "all_control_external_raw_omnibus_trace_identity"
            ],
            "corrected_physical_control_identity_passed": gates[
                "all_control_normalized_microscopic_and_dynamic_identity"
            ],
            "bundle_sha256": identity_audit["bundle_sha256"],
            "files": identity_audit["files"],
            "excluded_supervisor_observation_index": 10,
        },
        "branch_count": 14,
        "independent_seed_count": len(primary),
        "repeat_seed": REPEAT_SEED,
        "per_seed": [
            {
                "seed": seed,
                "gate_f_admitted": row["classification"]["gate_f_admitted"],
                "classification": row["classification"]["first_stage_label"],
                "dose_fraction": row["treatment_contract"]["treatment_dose"][
                    "realized_duration_fraction"
                ],
                "lane_coverage_fraction": row["treatment_contract"]["treatment_dose"][
                    "lane_coverage_fraction"
                ],
                "safety_admissible": row["classification"]["safety_admissible"],
                "service_noninferior": row["classification"]["service_noninferior"],
                "physical_effect": row["primary_physical_effect"],
                "control_primary": row["arms"]["0"][
                    "primary_physical_congestion_900_to_1080"
                ],
                "treatment_primary": row["arms"]["1"][
                    "primary_physical_congestion_900_to_1080"
                ],
                "safety_service_effects": row["safety_service_guardrail"]["effects"],
                "external_control_identity": {
                    "original_omnibus_and_aggregate": external[seed],
                    "normalized_microscopic_and_dynamic": normalized_external[seed],
                },
            }
            for seed, row in sorted(primary.items())
        ],
        "classification_counts": {
            label: labels.count(label) for label in sorted(set(labels))
        },
        "repeat_check": repeat_result,
        "gates": gates,
        "execution_integrity_passed": execution_integrity,
        "expanded_counterfactual_generation_recommended": expansion,
        "predictor_training_authorized": False,
        "validation_or_final_seeds_opened": False,
        "claims": {
            "predictor_works": False,
            "wall_prevents_congestion": False,
            "generalizes_beyond_development_seeds": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--sumo-binary", default=str(DEFAULT_SUMO_EXE))
    parser.add_argument("--gate-f", type=Path, default=DEFAULT_GATE_F)
    parser.add_argument("--density-mapper", type=Path, default=DEFAULT_DENSITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--identity-audit", type=Path, default=DEFAULT_IDENTITY_AUDIT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gate_f_root = args.gate_f.resolve()
    density_root = args.density_mapper.resolve()
    output = args.output.resolve()
    identity_root = args.identity_audit.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(parents=True, exist_ok=True)
    gate_summary = json.loads((gate_f_root / "summary.json").read_text(encoding="utf-8"))
    if not bool(gate_summary.get("selection_passed")):
        raise RuntimeError("frozen treatment-blind Gate F did not pass")
    if gate_summary["contract"] != contract_manifest():
        raise RuntimeError("Gate-F and matched-pilot contract fingerprints differ")
    if _sha256_file(density_root / "summary.json") != DENSITY_MAPPER_SUMMARY_SHA256:
        raise RuntimeError("density-mapper timing-selection provenance hash mismatch")
    for seed in DEVELOPMENT_SEEDS:
        _gate_f_row(gate_f_root, seed)
    admitted_seeds = [
        seed
        for seed in DEVELOPMENT_SEEDS
        if bool(_gate_f_row(gate_f_root, seed)["gate_f"]["admitted"])
    ]
    if not admitted_seeds or REPEAT_SEED != min(admitted_seeds):
        raise RuntimeError("repeat seed is not the treatment-blind lowest admitted seed")
    gate_bundle = _gate_f_bundle_provenance(gate_f_root)
    density_bundle = _density_bundle_provenance(density_root)
    cases = [
        {
            "seed": seed,
            "repeat": False,
            "gate_f": str(gate_f_root),
            "gate_f_bundle_sha256": gate_bundle["bundle_sha256"],
            "density_bundle_sha256": density_bundle["bundle_sha256"],
            "output": str(output),
            "sumo_binary": args.sumo_binary,
            "force": args.force,
        }
        for seed in DEVELOPMENT_SEEDS
    ] + [
        {
            "seed": REPEAT_SEED,
            "repeat": True,
            "gate_f": str(gate_f_root),
            "gate_f_bundle_sha256": gate_bundle["bundle_sha256"],
            "density_bundle_sha256": density_bundle["bundle_sha256"],
            "output": str(output),
            "sumo_binary": args.sumo_binary,
            "force": args.force,
        }
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_unix_s": time.time(),
        "contract": contract_manifest(),
        "gate_f_summary_fingerprint": gate_summary["contract"][
            "contract_fingerprint_sha256"
        ],
        "gate_f_root": str(gate_f_root),
        "gate_f_bundle_provenance": gate_bundle,
        "density_mapper_root": str(density_root),
        "density_bundle_provenance": density_bundle,
        "density_mapper_summary_sha256": DENSITY_MAPPER_SUMMARY_SHA256,
        "case_plan": [{"seed": row["seed"], "repeat": row["repeat"]} for row in cases],
        "branch_count": 14,
        "workers": min(max(1, args.workers), 8),
        "sumo_binary": args.sumo_binary,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_unix_s", None)
        expected = dict(manifest)
        expected.pop("created_unix_s", None)
        if comparable != expected:
            raise RuntimeError("existing wall-pilot manifest does not match frozen run")
    else:
        atomic_write_json(manifest_path, manifest)
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 8, len(cases))) as pool:
        futures = [pool.submit(run_case, case) for case in cases]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)
    summary = build_summary(output, gate_f_root, density_root, identity_root)
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["execution_integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
