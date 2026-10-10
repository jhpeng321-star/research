from types import SimpleNamespace

from experiments.idm.run_idm_v5_5_t880_wall_pilot import (
    _absolute_timing_integrity,
    _case_classification,
    _causal_integrity,
    _dose_complete,
    _physical_effect,
    _relative_decision,
    _telemetry_integrity,
    _terminal_integrity,
)
from idm.wall_t880 import congestion_run_summary, upstream_storage_summary


def _trace(*, treatment: bool = True, natural: bool = False):
    reset_time = 4.0
    decision = 876
    final_step = 1030 if natural else 1076
    snapshots = tuple(
        SimpleNamespace(
            relative_step=step,
            sim_time=reset_time + step,
            vehicle_count=0 if step == final_step and natural else 1,
        )
        for step in range(final_step + 1)
    )
    audits = []
    for step in range(final_step + 1):
        command = int(treatment and decision < step <= decision + 20)
        audits.append(
            {
                "relative_step": step,
                "command": command,
                "wall_override_ids": ["cav"] if command else [],
                "wall_active": bool(command),
            }
        )
    metadata = {
        "termination_reason": "terminated" if natural else "horizon_reached",
        "terminal_expected_vehicle_count": 0 if natural else None,
    }
    return SimpleNamespace(
        snapshots=snapshots,
        controller_audits=tuple(audits),
        controller_metadata=metadata,
    )


def test_relative_decision_uses_seed_specific_reset_offset() -> None:
    assert _relative_decision({"reset_info": {"episode_time": 4.0}}) == 876
    assert _relative_decision({"reset_info": {"episode_time": 1.0}}) == 879
    assert _relative_decision({"reset_info": {"episode_time": 2.0}}) == 878


def test_absolute_timing_contract_is_not_off_by_one() -> None:
    result = _absolute_timing_integrity(_trace(), decision_step=876, treatment=True)
    assert result["passed"] is True
    assert result["pulse_action_times_s"] == list(map(float, range(880, 900)))
    assert result["pulse_result_times_s"] == list(map(float, range(881, 901)))
    assert result["release_action_time_s"] == 900.0


def test_natural_clearance_requires_zero_live_and_expected_vehicles() -> None:
    result = _terminal_integrity(_trace(natural=True))
    assert result["proven_natural_clearance"] is True
    assert result["valid_common_horizon"] is True
    trace = _trace(natural=True)
    trace.controller_metadata["terminal_expected_vehicle_count"] = 1
    assert _terminal_integrity(trace)["valid_common_horizon"] is False


def test_telemetry_integrity_rejects_an_interior_clock_gap() -> None:
    density = {
        "telemetry": [
            {"sim_time_s": float(second)} for second in range(880, 901)
        ],
        "terminal": {"sim_time_s": 900.0, "reason": "natural_clearance"},
        "clock_integrity": {"one_hz": True, "exact_anchor_observed": True},
    }
    assert _telemetry_integrity(density)["passed"] is True
    density["telemetry"].pop(10)
    assert _telemetry_integrity(density)["passed"] is False


def test_full_twenty_second_dose_is_required() -> None:
    audit = {
        "checks": {
            "taper_is_monotone_and_reaches_zero": True,
            "release_returns_to_damping": True,
        },
        "treatment_dose": {
            "realized_duration_fraction": 1.0,
            "lane_coverage_fraction": 1.0,
            "assembly_completed": True,
            "abort_count": 0,
            "completion_reason": "planned_duration_complete",
        },
    }
    assert _dose_complete(audit) is True
    audit["treatment_dose"]["realized_duration_fraction"] = 0.95
    assert _dose_complete(audit) is False


def test_every_causal_identity_layer_is_required_even_without_engagement() -> None:
    audit = {
        "checks": {
            "prefix_base_action_identity": True,
            "first_request_same_base_actions": True,
            "override_only_during_treatment": True,
            "out_of_band_damping_preserved": True,
            "wall_ids_removed_from_env_action_path": True,
            "release_returns_to_damping": True,
        }
    }
    assert _causal_integrity(audit) is True
    audit["checks"]["out_of_band_damping_preserved"] = False
    assert _causal_integrity(audit) is False


def test_failure_stage_order_and_safe_benefit() -> None:
    safe = {"safety_admissible": True, "traffic_noninferior": True}
    benefit = {"primary_physical_benefit": True}
    assert _case_classification(
        gate_f_admitted=False,
        integrity_valid=True,
        dose_complete=False,
        safety_service=safe,
        physical_effect=benefit,
    )["first_stage_label"] == "infeasible_formation"
    assert _case_classification(
        gate_f_admitted=True,
        integrity_valid=True,
        dose_complete=False,
        safety_service=safe,
        physical_effect=benefit,
    )["first_stage_label"] == "failed_realized_dose"
    assert _case_classification(
        gate_f_admitted=True,
        integrity_valid=True,
        dose_complete=True,
        safety_service=safe,
        physical_effect=benefit,
    )["first_stage_label"] == "safe_beneficial"
    unsafe_service = {"safety_admissible": True, "traffic_noninferior": False}
    assert _case_classification(
        gate_f_admitted=True,
        integrity_valid=True,
        dose_complete=True,
        safety_service=unsafe_service,
        physical_effect={"primary_physical_benefit": False},
    )["first_stage_label"] == "service_failure"


def test_prevention_cannot_game_the_sixty_second_threshold() -> None:
    def arm(sustained, sustained_duration, predicate_duration, longest):
        return {
            "primary_physical_congestion_900_to_1080": {
                "sustained_event": sustained,
                "sustained_duration_s": sustained_duration,
                "predicate_duration_s": predicate_duration,
                "longest_run_s": longest,
            }
        }

    control = arm(True, 60, 60, 60)
    treatment = arm(False, 0, 59, 59)
    assert _physical_effect(control, treatment)["primary_physical_benefit"] is False
    treatment = arm(False, 0, 40, 40)
    assert _physical_effect(control, treatment)["primary_physical_benefit"] is True


def test_physical_run_and_storage_summaries() -> None:
    telemetry = [
        {
            "sim_time_s": float(second),
            "primary_congestion_predicate": 900 <= second <= 959,
            "premerge_count": 3,
            "ramp_count": 2,
        }
        for second in range(880, 961)
    ]
    congestion = congestion_run_summary(
        telemetry, start_s=900, end_s=960, sustain_s=60
    )
    assert congestion["sustained_event"] is True
    assert congestion["sustained_duration_s"] == 60
    assert congestion["first_sustained_confirmation_s"] == 959.0
    storage = upstream_storage_summary(telemetry, start_s=880, end_s=960)
    assert storage["combined_upstream_vehicle_seconds"] == 400.0
