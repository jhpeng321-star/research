from __future__ import annotations

from pathlib import Path
import argparse
import json

import pytest

from counterfactual.twin_replay import ReplayTrace, StepSnapshot
from experiments.idm.idm_v5_5_density_mapper_contract import (
    ANALYSIS_CLOCK,
    ANALYSIS_END_S,
    ANALYSIS_START_S,
    CAV_PENETRATION,
    DEVELOPMENT_SEEDS,
    MAINLINE_DEMANDS_VPH,
    PREVALENCE_LOOKBACK_S,
    PRIMARY_QUEUE_COUNT,
    PRIMARY_SPEED_THRESHOLD_MPS,
    PRIMARY_SUSTAIN_S,
    RAMP_DEMAND_VPH,
    REPEAT_SEED,
    SECONDARY_HEALTH_SUSTAIN_S,
    SECONDARY_HEALTH_THRESHOLD,
    scenario_path,
    validate_inputs,
)
from idm.density_mapper import branch_integrity, classify_endpoint, summarize_density_trace
from experiments.idm.run_idm_v5_5_damping_density_mapper import (
    _manifest_arguments,
    _regime,
    _repeat_check,
)


def _rows(start: int, end: int, active: set[int]) -> list[dict]:
    return [
        {"sim_time_s": float(second), "active": second in active}
        for second in range(start, end + 1)
    ]


def _classify(rows: list[dict], terminal_reason: str = "horizon_reached") -> dict:
    return classify_endpoint(
        rows,
        predicate=lambda row: bool(row["active"]),
        sustain_s=60,
        analysis_start_s=620,
        analysis_end_s=1080,
        terminal_reason=terminal_reason,
    )


def test_prevalence_requires_a_complete_run_at_anchor() -> None:
    prevalent = _classify(_rows(500, 1080, set(range(561, 701))))
    incipient = _classify(_rows(500, 1080, set(range(562, 622))))
    assert prevalent["classification"] == "prevalent_at_anchor"
    assert prevalent["selected_onset_s"] == 561.0
    assert incipient["classification"] == "incident_after_anchor"
    assert incipient["selected_onset_s"] == 562.0
    assert incipient["selected_confirmation_s"] == 621.0


def test_late_unfinished_run_is_right_censored() -> None:
    result = _classify(_rows(500, 1080, set(range(1050, 1081))))
    assert result["classification"] == "censored_or_invalid"
    assert result["terminal_unfinished_run_s"] == 31
    assert result["latest_fully_observable_onset_s"] == 1021


def test_natural_clearance_is_recovery_not_fabricated_tail() -> None:
    result = _classify(
        _rows(500, 950, set(range(700, 710))), terminal_reason="natural_clearance"
    )
    assert result["classification"] == "no_incident_or_recovered"
    assert result["natural_clearance_treated_as_recovery"] is True


def test_secondary_health_screen_does_not_skip_first_100_window_samples() -> None:
    rows = _rows(620, 679, set(range(620, 680)))
    result = classify_endpoint(
        rows,
        predicate=lambda row: bool(row["active"]),
        sustain_s=60,
        analysis_start_s=620,
        analysis_end_s=1080,
        terminal_reason="natural_clearance",
    )
    assert result["sustained_event"] is True
    assert result["first_sustained_onset_s"] == 620.0


def _snapshot(step: int, *, health: float, congested: bool, empty: bool = False) -> StepSnapshot:
    vehicles = ()
    if not empty:
        speed = 10.0 if congested else 25.0
        vehicles = tuple(
            (
                f"mainline.{index}",
                "1191554636",
                f"1191554636_{index % 3}",
                900.0 - index * 10.0,
                speed,
                0.0,
                0.0,
                0,
                "normal_car",
                1.1,
            )
            for index in range(5)
        )
    return StepSnapshot(
        relative_step=step,
        sim_time=float(step),
        action=0,
        supervisor_observation=(health,),
        vehicles=vehicles,
        active_cavs=(() if empty else ("mainline.0",)),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=f"step-{step}",
        vehicle_count=len(vehicles),
        mean_speed=(0.0 if empty else (10.0 if congested else 25.0)),
        active_time_loss_sum=0.0,
    )


def test_trace_summary_uses_absolute_time_and_independent_health_label() -> None:
    snapshots = tuple(
        _snapshot(
            step,
            health=(0.10 if 621 <= step <= 680 else 0.90),
            congested=(700 <= step <= 759),
        )
        for step in range(560, 1081)
    )
    audits = tuple(
        {
            "relative_step": step,
            "command": 0,
            "wall_active": False,
            "wall_override_ids": (),
            "base_diagnostics": {"commands": ()},
        }
        for step in range(560, 1081)
    )
    trace = ReplayTrace(
        seed=4401,
        decision_step=0,
        horizon_steps=1080,
        prefix_action=0,
        post_action=0,
        treatment_steps=None,
        reset_info={"episode_time": 560.0},
        snapshots=snapshots,
        terminated_early=False,
        controller_metadata={"termination_reason": "horizon_reached"},
        controller_audits=audits,
    )
    record = summarize_density_trace(
        trace,
        analysis_start_s=620,
        analysis_end_s=1080,
        primary_queue_count=5,
        primary_speed_threshold_mps=15.0,
        primary_sustain_s=60,
        secondary_health_threshold=0.15,
        secondary_health_sustain_s=60,
    )
    assert record["clock_integrity"]["one_hz"] is True
    assert record["clock_integrity"]["reset_offset_s"] == 560.0
    assert record["primary_premerge_endpoint"]["classification"] == "incident_after_anchor"
    assert record["primary_premerge_endpoint"]["selected_onset_s"] == 700.0
    assert record["secondary_legacy_health_diagnostic"]["selected_onset_s"] == 621.0
    assert record["label_agreement"] is True
    assert all(
        branch_integrity(
            record,
            analysis_start_s=620,
            analysis_end_s=1080,
            prevalence_lookback_s=60,
        ).values()
    )


def test_frozen_contract_resolves_2200_separately_and_rejects_p20() -> None:
    assert scenario_path(2200).name == "osm.dense.idm_m2200r400.sumocfg"
    assert scenario_path(2400).name == "osm.dense.idm_m2400r400.sumocfg"
    assert scenario_path(2700).name == "osm.dense.idm_m2700r400.sumocfg"
    common = dict(
        demands=MAINLINE_DEMANDS_VPH,
        ramp_demand=RAMP_DEMAND_VPH,
        cav_penetration=CAV_PENETRATION,
        seeds=DEVELOPMENT_SEEDS,
        analysis_clock=ANALYSIS_CLOCK,
        analysis_start=ANALYSIS_START_S,
        analysis_end=ANALYSIS_END_S,
        prevalence_lookback_s=PREVALENCE_LOOKBACK_S,
        primary_queue_count=PRIMARY_QUEUE_COUNT,
        primary_speed_threshold_mps=PRIMARY_SPEED_THRESHOLD_MPS,
        primary_sustain_s=PRIMARY_SUSTAIN_S,
        secondary_health_threshold=SECONDARY_HEALTH_THRESHOLD,
        secondary_health_sustain_s=SECONDARY_HEALTH_SUSTAIN_S,
        repeat_seed=REPEAT_SEED,
    )
    validate_inputs(**common)
    with pytest.raises(ValueError, match="frozen contract"):
        validate_inputs(**{**common, "cav_penetration": 0.20})


def test_manifest_arguments_serialize_path() -> None:
    values = _manifest_arguments(
        argparse.Namespace(output=Path("runs/probe/example"), workers=8, force=False)
    )
    assert values["output"] == str(Path("runs/probe/example"))
    json.dumps(values)


def _regime_row(
    classification: str,
    *,
    diagnostic_right_censored: bool = False,
    health_event: bool = False,
) -> dict:
    primary_event = classification in {"prevalent_at_anchor", "incident_after_anchor"}
    return {
        "primary_premerge_endpoint": {
            "classification": classification,
            "sustained_event": primary_event,
            "diagnostic_right_censored": diagnostic_right_censored,
        },
        "secondary_legacy_health_diagnostic": {"sustained_event": health_event},
        "label_agreement": primary_event == health_event,
    }


@pytest.mark.parametrize(
    ("incident_count", "expected"),
    [
        (2, "lower_transition_candidate"),
        (3, "upper_transition_candidate"),
        (4, "upper_transition_candidate"),
        (5, "overload_candidate"),
    ],
)
def test_regime_categories_use_actionable_incidence(
    incident_count: int, expected: str
) -> None:
    rows = [_regime_row("incident_after_anchor") for _ in range(incident_count)]
    rows.extend(_regime_row("no_incident_or_recovered") for _ in range(6 - incident_count))
    assert _regime(rows)["category"] == expected


def test_prevalence_and_adverse_trend_do_not_manufacture_headroom() -> None:
    prevalence = [_regime_row("prevalent_at_anchor") for _ in range(2)]
    prevalence.extend(_regime_row("no_incident_or_recovered") for _ in range(4))
    assert _regime(prevalence)["category"] == "prevalence_contaminated"

    adverse = [_regime_row("incident_after_anchor") for _ in range(2)]
    adverse.extend(_regime_row("no_incident_or_recovered") for _ in range(3))
    adverse.append(
        _regime_row("no_incident_or_recovered", diagnostic_right_censored=True)
    )
    assert _regime(adverse)["category"] == "censored_or_invalid"


def test_health_diagnostic_cannot_change_primary_regime() -> None:
    false_health = [_regime_row("incident_after_anchor", health_event=False) for _ in range(2)]
    false_health.extend(_regime_row("no_incident_or_recovered", health_event=False) for _ in range(4))
    true_health = [_regime_row("incident_after_anchor", health_event=True) for _ in range(2)]
    true_health.extend(_regime_row("no_incident_or_recovered", health_event=True) for _ in range(4))
    assert _regime(false_health)["category"] == _regime(true_health)["category"]


def test_short_horizon_and_missing_lookback_fail_integrity() -> None:
    record = {
        "terminal": {"reason": "horizon_reached", "sim_time_s": 700.0},
        "clock_integrity": {
            "one_hz": True,
            "first_sim_time_s": 600.0,
            "exact_anchor_observed": True,
            "exact_endpoint_observed": False,
        },
        "control": {"wall_command_or_override_count": 0},
        "safety": {"collision_vehicle_steps": 0, "starting_teleport_vehicle_steps": 0},
        "telemetry": [{"sim_time_s": float(value)} for value in range(600, 701)],
    }
    gates = branch_integrity(
        record,
        analysis_start_s=620,
        analysis_end_s=1080,
        prevalence_lookback_s=60,
    )
    assert gates["prevalence_lookback_observed"] is False
    assert gates["absolute_window_coverage"] is False
    assert gates["exact_endpoint_or_proven_natural_clearance"] is False


def _trace(
    snapshots: tuple[StepSnapshot, ...],
    *,
    terminated_early: bool,
    termination_reason: str,
) -> ReplayTrace:
    audits = tuple(
        {
            "relative_step": snapshot.relative_step,
            "command": 0,
            "wall_active": False,
            "wall_override_ids": (),
            "base_diagnostics": {"commands": ()},
        }
        for snapshot in snapshots
    )
    return ReplayTrace(
        seed=4401,
        decision_step=0,
        horizon_steps=1080,
        prefix_action=0,
        post_action=0,
        treatment_steps=None,
        reset_info={"episode_time": float(snapshots[0].sim_time)},
        snapshots=snapshots,
        terminated_early=terminated_early,
        controller_metadata={"termination_reason": termination_reason},
        controller_audits=audits,
    )


def _summarize(trace: ReplayTrace) -> dict:
    return summarize_density_trace(
        trace,
        analysis_start_s=620,
        analysis_end_s=1080,
        prevalence_lookback_s=60,
        primary_queue_count=5,
        primary_speed_threshold_mps=15.0,
        primary_sustain_s=60,
        secondary_health_threshold=0.15,
        secondary_health_sustain_s=60,
    )


def test_post_endpoint_clearance_cannot_change_1080_censoring_or_population() -> None:
    snapshots = tuple(
        _snapshot(
            step,
            health=0.9,
            congested=1050 <= step <= 1081,
            empty=step == 1082,
        )
        for step in range(560, 1083)
    )
    record = _summarize(
        _trace(snapshots, terminated_early=True, termination_reason="terminated")
    )
    assert record["terminal"]["reason"] == "horizon_reached"
    assert record["terminal"]["sim_time_s"] == 1080.0
    assert record["terminal"]["vehicle_count"] == 5
    assert record["terminal"]["post_endpoint_samples_discarded"] == 2
    assert record["primary_premerge_endpoint"]["classification"] == "censored_or_invalid"


def test_incident_then_natural_clearance_reports_recovered_event() -> None:
    snapshots = tuple(
        _snapshot(
            step,
            health=0.9,
            congested=700 <= step <= 759,
            empty=step == 800,
        )
        for step in range(560, 801)
    )
    record = _summarize(
        _trace(snapshots, terminated_early=True, termination_reason="terminated")
    )
    endpoint = record["primary_premerge_endpoint"]
    assert endpoint["classification"] == "incident_after_anchor"
    assert endpoint["terminal_recovered"] is True
    assert endpoint["sustained_then_recovered"] is True
    assert endpoint["latest_fully_observable_onset_s"] == 741.0


def _repeat_payload() -> dict:
    return {
        "trace_digest": "trace",
        "controller_audit_digest": "audit",
        "telemetry_digest": "telemetry",
        "primary_premerge_endpoint": {"classification": "incident_after_anchor"},
        "secondary_legacy_health_diagnostic": {"classification": "no_incident_or_recovered"},
        "terminal": {"reason": "horizon_reached"},
        "flow_and_population": {"downstream_discharges": 10},
    }


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("trace_digest", "other"),
        ("controller_audit_digest", "other"),
        ("telemetry_digest", "other"),
        ("terminal", {"reason": "natural_clearance"}),
    ],
)
def test_repeat_check_fails_each_identity_layer(key: str, replacement: object) -> None:
    primary = _repeat_payload()
    repeat = _repeat_payload()
    repeat[key] = replacement
    assert _repeat_check(primary, repeat)["passed"] is False
