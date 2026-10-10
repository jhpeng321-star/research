from counterfactual.safety_service_oracle import (
    ORACLE_VERSION,
    classify_safety_service_case,
    evaluate_safety_service_baselines,
    summarize_staged_safety_service,
)


def _outcomes(
    *,
    delay=1000.0,
    discharge=20,
    hard_brake_steps=10,
    hard_brake_onsets=6,
    hard_brake_unique=5,
    low_ttc=5,
    collisions=0,
):
    return {
        "incremental_delay_s": delay,
        "merge_discharge_count": discharge,
        "hard_brake_vehicle_steps": hard_brake_steps,
        "hard_brake_attribution": {
            "event_start_count": hard_brake_onsets,
            "unique_vehicle_count": hard_brake_unique,
        },
        "low_ttc_exposure_steps": low_ttc,
        "collision_unique_vehicle_count": collisions,
        "starting_teleport_unique_vehicle_count": 0,
    }


def _row(seed=1, *, wall=None, engaged=True):
    return {
        "case_id": f"case-{seed}",
        "seed": seed,
        "eligible": True,
        "arms": {
            "0": {"outcomes": _outcomes()},
            "1": {
                "wall_engaged": engaged,
                "outcomes": wall or _outcomes(),
            },
        },
    }


def test_delay_gain_alone_is_not_a_safety_service_benefit():
    wall = _outcomes(delay=500.0)
    result = classify_safety_service_case(_row(wall=wall))
    assert result["oracle_version"] == ORACLE_VERSION
    assert result["wall_selected"] is False
    assert result["service_noninferior"] is True
    assert result["meaningful_benefit"] is False


def test_hard_brake_onset_reduction_selects_wall_with_guardrails():
    wall = _outcomes(delay=1050.0, hard_brake_onsets=4)
    result = classify_safety_service_case(_row(wall=wall))
    assert result["wall_selected"] is True
    assert result["benefit_components"]["hard_brake_onset_reduction"] is True


def test_slowing_to_reduce_braking_fails_delay_noninferiority():
    wall = _outcomes(delay=1600.0, hard_brake_onsets=2)
    result = classify_safety_service_case(_row(wall=wall))
    assert result["meaningful_benefit"] is True
    assert result["service_noninferior"] is False
    assert result["wall_selected"] is False


def test_collision_and_onset_excess_are_safety_vetoes():
    collision = classify_safety_service_case(_row(wall=_outcomes(discharge=24, collisions=1)))
    onset_excess = classify_safety_service_case(
        _row(wall=_outcomes(discharge=24, hard_brake_onsets=9))
    )
    assert collision["safety_admissible"] is False
    assert onset_excess["safety_admissible"] is False


def test_staged_summary_separates_veto_safety_and_benefit():
    rows = [
        _row(1, engaged=False),
        _row(2, wall=_outcomes(discharge=24, collisions=1)),
        _row(3, wall=_outcomes(hard_brake_onsets=4)),
        _row(4),
    ]
    result = summarize_staged_safety_service(rows)
    assert result["formation_or_runtime_veto_case_count"] == 1
    assert result["engaged_case_count"] == 3
    assert result["safe_engagement_case_count"] == 2
    assert result["unsafe_engagement_case_count"] == 1
    assert result["selected_case_count"] == 1
    assert result["model_fit_permitted"] is False


def test_simple_baselines_report_precision_recall_and_burden():
    rows = [
        {
            **_row(1, wall=_outcomes(hard_brake_onsets=4)),
            "decision_step": 240,
            "scenario": {"cav_penetration": 0.3},
        },
        {**_row(2), "decision_step": 180, "scenario": {"cav_penetration": 0.2}},
        {**_row(3, engaged=False), "decision_step": 120, "scenario": {"cav_penetration": 0.1}},
    ]
    for row in rows:
        row["arms"]["1"]["wall_override_step_count"] = 5 if row["arms"]["1"]["wall_engaged"] else 0
    report = evaluate_safety_service_baselines(rows)
    assert report["paired_oracle_selected_case_count"] == 1
    assert report["baselines"]["always_command_wall"]["command_count"] == 3
    assert report["baselines"]["always_command_wall"]["engagement_count"] == 2
    assert report["baselines"]["step_240_only"]["beneficial_precision_among_engagements"] == 1.0
    assert report["baselines"]["paired_oracle"]["realized_override_steps"] == 5
