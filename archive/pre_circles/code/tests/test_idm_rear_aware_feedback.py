import pytest

from idm.rear_aware_feedback import (
    FeedbackResponse,
    REAR_AWARE_FEEDBACK_V1,
    REAR_AWARE_FEEDBACK_V2,
    feedback_release_reason,
    treatment_strength,
)


def _response(**changes):
    values = {
        "disturbance_present": True,
        "disturbance_std_mps": 4.0,
        "rear_gap_m": 30.0,
        "rear_closing_speed_mps": 2.0,
        "follower_acceleration_mps2": -0.5,
        "local_acceleration_energy_m2ps4": 2.0,
        "rear_floor_feasible": True,
        "projection_survived": True,
        "command_delta_from_native_mps2": 0.2,
    }
    return FeedbackResponse(**(values | changes))


def test_probe_strength_ramps_then_holds_full_authority():
    assert [treatment_strength(step) for step in range(1, 7)] == [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.0,
    ]
    with pytest.raises(ValueError):
        treatment_strength(0)


def test_probe_holds_only_after_response_improves():
    baseline = _response()
    improved = _response(rear_closing_speed_mps=1.5)
    assert (
        feedback_release_reason(
            active_age_steps=REAR_AWARE_FEEDBACK_V1.probe_steps,
            baseline=baseline,
            current=improved,
            projection_survival_count=5,
            meaningful_command_count=5,
        )
        is None
    )
    assert (
        feedback_release_reason(
            active_age_steps=REAR_AWARE_FEEDBACK_V1.probe_steps,
            baseline=baseline,
            current=baseline,
            projection_survival_count=5,
            meaningful_command_count=5,
        )
        == "probe_no_improving_response"
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"rear_floor_feasible": False}, "rear_floor_infeasible"),
        ({"projection_survived": False}, "projection_conflict"),
        ({"follower_acceleration_mps2": -3.1}, "severe_follower_braking"),
    ],
)
def test_feedback_releases_on_frozen_safety_and_response_conditions(changes, reason):
    assert (
        feedback_release_reason(
            active_age_steps=3,
            baseline=_response(),
            current=_response(**changes),
            projection_survival_count=3,
            meaningful_command_count=3,
        )
        == reason
    )


def test_v2_soft_rear_trends_wait_for_probe_and_require_actual_risk():
    baseline = _response()
    ordinary_following = _response(
        rear_gap_m=26.0,
        rear_closing_speed_mps=3.0,
        disturbance_std_mps=3.8,
    )
    assert (
        feedback_release_reason(
            active_age_steps=3,
            baseline=baseline,
            current=ordinary_following,
            projection_survival_count=3,
            meaningful_command_count=3,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        is None
    )
    assert (
        feedback_release_reason(
            active_age_steps=5,
            baseline=baseline,
            current=ordinary_following,
            projection_survival_count=5,
            meaningful_command_count=5,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        is None
    )
    risk_relevant = _response(rear_gap_m=12.0, rear_closing_speed_mps=3.0)
    assert (
        feedback_release_reason(
            active_age_steps=5,
            baseline=baseline,
            current=risk_relevant,
            projection_survival_count=5,
            meaningful_command_count=5,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        == "rear_gap_compression"
    )


def test_v1_diagnostic_contract_retains_original_immediate_gap_guard():
    assert (
        feedback_release_reason(
            active_age_steps=3,
            baseline=_response(),
            current=_response(rear_gap_m=27.0),
            projection_survival_count=3,
            meaningful_command_count=3,
            contract=REAR_AWARE_FEEDBACK_V1,
        )
        == "rear_gap_compression"
    )


def test_v2_is_selected_by_default_and_identical_to_explicit_v2():
    baseline = _response()
    risky = _response(rear_gap_m=12.0, rear_closing_speed_mps=3.0)
    default_reason = feedback_release_reason(
        active_age_steps=3,
        baseline=baseline,
        current=risky,
        projection_survival_count=3,
        meaningful_command_count=3,
    )
    assert default_reason is None
    assert feedback_release_reason(
        active_age_steps=3,
        baseline=baseline,
        current=risky,
        projection_survival_count=3,
        meaningful_command_count=3,
        contract=REAR_AWARE_FEEDBACK_V2,
    ) == default_reason
    assert feedback_release_reason(
        active_age_steps=3,
        baseline=baseline,
        current=risky,
        projection_survival_count=3,
        meaningful_command_count=3,
        contract=REAR_AWARE_FEEDBACK_V1,
    ) == "rear_gap_compression"


def test_v1_and_v2_contract_identities_are_distinct():
    assert REAR_AWARE_FEEDBACK_V1.schema != REAR_AWARE_FEEDBACK_V2.schema
    v1 = REAR_AWARE_FEEDBACK_V1.to_dict()
    v2 = REAR_AWARE_FEEDBACK_V2.to_dict()
    assert v1["soft_response_checks_after_probe"] is False
    assert v2["soft_response_checks_after_probe"] is True
    assert v1["risk_conditioned_rear_response"] is False
    assert v2["risk_conditioned_rear_response"] is True
    assert v1 != v2


def test_v2_hard_rear_floor_releases_at_probe_step_one():
    assert (
        feedback_release_reason(
            active_age_steps=1,
            baseline=_response(),
            current=_response(rear_floor_feasible=False),
            projection_survival_count=1,
            meaningful_command_count=1,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        == "rear_floor_infeasible"
    )


def test_v2_severe_follower_braking_releases_immediately_during_probe():
    assert (
        feedback_release_reason(
            active_age_steps=1,
            baseline=_response(),
            current=_response(follower_acceleration_mps2=-3.1),
            projection_survival_count=1,
            meaningful_command_count=1,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        == "severe_follower_braking"
    )


def test_v2_same_follower_ordinary_trends_do_not_end_the_probe():
    baseline = _response(follower_id="follower.1")
    following = _response(
        follower_id="follower.1",
        rear_gap_m=26.0,
        rear_closing_speed_mps=3.0,
        disturbance_std_mps=3.8,
    )
    assert (
        feedback_release_reason(
            active_age_steps=3,
            baseline=baseline,
            current=following,
            projection_survival_count=3,
            meaningful_command_count=3,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        is None
    )


@pytest.mark.parametrize(
    ("baseline_id", "current_id"),
    [
        (None, "follower.1"),
        ("follower.1", None),
        ("follower.1", "follower.2"),
    ],
)
def test_follower_identity_appearance_disappearance_replacement_release(
    baseline_id, current_id
):
    assert (
        feedback_release_reason(
            active_age_steps=3,
            baseline=_response(follower_id=baseline_id),
            current=_response(follower_id=current_id),
            projection_survival_count=3,
            meaningful_command_count=3,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        == "follower_identity_changed"
    )


def test_follower_identity_change_takes_priority_over_hard_safety_reasons():
    assert (
        feedback_release_reason(
            active_age_steps=3,
            baseline=_response(follower_id="follower.1"),
            current=_response(follower_id="follower.2", rear_floor_feasible=False),
            projection_survival_count=3,
            meaningful_command_count=3,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        == "follower_identity_changed"
    )


def test_identity_change_releases_after_probe_too():
    improved_same_follower = _response(follower_id="follower.1", rear_closing_speed_mps=1.5)
    assert (
        feedback_release_reason(
            active_age_steps=6,
            baseline=_response(follower_id="follower.1"),
            current=improved_same_follower,
            projection_survival_count=6,
            meaningful_command_count=6,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        is None
    )
    assert (
        feedback_release_reason(
            active_age_steps=6,
            baseline=_response(follower_id="follower.1"),
            current=_response(follower_id="follower.9", rear_closing_speed_mps=1.5),
            projection_survival_count=6,
            meaningful_command_count=6,
            contract=REAR_AWARE_FEEDBACK_V2,
        )
        == "follower_identity_changed"
    )
