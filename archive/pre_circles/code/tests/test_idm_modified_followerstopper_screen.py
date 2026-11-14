"""Focused contracts for the matched modified-FollowerStopper analyzer."""

from tools.analyze_idm_modified_followerstopper_screen import (
    EXPECTED_VALIDATION_CASE_IDS,
    _command_contrast,
    _local_safety,
)


def _branch(accelerations, *, feedback=False):
    return {
        "target_vehicle_id": "cav",
        "command_rows": [
            {
                "relative_step": index,
                "vehicle_id": "cav",
                "projected_acceleration_mps2": value,
                "modified_follower_stopper": (
                    {"zone": "free" if index < 3 else "closing_transition"}
                    if feedback
                    else None
                ),
                "treatment_infeasible": False,
            }
            for index, value in enumerate(accelerations, start=1)
        ],
        "phase_response": {
            "steps": [
                {
                    "upstream_followers": [
                        {
                            "vehicle_id": "f1",
                            "ttc_s": 8.0 - index,
                            "acceleration_mps2": -1.0 - index,
                        },
                        {
                            "vehicle_id": "churn",
                            "ttc_s": 0.5,
                            "acceleration_mps2": -9.0,
                        },
                    ],
                    "target_states": [
                        {
                            "immediate_follower": {
                                "vehicle_id": "f1",
                                "ttc_s": 8.0 - index,
                                "acceleration_mps2": -1.0 - index,
                            }
                        }
                    ],
                }
                for index in range(2)
            ]
        },
    }


def test_command_contrast_requires_two_meaningful_steps_and_feedback_rows():
    contrast = _command_contrast(
        _branch((0.0, 0.0, 0.0)),
        _branch((0.0, -0.1, -0.2), feedback=True),
    )
    assert contrast["shared_command_steps"] == 3
    assert contrast["distinct_command_steps"] == 2
    assert contrast["meaningfully_distinct"] is True
    assert contrast["all_feedback_diagnostics_present"] is True
    assert contrast["all_treatment_commands_compared"] is True
    assert contrast["treatment_zone_counts"] == {
        "closing_transition": 1,
        "free": 2,
    }


def test_local_safety_uses_the_latched_cohort_and_reports_immediate_follower():
    safety = _local_safety(_branch((0.0, 0.0)), ("f1",))
    stable = safety["stable_upstream_cohort"]
    immediate = safety["dynamic_immediate_follower"]
    assert stable["vehicle_step_count"] == 2
    assert stable["minimum_ttc_s"] == 7.0
    assert stable["minimum_acceleration_mps2"] == -2.0
    assert stable["severe_braking_exposure_steps"] == 0
    assert immediate == stable


def test_frozen_validation_case_set_excludes_the_viewed_design_case():
    assert len(EXPECTED_VALIDATION_CASE_IDS) == 14
    assert "m2000.seed4129.t354" not in EXPECTED_VALIDATION_CASE_IDS
