"""Focused contracts for projected-blend command-credit analysis."""

from tools.analyze_idm_p30_projected_blend import (
    EXPECTED_CASE_IDS,
    _case_has_three_levels,
    _command_contrast,
    _local_safety_complete,
    _arm_metrics,
)


def _branch(values, strength):
    return {
        "target_vehicle_id": "cav",
        "command_rows": [
            {
                "relative_step": index,
                "vehicle_id": "cav",
                "projected_acceleration_mps2": value,
                "projected_blend": {
                    "strength": strength,
                    "final_projection_idempotence_error_mps2": 0.0,
                },
                "treatment_infeasible": False,
            }
            for index, value in enumerate(values, start=1)
        ],
    }


def test_command_contrast_requires_two_distinct_steps_and_complete_diagnostics():
    reference = _branch((-1.0, -1.0, -1.0), 1.0)
    treatment = _branch((-0.9, -0.95, -1.0), 0.75)
    contrast = _command_contrast(reference, treatment)
    assert contrast["shared_command_steps"] == 3
    assert contrast["distinct_command_steps"] == 2
    assert contrast["meaningfully_distinct"] is True
    assert contrast["all_blend_diagnostics_present"] is True
    assert contrast["maximum_idempotence_error_mps2"] == 0.0


def test_case_level_strength_span_requires_three_levels_on_two_steps():
    branches = {
        1: _branch((0.0, 0.0), 0.25),
        2: _branch((-0.05, -0.05), 0.50),
        3: _branch((-0.10, -0.10), 0.75),
        4: _branch((-0.15, -0.15), 1.00),
    }
    assert _case_has_three_levels(branches) is True


def test_frozen_authority_case_set_has_fifteen_existing_decisions():
    assert len(EXPECTED_CASE_IDS) == 15
    assert "m2000.seed4129.t354" in EXPECTED_CASE_IDS


def test_nonclosing_populated_follower_cohort_is_infinite_ttc_not_missing():
    local = {
        "decision_cohort_vehicle_ids": ["upstream"],
        "stable_upstream_cohort": {
            "vehicle_step_count": 10,
            "minimum_ttc_s": 8.0,
            "minimum_acceleration_mps2": -1.0,
        },
        "dynamic_immediate_follower": {
            "vehicle_step_count": 5,
            "minimum_ttc_s": None,
            "minimum_acceleration_mps2": -0.5,
        },
    }
    assert _local_safety_complete(local) is True


def test_nonclosing_ttc_does_not_hide_missing_follower_samples():
    local = {
        "decision_cohort_vehicle_ids": ["upstream"],
        "stable_upstream_cohort": {
            "vehicle_step_count": 10,
            "minimum_ttc_s": 8.0,
            "minimum_acceleration_mps2": -1.0,
        },
        "dynamic_immediate_follower": {
            "vehicle_step_count": 0,
            "minimum_ttc_s": None,
            "minimum_acceleration_mps2": None,
        },
    }
    assert _local_safety_complete(local) is False


def test_arm_metrics_count_constraint_failure_classification_as_harmful():
    rows = [
        {"effect": {"score": 0.09, "classification": "harmful"}},
        {"effect": {"score": 0.01, "classification": "neutral"}},
    ]
    metrics = _arm_metrics(rows)
    assert metrics["classification_counts"] == {"harmful": 1, "neutral": 1}
    assert metrics["harmful_case_count"] == 1
    assert metrics["below_minus_0p05_count"] == 0
