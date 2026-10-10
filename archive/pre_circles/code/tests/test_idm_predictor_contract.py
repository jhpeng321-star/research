from __future__ import annotations

import pytest

from counterfactual.idm_predictor_contract import (
    OBSERVATION_V2_FEATURES,
    complete_group_id,
    feature_contract_manifest,
    preflight_features,
    staged_targets,
    validate_decision_features,
)


def _row() -> dict:
    return {
        "seed": 4401,
        "decision_step": 660,
        "demand_mainline_vph": 2700,
        "cav_penetration": 0.2,
        "hdv_model": "IDM",
        "timing_scan": {"admitted": True},
        "features": {"health": 0.4, "decision_queue_count": 7},
        "treatment_dose": {
            "activation_feasible": True,
            "realized_duration_fraction": 1.0,
            "lane_coverage_fraction": 1.0,
            "assembly_completed": True,
            "holding_fraction_of_override": 0.8,
            "wall_geometry": {
                "geometry_mode": "compact_moving",
                "fraction_within_configured_span": 1.0,
            },
        },
        "safety_service_oracle": {
            "safety_admissible": True,
            "wall_selected": True,
            "effects": {
                "hard_brake_onset_excess": -3,
                "low_ttc_excess_steps": -4,
                "merge_discharge_benefit": 2,
                "delay_benefit_s": 250.0,
            },
        },
    }


def test_group_id_keeps_all_windows_from_one_seed_together() -> None:
    first = _row()
    second = {**first, "decision_step": 700}
    assert complete_group_id(first) == complete_group_id(second)
    assert complete_group_id(first) == "m2700|p0.200000|IDM|seed4401"


def test_feature_validator_is_allow_listed_and_rejects_outcomes() -> None:
    row = _row()
    assert validate_decision_features(
        row, ("context_decision_step", "health", "decision_queue_count")
    ) == (660.0, 0.4, 7.0)
    with pytest.raises(ValueError, match="outside the frozen allow-list"):
        validate_decision_features(row, ("delay_benefit_s",))


def test_effect_targets_exist_only_after_safe_completed_treatment() -> None:
    targets = staged_targets(_row())
    assert targets.operationally_admissible
    assert targets.joint_safe_beneficial
    assert targets.hard_brake_onset_reduction == 3.0
    assert targets.hard_brake_exposure_reduction_vehicle_steps is None
    assert targets.low_ttc_exposure_reduction_steps == 4.0
    incomplete = _row()
    incomplete["treatment_dose"] = {
        **incomplete["treatment_dose"],
        "realized_duration_fraction": 0.25,
    }
    refused = staged_targets(incomplete)
    assert not refused.operationally_admissible
    assert not refused.joint_safe_beneficial
    assert refused.delay_benefit_s is None


def test_targets_accept_current_vehicle_step_safety_effect() -> None:
    row = _row()
    effects = row["safety_service_oracle"]["effects"]
    effects["hard_brake_excess_vehicle_steps"] = -5
    effects.pop("hard_brake_onset_excess")
    targets = staged_targets(row)
    assert targets.hard_brake_onset_reduction is None
    assert targets.hard_brake_exposure_reduction_vehicle_steps == 5.0


def test_preflight_features_preserve_selected_wall_geometry() -> None:
    values = preflight_features(
        {
            "active_cav_count": 22,
            "damping_command_count": 2,
            "formation": {
                "controlled_cav_ids": ["c0", "c1"],
                "required_lane_ids": ["lane0", "lane1"],
                "longitudinal_span_m": 5.9,
                "predicted_max_span_m": 8.0,
                "predicted_terminal_span_m": 2.5,
            },
            "activation_safety": {
                "min_front_gap_m": 34.6,
                "min_rear_gap_m": 31.5,
                "min_front_ttc_s": None,
                "min_rear_ttc_s": 101.9,
                "min_acceleration_mps2": -0.1,
            },
        }
    )
    assert values["preflight_member_count"] == 2.0
    assert values["preflight_required_lane_count"] == 2.0
    assert values["preflight_min_front_ttc_s_capped"] == 120.0
    assert values["preflight_min_rear_gap_m"] == 31.5


def test_feature_contract_manifest_is_stable_and_complete() -> None:
    first = feature_contract_manifest()
    second = feature_contract_manifest()
    assert first == second
    assert first["feature_count"] == len(first["feature_names"])
    assert first["primary_feature_count"] == 29 + len(OBSERVATION_V2_FEATURES)
    assert first["expanded_feature_count"] == 74 + len(OBSERVATION_V2_FEATURES)
    assert set(first["primary_feature_names"]) < set(first["expanded_feature_names"])
    assert len(first["feature_names_sha256"]) == 64
    assert "context_is_hyper" not in first["feature_names"]
    assert first["feature_time_scope"] == ("control prefix through declared decision only")
    assert first["contract_version"] == "idm_counterfactual_escalation_predictor/2"
    assert first["observation_v2_feature_count"] == 27
    assert all(
        row["future_samples"] == 0
        for row in first["observation_v2_provenance"].values()
    )
