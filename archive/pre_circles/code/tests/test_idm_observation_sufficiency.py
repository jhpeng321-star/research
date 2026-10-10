from __future__ import annotations

import numpy as np

from counterfactual.idm_observation_sufficiency import (
    audit_features,
    current_model_error_analysis,
    development_outer_folds,
    feature_families,
    nearest_neighbor_observability,
    prepare_population,
    shadow_rollout_reconstructibility,
)


def _case(seed: int, index: int, *, safe: bool, benefit: bool) -> dict:
    return {
        "case_id": f"seed{seed}-step{600 + index}",
        "seed": seed,
        "decision_step": 600 + index,
        "cav_penetration": 0.30,
        "demand_mainline_vph": 2700,
        "hdv_model": "IDM",
        "predecision_supervisor": {"admitted": True},
        "features": {
            "health": float(index),
            "pattern60_queue_count_mean": float(seed % 7),
            "local_lane_0_mean_speed_mps": 10.0 + float(index),
            "placement_pre_merge_cav_count": float((seed + index) % 5),
            "preflight_predicted_max_span_m": 20.0 + float(index),
            "preflight_min_rear_ttc_s_capped": 5.0 + float(seed % 3),
            "duplicate_health": float(index),
            "constant": 1.0,
        },
        "treatment_dose": {
            "activation_feasible": True,
            "realized_duration_fraction": 1.0,
            "lane_coverage_fraction": 1.0,
            "assembly_completed": True,
            "holding_fraction_of_override": 1.0,
            "wall_geometry": {
                "geometry_mode": "compact_moving",
                "fraction_within_configured_span": 1.0,
            },
        },
        "safety_service_oracle": {
            "safety_admissible": safe,
            "wall_selected": benefit,
            "effects": {"hard_brake_onset_reduction": float(benefit)},
        },
    }


def _population_rows() -> list[dict]:
    rows = []
    # Each seed has both benefit classes among safe cases, so every grouped
    # training fold remains estimable for all three diagnostic targets.
    for seed in range(100, 120):
        rows.append(_case(seed, 0, safe=True, benefit=bool(seed % 2)))
        rows.append(_case(seed, 1, safe=True, benefit=not bool(seed % 2)))
        rows.append(_case(seed, 2, safe=False, benefit=False))
    return rows


def test_feature_audit_reports_constants_duplicates_and_provenance() -> None:
    rows = _population_rows()
    audit = audit_features(rows)
    records = {row["name"]: row for row in audit["features"]}
    assert records["constant"]["constant"] is True
    duplicate_groups = [set(row["feature_names"]) for row in audit["exact_duplicate_groups"]]
    assert {"health", "duplicate_health"} in duplicate_groups
    assert records["preflight_predicted_max_span_m"]["source"] == (
        "deterministic_gate_f_preflight_at_decision"
    )
    rejected = {
        row["field"]: row for row in audit["explicitly_rejected_future_or_outcome_fields"]
    }
    assert rejected["treatment_dose"]["used_as_model_input"] is False
    assert rejected["safety_service_oracle"]["used_as_model_input"] is False


def test_feature_families_keep_static_baseline_physically_distinct() -> None:
    families = feature_families(audit_features(_population_rows()))
    assert "health" in families["A_static_traffic"]
    assert "pattern60_queue_count_mean" in families["B_temporal_speed_gap_density"]
    assert "local_lane_0_mean_speed_mps" in families["C_lane_resolved_traffic"]
    assert "placement_pre_merge_cav_count" in families[
        "D_wall_geometry_placement_authority"
    ]
    assert "preflight_predicted_max_span_m" in families[
        "E_predecision_dose_assembly_projection"
    ]
    assert "preflight_min_rear_ttc_s_capped" in families[
        "F_follower_compression_rear_safety"
    ]
    assert set(families["A_static_traffic"]).isdisjoint(
        set(families["C_lane_resolved_traffic"])
    )
    assert set().union(*(set(values) for key, values in families.items() if key != "G_combined_physical")) == set(
        families["G_combined_physical"]
    )


def test_outer_folds_never_split_one_seed() -> None:
    population = prepare_population(_population_rows())
    test_indices = []
    for train, test in development_outer_folds(population):
        assert not (set(population.groups[train]) & set(population.groups[test]))
        test_indices.extend(test.tolist())
    assert sorted(test_indices) == list(range(len(population.rows)))


def test_current_model_reports_separate_head_and_joint_confusions() -> None:
    rows = [
        _case(100, 0, safe=True, benefit=True),
        _case(101, 0, safe=True, benefit=False),
        _case(102, 0, safe=False, benefit=False),
        _case(103, 0, safe=True, benefit=True),
    ]
    payload = {
        "case_ids": [row["case_id"] for row in rows],
        "observed_safety": [1, 1, 0, 1],
        "observed_joint_benefit": [1, 0, 0, 1],
        "candidates": {
            "primary_shallow_boosting": {
                "safety_probability_oof": [0.9, 0.8, 0.8, 0.2],
                "conditional_benefit_probability_oof": [0.9, 0.9, 0.1, 0.9],
            }
        },
    }
    report = current_model_error_analysis(rows, payload)
    assert report["safety_head_confusion"] == {
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 0,
    }
    assert report["conditional_benefit_head_confusion_on_observed_safe_cases"] == {
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 0,
        "true_negative": 0,
    }
    assert report["joint_command_confusion"]["true_positive"] == 1


def test_nearest_neighbor_diagnosis_is_seed_grouped() -> None:
    population = prepare_population(_population_rows())
    report = nearest_neighbor_observability(population, ("health",))
    assert report["safety_admissible"]["independent_seed_count"] == 20
    for pair in report["safety_admissible"]["closest_opposite_label_pairs"]:
        assert pair["seed"] != pair["opposite_seed"]


def test_saved_aggregate_row_requires_new_shadow_extractor() -> None:
    report = shadow_rollout_reconstructibility(_population_rows())
    assert report["can_compute_exact_shadow_rollout_from_saved_rows"] is False
    protocol = report["deterministic_shadow_rollout_protocol"]
    assert protocol["planning_horizon_s"] == 10.0
    assert "projected_wall_induced_inflow_reduction_vph" in protocol["required_outputs"]
    assert report["required_new_predecision_extractor"]["latest_timestamp"] == (
        "declared_decision_before_real_branch_command"
    )
