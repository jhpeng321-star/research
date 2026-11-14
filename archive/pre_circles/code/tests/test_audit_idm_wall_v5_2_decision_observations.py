from tools.audit_idm_wall_v5_2_decision_observations import audit_rows


def _row(seed, *, admitted, completed, safe, selected):
    geometry_fraction = 1.0 if completed else 0.5
    return {
        "case_id": f"seed{seed}",
        "seed": seed,
        "decision_step": 600 + seed,
        "feature_contract_version": "contract/1",
        "eligible": admitted,
        "features": {
            "health": 0.0,
            "pattern60_queue_count_slope_per_step": float(seed),
            "preflight_member_count": 2.0,
            "preflight_required_lane_count": 2.0,
            "preflight_predicted_max_span_m": 20.0,
            "preflight_min_front_ttc_s_capped": 30.0,
            "preflight_min_rear_ttc_s_capped": 30.0,
            "local_downstream_minus_upstream_speed_mps": 1.0,
        },
        "predecision_supervisor": {"admitted": admitted},
        "timing_scan": {"admitted": admitted},
        "prefix_identity": {"passed": True},
        "treatment_dose": {
            "activation_feasible": admitted,
            "realized_duration_fraction": 1.0,
            "lane_coverage_fraction": 1.0,
            "assembly_completed": True,
            "holding_fraction_of_override": 0.8,
            "wall_geometry": {
                "geometry_mode": "compact_moving",
                "fraction_within_configured_span": geometry_fraction,
            },
        },
        "arms": {
            "0": {"outcomes": {"hard_brake_vehicle_steps": 1}},
            "1": {
                "wall_engaged": admitted,
                "outcomes": {
                    "hard_brake_vehicle_steps": 2,
                    "hard_brake_attribution": {"event_starts": []},
                    "low_ttc_attribution": {"events": []},
                },
            },
        },
        "safety_service_oracle": {
            "safety_admissible": safe,
            "service_noninferior": selected,
            "wall_selected": selected,
            "refusal_reasons": [] if selected else ["test"],
            "effects": {
                "hard_brake_excess_vehicle_steps": 1,
                "collision_excess_vehicle_count": 0,
                "teleport_excess_vehicle_count": 0,
                "low_ttc_excess_steps": 0,
                "delay_benefit_s": 10.0,
                "merge_discharge_benefit": 1,
            },
        },
    }


def test_audit_separates_dose_and_joint_headroom_and_flags_downstream_gap():
    rows = [
        _row(1, admitted=True, completed=True, safe=True, selected=True),
        _row(2, admitted=True, completed=True, safe=False, selected=False),
        _row(3, admitted=True, completed=False, safe=False, selected=False),
    ]
    result = audit_rows(rows)
    assert result["dose_qualified_case_count"] == 2
    assert result["failed_dose_case_count"] == 1
    assert result["joint_positive_case_count"] == 1
    assert result["joint_negative_case_count"] == 1
    assert result["feature_audit"]["has_explicit_post_merge_edge_state"] is False
    assert result["feature_audit"][
        "premerge_relative_feature_names_that_are_not_post_merge_observations"
    ] == ["local_downstream_minus_upstream_speed_mps"]
    assert result["minimum_headroom_pass"] is False
    assert result["target_contract_error_by_case"] == {}
    assert result["target_joint_label_mismatch_case_ids"] == []


def test_audit_recognizes_explicit_post_merge_state_without_outcome_leakage():
    rows = [
        _row(seed, admitted=True, completed=seed <= 3, safe=seed <= 3, selected=seed <= 3)
        for seed in range(1, 7)
    ]
    for row in rows:
        row["features"]["decision_post_merge_mean_speed_mps"] = 20.0 + row["seed"]
    result = audit_rows(rows)
    assert result["feature_audit"]["has_explicit_post_merge_edge_state"] is True
    assert result["feature_audit"]["forbidden_token_feature_names"] == []
    assert result["minimum_headroom_checks"][
        "gate_f_failed_dose_spans_at_least_3_seeds"
    ] is True
