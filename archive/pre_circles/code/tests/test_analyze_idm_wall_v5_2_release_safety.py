from tools.analyze_idm_wall_v5_2_release_safety import (
    _time_bin,
    analyze_case,
    analyze_payloads,
)


def _outcomes(hard_brakes, events, phase_counts):
    return {
        "hard_brake_vehicle_steps": hard_brakes,
        "step_length_s": 1.0,
        "hard_brake_attribution": {
            "event_starts": events,
            "exposure_counts": {"phase": phase_counts},
        },
    }


def _event(step, road="downstream", vehicle_class="hdv"):
    return {
        "relative_step": step,
        "road_id": road,
        "vehicle_class": vehicle_class,
    }


def _case(*, safe=False):
    return {
        "case_id": "seed1_step100",
        "seed": 1,
        "decision_step": 100,
        "treatment_dose": {
            "activation_feasible": True,
            "requested_duration_steps": 20,
            "realized_duration_fraction": 1.0,
            "lane_coverage_fraction": 1.0,
            "assembly_completed": True,
            "holding_fraction_of_override": 0.8,
            "wall_geometry": {
                "geometry_mode": "compact_moving",
                "fraction_within_configured_span": 1.0,
                "samples": [
                    {"longitudinal_span_m": 10.0, "member_speed_std_mps": 0.5},
                    {"longitudinal_span_m": 12.0, "member_speed_std_mps": 0.8},
                ],
            },
            "safety_projection_clip_count": 1,
            "terminal_tail_speed_delivery": {
                "mean_absolute_target_error_mps": 0.3,
                "fraction_within_1_mps": 1.0,
            },
        },
        "arms": {
            "0": {
                "wall_engaged": False,
                "outcomes": _outcomes(2, [_event(110), _event(170)], {"damping": 2}),
            },
            "1": {
                "wall_engaged": True,
                "outcomes": _outcomes(
                    8,
                    [_event(110), _event(125), _event(150), _event(190)],
                    {"holding": 1, "cooldown": 1, "damping": 6},
                ),
            },
        },
        "safety_service_oracle": {
            "safety_admissible": safe,
            "budgets": {"hard_brake_excess_budget": 2},
            "effects": {"hard_brake_excess_vehicle_steps": 6},
        },
    }


def test_release_aligned_time_bins_are_boundary_exact():
    assert _time_bin(100, decision_step=100, release_step=121) == "through_decision"
    assert _time_bin(120, decision_step=100, release_step=121) == "active_wall"
    assert _time_bin(121, decision_step=100, release_step=121) == "post_release_0_20s"
    assert _time_bin(140, decision_step=100, release_step=121) == "post_release_0_20s"
    assert _time_bin(141, decision_step=100, release_step=121) == "post_release_20_60s"
    assert _time_bin(181, decision_step=100, release_step=121) == "post_release_after_60s"


def test_case_analysis_derives_release_and_preserves_paired_timing():
    result = analyze_case(_case())
    assert result is not None
    assert result["derived_release_step"] == 121
    assert result["paired_onset_delta_by_release_aligned_time"] == {
        "active_wall": 0,
        "post_release_0_20s": 1,
        "post_release_20_60s": 0,
        "post_release_after_60s": 1,
    }
    assert result["treatment_post_active_exposure_fraction"] == 7 / 8
    assert result["temporal_pattern"] == "mixed"
    assert result["terminal_mechanics"]["terminal_span_m"] == 12.0
    assert result["terminal_mechanics"]["terminal_five_sample_span_growth_m"] == 2.0


def test_aggregate_does_not_call_delayed_divergence_a_release_mechanism():
    case = _case()
    case["arms"]["1"]["outcomes"]["hard_brake_attribution"]["event_starts"] = [
        _event(110),
        _event(150),
        _event(190),
        _event(191),
    ]
    result = analyze_payloads([{"cases": [case]}])
    assert result["unsafe_dose_qualified_case_count"] == 1
    assert result["release_boundary_localized"] is False
    assert "does not localize" in result["interpretation"]


def test_inadequate_compact_geometry_is_not_analyzed_as_completed():
    case = _case()
    case["treatment_dose"]["wall_geometry"]["fraction_within_configured_span"] = 0.55
    assert analyze_case(case) is None
