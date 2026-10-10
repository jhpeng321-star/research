from experiments.counterfactual.run_counterfactual_terminal_speed_sensitivity import (
    _strict_dose_geometry,
    _summarize,
)


def _result(
    case_id,
    fraction,
    *,
    strict=True,
    safe=True,
    selected=True,
    prevented=False,
):
    return {
        "case_id": case_id,
        "reduction_fraction": fraction,
        "wall_engaged": True,
        "strict_dose_geometry": strict,
        "safety_admissible": safe,
        "wall_selected": selected,
        "collapse_prevented": prevented,
        "requested_common_target_mps": 20.0,
        "terminal_actual_speed_mps": 19.5,
        "terminal_fraction_within_1_mps": 1.0,
        "hard_brake_exposure_excess_vehicle_steps": -1,
        "low_ttc_excess_steps": -1,
        "merge_discharge_benefit": 2,
        "delay_benefit_s": 100.0,
    }


def test_strict_dose_geometry_requires_both_contract_parts():
    assert _strict_dose_geometry(
        {
            "treatment_dose": {
                "dose_at_least_80_percent": True,
                "wall_geometry": {
                    "fraction_within_configured_span": 0.8,
                },
            }
        }
    )
    assert not _strict_dose_geometry(
        {
            "treatment_dose": {
                "dose_at_least_80_percent": False,
                "wall_geometry": {
                    "fraction_within_configured_span": 1.0,
                },
            }
        }
    )
    assert not _strict_dose_geometry(
        {
            "treatment_dose": {
                "dose_at_least_80_percent": True,
                "wall_geometry": {
                    "fraction_within_configured_span": 0.79,
                },
            }
        }
    )


def test_summary_keeps_speed_levels_and_state_dependent_selection_separate():
    summary = _summarize(
        [
            _result("case-a", 0.1, selected=True),
            _result("case-a", 0.3, selected=False, safe=False),
            _result("case-b", 0.1, selected=False, safe=False),
            _result("case-b", 0.3, selected=True, prevented=True),
        ]
    )

    assert summary["level_summary"]["r10"]["case_count"] == 2
    assert summary["level_summary"]["r30"]["wall_selected_strict_count"] == 1
    assert summary["level_summary"]["r30"]["collapse_prevention_strict_count"] == 1
    assert summary["case_comparison"]["case-a"]["selected_reduction_fractions"] == [0.1]
    assert summary["case_comparison"]["case-b"]["selected_reduction_fractions"] == [0.3]
    assert summary["state_dependent_selected_sets_observed"] is True
