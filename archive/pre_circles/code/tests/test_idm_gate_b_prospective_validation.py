from copy import deepcopy

import numpy as np

from counterfactual.idm_gate_b_prospective_validation import (
    evaluate_fixed_policy,
    score_admitted_rows,
    wilson_interval,
)
from experiments.idm.wall_v5_4_predictor_contract import (
    DEVELOPMENT_SEEDS,
    VALIDATION_SEEDS,
    predictor_manifest,
    validation_availability_rule,
)
from experiments.idm.wall_v5_4_predictor_validation_contract import (
    BENEFIT_THRESHOLD,
    SAFETY_THRESHOLD,
    prospective_gate,
)


class _ConstantModel:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, matrix):
        positive = np.full(len(matrix), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def _row(
    index: int,
    *,
    admitted: bool = True,
    dose: bool = True,
    safe: bool = False,
    beneficial: bool = False,
    catastrophic: bool = False,
):
    return {
        "case_id": f"case-{index}",
        "seed": 4601 + (index % 4),
        "decision_step": 620 + index,
        "features": {},
        "predecision_supervisor": {"admitted": admitted},
        "treatment_dose": {
            "activation_feasible": dose,
            "realized_duration_fraction": 1.0 if dose else 0.0,
            "lane_coverage_fraction": 1.0 if dose else 0.0,
            "assembly_completed": dose,
            "holding_fraction_of_override": 1.0 if dose else 0.0,
            "completion_reason": "completed" if dose else "aborted",
            "wall_geometry": {
                "geometry_mode": "compact_moving",
                "fraction_within_configured_span": 1.0 if dose else 0.0,
            },
        },
        "safety_service_oracle": {
            "safety_admissible": safe,
            "wall_selected": beneficial,
            "traffic_noninferior": beneficial,
            "meaningful_benefit": beneficial,
            "effects": {
                "collision_excess_vehicle_count": int(catastrophic),
                "teleport_excess_vehicle_count": 0,
                "delay_benefit_s": 0.0,
                "hard_brake_excess_vehicle_steps": 0.0,
                "hard_brake_onset_excess": 0.0,
                "low_ttc_excess_steps": 0.0,
                "merge_discharge_benefit": 0.0,
                "queue_area_benefit_vehicle_s": 0.0,
            },
        },
    }


def test_validation_partition_and_modest_coverage_gate_are_frozen() -> None:
    assert VALIDATION_SEEDS == tuple(range(4601, 4661))
    assert validation_availability_rule()["minimum_admitted_seeds"] == 30
    assert prospective_gate()["minimum_command_cases"] == 5
    assert prospective_gate()["minimum_command_seeds"] == 4
    assert prospective_gate()["minimum_selected_safety_rate"] == 0.80
    assert predictor_manifest("development", DEVELOPMENT_SEEDS)[
        "predictor_contract_sha256"
    ] == "46536de4077f60894126d716ea86af3b427a0660c27a0d7bd7727bda1c1713e9"


def test_wilson_interval_known_and_empty_denominator() -> None:
    interval = wilson_interval(8, 10)
    assert 0.49 < interval["lower"] < 0.50
    assert 0.94 < interval["upper"] < 0.95
    assert wilson_interval(0, 0) == {"lower": 0.0, "upper": 1.0}


def test_scoring_uses_only_gate_f_and_predecision_features() -> None:
    rows = [_row(0), _row(1, admitted=False)]
    bundle = {
        "feature_names": ["context_decision_step"],
        "safety_model": _ConstantModel(SAFETY_THRESHOLD),
        "conditional_benefit_model": _ConstantModel(BENEFIT_THRESHOLD),
    }
    original = score_admitted_rows(rows, bundle)
    mutated = deepcopy(rows)
    mutated[0]["treatment_dose"] = {"future": "mutated"}
    mutated[0]["safety_service_oracle"] = {"future": "mutated"}
    assert score_admitted_rows(mutated, bundle) == original
    assert [row["case_id"] for row in original] == ["case-0"]
    assert original[0]["selected"] is True


def test_operational_denominator_retains_selected_dose_failure() -> None:
    selected_rows = [
        _row(0, safe=True, beneficial=True),
        _row(1, safe=True, beneficial=True),
        _row(2, safe=True, beneficial=True),
        _row(3, safe=False, beneficial=False),
        _row(4, dose=False, safe=True, beneficial=True),
    ]
    baseline_only = [
        _row(5, safe=True),
        _row(6, safe=True),
        _row(7),
        _row(8),
        _row(9),
    ]
    rows = selected_rows + baseline_only
    predictions = [
        {
            "case_id": row["case_id"],
            "seed": row["seed"],
            "decision_step": row["decision_step"],
            "selected": index < 5,
        }
        for index, row in enumerate(rows)
    ]
    report = evaluate_fixed_policy(rows, predictions, integrity_passed=True)
    assert report["population"]["gate_f_admitted_case_count"] == 10
    assert report["feasible_always_wall_baseline"]["safety_rate"] == 0.6
    assert report["feasible_always_wall_baseline"]["joint_benefit_rate"] == 0.3
    assert report["selected_policy"]["selected_safety_rate"] == 0.8
    assert report["selected_policy"]["joint_benefit_precision"] == 0.6
    assert report["selected_policy"]["selected_dose_success_rate"] == 0.8
    assert report["selected_policy"]["joint_benefit_enrichment"] == 0.3
    assert report["validation_passed"] is True
    dose_failures = [
        row for row in report["selected_failure_or_nonbenefit_cases"]
        if not row["dose_qualified"]
    ]
    assert [row["case_id"] for row in dose_failures] == ["case-4"]


def test_selected_catastrophic_case_fails_without_being_removed() -> None:
    rows = [_row(index, safe=True, beneficial=True) for index in range(5)]
    rows[0]["safety_service_oracle"]["effects"]["collision_excess_vehicle_count"] = 1
    predictions = [
        {"case_id": row["case_id"], "selected": True} for row in rows
    ]
    report = evaluate_fixed_policy(rows, predictions, integrity_passed=True)
    assert report["gates"]["zero_selected_excess_collision_or_teleport"] is False
    assert report["validation_passed"] is False
    assert report["selected_policy"]["command_case_count"] == 5
