"""Focused contract tests for the bounded IDM Gate-B model screen."""

from __future__ import annotations

import numpy as np

from counterfactual.escalation_v2_model import grouped_splits
from counterfactual.idm_gate_b_model_screen import (
    CANDIDATE_SPECS,
    OBSERVATION_V3_CANDIDATE_SPECS,
    feature_diagnostics,
    grouped_oof_candidate,
    make_candidate_estimator,
    run_model_screen,
    run_observation_v3_screen,
    screen_manifest,
)
from counterfactual.idm_observation_v3_contract import (
    IDM_OBSERVATION_V3_FEATURES,
    feature_contract_manifest as observation_v3_contract_manifest,
)
from counterfactual.idm_predictor_contract import (
    IDM_DECISION_FEATURES,
    IDM_PRIMARY_FEATURES,
)
from counterfactual.idm_two_stage_gate import PreparedGateB


def _grouped_prepared() -> PreparedGateB:
    """Give every seed both safe benefit classes plus one unsafe case."""
    rows = []
    matrix = []
    groups = []
    safety = []
    benefit = []
    for seed in range(10):
        for case_index, (safe, selected) in enumerate(((0, 0), (1, 0), (1, 1))):
            rows.append({"case_id": f"case-{seed}-{case_index}"})
            matrix.append((float(seed), float(seed * seed)))
            groups.append(f"seed-{seed}")
            safety.append(safe)
            benefit.append(selected)
    return PreparedGateB(
        rows=tuple(rows),
        matrix=np.asarray(matrix),
        groups=np.asarray(groups),
        safety=np.asarray(safety),
        joint_benefit=np.asarray(benefit),
    )


def _screen_rows() -> list[dict]:
    rows = []
    for seed in range(10):
        features = {
            name: (float(seed) if name == "health" else 1.0)
            for name in IDM_OBSERVATION_V3_FEATURES
            if name not in {"context_decision_step", "context_cav_penetration"}
        }
        labels = ((False, False), (True, False), (True, True))
        for case_index, (safe, selected) in enumerate(labels):
            rows.append(
                {
                    "case_id": f"screen-{seed}-{case_index}",
                    "seed": seed,
                    "decision_step": 180,
                    "demand_mainline_vph": 2400,
                    "cav_penetration": 0.2,
                    "hdv_model": "IDM",
                    "features": dict(features),
                    "treatment_dose": {
                        "activation_feasible": True,
                        "realized_duration_fraction": 1.0,
                        "lane_coverage_fraction": 1.0,
                    },
                    "safety_service_oracle": {
                        "safety_admissible": safe,
                        "wall_selected": selected,
                    },
                }
            )
    return rows


def test_screen_manifest_preserves_declared_candidate_order() -> None:
    expected = [
        ("primary_logistic_reference", "primary", "regularized_logistic"),
        ("expanded_logistic", "expanded", "regularized_logistic"),
        ("primary_additive_spline", "primary", "additive_spline_logistic"),
        ("expanded_additive_spline", "expanded", "additive_spline_logistic"),
        ("primary_shallow_boosting", "primary", "shallow_gradient_boosting"),
        ("expanded_shallow_boosting", "expanded", "shallow_gradient_boosting"),
    ]
    assert [
        (spec.name, spec.feature_scope, spec.model_family)
        for spec in CANDIDATE_SPECS
    ] == expected

    manifest = screen_manifest()
    assert manifest["grouped_fold_count"] == 5
    assert [candidate["name"] for candidate in manifest["candidates"]] == [
        name for name, _, _ in expected
    ]
    assert [candidate["feature_count"] for candidate in manifest["candidates"]] == [
        len(IDM_PRIMARY_FEATURES),
        len(IDM_DECISION_FEATURES),
        len(IDM_PRIMARY_FEATURES),
        len(IDM_DECISION_FEATURES),
        len(IDM_PRIMARY_FEATURES),
        len(IDM_DECISION_FEATURES),
    ]


def test_grouped_oof_candidate_is_finite_and_keeps_seed_groups_disjoint() -> None:
    prepared = _grouped_prepared()
    folds = grouped_splits(
        prepared.safety,
        prepared.groups,
        n_splits=5,
        random_state=17,
    )
    safety_probability, benefit_probability, details = grouped_oof_candidate(
        prepared,
        make_candidate_estimator("regularized_logistic", random_state=17),
        folds,
    )

    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(
        range(len(prepared.rows))
    )
    assert np.all(np.isfinite(safety_probability))
    assert np.all(np.isfinite(benefit_probability))
    assert np.all((0.0 <= safety_probability) & (safety_probability <= 1.0))
    assert np.all((0.0 <= benefit_probability) & (benefit_probability <= 1.0))
    for detail, (train, test) in zip(details, folds):
        train_groups = set(prepared.groups[train])
        test_groups = set(prepared.groups[test])
        assert train_groups.isdisjoint(test_groups)
        assert detail["test_groups"] == sorted(str(group) for group in test_groups)


def test_feature_diagnostics_detect_constant_and_rank_redundant_features() -> None:
    prepared = PreparedGateB(
        rows=tuple({"case_id": str(index)} for index in range(6)),
        matrix=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 4.0],
                [1.0, 3.0, 6.0],
                [1.0, 4.0, 8.0],
                [1.0, 5.0, 10.0],
            ]
        ),
        groups=np.asarray([f"seed-{index}" for index in range(6)]),
        safety=np.asarray([0, 0, 1, 1, 1, 0]),
        joint_benefit=np.asarray([0, 0, 0, 1, 1, 0]),
    )

    diagnostics = feature_diagnostics(
        prepared, ("constant", "signal", "signal_scaled_copy")
    )

    assert diagnostics["constant_feature_count"] == 1
    assert diagnostics["missing_or_nonfinite_value_count"] == 0
    assert diagnostics["features"][0]["unique_count"] == 1
    assert diagnostics["features"][0]["safety_direction_invariant_auc"] is None
    assert diagnostics["absolute_spearman_at_least_0_98"] == [
        {"left": "signal", "right": "signal_scaled_copy", "spearman": 1.0}
    ]


def test_run_model_screen_returns_all_candidates_and_finite_oof_payloads() -> None:
    rows = _screen_rows()

    report, predictions = run_model_screen(rows)

    expected_names = [spec.name for spec in CANDIDATE_SPECS]
    assert [candidate["name"] for candidate in report["candidate_reports"]] == expected_names
    assert [candidate["candidate_order"] for candidate in report["candidate_reports"]] == list(
        range(len(CANDIDATE_SPECS))
    )
    assert report["cohort"]["case_count"] == len(rows)
    assert report["cohort"]["seed_count"] == 10
    assert predictions["case_ids"] == [row["case_id"] for row in rows]
    assert list(predictions["candidates"]) == expected_names
    for payload in predictions["candidates"].values():
        safety_probability = np.asarray(payload["safety_probability_oof"])
        benefit_probability = np.asarray(payload["conditional_benefit_probability_oof"])
        assert safety_probability.shape == (len(rows),)
        assert benefit_probability.shape == (len(rows),)
        assert np.all(np.isfinite(safety_probability))
        assert np.all(np.isfinite(benefit_probability))


def test_observation_v3_contract_and_screen_are_separately_versioned() -> None:
    manifest = observation_v3_contract_manifest()
    assert manifest["contract_version"] == "idm_counterfactual_escalation_observation/3"
    assert manifest["feature_count"] == len(IDM_OBSERVATION_V3_FEATURES)
    assert len(IDM_OBSERVATION_V3_FEATURES) > len(IDM_DECISION_FEATURES)
    assert manifest["future_samples"] == 0
    assert manifest["new_sumo_outcomes"] is False

    report, predictions = run_observation_v3_screen(_screen_rows())
    expected = [spec.name for spec in OBSERVATION_V3_CANDIDATE_SPECS]
    assert report["screen_contract"]["repair_iteration"] == 1
    assert report["screen_contract"]["further_observation_repairs_authorized"] is False
    assert [row["name"] for row in report["candidate_reports"]] == expected
    assert list(predictions["candidates"]) == expected
