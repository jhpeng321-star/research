from __future__ import annotations

import numpy as np
import pytest

from counterfactual.idm_predictor_contract import IDM_PRIMARY_FEATURES
from counterfactual.idm_shadow_observation_v2_contract import (
    CAPTURE_SCHEMA,
    SEED_SCHEMA,
    contract_manifest,
)
from tools.analyze_idm_v5_4_shadow_observation_v2_development import (
    CAPTURE_PHASE,
    CASE_SCHEMA,
    COMBINED_BLOCK,
    DEPLOYABLE_BASELINE,
    EXACT_BASELINE,
    PHYSICAL_BLOCK,
    SHADOW_BLOCK,
    _raw_tensor_payload,
    _sequence_matrix,
    _torch_tcn_fit_predict,
    _policy_metrics,
    development_outer_folds,
    join_capture_labels,
    prepare_population,
    temporal_convolution_screen,
    validate_capture_payloads,
)


DECISIONS = (620, 660, 680, 700)


def _dose(success: bool) -> dict:
    return {
        "activation_feasible": True,
        "realized_duration_fraction": 1.0 if success else 0.5,
        "lane_coverage_fraction": 1.0,
        "assembly_completed": success,
        "holding_fraction_of_override": 1.0 if success else 0.0,
        "wall_geometry": {
            "geometry_mode": "compact_moving",
            "fraction_within_configured_span": 1.0,
        },
    }


def _capture_case(seed: int, step: int, case_index: int) -> dict:
    case_id = f"synthetic-seed{seed}-step{step}"
    physical = {
        "sectional_transitional_instability": {
            "wave_speed": float(case_index),
            "wave_growth": float((seed + case_index) % 5),
        },
        "merge_pressure_breakdown_state": {
            "excess_demand": float(case_index - 1),
            "storage": float(seed % 7),
        },
        "trajectory_physical_20s": {
            "compression": float(case_index == 1),
            "headway_cv": float((seed * 3 + case_index) % 11),
        },
    }
    return {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "seed": seed,
        "decision_step": step,
        "gate_f_admitted": True,
        "capture_phase": CAPTURE_PHASE,
        "predecision_record": {},
        "baseline_observation": {
            "schema": "idm_shadow_predecision_record_v2/1",
            "feature_blocks": physical,
            "model_features": {
                "wave_speed": float(case_index),
                "headway_cv": float((seed + case_index) % 11),
            },
            "integrity": {},
        },
        "shadow_observation": {
            "schema": "idm_shadow_twin_features_v2/1",
            "feature_blocks": {
                "proposed_wall_shadow_10s": {
                    "authority_margin": float(case_index - 0.5),
                    "projected_rear_ttc": float(10 - case_index),
                }
            },
            "model_features": {
                "authority_margin": float(case_index - 0.5),
                "projected_rear_ttc": float(10 - case_index),
            },
            "integrity": {"prefix_identity_passed": True},
        },
        "integrity": {
            "feasibility_source_treatment_blind": True,
            "prefix_identity_passed": True,
            "shadow_generated": True,
        },
    }


def _label_case(seed: int, step: int, case_index: int) -> dict:
    # Every seed contains dose failure, unsafe, safe nonbenefit and safe benefit.
    dose = case_index != 0
    safe = case_index >= 2
    benefit = case_index == 3
    features = {
        name: float((seed % 13) * 0.1 + case_index)
        for name in IDM_PRIMARY_FEATURES
    }
    features["health"] = float(benefit)
    features["health_change_30"] = float(safe)
    return {
        "case_id": f"synthetic-seed{seed}-step{step}",
        "seed": seed,
        "decision_step": step,
        "features": features,
        "predecision_supervisor": {"admitted": True},
        "treatment_dose": _dose(dose),
        "safety_service_oracle": {
            "safety_admissible": safe,
            "wall_selected": benefit,
            "effects": {
                "hard_brake_onset_excess": float(1 if case_index == 1 else 0),
                "hard_brake_excess_vehicle_steps": float(2 if case_index == 1 else 0),
                "low_ttc_excess_steps": float(1 if case_index == 1 else 0),
                "merge_discharge_benefit": float(case_index - 2),
                "delay_benefit_s": float(20 * (case_index - 2)),
                "collision_excess_vehicle_count": 0,
                "teleport_excess_vehicle_count": 0,
            },
        },
    }


def _population(seed_count: int = 20):
    captures = []
    labels = []
    for seed in range(100, 100 + seed_count):
        for case_index, step in enumerate(DECISIONS):
            captures.append(_capture_case(seed, step, case_index))
            labels.append(_label_case(seed, step, case_index))
    return prepare_population(join_capture_labels(captures, labels))


def test_capture_validation_is_complete_and_label_free() -> None:
    seeds = (101, 102)
    contract_sha = contract_manifest()["contract_sha256"]
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "contract_sha256": contract_sha,
        "seeds": list(seeds),
        "decision_steps": list(DECISIONS),
    }
    payloads = []
    for seed in seeds:
        payloads.append(
            {
                "schema": SEED_SCHEMA,
                "contract_sha256": contract_sha,
                "seed": seed,
                "decision_steps": list(DECISIONS),
                "cases": [
                    _capture_case(seed, step, case_index)
                    for case_index, step in enumerate(DECISIONS)
                ],
            }
        )

    rows = validate_capture_payloads(
        manifest,
        payloads,
        expected_seeds=seeds,
        expected_decisions=DECISIONS,
    )

    assert len(rows) == 8
    assert all("safety_service_oracle" not in row for row in rows)


def test_capture_validation_rejects_future_or_outcome_feature_names() -> None:
    row = _capture_case(101, 620, 0)
    row["shadow_observation"]["model_features"]["actual_outcome"] = 1.0
    contract_sha = contract_manifest()["contract_sha256"]
    manifest = {
        "schema": CAPTURE_SCHEMA,
        "contract_sha256": contract_sha,
        "seeds": [101],
        "decision_steps": [620],
    }
    payload = {
        "schema": SEED_SCHEMA,
        "contract_sha256": contract_sha,
        "seed": 101,
        "decision_steps": [620],
        "cases": [row],
    }
    with pytest.raises(ValueError, match="prohibited input token"):
        validate_capture_payloads(
            manifest,
            [payload],
            expected_seeds=(101,),
            expected_decisions=(620,),
        )


def test_exact_case_join_refuses_missing_or_metadata_mismatch() -> None:
    capture = _capture_case(101, 620, 0)
    label = _label_case(101, 620, 0)
    assert join_capture_labels([capture], [label])[0]["case_id"] == capture["case_id"]

    wrong = dict(label, seed=999)
    with pytest.raises(ValueError, match="metadata mismatch"):
        join_capture_labels([capture], [wrong])
    with pytest.raises(ValueError, match="do not match exactly"):
        join_capture_labels([capture], [])


def test_v2_blocks_exclude_health_while_exact_benchmark_retains_it() -> None:
    population = _population()
    assert any(name.endswith("::health") for name in population.blocks[EXACT_BASELINE].names)
    assert not any("health" in name for name in population.blocks[DEPLOYABLE_BASELINE].names)


def test_exact_v1_context_features_are_derived_from_case_metadata() -> None:
    capture = _capture_case(101, 620, 0)
    label = _label_case(101, 620, 0)
    label["features"].pop("context_decision_step", None)
    label["features"].pop("context_cav_penetration", None)
    label["cav_penetration"] = 0.30
    population = prepare_population(join_capture_labels([capture], [label]))
    block = population.blocks[EXACT_BASELINE]
    values = dict(zip(block.names, block.values[0]))
    assert values[f"{EXACT_BASELINE}::context_decision_step"] == 620.0
    for name in (PHYSICAL_BLOCK, SHADOW_BLOCK, COMBINED_BLOCK):
        assert not any("health" in feature.lower() for feature in population.blocks[name].names)
    assert population.blocks[COMBINED_BLOCK].values.shape[1] == (
        population.blocks[PHYSICAL_BLOCK].values.shape[1]
        + population.blocks[SHADOW_BLOCK].values.shape[1]
    )


def test_outer_folds_are_complete_and_never_split_a_seed() -> None:
    population = _population()
    tests = []
    for train, test in development_outer_folds(population):
        assert set(population.groups[train]).isdisjoint(set(population.groups[test]))
        tests.extend(test.tolist())
    assert sorted(tests) == list(range(len(population.rows)))


def test_policy_inventory_keeps_dose_failures_unsafe_and_harmful_cases() -> None:
    population = _population(seed_count=10)
    command = np.ones(len(population.rows), dtype=bool)
    probabilities = {
        "dose": np.full(len(command), 0.8),
        "safety": np.full(len(command), 0.8),
        "benefit": np.full(len(command), 0.8),
    }
    report = _policy_metrics(population, command, probabilities)

    assert report["selected_failed_dose_case_count"] == 10
    assert report["selected_unsafe_case_count"] == 20
    assert report["selected_nonbeneficial_case_count"] == 30
    assert report["selected_physically_harmful_case_count"] == 10
    assert len(report["inventories"]["failed_dose"]) == 10
    assert len(report["inventories"]["unsafe"]) == 20


def test_temporal_convolution_is_reported_unavailable_without_raw_tensor() -> None:
    report = temporal_convolution_screen(_population(seed_count=10))
    assert report["status"].startswith("unavailable_raw_tensor_not_captured:")
    assert report["fabricated_sequence_features"] is False


def test_small_raw_tensor_runs_the_real_causal_convolution() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(20260831)
    payloads = []
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 1], dtype=int)
    for index, label in enumerate(labels):
        # Variable anonymous-slot counts exercise the permutation-invariant
        # pooling used before the causal convolution.  A weak temporal signal
        # makes the fit nondegenerate without asserting scientific efficacy.
        values = rng.normal(0.0, 0.05, size=(6, 3 + index % 2, 2))
        values[3:, :, 0] += float(label)
        mask = np.ones(values.shape[:2], dtype=float)
        payloads.append((values, mask))

    sequence, audit = _sequence_matrix(payloads)
    train_probability, held_probability = _torch_tcn_fit_predict(
        sequence,
        labels,
        np.arange(6),
        np.arange(6, 8),
        channels=4,
        layers=1,
        weight_decay=0.01,
        random_state=20260831,
    )

    assert audit["slot_pooling"].startswith("permutation_invariant")
    assert sequence.shape == (8, 6, 7)
    assert train_probability.shape == (6,)
    assert held_probability.shape == (2,)
    assert np.all((train_probability >= 0.0) & (train_probability <= 1.0))
    assert np.all((held_probability >= 0.0) & (held_probability <= 1.0))


def test_raw_tensor_preserves_anonymous_exit_event_with_event_mask() -> None:
    row = {
        "capture": {
            "baseline_observation": {
                "history_20s": {
                    "tensor_values": [
                        [[1.0, 0.0]],
                        [[0.0, 1.0]],
                    ],
                    "tensor_mask": [[1.0], [0.0]],
                    "event_mask": [[0.0], [1.0]],
                    "channel_names": ["present", "exit_event"],
                }
            }
        }
    }
    values, mask = _raw_tensor_payload(row, "history_20s")
    assert values[1, 0, 1] == 1.0
    assert mask[1, 0, 0] == 0.0
    assert mask[1, 0, 1] == 1.0
