import numpy as np

from tools.analyze_idm_rsu_observation_diagnostic import (
    MODELS,
    _candidate_authority,
    _evaluate_nested,
)


def test_authority_features_are_predecision_and_projection_based():
    observation = np.zeros(134, dtype=np.float32)
    observation[1] = 28.0 / 35.0
    candidate = {
        "runway_m": 600.0,
        "predicted_dose_s": 22.0,
        "prospective_rear_headroom_m": 15.0,
        "preview_reference_speed_mps": 20.0,
        "terminal_speed_previews": {
            "p30": {
                "reference_speed_mps": 20.0,
                "desired_acceleration_mps2": -1.2,
                "projected_acceleration_mps2": -0.5,
            }
        },
    }
    values = _candidate_authority(candidate, observation)
    assert np.allclose(values, [600.0, 22.0, 15.0, 8.0, 0.7])


def test_nested_grouped_models_select_thresholds_without_test_fold_access():
    seeds = np.repeat(np.arange(1, 7), 2)
    labels = np.tile(np.asarray([0, 1], dtype=int), 6)
    recurrences = np.tile(np.asarray([1, 2], dtype=int), 6)
    # A deliberately simple signal with small seed-specific nuisance terms.
    features = np.column_stack(
        [labels.astype(float) * 2.0 - 1.0, seeds.astype(float) / 100.0]
    )
    for model_name in MODELS:
        result = _evaluate_nested(
            features,
            labels,
            seeds,
            recurrences,
            model_name,
            bootstrap_replicates=200,
        )
        assert len(result["scores"]) == len(labels)
        assert len(result["folds"]) == 6
        assert all(
            0.0 <= row["inner_selected_threshold"] <= 1.0
            for row in result["folds"]
        )
        assert result["auroc"] >= 0.9
        assert result["balanced_accuracy"] >= 0.8
