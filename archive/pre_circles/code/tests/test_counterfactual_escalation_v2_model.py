from __future__ import annotations

import numpy as np
import pytest

from counterfactual.escalation_v2_model import (
    complete_group_id,
    feature_matrix,
    grouped_splits,
)


def _row(seed: int = 1) -> dict:
    return {
        "case_id": f"case-{seed}",
        "seed": seed,
        "decision_step": 180,
        "scenario": {
            "demand": "hyper",
            "cav_penetration": 0.2,
            "hdv_model": "Krauss",
        },
        "features": {"health": 0.4, "decision_queue_count": 7},
    }


def test_complete_group_id_contains_full_scenario_and_seed() -> None:
    assert complete_group_id(_row(17)) == "hyper|p0.200000|Krauss|seed17"


def test_feature_matrix_uses_only_explicit_decision_time_paths() -> None:
    row = _row()
    matrix = feature_matrix(
        [row],
        (
            "health",
            "decision_queue_count",
            "context_decision_step",
            "context_is_hyper",
            "context_cav_penetration",
        ),
    )
    assert matrix.tolist() == [[0.4, 7.0, 180.0, 1.0, 0.2]]
    with pytest.raises(ValueError, match="missing feature"):
        feature_matrix([row], ("incremental_delay_s",))


def test_grouped_splits_never_split_one_scenario_seed() -> None:
    labels = np.asarray([0, 1] * 10)
    groups = np.asarray([f"group-{index // 2}" for index in range(20)])
    for train, test in grouped_splits(labels, groups, n_splits=5):
        assert set(groups[train]).isdisjoint(set(groups[test]))
