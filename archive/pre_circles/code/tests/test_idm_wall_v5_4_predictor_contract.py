from pathlib import Path

import pytest

from experiments.idm.wall_v5_4_predictor_contract import (
    DEVELOPMENT_SEEDS,
    FINAL_TEST_SEEDS,
    VALIDATION_SEEDS,
    fitting_gates,
    predictor_manifest,
    require_support_authorization,
)


def test_predictor_partitions_are_disjoint_and_final_seeds_remain_sealed() -> None:
    assert DEVELOPMENT_SEEDS == tuple(range(4401, 4521))
    assert VALIDATION_SEEDS == tuple(range(4601, 4661))
    assert FINAL_TEST_SEEDS == tuple(range(5001, 5061))
    assert not set(DEVELOPMENT_SEEDS) & set(VALIDATION_SEEDS)
    assert not set(DEVELOPMENT_SEEDS) & set(FINAL_TEST_SEEDS)
    assert not set(VALIDATION_SEEDS) & set(FINAL_TEST_SEEDS)


def test_predictor_contract_separates_deterministic_f_from_learned_b() -> None:
    manifest = predictor_manifest("development", DEVELOPMENT_SEEDS)
    assert manifest["gate_f"]["learned"] is False
    assert manifest["gate_b"]["training_population"] == "gate_f_admitted_and_dose_qualified"
    assert manifest["gate_b"]["label_contract"]["delay_is_standalone_benefit"] is False
    assert fitting_gates()["minimum_unsafe_independent_seeds"] == 20


def test_development_generation_fails_closed_without_passing_support(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        require_support_authorization(tmp_path)
