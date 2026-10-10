from tools.screen_idm_observation_v2_value import (
    _feature_screen,
    _fold_audit,
    _standardized_difference,
)


def test_standardized_difference_direction_and_zero_variance() -> None:
    assert _standardized_difference([2.0, 3.0], [0.0, 1.0]) > 0.0
    assert _standardized_difference([1.0], [1.0]) == 0.0


def test_descriptive_screen_uses_grouped_leave_one_seed_out_without_fit() -> None:
    rows = [
        {"seed": seed, "features": {"x": float(seed + label)}, "label": bool(label)}
        for seed in range(4)
        for label in (0, 1)
    ]
    result = _feature_screen(rows, "label")
    assert result["top"][0]["feature"] == "x"
    assert result["top"][0]["leave_one_seed_out_count"] == 4
    assert "model" not in result


def test_fold_audit_never_splits_one_seed() -> None:
    rows = [
        {"seed": 1},
        {"seed": 1},
        {"seed": 2},
        {"seed": 3},
    ]
    audit = _fold_audit(rows)
    assert audit["passed"]
    assert not audit["seed_spans_multiple_folds"]
