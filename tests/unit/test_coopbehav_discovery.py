"""Synthetic-only checks for deterministic physical-signature discovery."""

import copy
import json

import numpy as np
import pytest

from tools.coopbehav.discovery import discover


def _record(identifier, gain=1.0, explained_by=None, **features):
    return {
        "id": identifier, "C_G": gain, "features": features,
        "explained_by": [] if explained_by is None else explained_by,
    }


def _physical_groups():
    records = []
    for index, offset in enumerate((-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)):
        records.append(_record(
            "open-{}".format(index), gain=2 + index / 10,
            speed_change=4 + offset, gap_change=10 + offset,
        ))
        records.append(_record(
            "brake-{}".format(index), gain=1 + index / 10,
            explained_by=["braking_detector"] if index == 0 else [],
            speed_change=-4 + offset, gap_change=-10 + offset,
        ))
    return records


def test_two_physical_groups_are_ranked_described_and_flagged():
    records = _physical_groups()
    original = copy.deepcopy(records)
    result = discover(records)
    assert records == original
    assert result["status"] == "ok"
    assert result["method"]["selected_k"] == 2
    assert result["method"]["k_candidates"] == [2, 3, 4]
    groups = {frozenset(cluster["member_ids"]): cluster for cluster in result["clusters"]}
    open_group = groups[frozenset("open-{}".format(i) for i in range(8))]
    brake_group = groups[frozenset("brake-{}".format(i) for i in range(8))]
    assert "Higher mean speed change" in open_group["description"]
    assert "Lower mean gap change" in brake_group["description"]
    assert open_group["physical_means"]["speed_change"] > 0
    assert brake_group["standardized_means"]["speed_change"] < 0
    assert open_group["unexplained"] is True
    # Even one explained member prevents the whole cluster being flagged.
    assert brake_group["unexplained"] is False
    assert brake_group["explained_by"] == ["braking_detector"]
    assert [item["cluster_id"] for item in result["unanticipated_candidates"]] == [open_group["cluster_id"]]
    assert "not causal proof" in result["unanticipated_candidates"][0]["interpretation"]
    assert [row["C_G"] for row in result["assignments"]] == sorted(
        [row["C_G"] for row in records], reverse=True,
    )
    assert [row["rank"] for row in result["assignments"]] == list(range(1, 17))
    json.dumps(result, allow_nan=False)


def test_order_independence_ties_and_repeatability():
    records = _physical_groups()
    for row in records:
        row["C_G"] = 1.0
    expected = discover(records)
    assert discover(records) == expected
    assert discover(reversed(records)) == expected
    permutation = np.random.default_rng(72).permutation(len(records))
    assert discover([records[i] for i in permutation]) == expected


def test_only_strictly_positive_finite_measured_gains_are_ranked():
    records = [_record("positive", gain=0.01, speed=2)]
    records += [_record(str(index), gain=gain, speed=100) for index, gain in enumerate(
        (None, 0, -1, float("nan"), float("inf"), -float("inf"), "3", True)
    )]
    records.append({"id": "descriptive_arm", "features": {"speed": 900}})
    result = discover(records)
    assert [row["id"] for row in result["assignments"]] == ["positive"]
    assert result["features"][0]["mean"] == 2
    assert result["counts"] == {"input": 10, "ranked": 1, "clustered": 1}


@pytest.mark.parametrize("records", [[], [
    {"id": "descriptive", "features": {"speed": 5}},
    _record("invalid", gain=float("nan"), speed=1),
    _record("nonpositive", gain=0, speed=2),
]])
def test_no_gain_cases(records):
    result = discover(records)
    assert result["status"] == "no_positive_c_g"
    assert result["assignments"] == result["clusters"] == result["unanticipated_candidates"] == []
    assert result["method"]["selected_k"] == 0
    json.dumps(result, allow_nan=False)


def test_missing_and_nonfinite_features_are_imputed_without_inventing_records():
    records = [
        _record("a", gain=5, speed=1, gap=float("inf"), dead="invalid"),
        _record("b", gain=4, speed=3, gap=2),
        _record("c", gain=3, speed=float("nan"), gap=4),
        _record("empty", gain=9),
        _record("invalid", gain=8, speed=float("inf"), gap=None, dead=True),
    ]
    result = discover(records)
    features = {item["name"]: item for item in result["features"]}
    assert features["speed"]["mean"] == 2
    assert features["speed"]["scale"] == 1
    assert features["speed"]["imputed_count"] == 1
    assert features["gap"]["mean"] == 3
    assert features["dead"]["mean"] is None
    assert features["dead"]["used_for_clustering"] is False
    assert result["counts"]["ranked"] == 5
    assert result["counts"]["clustered"] == 3
    for assignment in result["assignments"][:2]:
        assert assignment["cluster_id"] is None
        assert assignment["reason"] == "no_finite_physical_features"
    assert result["clusters"][0]["physical_means"] == {"gap": 3.0, "speed": 2.0}
    json.dumps(result, allow_nan=False)


def test_no_usable_features_keeps_complete_gain_ranking():
    result = discover([_record("b", gain=1), _record("a", gain=2, x=float("nan"))])
    assert result["status"] == "no_usable_features"
    assert [row["id"] for row in result["assignments"]] == ["a", "b"]
    assert all(row["cluster_id"] is None for row in result["assignments"])
    assert result["clusters"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("count", [1, 2, 3, 9])
def test_constant_signatures_and_small_samples_use_one_cluster(count):
    result = discover([_record(str(i), speed=3, gap=0) for i in range(count)])
    assert result["status"] == "degenerate"
    assert result["method"]["selected_k"] == 1
    assert result["clusters"][0]["size"] == count
    assert all(not feature["used_for_clustering"] for feature in result["features"])
    assert "Baseline mean speed" in result["clusters"][0]["description"]


def test_feature_units_and_constant_column_do_not_change_partition():
    records = _physical_groups()
    expected = discover(records)
    for row in records:
        row["features"]["gap_change"] *= 1000
        row["features"]["constant"] = 9
    result = discover(records)
    assert result["assignments"] == expected["assignments"]
    assert result["method"]["selected_k"] == 2
    assert result["features"][0]["used_for_clustering"] is False


def test_finite_extreme_values_do_not_overflow_scaling_or_json():
    records = [_record(str(i), gain=1e308, physical=(1 if i < 4 else -1) * 1e308)
               for i in range(8)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = discover(records)
    assert result["method"]["selected_k"] == 2
    assert result["features"][0]["scale"] == 1e308
    json.dumps(result, allow_nan=False)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        discover([_record("same", speed=1), _record("same", speed=2)])
