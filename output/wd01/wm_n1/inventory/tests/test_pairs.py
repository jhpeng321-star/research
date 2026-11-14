"""Pair grouping by world + t0 (sibling arms vs the genuine native)."""
import pandas as pd

import inventory_lib as inv
from fixtures import ALL_SPECS, TRAIN_WORLD, T0, build_fixture_root


def _build(root):
    guard = inv.make_guard(root)
    rows, _holdout, _planned = inv.build_rows(root, guard, specs=ALL_SPECS)
    return inv.build_pairs(pd.DataFrame(rows))


def test_pairs_group_by_world_and_t0(tmp_path):
    root = build_fixture_root(tmp_path)
    pairs = _build(root)
    assert pairs
    for p in pairs:
        assert p["world"] == TRAIN_WORLD
        assert p["t0_s"] == T0
        assert p["pair_key"] == f"{TRAIN_WORLD}@{T0}"


def test_pair_categories_and_native_reference(tmp_path):
    root = build_fixture_root(tmp_path)
    pairs = _build(root)
    categories = {p["category"] for p in pairs}
    assert "damping-native" in categories
    assert "pi-native" in categories
    assert "switching-native" in categories
    assert "fixed-native" in categories
    assert "wall-native" in categories
    for p in pairs:
        assert p["native_run"] == f"{TRAIN_WORLD}__native"
        assert p["native_batch"] == "truearms_chunk2"


def test_same_base_state_flag(tmp_path):
    root = build_fixture_root(tmp_path)
    pairs = _build(root)
    pi_pair = [p for p in pairs if p["category"] == "pi-native" and p["other_batch"] == "truearms_chunk2"][0]
    assert pi_pair["same_base_state"] is True
    damping_pair = [p for p in pairs if p["category"] == "damping-native"][0]
    assert damping_pair["same_base_state"] is None
