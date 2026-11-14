"""OOF tree predictions: world-grouped folds, leakage, full-TRAIN path."""
from __future__ import annotations

import numpy as np
import pytest

from combine.guards import SplitLeakError
from combine.oof_tree import (FULL_FOLD, RowTableDataset,
                              build_oof_predictions, fit_guarded,
                              load_oof, partition_worlds)
from combine.synthetic import SyntheticQuantileModel, synthetic_rows

TAUS = (60.0, 120.0, 300.0)


def test_partition_worlds_is_a_partition():
    worlds = [f"w{i}" for i in range(7)]
    assign, k = partition_worlds(worlds, 5)
    assert k == 5
    folds = {}
    for w, f in assign.items():
        folds.setdefault(f, []).append(w)
    assert sorted(sum(folds.values(), [])) == sorted(worlds)
    assert all(0 <= f < k for f in folds)


def test_oof_no_world_leakage_and_partition():
    rows = synthetic_rows(n_train=8, n_cal=4, seed=1)
    created = []

    def factory():
        m = SyntheticQuantileModel()
        created.append(m)
        return m

    res = build_oof_predictions(rows, factory, k=5, targets=["delay"])
    assert res.k == 5
    train_worlds = sorted(set(rows.loc[rows.split == "TRAIN", "world"]))

    # (ii) folds partition the TRAIN worlds
    all_test = sorted(sum((list(v) for v in res.fold_worlds.values()), []))
    assert all_test == train_worlds
    seen = set()
    for f, w in res.fold_worlds.items():
        assert seen.isdisjoint(w)
        seen |= set(w)
        assert sorted(res.fit_worlds[f]) == sorted(set(train_worlds) - set(w))

    # the fold model created at call f is the one that fitted fit_worlds[f]
    for f in res.fit_worlds:
        assert sorted(created[f].fitted_worlds) == sorted(res.fit_worlds[f])

    # (i) a TRAIN row's predictor never saw any row of its own world
    oof = res.table[res.table.model_kind == "oof"]
    assert len(oof) == int((rows.split == "TRAIN").sum())
    for _, r in oof.iterrows():
        f = int(r.fold_id)
        assert 0 <= f < res.k
        assert r.world not in created[f].fitted_worlds


def test_non_train_rows_use_only_full_train_model():
    rows = synthetic_rows(n_train=8, n_cal=4, seed=2)
    created = []

    def factory():
        m = SyntheticQuantileModel()
        created.append(m)
        return m

    res = build_oof_predictions(rows, factory, k=4, targets=["delay"])
    train_worlds = sorted(set(rows.loc[rows.split == "TRAIN", "world"]))
    cal_worlds = sorted(set(rows.loc[rows.split == "CALIBRATION", "world"]))

    full_model = created[-1]
    assert sorted(full_model.fitted_worlds) == train_worlds

    full = res.table[res.table.model_kind == "full"]
    assert set(full["world"]) == set(cal_worlds)
    assert (full["fold_id"] == FULL_FOLD).all()

    # the CAL predictions equal the full model's own predictions
    cal_rows = rows[rows.split == "CALIBRATION"].reset_index(drop=True)
    ds_cal = RowTableDataset(cal_rows)
    idx = np.arange(len(cal_rows))
    got = full_model.predict_quantiles_rows(ds_cal, idx, "delay")
    for _, r in full.iterrows():
        pos = cal_rows.index[cal_rows.row_id == r.row_id][0]
        assert np.isclose(r["q0.5"], got[0.5][pos])


def test_fit_guard_raises_on_cal_and_test():
    rows = synthetic_rows(n_train=4, n_cal=2, n_test=2, seed=3)
    for split in ("CALIBRATION", "TEST"):
        sub = rows[rows.split == split].reset_index(drop=True)
        with pytest.raises(SplitLeakError):
            fit_guarded(lambda: SyntheticQuantileModel(),
                        RowTableDataset(sub), f"fit-{split}")


def test_build_raises_when_builder_leaks_non_train():
    rows = synthetic_rows(n_train=4, n_cal=2, seed=4)

    def leaky_builder(_worlds):
        return RowTableDataset(rows)          # ignores the world filter

    with pytest.raises(SplitLeakError):
        build_oof_predictions(rows, lambda: SyntheticQuantileModel(),
                              leaky_builder, k=2, targets=["delay"])


def test_save_load_roundtrip(tmp_path):
    rows = synthetic_rows(n_train=6, n_cal=2, seed=5)
    res = build_oof_predictions(rows, lambda: SyntheticQuantileModel(),
                                k=3, targets=["delay"])
    paths = res.save(tmp_path / "oof")
    assert paths["parquet"].endswith(".parquet")
    assert paths["npz"].endswith(".npz")
    back = load_oof(tmp_path / "oof")
    assert back.k == res.k
    assert list(back.table.columns) == list(res.table.columns)
    assert np.allclose(back.table["q0.5"], res.table["q0.5"], equal_nan=True)
    assert back.full_fit_worlds == res.full_fit_worlds
