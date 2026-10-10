"""Tests for the TRAIN-scoped validation holdout (WM-S2 held-out monitoring)."""

from __future__ import annotations

import json

from wms.data.val_split import (holdout_worlds, materialise_holdout,
                                split_document)


def _doc(n_worlds=20, arms=("native", "damping")):
    windows = []
    for i in range(n_worlds):
        for arm in arms:
            windows.append({"pack": "pack_hetdata_train", "batch": "hetdata",
                            "chunk": "hetdata", "world": f"w{8900 + i}m2700p05",
                            "arm": arm, "t0_s": 0.0, "held_out": False,
                            "key": f"hetdata/w{8900 + i}m2700p05__{arm}"})
    return {"schema": "wm_s_lpo_windows/1", "split": "TRAIN", "batch": "hetdata",
            "pack": "pack_hetdata_train", "n_windows": len(windows),
            "windows": windows}


def test_holdout_is_deterministic_and_about_ten_percent():
    doc = _doc(20)
    a = holdout_worlds(doc, frac=0.1, seed=0)
    b = holdout_worlds(doc, frac=0.1, seed=0)
    assert a == b and len(a) == 2
    assert set(a).issubset({w["world"] for w in doc["windows"]})
    assert holdout_worlds(doc, frac=0.1, seed=1) != a


def test_split_partitions_windows_and_keeps_pack():
    doc = _doc(20)
    train, val = split_document(doc, frac=0.1, seed=7)
    assert train["n_windows"] + val["n_windows"] == doc["n_windows"]
    assert {w["key"] for w in train["windows"]}.isdisjoint(
        {w["key"] for w in val["windows"]})
    assert train["pack"] == val["pack"] == "pack_hetdata_train"
    assert train["holdout_worlds"] == val["holdout_worlds"]
    assert all(w["world"] in val["holdout_worlds"] for w in val["windows"])
    assert all(w["world"] not in val["holdout_worlds"] for w in train["windows"])


def test_materialise_writes_both_lists(tmp_path):
    src = tmp_path / "windows_train_hetdata.json"
    src.write_text(json.dumps(_doc(20)), encoding="utf-8")
    train_path, val_path = materialise_holdout(src, tmp_path / "run",
                                               frac=0.1, seed=3)
    train = json.loads(open(train_path, encoding="utf-8").read())
    val = json.loads(open(val_path, encoding="utf-8").read())
    assert train["n_windows"] > 0 and val["n_windows"] > 0
    assert train["pack"] == "pack_hetdata_train"
    assert train["n_windows"] + val["n_windows"] == 40
