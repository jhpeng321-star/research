"""ROUND-3 fix #2: jam-window detection and oversampling (flags default OFF)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from wms.data import jam as jamdata
from wms.data.mixdata import MixDataRefusal, MixedTrajectoryDataset, MixSource
from wms.data.dataset import TrajectoryDataset


def _arrays(n, *, min_speeds):
    """``n`` windows ``[T=6, K=3, 7]``; one HDV column carries the min speed."""
    states = np.zeros((n, 6, 3, 7), np.float32)
    states[..., 1] = 1.0
    present = np.ones((n, 6, 3), bool)
    is_cav = np.zeros((n, 6, 3), bool)
    is_cav[..., 0] = True
    for i, speed in enumerate(min_speeds):
        states[i, :, 1:, 4] = 20.0
        states[i, 2, 1, 4] = float(speed)
    return {"states": states, "present": present, "is_cav": is_cav}


class _Src:
    def __init__(self, records):
        self.records = records
        self.receipt = {"mode": "fake", "n_records": len(records)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return dict(self.records[i])


def _records(min_speeds):
    arrays = _arrays(len(min_speeds), min_speeds=min_speeds)
    ds = TrajectoryDataset.from_arrays(arrays)
    return [ds[i] for i in range(len(min_speeds))], arrays


def test_record_hdv_min_speed_excludes_cav_and_missing():
    rec = {"states": np.zeros((2, 3, 7), np.float32),
           "present": np.ones((2, 3), bool),
           "is_cav": np.asarray([[True, False, False], [True, False, False]])}
    rec["states"][:, 1, 4] = 4.0
    rec["states"][:, 0, 4] = 0.1          # CAV: must be ignored
    rec["states"][:, 2, 4] = np.nan
    minimum, n = jamdata.record_hdv_min_speed(rec)
    assert n == 2 and minimum == pytest.approx(4.0)
    empty = {"states": np.zeros((2, 1, 7), np.float32),
             "present": np.ones((2, 1), bool),
             "is_cav": np.ones((2, 1), bool)}
    assert jamdata.record_hdv_min_speed(empty) == (float("inf"), 0)


def test_compute_jam_flags_counts_and_cache_roundtrip(tmp_path):
    records, _ = _records([6.0, 4.9, 20.0, 4.99])
    ds = _Src(records)
    cache = tmp_path / "jam.json"
    first = jamdata.compute_jam_flags(ds, speed_mps=5.0, cache_path=cache,
                                      source_key="k")
    assert first["flags"].tolist() == [False, True, False, True]
    assert first["n_jam"] == 2 and first["n_hdv_windows"] == 4
    assert first["min_speed_mps"] == pytest.approx(4.9)
    assert first["cache_hit"] is False and cache.is_file()
    document = json.loads(cache.read_text())
    assert document["schema"] == jamdata.JAM_SCHEMA and document["n_jam"] == 2
    second = jamdata.compute_jam_flags(ds, speed_mps=5.0, cache_path=cache,
                                       source_key="k")
    assert second["cache_hit"] is True
    assert second["flags"].tolist() == first["flags"].tolist()
    # a different key/threshold invalidates the cache
    third = jamdata.compute_jam_flags(ds, speed_mps=6.0, cache_path=cache,
                                     source_key="k")
    assert third["cache_hit"] is False and third["n_jam"] == 2


def test_compute_jam_flags_rejects_bad_speed(tmp_path):
    ds = _Src(_records([6.0])[0])
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(jamdata.JamRefusal):
            jamdata.compute_jam_flags(ds, speed_mps=bad)


def test_mixsource_jam_shape_is_validated():
    with pytest.raises(MixDataRefusal):
        MixSource("original", _Src(_records([6.0])[0]), share=1.0,
                  jam=np.zeros(3, bool))


def test_jam_off_weights_are_bit_identical():
    over = 3.0
    events = np.asarray([[2, 99], [99, 99], [99, 20]], np.int64)
    jam = np.asarray([True, False, True])
    old = MixedTrajectoryDataset(
        [MixSource("original", _Src(_records([6.0, 20.0])[0]), share=0.5),
         MixSource("switching", _Src(_records([6.0, 20.0, 20.0])[0]), share=0.5,
                   events=events)],
        event_oversample=over)
    # jam attached but oversample 1 == the OFF path (no reweighting at all)
    same = MixedTrajectoryDataset(
        [MixSource("original", _Src(_records([6.0, 20.0])[0]), share=0.5,
                   jam=np.asarray([False, True])),
         MixSource("switching", _Src(_records([6.0, 20.0, 20.0])[0]), share=0.5,
                   events=events, jam=jam)],
        event_oversample=over, jam_oversample=1.0)
    for stage in (10.0, 30.0, 120.0):
        assert old.stage_weights(stage) == same.stage_weights(stage)
    assert "jam_oversample" not in same.receipt


def test_jam_oversampling_composes_with_event_and_share():
    over, jam_k, ratio = 3.0, 4.0, 0.25
    events = np.asarray([[2, 99], [99, 99]], np.int64)      # item 0 is an event
    jam = np.asarray([True, False])                          # item 0 is also jam
    ds = MixedTrajectoryDataset(
        [MixSource("original", _Src(_records([6.0, 20.0])[0]), share=1.0 - ratio),
         MixSource("switching", _Src(_records([6.0, 20.0])[0]), share=ratio,
                   events=events, jam=jam)],
        event_oversample=over, jam_oversample=jam_k)
    w = np.asarray(ds.stage_weights(10.0))
    # switching share is preserved; item 0 (event AND jam) == over * jam_k
    assert w[2:].sum() == pytest.approx(ratio)
    assert w[2] == pytest.approx(ratio * (over * jam_k) / (over * jam_k + 1.0))
    assert w[3] == pytest.approx(ratio * 1.0 / (over * jam_k + 1.0))
    # original share is uniform (no jam there)
    assert np.allclose(w[:2], (1.0 - ratio) / 2.0)
    rows = ds.describe_stages((10.0,))
    assert rows[0]["sources"]["switching"]["n_jam_items"] == 1
    assert "n_jam_items" not in rows[0]["sources"]["original"]


def test_jam_oversampled_dataset_wrapper(tmp_path):
    records, _ = _records([6.0, 20.0, 4.0])
    base = TrajectoryDataset.from_arrays(_arrays(3, min_speeds=[6.0, 20.0, 4.0]))
    report = jamdata.compute_jam_flags(base, speed_mps=5.0)
    report["oversample"] = 2.0
    ds = jamdata.JamOversampledDataset(base, report["flags"], oversample=2.0,
                                       report=report)
    assert len(ds) == 3 and ds[0]["states"].shape[0] == 6
    assert ds.receipt["jam"]["n_jam"] == 1
    w = np.asarray(ds.stage_weights(10.0))
    assert w.sum() == pytest.approx(1.0)
    assert w[2] / w[0] == pytest.approx(2.0)
    assert ds.describe_stages((10.0,))[0]["sources"]["original"]["n_jam_items"] == 1


def test_jam_dataset_source_key_is_deterministic():
    base = TrajectoryDataset.from_arrays(_arrays(2, min_speeds=[6.0, 4.0]))
    assert jamdata.dataset_source_key(base) == jamdata.dataset_source_key(base)


