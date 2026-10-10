"""Tests for leave-one-controller-out window filtering (fake corpus, metadata only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wms.core.errors import TrainDataRefusal
from wms.core.io import run_key
from wms.data.dataset import TrajectoryDataset
from wms.data.windows import load_exclusion_keys, load_window_list


class _FakeCorpus:
    """Minimal IdeaACorpus stand-in for the pre-load split/filter path."""

    def __init__(self, items, *, global_packs=None, global_positions=None, k_max=3):
        self._items = list(items)
        self._order = list(range(len(items)))
        self._global_packs = (list(global_packs) if global_packs is not None
                              else [0] * len(items))
        self._global_positions = (list(global_positions) if global_positions is not None
                                  else list(range(len(items))))
        self.k_max = k_max
        self._v1 = [None] * len(items)
        self._runs = [(0, 1_000_000)]

    def __len__(self):
        return len(self._order)


def _install_fake(monkeypatch, corpus):
    import wms.data.corpus as pack_corpus
    import wms.data.dataset as dataset_module
    monkeypatch.setattr(pack_corpus, "IdeaACorpus", lambda *a, **k: corpus)

    class _AnyPositions(dict):
        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            return np.zeros((1, 1), dtype=np.int64)

    monkeypatch.setattr(dataset_module, "read_identity_rows",
                        lambda sidecar, *, corpus_k: _AnyPositions())


def _items(*specs):
    return [{"batch": b, "chunk": b, "world": w, "arm": a, "tick": 0}
            for b, w, a in specs]


def test_run_key_prefers_batch_and_falls_back_to_chunk():
    assert run_key({"batch": "chunk2", "chunk": "x", "world": "w1", "arm": "pi"}) == \
        "chunk2/w1__pi"
    assert run_key({"chunk": "chunk1_fix27", "world": "w9", "arm": "native"}) == \
        "chunk1_fix27/w9__native"


def test_load_exclusion_keys_accepts_all_three_shapes(tmp_path):
    a = tmp_path / "keys.json"
    a.write_text(json.dumps({"schema": "s", "keys": ["k1", "k2"]}), encoding="utf-8")
    b = tmp_path / "windows.json"
    b.write_text(json.dumps({"windows": [{"key": "k3"}, {"key": "k4"}]}), encoding="utf-8")
    c = tmp_path / "list.json"
    c.write_text(json.dumps(["k5", "k6"]), encoding="utf-8")
    assert load_exclusion_keys([a, b, c]) == {"k1", "k2", "k3", "k4", "k5", "k6"}


def test_load_exclusion_keys_refuses_unknown_shape(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_exclusion_keys([bad])


def test_load_window_list_requires_pack_and_key(tmp_path):
    good = tmp_path / "val.json"
    good.write_text(json.dumps({"windows": [
        {"pack": "pack_v3", "key": "chunk1/w1__base"}]}), encoding="utf-8")
    assert load_window_list(good) == {("pack_v3", "chunk1/w1__base")}

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"windows": [{"key": "chunk1/w1__base"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_window_list(bad)


def test_from_real_packs_excludes_matching_keys(tmp_path, monkeypatch):
    corpus = _FakeCorpus(_items(("chunk1", "w1", "pi"), ("chunk1", "w1", "base"),
                                ("chunk2", "w2", "base")))
    _install_fake(monkeypatch, corpus)
    ds = TrajectoryDataset.from_real_packs(
        [tmp_path / "pack_v3"], identity_dirs=[tmp_path / "ids"],
        exclude_keys={"chunk1/w1__pi"})
    receipt = ds.receipt
    assert len(ds) == 2
    assert receipt["n_records"] == 2
    assert receipt["n_corpus_windows"] == 3
    assert receipt["excluded_records"] == 1
    assert receipt["exclude_keys_requested"] == 1
    assert receipt["kept_by_pack"] == {"pack_v3": 2}
    assert receipt["total_by_pack"] == {"pack_v3": 3}


def test_from_real_packs_refuses_absent_exclusion_key(tmp_path, monkeypatch):
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base")))
    _install_fake(monkeypatch, corpus)
    with pytest.raises(TrainDataRefusal):
        TrajectoryDataset.from_real_packs(
            [tmp_path / "pack_v3"], identity_dirs=[tmp_path / "ids"],
            exclude_keys={"chunk9/nope__pi"})


def test_from_real_packs_window_list_selects_only_listed(tmp_path, monkeypatch):
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base"), ("chunk1", "w2", "base")))
    _install_fake(monkeypatch, corpus)
    ds = TrajectoryDataset.from_real_packs(
        [tmp_path / "pack_v3"], identity_dirs=[tmp_path / "ids"],
        select_windows={("pack_v3", "chunk1/w2__base")})
    assert len(ds) == 1
    assert ds.receipt["kept_by_pack"] == {"pack_v3": 1}
    assert ds.receipt["select_windows_requested"] == 1


def test_from_real_packs_refuses_absent_window_list_entry(tmp_path, monkeypatch):
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base")))
    _install_fake(monkeypatch, corpus)
    with pytest.raises(TrainDataRefusal):
        TrajectoryDataset.from_real_packs(
            [tmp_path / "pack_v3"], identity_dirs=[tmp_path / "ids"],
            select_windows={("pack_v3", "chunk1/w9__base")})


def test_from_real_packs_scopes_window_list_by_pack_name(tmp_path, monkeypatch):
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base"), ("chunk1", "w1", "base")),
                         global_packs=[0, 1])
    _install_fake(monkeypatch, corpus)
    ds = TrajectoryDataset.from_real_packs(
        [tmp_path / "pack_v3", tmp_path / "pack_dspeed"],
        identity_dirs=[tmp_path / "ids_a", tmp_path / "ids_b"],
        select_windows={("pack_dspeed", "chunk1/w1__base")})
    assert len(ds) == 1
    assert ds.receipt["kept_by_pack"] == {"pack_dspeed": 1}
    assert ds.receipt["total_by_pack"] == {"pack_dspeed": 1, "pack_v3": 1}


def test_from_real_packs_refuses_empty_selection(tmp_path, monkeypatch):
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base")))
    _install_fake(monkeypatch, corpus)
    with pytest.raises(TrainDataRefusal):
        TrajectoryDataset.from_real_packs(
            [tmp_path / "pack_v3"], identity_dirs=[tmp_path / "ids"],
            exclude_keys={"chunk1/w1__base"})


def test_from_real_packs_refuses_window_both_selected_and_excluded(tmp_path, monkeypatch):
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base")))
    _install_fake(monkeypatch, corpus)
    with pytest.raises(TrainDataRefusal):
        TrajectoryDataset.from_real_packs(
            [tmp_path / "pack_v3"], identity_dirs=[tmp_path / "ids"],
            exclude_keys={"chunk1/w1__base"},
            select_windows={("pack_v3", "chunk1/w1__base")})


def _write_window_list(path, windows):
    path.write_text(json.dumps({
        "schema": "wm_s_lpo_windows/1", "list": "test", "n_windows": len(windows),
        "windows": windows}), encoding="utf-8")
    return path


def test_from_real_packs_resolves_staged_pack_basename(tmp_path, monkeypatch):
    """pack_v3 windows are kept even when the pack directory is staged under a
    different basename (the box failure: PACK_V3=.../work/pack)."""
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base"), ("chunk2", "w2", "base")),
                         global_packs=[0, 1])
    _install_fake(monkeypatch, corpus)
    wl = _write_window_list(tmp_path / "wl.json", [
        {"pack": "pack_v3", "batch": "chunk1", "world": "w1", "arm": "base",
         "t0_s": 0.0},
        {"pack": "pack_dspeed", "batch": "chunk2", "world": "w2", "arm": "base",
         "t0_s": 0.0},
    ])
    ds = TrajectoryDataset.from_real_packs(
        [tmp_path / "pack", tmp_path / "pack_dspeed"],
        identity_dirs=[tmp_path / "ids_a", tmp_path / "ids_b"],
        window_list=wl)
    assert ds.receipt["kept_by_pack"] == {"pack": 1, "pack_dspeed": 1}
    assert ds.receipt["window_filter"]["pack_name_resolution"] == {"pack_v3": "pack"}
    assert ds.receipt["window_filter"]["requested"] == 2
    assert ds.receipt["window_filter"]["matched"] == 2


def test_from_real_packs_refuses_unresolvable_window_pack_name(tmp_path, monkeypatch):
    """An unmatched list pack name with no corpus overlap fails closed instead
    of silently contributing zero windows."""
    corpus = _FakeCorpus(_items(("chunk1", "w1", "base")))
    _install_fake(monkeypatch, corpus)
    wl = _write_window_list(tmp_path / "wl.json", [
        {"pack": "pack_v3", "batch": "chunk9", "world": "w9", "arm": "base",
         "t0_s": 0.0},
    ])
    with pytest.raises(TrainDataRefusal):
        TrajectoryDataset.from_real_packs(
            [tmp_path / "pack"], identity_dirs=[tmp_path / "ids"],
            window_list=wl)


