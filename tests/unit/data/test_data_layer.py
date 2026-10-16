"""Synthetic unit tests of the clean data layer (always run; no real data)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.core.errors import TrainDataRefusal
from wms.data import windows as W
from wms.data.collate import collate
from wms.data.dataset import TrajectoryDataset, timeline_controller_ids
from wms.data.history import history_spec, long_history_features
from wms.data.sources import validate_train_source

ROOT = Path(__file__).resolve().parents[3]


def _arrays(n=2, t1=5, k=3):
    states = np.zeros((n, t1, k, 7), np.float32)
    states[..., 1] = 1.0
    states[..., 0] = np.arange(k) * 20.0
    states[..., 4] = 15.0
    states[..., 6] = 4.5
    return {"states": states, "is_cav": np.broadcast_to(np.eye(k, dtype=bool)[0], (n, t1, k)).copy()}


def test_validate_train_source_refuses_restricted_tokens(tmp_path):
    assert validate_train_source(tmp_path / "pack_v3") == tmp_path / "pack_v3"
    for bad in ("x/HOLDOUT/pack", "x/chunk5/p", "x/J4B_run", "x/INC1/p", "x/sealed/p",
                "x/CAL_SPLIT_A4.json"):
        with pytest.raises(TrainDataRefusal):
            validate_train_source(bad)
    with pytest.raises(TrainDataRefusal, match="split"):
        validate_train_source("ok", {"split": "CAL"})


def test_from_arrays_collate_keys_and_dtypes():
    ds = TrajectoryDataset.from_arrays(_arrays())
    batch = collate([ds[0], ds[1]])
    assert batch["states"].dtype == torch.float32 and batch["states"].shape == (2, 5, 3, 7)
    assert batch["present"].dtype == torch.bool and batch["slot_index"].dtype == torch.int64
    assert batch["is_cav"].shape == (2, 5, 3) and batch["cav_teacher"].shape == (2, 5, 3, 4)
    assert "driver_hist" not in batch and batch["identities"] == [None, None]


def test_history_on_adds_driver_history_to_every_record():
    ds = TrajectoryDataset.from_arrays(_arrays(), history_s=10.0, history_stride_s=1.0)
    batch = collate([ds[0], ds[1]])
    assert batch["driver_hist"].shape == (2, 3, 11, 9)
    assert batch["driver_hist_mask"].shape == (2, 3, 11)
    with pytest.raises(ValueError, match="driver history"):
        collate([ds[0], {k: v for k, v in ds[1].items() if "driver" not in k}])


def test_history_spec_and_front_masking():
    assert history_spec(None) == (10, 1, 10)
    assert history_spec(60.0, 1.0) == (121, 2, 61)
    with pytest.raises(ValueError):
        history_spec(60.0, 0.75)
    kin = torch.zeros(1, 2, 7)
    kin[..., 1] = 1.0
    feat, mask = long_history_features(
        kin, torch.ones(1, 2, dtype=torch.bool), torch.zeros(0, 2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool), history_s=10.0)
    assert feat.shape == (2, 11, 9) and mask[:, :-1].sum() == 0 and mask[:, -1].all()


def test_timeline_ids_normalisation_refuses_ambiguity():
    assert timeline_controller_ids(np.array([[0, 2]]), {}).tolist() == [[0, 2]]
    with pytest.raises(ValueError, match="ambiguous"):
        timeline_controller_ids(np.array([[1]]), {})
    assert timeline_controller_ids(np.array([[6, 3]]), {}).tolist() == [[3, 1]]


def test_window_keys_prefer_batch_and_round_t0():
    assert W.window_key({"batch": "b", "chunk": "c", "world": "w", "arm": "a",
                         "t0_s": 1.00049}) == ("b", "w", "a", 1.0)
    assert W.item_key({"chunk": "c", "world": "w", "arm": "a"}) == ("c", "w", "a", 0.0)


def test_filter_positions_is_fail_closed_and_batch_scoped():
    items = [{"batch": "chunk1", "world": "w1", "arm": "pi", "t0_s": 0.0},
             {"batch": "chunk2", "world": "w1", "arm": "pi", "t0_s": 0.0},
             {"batch": "chunk1", "world": "w2", "arm": "base", "t0_s": 0.0}]
    kept, info = W.filter_positions(items, [0, 0, 0], ["pack_v3"],
                                    exclude={("chunk1", "w1__pi")})
    assert kept == [1, 2] and info["dropped"] == 1      # same run in another batch survives
    keep = {"pack_v3": {("chunk1", "w2", "base", 0.0), ("chunk9", "w9", "base", 0.0)}}
    with pytest.raises(ValueError, match="absent from corpus"):
        W.filter_positions(items, [0, 0, 0], ["pack_v3"], keep=keep)


class _Fake:
    def __init__(self, items, packs, names):
        self.items, self.packs = items, packs
        self.pack_dirs = [Path(n) for n in names]
        self.pack_names = list(names)

    def __len__(self):
        return len(self.items)

    def meta(self, i):
        return self.items[i]

    def pack_index(self, i):
        return self.packs[i]


def test_pack_name_resolution_maps_staged_basename_and_refuses_ambiguity():
    items = [{"batch": "c1", "world": "w1", "arm": "base"},
             {"batch": "c2", "world": "w2", "arm": "base"}]
    corpus = _Fake(items, [0, 1], ["pack", "pack_dspeed"])
    keep = {"pack_v3": {("c1", "w1", "base", 0.0)}, "pack_dspeed": {("c2", "w2", "base", 0.0)}}
    resolved, mapping = W.resolve_window_pack_names(corpus, corpus.pack_names, keep)
    assert mapping["pack_v3"] == "pack" and set(resolved) == {"pack", "pack_dspeed"}
    with pytest.raises(TrainDataRefusal, match="cannot uniquely"):
        W.resolve_window_pack_names(corpus, corpus.pack_names,
                                    {"pack_v3": {("zz", "w", "a", 0.0)}})
    with pytest.raises(TrainDataRefusal, match="unique"):
        W.resolve_window_pack_names(corpus, ["a", "a"], keep)


def test_select_windows_counts_by_pack(tmp_path):
    items = [{"batch": "c1", "world": f"w{i}", "arm": "base", "t0_s": 0.0} for i in range(4)]
    corpus = _Fake(items, [0, 0, 1, 1], ["pack_v3", "pack_dspeed"])
    listing = tmp_path / "list.json"
    windows = [{"pack": "pack_v3" if i < 2 else "pack_dspeed", "batch": "c1",
                "world": f"w{i}", "arm": "base", "t0_s": 0.0} for i in (0, 1, 3)]
    listing.write_text(json.dumps({"schema": W.WINDOW_SCHEMA, "n_windows": 3,
                                   "windows": windows}), encoding="utf-8")
    selection = W.select_windows(corpus, window_list=listing)
    assert selection.positions == (0, 1, 3)
    assert selection.kept_by_pack == {"pack_dspeed": 1, "pack_v3": 2}
    counts = W.window_counts(corpus, train_list=listing, val_list=listing)
    assert counts == {"pack_dspeed": (1, 1), "pack_v3": (2, 2)}


def test_controller_exclusion_drops_pi_and_wall_windows():
    ticks, k = 6, 3
    veh = np.zeros((ticks, k, 14), np.float32)
    veh[:, :, 7] = 1                                  # all CAV
    veh[:, :, 8] = 1                                  # native owners
    veh[3, 1, 8], veh[3, 1, 10] = 0, 1                # PI at tick 3 of window 1
    mask = np.ones((ticks, k), bool)
    corpus = SimpleNamespace(
        meta=lambda i: {"world": "w", "tick": [0, 4][i]}, run_end=lambda t: ticks,
        array=lambda name: veh if name == "veh_step" else mask)
    kept, receipt = W.ControllerExclusion(horizon_steps=5, leave_pi_wall_out=True)(
        corpus, [0, 1])
    assert kept == [1] and receipt["controller_windows_dropped"] == 1
    kept, _ = W.ControllerExclusion(horizon_steps=5)(corpus, [0, 1])
    assert kept == [0, 1]


def test_new_sources_contain_no_restricted_absolute_path_literals():
    restricted = re.compile(r"[A-Za-z]:[/\][^\"']*(sealed|holdout|chunk-?5|j4b|inc1|CAL_SPLIT)",
                            re.I)
    offenders = []
    for path in (ROOT / "src" / "wms").rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if restricted.search(line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, offenders[:5]


def test_guard_refuses_restricted_tokens_and_excluded_worlds(tmp_path):
    """``tools/data_guard.py`` (WP-C window-list DataGuard; renamed from ``holdout_guard.py``
    so the planner/reconstructor ``tools/holdout_guard.py`` keeps its own name)."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import data_guard as hg
    finally:
        sys.path.pop(0)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": hg.SPEC_SCHEMA,
                                "excluded": [{"world": "w8859m2700p10"}]}), encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    guard = hg.DataGuard(spec, [data], workspace=tmp_path)
    assert guard.check_path(data / "pack_v3" / "INDEX.json")
    for bad in (data / "chunk5" / "x.json", data / "J4B_run.json", data / "w8859m2700p10.json",
                tmp_path / "elsewhere" / "x.json"):
        with pytest.raises(hg.GuardError):
            guard.check_path(bad)
    with pytest.raises(hg.GuardError, match="TRAIN"):
        guard.check_record({"split": "CAL", "world": "w1"})
    assert guard.check_record({"split": "TRAIN", "world": "w1", "pack": "pack_v3"})
    with pytest.raises(hg.GuardError, match="forbidden source identity"):
        guard.check_record({"split": "TRAIN", "world": "w8859m2700p10"})
    assert guard.audit()["n_refusals"] >= 3
    with pytest.raises(hg.GuardError):
        hg.assert_unrestricted("a/sealed/b")
