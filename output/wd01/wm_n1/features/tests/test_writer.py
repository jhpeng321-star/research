"""Writer tests: memmap round-trip, resumable skip, overwrite refusal."""
from __future__ import annotations

import json

import numpy as np
import pytest

from features import featurizer as F
from features import writer as W
from synth_features import MAIN_RIGHT, plan_row, row, veh, write_run


def _shard(tmp_path):
    rows = [row(0.5, {"mainline.0": veh(MAIN_RIGHT, 10, 30),
                      "mainline.1": veh(MAIN_RIGHT, 20, 28)}),
            row(1.0, {"mainline.0": veh(MAIN_RIGHT, 10.5, 30)})]
    root = write_run(tmp_path / "base", rows)
    return F.build_shard("chunk4", "wsynth", "base", root, split="TRAIN",
                         plan_row=plan_row(0.0), outcomes_root=None)


def test_round_trip_write_read_equality(tmp_path):
    sh = _shard(tmp_path)
    data_root = tmp_path / "data"
    rec = W.write_shard(sh, data_root)
    assert rec["status"] == "written"
    dirp = W.shard_dir(data_root, "chunk4", "wsynth", "base")
    for name in ("veh_step", "veh_mask", "slot_index", "plan", "wall",
                 "wall_mask", "counter", "accel_next", "target_mask",
                 "lane_next", "arrivals"):
        back = np.load(dirp / f"{name}.npy")
        assert np.array_equal(back, sh[name])
    assert json.loads((dirp / "slot_vid.json").read_text()) == sh["slot_vid"]


def test_resumable_skip_and_index(tmp_path):
    sh = _shard(tmp_path)
    data_root = tmp_path / "data"
    W.write_shard(sh, data_root)
    rec2 = W.write_shard(sh, data_root)
    assert rec2["status"] == "skipped"
    index = W.load_index(data_root)
    index = W.update_index(index, rec2["record"])
    W.save_index(data_root, index)
    W.write_sums(data_root, index)
    assert (data_root / "INDEX.json").is_file()
    assert (data_root / "SHA256SUMS").is_file()
    rep = W.verify_index(data_root, index)
    assert rep["ok"], rep["errors"]
    key = W.pair_key("wsynth", 0.0)
    assert index["pairs"][key] == ["chunk4/wsynth__base"]


def test_refuses_overwrite(tmp_path):
    sh = _shard(tmp_path)
    data_root = tmp_path / "data"
    dirp = W.shard_dir(data_root, "chunk4", "wsynth", "base")
    dirp.mkdir(parents=True)
    (dirp / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(W.WriterError):
        W.write_shard(sh, data_root)
    # --force rebuilds
    rec = W.write_shard(sh, data_root, force=True)
    assert rec["status"] == "written"


def test_estimate_positive():
    assert W.expected_bytes(2281, 128) > 0
    assert W.expected_bytes(2281, 128) > W.expected_bytes(1000, 128)
