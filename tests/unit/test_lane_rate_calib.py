"""LANERATECAL (2026-10-19): closed-loop per-head lane-change RATE calibration.

* ``tools/lane_rate_calib.py``: event counting, the damped clipped fixed point and convergence;
* ``wms.model.checkpoint``: ``lane_calib.json`` is applied at load (absent = bit-identical
  no-op), additive on the config offsets, and refuses a calibration of another checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("lane_rate_calib",
                                                  ROOT / "tools" / "lane_rate_calib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lane_events_counts_each_transition():
    tool = _tool()
    lane = np.array([[1, 0, 2], [0, 1, 1], [0, 1, 1], [1, 0, 1]])
    msk = np.ones_like(lane, bool)
    counts = tool.lane_events(lane, msk, np.ones(3, bool))
    assert counts == {"overtake": 2, "return": 2, "merge": 1}


def test_lane_events_ignores_missing_endpoints():
    tool = _tool()
    lane = np.array([[1], [0], [0]])
    msk = np.array([[True], [True], [False]])
    assert tool.lane_events(lane, msk, np.ones(1, bool)) == {"overtake": 1, "return": 0,
                                                             "merge": 0}


def test_rate_update_is_damped_clipped_and_converges():
    tool = _tool()
    heads = ("overtake", "return")
    target = {"overtake": 100, "return": 100}
    offsets = {"overtake": 0.0, "return": 0.0, "merge": 0.0}
    got = {"overtake": 300, "return": 25, "merge": 0}
    new = tool.rate_update(offsets, got, target, heads, damping=1.0, max_step=2.0)
    assert new["overtake"] == pytest.approx(np.log(100 / 300), abs=1e-6)
    assert new["return"] == pytest.approx(np.log(4.0), abs=1e-6)
    clipped = tool.rate_update(offsets, got, target, heads, damping=1.0, max_step=1.0)
    assert clipped["return"] == pytest.approx(1.0, abs=1e-6)      # clipped log(4)
    assert not tool.converged(got, target, heads, 0.10)
    assert tool.converged({"overtake": 105, "return": 95}, target, heads, 0.10)
    assert tool.converged({"overtake": 100, "return": 100}, target, heads, 0.0)
    assert not tool.converged({"overtake": 0, "return": 100}, target, heads, 0.10)
    assert tool.relation({"overtake": 0, "return": 100}, target, heads)["overtake"] == 1.0
    assert tool.relation({"overtake": 3, "return": 0}, {"overtake": 0, "return": 0},
                         heads)["overtake"] == float("inf")


def _saved_checkpoint(tmp_path, offsets=(0.0, 0.0, 0.0)):
    from wms.cli.assemble import build_model as assemble_model, build_physics
    from wms.config.load import load_train_config
    from wms.model.checkpoint import save_checkpoint
    flat = ["runtime.device=cpu"]
    if any(offsets):
        flat.append(f"model.lane_logit_offsets=[{offsets[0]},{offsets[1]},{offsets[2]}]")
    cfg = load_train_config("configs/c4_wm_v5_lane.toml", flat)
    physics = build_physics(cfg, lambda n: n)
    model = assemble_model(cfg, physics)
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, cfg)
    (tmp_path / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")
    return path


def test_checkpoint_without_calibration_is_noop(tmp_path):
    from wms.model.checkpoint import load_checkpoint
    path = _saved_checkpoint(tmp_path, offsets=(0.5, -0.5, 0.0))
    loaded = load_checkpoint(path).model
    assert loaded.lane_rate_calib is None
    assert loaded.lane_logit_offsets == (0.5, -0.5, 0.0)


def test_checkpoint_reads_lane_calib_json_additively(tmp_path):
    from wms.model.checkpoint import LANE_CALIB_FORMAT, load_checkpoint, sha256_file
    path = _saved_checkpoint(tmp_path, offsets=(0.5, 0.0, 0.0))
    (tmp_path / "lane_calib.json").write_text(json.dumps({
        "format": LANE_CALIB_FORMAT, "checkpoint_sha256": sha256_file(path),
        "offsets": {"overtake": -1.25, "return": 0.75, "merge": 0.0}}), encoding="utf-8")
    loaded = load_checkpoint(path).model
    assert loaded.lane_logit_offsets == pytest.approx((-0.75, 0.75, 0.0))
    assert loaded.lane_rate_calib["offset_vector"] == [-1.25, 0.75, 0.0]


def test_checkpoint_refuses_calibration_of_another_checkpoint(tmp_path):
    from wms.model.checkpoint import (CheckpointError, LANE_CALIB_FORMAT, load_checkpoint,
                                      sha256_file)
    path = _saved_checkpoint(tmp_path)
    other = sha256_file(path)[:-1] + ("0" if sha256_file(path)[-1] != "0" else "1")
    (tmp_path / "lane_calib.json").write_text(json.dumps({
        "format": LANE_CALIB_FORMAT, "checkpoint_sha256": other,
        "offset_vector": [0.0, 0.0, 0.0]}), encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_checkpoint_refuses_malformed_calibration(tmp_path):
    from wms.model.checkpoint import CheckpointError, load_checkpoint
    path = _saved_checkpoint(tmp_path)
    (tmp_path / "lane_calib.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)
    (tmp_path / "lane_calib.json").write_text(json.dumps({"offsets": [1.0, 2.0]}),
                                              encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_tool_guards_restricted_paths():
    tool = _tool()
    with pytest.raises(SystemExit):
        tool.guard("data/sealed/holdout.pkl")
    assert tool.guard("data/v5_native") == "data/v5_native"
