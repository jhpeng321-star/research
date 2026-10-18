"""MERGECAL (2026-10-18): per-position merge-head calibration (inference only).

* ``lv5.merge_position_offset`` / ``lane_logits_v5(merge_calib=)``: off = bit-identical, offsets
  reach only ramp-lane rows of their distance bin;
* ``load_checkpoint`` applies the payload key ``lane_v5_merge_calib`` (and config gates);
* ``tools/merge_calib.py``: exposure / hazard bookkeeping, the fixed-point update and the
  calibrated checkpoint copy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from .test_lanefix import _phys
from .test_lanev5 import _batch, _rollout, _single_cars, _v5_model
from wms.model import lane_v5 as lv5
from wms.model.checkpoint import load_checkpoint, save_checkpoint
from wms.physics.mobil import mobil_merge_logits

ROOT = Path(__file__).resolve().parents[2]
EDGES = (0.0, 5.0, 15.0, 25.0, 80.0)


def _tool():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("merge_calib", ROOT / "tools" / "merge_calib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _logits(rows, calib=None, bias=-3.0):
    phys = _phys()
    net = lv5.LaneV5Net()
    with torch.no_grad():
        net.heads.bias.fill_(bias)
    kin = _single_cars(rows)
    present = torch.ones(kin.shape[0], 1, dtype=torch.bool)
    base = mobil_merge_logits(kin, present, phys)
    acc = lv5.initial_acc(present, kin, phys, None, None, 1)
    raw, _, inst = lv5.step_features(acc, kin, present, phys, None)
    return lv5.lane_logits_v5(net, raw, inst, present, base, phys, merge_calib=calib), phys


def test_offset_bins_and_lane_mask():
    phys = _phys()
    s1 = phys.merge_zone_s1_m
    s = torch.tensor([[s1 - 2.0, s1 - 10.0, s1 - 20.0, s1 - 60.0, s1 - 90.0, s1 - 10.0]])
    lane = torch.tensor([[2, 2, 2, 2, 2, 1]])
    off = lv5.merge_position_offset(lane, s, phys, (EDGES, (0.5, 1.0, 2.0, 3.0)))
    assert off.tolist() == [[0.5, 1.0, 2.0, 3.0, 0.0, 0.0]]     # d=90 outside, lane 1 untouched


def test_calibration_validation():
    with pytest.raises(ValueError):
        lv5.check_merge_calib(((0.0, 5.0), (1.0, 2.0)))
    with pytest.raises(ValueError):
        lv5.check_merge_calib(((0.0, 5.0, 5.0), (1.0, 2.0)))
    with pytest.raises(ValueError):
        lv5.check_merge_calib(((0.0, 5.0), (float("nan"),)))
    assert lv5.check_merge_calib(([0, 5], [1])) == ((0.0, 5.0), (1.0,))


def test_logits_off_is_identical_and_offset_moves_only_merge_rows():
    phys = _phys()
    s1 = phys.merge_zone_s1_m
    rows = [(s1 - 20.0, 2, 8.0), (s1 - 40.0, 1, 20.0), (s1 - 30.0, 0, 25.0)]
    (lg0, h0, a0), _ = _logits(rows)
    (lgz, hz, az), _ = _logits(rows, (EDGES, (0.0, 0.0, 0.0, 0.0)))
    assert torch.equal(lg0, lgz) and torch.equal(h0, hz) and torch.equal(a0, az)
    (lg1, h1, _), _ = _logits(rows, (EDGES, (0.0, 0.0, 1.5, 0.0)))
    assert float((h1[0, 0] - h0[0, 0]).detach()) == pytest.approx(1.5, abs=1e-5)   # ramp car, d = 20 m
    assert torch.equal(h1[1:], h0[1:])                                  # mainline heads untouched


def test_forced_window_deterministic_mode_unaffected():
    phys = _phys()
    s1 = phys.merge_zone_s1_m
    (lg0, _, _), _ = _logits([(s1 - 2.0, 2, 1.0)])
    (lg1, _, _), _ = _logits([(s1 - 2.0, 2, 1.0)], ((0.0, 5.0), (-5.0,)))
    assert torch.equal(lg0, lg1)          # feasible forced-window merge stays forced


def test_model_attribute_reaches_rollout():
    model, _ = _v5_model(['model.lane_decision="sample"'])
    model.eval()
    b = _batch()
    assert model.lane_v5_merge_calib is None
    ref = _rollout(model, b, lane_seed=3)
    model.lane_v5_merge_calib = ((0.0, 1000.0), (0.0,))
    zero = _rollout(model, b, lane_seed=3)
    for key in ("lanes", "positions", "masks"):
        assert torch.equal(ref[key], zero[key])


def test_checkpoint_payload_applies_calibration(tmp_path):
    from wms.cli.assemble import build_model as assemble_model, build_physics
    from wms.config.load import load_train_config
    cfg = load_train_config("configs/c4_wm_v5_lane.toml", ["runtime.device=cpu"])
    physics = build_physics(cfg, lambda n: n)
    model = assemble_model(cfg, physics)
    path = tmp_path / "wm_s.pt"
    save_checkpoint(path, model, cfg)
    (tmp_path / "physics.json").write_text(json.dumps(asdict(physics)), encoding="utf-8")
    assert load_checkpoint(path).model.lane_v5_merge_calib is None
    tool = _tool()
    out = tmp_path / "cal" / "wm_s_mcal.pt"
    tool.write_checkpoint(str(path), str(out), 0.05, EDGES, (0.0, 0.5, 1.0, 2.0), {"iter": 2})
    loaded = load_checkpoint(out).model
    assert loaded.lane_v5_merge_calib == (EDGES, (0.0, 0.5, 1.0, 2.0))
    assert loaded.lane_v5_gates == (0.25, 0.10, 0.05)
    assert (out.parent / "physics.json").is_file()
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_ramp_exposure_counts_risk_and_first_departure():
    tool = _tool()
    z = tool.ZONE_END_M
    # car 0: in zone at d = 30, 22, 12 then merges; car 1: upstream then never merges
    s = np.array([[z - 30, -10], [z - 22, -5], [z - 12, z - 70], [z - 8, z - 60]], float)
    lane = np.array([[2, 2], [2, 2], [2, 2], [1, 2]])
    present = np.ones((4, 2), bool)
    ex = tool.ramp_exposure(s, lane, present, [0, 1], edges=(0.0, 5.0, 15.0, 25.0, 80.0))
    assert ex["at_risk"].tolist() == [0, 1, 1, 2] and ex["events"].tolist() == [0, 1, 0, 0]
    assert ex["dep_d"] == [pytest.approx(12.0)] and ex["dep_t"] == [1.5]
    present[2, 1] = False           # exit ends risk for the step before it
    ex = tool.ramp_exposure(s, lane, present, [1], edges=(0.0, 5.0, 15.0, 25.0, 80.0))
    assert ex["at_risk"].sum() == 0


def test_update_moves_towards_sumo_and_respects_window_and_exposure():
    tool = _tool()
    st = {"sumo": {"at_risk": [100, 100, 100, 10], "events": [5, 10, 10, 1]},
          "wm": {"at_risk": [100, 100, 100, 10], "events": [20, 5, 10, 0]}}
    r = tool.log_ratio(st)
    assert r[1] > 0 and r[0] < 0 and abs(r[2]) < 1e-12
    assert tool.fitted_bins(st, False) == [1, 2]       # window off, bin 3 too little exposure
    assert tool.fitted_bins(st, True) == [0, 1, 2]
    new = tool.update(np.zeros(4), st, False)
    assert new[0] == 0.0 and new[3] == 0.0 and new[1] == pytest.approx(r[1])
    st["wm"]["events"] = [20, 0, 10, 0]
    assert tool.update(np.zeros(4), st, False)[1] == pytest.approx(min(tool.MAX_STEP,
                                                                        tool.log_ratio(st)[1]))


def test_pooled_summary():
    tool = _tool()
    ex = {"at_risk": np.array([1.0, 2.0]), "events": np.array([1.0, 0.0]), "dep_d": [3.0],
          "dep_t": [1.0]}
    res = [{"idx": 0, "error": None, "n_ramp": 2, "wm": ex, "sumo": ex,
            "t_wm": [1.0, np.inf], "t_rec": [3.0, 2.0], "ds60": 1.0, "ds120": 2.0},
           {"idx": 1, "error": "x"}]
    st = tool.pooled(res)
    assert st["starts"] == 1 and st["errors"] == 1 and st["stuck_wm"] == 1
    assert st["median_dep_err_s"] == -2.0 and st["wm"]["share_last5m"] == 1.0


def test_secant_update_uses_measured_response():
    tool = _tool()
    st = {"sumo": {"at_risk": [100, 100], "events": [10, 10]},
          "wm": {"at_risk": [100, 100], "events": [10, 5]}}
    r = tool.log_ratio(st)
    prev = (np.array([0.0, 0.0]), np.array([0.0, r[1] + 0.5]))   # +1 logit moved r by 0.5
    new = tool.update(np.array([0.0, 1.0]), st, True, prev)
    assert new[0] == pytest.approx(0.0)                          # r = 0: no step
    assert new[1] == pytest.approx(1.0 + r[1] / 0.5)
    prev = (np.array([0.0, 0.0]), np.array([0.0, r[1] - 3.0]))   # wrong-way response -> floor
    new = tool.update(np.array([0.0, 1.0]), st, True, prev)
    assert new[1] == pytest.approx(1.0 + min(r[1] / tool.MIN_SLOPE, tool.MAX_SECANT_STEP))
