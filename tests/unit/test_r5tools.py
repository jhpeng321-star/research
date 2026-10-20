"""R5TOOLS: optimizer parameter protection, merge-timing helpers, v8wm3 k>1 summaries."""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.config.train import TrainConfig
from wms.train.settings import TrainSettings
from wms.train.trainer import WmSTrainer, build_optimizer

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MT = _load("wm_merge_timing", REPO / "scripts" / "c4" / "wm_merge_timing.py")
V8 = _load("v8wm3_check", REPO / "tools" / "v8wm3_check.py")


# --------------------------------------------------------------------------- (A) optimizer
class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(3, 4)
        self.trunk_head = torch.nn.Linear(4, 2)
        self.head = torch.nn.Linear(2, 1)


def test_optimizer_defaults_are_bit_identical():
    model = _Tiny()
    opt, report = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 1
    assert [id(p) for p in opt.param_groups[0]["params"]] == [
        id(p) for p in model.parameters()]
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)
    assert report["frozen_parameters"] == 0
    assert report["groups"][0]["prefix"] is None


def test_optimizer_freezes_prefixes_and_scales_longest_prefix():
    model = _Tiny()
    opt, report = build_optimizer(
        model, lr=0.1, weight_decay=0.0, freeze_prefixes=("head",),
        lr_scale={"trunk": 0.5, "trunk_head": 0.1})
    assert not model.head.weight.requires_grad
    frozen = {id(p) for p in model.head.parameters()}
    used = {id(p) for group in opt.param_groups for p in group["params"]}
    assert not frozen & used
    assert report["frozen_tensors"] == 2
    assert report["frozen_parameters"] == 3
    assert report["unmatched_freeze_prefixes"] == []
    # longest prefix wins: trunk_head (0.01) over trunk (0.05)
    assert sorted(group["lr"] for group in opt.param_groups) == pytest.approx([0.01, 0.05])
    assert report["unmatched_lr_scale_prefixes"] == []


def test_stage_lr_keeps_group_scale():
    model = _Tiny()
    opt, _ = build_optimizer(model, lr=0.1, weight_decay=0.0,
                             lr_scale={"trunk_head": 0.1})
    trainer = object.__new__(WmSTrainer)
    trainer.optimizer = opt
    trainer.set_stage_lr(0.2)
    assert sorted(group["lr"] for group in opt.param_groups) == pytest.approx([0.02, 0.2])


def test_settings_and_config_roundtrip():
    assert TrainSettings().freeze_prefixes == ()
    assert TrainSettings().lr_scale is None
    flat = TrainConfig().to_legacy_flat()
    assert "freeze_prefixes" not in flat and "lr_scale" not in flat
    cfg = TrainConfig.from_dict({"optim": {"freeze_prefixes": ["head"],
                                           "lr_scale": {"trunk": 0.25}}})
    flat = cfg.to_legacy_flat()
    assert flat["freeze_prefixes"] == ["head"] and flat["lr_scale"] == {"trunk": 0.25}
    settings = TrainSettings.from_config(cfg)
    assert settings.freeze_prefixes == ("head",)
    assert settings.lr_scale == {"trunk": 0.25}
    assert TrainConfig.from_legacy_flat(flat).optim.lr_scale == {"trunk": 0.25}
    with pytest.raises(ValueError):
        TrainSettings(freeze_prefixes=("",))
    with pytest.raises(ValueError):
        TrainSettings(lr_scale={"trunk": 0.0})


# --------------------------------------------------------------------------- (B) merge timing
def _arrays(lane, pos=None, mask=None):
    lane = np.asarray(lane, dtype=float)
    rows, n = lane.shape
    return {
        "lane": lane,
        "pos": np.zeros_like(lane) if pos is None else np.asarray(pos, float),
        "mask": np.ones_like(lane, bool) if mask is None else np.asarray(mask, bool),
    }


def test_merge_time_censor_and_position():
    a = _arrays([[2, 1], [2, 1], [1, 1], [1, 1]])
    assert MT.merge_time(a["lane"], a["mask"], 0) == pytest.approx(1.0)
    assert MT.merge_time(a["lane"], a["mask"], 1) == 0.0
    assert MT.merge_time(a["lane"], a["mask"], 0) == MT.DT_S * 2
    b = _arrays([[2], [2], [2]], pos=[[10.0], [20.0], [30.0]])
    assert MT.merge_time(b["lane"], b["mask"], 0) == float("inf")
    assert MT.censor_time(b["mask"], 0) == pytest.approx(1.0)
    assert np.isnan(MT.merge_position(b["pos"], b["lane"], b["mask"], 0))
    c = _arrays([[2], [2], [1]], pos=[[10.0], [20.0], [30.0]])
    assert MT.merge_position(c["pos"], c["lane"], c["mask"], 0) == pytest.approx(20.0)


def test_km_survival_and_quantiles():
    curve = MT.km_survival([1.0, 2.0, 3.0, 4.0], [True, False, True, False])
    assert [row["survival"] for row in curve] == pytest.approx([0.75, 0.375])
    assert MT.km_quantile(curve, 0.5) == pytest.approx(3.0)
    assert MT.km_quantile(curve, 0.9) is None
    summary = MT.km_summary([1.0, 2.0, 3.0, 4.0], [True, False, True, False])
    assert summary["events"] == 2 and summary["censored"] == 2


def test_gap_buckets_and_t0_gap_identity():
    assert [MT.gap_bucket(g) for g in (10, 30, 60, float("inf"))] == [
        "tight", "medium", "open", "unknown"]
    kin = np.zeros((3, 7))
    kin[0, 0], kin[1, 0], kin[2, 0] = 0.0, 20.0, -15.0
    kin[0, 2 + 2] = 1.0   # ego on the ramp lane
    kin[1, 1 + 1] = 1.0
    kin[2, 1 + 1] = 1.0
    present = np.ones(3, bool)
    assert MT.t0_lane1_gap(kin, present, 0) == pytest.approx(15.0)


def test_zone_stats_and_timing_block():
    pos = np.array([[10.0, -5.0], [20.0, -10.0]])
    vel = np.array([[10.0, 1.0], [20.0, 2.0]])
    lane = np.array([[1, 2], [1, 2]])
    mask = np.ones_like(pos, bool)
    stats = MT.zone_stats(pos, vel, mask, lane)
    assert stats["occupancy"] == pytest.approx(1.0)
    assert stats["speed_mps"] == pytest.approx(15.0)
    assert stats["spillback_veh"] == pytest.approx(1.0)
    assert stats["spillback_step_fraction"] == pytest.approx(1.0)
    block = MT.timing_block([3.0, 1.0, float("inf")], [2.0, 4.0, 1.0])
    assert block["matched"] == 2
    assert block["median_signed_s"] == pytest.approx(-1.0)
    assert block["median_abs_s"] == pytest.approx(2.0)
    assert block["early_gt_2s"] == 1 and block["late_gt_2s"] == 0


def _row(key, sample, events, sumo_events, gaps):
    cars = [{"id": f"v{i}", "gap_m": gap, "wm_t": (2.0 if ev else None),
             "sumo_t": (3.0 if sev else None), "wm_duration_s": 2.0 if ev else 10.0,
             "sumo_duration_s": 3.0 if sev else 10.0, "wm_event": ev, "sumo_event": sev,
             "wm_pos": 70.0 if ev else float("nan"),
             "sumo_pos": 72.0 if sev else float("nan")}
            for i, (ev, sev, gap) in enumerate(zip(events, sumo_events, gaps))]
    zone = {"occupancy": 1.0, "speed_mps": 10.0, "spillback_veh": 0.0,
            "spillback_step_fraction": 0.0}
    return {"key": key, "sample": sample, "seed": sample, "cars": cars,
            "zone_wm": zone, "zone_sumo": zone}


def test_aggregate_variant_probabilities_and_strata():
    rows = [_row("s1", 0, [True, False], [True, False], [10.0, 100.0]),
            _row("s1", 1, [False, True], [True, False], [10.0, 100.0])]
    block = MT.aggregate_variant(rows, k=2, horizon_s=10.0)
    assert block["states"] == 1 and block["scored_states"] == 1
    assert block["wm"]["merge_fraction"] == pytest.approx(0.5)
    assert block["sumo"]["merge_fraction"] == pytest.approx(0.5)
    assert block["timing"]["matched"] == 1
    assert block["by_gap"]["tight"]["cars"] == 2
    assert block["by_gap"]["open"]["cars"] == 2
    states = {row["key"]: row for row in block["per_state"]}
    assert states["s1"]["merge_probability"] == pytest.approx(0.5)
    assert states["s1"]["all_merged_probability"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- (C) v8wm3 k>1
def test_v8wm3_state_helpers():
    rows = [{"ds60": 60, "ds120": 120}, {"ds60": 1, "ds120": 2}, {"error": "x"}]
    assert V8.state_blow_probability(rows) == pytest.approx(0.5)
    assert np.isnan(V8.state_blow_probability([{"error": "x"}]))
    assert V8.is_blowup({"ds60": 51, "ds120": 101})
    assert not V8.is_blowup({"ds60": 49, "ds120": 101})
    assert V8.severity_tail([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert V8.severity_tail([], 90) is None
    assert V8.paired_failure_counts({0: 0.0, 1: 1.0, 2: 0.5}, {0: 1.0, 1: 0.0, 2: 0.0}) == (1, 2)


def _v8_row(idx, variable, sample, ds60, ds120):
    return {"idx": idx, "var": variable, "sample": sample, "error": None,
            "t_wm": np.array([5.0]), "t_rec": np.array([4.0]),
            "ds60": ds60, "ds120": ds120, "ramp_ds60": ds60,
            "ramp_ds120": ds120, "ramp_ds_all60": np.array([ds60]),
            "ramp_ds_all120": np.array([ds120])}


def test_v8wm3_summarize_k1_unchanged_and_k2_details(tmp_path):
    k1 = tmp_path / "k1.pkl"
    results = [_v8_row(0, "a", 0, 10.0, 20.0), _v8_row(1, "a", 0, 60.0, 120.0)]
    k1.write_bytes(pickle.dumps({"specs": [("a", "x", {})], "results": results, "k": 1}))
    text1 = V8.summarize(k1)
    assert "| a | 2 | 2 |" in text1 and "| 1/2 |" in text1
    assert "details" not in text1
    k2 = tmp_path / "k2.pkl"
    results2 = [_v8_row(0, "a", 0, 60.0, 120.0), _v8_row(0, "a", 1, 1.0, 2.0),
                _v8_row(1, "a", 0, 1.0, 2.0), _v8_row(1, "a", 1, 1.0, 2.0),
                _v8_row(0, "b", 0, 1.0, 2.0), _v8_row(0, "b", 1, 1.0, 2.0),
                _v8_row(1, "b", 0, 60.0, 120.0), _v8_row(1, "b", 1, 60.0, 120.0)]
    k2.write_bytes(pickle.dumps({"specs": [("a", "x", {}), ("b", "y", {})],
                                 "results": results2, "k": 2}))
    text2 = V8.summarize(k2)
    assert "k=2 details" in text2
    assert "per-state blow-up probability" in text2
    assert "1/2 |" in text2                       # state a fails once of two samples
    assert "1 new failures, 1 recovered" in text2
    assert "scored roster: 2/2 states" in text2
