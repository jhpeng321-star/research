"""V8WM (2026-10-17): merge-timing survival loss, slimmed / capped merge loss, train-vs-inference
physics, gap-rule overrides and ramp-aware checkpoint selection.

* merge timing: exact survival NLL (event + censored), zero gradient off the at-risk set,
  masked event dropped, BPTT chunks add up, metrics (stuck fraction, merge-time error sign);
  on a hazard_v5 rollout the gradient reaches the lane network although the rollout logits are
  decoupled (``lane_hazard_detach``);
* merge loss ``speed_cells`` / ``max_share``;
* physics OFF in training + ON in physics.json, loaded by the checkpoint loader;
* every new switch defaults OFF and is bit-identical (flat forms, composer, unroll + grads);
* v8ft configs reach settings / composer / physics.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from .test_lanev5 import _v5_model
from .test_merge_loss import _ctx as _merge_ctx
from .test_merge_loss import _default_batch, _recorded
from .test_merge_loss import _run as _run_default
from .test_merge_loss import _targets as _merge_targets
from wms.cli.assemble import build_physics
from wms.config.load import load_train_config
from wms.config.train import TrainConfig
from wms.core.errors import ConfigError
from wms.model.checkpoint import load_checkpoint
from wms.model.wms import WmSModel
from wms.train.losses import merge as mg
from wms.train.losses import merge_timing as mt
from wms.train.losses.base import LossContext, TermValue, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.selection import SameHorizonSelection
from wms.train.settings import TrainSettings
from wms.train.unroll import UnrollRunner

NEW_FLAT_KEYS = ("merge_loss_speed_cells", "merge_loss_max_share", "merge_timing_weight",
                 "merge_timing_metrics", "physics_overrides", "physics_inference",
                 "selection_ramp_weight", "selection_stuck_weight")


# ----------------------------------------------------------------------- merge timing (unit)
def _ramp_record(b=1, t=8, k=3, merge_at=(3, None, None)):
    """Slots in lane 2 inside the merge zone; slot i leaves to lane 1 after step merge_at[i]."""
    states = torch.zeros(b, t + 1, k, 7)
    states[..., 0] = torch.tensor([10.0, 30.0, 50.0][:k])
    lanes = torch.full((b, t + 1, k), 2, dtype=torch.long)
    for i, m in enumerate(merge_at):
        if m is not None:
            lanes[:, m + 1:, i] = 1
    states[..., 1:4] = F.one_hot(lanes, 3).float()
    states[..., 4] = 3.0
    states[..., 6] = 5.0
    present = torch.ones(b, t + 1, k, dtype=torch.bool)
    is_cav = torch.zeros(b, t + 1, k, dtype=torch.bool)
    targets = UnrollTargets(
        states=states, n_steps=t, weights=torch.ones(1, t, 1) / t,
        hdv_mask=present[:, 1:], aux_mask=present[:, 1:], aux_count=0,
        denom=torch.tensor(1.0), lane_target=lanes[:, 1:], loss_valid=present, is_cav=is_cav,
        dt_s=0.5, position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)
    return targets, lanes


def _timing_ctx(targets, pred_lanes, head, acts, c0=0, c1=None):
    """Model lanes ``[B, T+1, K]`` (index 0 = t0), hazard logits ``[B, T, K]`` (stay = 0)."""
    c1 = targets.n_steps if c1 is None else c1
    s = targets.states[..., 0]
    b, _, k = pred_lanes.shape
    logits = torch.zeros(b, c1 - c0, k, 3)
    logits[..., 0] = -8.0
    pred = {"lanes": pred_lanes[:, c0 + 1:c1 + 1], "masks": torch.ones(b, c1 - c0, k, dtype=torch.bool),
            "positions": s[:, c0 + 1:c1 + 1], "lane_logits": logits,
            "lane_v5_head_logit": head[:, c0:c1], "lane_v5_acts": acts[:, c0:c1]}
    current = {"lane": pred_lanes[:, c0], "present": torch.ones(b, k, dtype=torch.bool),
               "s_m": s[:, c0]}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def test_merge_timing_exact_nll_event_and_censoring():
    targets, lanes = _ramp_record()
    term = mt.MergeTimingTerm(mt.MergeTimingConfig(weight=1.0))
    term.prepare(targets)
    head = torch.randn(1, 8, 3, generator=torch.Generator().manual_seed(0)).requires_grad_()
    acts = torch.ones(1, 8, 3, dtype=torch.bool)
    # the model keeps every car in lane 2 (late/stuck model)
    model_lanes = torch.full((1, 9, 3), 2, dtype=torch.long)
    value = term(_timing_ctx(targets, model_lanes, head, acts)).value
    h = torch.sigmoid(head.detach())
    # slot 0: survives steps 0..2, event at 3; slots 1, 2: censored over all 8 steps
    expect = -(torch.log1p(-h[0, :3, 0]).sum() + torch.log(h[0, 3, 0])
               + torch.log1p(-h[0, :, 1:]).sum()) / 3.0
    torch.testing.assert_close(value.detach(), expect, rtol=1e-5, atol=1e-6)
    value.backward()
    assert bool(torch.isfinite(head.grad).all())
    assert float(head.grad[0, 4:, 0].abs().sum()) == 0.0          # after the recorded event
    assert float(head.grad[0, 3, 0]) < 0.0                        # event: raise the hazard
    assert float(head.grad[0, :3, 0].min()) > 0.0                 # survival: lower it
    assert float(head.grad[0, :, 1:].min()) > 0.0                 # censored: lower it
    m = mt.merge_timing_metrics(targets.prepared["merge_timing"])
    assert m["merge_timing_stuck_rec_frac"] == pytest.approx(2 / 3)
    assert m["merge_timing_stuck_pred_frac"] == pytest.approx(1.0)
    assert m["merge_timing_vehicles"] == 3.0 and m["merge_timing_events_used"] == 1.0


def test_merge_timing_masked_event_early_model_and_chunking():
    targets, lanes = _ramp_record()
    cfg = mt.MergeTimingConfig(weight=2.0)
    head = torch.randn(1, 8, 3, generator=torch.Generator().manual_seed(1))
    acts = torch.ones(1, 8, 3, dtype=torch.bool)
    acts[0, 3, 0] = False                         # physics forbids slot 0's merge at t*
    model_lanes = torch.full((1, 9, 3), 2, dtype=torch.long)
    model_lanes[0, 2:, 1] = 1                     # model's slot 1 leaves after step 1 (early)
    term = mt.MergeTimingTerm(cfg)
    term.prepare(targets)
    whole = term(_timing_ctx(targets, model_lanes, head, acts)).value
    h = torch.sigmoid(head)
    expect = -2.0 * (torch.log1p(-h[0, :3, 0]).sum() + torch.log1p(-h[0, :2, 1]).sum()
                     + torch.log1p(-h[0, :, 2]).sum()) / 3.0
    torch.testing.assert_close(whole, expect, rtol=1e-5, atol=1e-6)
    m = mt.merge_timing_metrics(targets.prepared["merge_timing"])
    assert m["merge_timing_event_masked"] == 1.0
    assert m["merge_timing_stuck_pred_frac"] == pytest.approx(2 / 3)
    term.prepare(targets)
    first = term(_timing_ctx(targets, model_lanes, head, acts, 0, 3)).value
    term(_timing_ctx(targets, model_lanes, head, acts, 0, 3))     # a recomputed chunk
    second = term(_timing_ctx(targets, model_lanes, head, acts, 3, 8)).value
    torch.testing.assert_close(first + second, whole)
    m2 = mt.merge_timing_metrics(targets.prepared["merge_timing"])
    assert m2["merge_timing_stuck_pred_n"] == m["merge_timing_stuck_pred_n"]


def test_merge_timing_error_sign_padding_and_metrics_only():
    targets, lanes = _ramp_record(merge_at=(2, 5, None))
    targets.loss_valid[:, 7:, 2] = False          # slot 2's record ends: censored, not 'stuck'
    term = mt.MergeTimingTerm(mt.MergeTimingConfig(weight=0.0, metrics=True))
    term.prepare(targets)
    model_lanes = torch.full((1, 9, 3), 2, dtype=torch.long)
    model_lanes[0, 5:, 0] = 1                     # model merges slot 0 two steps late
    model_lanes[0, 6:, 1] = 1                     # slot 1 0.0 s late
    out = term(_timing_ctx(targets, model_lanes, torch.zeros(1, 8, 3),
                           torch.ones(1, 8, 3, dtype=torch.bool)))
    assert out is None                            # metrics only: no loss term
    m = mt.merge_timing_metrics(targets.prepared["merge_timing"])
    assert m["merge_timing_stuck_den"] == 2.0 and m["merge_timing_stuck_pred_frac"] == 0.0
    assert sorted(m[mt.ERRORS_KEY]) == [0.0, 1.0]
    assert m["merge_timing_mean_err_s"] == pytest.approx(0.5)


# ----------------------------------------------------------------------- merge loss v8 options
def test_speed_cells_select_cells_and_default_is_unchanged():
    states, present, is_cav, lane = _recorded()
    s, v = states[..., 0] - 3.0, states[..., 4] * 0.8
    vals = {}
    for cells in (None, (0, 1, 2), (2,)):
        targets = _merge_targets(states, present, is_cav)
        loss = mg.MergeLoss(mg.MergeLossConfig(speed_weight=1.0, speed_cells=cells))
        loss.prepare(targets)
        vals[cells] = loss.components(_merge_ctx(targets, s, v, lane, present))["merge_speed"]
    torch.testing.assert_close(vals[None], vals[(0, 1, 2)])
    assert float(vals[(2,)]) > 0.0 and float(vals[(2,)]) != float(vals[None])
    with pytest.raises(ValueError):
        mg.MergeLossConfig(speed_cells=(3,))
    with pytest.raises(ConfigError):
        TrainConfig.from_dict({"loss": {"merge": {"speed_cells": [5]}}})


def test_max_share_caps_the_value_share():
    def terms(m1, m2, other):
        return {"position": TermValue(torch.tensor(other)),
                "merge_speed": TermValue(torch.tensor(m1)),
                "merge_discharge": TermValue(torch.tensor(m2))}
    t = terms(30.0, 10.0, 60.0)                                   # share 0.4
    prep = {}
    assert mg.cap_merge_share(t, 0.15, prep, 0)
    merge = float(t["merge_speed"].value + t["merge_discharge"].value)
    assert merge / (merge + 60.0) == pytest.approx(0.15)
    assert float(t["merge_speed"].value) / float(t["merge_discharge"].value) == pytest.approx(3.0)
    t = terms(3.0, 1.0, 60.0)
    assert not mg.cap_merge_share(t, 0.15, prep, 5) and float(t["merge_speed"].value) == 3.0
    assert prep["cap_scale"][5] == 1.0 and prep["cap_scale"][0] < 1.0


# ----------------------------------------------------------------------- defaults off
def test_switches_default_off_and_flat_forms_unchanged():
    flat = TrainConfig().to_legacy_flat()
    assert not set(NEW_FLAT_KEYS) & set(flat)
    for name in ("c4_wm", "configs/c4_wm_v5.toml", "configs/c4_wm_v7ft.toml"):
        assert not set(NEW_FLAT_KEYS) & set(load_train_config(name).to_legacy_flat()), name
    s = TrainSettings()
    assert not s.merge_timing_on and build_loss(s).merge_share_cap is None
    assert "merge_timing" not in build_loss(s).names
    assert not WmSModel.emit_merge_hazard


def test_explicit_off_is_bit_identical():
    base_rec, base_grads = _run_default(TrainSettings(), _default_batch(), truncation=2)
    off_rec, off_grads = _run_default(TrainSettings(
        merge_timing_weight=0.0, merge_timing_metrics=False, merge_loss_speed_cells=None,
        merge_loss_max_share=0.15, physics_overrides={}, physics_inference={},
        selection_ramp_weight=0.0), _default_batch(), truncation=2)
    assert base_rec == off_rec
    assert all(torch.equal(a, b) for a, b in zip(base_grads, off_grads))


def test_merge_timing_on_logs_and_leaves_other_terms_unchanged():
    base_rec, _ = _run_default(TrainSettings(), _default_batch(), truncation=2)
    rec, grads = _run_default(TrainSettings(merge_timing_weight=2.0), _default_batch(),
                              truncation=2)
    for key in ("position", "speed", "lane", "valid_targets"):
        assert rec[key] == base_rec[key], key
    for key in ("merge_timing_stuck_pred_frac", "ramp_abs_ds_m", "merge_timing_vehicles"):
        assert key in rec, key
    assert all(bool(torch.isfinite(g).all()) for g in grads)


# ----------------------------------------------------------------------- hazard_v5 rollout
def _ramp_batch():
    b, t, k = 2, 13, 6
    states = torch.zeros(b, t, k, 7)
    s0 = torch.tensor([-60.0, 5.0, 25.0, 45.0, -20.0, 0.0])
    v = torch.tensor([10.0, 4.0, 4.0, 4.0, 10.0, 0.0])
    states[..., 0] = s0 + v * 0.5 * torch.arange(t).view(1, t, 1)
    lanes = torch.tensor([1, 2, 2, 2, 1, 1]).view(1, 1, k).expand(b, t, k).clone()
    lanes[:, 5:, 2] = 1                            # slot 2 merges after step 4
    states[..., 1:4] = F.one_hot(lanes, 3).float()
    states[..., 4] = v
    states[..., 6] = 5.0
    present = torch.ones(b, t, k, dtype=torch.bool)
    present[..., 5] = False
    is_cav = torch.zeros(b, t, k, dtype=torch.bool)
    is_cav[..., 0] = True
    teacher = torch.zeros(b, t, k, 4)
    teacher[..., 0], teacher[..., 1], teacher[..., 2] = states[..., 0], lanes.float(), states[..., 4]
    g = torch.Generator().manual_seed(3)
    dh = torch.rand(b, k, 61, 9, generator=g)
    dh[..., 2:5] = F.one_hot(lanes[:, 0], 3).float().unsqueeze(2)
    return {"states": states, "present": present, "is_cav": is_cav, "cav_teacher": teacher,
            "hist": torch.rand(b, k, 10, 6, generator=g),
            "hist_mask": torch.ones(b, k, 10, dtype=torch.bool), "driver_hist": dh,
            "driver_hist_mask": torch.ones(b, k, 61, dtype=torch.bool)}


def test_hazard_v5_gradient_reaches_the_merge_head_despite_decouple():
    model, settings = _v5_model(["loss.merge_timing.weight=2.0"])
    model.lane_hazard_detach = True                # the LANEFIX decouple of the trainer
    model.train()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    assert model.emit_merge_hazard
    rec = runner.run(_ramp_batch(), unroll_steps=12, backward=True, truncation_steps=5)
    assert rec["merge_timing"] > 0.0 and math.isfinite(rec["merge_timing"])
    assert rec["merge_timing_risk_steps"] > 0
    grad = model.lane_v5.heads.weight.grad
    assert grad is not None and float(grad[2].abs().sum()) > 0.0   # merge head row


# ----------------------------------------------------------------------- physics split
def _fit(tmp_path, extra):
    from wms.train.trainer import WmSTrainer

    cfg = TrainConfig.from_dict({
        "runtime": {"device": "cpu", "threads": 2, "omp": 2},
        "curriculum": {"unroll_s": [3.0], "stage_steps": [2], "truncation_steps": 2},
        "optim": {"batch_size": 2},
        "selection": {"select_by_long_horizon": True, "val_every": 1}, **extra})
    batch = _default_batch()

    class Windows(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            rec = {k: v[index].numpy() for k, v in batch.items() if torch.is_tensor(v)}
            rec["slot_index"] = np.arange(4)
            rec["meta"] = {"world": "world-A"}
            return rec

    torch.manual_seed(0)
    from wms.cli.assemble import build_model
    physics = build_physics(cfg, resolver=lambda n: n)
    model = build_model(cfg, physics)
    trainer = WmSTrainer(cfg, model, Windows(), val_dataset=Windows(), out_dir=tmp_path)
    trainer.fit()
    return trainer


def test_train_off_inference_on_physics_and_two_step_smoke(tmp_path):
    trainer = _fit(tmp_path, {
        "physics": {"merge_end_wait": False, "mainline_merge_yield": False,
                    "overrides": {"gap_stopped_min_m": 1.25},
                    "inference": {"merge_end_wait": True, "mainline_merge_yield": True}},
        "loss": {"merge": {"speed_weight": 10.0, "discharge_weight": 5.0,
                           "cell_edges_m": [0.0, 60.0, 200.0], "line_m": 60.0,
                           "speed_cells": [1], "max_share": 0.15},
                 "merge_timing": {"weight": 2.0}}})
    assert not trainer.model.physics.merge_end_wait
    assert trainer.model.physics.gap_stopped_min_m == 1.25
    on = json.loads((tmp_path / "physics.json").read_text())
    off = json.loads((tmp_path / "physics_train.json").read_text())
    assert on["merge_end_wait"] and on["mainline_merge_yield"] and on["gap_stopped_min_m"] == 1.25
    assert not off["merge_end_wait"] and not off["mainline_merge_yield"]
    receipt = json.loads((tmp_path / "TRAIN_RECEIPT.json").read_text())
    assert receipt["physics"]["merge_end_wait"] and not receipt["physics_train"]["merge_end_wait"]
    loaded = load_checkpoint(tmp_path / "wm_s.pt")
    assert loaded.model.physics.merge_end_wait and loaded.model.physics.mainline_merge_yield
    rows = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["merge_timing"] >= 0.0 and "merge_timing_value_share" in row
        assert "merge_cap_scale_mean" in row and row["merge_cap_scale_min"] <= 1.0
    val = [json.loads(x) for x in (tmp_path / "val_log.jsonl").read_text().splitlines()]
    assert val and all("mean_merge_timing" in r and "mean_ramp_abs_ds_m" in r
                       and "mean_merge_timing_stuck_pred_frac" in r for r in val)


def test_physics_override_and_inference_key_validation():
    with pytest.raises(ConfigError):
        TrainConfig.from_dict({"physics": {"overrides": {"lane_gap_rule": 1.0}}})
    with pytest.raises(ConfigError):
        TrainConfig.from_dict({"physics": {"inference": {"merge_end_wait": 1.0}}})
    cfg = TrainConfig.from_dict({"physics": {"overrides": {"gap_t_lead_s": 0.4}}})
    assert build_physics(cfg, resolver=lambda n: n).gap_t_lead_s == 0.4
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).physics == cfg.physics


# ----------------------------------------------------------------------- selection
def _rows(step, pos60, ramp60, stuck_pred):
    return [{"horizon_s": 60.0, "valid_targets": 10, "mean_position": pos60,
             "mean_ramp_abs_ds_m": ramp60, "mean_merge_timing_stuck_pred_frac": stuck_pred,
             "mean_merge_timing_stuck_rec_frac": 0.02}]


def test_selection_ramp_terms_change_the_choice_only_when_on():
    plain = SameHorizonSelection(horizons=(60.0,), references=(0.0,), eligible_stage_s=60.0)
    ramp = SameHorizonSelection(horizons=(60.0,), references=(0.0,), eligible_stage_s=60.0,
                                ramp_weight=1.0, stuck_weight=1.0)
    for sel in (plain, ramp):
        sel.consider_event(_rows(1, 50.0, 100.0, 0.10), stage_s=60.0, step=1)
        sel.consider_event(_rows(2, 50.5, 50.0, 0.02), stage_s=60.0, step=2, stage_end=True)
    plain.finalize(lambda step, hs: [])
    ramp.finalize(lambda step, hs: [])
    assert plain.best["step"] == 1 and ramp.best["step"] == 2
    assert "ramp_weight" in ramp.summary() and "ramp_weight" not in plain.summary()


# ----------------------------------------------------------------------- configs
@pytest.mark.parametrize("path,steps", [("configs/c4_wm_v8ft.toml", (300, 250)),
                                        ("configs/c4_wm_v8ft_short.toml", (80, 60))])
def test_v8ft_configs(path, steps):
    cfg = load_train_config(path)
    s = TrainSettings.from_config(cfg)
    assert cfg.curriculum.stage_steps == steps and cfg.curriculum.unroll_s == (60.0, 120.0)
    assert not s.merge_end_wait and not s.mainline_merge_yield
    assert dict(s.physics_inference) == {"merge_end_wait": True, "mainline_merge_yield": True}
    assert s.merge_loss_speed_cells == (2,) and s.merge_loss_discharge_weight <= 5.0
    assert s.merge_loss_max_share == 0.15 and s.merge_timing_weight > 0.0
    ratios = {src.name: src.mix_ratio for src in cfg.data.c4_sources}
    assert ratios == {"native_v5": 0.40, "genrand_plain": 0.20, "v6_policy": 0.35,
                      "genrand_branch": 0.05}
    assert cfg.selection.val_every == 50 and cfg.model.init_from is not None
    composer = build_loss(s)
    assert composer.names.index("merge_timing") > composer.names.index("merge_discharge")
    assert composer.merge_share_cap == 0.15
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()).loss == cfg.loss
    physics = build_physics(cfg, resolver=lambda n: n)
    assert not physics.merge_end_wait
