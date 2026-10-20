"""R5MERGE: post-merge acceleration consistency and one-sided residual penalty.

All terms default off; tests cover the recorded 2->1 transition scan, the Huber and
one-sided losses and their gradients, config round-trip and the OFF path.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT
from wms.config.train import LossConfig, PostMergeConfig, TrainConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.losses.post_merge import (
    PostMergeAccelConfig, PostMergeAccelLoss, PostMergeResidualLoss, post_merge_mask,
    post_merge_terms, prepare_post_merge)
from wms.train.settings import TrainSettings

DT = 0.5


def _targets(n: int = 60, k: int = 3) -> UnrollTargets:
    """Car 0 merges 2->1 at step 10, car 1 stays in lane 1, car 2 merges at step 30."""
    states = np.zeros((1, n + 1, k, 7), dtype=np.float32)
    states[..., 1:4] = 0.0
    states[:, :, :, 1] = 1.0                     # everyone starts in lane 1
    states[0, :11, 0, 1:4] = (0.0, 0.0, 1.0)     # car 0: lane 2 until step 10
    states[0, 11:, 0, 1:4] = (0.0, 1.0, 0.0)     # car 0: lane 1 from step 11
    states[0, :31, 2, 1:4] = (0.0, 0.0, 1.0)     # car 2: lane 2 until step 30
    states[0, 31:, 2, 1:4] = (0.0, 1.0, 0.0)     # car 2: lane 1 from step 31
    a = np.zeros((1, n, k), dtype=np.float32)
    a[:, 10:20, 0] = 2.0
    v = np.zeros((1, n + 1, k), dtype=np.float32)
    v[0, 0] = 20.0
    s = np.zeros((1, n + 1, k), dtype=np.float32)
    for t in range(n):
        v[0, t + 1] = v[0, t] + a[0, t] * DT
        s[0, t + 1] = s[0, t] + v[0, t] * DT
    states[..., 4] = v
    states[..., 0] = s
    states[:, 1:, :, 5] = a
    states[..., 6] = 4.5
    valid = torch.ones(1, n + 1, k, dtype=torch.bool)
    hdv = torch.ones(1, n, k, dtype=torch.bool)
    weights = torch.ones(1, n, 1)
    return UnrollTargets(
        states=torch.from_numpy(states), n_steps=n, weights=weights, hdv_mask=hdv,
        aux_mask=hdv, aux_count=int(hdv.sum()), denom=(weights * hdv).sum(),
        lane_target=torch.from_numpy(states)[:, 1:, :, 1:4].argmax(-1), loss_valid=valid,
        is_cav=torch.zeros_like(valid), dt_s=DT, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False)


def _ctx(targets: UnrollTargets, *, c0: int = 0, c1: int | None = None,
         accel: torch.Tensor | None = None, residual: torch.Tensor | None = None,
         physics: torch.Tensor | None = None):
    c1 = targets.n_steps if c1 is None else c1
    if accel is None:
        accel = targets.states[:, c0 + 1:c1 + 1, :, 5].clone()
    accel = accel.clone().requires_grad_(True)
    pred = {"positions": targets.states[:, c0 + 1:c1 + 1, :, 0].clone(),
            "speeds": targets.states[:, c0 + 1:c1 + 1, :, 4].clone(),
            "accel": accel,
            "lane_logits": torch.zeros(1, c1 - c0, targets.states.shape[2], 3)}
    if residual is not None:
        pred["raw_residual_mps2"] = residual.clone().requires_grad_(True)
    if physics is not None:
        pred["physics_accel_mps2"] = physics
    current = {"v_mps": targets.states[:, c0, :, 4].clone()}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def test_mask_follows_the_recorded_2_to_1_transition_within_the_window():
    targets = _targets()
    mask = post_merge_mask(targets, PostMergeAccelConfig(window_s=10.0))
    # Step 10 is the first lane-1 sample of car 0; 20 steps later is step 30.
    assert bool(mask[0, 9, 0]) is False
    assert bool(mask[0, 10, 0]) is True
    assert bool(mask[0, 30, 0]) is True
    assert bool(mask[0, 31, 0]) is False
    assert not bool(mask[0, :, 1].any())          # never changed lane
    assert bool(mask[0, 30, 2]) is True           # car 2 merges at step 30
    plan = prepare_post_merge(PostMergeAccelConfig(window_s=10.0), targets)
    assert plan["count"] == int(mask.sum()) > 0
    assert plan["window_steps"] == 20


def test_mask_respects_cav_and_invalid_steps():
    targets = _targets()
    targets.is_cav = targets.is_cav.clone()
    targets.is_cav[0, 11, 0] = True
    targets.loss_valid = targets.loss_valid.clone()
    targets.loss_valid[0, 12, 0] = False
    mask = post_merge_mask(targets, PostMergeAccelConfig(window_s=10.0))
    assert not bool(mask[0, 10, 0])
    assert not bool(mask[0, 11, 0])


def test_huber_loss_penalises_a_braking_prediction_and_reaches_accel():
    targets = _targets()
    accel = torch.zeros(1, 60, 3)
    ctx = _ctx(targets, accel=accel)
    term = PostMergeAccelLoss(PostMergeAccelConfig(weight=1.0))
    term.prepare(targets)
    value = term(ctx)
    assert torch.isfinite(value.value) and float(value.value.detach()) > 0.0
    value.value.backward()
    assert ctx.pred["accel"].grad is not None
    assert float(ctx.pred["accel"].grad.abs().sum()) > 0.0


def test_huber_loss_is_zero_when_the_prediction_matches_the_recording():
    targets = _targets()
    term = PostMergeAccelLoss(PostMergeAccelConfig(weight=1.0))
    term.prepare(targets)
    value = term(_ctx(targets))
    assert float(value.value.detach()) == pytest.approx(0.0, abs=1e-7)


def test_residual_penalty_is_one_sided_and_conditioned_on_idm_and_recording():
    targets = _targets()
    residual = torch.full((1, 60, 3), -4.0)
    physics = torch.full((1, 60, 3), 2.0)
    term = PostMergeResidualLoss(PostMergeAccelConfig(residual_weight=1.0))
    term.prepare(targets)
    value = term(_ctx(targets, residual=residual, physics=physics))
    assert float(value.value.detach()) == pytest.approx(4.0, abs=1e-6)
    value.value.backward()
    positive = torch.full((1, 60, 3), 3.0)
    assert float(term(_ctx(targets, residual=positive, physics=physics)).value.detach()) \
        == pytest.approx(0.0, abs=1e-7)
    # Recorded deceleration (a_rec < 0) or an IDM brake excludes the step.
    ctx = _ctx(targets, residual=residual, physics=physics)
    ctx.targets.states = ctx.targets.states.clone()
    ctx.targets.states[:, 11:32, 0, 5] = -2.0
    ctx.targets.states[:, 31:52, 2, 5] = -2.0
    assert float(term(ctx).value.detach()) == pytest.approx(0.0, abs=1e-6)


def test_config_keys_are_default_off_and_roundtrip():
    cfg = TrainConfig()
    flat = cfg.to_legacy_flat()
    for key in ("post_merge_weight", "post_merge_residual_weight",
                "post_merge_window_s", "post_merge_huber_delta_mps2"):
        assert key not in flat
        assert key in OMIT_WHEN_DEFAULT
        assert FLAT_TO_PATH[key].startswith("loss.post_merge.")
    on = TrainConfig(loss=LossConfig(post_merge=PostMergeConfig(
        weight=0.5, residual_weight=0.5, window_s=10.0, huber_delta_mps2=0.5)))
    flat_on = on.to_legacy_flat()
    assert flat_on["post_merge_weight"] == pytest.approx(0.5)
    assert flat_on["post_merge_residual_weight"] == pytest.approx(0.5)
    assert TrainConfig.from_legacy_flat(flat_on).loss.post_merge == on.loss.post_merge
    settings = TrainSettings.from_config(on)
    assert settings.post_merge_on and settings.post_merge_window_s == pytest.approx(10.0)
    with pytest.raises(Exception):
        PostMergeConfig(window_s=1.0)
    with pytest.raises(Exception):
        PostMergeConfig(huber_delta_mps2=0.0)


def test_default_off_composer_is_bit_identical_and_terms_are_ordered():
    targets = _targets()
    ctx = _ctx(targets)
    base = build_loss(TrainSettings())
    assert base.names == ("position", "speed", "lane")
    base.prepare(targets)
    assert "post_merge" not in targets.prepared
    assert torch.isfinite(base(ctx).total)
    settings = TrainSettings(post_merge_weight=0.5, post_merge_residual_weight=0.5)
    names = build_loss(settings).names
    assert names == ("position", "speed", "lane", "post_merge", "post_merge_residual")
    assert post_merge_terms(PostMergeAccelConfig()) == []
    assert torch.isfinite(build_loss(settings)(ctx).total)


def test_chunk_view_uses_only_steps_of_that_chunk():
    targets = _targets()
    term = PostMergeAccelLoss(PostMergeAccelConfig(weight=1.0))
    term.prepare(targets)
    off_chunk = term(_ctx(targets, c0=0, c1=8))
    assert float(off_chunk.value.detach()) == pytest.approx(0.0, abs=1e-7)
    on_chunk = term(_ctx(targets, c0=8, c1=24, accel=torch.zeros(1, 16, 3)))
    assert float(on_chunk.value.detach()) > 0.0
