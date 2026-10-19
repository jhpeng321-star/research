"""BRAKE_DIAG V10PERSIST: onset persistence, quantile proxies and event balancing.

All terms are default off; the tests cover the recorded onset scan, the multi-step
speed-change error, the pinball proxies, inverse-frequency event weights, the
chunk-boundary restriction and the bit-identical OFF path.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT
from wms.config.train import BrakePersistConfig, LossConfig, TrainConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.brake_persist import (
    BrakePersistConfig as TermConfig, BrakePersistLoss, BrakePersistQuantileLoss,
    brake_persist_terms, onset_mask, prepare_brake_persist)
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings

DT = 0.5


def _targets(n: int = 40, k: int = 3) -> UnrollTargets:
    states = np.zeros((1, n + 1, k, 7), dtype=np.float32)
    states[..., 1] = 1.0                       # all vehicles in lane 0
    # vehicles 0/1/2 are spaced 20 m apart, leader first
    states[0, 0, :, 0] = np.array([100.0, 80.0, 60.0], dtype=np.float32)
    a = np.zeros((1, n, k), dtype=np.float32)
    a[:, 6:20, 0] = -1.8                       # hard leader braking
    a[:, 6:20, 1] = -1.5                       # follower onset at the same step
    a[:, 6:20, 2] = -1.5
    v = np.zeros((1, n + 1, k), dtype=np.float32)
    v[0, 0] = 20.0
    s = np.zeros((1, n + 1, k), dtype=np.float32)
    s[0, 0] = states[0, 0, :, 0]
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


def _ctx(targets: UnrollTargets, c0: int | None = None, c1: int | None = None,
         offset: float = -0.3):
    c0 = 0 if c0 is None else c0
    c1 = targets.n_steps if c1 is None else c1
    recorded = targets.states[:, c0 + 1:c1 + 1, :, 4]
    drift = torch.sin(torch.arange(c1 - c0, dtype=recorded.dtype)).view(1, -1, 1)
    speeds = (recorded + 0.3 * drift).clone().requires_grad_(True)
    pred = {"positions": targets.states[:, c0 + 1:c1 + 1, :, 0].clone(),
            "speeds": speeds,
            "lane_logits": torch.zeros(speeds.shape[0], speeds.shape[1],
                                       speeds.shape[2], 3)}
    current = {"v_mps": targets.states[:, c0, :, 4].clone() + offset}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def test_onset_mask_finds_only_fresh_braking_after_one_second():
    targets = _targets()
    onset = onset_mask(targets)
    assert onset.shape == (1, 40, 3)
    assert int(onset.sum()) == 3
    assert bool(onset[0, 6].all())
    assert not bool(onset[0, 5].any())
    assert not bool(onset[0, 7].any())       # still braking: not a fresh onset


def test_persist_loss_is_finite_positive_and_reaches_speeds():
    targets = _targets()
    ctx = _ctx(targets)
    term = BrakePersistLoss(TermConfig(weight=1.0))
    term.prepare(targets)
    value = term(ctx)
    assert torch.isfinite(value.value)
    assert float(value.value.detach()) > 0.0
    value.value.backward()
    assert ctx.pred["speeds"].grad is not None
    assert float(ctx.pred["speeds"].grad.abs().sum()) > 0.0


def test_quantile_proxies_are_finite_and_differentiable():
    targets = _targets()
    ctx = _ctx(targets)
    term = BrakePersistQuantileLoss(TermConfig(quantile_weight=1.0))
    term.prepare(targets)
    value = term(ctx)
    assert torch.isfinite(value.value)
    assert float(value.value.detach()) > 0.0
    value.value.backward()
    assert ctx.pred["speeds"].grad is not None
    assert float(ctx.pred["speeds"].grad.abs().sum()) > 0.0


def test_events_are_restricted_to_windows_inside_the_chunk():
    targets = _targets()
    full = _ctx(targets)
    term = BrakePersistLoss(TermConfig(weight=1.0))
    term.prepare(targets)
    assert float(term(full).value.detach()) > 0.0
    # onset at step 6 needs steps 6..14; a chunk ending at 8 cannot carry it
    short = _ctx(targets, c0=0, c1=8)
    assert float(term(short).value.detach()) == pytest.approx(0.0)


def test_event_balance_weights_are_inverse_frequency_and_mean_one():
    targets = _targets()
    config = TermConfig(weight=1.0, event_balance=1.0)
    plan = prepare_brake_persist(config, targets)
    weight = plan["weights"][plan["onset"]]
    assert float(weight.mean()) == pytest.approx(1.0, abs=1e-6)
    assert plan["count"] == 3 and plan["n_cells"] >= 2
    # vehicle 0 has no leader (rare cell) -> higher weight than the two followers
    assert float(plan["weights"][0, 6, 0]) > float(plan["weights"][0, 6, 1])
    off = prepare_brake_persist(TermConfig(weight=1.0, event_balance=0.0), targets)
    assert torch.allclose(off["weights"], torch.ones_like(off["weights"]))


def test_config_keys_are_default_off_and_roundtrip():
    cfg = TrainConfig()
    flat = cfg.to_legacy_flat()
    for key in ("brake_persist_weight", "brake_persist_quantile_weight",
                "brake_persist_event_balance", "brake_persist_window_s"):
        assert key not in flat
        assert key in OMIT_WHEN_DEFAULT
    assert FLAT_TO_PATH["brake_persist_weight"] == "loss.brake_persist.weight"
    assert FLAT_TO_PATH["brake_persist_quantile_weight"] == \
        "loss.brake_persist.quantile_weight"
    assert FLAT_TO_PATH["brake_persist_event_balance"] == \
        "loss.brake_persist.event_balance"
    on = TrainConfig(loss=LossConfig(brake_persist=BrakePersistConfig(
        weight=0.5, quantile_weight=0.25, event_balance=1.0, window_s=3.5)))
    flat_on = on.to_legacy_flat()
    assert flat_on["brake_persist_weight"] == pytest.approx(0.5)
    assert flat_on["brake_persist_quantile_weight"] == pytest.approx(0.25)
    assert flat_on["brake_persist_event_balance"] == pytest.approx(1.0)
    assert flat_on["brake_persist_window_s"] == pytest.approx(3.5)
    back = TrainConfig.from_legacy_flat(flat_on)
    assert back.loss.brake_persist == on.loss.brake_persist
    settings = TrainSettings.from_config(on)
    assert settings.brake_persist_weight == pytest.approx(0.5)
    assert settings.brake_persist_quantile_weight == pytest.approx(0.25)
    assert settings.brake_persist_on
    with pytest.raises(Exception):
        BrakePersistConfig(window_s=2.0)
    with pytest.raises(Exception):
        BrakePersistConfig(event_balance=5.0)


def test_default_off_composer_is_bit_identical():
    targets = _targets()
    ctx = _ctx(targets)
    base = build_loss(TrainSettings())
    assert base.names == ("position", "speed", "lane")
    assert "brake_persist" not in base.names
    balanced_only = build_loss(TrainSettings(brake_persist_event_balance=1.0))
    assert balanced_only.names == base.names
    base.prepare(targets)
    balanced_only.prepare(targets)
    first = base(ctx)
    second = balanced_only(ctx)
    assert torch.equal(first.total, second.total)
    assert "brake_persist" not in targets.prepared


def test_enabled_terms_are_built_in_order():
    settings = TrainSettings(brake_persist_weight=0.5,
                             brake_persist_quantile_weight=0.25)
    names = build_loss(settings).names
    assert names == ("position", "speed", "lane", "brake_persist",
                     "brake_persist_quantile")
    assert [type(t).__name__ for t in brake_persist_terms(TermConfig())] == []
