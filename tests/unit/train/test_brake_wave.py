"""WAVE: onset-conditioned speed-drop, residual cancellation and spurious pulses.

All terms default off; tests cover the recorded onset/drop scan, the three losses and
their gradients, the chunk-boundary restriction, config round-trip and the OFF path.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT
from wms.config.train import BrakeWaveConfig as ConfigBrakeWave, LossConfig, TrainConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.brake_wave import (
    BrakeWaveConfig, BrakeWaveDropLoss, BrakeWaveResidualLoss, BrakeWaveSpuriousLoss,
    brake_wave_terms, prepare_brake_wave)
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings

DT = 0.5


def _targets(n: int = 40, k: int = 3) -> UnrollTargets:
    states = np.zeros((1, n + 1, k, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[0, 0, :, 0] = np.array([100.0, 80.0, 60.0], dtype=np.float32)
    a = np.zeros((1, n, k), dtype=np.float32)
    a[:, 6:20, 0] = -2.0
    a[:, 6:20, 1] = -1.5
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


def _ctx(targets: UnrollTargets, *, c0: int = 0, c1: int | None = None,
         speeds: torch.Tensor | None = None, residual: torch.Tensor | None = None):
    c1 = targets.n_steps if c1 is None else c1
    recorded = targets.states[:, c0 + 1:c1 + 1, :, 4]
    if speeds is None:
        speeds = recorded.clone()
    speeds = speeds.clone().requires_grad_(True)
    pred = {"positions": targets.states[:, c0 + 1:c1 + 1, :, 0].clone(),
            "speeds": speeds,
            "lane_logits": torch.zeros(speeds.shape[0], speeds.shape[1],
                                       speeds.shape[2], 3)}
    if residual is not None:
        pred["raw_residual_mps2"] = residual
    current = {"v_mps": targets.states[:, c0, :, 4].clone()}
    return LossContext(pred=pred, current=current, targets=targets, c0=c0, c1=c1)


def test_plan_finds_recorded_onsets_and_drop_targets():
    targets = _targets()
    plan = prepare_brake_wave(BrakeWaveConfig(weight=1.0), targets)
    assert plan["count"] == 3
    assert bool(plan["onset"][0, 6].all())
    assert not bool(plan["onset"][0, 7].any())
    # 14 braking steps of -2.0 / -1.5 m/s^2: the leader drops more than 8 m/s
    assert float(plan["drop_target"][0, 6, 0]) >= 8.0
    assert float(plan["drop_target"][0, 6, 1]) >= 6.0
    assert float(plan["weights"][plan["onset"]].mean()) == pytest.approx(1.0, abs=1e-5)


def test_drop_loss_penalises_a_shorter_brake_and_reaches_speeds():
    targets = _targets()
    recorded = targets.states[:, 1:, :, 4]
    shorter = recorded + torch.linspace(0.0, 1.0, recorded.shape[1]).view(1, -1, 1)
    ctx = _ctx(targets, speeds=shorter)
    term = BrakeWaveDropLoss(BrakeWaveConfig(weight=1.0))
    term.prepare(targets)
    value = term(ctx)
    assert torch.isfinite(value.value) and float(value.value.detach()) > 0.0
    value.value.backward()
    assert ctx.pred["speeds"].grad is not None
    assert float(ctx.pred["speeds"].grad.abs().sum()) > 0.0


def test_drop_loss_is_zero_when_the_prediction_matches_the_recording():
    targets = _targets()
    term = BrakeWaveDropLoss(BrakeWaveConfig(weight=1.0))
    term.prepare(targets)
    value = term(_ctx(targets))
    assert float(value.value.detach()) == pytest.approx(0.0, abs=1e-6)


def test_drop_loses_events_outside_the_chunk():
    targets = _targets()
    term = BrakeWaveDropLoss(BrakeWaveConfig(weight=1.0))
    term.prepare(targets)
    ctx = _ctx(targets, c0=0, c1=8)
    assert float(term(ctx).value.detach()) == pytest.approx(0.0, abs=1e-7)


def test_residual_loss_penalises_positive_residual_only_during_recorded_braking():
    targets = _targets()
    residual = torch.zeros(1, 40, 3)
    residual[:, 6:20, 0] = 0.7
    value = BrakeWaveResidualLoss(BrakeWaveConfig(residual_weight=1.0))(
        _ctx(targets, residual=residual.clone().requires_grad_(True)))
    assert torch.isfinite(value.value) and float(value.value.detach()) > 0.0
    value.value.backward()
    negative = -torch.ones(1, 40, 3)
    negative[:, 6:20, 0] = -0.7
    value_neg = BrakeWaveResidualLoss(BrakeWaveConfig(residual_weight=1.0))(
        _ctx(targets, residual=negative))
    assert float(value_neg.value.detach()) == pytest.approx(0.0, abs=1e-7)


def test_spurious_loss_penalises_a_one_step_pulse_on_a_recorded_non_braking_step():
    targets = _targets()
    recorded = targets.states[:, 1:, :, 4]
    pulsed = recorded.clone()
    pulsed[:, 2, 2] -= 0.9          # -1.8 m/s^2 for one 0.5 s step
    pulsed[:, 3, 2] += 0.9          # released next step
    term = BrakeWaveSpuriousLoss(BrakeWaveConfig(spurious_weight=1.0))
    value = term(_ctx(targets, speeds=pulsed))
    assert torch.isfinite(value.value) and float(value.value.detach()) > 0.0
    clean = term(_ctx(targets))
    assert float(clean.value.detach()) == pytest.approx(0.0, abs=1e-7)


def test_config_keys_are_default_off_and_roundtrip():
    cfg = TrainConfig()
    flat = cfg.to_legacy_flat()
    keys = ("brake_wave_weight", "brake_wave_residual_weight", "brake_wave_spurious_weight",
            "brake_wave_window_s", "brake_wave_onset_decel_mps2",
            "brake_wave_spurious_decel_mps2")
    for key in keys:
        assert key not in flat
        assert key in OMIT_WHEN_DEFAULT
        assert FLAT_TO_PATH[key].startswith("loss.brake_wave.")
    on = TrainConfig(loss=LossConfig(brake_wave=ConfigBrakeWave(
        weight=0.5, residual_weight=0.25, spurious_weight=0.5, window_s=3.5,
        onset_decel_mps2=1.0, spurious_decel_mps2=1.5)))
    flat_on = on.to_legacy_flat()
    assert flat_on["brake_wave_weight"] == pytest.approx(0.5)
    assert flat_on["brake_wave_residual_weight"] == pytest.approx(0.25)
    assert flat_on["brake_wave_spurious_weight"] == pytest.approx(0.5)
    assert TrainConfig.from_legacy_flat(flat_on).loss.brake_wave == on.loss.brake_wave
    settings = TrainSettings.from_config(on)
    assert settings.brake_wave_on and settings.brake_wave_weight == pytest.approx(0.5)
    with pytest.raises(Exception):
        ConfigBrakeWave(window_s=2.0)
    with pytest.raises(Exception):
        ConfigBrakeWave(spurious_decel_mps2=0.0)


def test_default_off_composer_is_bit_identical_and_terms_are_ordered():
    targets = _targets()
    ctx = _ctx(targets)
    base = build_loss(TrainSettings())
    assert base.names == ("position", "speed", "lane")
    assert "brake_wave" not in base.names
    base.prepare(targets)
    assert "brake_wave" not in targets.prepared
    assert torch.isfinite(base(ctx).total)
    settings = TrainSettings(brake_wave_weight=0.5, brake_wave_residual_weight=0.25,
                             brake_wave_spurious_weight=0.5)
    names = build_loss(settings).names
    assert names == ("position", "speed", "lane", "brake_wave", "brake_wave_residual",
                     "brake_wave_spurious")
    assert [type(t).__name__ for t in brake_wave_terms(BrakeWaveConfig())] == []
    second = build_loss(settings)(ctx)
    assert torch.isfinite(second.total)
