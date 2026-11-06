"""SLOWRES / astra #3: free-rollout onset-statistics auxiliary (default OFF)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.config.train import TrainConfig
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.compose import build_loss
from wms.train.losses.onset_stats import OnsetStatsConfig, OnsetStatsLoss, onset_stats_terms
from wms.train.settings import TrainSettings

DT = 0.5


def _targets(n: int = 40, k: int = 2) -> UnrollTargets:
    states = np.zeros((1, n + 1, k, 7), dtype=np.float32)
    states[..., 1] = 1.0
    a = np.zeros((1, n, k), dtype=np.float32)
    a[:, 4:8, 0] = -2.0          # one 2 s braking event
    a[:, 12:13, 1] = -2.0        # one one-step pulse
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


def _ctx(targets: UnrollTargets, speeds: torch.Tensor | None = None) -> LossContext:
    recorded = targets.states[:, 1:, :, 4]
    if speeds is None:
        speeds = recorded.clone()
    speeds = speeds.clone().requires_grad_(True)
    pred = {"positions": targets.states[:, 1:, :, 0].clone(), "speeds": speeds,
            "lane_logits": torch.zeros(speeds.shape[0], speeds.shape[1],
                                       speeds.shape[2], 3)}
    return LossContext(pred=pred, current={"v_mps": targets.states[:, 0, :, 4].clone()},
                       targets=targets, c0=0, c1=targets.n_steps)


def test_off_by_default_and_config_round_trip():
    assert not OnsetStatsConfig().enabled
    assert onset_stats_terms(OnsetStatsConfig()) == []
    cfg = TrainConfig.from_legacy_flat({
        "onset_stats_weight": 0.25, "onset_stats_max_duration_s": 6.0})
    flat = cfg.to_legacy_flat()
    assert flat["onset_stats_weight"] == pytest.approx(0.25)
    settings = TrainSettings.from_config(cfg)
    assert settings.onset_stats_on
    assert "onset_stats" in [t.name for t in build_loss(settings).chunk_terms]


def test_matching_prediction_has_zero_loss_and_a_pulse_has_gradient():
    targets = _targets()
    term = OnsetStatsLoss(OnsetStatsConfig(weight=1.0))
    matched = term(_ctx(targets))
    assert float(matched.value.detach()) < 0.05
    speeds = targets.states[:, 1:, :, 4].clone()
    speeds[:, :, 0] -= 2.0  # deeper/longer braking than the recorded chunk
    ctx = _ctx(targets, speeds)
    value = term(ctx)
    assert torch.isfinite(value.value) and float(value.value.detach()) > 0.0
    value.value.backward()
    grad = ctx.pred["speeds"].grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0.0
