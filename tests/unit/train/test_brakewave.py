"""V10BRAKEWAVE: HDV brake-event replay flags and follower-chain loss weighting.

Both switches default off.  The tests cover the recorded brake-event scan, the static
replay multiplier, the per-step follower mask and the bit-identical OFF path.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from wms.config.data import DataConfig, ReplayConfig
from wms.config.legacy_flat import FLAT_TO_PATH, OMIT_WHEN_DEFAULT
from wms.config.train import LossConfig, TrainConfig
from wms.data.replay_flags import BRAKE_EVENT_ACCEL_MPS2, window_brake_event
from wms.train.losses.base import LossContext, UnrollTargets
from wms.train.losses.trajectory import (
    PositionLoss, SpeedLoss, brake_follow_mask, brake_follow_step_weights)
from wms.train.replay import PriorityReplayDataset, ReplayRefusal
from wms.train.settings import TrainSettings


def _window(b=1, steps=4, k=2):
    states = np.zeros((b, steps + 1, k, 7), dtype=np.float32)
    states[..., 1] = 1.0
    states[:, :, 0, 0] = 100.0
    states[:, :, 1, 0] = 80.0
    states[..., 4] = 10.0
    states[:, 1:, 0, 5] = -2.0
    states[:, 1:, 1, 5] = -0.5
    present = np.ones((b, steps + 1, k), dtype=bool)
    is_cav = np.zeros((b, steps + 1, k), dtype=bool)
    return {"states": states, "present": present, "is_cav": is_cav}


def _targets(steps=4, k=2):
    rec = _window(steps=steps, k=k)
    states = torch.from_numpy(rec["states"])
    valid = torch.ones(1, steps + 1, k, dtype=torch.bool)
    weights = torch.ones(1, steps, 1)
    hdv = torch.ones(1, steps, k, dtype=torch.bool)
    return UnrollTargets(
        states=states, n_steps=steps, weights=weights, hdv_mask=hdv,
        aux_mask=hdv, aux_count=int(hdv.sum()), denom=(weights * hdv).sum(),
        lane_target=states[:, 1:, :, 1:4].argmax(-1), loss_valid=valid,
        is_cav=torch.zeros_like(valid), dt_s=0.5, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False)


def _ctx(targets, offset=1.0):
    b, n, k = targets.hdv_mask.shape
    pred = {"positions": targets.states[:, 1:, :, 0] + offset,
            "speeds": targets.states[:, 1:, :, 4].clone()}
    return LossContext(pred=pred, current={"lane": torch.zeros(b, k, dtype=torch.long)},
                       targets=targets, c0=0, c1=n)


def test_window_brake_event_detects_hdv_leader_braking_with_follower():
    rec = {key: value[0] for key, value in _window().items()}
    assert window_brake_event(rec)
    assert BRAKE_EVENT_ACCEL_MPS2 == -1.5
    def single():
        return {key: value[0].copy() for key, value in _window().items()}

    no_gap = single()
    no_gap["states"][:, 1, 0] = -20.0
    assert not window_brake_event(no_gap)
    mild = single()
    mild["states"][1:, 0, 5] = -1.0
    assert not window_brake_event(mild)
    other_lane = single()
    other_lane["states"][:, 1, 1:4] = np.array([0.0, 1.0, 0.0], np.float32)
    assert not window_brake_event(other_lane)
    cav_leader = single()
    cav_leader["is_cav"][:, 0] = True
    assert not window_brake_event(cav_leader)


class _TwoWindows:
    receipt: dict = {}

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {}


def test_priority_replay_brake_oversample_composes():
    ds = _TwoWindows()
    merge = np.array([False, False])
    follower = np.array([False, False])
    brake = np.array([True, False])
    replay = PriorityReplayDataset(ds, merge_zone=merge, cav_follower=follower,
                                   brake_event=brake, brake_oversample=4.0)
    weights = np.asarray(replay.stage_weights(10.0))
    assert weights[0] > weights[1]
    assert replay.receipt["replay"]["n_brake_event"] == 1
    off = PriorityReplayDataset(ds, merge_zone=merge, cav_follower=follower,
                                brake_event=brake)
    assert np.allclose(off.stage_weights(10.0), 0.5)
    with pytest.raises(ReplayRefusal):
        PriorityReplayDataset(ds, merge_zone=merge, cav_follower=follower,
                              brake_event=brake, brake_oversample=0.5)


def test_brake_follow_mask_selects_hdv_followers_only():
    targets = _targets()
    mask = brake_follow_mask(targets)
    assert mask.shape == (1, 4, 2)
    assert bool(mask[0, :, 1].all())       # follower behind the braking leader
    assert not bool(mask[0, :, 0].any())   # the leader has no same-lane leader ahead
    multiplier = brake_follow_step_weights(targets, 3.0)
    assert float(multiplier[0, 0, 1]) == pytest.approx(3.0)
    assert float(multiplier[0, 0, 0]) == pytest.approx(1.0)


def test_brake_follow_weight_emphasises_followers_and_can_be_off():
    targets = _targets()
    plain = PositionLoss(1.0)(_ctx(targets))
    weighted = PositionLoss(1.0, brake_follow_weight=3.0)(_ctx(targets))
    assert float(weighted.value) > float(plain.value)
    speed_plain = SpeedLoss(1.0)(_ctx(targets))
    speed_weighted = SpeedLoss(1.0, brake_follow_weight=3.0)(_ctx(targets))
    assert float(speed_weighted.value) >= float(speed_plain.value)
    off_targets = _targets()
    off = PositionLoss(1.0, brake_follow_weight=1.0)(_ctx(off_targets))
    assert torch.equal(off.value, plain.value)
    assert "brake_follow" not in off_targets.prepared


def test_config_keys_default_off_and_roundtrip():
    cfg = TrainConfig()
    flat = cfg.to_legacy_flat()
    assert "brake_follow_weight" not in flat
    assert "replay_brake_oversample" not in flat
    assert "brake_follow_weight" in OMIT_WHEN_DEFAULT
    assert "replay_brake_oversample" in OMIT_WHEN_DEFAULT
    assert FLAT_TO_PATH["brake_follow_weight"] == "loss.brake_follow_weight"
    assert FLAT_TO_PATH["replay_brake_oversample"] == "data.replay.brake_oversample"
    on = TrainConfig(loss=LossConfig(brake_follow_weight=3.0),
                     data=DataConfig(replay=ReplayConfig(enabled=True,
                                                         brake_oversample=5.0)))
    settings = TrainSettings.from_config(on)
    assert settings.brake_follow_weight == pytest.approx(3.0)
    assert settings.replay_brake_oversample == pytest.approx(5.0)
    with pytest.raises(ValueError):
        TrainSettings(brake_follow_weight=0.5)
    with pytest.raises(ValueError):
        ReplayConfig(brake_oversample=0.5)
