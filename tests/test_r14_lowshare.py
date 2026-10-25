"""R14 D3 contracts: low-CAV-share replay weight (synthetic windows, CPU only, no data)."""

import numpy as np
import pytest

from wms.cli.assemble import _wrap_replay
from wms.config.data import ReplayConfig
from wms.config.train import TrainConfig
from wms.data.replay_flags import ReplayRefusal, compute_replay_flags, window_cav_share
from wms.train.replay import ReplayRefusal as ReplayDatasetRefusal
from wms.train.replay import PriorityReplayDataset


def record(n_present=10, n_cav=1, steps=4):
    states = np.zeros((steps, 40, 7), np.float32)
    states[..., 0] = np.arange(40) * 30.0 - 900.0
    states[..., 2] = 1.0                       # lane 1 one-hot
    states[..., 4] = 10.0
    present = np.zeros((steps, 40), bool)
    present[:, :n_present] = True
    is_cav = np.zeros((steps, 40), bool)
    is_cav[:, :n_cav] = True
    return {"states": states, "present": present, "is_cav": is_cav}


class Windows:
    def __init__(self, records, base=None):
        self.records, self.base = records, base
        self.receipt = {"mode": "synthetic"}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]

    def stage_weights(self, stage_s):
        return list(self.base) if self.base is not None else [1.0 / len(self)] * len(self)


def test_config_defaults_validation_and_flat_roundtrip():
    r = ReplayConfig()
    assert r.low_share_oversample == 1.0 and r.low_share_max == 0.15 and r.low_share_min_stage_s == 120.0
    flat = TrainConfig().to_legacy_flat()
    assert not any(k.startswith("replay_low_share") for k in flat)
    cfg = TrainConfig.from_dict({"data": {"replay": {"enabled": True, "low_share_oversample": 1.8,
                                                     "low_share_max": 0.15, "low_share_min_stage_s": 120.0}}})
    assert TrainConfig.from_legacy_flat(cfg.to_legacy_flat()) == cfg
    assert TrainConfig.from_dict(cfg.to_dict()) == cfg
    for bad in ({"low_share_oversample": 0.5}, {"low_share_max": 0.0}, {"low_share_max": 1.0},
                {"low_share_min_stage_s": -1.0}, {"low_share_oversample": float("nan")}):
        with pytest.raises(ValueError, match="low_share"):
            ReplayConfig(**bad)


def test_window_cav_share_first_row_and_fail_closed():
    assert window_cav_share(record(10, 1)) == pytest.approx(0.1)
    r = record(10, 2)
    r["is_cav"][1:] = False                    # only the first row counts
    assert window_cav_share(r) == pytest.approx(0.2)
    with pytest.raises(ReplayRefusal):
        window_cav_share({"present": record()["present"], "is_cav": None})
    empty = record(0, 0)
    with pytest.raises(ReplayRefusal):
        window_cav_share(empty)


def test_flags_off_unchanged_and_on_flags_low_share():
    ds = Windows([record(20, 1), record(20, 3), record(20, 4), record(10, 4)])
    off = compute_replay_flags(ds)
    assert "low_share" not in off and "n_low_share" not in off
    on = compute_replay_flags(ds, low_share_max=0.15)
    assert on["low_share"].tolist() == [True, True, False, False] and on["n_low_share"] == 2
    for k in ("merge_zone", "cav_follower", "brake_event"):
        assert np.array_equal(off[k], on[k])


def test_stage_dependent_multiplier_and_off_identity():
    base = np.array([0.1, 0.2, 0.3, 0.4])
    ds = Windows([record()] * 4, base)
    flags = dict(merge_zone=np.zeros(4, bool), cav_follower=np.zeros(4, bool))
    plain = PriorityReplayDataset(ds, **flags)
    off = PriorityReplayDataset(ds, **flags, low_share=np.array([1, 1, 0, 0], bool), low_share_oversample=1.0)
    assert off.low_share is None
    for stage in (60.0, 120.0, 300.0):
        assert plain.stage_weights(stage) == off.stage_weights(stage)
    w = 1.75
    on = PriorityReplayDataset(ds, **flags, low_share=np.array([1, 1, 0, 0], bool), low_share_oversample=w,
                               low_share_min_stage_s=120.0)
    assert on.stage_weights(60.0) == plain.stage_weights(60.0)
    got = np.array(on.stage_weights(120.0))
    expect = base * np.array([w, w, 1, 1])
    assert np.allclose(got, expect / expect.sum())
    m_low, m_high = base[:2].sum(), base[2:].sum()
    row = on.describe_stages([300.0])[0]
    assert row["low_share_draw_fraction"] == round(w * m_low / (w * m_low + m_high), 4)
    assert row["low_share_draw_fraction_before"] == round(m_low / (m_low + m_high), 4)
    assert on.receipt["replay"]["n_low_share"] == 2
    with pytest.raises(ReplayDatasetRefusal):
        PriorityReplayDataset(ds, **flags, low_share_oversample=2.0)
    with pytest.raises(ReplayDatasetRefusal):
        PriorityReplayDataset(ds, **flags, low_share=np.ones(3, bool), low_share_oversample=2.0)


def test_wrap_replay_wiring_off_and_on():
    ds = Windows([record(20, 1), record(20, 6), record(20, 2), record(20, 8)])
    cfg_off = ReplayConfig(enabled=True)
    r_off = _wrap_replay(ds, cfg_off, (120.0, 300.0))
    assert r_off.low_share is None and "n_low_share" not in r_off.receipt["replay"]["scan"]
    cfg_on = ReplayConfig(enabled=True, low_share_oversample=2.0)
    r_on = _wrap_replay(ds, cfg_on, (120.0, 300.0))
    assert r_on.low_share.tolist() == [True, False, True, False]
    assert r_on.receipt["replay"]["scan"]["n_low_share"] == 2
    stages = r_on.receipt["replay"]["stage_sampling"]
    assert [s["low_share_draw_fraction"] for s in stages] == [round(2 / 3, 4)] * 2
