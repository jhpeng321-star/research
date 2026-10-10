"""A11 full-horizon (long) plan input at scoring time.

Synthetic fakes only, plus one opt-in integration test that loads the real
TRAIN-only A11 dry-run checkpoint when it is present on this box.  No
CAL/HOLDOUT/chunk-5/J4B/INC1 data is opened and no real scoring run happens.

The long plan vector is the SAME treatment-arm CAV-motion summary training
builds (``train/idea_a.py`` A11 block) through
``model.paired_effect.long_plan_summary``; the scorer must reproduce it exactly
and pass it as ``plan_long`` to the plan-aware head.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evalh.protocol import N_QUANTILES
from evalh.scorer import predictors as P

from model import constants as MC

HORIZON_S = 120.0
K_SLOTS = 3
N_TICKS = 320
TICK = 4
IS_CAV_IDX = 7

#: One registered long-plan width for H=120: 12 bins x 5 features.
LONG_F = 60

#: The real TRAIN-only A11 dry run and a local TRAIN climatology copy (the
#: recorded path is also present on this box; the env override is set anyway).
DRY_CKPT = Path("E:/rd-wt/a11/_out/dry_a11m/member_1.pt")
TRAIN_CLIMATOLOGY = (
    "E:/research-dense/runs/active/WM-N1-SCORE-A9-20261005/TRAIN_CLIMATOLOGY.json")


def _synthetic_arrays():
    rng = np.random.default_rng(20261005)
    kin = np.zeros((N_TICKS, K_SLOTS, 7), np.float32)
    kin[:, :, 0] = (np.arange(N_TICKS)[:, None] * 2.0
                    + np.arange(K_SLOTS)[None, :] * 15.0)
    kin[:, :, 1] = 1.0
    kin[:, :, 4] = 12.0 + rng.random((N_TICKS, K_SLOTS)).astype(np.float32)
    kin[:, :, 5] = rng.normal(0, 0.3, (N_TICKS, K_SLOTS)).astype(np.float32)
    mask = np.ones((N_TICKS, K_SLOTS), np.float32)
    target = np.ones((N_TICKS, K_SLOTS), np.float32)
    step = np.zeros((N_TICKS, K_SLOTS, 16), np.float32)
    step[:, 0, IS_CAV_IDX] = 1.0
    slot = np.tile(np.arange(K_SLOTS, dtype=np.int64), (N_TICKS, 1))
    counter = np.zeros((N_TICKS, 2, 30), np.float64)
    counter[..., 0] = 1.0
    plan = np.zeros((N_TICKS, K_SLOTS, 8), np.float32)
    plan[:, :, 0] = 12.0                        # planned speed (unscaled)
    plan[:, :, 1] = 0.4                         # planned accel (unscaled)
    return dict(veh_kin=kin, veh_mask=mask, target_mask=target, veh_step=step,
                slot_index=slot, counter=counter, cav_plan_ch=plan)


class _FakeSource:
    """Minimal released-shard stand-in: ``_checked_file`` -> a pinned .npy."""

    def __init__(self, arrays, tmp_path):
        self._dir = Path(tmp_path)
        for name, value in arrays.items():
            np.save(self._dir / f"{name}.npy", value)
        self.selected = []

    def _checked_file(self, item, name):
        return str(self._dir / name)


def _item_run():
    return {"tick": TICK}, {"tick_start": 0}


def _window(source, arrays):
    item, run = _item_run()
    is_cav = arrays["veh_step"][TICK, :, IS_CAV_IDX] > 0.5
    return P.ItemWorld(
        item=item, run=run, local_t0_tick=TICK, world=None,
        counter_t0=np.asarray(arrays["counter"][TICK], np.float64),
        is_cav_t0=is_cav)


# ------------------------------------------------------- vector parity -------
def test_training_plan_long_shape_values_and_determinism(tmp_path):
    from features.idea_a import CAV_ACCEL_SCALE, CAV_TGT_SPEED_SCALE
    from model.paired_effect import long_plan_raw_f

    arrays = _synthetic_arrays()
    source = _FakeSource(arrays, tmp_path)
    window = _window(source, arrays)
    vec = P.training_plan_long(source, window.item, window.run,
                               horizon_s=HORIZON_S, is_cav=window.is_cav_t0)
    assert tuple(vec.shape) == (1, long_plan_raw_f(HORIZON_S))
    assert int(vec.shape[-1]) == LONG_F
    # Only slot 0 is a CAV; bin 0 features: mean speed, min speed, mean accel,
    # decel fraction, mean CAV count per tick.
    assert vec[0, 0].item() == pytest.approx(12.0 * float(CAV_TGT_SPEED_SCALE),
                                             rel=1e-5)
    assert vec[0, 1].item() == pytest.approx(12.0 * float(CAV_TGT_SPEED_SCALE),
                                             rel=1e-5)
    assert vec[0, 2].item() == pytest.approx(0.4 * float(CAV_ACCEL_SCALE),
                                             rel=1e-5)
    assert vec[0, 3].item() == pytest.approx(0.0)
    assert vec[0, 4].item() == pytest.approx(1.0)
    again = P.training_plan_long(source, window.item, window.run,
                                 horizon_s=HORIZON_S, is_cav=window.is_cav_t0)
    assert torch.equal(vec, again)


def test_training_plan_long_refuses_missing_channels(tmp_path):
    arrays = _synthetic_arrays()
    source = _FakeSource(arrays, tmp_path)
    window = _window(source, arrays)
    (Path(tmp_path) / "cav_plan_ch.npy").unlink()
    with pytest.raises(P.PredictorRefusal, match="cav_plan_ch"):
        P.training_plan_long(source, window.item, window.run,
                             horizon_s=HORIZON_S, is_cav=window.is_cav_t0)


def test_bundle_plan_horizon_s_reads_and_refuses():
    good = SimpleNamespace(config=SimpleNamespace(
        paired_effect_plan_horizon_s=120.0))
    assert P._bundle_plan_horizon_s(good) == 120.0
    assert P._bundle_plan_horizon_s(SimpleNamespace(config=SimpleNamespace())) \
        is None
    bad = SimpleNamespace(config=SimpleNamespace(paired_effect_plan_horizon_s=0))
    with pytest.raises(P.PredictorRefusal):
        P._bundle_plan_horizon_s(bad)


# ------------------------------------------------------- long-head wiring ----
def _fake_window():
    return SimpleNamespace(item={}, run={}, is_cav_t0=np.zeros(K_SLOTS, bool),
                           counter_t0=np.zeros((2, 30)))


def test_plan_aware_long_head_receives_plan_long(monkeypatch):
    calls = {}

    def fake_helper(bundle, *, t0_kin, t0_mask, t0_counter, summary_raw,
                    plan_long=None):
        calls["plan"] = plan_long
        calls["summary"] = summary_raw
        return np.stack([np.full(N_QUANTILES, 3.0 + i)
                         for i in range(len(MC.TAUS))])

    sentinel = torch.arange(LONG_F, dtype=torch.float32).reshape(1, LONG_F)
    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: fake_helper)
    monkeypatch.setattr(P, "training_summary_raw", lambda *a, **k: "SUMMARY")
    monkeypatch.setattr(P, "training_plan_long", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        P, "_t0_state_channels",
        lambda s, w: (torch.zeros(1, K_SLOTS, 7),
                      torch.zeros(1, K_SLOTS, dtype=torch.bool),
                      torch.zeros(1, 2, 30)))

    head = torch.nn.Linear(1, 1)
    head.plan_raw_f = LONG_F
    cfg = SimpleNamespace(paired_effect_plan_aware=True,
                          paired_effect_plan_horizon_s=HORIZON_S)
    bundle = SimpleNamespace(paired_effect=head, state_enc=None, config=cfg)
    pred = P.ModelAPredictor(bundle, source=SimpleNamespace(),
                             cav_motion="teacher")
    assert pred.plan_aware is True
    dv = pred.paired_head_arm_dv(_fake_window(), 60.0)
    assert dv.shape == (N_QUANTILES,)
    assert calls["summary"] == "SUMMARY"
    assert calls["plan"] is sentinel


def test_long_head_without_recorded_horizon_refuses(monkeypatch):
    def fake_helper(bundle, *, t0_kin, t0_mask, t0_counter, summary_raw,
                    plan_long=None):
        raise AssertionError("must refuse before calling the head")

    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: fake_helper)
    monkeypatch.setattr(P, "training_summary_raw", lambda *a, **k: "S")
    monkeypatch.setattr(
        P, "_t0_state_channels",
        lambda s, w: (torch.zeros(1, K_SLOTS, 7),
                      torch.zeros(1, K_SLOTS, dtype=torch.bool),
                      torch.zeros(1, 2, 30)))
    head = torch.nn.Linear(1, 1)
    head.plan_raw_f = LONG_F
    cfg = SimpleNamespace(paired_effect_plan_aware=True,
                          paired_effect_plan_horizon_s=None)
    bundle = SimpleNamespace(paired_effect=head, state_enc=None, config=cfg)
    pred = P.ModelAPredictor(bundle, source=SimpleNamespace(),
                             cav_motion="teacher")
    with pytest.raises(P.PredictorRefusal, match="plan_horizon_s"):
        pred.paired_head_arm_dv(_fake_window(), 60.0)


def test_short_head_still_passes_summary_only(monkeypatch):
    calls = {}

    def fake_helper(bundle, *, t0_kin, t0_mask, t0_counter, summary_raw,
                    plan_long=None):
        calls["plan"] = plan_long
        return np.stack([np.full(N_QUANTILES, 1.0 + i)
                         for i in range(len(MC.TAUS))])

    monkeypatch.setattr(P, "_plan_aware_head_helper", lambda: fake_helper)
    monkeypatch.setattr(P, "training_summary_raw", lambda *a, **k: "SHORT")
    monkeypatch.setattr(
        P, "_t0_state_channels",
        lambda s, w: (torch.zeros(1, K_SLOTS, 7),
                      torch.zeros(1, K_SLOTS, dtype=torch.bool),
                      torch.zeros(1, 2, 30)))
    head = torch.nn.Linear(1, 1)          # no plan_raw_f -> historical 16-D
    cfg = SimpleNamespace(paired_effect_plan_aware=True,
                          paired_effect_plan_horizon_s=None)
    bundle = SimpleNamespace(paired_effect=head, state_enc=None, config=cfg)
    pred = P.ModelAPredictor(bundle, source=SimpleNamespace(),
                             cav_motion="teacher")
    dv = pred.paired_head_arm_dv(_fake_window(), 60.0)
    assert dv.shape == (N_QUANTILES,)
    assert calls["plan"] is None


# ------------------------------------------------- real dry-run integration --
@pytest.mark.skipif(not DRY_CKPT.is_file(),
                    reason="A11 TRAIN-only dry-run checkpoint not present")
def test_real_a11_dry_run_checkpoint_long_plan_path(tmp_path, monkeypatch):
    monkeypatch.setenv("WM_N1_PAIRED_EFFECT_CLIMATOLOGY", TRAIN_CLIMATOLOGY)
    from model.paired_effect import long_plan_raw_f
    from wm_api.checkpoint import load_checkpoint

    loaded = load_checkpoint(str(DRY_CKPT.parent), verify_sha=False)
    bundle = loaded.members[0].bundle
    head = bundle.paired_effect
    horizon = bundle.config.paired_effect_plan_horizon_s
    assert horizon == HORIZON_S
    assert int(head.plan_raw_f) == long_plan_raw_f(horizon)
    assert int(head.plan_raw_f) == LONG_F

    arrays = _synthetic_arrays()
    source = _FakeSource(arrays, tmp_path)
    window = _window(source, arrays)
    pred = P.ModelAPredictor(bundle, source=source, cav_motion="teacher")
    assert pred.plan_aware is True
    dv = pred.paired_head_arm_dv(window, 60.0)
    assert dv.shape == (N_QUANTILES,)
    assert np.isfinite(dv).all()
