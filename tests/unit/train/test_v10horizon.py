"""V10HORIZON: plateau-driven horizon curriculum and prioritised mixed replay.

All switches default OFF; the OFF path must be bit-identical to the fixed-step
curriculum.  These tests cover the plateau rule, its trainer integration, the
priority-replay sampler weights and the per-window error probe.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from _compat import legacy
from synth import make_arrays, make_batch, small_model
from wms.train.curriculum import PlateauRule
from wms.train.replay import PriorityReplayDataset, replay_flags
from wms.train.trainer import WmSTrainer

train_mod = legacy("train")


def _flat(tmp_path, **kw):
    from _compat import FlatConfig
    flat = dict(device="cpu", threads=1, truncation_steps=4, batch_size=2, seed=3,
                num_workers=0, epochs_per_stage=1, steps_per_epoch=2000,
                unroll_curriculum_s=[5.0, 10.0], stage_steps=[10, 10], out_dir=str(tmp_path))
    flat.update(kw)
    return FlatConfig(flat)


def _dataset():
    return train_mod.TrajectoryDataset.from_arrays(make_arrays())


# ------------------------------------------------------------------ plateau ---
def test_plateau_rule_advances_on_staleness():
    r = PlateauRule(patience=2, min_steps=1, min_delta=0.01)
    assert not r.observe(1, 10.0)      # first value is always an improvement
    assert not r.observe(2, 9.8)       # > 1 % improvement resets the counter
    assert not r.observe(3, 9.8)       # stale = 1
    assert r.observe(4, 9.8)           # stale = 2 >= patience -> advance
    assert r.advanced and r.reason == "plateau"
    assert r.observations == 4


def test_plateau_rule_min_steps_blocks_premature_advance():
    r = PlateauRule(patience=1, min_steps=5, min_delta=0.0)
    for step in range(1, 5):
        assert not r.observe(step, 10.0)      # stale grows but min_steps not reached
    assert r.observe(5, 10.0)
    assert r.advanced and r.reason == "plateau"


def test_plateau_rule_max_steps_cap():
    r = PlateauRule(patience=99, min_steps=1, max_steps=3, min_delta=0.5)
    assert not r.observe(1, 10.0)
    assert not r.observe(2, 10.0)
    assert r.observe(3, 10.0)
    assert r.advanced and r.reason == "max_steps"


def test_plateau_rule_ignores_non_finite():
    r = PlateauRule(patience=1, min_steps=1)
    assert not r.observe(1, float("nan"))
    assert r.observations == 0 and not r.advanced


def test_plateau_rule_rejects_bad_params():
    with pytest.raises(ValueError):
        PlateauRule(patience=0)
    with pytest.raises(ValueError):
        PlateauRule(min_delta=-1.0)


def test_plateau_trainer_advances_before_budget(tmp_path):
    cfg = _flat(tmp_path, plateau_advance=True, plateau_patience=1, plateau_min_steps=1,
                plateau_min_delta=1.0, val_every=1, select_by_long_horizon=True)
    ds = _dataset()
    trainer = WmSTrainer(cfg, small_model(), ds, val_dataset=ds, out_dir=tmp_path)
    result = trainer.fit()
    rows = [json.loads(x) for x in (tmp_path / "val_log.jsonl").read_text().splitlines()]
    plateau_rows = [r for r in rows if r.get("plateau_advanced")]
    assert plateau_rows, "the plateau rule never advanced a stage"
    # every curriculum stage recorded a plateau decision
    stage_rows = [r for r in rows if "plateau" in r]
    assert stage_rows, "no stage record carried the plateau summary"
    assert all("curriculum_advance" in r for r in stage_rows)
    assert result.receipt["config_fingerprint"] == cfg.fingerprint()


def test_plateau_off_is_fixed_budget_and_byte_identical_keys(tmp_path):
    cfg = _flat(tmp_path)
    flat = cfg.to_legacy_flat()
    assert "plateau_advance" not in flat
    trainer = WmSTrainer(cfg, small_model(), _dataset(), out_dir=tmp_path)
    result = trainer.fit()
    hist = result.history
    assert all("plateau" not in h for h in hist)
    # the fixed recipe ran exactly the two stage budgets
    assert len(hist) == 20


# ------------------------------------------------------------------- replay ---
def _flags(n=3, steps=24, k=6):
    arrays = make_arrays(steps=steps, k=k, n=n)
    return arrays


def test_replay_flags_detect_lane2_and_cav_follower():
    base = make_arrays(n=1)
    rec = {key: base[key][0].copy() for key in base}
    rec["states"][:, 3, 1:4] = np.array([0.0, 0.0, 1.0], np.float32)  # a ramp slot in lane 2
    is_cav = np.zeros(rec["present"].shape, dtype=bool)
    is_cav[:, 0] = True                    # the front vehicle is a CAV with an HDV behind it
    rec["is_cav"] = is_cav
    flags = replay_flags([rec])
    assert flags["merge_zone"].shape == (1,)
    assert flags["cav_follower"].shape == (1,)
    assert bool(flags["merge_zone"][0]), "a lane-2 ramp slot should flag the window"
    assert bool(flags["cav_follower"][0]), "an HDV behind the middle CAV should flag it"


def test_replay_flags_plain_single_lane_has_no_merge_zone():
    base = make_arrays(n=1)
    rec = {key: base[key][0].copy() for key in base}
    rec["states"][..., 1:4] = np.array([1.0, 0.0, 0.0], np.float32)  # everyone in lane 0
    flags = replay_flags([rec])
    assert not bool(flags["merge_zone"][0])


def test_priority_replay_weights_compose(tmp_path):
    from output.wd01.wm_n1.wms.train import TrajectoryDataset

    ds = TrajectoryDataset.from_arrays(make_arrays(n=4))
    merge = np.array([True, False, False, True])
    follower = np.array([False, True, False, True])
    replay = PriorityReplayDataset(ds, merge_zone=merge, cav_follower=follower,
                                   merge_oversample=3.0, cav_oversample=2.0)
    w = np.asarray(replay.stage_weights(10.0))
    assert w.shape == (4,)
    assert w[3] > w[0] > w[1]        # both flags > merge only > neither
    # high-error windows are boosted above their static weight
    assert np.allclose(replay.error_multiplier(), 1.0)
    replay.update_errors(np.array([1, 2]), np.array([100.0, 1.0]), decay=0.0)
    replay.error_alpha = 1.0
    em = replay.error_multiplier()
    assert em[1] > 1.0            # above-average error window is oversampled
    assert em[2] == pytest.approx(1.0)   # below-average stays at its static weight
    assert em[0] == pytest.approx(1.0) and em[3] == pytest.approx(1.0)


def test_priority_replay_receipt_and_getitem(tmp_path):
    from output.wd01.wm_n1.wms.train import TrajectoryDataset

    ds = TrajectoryDataset.from_arrays(make_arrays(n=2))
    replay = PriorityReplayDataset(ds, merge_zone=np.array([True, False]),
                                   cav_follower=np.array([False, True]),
                                   merge_oversample=2.0, cav_oversample=2.0)
    rec = replay[0]
    assert rec["states"].shape == ds[0]["states"].shape
    assert replay.receipt["replay"]["n_merge_zone"] == 1
    assert replay.receipt["replay"]["n_cav_follower"] == 1
    assert len(replay) == 2


def test_trainer_replay_stage_probe_updates_priorities(tmp_path):
    base = _dataset()
    replay = PriorityReplayDataset(base, merge_zone=np.array([True, False]),
                                   cav_follower=np.array([False, True]),
                                   merge_oversample=2.0, cav_oversample=2.0)
    cfg = _flat(tmp_path, replay_enabled=True, replay_probe_windows=2, replay_refresh_every=2)
    trainer = WmSTrainer(cfg, small_model(), replay, val_dataset=base, out_dir=tmp_path)
    trainer.fit()
    assert replay.receipt["replay"]["error_weighted_windows"] >= 1
    rows = [json.loads(x) for x in (tmp_path / "val_log.jsonl").read_text().splitlines()]
    assert any("replay" in r for r in rows)


def test_batch_size_per_stage_plumbing(tmp_path):
    from wms.train.settings import TrainSettings

    flat = _flat(tmp_path, batch_size_per_stage=[2, 1])
    assert TrainSettings.from_flat(flat.to_legacy_flat()).batch_size_per_stage == (2, 1)
    with pytest.raises(ValueError):
        TrainSettings(batch_size_per_stage=(1,))
    trainer = WmSTrainer(_flat(tmp_path, batch_size_per_stage=[2, 1]), small_model(),
                         _dataset(), out_dir=tmp_path)
    assert len(trainer.fit().history) == 20


def test_v10horizon_configs_load_and_carry_the_switches():
    from pathlib import Path

    from wms.config.load import load_train_config
    from wms.train.settings import TrainSettings

    root = Path(__file__).resolve().parents[3]
    full = TrainSettings.from_config(load_train_config(root / "configs" / "c4_wm_v10horizon.toml"))
    assert full.plateau_advance and full.replay_enabled
    assert full.unroll_curriculum_s == (10.0, 30.0, 60.0, 120.0)
    assert full.batch_size_per_stage == (8, 6, 4, 2)
    smoke = TrainSettings.from_config(
        load_train_config(root / "configs" / "c4_wm_v10horizon_smoke.toml"))
    assert smoke.plateau_advance and smoke.replay_enabled
    assert smoke.stage_steps == (6, 6, 6, 6)


def test_per_window_error_probe_returns_one_value_per_window(tmp_path):
    cfg = _flat(tmp_path)
    ds = _dataset()
    trainer = WmSTrainer(cfg, small_model(), ds, out_dir=tmp_path)
    batch = make_batch()
    errors = trainer.runner.window_errors(batch, unroll_steps=8)
    assert len(errors) == int(batch["states"].shape[0])
    assert all(np.isfinite(e) for e in errors)
