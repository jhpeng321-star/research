"""Unit tests of the train layer: settings, curriculum, selection, gates, preflight, trainer."""
from __future__ import annotations

import json

import pytest
import torch

from _compat import FlatConfig, legacy
from synth import make_arrays, make_batch, small_model
from wms.train import curriculum, gates, preflight, selection
from wms.train.losses.compose import build_loss
from wms.train.losses.lane_rate import lane_rate_loss
from wms.train.settings import TrainSettings
from wms.train.trainer import HardCapRefused, WmSTrainer

train_mod = legacy("train")


def _cfg(tmp_path, **kw):
    base = dict(device="cpu", threads=1, truncation_steps=8, batch_size=2, seed=3,
                unroll_curriculum_s=(5.0, 10.0), stage_steps=(1, 1), out_dir=str(tmp_path))
    base.update(kw)
    return FlatConfig.from_legacy(train_mod.WmSTrainConfig(**base))


def _dataset():
    return train_mod.TrajectoryDataset.from_arrays(make_arrays())


def test_settings_roundtrip_and_validation():
    flat = train_mod.WmSTrainConfig(asym_speed_alpha=2.0).to_dict()
    assert TrainSettings.from_flat(flat).asym_speed_alpha == 2.0
    with pytest.raises(ValueError, match="unknown"):
        TrainSettings.from_flat({"nope": 1})
    with pytest.raises(ValueError):
        TrainSettings(stage_steps=(1,))
    assert TrainSettings(residual_cap_mps2=1.5).uses_hard_cap
    assert not TrainSettings(residual_soft_bound_mps2=1.5).uses_hard_cap


def test_hard_cap_refused_for_new_runs_but_replayable(tmp_path):
    cfg = _cfg(tmp_path, residual_cap_mps2=1.5)
    with pytest.raises(HardCapRefused):
        WmSTrainer(cfg, small_model(), _dataset(), out_dir=tmp_path)
    trainer = WmSTrainer(cfg, small_model(), _dataset(), out_dir=tmp_path,
                         allow_rejected_hard_cap=True)
    assert trainer.model.residual_cap_mps2 == 1.5


def test_composer_term_order_and_enabling():
    names = build_loss(TrainSettings(), physics=small_model().physics).names
    assert names == ("position", "speed", "lane")
    r3 = TrainSettings(rollout_mode="hybrid", lane_head="history", residual_mag_weight=0.05,
                       lane_rate_weight=0.1, asym_speed_alpha=2.0, decel_weight=5.0,
                       propagation_weight=1.0, stability_weight=0.1, conservation_weight=0.1)
    assert build_loss(r3, physics=small_model().physics).names == (
        "position", "speed", "lane", "residual_mag", "lane_rate", "asym_speed", "decel_weight",
        "propagation", "residual_penalty", "conservation", "stability")


def test_lane_rate_chunk_derivative_matches_whole_batch_gradient():
    p = torch.rand(3, 5, requires_grad=True)
    valid = torch.ones(3, 5, dtype=torch.bool)
    lane_rate_loss(p, valid, 0.1, 0.0123).backward()
    mean = p.detach().mean()
    expected = 2 * 0.1 * (mean - 0.0123) / p.numel()
    torch.testing.assert_close(p.grad, torch.full_like(p, float(expected)))


def test_curriculum_plan_and_projection(tmp_path):
    s = TrainSettings(stage_steps=(2500, 2500, 1875, 1250))
    plans = curriculum.stage_plan(s, 0.5)
    assert [p.unroll_steps for p in plans] == [20, 60, 120, 240]
    assert [p.steps for p in plans] == [2500, 2500, 1875, 1250]
    log = tmp_path / "log.jsonl"
    log.write_text("".join(json.dumps({"stage_s": st, "seconds": 1.0, "unroll_steps": 20}) + "\n"
                           for st in (10.0, 30.0, 60.0, 120.0) for _ in range(3)))
    info = curriculum.project_hours(log, s.stage_steps)
    assert info["projected_hours"] == pytest.approx(8125 / 3600)
    steps, _ = curriculum.plan_from_log(log, 3600.0)
    assert len(steps) == 4 and steps[0] >= steps[-1]


def test_long_horizon_selection_policy():
    sel = selection.LongHorizonSelection()
    rec = lambda h, s, stage=120.0: dict(horizon_s=h, mean_position=s, stage_s=stage,  # noqa: E731
                                         valid_targets=5)
    assert not sel.consider(rec(30.0, 1.0, 30.0))
    assert sel.consider(rec(60.0, 5.0))
    assert not sel.consider(rec(60.0, 6.0))
    assert sel.consider(rec(60.0, 4.0))
    assert sel.consider(rec(120.0, 9.0))
    assert not sel.consider(rec(60.0, 0.1))
    assert selection.validation_horizons(120.0) == (60.0, 120.0)
    assert selection.validation_horizons(30.0) == (30.0,)


def test_fit_logs_residual_size_and_lane_change_rate(tmp_path):
    cfg = _cfg(tmp_path, select_by_long_horizon=True, val_every=1, lane_rate_weight=0.1,
               unroll_curriculum_s=(30.0,), stage_steps=(1,))
    ds = _dataset()
    trainer = WmSTrainer(cfg, small_model(), ds, val_dataset=ds, out_dir=tmp_path)
    result = trainer.fit()
    row = json.loads((tmp_path / "train_log.jsonl").read_text().splitlines()[0])
    for key in ("raw_residual_abs_mean_mps2", "predicted_lane_change_rate", "lane_rate"):
        assert key in row
    assert result.receipt["config_fingerprint"] == cfg.fingerprint()
    assert result.receipt["long_horizon_selection"]["enabled"]


def test_gate_passes_and_fails_closed(tmp_path):
    cfg = _cfg(tmp_path / "unused")
    with pytest.raises(gates.GateFailed, match="real data path"):
        gates.run_gate("stages", cfg, tmp_path / "g", model=small_model(), dataset=_dataset())
    res = gates.run_gate("stages", cfg, tmp_path / "g2", model=small_model(), dataset=_dataset(),
                         allow_fixture=True)
    assert res.passed, res.failures
    assert res.records == 24 and sorted(res.loss_share) == [10.0, 30.0, 60.0, 120.0]
    res.require()


def test_gate_check_log_detects_nonfinite_and_dominance(tmp_path):
    spec = gates.GateSpec(stages_s=(10.0,), steps_per_stage=1, require_loss_share=False)
    good = dict(step=1, stage_s=10.0, finite=True, gate_monotone=True, gate_g0=0.0,
                valid_targets=3, gate_curve=[0.0, 0.1], seconds=1.0, position=1.0, speed=1.0,
                lane=1.0)
    path = tmp_path / "train_log.jsonl"
    path.write_text(json.dumps(good) + "\n")
    assert gates.check_log(path, spec, ("position", "speed", "lane")).passed
    bad = dict(good, lane=1000.0)
    path.write_text(json.dumps(bad) + "\n")
    assert any("exceeds" in f for f in gates.check_log(path, spec, ("position", "speed", "lane")).failures)
    path.write_text(json.dumps(dict(good, speed=float("nan"))) + "\n")
    assert not gates.check_log(path, spec, ("position", "speed", "lane")).passed


def test_preflight_checks(tmp_path):
    cfg = _cfg(tmp_path / "fresh", stage_steps=(2500, 2500))
    assert preflight.preflight(cfg, out_dir=tmp_path / "fresh").ok
    bad = preflight.preflight(cfg, out_dir=tmp_path, inputs=[tmp_path / "HOLDOUT" / "x.json"])
    assert any("output exists" in f for f in bad.failures)
    assert any("restricted" in f for f in bad.failures)
    script = tmp_path / "a.sh"
    script.write_bytes(b"echo\r\n")
    assert any("CR bytes" in f for f in preflight.preflight(cfg, scripts=[script]).failures)
    capped = _cfg(tmp_path / "x", residual_cap_mps2=1.5)
    assert any("REJECTED" in f for f in preflight.preflight(capped).failures)


def test_dry_run_two_updates(tmp_path):
    receipt = preflight.dry_run(_cfg(tmp_path), small_model(), _dataset(), out_dir=tmp_path / "dry")
    assert receipt["status"] == "passed" and len(receipt["updates"]) == 2


def test_trainer_hybrid_identity_survives_backward_chunks(tmp_path):
    model = small_model("hybrid", "history")
    trainer = WmSTrainer(_cfg(tmp_path, rollout_mode="hybrid", lane_head="history",
                              truncation_steps=2), model, _dataset(), out_dir=tmp_path)
    encodings, conditioning = [], []
    h1 = model.driver_encoder.register_forward_hook(
        lambda _m, _a, out: encodings.append(out["embedding"].detach().clone()))
    h2 = model.residual_net.register_forward_pre_hook(
        lambda _m, _a, kw: conditioning.append(kw["conditioning"].detach().clone()),
        with_kwargs=True)
    try:
        record = trainer.train_batch(make_batch(history=True), unroll_steps=6)
    finally:
        h1.remove(), h2.remove()
    assert record["finite"] and len(encodings) == 1 and len(conditioning) == 6
    assert all(torch.equal(c, encodings[0]) for c in conditioning)
