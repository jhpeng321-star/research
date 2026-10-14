"""INFRAV5 (2026-10-14, INFRAV5_NOTES.md): non-lane v5 world-model fixes, all default OFF.

1. ``curriculum.selection = "multi_horizon_v2"``: same-horizon checkpoint selection.
2. Jam oversampling covers the C4 / GENRAND sources (receipt: jam shares before/after).
3. ``loss.residual.residual_mag_weight`` (existing key) works with the hybrid residual.
4. ``loss.driver_prior``: population-mean anchor of the encoded driver settings.
5. ``WmSModel.cav_exec_cap = "idm_normal[:textbook|legacy]"``: execution cap with fixed CAV
   vType settings (textbook cav_typical by default, legacy normal_car).
6. ``data.jam.min_fraction``: share-based jam-window rule (0 = legacy any-observation rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "train"))

from _compat import FlatConfig, legacy  # noqa: E402
from synth import make_arrays, small_model  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.config.train import TrainConfig  # noqa: E402
from wms.core.errors import ConfigError  # noqa: E402
from wms.model import cav as cav_motion  # noqa: E402
from wms.model.driver_encoder import PARAMETER_NAMES  # noqa: E402
from wms.model.residual_net import HdvConfig  # noqa: E402
from wms.model.spec import ModelSpec  # noqa: E402
from wms.model.wms import WmSModel  # noqa: E402
from wms.physics.accel import baseline_accel  # noqa: E402
from wms.physics.state import state_kin, state_present  # noqa: E402
from wms.train import selection  # noqa: E402
from wms.train.losses.compose import build_loss  # noqa: E402
from wms.train.losses.driver_prior import driver_prior_loss  # noqa: E402
from wms.train.settings import TrainSettings  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402
from wms.train.unroll import UnrollRunner  # noqa: E402

train_mod = legacy("train")
REPO = Path(__file__).resolve().parents[2]
NEW_FLAT_KEYS = ("driver_prior_weight", "driver_prior_accel", "driver_prior_time_headway",
                 "driver_prior_decel", "driver_prior_min_gap", "driver_prior_v0_ratio")


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


# ============================================================================ defaults / off
def test_defaults_off_and_old_flat_forms_unchanged():
    s = TrainSettings()
    assert s.selection_mode == "long_horizon" and s.driver_prior_weight == 0.0
    flat = TrainConfig().to_legacy_flat()
    assert not any(k in flat for k in NEW_FLAT_KEYS)
    for name in ("c4_wm_v3.toml", "c4_wm_v4.toml"):
        f = load_train_config(REPO / "configs" / name).to_legacy_flat()
        assert not any(k in f for k in NEW_FLAT_KEYS)
        assert TrainConfig.from_legacy_flat(f).to_legacy_flat() == f
    assert cav_motion.check_exec_cap_mode(None) == "off"


def test_validation_of_new_switches():
    with pytest.raises(ValueError):           # v2 needs every horizon validated
        TrainSettings(select_by_long_horizon=True, selection_mode="multi_horizon_v2")
    TrainSettings(select_by_long_horizon=True, selection_mode="multi_horizon_v2",
                  val_all_horizons=True)
    with pytest.raises(ValueError):           # prior needs the single t0 encoding
        TrainSettings(driver_prior_weight=0.1, rollout_mode="hybrid")
    with pytest.raises(ValueError):
        TrainSettings(driver_prior_weight=-1.0)
    TrainSettings(driver_prior_weight=0.1, rollout_mode="hybrid", driver_grad_across_chunks=True)
    with pytest.raises(ConfigError):
        load_train_config(REPO / "configs" / "c4_wm_v5_base.toml",
                          ["loss.driver_prior.weight=0.1",
                           "curriculum.driver_grad_across_chunks=false"]).validate_for_training()
    with pytest.raises(ConfigError):
        load_train_config(REPO / "configs" / "c4_wm_v5_base.toml",
                          ["loss.driver_prior.accel=0.0"])


def _batch():
    b, t, k = 2, 7, 4
    torch.manual_seed(0)
    states = torch.zeros(b, t, k, 7)
    states[..., 0] = torch.rand(b, t, k) * 80 + torch.arange(k) * 22
    states[..., 2] = 1.0
    states[..., 4] = torch.rand(b, t, k) * 15 + 5
    states[..., 5] = 0.1
    states[..., 6] = 5.0
    present = torch.ones(b, t, k, dtype=torch.bool)
    present[:, :, 3] = False
    is_cav = torch.zeros(b, t, k, dtype=torch.bool)
    is_cav[:, :, 0] = True
    hist = torch.randn(b, k, 10, 6) * 0.3
    hist[..., 0] += 40.0
    hist[..., 4] += 18.0
    return {"states": states, "present": present, "is_cav": is_cav,
            "cav_teacher": torch.zeros(b, t, k, 4), "hist": hist,
            "hist_mask": torch.ones(b, k, 10, dtype=torch.bool)}


def _hybrid_model():
    torch.manual_seed(0)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=1, ffn=32))
    model.train()
    return model


def _run(settings, steps=6, backward=True):
    model = _hybrid_model()
    runner = UnrollRunner(model, settings, build_loss(settings, physics=model.physics),
                          projection=None, device=torch.device("cpu"))
    torch.manual_seed(1)
    record = runner.run(_batch(), unroll_steps=steps, backward=backward, truncation_steps=3)
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    return record, grads


HYBRID = dict(rollout_mode="hybrid", driver_grad_across_chunks=True, truncation_steps=3)


def test_prior_off_is_bit_identical_one_training_step():
    base_rec, base_grads = _run(TrainSettings(**HYBRID))
    off_rec, off_grads = _run(TrainSettings(**HYBRID, driver_prior_weight=0.0,
                                            driver_prior_accel=9.0, driver_prior_v0_ratio=0.5))
    assert base_rec == off_rec and "driver_prior" not in off_rec
    assert base_grads.keys() == off_grads.keys()
    assert all(torch.equal(base_grads[k], off_grads[k]) for k in base_grads)


def test_off_fit_is_bit_identical_with_explicit_off_values(tmp_path):
    def run(sub, extra):
        flat = train_mod.WmSTrainConfig(
            device="cpu", threads=1, truncation_steps=8, batch_size=2, seed=3,
            unroll_curriculum_s=(5.0, 10.0), stage_steps=(2, 2), out_dir=str(tmp_path / sub),
            select_by_long_horizon=True, val_every=1).to_dict()
        flat.update(extra)
        ds = train_mod.TrajectoryDataset.from_arrays(make_arrays())
        trainer = WmSTrainer(FlatConfig(flat), small_model(), ds, val_dataset=ds,
                             out_dir=tmp_path / sub)
        result = trainer.fit()
        return trainer.model.state_dict(), result

    a, ra = run("a", {})
    b, rb = run("b", {"driver_prior_weight": 0.0, "driver_prior_decel": 3.0,
                      "selection_mode": "long_horizon"})
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [r["loss"] for r in ra.history] == [r["loss"] for r in rb.history]
    assert ra.val_by_stage == rb.val_by_stage and ra.selection is None and rb.selection is None


# ============================================================================ 1. selection v2
def _rows(stage, step, values):
    return [dict(horizon_s=h, mean_position=v, stage_s=stage, step=step, valid_targets=5)
            for h, v in values.items()]


#: The c4_wm_v4_cont validation log (RUN_SUMMARY selection table, mean_position = ratio x
#: reference); the 120 s value of the final event is v4_cont's own (144.59).
REFS = (7.0356, 49.4889, 128.7179, 0.0)
V4_LOG = [(60.0, 200, {10.0: 0.8886923 * 7.0356, 30.0: 0.8477449 * 49.4889,
                       60.0: 0.7517397 * 128.7179}, False),
          (60.0, 400, {10.0: 0.8949126 * 7.0356, 30.0: 0.8480655 * 49.4889,
                       60.0: 0.7521272 * 128.7179}, True),
          (120.0, 560, {10.0: 0.9085161 * 7.0356, 30.0: 0.8663431 * 49.4889,
                        60.0: 0.7451378 * 128.7179, 120.0: 144.5919}, True)]
#: Synthetic 120 s re-validation of the 60 s-stage checkpoints (on unseen walls v4's 60 s-stage
#: step-200 checkpoint was ~2x worse at 120 s than the final one: 142.4 vs 71.7).
REVAL_120 = {200: 210.0, 400: 205.0}


def test_v1_on_the_v4_log_picks_the_60s_stage_checkpoint():
    old = selection.MultiHorizonSelection(references=REFS, eligible_stage_s=60.0)
    for stage, step, values, _ in V4_LOG:
        old.consider_event(_rows(stage, step, values), stage_s=stage, step=step)
    assert old.best["step"] == 200       # the v4 defect: 120 s ratio = own first = 1.0


def test_v2_on_the_v4_log_compares_on_one_horizon_set_and_picks_the_final():
    sel = selection.SameHorizonSelection(references=REFS, eligible_stage_s=60.0)
    assert sel.common_horizons == (10.0, 30.0, 60.0)
    flags = [sel.consider_event(_rows(stage, step, values), stage_s=stage, step=step,
                                stage_end=end) for stage, step, values, end in V4_LOG]
    # online (10/30/60 s only, fixed refs): step 200 is still the provisional best
    assert [f["global_best"] for f in flags] == [True, False, False]
    assert sel.table[2]["score"] == pytest.approx((0.9085161 + 0.8663431 + 0.7451378) / 3)
    assert [c["step"] for c in sel.candidates()] == [200, 400, 560]
    assert sel.full_horizons() == (10.0, 30.0, 60.0, 120.0)
    calls = []

    def revalidate(step, horizons):
        calls.append((step, tuple(horizons)))
        return _rows(60.0, step, {120.0: REVAL_120[step]})

    final = sel.finalize(revalidate)
    assert calls == [(200, (120.0,)), (400, (120.0,))]       # the final event is complete
    assert final["normalisers"]["120"] == pytest.approx(144.5919)   # best candidate, not own-first
    table = {r["step"]: r for r in final["table"]}
    assert table[560]["score"] == pytest.approx((0.9085161 + 0.8663431 + 0.7451378 + 1.0) / 4)
    assert table[200]["score"] == pytest.approx(
        (0.8886923 + 0.8477449 + 0.7517397 + 210.0 / 144.5919) / 4)
    assert final["best_step"] == 560 and table[560]["selected"]
    assert all(r["complete"] for r in table.values())
    assert sel.best["step"] == 560 and sel.summary()["final_comparison"] is final
    # if the 60 s-stage checkpoint were as good at 120 s, it would win (same horizon set)
    sel2 = selection.SameHorizonSelection(references=REFS, eligible_stage_s=60.0)
    for stage, step, values, end in V4_LOG:
        sel2.consider_event(_rows(stage, step, values), stage_s=stage, step=step, stage_end=end)
    final2 = sel2.finalize(lambda step, hs: _rows(60.0, step, {120.0: 140.0}))
    assert final2["best_step"] == 200


def test_v2_ignores_ineligible_and_incomplete_events():
    sel = selection.SameHorizonSelection(references=(10.0, 50.0, 0.0), horizons=(10.0, 30.0, 60.0),
                                         eligible_stage_s=30.0)
    assert sel.common_horizons == (10.0, 30.0)
    f0 = sel.consider_event(_rows(10.0, 1, {10.0: 1.0}), stage_s=10.0, step=1, stage_end=True)
    assert not any(f0.values()) and not sel.table[0]["eligible"]
    f1 = sel.consider_event(_rows(30.0, 2, {10.0: 9.0}), stage_s=30.0, step=2)  # 30 s missing
    assert not any(f1.values()) and sel.table[1]["score"] == float("inf")
    sel.consider_event(_rows(30.0, 3, {10.0: 9.0, 30.0: 40.0}), stage_s=30.0, step=3,
                       stage_end=True)
    sel.consider_event(_rows(60.0, 4, {10.0: 9.0, 30.0: 45.0, 60.0: 100.0}), stage_s=60.0, step=4,
                       stage_end=True)
    final = sel.finalize(lambda step, hs: _rows(30.0, step, {60.0: 90.0}))
    assert [r["step"] for r in final["table"]] == [3, 4]
    assert final["normalisers"]["60"] == 90.0 and final["best_step"] == 3


def _fake_fit(tmp_path, extra, reval_score=None):
    flat = train_mod.WmSTrainConfig(
        device="cpu", threads=1, truncation_steps=8, batch_size=2, seed=3,
        unroll_curriculum_s=(10.0, 30.0, 60.0, 120.0), stage_steps=(3, 4, 4, 2),
        out_dir=str(tmp_path), select_by_long_horizon=True, val_every=2).to_dict()
    flat.update(extra)
    ds = train_mod.TrajectoryDataset.from_arrays(make_arrays())
    trainer = WmSTrainer(FlatConfig(flat), small_model(), ds, val_dataset=ds, out_dir=tmp_path)
    calls, state = [], {"step": 0}
    marker_name, marker = next(iter(trainer.model.named_parameters()))
    trainer.marker_name = marker_name

    def train_batch(batch, *, unroll_steps, truncation_steps=None):
        state["step"] += 1
        with torch.no_grad():
            marker.view(-1)[0] = float(state["step"])     # weights identify the step
        return {"loss": 1.0, "unroll_steps": int(unroll_steps), "seconds": 0.0, "finite": True}

    def evaluate(val_dataset, *, unroll_steps, require_full_horizon=False):
        weight_step = int(round(float(marker.detach().view(-1)[0])))
        calls.append((state["step"], weight_step, unroll_steps))
        h = unroll_steps * 0.5
        # 60 s-stage weights look best on 10/30/60 s; at 120 s they are bad
        score = {7: 0.8, 9: 0.85, 11: 0.9, 13: 0.95}.get(weight_step, 1.0) * h
        if h == 120.0:
            score = (reval_score or {}).get(weight_step, 1.0 * h)
        return {"n_batches": 1, "valid_targets": 5, "reached_steps": unroll_steps,
                "available_steps": unroll_steps, "unroll_steps": unroll_steps,
                "mean_position": score}

    trainer.train_batch = train_batch
    trainer.evaluate = evaluate
    return trainer.fit(), calls, trainer, marker


V2 = {"val_all_horizons": True, "selection_mode": "multi_horizon_v2",
      "selection_reference": [10.0, 30.0, 60.0, 0.0], "selection_min_stage_s": 60.0}


def test_v2_fit_revalidates_candidates_and_restores_final_weights(tmp_path):
    result, calls, trainer, marker = _fake_fit(tmp_path, V2, reval_score={9: 300.0, 11: 290.0})
    sel = result.selection
    assert sel["mode"] == "multi_horizon_v2"
    online = [e for e in sel["table"] if e["eligible"]]
    assert [e["step"] for e in online] == [8, 10, 11, 12, 13]
    # candidates: 60 s stage best (step 8 weights? scores) + last (11); 120 s best + last (13)
    final = sel["final_comparison"]
    steps = [r["step"] for r in final["table"]]
    assert 11 in steps and 13 in steps
    reval = [c for c in calls if c[0] == 13 and c[1] != 13]
    assert reval and all(c[2] == 240 for c in reval)                # only 120 s re-validated
    assert {c[1] for c in reval} == {r["step"] for r in final["table"]
                                      if r["revalidated_s"]}
    assert final["best_step"] == 13 and result.receipt["long_horizon_selection"][
        "policy"] == "same-horizon-final-comparison-of-stage-candidates"
    ckpt = torch.load(tmp_path / "wm_s_best.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == 13 and ckpt["selection"]["selected"]
    assert int(round(float(marker.detach().view(-1)[0]))) == 13            # final weights restored
    logged = [json.loads(x) for x in (tmp_path / "val_log.jsonl").read_text().splitlines()]
    assert sum(1 for r in logged if r.get("revalidated")) == len(reval)


def test_v2_fit_picks_a_60s_checkpoint_when_it_is_also_better_at_120s(tmp_path):
    result, _, trainer, marker = _fake_fit(tmp_path, V2, reval_score={9: 100.0, 11: 100.0,
                                                                     13: 130.0, 12: 130.0})
    final = result.selection["final_comparison"]
    best = final["best_step"]
    assert best in (8, 9, 10, 11)
    ckpt = torch.load(tmp_path / "wm_s_best.pt", map_location="cpu", weights_only=True)
    assert ckpt["step"] == best
    weights = ckpt["model"][trainer.marker_name]
    assert int(round(float(weights.view(-1)[0]))) == best         # the candidate's weights
    assert int(round(float(marker.detach().view(-1)[0]))) == 13


def test_v2_real_fit_end_to_end(tmp_path):
    """Real compute: v2 validates, keeps candidates, completes them and writes wm_s_best.pt."""
    flat = train_mod.WmSTrainConfig(
        device="cpu", threads=1, truncation_steps=8, batch_size=2, seed=3,
        unroll_curriculum_s=(5.0, 10.0), stage_steps=(2, 2), out_dir=str(tmp_path),
        select_by_long_horizon=True, val_every=1).to_dict()
    flat.update({"val_all_horizons": True, "selection_mode": "multi_horizon_v2",
                 "selection_horizons_s": [5.0, 10.0], "selection_reference": [0.0, 0.0],
                 "selection_min_stage_s": 5.0})
    ds = train_mod.TrajectoryDataset.from_arrays(make_arrays())
    trainer = WmSTrainer(FlatConfig(flat), small_model(), ds, val_dataset=ds, out_dir=tmp_path)
    result = trainer.fit()
    final = result.selection["final_comparison"]
    assert final["horizons_s"] == [5.0, 10.0]
    assert all(r["complete"] for r in final["table"])
    assert [r["revalidated_s"] for r in final["table"] if r["stage_s"] == 5.0][0] == [10.0]
    best = torch.load(tmp_path / "wm_s_best.pt", map_location="cpu", weights_only=True)
    assert best["step"] == final["best_step"]
    last = torch.load(tmp_path / "wm_s.pt", map_location="cpu", weights_only=True)
    state = trainer.model.state_dict()
    assert all(torch.equal(last["model"][k], state[k]) for k in state)  # final weights kept
    receipt = json.loads((tmp_path / "TRAIN_RECEIPT.json").read_text())
    assert receipt["long_horizon_selection"]["multi_horizon_v2"]["final_comparison"] == final


# ============================================================================ 2. jam / C4 sources
class _Windows(torch.utils.data.Dataset):
    """Windows whose min HDV speed is 2 m/s (jam) or 20 m/s."""

    def __init__(self, jam_flags):
        self.flags = list(jam_flags)
        self.receipt = {"n_records": len(self.flags)}

    def __len__(self):
        return len(self.flags)

    def __getitem__(self, i):
        states = np.zeros((5, 3, 7), np.float32)
        states[..., 4] = 20.0
        if self.flags[i]:
            states[3, 1, 4] = 2.0
        return {"states": states, "present": np.ones((5, 3), bool),
                "is_cav": np.array([[True, False, False]] * 5)}


def _c4(flags):
    return {"train": _Windows(flags), "val": None, "train_events": None, "receipt": {}}


@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_jam_oversampling_covers_c4_sources(factor):
    from wms.data import jam as jamdata
    from wms.data.c4_pairs import add_c4_sources

    orig = _Windows([True, False, False, False])                 # 25 % jam
    base = (orig if factor == 1.0 else jamdata.JamOversampledDataset(
        orig, np.array(orig.flags), oversample=factor))
    plain = [True] * 2 + [False] * 8                              # GENRAND-like: 20 % jam
    mixed, _ = add_c4_sources(base, None, [("genrand_plain", _c4(plain), 0.15),
                                           ("genrand_branch", _c4([False, True]), 0.05)])
    w = np.asarray(mixed.stage_weights(60.0))
    shares = dict(zip([s.name for s in mixed.sources], mixed.shares))
    assert shares == pytest.approx({"original": 0.8, "genrand_plain": 0.15,
                                    "genrand_branch": 0.05})
    for k, src in enumerate(mixed.sources):
        lo = mixed.offsets[k]
        ws = w[lo:lo + len(src.dataset)]
        assert ws.sum() == pytest.approx(shares[src.name])        # source shares unchanged
        jam = np.asarray(src.dataset.flags)
        if factor == 1.0:
            assert src.jam is None and np.allclose(ws, ws[0])
        else:
            assert np.array_equal(src.jam, jam)
            assert ws[jam][0] == pytest.approx(factor * ws[~jam][0])
    desc = mixed.describe_stages([60.0])[0]
    if factor == 1.0:
        assert "jam_share_total" not in desc
        assert "jam_fraction_sampled" not in desc["sources"]["genrand_plain"]
    else:
        tot = desc["jam_share_total"]
        before = 0.8 * 0.25 + 0.15 * 0.2 + 0.05 * 0.5
        after = 0.8 * (2 / 5) + 0.15 * (4 / 12) + 0.05 * (2 / 3)
        assert tot["before"] == pytest.approx(before, abs=1e-4)
        assert tot["after"] == pytest.approx(after, abs=1e-4)
        assert desc["sources"]["genrand_plain"]["jam_fraction_sampled"] == pytest.approx(
            4 / 12, abs=1e-4)


def test_paired_batches_are_unaffected_by_jam_weights():
    from wms.data import jam as jamdata
    from wms.data.c4_pairs import PairedBatchDataset, add_c4_sources

    orig = _Windows([True, False, False, False])
    mixed, _ = add_c4_sources(
        jamdata.JamOversampledDataset(orig, np.array(orig.flags), oversample=2.0), None,
        [("genrand_plain", _c4([True, False]), 0.15)])

    class _Pairs:
        receipt = {"n_groups": 1}
        groups = [{"rows": [0, 1, 2]}]

        def group_records(self, gi, take, base_row=0):
            return [{"pair_row": r} for r in take]

    wrapped = PairedBatchDataset(mixed, _Pairs(), fraction=1.0, groups=1, max_arms=3,
                                 batch_size=3)
    assert wrapped.stage_weights(60.0) == mixed.stage_weights(60.0)   # delegated
    batch = wrapped.paired_batch(np.random.default_rng(0))
    assert sorted(r["pair_row"] for r in batch) == [0, 1, 2]          # all arms, no jam reweight


def test_v5_config_jam_and_sources():
    cfg = load_train_config(REPO / "configs" / "c4_wm_v5_base.toml")
    assert cfg.data.jam.oversample == 2.0
    assert {s.name: (s.mix_ratio, s.paired) for s in cfg.data.c4_sources} == {
        "genrand_plain": (0.15, False), "genrand_branch": (0.05, True)}
    assert cfg.loss.paired_diff.weight == 1.0 and cfg.loss.paired_diff.fraction == 0.25
    assert cfg.data.jam.min_fraction == 0.3
    assert tuple(cfg.curriculum.stage_steps) == (1200, 900, 650, 250) and cfg.selection.val_every == 400
    assert cfg.loss.driver_prior.weight == 0.0 and cfg.loss.driver_prior.target == "legacy"


# ============================================================================ 3. residual mag
def test_residual_mag_weight_key_and_hybrid_residual():
    cfg = load_train_config(REPO / "configs" / "c4_wm_v5_base.toml")
    flat = cfg.to_legacy_flat()
    assert flat["residual_mag_weight"] == 0.05 and cfg.model.rollout_mode == "hybrid"
    s = TrainSettings.from_flat(flat)
    names = build_loss(s).names
    assert "residual_mag" in names and "residual_penalty" in names
    rec_lo, _ = _run(TrainSettings(**HYBRID, residual_mag_weight=0.01))
    rec_hi, _ = _run(TrainSettings(**HYBRID, residual_mag_weight=0.05))
    assert rec_lo["residual_mag"] > 0.0
    assert rec_hi["residual_mag"] == pytest.approx(5.0 * rec_lo["residual_mag"], rel=1e-5)
    assert rec_hi["position"] == rec_lo["position"]


# ============================================================================ 4. driver prior
def _values(b=2, k=5, seed=0):
    torch.manual_seed(seed)
    v = torch.zeros(b, k, len(PARAMETER_NAMES))
    base = {"v0": 30.0, "time_headway": 1.4, "min_gap": 3.0, "accel": 1.8, "decel": 3.3}
    for name, value in base.items():
        v[..., PARAMETER_NAMES.index(name)] = value + 0.2 * torch.randn(b, k)
    return v.requires_grad_(True)


TARGETS = {"accel": 2.6, "time_headway": 1.1, "decel": 4.5, "min_gap": 2.5, "v0": 1.0}


def test_prior_value_is_relative_l2_of_batch_means():
    v = _values()
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 4] = False
    value, means = driver_prior_loss(v, mask, TARGETS, weight=0.5, speed_limit_mps=30.0)
    expected = 0.0
    for name, tgt in TARGETS.items():
        col = v[..., PARAMETER_NAMES.index(name)][mask]
        if name == "v0":
            col = col / 30.0
        expected += ((float(col.mean()) - tgt) / tgt) ** 2
    assert float(value) == pytest.approx(0.5 * expected, rel=1e-5)
    assert means["accel"] == pytest.approx(float(v[..., 3][mask].mean()), rel=1e-6)
    empty, _ = driver_prior_loss(v, torch.zeros(2, 5, dtype=torch.bool), TARGETS, weight=1.0,
                                 speed_limit_mps=30.0)
    assert float(empty) == 0.0


def test_prior_pulls_only_the_batch_mean():
    v = _values()
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 0] = False
    value, _ = driver_prior_loss(v, mask, TARGETS, weight=1.0, speed_limit_mps=30.0)
    (grad,) = torch.autograd.grad(value, v)
    for i in range(len(PARAMETER_NAMES)):
        g = grad[..., i]
        assert torch.allclose(g[mask], g[mask][0].expand_as(g[mask]))   # same pull for all
        assert float(g[~mask].abs().sum()) == 0.0
    # zero-mean per-driver deviations (identification) do not change the prior
    dev = torch.randn(2, 5, len(PARAMETER_NAMES)) * mask.unsqueeze(-1)
    dev = dev - dev.sum((0, 1), keepdim=True) / mask.sum() * mask.unsqueeze(-1)
    moved, _ = driver_prior_loss((v + dev).detach(), mask, TARGETS, weight=1.0,
                                 speed_limit_mps=30.0)
    assert float(moved) == pytest.approx(float(value), rel=1e-5)
    # names that are not anchored get no gradient
    for name in ("politeness", "mobil_threshold", "merge_gap", "merge_urgency"):
        assert float(grad[..., PARAMETER_NAMES.index(name)].abs().sum()) == 0.0


def test_prior_on_reaches_only_the_encoder_and_is_logged():
    base_rec, base_grads = _run(TrainSettings(**HYBRID))
    rec, grads = _run(TrainSettings(**HYBRID, driver_prior_weight=1.0))
    assert rec["driver_prior"] > 0.0 and 0.0 < rec["driver_prior_grad_share"] <= 1.0
    assert set(rec["driver_prior_means"]) == {"accel", "time_headway", "decel", "min_gap",
                                              "v0_ratio"}
    assert rec["loss"] == pytest.approx(base_rec["loss"] + rec["driver_prior"])
    for key in ("position", "speed", "lane"):
        assert rec[key] == base_rec[key]
    encoder = {k for k in grads if k.startswith("driver_encoder.")}
    assert encoder and any(not torch.equal(grads[k], base_grads[k]) for k in encoder)
    assert all(torch.equal(grads[k], base_grads[k]) for k in grads if k not in encoder)
    # forward-only (validation): value logged, no gradient bookkeeping
    ev, _ = _run(TrainSettings(**HYBRID, driver_prior_weight=1.0), backward=False)
    assert ev["driver_prior"] == pytest.approx(rec["driver_prior"])
    assert "driver_prior_grad_share" not in ev


def test_prior_targets_tables():
    from wms.train.losses.driver_prior import prior_targets

    explicit = prior_targets(TrainSettings(driver_prior_accel=2.0))
    assert explicit["accel"] == 2.0 and explicit["v0"] == 1.0
    assert prior_targets(TrainSettings(driver_prior_target="textbook", driver_prior_accel=2.0)) == {
        "accel": 1.2, "time_headway": 1.2, "decel": 2.0, "min_gap": 2.0, "v0": 1.0}
    assert prior_targets(TrainSettings(driver_prior_target="legacy")) == {
        "accel": 2.6, "time_headway": 1.1, "decel": 4.5, "min_gap": 2.5, "v0": 1.0}
    with pytest.raises(ValueError):
        TrainSettings(driver_prior_target="median")
    cfg = load_train_config(REPO / "configs" / "c4_wm_v5_base.toml",
                            ["loss.driver_prior.target=\"textbook\""])
    assert TrainSettings.from_flat(cfg.to_legacy_flat()).driver_prior_target == "textbook"
    rec_tb, _ = _run(TrainSettings(**HYBRID, driver_prior_weight=1.0,
                                   driver_prior_target="textbook"))
    rec_lg, _ = _run(TrainSettings(**HYBRID, driver_prior_weight=1.0,
                                   driver_prior_target="legacy"))
    assert rec_tb["driver_prior_means"] == rec_lg["driver_prior_means"]
    assert rec_tb["driver_prior"] != rec_lg["driver_prior"]


# ============================================================================ 6. jam min_fraction
def test_jam_min_fraction_rule_and_cache(tmp_path):
    from wms.data import jam as jamdata

    class _Frac(torch.utils.data.Dataset):
        """Window i: i of 10 HDV observations below 5 m/s (2 HDVs x 5 steps)."""

        receipt = {"n_records": 6}

        def __len__(self):
            return 6

        def __getitem__(self, i):
            states = np.zeros((5, 3, 7), np.float32)
            states[..., 4] = 20.0
            flat = states[:, 1:, 4].reshape(-1)
            flat[:i] = 1.0
            states[:, 1:, 4] = flat.reshape(5, 2)
            return {"states": states, "present": np.ones((5, 3), bool),
                    "is_cav": np.array([[True, False, False]] * 5)}

    ds = _Frac()
    legacy_flags = jamdata.compute_jam_flags(ds, speed_mps=5.0)["flags"]
    assert legacy_flags.tolist() == [False, True, True, True, True, True]
    cache = tmp_path / "jam.json"
    rep = jamdata.compute_jam_flags(ds, speed_mps=5.0, min_fraction=0.25, cache_path=cache)
    assert rep["flags"].tolist() == [False, False, False, True, True, True]   # > 25 %
    assert json.loads(cache.read_text())["min_fraction"] == 0.25
    # a cache written for another rule is not reused
    other = jamdata.compute_jam_flags(ds, speed_mps=5.0, min_fraction=0.35, cache_path=cache)
    assert other["flags"].tolist() == [False, False, False, False, True, True]
    assert not other["cache_hit"]
    legacy_doc = tmp_path / "legacy.json"
    jamdata.compute_jam_flags(ds, speed_mps=5.0, cache_path=legacy_doc)
    assert "min_fraction" not in json.loads(legacy_doc.read_text())        # legacy bytes
    assert jamdata.record_hdv_jam_fraction(ds[3], 5.0) == pytest.approx(0.3)
    with pytest.raises(ConfigError):
        load_train_config(REPO / "configs" / "c4_wm_v5_base.toml", ["data.jam.min_fraction=1.5"])


# ============================================================================ 5. idm_normal cap
def _model():
    torch.manual_seed(11)
    return WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                    hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32)).eval()


def _state(s, v, lane, cav):
    k = len(s)
    return {"s_m": torch.tensor([s], dtype=torch.float32),
            "v_mps": torch.tensor([v], dtype=torch.float32), "a_mps2": torch.zeros(1, k),
            "lane": torch.tensor([lane], dtype=torch.long),
            "present": torch.ones(1, k, dtype=torch.bool), "is_cav": torch.tensor([cav]),
            "length_m": torch.full((1, k), 5.0)}


def _params(model, k, **overrides):
    defaults = model._default_driver_parameters(torch.zeros(1, k, 7), 1, k)[0, 0]
    out = {name: torch.full((1, k), float(defaults[i])) for i, name in enumerate(PARAMETER_NAMES)}
    for name, value in overrides.items():
        out[name] = torch.full((1, k), float(value))
    return out


TEXTBOOK = {"accel": 1.2, "decel": 2.0, "time_headway": 1.2, "min_gap": 2.0, "speed_factor": 1.0}
LEGACY = {"accel": 2.6, "decel": 4.5, "time_headway": 1.1, "min_gap": 2.5, "speed_factor": 1.0}


@pytest.mark.parametrize("mode,table", [("idm_normal", TEXTBOOK), ("idm_normal:textbook", TEXTBOOK),
                                        ("idm_normal:legacy", LEGACY)])
def test_idm_normal_cap_uses_the_fixed_cav_settings_table(mode, table):
    model = _model()
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0), (True, False, True))
    inferred = _params(model, 3, accel=0.5, v0=18.0, time_headway=2.0)   # conservative WM guess
    with torch.no_grad():
        out = model.forward(dict(x), driver_params=inferred)
    model.cav_exec_cap = "idm"
    cap_idm = model._exec_cap(out, x, inferred)
    model.cav_exec_cap = mode
    cap_normal = model._exec_cap(out, x, inferred)
    assert cav_motion.exec_cap_settings(mode) == table
    params = _params(model, 3, accel=table["accel"], decel=table["decel"],
                     time_headway=table["time_headway"], min_gap=table["min_gap"],
                     v0=model.physics.speed_limit_mps * table["speed_factor"])
    kin = state_kin(x).float()
    physics = baseline_accel(kin, state_present(x, kin), model.physics, params)
    v = x["v_mps"]
    expected = torch.minimum(torch.minimum(physics, (model.physics.speed_limit_mps - v) / 0.5),
                             torch.tensor(table["accel"]))
    assert torch.allclose(cap_normal, expected)
    # independent of the inferred settings; the "idm" cap follows them (accel 0.5)
    cap_normal2 = model._exec_cap(out, x, _params(model, 3, accel=3.0))
    assert torch.equal(cap_normal, cap_normal2)
    assert float(cap_idm[0, 2]) <= 0.5 + 1e-6 < float(cap_normal[0, 2])
    model.cav_exec_cap = "off"


def test_cav_settings_tables_and_modes():
    assert cav_motion.CAV_SETTINGS_TABLES == {"textbook": TEXTBOOK, "legacy": LEGACY}
    assert cav_motion.DEFAULT_CAV_SETTINGS_TABLE == "textbook"
    assert cav_motion.NORMAL_CAR_SETTINGS == LEGACY
    assert cav_motion.check_exec_cap_mode("IDM_NORMAL:Legacy") == "idm_normal:legacy"
    with pytest.raises(ValueError):
        cav_motion.check_exec_cap_mode("idm_normal:bogus")
    with pytest.raises(ValueError):
        cav_motion.exec_cap_settings("idm")


def test_textbook_constants_match_population_module_when_merged():
    population = pytest.importorskip("traffic_shepherd.c4.population")
    if not hasattr(population, "TEXTBOOK_CAV_PARAMS"):
        pytest.skip("c4/popfix population module not merged")
    cav = population.TEXTBOOK_CAV_PARAMS
    assert TEXTBOOK == {"accel": cav["accel_mps2"], "decel": cav["decel_mps2"],
                        "time_headway": cav["tau_s"], "min_gap": cav["min_gap_m"],
                        "speed_factor": cav["speed_factor"]}
    from wms.train.losses.driver_prior import TEXTBOOK_POPULATION_MEDIANS as ours
    med = population.TEXTBOOK_POPULATION_MEDIANS
    assert ours == {"accel": med["accel"], "time_headway": med["time_headway"],
                    "decel": med["decel"], "min_gap": med["min_gap"], "v0": med["speed_factor"]}


def _bridge(value):
    from wms.control.bridge import make_cav_controller
    from wms.control.safety import safe_accel

    def policy(obs, signal):
        return torch.full(obs.cav_mask.shape, float(value)), torch.zeros(obs.cav_mask.shape)
    policy.accel_max = 2.6
    return make_cav_controller(policy, lambda obs, s, field=None: torch.zeros(s.shape[0], 4),
                               safe_accel, accel_max=2.6)


def _step(model, state, mode, **kw):
    model.cav_exec_cap = mode
    try:
        with torch.no_grad():
            return model.step(dict(state), k_steps=0, **kw)
    finally:
        model.cav_exec_cap = "off"


def test_idm_normal_caps_commanded_cavs_only_and_leaves_hdvs_and_teachers():
    model = _model()
    # CAV at 15 m/s, slow HDV leader 60 m ahead at 10 m/s: safety passes +2.6, IDM binds
    x = _state((-600.0, -540.0, -500.0), (15.0, 10.0, 22.0), (1, 1, 0), (True, False, True))
    params = _params(model, 3, accel=1.0, time_headway=2.0)      # WM-inferred, conservative
    normal = cav_motion.CAV_SETTINGS_TABLES["legacy"]
    nparams = _params(model, 3, accel=normal["accel"], decel=normal["decel"],
                      time_headway=normal["time_headway"], min_gap=normal["min_gap"],
                      v0=model.physics.speed_limit_mps)
    idm_normal = float(model.physics_accel(x, nparams)[0, 0])
    off = _step(model, x, "off", driver_params=params, cav_controller=_bridge(2.6))
    on = _step(model, x, "idm_normal:legacy", driver_params=params, cav_controller=_bridge(2.6))
    inferred = _step(model, x, "idm", driver_params=params, cav_controller=_bridge(2.6))
    assert 0.0 < idm_normal < float(off["accel_mps2"][0, 0]) - 0.5
    assert float(on["accel_mps2"][0, 0]) == pytest.approx(idm_normal, rel=1e-5)
    assert float(inferred["accel_mps2"][0, 0]) < idm_normal          # inferred accel 1.0 binds
    # HDV slot unchanged
    for key in ("accel_mps2", "next_s_m", "next_v_mps"):
        assert torch.equal(on[key][0, 1], off[key][0, 1]), key
    # recorded teachers are not capped
    steps = 6
    teacher = torch.zeros(1, steps + 1, 3, 4)
    for i in range(steps + 1):
        v = x["v_mps"][0] + 2.5 * 0.5 * i
        teacher[0, i, :, 0] = x["s_m"][0] + (x["v_mps"][0] + v) * 0.25 * i
        teacher[0, i, :, 1] = x["lane"][0].float()
        teacher[0, i, :, 2] = v
        teacher[0, i, :, 3] = 2.5
    out = {}
    for mode in ("off", "idm_normal"):
        model.cav_exec_cap = mode   # textbook table: would cap the +2.5 teacher if applied
        with torch.no_grad():
            out[mode] = model.rollout(dict(x), n_steps=steps, cav_teacher=teacher)
    model.cav_exec_cap = "off"
    for key in ("positions", "speeds", "accel"):
        assert torch.equal(out["off"][key], out["idm_normal"][key]), key


def test_idm_normal_is_wired_through_every_switch(monkeypatch):
    from types import SimpleNamespace

    import wms.model.checkpoint as checkpoint_module
    from wms.control.dreamer import DreamerConfig
    from wms.control.train_cli import _read_config
    from wms.eval import backend

    assert DreamerConfig(cav_exec_cap="idm_normal").cav_exec_cap == "idm_normal"
    final = _read_config(REPO / "configs" / "c4_policy_final.toml")
    assert DreamerConfig(**final["trainer"]).cav_exec_cap == "idm_normal:textbook"
    assert DreamerConfig(cav_exec_cap="idm_normal:legacy").cav_exec_cap == "idm_normal:legacy"
    monkeypatch.setattr(checkpoint_module, "load_checkpoint",
                        lambda path, **kw: SimpleNamespace(model=_model(), legacy_config={},
                                                           projection=None))
    monkeypatch.setenv(cav_motion.EXEC_CAP_ENV, "idm_normal")
    assert backend.load_backend("x.pt").model.cav_exec_cap == "idm_normal"
    from wms.control.deploy import parser
    action = next(a for a in parser()._actions if a.dest == "cav_exec_cap")
    assert {"idm_normal", "idm_normal:textbook", "idm_normal:legacy"} <= set(action.choices)
    from wms.control.rsu_check import RsuChecker
    import inspect
    assert "cav_exec_cap" in inspect.signature(RsuChecker).parameters
