"""E8 contract-3 tests: teacher-mode correction summary matches training.

The WM-N1 teacher mode must feed the correction head the SAME ``summary_raw``
the trainer built (``train/idea_a.py:1031-1085``) from the pinned packed
``cav_plan_ch`` speed/accel channels, de-scaled by
``features.idea_a.CAV_TGT_SPEED_SCALE`` / ``CAV_ACCEL_SCALE`` and
alive-since-t0 forward-filled, instead of the executed rollout's trajectory.
Controller mode keeps the executed-rollout summary (``outcome_head``).

Everything here is GENERATED under ``tmp_path``: synthetic shard arrays, a
random-init Idea A bundle and the frozen training window builder.  No CAL
shard, CAL split contents, HOLDOUT/chunk-5/J4B/INC1 data, real checkpoint or
SUMO is ever opened (R1/R6).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import train.idea_a as T
from evalh.protocol import N_QUANTILES
from evalh.scorer import predictors as P
from evalh.scorer.labels import planned_reads
from evalh.scorer.predictors import (CAV_MOTION_CONTROLLER, CAV_MOTION_TEACHER,
                                     ModelAPredictor, PredictorRefusal,
                                     head_quantiles_from_summary,
                                     training_summary_raw)
from evalh.tests import test_scorer_predictors as TSP

_K = 3
_N = 300
_HORIZON = 20

# capture the E7 fixture generator before the tests monkeypatch it.
_ORIG_ARRAYS = TSP._arrays


def _small_config(on: bool = True) -> T.IdeaATrainConfig:
    return T.IdeaATrainConfig(
        hdv_d_model=16, hdv_n_heads=2, hdv_n_layers=1, hdv_ffn=32,
        arrival_counter_channels=8, state_global_token=on, num_workers=0)


def _nondegenerate_arrays(arm):
    """The E7 generated shard arrays with NONTRIVIAL plan/alive channels.

    Deterministic (fixed RNG) so the writing pass and the assertions build the
    identical window.
    """
    a = dict(_ORIG_ARRAYS(arm))
    n, k = a["veh_kin"].shape[:2]
    rng = np.random.default_rng(20261004)
    kin = a["veh_kin"].astype(np.float32)
    kin[..., 0] = 50.0 + np.cumsum(rng.uniform(0.5, 2.0, size=n))[:, None]
    kin[..., 4] = 15.0 + 5.0 * rng.random((n, k)).astype(np.float32)
    kin[..., 5] = rng.uniform(-2.0, 1.5, size=(n, k)).astype(np.float32)
    a["veh_kin"] = kin.astype(np.float16)
    plan = np.zeros((n, k, 8), np.float16)
    plan[:, 0, 0] = rng.uniform(0.1, 1.0, size=n).astype(np.float16)  # speed/40
    plan[:, 0, 1] = rng.uniform(-1.0, 1.0, size=n).astype(np.float16)  # accel/3
    plan[:, 1, 0] = rng.uniform(0.2, 0.9, size=n).astype(np.float16)
    plan[:, 1, 1] = rng.uniform(-1.0, 1.0, size=n).astype(np.float16)
    a["cav_plan_ch"] = plan
    target = np.ones((n, k), np.uint8)
    target[10:15, 0] = 0        # CAV slot departs mid-window -> forward-fill
    target[5:, 1] = 0           # non-CAV slot departs early
    a["target_mask"] = target
    return a


class _FakeTrainCorpus:
    """Drive the REAL ``IdeaACorpus.window`` on generated arrays (test only)."""

    def __init__(self, arrays, horizon_steps):
        self._arrays = arrays
        self.horizon_steps = int(horizon_steps)
        self.split = "CALIBRATION"
        self._lane_of = T.IdeaACorpus._lane_of

    def _array(self, name):
        try:
            return self._arrays[name]
        except KeyError as exc:                                    # pragma: no cover
            raise AssertionError(
                f"unexpected training array read {name!r}") from exc

    def _run_end(self, tick):
        return int(self._arrays["veh_kin"].shape[0])


def _train_item(arrays, *, tick=0):
    k = int(arrays["veh_kin"].shape[1])
    return {
        "world": "w1", "arm": "native", "chunk": "chunk2",
        "split": "CALIBRATION", "t0_s": 300.0,
        "pair_key": "w1@t0=300", "policy": "native", "tick": int(tick),
        "veh_mask": torch.as_tensor(arrays["veh_mask"][tick].astype(bool)),
        "veh_step": torch.as_tensor(
            arrays["veh_step"][tick].astype(np.float32)),
        "accel": torch.zeros(k),
        "target_mask": torch.as_tensor(
            arrays["target_mask"][tick].astype(bool)),
        "lane": torch.zeros(k, dtype=torch.long),
        "counter": torch.as_tensor(
            arrays["counter"][tick].astype(np.float32)),
        "arrivals": torch.zeros(2),
        "outcome_quantiles": torch.zeros(5, 3, 19),
        "outcome_mask": torch.zeros(5, 3, dtype=torch.bool),
        "collision": torch.zeros(1),
    }


def _train_arrays_for_window(arrays):
    n, k = arrays["veh_kin"].shape[:2]
    return {
        "veh_kin": arrays["veh_kin"],
        "cav_plan_ch": arrays["cav_plan_ch"],
        "target_mask": arrays["target_mask"],
        "hdv_base": np.zeros((n, k, 6), np.float32),
        "event_tok": np.zeros((n, k, 8), np.float32),
        "event_mask": np.zeros((n, k, 8), np.float32),
        "accel_next": np.zeros((n, k), np.float32),
    }


def _training_summary(arrays, *, tick=0, n_steps=_HORIZON):
    """The ACTUAL training window builder's ``summary_raw``."""
    fake = _FakeTrainCorpus(_train_arrays_for_window(arrays),
                            horizon_steps=n_steps)
    out = T.IdeaACorpus.window(fake, _train_item(arrays, tick=tick))
    return out["summary_raw"]


def _tune(tmp_path, monkeypatch):
    monkeypatch.setattr(TSP, "_arrays", _nondegenerate_arrays)
    return TSP._tune_source(tmp_path, monkeypatch, "a" * 64)


# ------------------------------------------------------------- summary parity --
def test_teacher_summary_raw_equals_training_window(tmp_path, monkeypatch):
    _ctx, source = _tune(tmp_path, monkeypatch)
    _pos, item, run = source.selected[0]
    tick = int(item["tick"]) - int(run["tick_start"])
    arrays = _nondegenerate_arrays(item["arm"])
    train_summary = _training_summary(arrays, tick=tick)

    window = P.build_item_world(source, item, run)
    got = training_summary_raw(source, item, run, n_steps=_HORIZON,
                               is_cav=window.is_cav_t0)
    assert got.shape == (1, 16)
    assert _training_summary(arrays, tick=tick).shape == (1, 16)
    assert torch.allclose(got, train_summary, atol=1e-6), (
        float((got - train_summary).abs().max()))
    # the window really exercises the de-scaled plan + forward-fill path
    assert not torch.allclose(got, torch.zeros_like(got), atol=1e-6)
    assert not torch.allclose(got[0, :2], torch.zeros(2), atol=1e-6)


def test_teacher_head_output_equals_training_head(tmp_path, monkeypatch):
    _ctx, source = _tune(tmp_path, monkeypatch)
    _pos, item, run = source.selected[0]
    tick = int(item["tick"]) - int(run["tick_start"])
    arrays = _nondegenerate_arrays(item["arm"])
    summary_train = _training_summary(arrays, tick=tick)

    torch.manual_seed(23)
    bundle = T.build_idea_a_bundle(_small_config(True))
    bundle.corr.eval()
    if bundle.state_enc is not None:
        bundle.state_enc.eval()
    kin0 = torch.as_tensor(
        arrays["veh_kin"][tick].astype(np.float32)).unsqueeze(0)
    mask0 = torch.as_tensor(
        arrays["veh_mask"][tick].astype(bool)).unsqueeze(0)
    counter0 = torch.as_tensor(
        arrays["counter"][tick].astype(np.float32)).unsqueeze(0)

    # training-path head: global_token(batch) then corr(summary, token)
    trainer = T.IdeaATrainer(
        T.IdeaATrainConfig(out_dir="unused", num_workers=0,
                           state_global_token=True),
        corpus=None, bundle=bundle)
    token = trainer.global_token(
        {"kin": kin0, "kin_mask": mask0, "counter": counter0})
    q_train = bundle.corr(summary_train, token)["outcome_quantiles"][0]

    window = P.build_item_world(source, item, run)
    t0_kin, t0_mask, t0_counter = P._t0_state_channels(source, window)
    q_scorer = head_quantiles_from_summary(
        bundle, summary_train, t0_kin=t0_kin, t0_mask=t0_mask,
        t0_counter=t0_counter)
    assert np.allclose(
        q_scorer, q_train.detach().cpu().numpy(), atol=1e-6), (
        float(np.abs(q_scorer - q_train.detach().cpu().numpy()).max()))


# ------------------------------------------------------------------- modes ----
def test_summary_source_recorded_per_mode(tmp_path, monkeypatch):
    _ctx, source = _tune(tmp_path, monkeypatch)
    bound = TSP._load(TSP._checkpoint(tmp_path)[0])
    ctrl = ModelAPredictor(bound.bundle, source=source,
                           cav_motion=CAV_MOTION_CONTROLLER)
    teach = ModelAPredictor(bound.bundle, source=source,
                            cav_motion=CAV_MOTION_TEACHER)
    assert ctrl.summary_source == "rollout_executed"
    assert teach.summary_source == "training_packed_plan"

    rows = TSP._rows()
    q_ctrl = ctrl.predict_quantiles(rows)
    q_teach = teach.predict_quantiles(rows)
    assert q_ctrl.shape == (len(rows), N_QUANTILES)
    assert q_teach.shape == (len(rows), N_QUANTILES)
    assert np.isfinite(q_ctrl).all() and np.isfinite(q_teach).all()


def test_planned_reads_include_cav_plan_ch(tmp_path, monkeypatch):
    _ctx, source = _tune(tmp_path, monkeypatch)
    reads = planned_reads(source)
    assert ("chunk2/w1/native/cav_plan_ch.npy", "cav_plan_ch") in reads
    assert ("chunk2/w1/damping/cav_plan_ch.npy", "cav_plan_ch") in reads


def test_teacher_refuses_missing_cav_plan_ch(tmp_path, monkeypatch):
    _ctx, source = _tune(tmp_path, monkeypatch)
    bound = TSP._load(TSP._checkpoint(tmp_path)[0])
    key = next(iter(source.pins))
    source.pins[key]["files"].pop("cav_plan_ch.npy")
    teach = ModelAPredictor(bound.bundle, source=source,
                            cav_motion=CAV_MOTION_TEACHER)
    assert teach.summary_source == "training_packed_plan"
    with pytest.raises(PredictorRefusal, match="cav_plan_ch"):
        teach.predict_quantiles(TSP._rows())
