"""Synthetic test (c): the eval/API correction summary uses the TRAIN window.

The evaluator and the scoring API roll out horizons far longer than training
(the correction head is trained on ``rollout_horizon_steps``).  These tests
prove the summary is built over the single-source
``model.constants.CORRECTION_WINDOW_STEPS`` window: steps beyond it cannot
change the summary or the outcome quantiles.  Synthetic weights only; no
checkpoint, pack or data file is opened.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from corr import CorrConfig
from corr.evaluation import trajectory_summary
from evalh.idea_a import IdeaAVariantPredictor
from model import constants as C
from train.g4_real import RealCorrBundle
from train.idea_a import IdeaATrainConfig
from wm_api.engine import outcome_head

_K = 3


def _bundle():
    torch.manual_seed(11)
    corr = RealCorrBundle(CorrConfig(d_model=16)).eval()
    return SimpleNamespace(corr=corr, state_enc=None,
                           cfg=SimpleNamespace(state_global_token=False))


def _window(n_states):
    return C.correction_window_steps(n_states)


def _long_rollout(*, steps: int = 29, garbage_tail: bool = True,
                  sample_axis: bool = False) -> dict:
    """A valid trajectory of ``steps`` transitions with a garbage tail.

    The first ``C.CORRECTION_WINDOW_STEPS + 1`` states are a coherent CAV
    trajectory; every later state keeps the SAME vehicle identity (so the
    survival mask alone would not exclude it) but is wildly displaced, so any
    consumer that did NOT slice to the registered window would see it.
    """
    T = int(steps)
    pos = torch.zeros(T + 1, 1, _K)
    spd = torch.zeros(T + 1, 1, _K)
    acc = torch.zeros(T, 1, _K)
    mask = torch.zeros(T + 1, 1, _K, dtype=torch.bool)
    ids = torch.full((T + 1, 1, _K), -1, dtype=torch.long)
    for t in range(T + 1):
        pos[t, 0] = torch.tensor([10.0 + 2.0 * t, 60.0 + 0.5 * t, 90.0])
        spd[t, 0] = torch.tensor([12.0, 11.0, 9.0])
        mask[t, 0, :2] = True
        ids[t, 0, :2] = torch.tensor([7, 8])
        if garbage_tail and t > _window(T + 1):
            pos[t, 0, :2] += 5000.0
            spd[t, 0, :2] += 300.0
    for t in range(T):
        acc[t, 0] = torch.tensor([0.5, -0.2, 0.0])
    kin = torch.zeros(1, _K, 7)
    kin[0, :, 0] = pos[0, 0, :]
    kin[0, :, 4] = spd[0, 0, :]
    kin[0, :, 1] = 1.0
    kin[0, :, 5] = acc[0, 0, :]
    kin[0, :, 6] = 5.0
    data = dict(positions=pos, speeds=spd, accel=acc, masks=mask,
                slot_ids=ids, t0_kin=kin, t0_mask=mask[0].clone(),
                t0_is_cav=torch.tensor([[True, True, False]]),
                t0_slot_id=ids[0].clone(),
                t0_counter=torch.zeros(1, 2, 30))
    if not sample_axis:
        return data
    per_sample = ("positions", "speeds", "accel", "masks", "slot_ids")
    return {k: (v.unsqueeze(0) if k in per_sample else v)
            for k, v in data.items()}


def _prefix(data: dict, states: int, *, sample_axis: bool = False) -> dict:
    keep = states + 1
    out = {}
    for key, value in data.items():
        if key == "accel":
            out[key] = value[:, :states] if sample_axis else value[:states]
        elif key in ("positions", "speeds", "masks", "slot_ids"):
            out[key] = value[:, :keep] if sample_axis else value[:keep]
        else:
            out[key] = value
    return out


def test_constant_is_the_training_horizon_default_and_caps_cleanly():
    assert C.CORRECTION_WINDOW_STEPS == 20
    assert IdeaATrainConfig().rollout_horizon_steps == C.CORRECTION_WINDOW_STEPS
    assert _window(1000) == C.CORRECTION_WINDOW_STEPS
    assert _window(4) == 3                  # short trajectory: use it all
    assert _window(1) == 1                  # never zero


def test_outcome_head_ignores_steps_beyond_the_training_window():
    bundle = _bundle()
    data = _long_rollout()
    window = _window(data["positions"].shape[0])
    pref = _prefix(data, window)
    args = ("t0_kin", "t0_mask", "t0_is_cav", "t0_slot_id")
    q_long, raw_long = outcome_head(
        bundle, data["positions"], data["speeds"], data["accel"],
        data["masks"], slot_ids=data["slot_ids"],
        **{k: data[k] for k in args})
    q_pref, raw_pref = outcome_head(
        bundle, pref["positions"], pref["speeds"], pref["accel"],
        pref["masks"], slot_ids=pref["slot_ids"],
        **{k: pref[k] for k in args})
    np.testing.assert_array_equal(q_long, q_pref)
    np.testing.assert_array_equal(raw_long, raw_pref)
    # sanity: the garbage tail WOULD change an unsliced summary
    unsliced = trajectory_summary(
        data["positions"], data["speeds"], data["accel"], data["masks"],
        slot_ids=data["slot_ids"], **{k: data[k] for k in args})
    assert not torch.equal(unsliced, torch.as_tensor(raw_long).reshape_as(
        unsliced))


def test_evalh_outcome_ignores_steps_beyond_the_training_window():
    bundle = _bundle()
    predictor = IdeaAVariantPredictor(bundle=bundle)
    rows = [SimpleNamespace(tau_s=float(C.TAUS[0]))]
    data = _long_rollout(sample_axis=True)
    window = _window(data["positions"].shape[1])
    pref = _prefix(data, window, sample_axis=True)
    q_long = predictor._outcome(rows, data)
    q_pref = predictor._outcome(rows, pref)
    np.testing.assert_array_equal(q_long, q_pref)


def test_short_rollout_is_unchanged_by_the_window():
    bundle = _bundle()
    data = _long_rollout(steps=3, garbage_tail=False)
    args = ("t0_kin", "t0_mask", "t0_is_cav", "t0_slot_id")
    q, raw = outcome_head(
        bundle, data["positions"], data["speeds"], data["accel"],
        data["masks"], slot_ids=data["slot_ids"],
        **{k: data[k] for k in args})
    expected = trajectory_summary(
        data["positions"], data["speeds"], data["accel"], data["masks"],
        slot_ids=data["slot_ids"], **{k: data[k] for k in args})
    np.testing.assert_array_equal(raw, expected[0].numpy())
