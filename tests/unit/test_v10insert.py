"""V10INSERT: learned entrance insertion speed and acceptance hazard (default off)."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from wms.model import entries as E
from wms.model.entry_learned import (FEATURE_NAMES, LearnedEntryModel, _MLP,
                                     load_learned_entry, state_features)
from wms.model.entries import EntranceHistory, EntranceQueue, EntryConfig, TorchEntryProcessV2
from wms.physics.config import PhysicsConfig

DT = 0.5


def _model(*, speed_b2=20.0, hazard_b2=20.0):
    """Tiny deterministic fit: constant speed, and a hazard that increases with gap."""
    n = len(FEATURE_NAMES)
    zero = np.zeros(n, np.float64)
    w1 = np.zeros((n, 1), np.float64)
    w1[0, 0] = 1.0                                  # gap_50 drives the hazard logit
    speed = _MLP(mean=zero, scale=np.ones(n), w1=np.zeros((n, 1)), b1=np.zeros(1),
                 w2=np.zeros(1), b2=float(speed_b2), out_scale=1.0)
    hazard = _MLP(mean=zero, scale=np.ones(n), w1=w1, b1=np.zeros(1),
                  w2=np.ones(1), b2=float(hazard_b2), out_scale=1.0)
    return LearnedEntryModel(speed=speed, hazard=hazard)


def _state(cars, n_pad=12):
    k = len(cars) + n_pad
    s = torch.zeros(1, k, dtype=torch.float64)
    v = torch.zeros(1, k, dtype=torch.float64)
    lane = torch.zeros(1, k, dtype=torch.long)
    length = torch.full((1, k), 5.0, dtype=torch.float64)
    present = torch.zeros(1, k, dtype=torch.bool)
    for i, (si, vi, li) in enumerate(cars):
        s[0, i], v[0, i], lane[0, i], present[0, i] = si, vi, li, True
    pool = np.zeros(k, bool)
    pool[len(cars):] = True
    return {"s_m": s, "v_mps": v, "a_mps2": torch.zeros(1, k, dtype=torch.float64), "lane": lane,
            "length_m": length, "present": present,
            "is_cav": torch.zeros(1, k, dtype=torch.bool)}, pool


def _hist():
    return dataclasses.replace(EntranceHistory.defaults(), rate=np.asarray([0.0, 0.0]))


def _queue(backlog=(5, 0)):
    return EntranceQueue(backlog=np.asarray(backlog, float),
                         demand_rate=np.asarray([0.0, 0.0], float))


def test_switch_defaults_off_and_is_validated():
    assert PhysicsConfig().entry_learned is False
    assert EntryConfig().learned_entry is None
    assert PhysicsConfig(entry_learned=True).entry_learned is True
    with pytest.raises(ValueError):
        PhysicsConfig(entry_learned=1)


def test_shipped_fit_loads_and_responds_to_the_entrance_gap():
    model = load_learned_entry()
    blocked = model.insertion_speed(1.0, 0.0, 40.0, 0.65, 31.29, 0).reshape(-1)[0]
    free = model.insertion_speed(200.0, 0.0, 0.0, 0.65, 31.29, 0).reshape(-1)[0]
    ramp_free = model.insertion_speed(200.0, 0.0, 0.0, 0.1, 22.22, 1).reshape(-1)[0]
    assert blocked < 2.0 < 10.0 < free
    assert ramp_free > 15.0
    h_blocked = model.acceptance_hazard(0.0, 0.0, 40.0, 0.65, 31.29, 0).reshape(-1)[0]
    h_free = model.acceptance_hazard(200.0, 0.0, 40.0, 0.65, 31.29, 0).reshape(-1)[0]
    assert 0.0 <= h_blocked < h_free <= 1.0
    round_trip = LearnedEntryModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(round_trip.insertion_speed(200.0, 0.0, 0.0, 0.1, 22.22, 1),
                              ramp_free)


def test_features_are_finite_for_extreme_inputs():
    x = state_features(-5.0, 60.0, -3.0, 100.0, 0.0, 7.0)
    assert x.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(x))


def test_learned_acceptance_rejects_a_blocked_entrance():
    state, pool = _state([(E.ENTRY_S_M[0] + 5.0 + 0.1 + 5.0, 0.0, 0),
                          (E.ENTRY_S_M[1] + 5.0 + 0.1 + 5.0, 0.0, 1)])
    proc = TorchEntryProcessV2(histories=[_hist()], dt_s=DT, seed=0, batch=1,
                               cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like_v2",
                                               fresh_no_credit=True, learned_entry=_model(hazard_b2=-20.0)),
                               pool=pool, queues=[_queue((8, 0))])
    assert not proc.insert(state).any()
    assert proc.pending[0, 0] == 8
    assert proc.summary()["learned_entry"] is True


def test_learned_free_insertion_uses_the_fitted_speed_and_one_per_step():
    state, pool = _state([], n_pad=20)
    model = _model(speed_b2=20.0, hazard_b2=20.0)
    proc = TorchEntryProcessV2(histories=[_hist()], dt_s=DT, seed=0, batch=1,
                               cfg=EntryConfig(max_pending=10 ** 9, insert_rule="sumo_like_v2",
                                               fresh_no_credit=True, learned_entry=model),
                               pool=pool, queues=[_queue((5, 0))])
    new = proc.insert(state)
    assert int(new.sum()) == 1                       # one learned acceptance per entrance/step
    assert proc.pending[0, 0] == 4
    slot = int(np.flatnonzero(new.numpy()[0])[0])
    assert float(state["v_mps"][0, slot]) == pytest.approx(20.0)
    assert float(state["s_m"][0, slot]) == pytest.approx(E.ENTRY_S_M[int(state["lane"][0, slot])]
                                                         + 5.0 + E.SUMO_POSITION_EPS_M)


def test_queue_entry_process_accepts_the_learned_model():
    import types

    from wms.eval.entries_input import entry_process_queue

    rec = {"world": "w963004m1900p40t300__native", "entry_queue": [7, 2, 0.6, 0.1],
           "counter": np.ones((2, 30)).tolist(),
           "initial_state": {"time_s": 300.0, "vehicles": []}, "history": {"ids": []}}
    kin = np.zeros((2, 7), np.float32)
    kin[:, 0] = [-200.0, -100.0]
    kin[:, 2] = 1.0
    kin[:, 4] = 20.0
    kin[:, 6] = 5.0
    tools = types.SimpleNamespace(entries=E, counter_reader=None)
    model = _model()
    proc, _, _ = entry_process_queue(rec, 300.0, 60.0, 0, kin, tools, learned_entry=model)
    assert proc.cfg.learned_entry is model
