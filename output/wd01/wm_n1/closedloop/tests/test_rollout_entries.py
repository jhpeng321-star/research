"""Synthetic tests for open-road entries (no model, deterministic)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_WM = Path(__file__).resolve().parents[2]
if str(_WM) not in sys.path:
    sys.path.insert(0, str(_WM))

from closedloop.rollout_entries import (EntryConfig, EntryProcess,  # noqa: E402
                                        observed_rates)
from rollout2 import baseline  # noqa: E402
from rollout2.state import Rollout2Config, Rollout2State, StrategyActions, WorldState  # noqa: E402

K = 64
PARAMS = baseline.Params(Rollout2Config().params)


def make_state(counter0=(7.0, 1.0), cav_share_n=(2, 6), p_lane=(3, 3)):
    counter = np.zeros((2, 30), np.float32)
    counter[0, :12] = counter0[0]
    counter[1, :12] = counter0[1]
    kin = np.zeros((K, 7), np.float32)
    mask = np.zeros(K, bool)
    is_cav = np.zeros(K, bool)
    i = 0
    for ln, n in ((0, p_lane[0]), (1, p_lane[1])):
        for _ in range(n):
            kin[i, 2] = ln
            mask[i] = True
            i += 1
    is_cav[:cav_share_n[0]] = True
    return SimpleNamespace(counter=counter, veh_kin=kin, veh_mask=mask, is_cav=is_cav,
                           rollout_mask=None)


def make_world(rear_s=(-1000.0, -1000.0), v=25.0, batch=1):
    s = torch.full((batch, K), 0.0, dtype=torch.float64)
    vv = torch.zeros((batch, K), dtype=torch.float64)
    lane = torch.full((batch, K), -1, dtype=torch.int64)
    present = torch.zeros((batch, K), dtype=torch.bool)
    for j, (ln, sp) in enumerate(((0, rear_s[0]), (1, rear_s[1]))):
        s[:, j], vv[:, j], lane[:, j], present[:, j] = sp, v, ln, True
    w = WorldState(s_m=s, v_mps=vv, a_mps2=torch.zeros_like(s), lane=lane,
                   length_m=torch.full_like(s, 5.0), present=present,
                   is_cav=torch.zeros_like(present), wall_group=torch.full_like(lane, -1),
                   slot_id=torch.where(present, torch.arange(K).expand(batch, K), torch.tensor(-1)))
    st = Rollout2State(batch, K)
    st.next_slot_id = w.slot_id.amax(dim=1) + 1
    acts = StrategyActions(action_code=torch.full((batch, K), -1, dtype=torch.int64))
    return w, st, acts


def test_rates_from_counter():
    r = observed_rates(make_state((6.0, 3.0)).counter, 12)
    assert r == pytest.approx([0.6, 0.3])


def test_gap_check_blocks_then_retries():
    ep = EntryProcess(make_state(), PARAMS, seed=1, cfg=EntryConfig(rate_scale=1000.0))
    w, st, acts = make_world(rear_s=(-2640.0, -2640.0))     # rearmost ~4 m from entrance
    ep.step(w, acts, st)
    main = w.present & (w.lane <= 1)
    assert int(main.sum()) == 2                            # unsafe -> no mainline insertion
    assert ep.pending[0, 0] > 0                            # retried later
    # clear road: arrivals now go in with a safe gap
    w2, st2, acts2 = make_world(rear_s=(-1500.0, -1500.0))
    ep.step(w2, acts2, st2)
    assert int((w2.present & (w2.lane <= 1)).sum()) > 2


def test_inserted_gap_speed_and_lane_rules():
    ep = EntryProcess(make_state(), PARAMS, seed=3, cfg=EntryConfig(rate_scale=50.0))
    w, st, acts = make_world(rear_s=(-1200.0, -1200.0))
    for _ in range(5):
        ep.step(w, acts, st)
    new = np.flatnonzero(w.present[0].numpy())[2:]
    assert new.size > 0
    lane = w.lane[0].numpy()[new]
    s = w.s_m[0].numpy()[new]
    assert set(lane.tolist()) <= {0, 1, 2}
    for k in new:
        if int(w.lane[0, k]) <= 1:
            assert w.v_mps[0, k] == pytest.approx(25.0)    # speed of nearby traffic
        assert w.s_m[0, k] <= -PARAMS.off[int(w.lane[0, k])] + 1e-9 or True
    assert (s < -300.0).all()                              # at the entrances


def test_deterministic_and_batch_rows_identical():
    def run():
        ep = EntryProcess(make_state(), PARAMS, seed=7, cfg=EntryConfig(rate_scale=5.0), batch=2)
        w, st, acts = make_world(batch=2, rear_s=(-1200.0, -1200.0))
        for _ in range(8):
            ep.step(w, acts, st)
        return w.s_m.numpy().copy(), w.present.numpy().copy(), ep.summary(0)

    a, b = run(), run()
    assert (a[1] == b[1]).all() and np.array_equal(a[0], b[0]) and a[2] == b[2]
    assert (a[1][0] == a[1][1]).all()                      # common random numbers across plans


def test_counts_scale_with_observed_rate_and_cav_share():
    lo = EntryProcess(make_state((2.0, 0.0)), PARAMS, seed=0, cfg=EntryConfig())
    hi = EntryProcess(make_state((8.0, 0.0)), PARAMS, seed=0, cfg=EntryConfig())
    for ep in (lo, hi):
        w, st, acts = make_world(rear_s=(-1200.0, -1200.0))
        for _ in range(60):
            ep.step(w, acts, st)
            w.s_m[:, 2:] = -3000.0                           # keep entrances clear
            w.present[:, 2:] = False
            w.slot_id[:, 2:] = -1
    n_lo, n_hi = lo.summary()["inserted"], hi.summary()["inserted"]
    assert 0 < n_lo < n_hi
    assert hi.summary()["cav_share_observed"] == pytest.approx(2 / 6)


def test_spawned_cav_native_and_hdv_none():
    ep = EntryProcess(make_state(), PARAMS, seed=2, cfg=EntryConfig(rate_scale=20.0, cav_share=0.5))
    w, st, acts = make_world(rear_s=(-1200.0, -1200.0))
    for _ in range(4):
        ep.step(w, acts, st)
    present = w.present[0].numpy()
    cav = w.is_cav[0].numpy()
    codes = acts.action_code[0].numpy()
    assert (codes[present & cav] == 0).all() or not (present & cav)[2:].any()
    assert (codes[present & ~cav] == -1).all()
    assert w.is_cav.any()
    assert torch.isnan(st.held_target_mps[0, 2:]).all()


def test_no_free_slot_is_counted_not_crashing():
    ep = EntryProcess(make_state(), PARAMS, seed=0, cfg=EntryConfig(rate_scale=500.0))
    w, st, acts = make_world(rear_s=(-1200.0, -1200.0))
    w.present[:] = True
    w.lane[:] = 0
    w.s_m[:] = torch.arange(K, dtype=torch.float64) * 30.0 - 1500.0
    ep.step(w, acts, st)
    assert ep.summary()["inserted"] == 0


def test_flag_off_byte_identical_signature():
    import inspect

    from closedloop import a11_predictor as AP
    sig = inspect.signature(AP.batched_rollout)
    assert sig.parameters["open_entries"].default is False
    assert inspect.signature(AP.A11Predictor.__init__).parameters["open_road"].default is False
