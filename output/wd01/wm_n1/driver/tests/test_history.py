"""Tests for ``driver.history`` (shapes, masks, identity, no future leakage)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from driver.history import (HISTORY_F, RollingHistoryBuffer, build_history,
                            pack_history_tensor, tick_gap_lead_dv)

VEH_F = 16


def _kin_row(s, lane, v, a):
    """One ``veh_step`` row with the P9 kinematic columns set."""
    row = np.zeros(VEH_F, dtype=np.float32)
    row[0] = s
    if lane >= 0:
        row[1 + lane] = 1.0
    row[4] = v
    row[5] = a
    row[15] = 1.0
    return row


def test_tick_gap_and_lead_dv():
    s = np.array([100.0, 130.0, 400.0])
    lane = np.array([0, 0, 0])
    v = np.array([10.0, 12.0, 15.0])
    present = np.array([True, True, True])
    gap, dv = tick_gap_lead_dv(s, lane, v, present)
    assert abs(gap[0] - (130.0 - 100.0 - 5.0)) < 1e-9
    assert abs(dv[0] - (10.0 - 12.0)) < 1e-9
    assert np.isinf(gap[2]) and dv[2] == 0.0


def test_build_history_shape_and_fields():
    t, k, h = 12, 2, 10
    s = np.zeros((t, k))
    s[:, 0], s[:, 1] = 100.0, 140.0
    v = np.full((t, k), 12.0)
    a = np.full((t, k), 0.5)
    lane = np.zeros((t, k), np.int64)
    present = np.ones((t, k), bool)
    ids = np.tile(np.array([10, 20]), (t, 1))
    hist = build_history(s, lane, v, a, present, ids, H=h)
    assert hist.shape == (k, h, HISTORY_F)
    assert np.all(hist[:, :, 6] == 1.0)              # present channel
    assert np.allclose(hist[:, :, 3], 0.0)           # lane
    assert abs(hist[0, -1, 4] - (140.0 - 100.0 - 5.0)) < 1e-9
    assert abs(hist[0, -1, 5] - 0.0) < 1e-9
    assert np.isclose(hist[1, -1, 4], 1000.0)        # no leader -> clipped max


def test_slot_reuse_does_not_mix_identities():
    t, k, h = 6, 1, 4
    s = np.tile(np.arange(t, dtype=float)[:, None], (1, k))
    v = np.full((t, k), 10.0)
    a = np.zeros((t, k))
    lane = np.zeros((t, k), np.int64)
    present = np.ones((t, k), bool)
    ids = np.array([[10], [10], [10], [99], [99], [99]])   # reuse at t=3
    hist = build_history(s, lane, v, a, present, ids, H=h)
    # window = ticks 2..5; id 99 first appears at tick 3
    assert list(hist[0, :, 6]) == [0.0, 1.0, 1.0, 1.0]
    assert hist[0, 0, 0] == 0.0                      # old vehicle not copied


def test_rolling_buffer_matches_direct_build():
    k, h = 3, 5
    buf = RollingHistoryBuffer(k, H=h)
    rng = np.random.default_rng(0)
    s_hist, v_hist, a_hist, lane_hist, pres_hist, id_hist = [], [], [], [], [], []
    for i in range(h):
        s = rng.random(k) * 10 + 100
        v = rng.random(k) * 10 + 10
        a = rng.standard_normal(k)
        lane = np.zeros(k, np.int64)
        present = np.ones(k, bool)
        ids = np.arange(k, dtype=np.int64)
        buf.push(s, v, a, lane, present, ids)
        s_hist.append(s), v_hist.append(v), a_hist.append(a)
        lane_hist.append(lane), pres_hist.append(present), id_hist.append(ids)
    direct = build_history(np.stack(s_hist), np.stack(lane_hist),
                           np.stack(v_hist), np.stack(a_hist),
                           np.stack(pres_hist), np.stack(id_hist), H=h)
    assert buf.tensor().shape == (k, h, HISTORY_F)
    assert np.allclose(buf.tensor(), direct)


def _write_pack(root, veh, mask, tick):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "veh_step.npy", veh)
    np.save(root / "veh_mask.npy", mask)
    items = [dict(world="w", arm="native", chunk="chunk4", split="TRAIN",
                  t0_s=48.0, pair_key="w@t0=48", tick=int(tick), plan_row=0,
                  outcome_rows=[])]
    manifest = {
        "schema": "wm_n1_pack/1", "k_max": int(veh.shape[1]),
        "arrays": {
            "veh_step": {"file": "veh_step.npy", "dtype": str(veh.dtype),
                         "shape": list(veh.shape)},
            "veh_mask": {"file": "veh_mask.npy", "dtype": str(mask.dtype),
                         "shape": list(mask.shape)},
        },
        "items": items,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_pack_hook_no_future_leakage(tmp_path):
    from train.dataset import MemmapCorpus

    t, k, tick = 8, 2, 5
    rng = np.random.default_rng(1)
    veh = np.zeros((t, k, VEH_F), np.float32)
    mask = np.ones((t, k), np.uint8)
    for i in range(t):
        for j in range(k):
            veh[i, j] = _kin_row(100.0 + i + j, 0, 10.0 + j, 0.0)
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_pack(old, veh.copy(), mask, tick)
    future = veh.copy()
    future[tick + 1:] = rng.standard_normal(future[tick + 1:].shape).astype(np.float32)
    _write_pack(new, future, mask, tick)

    h_old = pack_history_tensor(MemmapCorpus(old, split="TRAIN"), 0, H=4)
    h_new = pack_history_tensor(MemmapCorpus(new, split="TRAIN"), 0, H=4)
    assert h_old.shape == (k, 4, HISTORY_F)
    assert np.allclose(h_old, h_new)                 # future rows never read
    assert np.all(h_old[:, :, 6] == 1.0)
