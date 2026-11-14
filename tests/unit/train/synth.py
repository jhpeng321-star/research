"""Deterministic synthetic batches and models built from the legacy objects."""
from __future__ import annotations

import numpy as np
import torch

from _compat import legacy


def small_model(mode="residual", lane_head="off", history_s=None, seed=9):
    model_mod = legacy("model")
    from output.wd01.wm_n1.hdv.config import HdvConfig

    torch.manual_seed(seed)
    model = model_mod.WmSModel(
        physics=model_mod.PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        rollout_mode=mode, lane_head=lane_head, history_s=history_s)
    if mode == "hybrid":
        with torch.no_grad():
            model.residual_net.cond_proj[-1].weight.uniform_(-0.2, 0.2)
    return model


def make_arrays(steps=24, k=6, n=2, cav=True, seed=3):
    """Platoon with a braking leader, one lane change, one CAV and a vanishing slot."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n, steps + 1, k, 7), dtype=np.float32)
    for b in range(n):
        s0 = -2000.0 + 30.0 * np.arange(k)[::-1] + rng.normal(0, 2, k)
        v = 12.0 + rng.normal(0, 0.5, (steps + 1, k))
        v[8:14, k - 1] -= np.linspace(0, 6, 6).tolist() * 1  # leader brakes
        v[14:, k - 1] = v[13, k - 1]
        v = np.clip(v, 0.5, None)
        a = np.zeros_like(v)
        a[1:] = (v[1:] - v[:-1]) / 0.5
        pos = s0[None] + np.cumsum(np.vstack((np.zeros((1, k)), v[:-1] * 0.5)), axis=0)
        states[b, ..., 0], states[b, ..., 4], states[b, ..., 5] = pos, v, a
        lane = np.zeros((steps + 1, k), dtype=np.int64)
        lane[:, k // 2:] = 1
        lane[12:, 1] = 1                      # one lane change
        states[b, ..., 1:4] = np.eye(3, dtype=np.float32)[lane]
        states[b, ..., 6] = 5.0
    present = np.ones((n, steps + 1, k), dtype=bool)
    present[:, 18:, 0] = False                # a slot leaves the road
    arrays = {"states": states, "present": present}
    if cav:
        is_cav = np.zeros(states.shape[:3], dtype=bool)
        is_cav[..., 2] = True
        arrays["is_cav"] = is_cav
    return arrays


def make_batch(history=False, entry=False, **kwargs):
    from output.wd01.wm_n1.wms.train import TrajectoryDataset, collate

    ds = TrajectoryDataset.from_arrays(make_arrays(**kwargs))
    records = [ds[i] for i in range(len(ds))]
    rng = np.random.default_rng(8)
    for r in records:
        if history:
            k = r["states"].shape[1]
            r["hist"] = rng.normal(0, 0.2, (k, 10, 6)).astype(np.float32)
            r["hist_mask"] = np.ones((k, 10), dtype=bool)
        if entry:
            from output.wd01.wm_n1.wms.entries import EntranceHistory

            feat = EntranceHistory.defaults().to_features().astype(np.float32)
            feat[0] = feat[1] = 5.0
            r["entry_feat"] = feat
    return collate(records)
