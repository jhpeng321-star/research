"""Shared CPU rollout machinery for the R10B checks (inference only, TRAIN states, entries off, recorded CAV arm replayed).

``init_arms`` loads the predictors of every arm into the worker process and wraps ``model.forward`` to capture, per
rollout step ``k_steps``, the final model acceleration and (when the courtesy head exists) the head offset, its inputs
and the pre-head acceleration.  ``roll_once`` returns the trajectories (rows 0..steps = state at the START of step i)
plus the captured per-step calls."""
from __future__ import annotations

import dataclasses
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

from common import DT, LEN_RAMP

G: dict = {}


def init_arms(specs, entries="off"):
    import torch
    import wm_eval_common as wc
    torch.set_num_threads(1)
    G["p"] = {}
    for name, path in specs:
        p, cap = wc.load_predictor(wc.guard(path), "AS_IS", hdv_exits="sumo", entries=entries)
        m = p.model
        cap["calls"] = []
        orig = m.forward

        def fwd(state, *a, _orig=orig, _cap=cap, **kw):
            out = _orig(state, *a, **kw)
            rec = {"k": int(kw.get("k_steps", len(_cap["calls"]))),
                   "accel": out["accel_mps2"].detach()[0].numpy().astype(np.float32)}
            if "courtesy_accel_mps2" in out:
                rec["offset"] = out["courtesy_accel_mps2"].detach()[0].numpy().astype(np.float32)
                rec["applies"] = out["courtesy_applies"].detach()[0].numpy().astype(bool)
                rec["x"] = out["courtesy_inputs"].detach()[0].numpy().astype(np.float32)
                rec["pre"] = out["accel_pre_courtesy_mps2"].detach()[0].numpy().astype(np.float32)
            _cap["calls"].append(rec)
            return out

        m.forward = fwd
        G["p"][name] = (p, cap)


def roll_once(name, rec, sample, steps, arm="native"):
    """-> dict(pos, lane, mask, v [steps+1, n]; is_cav, ids, t0, calls) ; rows are the state at the START of step i."""
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    p, cap = G["p"][name]
    t0, ids, kin, is_cav = t0_vehicles(rec)
    teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, steps, arm, p.backend.tools)
    if getattr(p, "flags", None) is not None:
        p.flags = dataclasses.replace(p.flags, lane_seed=int(sample))
    cap["calls"] = []
    p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm}, steps * DT, int(sample))
    r = cap["last"]
    n = len(ids)
    tr = teacher[0].numpy()
    rp = present_seq[0].numpy().astype(bool)
    f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
    return {
        "pos": f32(np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)),
        "lane": f32(np.concatenate([tr[:1, :, 1], r["lanes"][0, :, :n].numpy()], 0)),
        "mask": np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0),
        "v": f32(np.concatenate([tr[:1, :, 2], r["speeds"][0, :, :n].numpy()], 0)),
        "is_cav": np.asarray(is_cav, bool), "ids": list(ids), "t0": float(t0),
        "rec_pos": f32(tr[:, :, 0]), "rec_lane": f32(tr[:, :, 1]), "rec_v": f32(tr[:, :, 2]), "rec_mask": rp,
        "calls": list(cap["calls"]), "n": n, "dem": rec.get("demand_vph"), "pen": rec.get("penetration"),
    }


def traj_arrays(pos, lane, mask, v, is_cav, t0):
    """Arrays in the layout of diag_yield_extract.rows_from_arrays (acceleration = backward difference)."""
    lane_i = np.where((lane >= 0) & (lane <= 2), np.rint(lane), -1).astype(int)
    a = np.zeros_like(v, dtype=np.float64)
    a[1:] = (v[1:] - v[:-1]) / DT
    T, N = pos.shape
    vid = np.broadcast_to(np.arange(N)[None], (T, N)).copy()
    cav = np.broadcast_to(np.asarray(is_cav)[None], (T, N)).copy()
    tt = t0 + DT * np.arange(T)
    return pos.astype(np.float64), v.astype(np.float64), a, lane_i, mask.astype(bool), vid, cav, tt


def pairs_per_step(pos, lane, mask, v, physics):
    """Ramp partner (has_partner, partner slot) per (row, slot) from the state of every row."""
    import torch
    from wms.model.follower_courtesy import follower_courtesy_pairs
    T, N = pos.shape
    kin = torch.zeros(T, N, 8)
    kin[..., 0] = torch.as_tensor(pos)
    li = np.where((lane >= 0) & (lane <= 2), np.rint(lane), -1).astype(int)
    oh = np.zeros((T, N, 3), np.float32)
    for c in range(3):
        oh[..., c] = (li == c)
    kin[..., 1:4] = torch.as_tensor(oh)
    kin[..., 4] = torch.as_tensor(v)
    kin[..., 6] = LEN_RAMP
    pr = follower_courtesy_pairs(kin, torch.as_tensor(mask), physics)
    return pr["has_partner"].numpy(), pr["partner"].numpy()
