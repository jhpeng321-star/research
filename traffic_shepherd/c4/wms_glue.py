"""The single sanctioned ``traffic_shepherd -> wms`` edge (C4 SUMO runner glue).

Only optional C4 adapters live here and every import is lazy:

* :func:`wall_executor` - ``wms.physics.wall_exec`` (wall-law baseline in SUMO);
* :func:`driver_history_tensors` - the trainer's history featurisers
  (``wms.data.history.long_history_features``, ``wms.model.history.history_step_features``);
* :func:`contract` - ``wms.control.contract`` (RSU zones, group-signal width);
* :func:`trajectory_dataset` - ``TrajectoryDataset.from_pack`` (smoke load check);
* :func:`team_reward_terms` - ``wms.control.reward.team_reward`` on one SUMO physics step
  (saved-state branching metrics, :mod:`traffic_shepherd.c4.branching`);
* :func:`imagined_cooperation` - ``wms.control.coop_imagine`` (WM-vs-SUMO branch check).

The SUMO environment itself (env, conversion, recorder, collector) does not depend on
``wms``.  ``src`` is added to ``sys.path`` when ``PYTHONPATH=src`` was not set.
"""
from __future__ import annotations

import sys

from . import ROOT


def ensure_wms_importable() -> None:
    try:
        import wms  # noqa: F401
    except ImportError:
        src = str(ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)


def contract():
    ensure_wms_importable()
    from wms.control import contract as module
    return module


def trajectory_dataset(path, *, history_s=None, history_stride_s: float = 1.0):
    """``wms.data.dataset.TrajectoryDataset.from_pack`` (TRAIN npz windows of the recorder)."""
    ensure_wms_importable()
    from wms.data.dataset import TrajectoryDataset
    return TrajectoryDataset.from_pack(path, history_s=history_s,
                                       history_stride_s=history_stride_s)


def wall_executor(member_mask, wall_speed_mps, *, prior_controller):
    """``NumpyWallExecutor`` with law v2.2 merge-release (one wall, B = 1)."""
    ensure_wms_importable()
    from wms.physics.wall_exec import V22, NumpyWallExecutor
    return NumpyWallExecutor(member_mask, wall_speed_mps, prior_controller=prior_controller,
                             law=V22)


def driver_history_tensors(raw: dict, *, history_s, history_stride_s: float = 1.0,
                           history_len: int = 10, dt_s: float = 0.5):
    """``(driver_history [1,K,H,F], mask [1,K,H])`` from a slot-indexed raw history.

    ``raw`` = ``{"kin" [T,K,7], "mask" [T,K], "target_mask" [T-1,K]}`` ending at t0.
    ``history_s`` set: ``long_history_features`` (F = 9, the long-history trainer path).
    Otherwise the rolling ``history_step_features`` window (F = 6, ``history_len`` ticks)
    with the trainer's backward same-vehicle continuity and leader gap.
    """
    ensure_wms_importable()
    import torch
    from wms.data.history import long_history_features
    from wms.model.history import history_step_features

    kin = torch.as_tensor(raw["kin"], dtype=torch.float32)
    present = torch.as_tensor(raw["mask"], dtype=torch.bool)
    target = torch.as_tensor(raw["target_mask"], dtype=torch.bool)
    if history_s is not None:
        feat, mask = long_history_features(kin, present, target, present[-1],
                                           history_s=history_s,
                                           history_stride_s=history_stride_s, dt_s=dt_s)
        return feat[None], mask[None]
    t, k = present.shape
    same = torch.ones(t, k, dtype=torch.bool)
    for j in range(t - 2, -1, -1):
        same[j] = same[j + 1] & target[j]
    valid = same & present
    gap = torch.full((t, k), 250.0)
    for ti in range(t):
        lane = kin[ti, :, 1:4].argmax(-1)
        s_ = kin[ti, :, 0]
        ds = s_.unsqueeze(0) - s_.unsqueeze(1)
        ok = valid[ti].unsqueeze(0) & (lane.unsqueeze(0) == lane.unsqueeze(1)) & (ds > 0)
        g = torch.where(ok, ds - kin[ti, :, 6].unsqueeze(0), torch.full_like(ds, 250.0))
        gap[ti] = g.min(-1).values.clamp(0.3, 250.0)
    feat = history_step_features(kin, gap)
    use = min(t, history_len)
    out = torch.zeros(k, history_len, feat.shape[-1])
    out_mask = torch.zeros(k, history_len, dtype=torch.bool)
    out[:, history_len - use:] = feat[-use:].transpose(0, 1)
    out_mask[:, history_len - use:] = valid[-use:].transpose(0, 1)
    return out[None], out_mask[None]


def team_reward_terms(state: dict, next_state: dict, accel, weights=None) -> dict:
    """``wms.control.reward.team_reward`` (default ``RewardWeights``) on numpy ``[K]`` arrays.

    ``state`` / ``next_state``: ``s_m``, ``v_mps``, ``lane`` (int, -1 invalid), ``present``
    (bool), ``length_m``; slot identity must be preserved (exits carried in ``next_state``).
    Returns the seven contract metrics as Python floats.
    """
    ensure_wms_importable()
    import numpy as np
    import torch
    from wms.control.contract import RewardWeights
    from wms.control.reward import team_reward

    def tensors(d):
        return {"s_m": torch.as_tensor(np.asarray(d["s_m"], np.float32)[None]),
                "v_mps": torch.as_tensor(np.asarray(d["v_mps"], np.float32)[None]),
                "lane": torch.as_tensor(np.asarray(d["lane"], np.int64)[None]),
                "present": torch.as_tensor(np.asarray(d["present"], bool)[None]),
                "length_m": torch.as_tensor(np.asarray(d["length_m"], np.float32)[None])}

    with torch.no_grad():
        out = team_reward(tensors(state), tensors(next_state),
                          torch.as_tensor(np.asarray(accel, np.float32)[None]),
                          weights if weights is not None else RewardWeights())
    return {key: float(value[0]) for key, value in out.items()}


def imagined_cooperation(*args, **kwargs):
    """``wms.control.coop_imagine.imagined_cooperation`` (lazy)."""
    ensure_wms_importable()
    from wms.control.coop_imagine import imagined_cooperation as fn
    return fn(*args, **kwargs)
