"""Per-slot pre-t0 history and entry features read from a pack (training windows only).

``prior_history`` reconstructs, from ticks <= t0 of the memory-mapped pack, the motion
history that the driver encoder / history lane head consume.  The pure featurisation
(``history_spec``, ``long_history_features``) lives in :mod:`wms.data.history` and is shared
with evaluation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from wms.core.constants import HISTORY_F, HISTORY_H
from wms.data.corpus import PackCorpus
from wms.data.history import history_spec, history_step_features, long_history_features


def _same_lane_leader_gap(kin: Tensor, valid: Tensor) -> Tensor:
    """Per-step gap to the nearest valid same-lane leader, clamped to [0.3, 250] m."""
    t, k = kin.shape[:2]
    lane = kin[..., 1:4].argmax(-1)
    position = kin[..., 0]
    gap = torch.full((t, k), 250.0)
    for ti in range(t):
        ds = position[ti].unsqueeze(0) - position[ti].unsqueeze(1)      # s_j - s_i
        ok = (valid[ti].unsqueeze(0) & (lane[ti].unsqueeze(0) == lane[ti].unsqueeze(1))
              & (ds > 0))
        g = torch.where(ok, ds - kin[ti, :, 6].unsqueeze(0), torch.full_like(ds, 250.0))
        gap[ti] = g.min(-1).values.clamp(0.3, 250.0)
    return gap


def prior_history(corpus: PackCorpus, tick: int, present_now: np.ndarray, *,
                  history_s=None, history_stride_s=1.0):
    """Real pre-t0 motion history ``[K,H,F]`` + validity ``[K,H]`` (ticks <= t0 only).

    A slot's entry at ``tick-j`` is valid only if the same vehicle occupied it:
    ``target_mask[u] == 1`` iff the slot's vehicle is still present at ``u+1``, so
    validity is the backward cumulative product of ``target_mask`` and presence.
    Never crosses the run start.
    """
    native, _, _ = history_spec(history_s, history_stride_s)
    lo = max(corpus.run_start(tick), tick - native + 1)
    kin = torch.as_tensor(np.asarray(corpus.array("veh_kin")[lo:tick + 1], np.float32))
    pres = torch.as_tensor(np.asarray(corpus.array("veh_mask")[lo:tick + 1]).astype(bool))
    t, k = kin.shape[:2]
    tm = torch.as_tensor(np.asarray(corpus.array("target_mask")[lo:tick], bool))         if tick > lo else torch.zeros(0, k, dtype=torch.bool)
    if history_s is not None:
        feat, mask = long_history_features(kin, pres, tm, torch.as_tensor(present_now),
                                           history_s=history_s,
                                           history_stride_s=history_stride_s)
        return feat.numpy(), mask.numpy()
    same = torch.ones(t, k, dtype=torch.bool)       # same vehicle as at t0
    for j in range(t - 2, -1, -1):                  # row j -> j+1 continuity
        same[j] = same[j + 1] & tm[j]
    valid = same & pres
    valid[-1] = torch.as_tensor(present_now)
    feat = history_step_features(kin, _same_lane_leader_gap(kin, valid))      # [t,K,F]
    out = torch.zeros(k, HISTORY_H, HISTORY_F)
    mask = torch.zeros(k, HISTORY_H, dtype=torch.bool)
    out[:, HISTORY_H - t:] = feat.transpose(0, 1)
    mask[:, HISTORY_H - t:] = valid.transpose(0, 1)
    return out.numpy(), mask.numpy()


def entry_features(corpus: PackCorpus, item: dict) -> np.ndarray:
    """Observed pre-t0 entry-history feature vector for the entries-v2 trainer.

    Reads only the window's t0 ``counter`` and its pre-t0 ``veh_kin``/``veh_mask``/
    ``target_mask`` ticks (t <= t0).  A slot whose vehicle first appears at tick ``u``
    (present at ``u`` but not the same vehicle continued from ``u-1``) is an observed
    entrant; its entrance, speed, same-lane gap to the leader and the entry times are
    collected and passed to ``wms.model.entries.build_entrance_history``.  No recorded
    future is read.  (Lazy import: the entries model is the one sanctioned data->model edge.)
    """
    from wms.model.entries import build_entrance_history

    tick = int(item["tick"])
    counter = np.asarray(corpus.array("counter")[tick], np.float32)
    lo = max(corpus.run_start(tick), tick - 120)          # bounded pre-t0 history (60 s)
    kin = np.asarray(corpus.array("veh_kin")[lo:tick + 1], np.float32)
    mask = np.asarray(corpus.array("veh_mask")[lo:tick + 1]).astype(bool)
    tm = (np.asarray(corpus.array("target_mask")[lo:tick], bool)
          if tick > lo else np.zeros((0, kin.shape[1]), bool))
    times: list = []
    entrance: list = []
    speed: list = []
    local_speed: list = []
    lane_obs: list = []
    gaps: list = []
    gap_ent: list = []
    for j in range(kin.shape[0]):
        pres = mask[j]
        prev_same = np.zeros_like(pres) if j == 0 else (mask[j - 1] & tm[j - 1])
        lanes_j = np.argmax(kin[j, :, 1:4], axis=-1)
        for k in np.flatnonzero(pres & ~prev_same):
            ln = int(lanes_j[k])
            if ln not in (0, 1, 2):
                continue
            e = 0 if ln in (0, 1) else 1
            v_self = float(kin[j, k, 4])
            same_lane = pres & (lanes_j == ln)
            idx = np.flatnonzero(same_lane)
            idx = idx[np.argsort(kin[j, idx, 0])]
            local = float(kin[j, idx[:3], 4].mean()) if idx.size else v_self
            ahead = np.flatnonzero(same_lane & (kin[j, :, 0] > kin[j, k, 0]))
            if ahead.size:
                lead = ahead[np.argmin(kin[j, ahead, 0])]
                gap = float(kin[j, lead, 0] - kin[j, k, 0] - kin[j, k, 6])
                if gap > 0.0:
                    gaps.append(gap)
                    gap_ent.append(e)
            times.append((lo + j) * 0.5)
            entrance.append(e)
            speed.append(v_self)
            local_speed.append(max(local, 1.0))
            # One lane per observed entrant (aligned with ``entrance``); the builder keeps
            # only the mainline (e == 0) lanes for lane_p1.
            lane_obs.append(ln)
    history = build_entrance_history(
        counter=counter, times=np.asarray(times, np.float64),
        entrance=np.asarray(entrance, np.int64),
        speed=np.asarray(speed, np.float64),
        local_speed=np.asarray(local_speed, np.float64),
        lane=(np.asarray(lane_obs, np.int64) if lane_obs else None),
        gap=np.asarray(gaps, np.float64),
        gap_entrance=np.asarray(gap_ent, np.int64), t0_s=float(tick) * 0.5)
    return history.to_features().astype(np.float32)
