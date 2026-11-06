"""Per-window replay flags: merge-zone, CAV-follower and HDV brake-event windows.

A *merge-zone window* has, at some step, a present recorded HDV in the ramp / merge lane
(lane index 2 of the one-hot lane block ``states[..., 1:4]``) whose distance to the
acceleration-lane end is below ``MERGE_ZONE_END_M`` metres (V10ROUND1: distance to
``physics.merge_zone_s1_m`` in the model frame, default 40 m).  Merely containing a
lane-2 HDV upstream is not enough (that flagged ~every window).  A *CAV-follower window*
has, at some step, a recorded HDV directly behind a present CAV in the same lane
(``states[..., 0]`` = longitudinal position gets the nearest predecessor).  A
*brake-event window* (V10BRAKEWAVE) has, at some step, a present recorded HDV leader
decelerating below ``BRAKE_EVENT_ACCEL_MPS2`` with a present follower in the same lane
within ``BRAKE_EVENT_MAX_GAP_M`` metres.

All flags are computed once from the memory-mapped window records (one window at a
time, exactly like :mod:`wms.data.jam`) and feed :class:`wms.train.replay.PriorityReplayDataset`.
Nothing here reads CAL / HOLDOUT / sealed / chunk-5 / J4B / INC1 bytes.

The OFF path never calls this module.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from wms.core.errors import TrainDataRefusal

REPLAY_SCHEMA = "wms_replay_windows/1"
#: Lane channel of the ramp / merge lane in the one-hot lane block ``states[..., 1:4]``.
RAMP_LANE = 2
LANE_SLICE = slice(1, 4)
S_FEATURE = 0
A_FEATURE = 5
#: V10BRAKEWAVE: leader acceleration of a recorded HDV-led braking event.
BRAKE_EVENT_ACCEL_MPS2 = -1.5
#: V10BRAKEWAVE: maximum same-lane leader-follower gap of a brake-event pair.
BRAKE_EVENT_MAX_GAP_M = 40.0
#: V10ROUND1: distance-to-end band (m) that makes a lane-2 HDV a merge-zone vehicle
#: (``data.replay.merge_zone_end_m``; the end itself is ``physics.merge_zone_s1_m``).
MERGE_ZONE_END_M = 40.0
#: Default model-frame acceleration-lane end (= ``PhysicsConfig.merge_zone_s1_m``).
MERGE_ZONE_S1_M = 75.44


class ReplayRefusal(TrainDataRefusal):
    """A replay-window contract violation (fail closed)."""


def _lane_index(states: np.ndarray) -> np.ndarray:
    onehot = states[..., LANE_SLICE]
    if onehot.shape[-1] < 3:
        raise ReplayRefusal("replay flags need the 3-lane one-hot block states[..., 1:4]")
    return onehot.argmax(axis=-1)


def _has_cav_follower(present: np.ndarray, cav: np.ndarray, lane: np.ndarray,
                      s: np.ndarray) -> bool:
    """True when some present CAV has a present non-CAV slot directly behind it."""
    for t in range(present.shape[0]):
        p, cv = present[t], cav[t]
        if not cv.any():
            continue
        cav_slots = np.flatnonzero(cv)
        for c in cav_slots:
            behind = p & ~cv & (lane[t] == lane[t, c]) & (s[t] < s[t, c] - 1e-6)
            if behind.any():
                return True
    return False


def window_brake_event(record: dict, *, accel_mps2: float = BRAKE_EVENT_ACCEL_MPS2,
                       max_gap_m: float = BRAKE_EVENT_MAX_GAP_M) -> bool:
    """True when a recorded HDV leader brakes hard with a same-lane follower <= 40 m behind."""
    states = np.asarray(record["states"])
    if states.ndim != 3 or states.shape[-1] <= A_FEATURE:
        raise ReplayRefusal("brake-event flags need states [T, K, F>=6]")
    present = np.asarray(record["present"], dtype=bool)
    if present.shape != states.shape[:2]:
        raise ReplayRefusal("brake-event flags: present shape differs from states")
    is_cav = record.get("is_cav")
    cav = (np.zeros_like(present) if is_cav is None
           else np.asarray(is_cav, dtype=bool))
    if cav.shape != present.shape:
        raise ReplayRefusal("brake-event flags: is_cav shape differs from present")
    lane = _lane_index(states)
    on_road = states[..., LANE_SLICE].sum(-1) > 0
    s = states[..., S_FEATURE].astype(np.float64)
    a = states[..., A_FEATURE].astype(np.float64)
    leader_ok = present & on_road & ~cav & (a < float(accel_mps2))
    for t in range(present.shape[0]):
        ldr, p = leader_ok[t], present[t] & on_road[t]
        if not ldr.any():
            continue
        s_t, lane_t = s[t], lane[t]
        gap = s_t[:, None] - s_t[None, :]
        ahead = (p[None, :] & (lane_t[:, None] == lane_t[None, :])
                 & (gap > 0.0) & (gap <= float(max_gap_m)))
        if (ldr[:, None] & ahead).any():
            return True
    return False


def _merge_zone(present: np.ndarray, lane: np.ndarray, s: np.ndarray,
                hdv: np.ndarray, *, merge_end_m: float,
                merge_zone_s1_m: float) -> bool:
    """True when a present HDV in the ramp lane sits in the last ``merge_end_m`` metres."""
    near_end = (merge_zone_s1_m - s)
    hit = present & hdv & (lane == RAMP_LANE) & (near_end >= 0.0) & (near_end < float(merge_end_m))
    return bool(hit.any())


def window_replay_flags(record: dict, *, merge_end_m: float = MERGE_ZONE_END_M,
                        merge_zone_s1_m: float = MERGE_ZONE_S1_M) -> tuple[bool, bool]:
    """``(merge_zone, cav_follower)`` of one window record.

    ``merge_zone`` requires a present HDV (not CAV) in lane 2 within ``merge_end_m`` metres
    of ``merge_zone_s1_m`` at some step (V10ROUND1 definition).
    """
    states = np.asarray(record["states"])
    if states.ndim != 3 or states.shape[-1] <= 4:
        raise ReplayRefusal("replay flags need states [T, K, F>=5]")
    present = np.asarray(record["present"], dtype=bool)
    if present.shape != states.shape[:2]:
        raise ReplayRefusal("replay flags: present shape differs from states")
    is_cav = record.get("is_cav")
    if is_cav is None:
        cav = present                  # legacy follower semantics without CAV labels
        hdv = present                  # ... and every present vehicle counts as an HDV
    else:
        cav = present & np.asarray(is_cav, dtype=bool)
        hdv = present & ~np.asarray(is_cav, dtype=bool)
    if cav.shape != present.shape:
        raise ReplayRefusal("replay flags: is_cav shape differs from present")
    lane = _lane_index(states)
    s = states[..., S_FEATURE].astype(np.float64)
    merge_zone = _merge_zone(present, lane, s, hdv, merge_end_m=float(merge_end_m),
                             merge_zone_s1_m=float(merge_zone_s1_m))
    cav_follower = _has_cav_follower(present, cav, lane, s)
    return merge_zone, cav_follower


def window_cav_share(record: dict) -> float:
    """R14 D3: realized CAV share of the window's FIRST row (present CAVs / present slots).

    Sampling bookkeeping only (never a model input).  A window without CAV labels or
    without any present slot fails closed (it cannot be classified)."""
    present = np.asarray(record["present"], dtype=bool)
    is_cav = record.get("is_cav")
    if is_cav is None or present.ndim != 2 or not present[0].any():
        raise ReplayRefusal("low-share replay flag needs is_cav and a present first row")
    cav = np.asarray(is_cav, dtype=bool)
    if cav.shape != present.shape:
        raise ReplayRefusal("low-share replay flag: is_cav shape differs from present")
    return float((present[0] & cav[0]).sum() / present[0].sum())


def replay_flags(records: Iterable[dict], *, merge_end_m: float = MERGE_ZONE_END_M,
                 merge_zone_s1_m: float = MERGE_ZONE_S1_M) -> dict:
    """``(merge_zone, cav_follower, brake_event)`` bool arrays over an iterable of records."""
    merge, follower, brake = [], [], []
    for record in records:
        m, f = window_replay_flags(record, merge_end_m=merge_end_m,
                                   merge_zone_s1_m=merge_zone_s1_m)
        merge.append(m)
        follower.append(f)
        brake.append(window_brake_event(record))
    return {"merge_zone": np.asarray(merge, dtype=bool),
            "cav_follower": np.asarray(follower, dtype=bool),
            "brake_event": np.asarray(brake, dtype=bool)}


def compute_replay_flags(dataset, *, progress: Optional[int] = None,
                         merge_end_m: float = MERGE_ZONE_END_M,
                         merge_zone_s1_m: float = MERGE_ZONE_S1_M,
                         low_share_max: Optional[float] = None) -> dict:
    """Scan a dataset window by window and return the flag arrays + counts.

    ``progress`` (windows) prints a stderr line every that many windows (0/None = quiet).
    R14 D3 ``low_share_max`` (default ``None`` = not computed, output unchanged): also flag
    windows whose first-row realized CAV share is <= ``low_share_max``.
    """
    n = len(dataset)
    merge = np.zeros(n, dtype=bool)
    follower = np.zeros(n, dtype=bool)
    brake = np.zeros(n, dtype=bool)
    low = np.zeros(n, dtype=bool) if low_share_max is not None else None
    for i in range(n):
        record = dataset[i]
        m, f = window_replay_flags(record, merge_end_m=merge_end_m,
                                   merge_zone_s1_m=merge_zone_s1_m)
        merge[i], follower[i] = m, f
        brake[i] = window_brake_event(record)
        if low is not None:
            low[i] = window_cav_share(record) <= float(low_share_max)
    out = {"merge_zone": merge, "cav_follower": follower, "brake_event": brake,
           "merge_end_m": float(merge_end_m), "merge_zone_s1_m": float(merge_zone_s1_m),
           "n_windows": n, "n_merge_zone": int(merge.sum()),
           "n_cav_follower": int(follower.sum()), "n_brake_event": int(brake.sum()),
           "n_both": int((merge & follower).sum())}
    if low is not None:
        out.update(low_share=low, n_low_share=int(low.sum()), low_share_max=float(low_share_max))
    return out


__all__ = ["REPLAY_SCHEMA", "RAMP_LANE", "BRAKE_EVENT_ACCEL_MPS2",
           "BRAKE_EVENT_MAX_GAP_M", "MERGE_ZONE_END_M", "MERGE_ZONE_S1_M", "ReplayRefusal",
           "window_replay_flags", "window_brake_event", "window_cav_share", "replay_flags",
           "compute_replay_flags"]
