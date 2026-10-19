"""Per-window replay flags: merge-zone and CAV-follower windows (V10HORIZON).

A *merge-zone window* has at least one present recorded HDV in the ramp / merge lane
(lane index 2 of the one-hot lane block ``states[..., 1:4]``).  A *CAV-follower window*
has, at some step, a recorded HDV directly behind a present CAV in the same lane
(``states[..., 0]`` = longitudinal position gets the nearest predecessor).

Both flags are computed once from the memory-mapped window records (one window at a
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


def window_replay_flags(record: dict) -> tuple[bool, bool]:
    """``(merge_zone, cav_follower)`` of one window record."""
    states = np.asarray(record["states"])
    if states.ndim != 3 or states.shape[-1] <= 4:
        raise ReplayRefusal("replay flags need states [T, K, F>=5]")
    present = np.asarray(record["present"], dtype=bool)
    if present.shape != states.shape[:2]:
        raise ReplayRefusal("replay flags: present shape differs from states")
    is_cav = record.get("is_cav")
    cav = present if is_cav is None else (present & np.asarray(is_cav, dtype=bool))
    if cav.shape != present.shape:
        raise ReplayRefusal("replay flags: is_cav shape differs from present")
    lane = _lane_index(states)
    s = states[..., S_FEATURE].astype(np.float64)
    merge_zone = bool((present & (lane == RAMP_LANE)).any())
    cav_follower = _has_cav_follower(present, cav, lane, s)
    return merge_zone, cav_follower


def replay_flags(records: Iterable[dict]) -> dict:
    """``(merge_zone, cav_follower)`` bool arrays over an iterable of records."""
    merge, follower = [], []
    for record in records:
        m, f = window_replay_flags(record)
        merge.append(m)
        follower.append(f)
    return {"merge_zone": np.asarray(merge, dtype=bool),
            "cav_follower": np.asarray(follower, dtype=bool)}


def compute_replay_flags(dataset, *, progress: Optional[int] = None) -> dict:
    """Scan a dataset window by window and return the two flag arrays + counts.

    ``progress`` (windows) prints a stderr line every that many windows (0/None = quiet).
    """
    n = len(dataset)
    merge = np.zeros(n, dtype=bool)
    follower = np.zeros(n, dtype=bool)
    for i in range(n):
        m, f = window_replay_flags(dataset[i])
        merge[i], follower[i] = m, f
    return {"merge_zone": merge, "cav_follower": follower,
            "n_windows": n, "n_merge_zone": int(merge.sum()),
            "n_cav_follower": int(follower.sum()),
            "n_both": int((merge & follower).sum())}


__all__ = ["REPLAY_SCHEMA", "RAMP_LANE", "ReplayRefusal", "window_replay_flags",
           "replay_flags", "compute_replay_flags"]
