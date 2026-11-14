"""Counter history (design section 1, brief item 4).

Per entrance, 10 s departure bins over the last 300 s, from the physics
``departed_ids`` (source departures = insertions).  At tick t bin b covers
``[t - 300 + 10 b, t - 300 + 10 (b + 1))`` for b in 0..29; the current tick's
departure falls in the last bin (``dep <= t`` only).  Entrance is taken from
the departing vehicle's road edge: mainline entry ``1191554636`` -> entrance
0, ramp entry ``11553048#1`` -> entrance 1.
"""
from __future__ import annotations

import numpy as np

from . import constants as C
from .geometry import (ENTRANCE_MAINLINE, ENTRANCE_RAMP,
                       ENTRANCE_MAINLINE_EDGE, ENTRANCE_RAMP_EDGE)

_EDGE_TO_ENTRANCE = {
    ENTRANCE_MAINLINE_EDGE: ENTRANCE_MAINLINE,
    ENTRANCE_RAMP_EDGE: ENTRANCE_RAMP,
}


def entrance_of_edge(road_id: str) -> int:
    edge = str(road_id)
    if edge in _EDGE_TO_ENTRANCE:
        return _EDGE_TO_ENTRANCE[edge]
    # fall back on the downstream chain of the physical lanes
    if edge in ("1487606179", "1487606178"):
        return ENTRANCE_MAINLINE
    raise KeyError(f"road id {road_id!r} is not a known entrance edge")


class DepartureLog:
    """Departure (insertion) times per entrance, tick-indexed."""

    def __init__(self, t_origin: float):
        self.t_origin = float(t_origin)
        self.times = []          # list[float] departure times
        self.entrances = []      # list[int]

    def add(self, t: float, entrance: int) -> None:
        self.times.append(float(t))
        self.entrances.append(int(entrance))

    def arrays(self):
        return (np.asarray(self.times, np.float64),
                np.asarray(self.entrances, np.int64))

    def counter_at(self, t: float) -> np.ndarray:
        """float32 [2, 30] bin counts for departures in (t-300, t]."""
        out = np.zeros((C.N_ENTRANCES, C.COUNTER_BINS), np.float32)
        ts, ent = self.arrays()
        if len(ts) == 0:
            return out
        age = float(t) - ts
        keep = (age >= 0.0) & (age < C.COUNTER_BINS * C.COUNTER_BIN_S)
        if not keep.any():
            return out
        b = np.floor(age[keep] / C.COUNTER_BIN_S).astype(np.int64)
        e = ent[keep]
        np.add.at(out, (e, b), 1.0)
        return out

    def arrivals_by_tick(self, n_ticks: int) -> np.ndarray:
        """int16 [n_ticks, 2] departures per tick per entrance."""
        out = np.zeros((int(n_ticks), C.N_ENTRANCES), np.int16)
        ts, ent = self.arrays()
        for t, e in zip(ts, ent):
            k = int(round((t - self.t_origin) / C.DT_S))
            if 0 <= k < n_ticks:
                out[k, e] += 1
        return out
