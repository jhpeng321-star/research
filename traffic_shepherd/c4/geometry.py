"""SUMO lane id + lane position -> WM-S model frame (lane, s_down), slot pool.

This is the conversion every WM-S pack was built with: the frozen WM-N1 featurizer
(``output/wd01/wm_n1/features/geometry.py`` LANE_TABLE / HERE_OFFSETS and the
persistent slot pool of ``features/featurizer.py::parse_run``).  The tables are
copied here (tests assert byte-equality with the frozen module) so the live
environment needs no ``output/`` import.

Model frame: lane 0 = main left, 1 = main right, 2 = ramp/acceleration lane; ``s`` is
the merge-relative coordinate (merge zone 0..75.44 m, upstream negative, downstream
> 75.44).  Junction internal lanes (``:`` prefix) have no modelled lane: lane -1 and
``s`` carried from the vehicle's last modelled position (``junction`` flag set).
"""
from __future__ import annotations

import heapq
from typing import Iterable

#: lane_offsets_m of the frozen geometry (model s = OFFSETS[lane] + s_down).
HERE_OFFSETS = (2645.67, 2644.22, 378.66)
#: (edge, lane_index) -> (lane_phys, segment_start_m); frozen WM-30 table.
LANE_TABLE = {
    ("1191554636", 0): (1, 0.0),
    ("1191554636", 1): (0, 0.0),
    ("1487606179", 0): (2, 378.66),
    ("1487606179", 1): (1, 2644.22),
    ("1487606179", 2): (0, 2645.67),
    ("1487606178", 0): (1, 2719.66),
    ("1487606178", 1): (0, 2721.11),
    ("11553048#1", 0): (2, 0.0),
}
#: Pack length column (``features.constants.NOMINAL_LENGTH_M``).
NOMINAL_LENGTH_M = 5.0
MERGE_S0_M = 0.0
MERGE_S1_M = 75.44


class GeometryError(ValueError):
    """A SUMO lane outside the frozen lane table."""


def is_junction_lane(lane_id: str) -> bool:
    return str(lane_id).startswith(":")


def lane_and_s(lane_id: str, lane_position_m: float) -> tuple[int, float]:
    """``(lane_phys, s_down_m)`` for a non-junction SUMO lane (fails loudly otherwise)."""
    edge, sep, idx = str(lane_id).rpartition("_")
    if not sep or not idx.isdigit():
        raise GeometryError(f"unrecognised SUMO lane id {lane_id!r}")
    row = LANE_TABLE.get((edge, int(idx)))
    if row is None:
        raise GeometryError(f"lane id {lane_id!r} not in the frozen lane table")
    lane, start = row
    return lane, start + float(lane_position_m) - HERE_OFFSETS[lane]


class SlotPool:
    """Persistent slot pool of ``k`` slots with lowest-free-slot reuse.

    Identical to ``features.featurizer.parse_run``: per physics row, slots of vehicles
    that left are freed first (heap), then live vehicles in row order (sorted id) keep
    their slot or take the lowest free one; with no free slot a vehicle is dropped for
    that tick (and retried on the next).
    """

    def __init__(self, k: int):
        if int(k) < 1:
            raise ValueError("slot pool size must be positive")
        self.k = int(k)
        self._free = list(range(self.k))
        heapq.heapify(self._free)
        self.current: dict[str, int] = {}
        self.dropped_steps = 0

    def update(self, live_ids: Iterable[str]) -> tuple[dict[str, int], list[str]]:
        live = [str(v) for v in live_ids]
        live_set = set(live)
        for vid in [v for v in self.current if v not in live_set]:
            heapq.heappush(self._free, self.current.pop(vid))
        dropped = []
        for vid in live:
            if vid in self.current:
                continue
            if not self._free:
                dropped.append(vid)
                continue
            self.current[vid] = heapq.heappop(self._free)
        self.dropped_steps += len(dropped)
        return dict(self.current), dropped

    def slot_of(self, vid: str):
        return self.current.get(str(vid))
