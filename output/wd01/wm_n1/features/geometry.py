"""Frozen network geometry: SUMO lane id -> lane_phys and s_down.

Physical-lane convention (WM-30, ``partA_v2g/adapter.py``):

    lane_phys 0 = main_left  (edges 1191554636@1, 1487606179@2, 1487606178@1)
    lane_phys 1 = main_right (edges 1191554636@0, 1487606179@1, 1487606178@0)
    lane_phys 2 = accel      (edges 11553048#1@0, 1487606179@0)

``s`` is the physical-lane coordinate from the lane's upstream end (the same
``s`` the frozen partA_data.pdl mapping produces): ``s = segment_start +
lane_position_m``.  ``s_down`` is the merge-relative downstream coordinate of
the Part-A view: ``s_down = s - OFFSETS[lane_phys]`` (model s = OFFSETS +
s_down).  Negative upstream of the merge boundary, ~0 at the merge start.

The LANE_TABLE below is read off the same frozen network the rest of WD-01
uses (edge/lane ids and the segment starts in
``partB_v0data/params_nominal.json`` geometry block); it is covered by a
tested table so no per-row topology lookup is needed at featurization time.
Junction internal lanes (SUMO ``:``-prefixed) carry lane_phys -1/junction=1.
"""
from __future__ import annotations

HERE_OFFSETS = (2645.67, 2644.22, 378.66)     # lane_offsets_m
LANE_NAMES = ("main_left", "main_right", "accel")

# (edge, lane_index) -> (lane_phys, segment_start_m)
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

# entrance edges (design section 1 / brief): mainline entry edge and ramp
# entry edge taken from the network.
ENTRANCE_MAINLINE_EDGE = "1191554636"
ENTRANCE_RAMP_EDGE = "11553048#1"
ENTRANCE_MAINLINE, ENTRANCE_RAMP = 0, 1


class GeometryError(ValueError):
    """A lane id outside the frozen WM-30 lane table."""


def split_lane_id(lane_id: str):
    """('<edge>_<laneIndex>') -> (edge, lane_index)."""
    s = str(lane_id)
    edge, sep, idx = s.rpartition("_")
    if not sep or not idx.isdigit():
        raise GeometryError(f"unrecognised SUMO lane id {lane_id!r}")
    return edge, int(idx)


def is_junction_lane(lane_id: str) -> bool:
    return str(lane_id).startswith(":")


def lane_phys_of(lane_id: str):
    """lane_phys for a SUMO lane id, or None for a junction internal lane."""
    if is_junction_lane(lane_id):
        return None
    edge, idx = split_lane_id(lane_id)
    row = LANE_TABLE.get((edge, idx))
    if row is None:
        raise GeometryError(
            f"lane id {lane_id!r} (edge {edge!r}, index {idx}) is not in the "
            f"frozen WM-30 lane table; refusing to guess a lane_phys")
    return row[0]


def s_down_of(lane_id: str, lane_position_m: float):
    """(lane_phys, s_down_m) for a non-junction lane id."""
    edge, idx = split_lane_id(lane_id)
    row = LANE_TABLE.get((edge, idx))
    if row is None:
        raise GeometryError(f"lane id {lane_id!r} not in the frozen lane table")
    lane_phys, seg_start = row
    s = seg_start + float(lane_position_m)
    return lane_phys, s - HERE_OFFSETS[lane_phys]


def entrance_of_lane_phys(lane_phys: int) -> int:
    """lane_phys 2 (accel/ramp chain) -> ramp, otherwise mainline."""
    return ENTRANCE_RAMP if int(lane_phys) == 2 else ENTRANCE_MAINLINE
