"""wz3to2 geometry in the WM-S model frame (N1 map_descriptor / WM_PORT_API section 1).

SUMO: edge 'up' x in [0, 2696] m with 3 lanes (index 0 rightmost = the CLOSING lane, it ends at
X_B = 2696 m), internal junction ':n1_0' (8 m), edge 'wz' (2 lanes) to x = 4000 m.
WM frame: s = x - ANCHOR_X (closing-lane end at s = 75.44); WM lane 0 = left through lane
(SUMO up_2 / wz_1), WM lane 1 = right through lane (SUMO up_1 / wz_0), WM lane 2 = closing lane
(SUMO up_0).  All metrics of this suite are computed in the WM frame.
"""
from __future__ import annotations

import math
import re

X_B = 2696.0                     # SUMO x of the closing-lane end (nm2_core.X_B)
DROP_S = 75.44                   # WM s of the closing-lane end (FrameContract.merge_zone_s_m[1])
ANCHOR_X = X_B - DROP_S          # 2620.56
EDGE_OFFSET = {"up": 0.0, ":n1_0": 2696.0, "wz": 2704.0}
ROAD_END_X = 4000.0
ROAD_END_S = ROAD_END_X - ANCHOR_X          # 1379.44
ENTRY_S = -ANCHOR_X                         # -2620.56
SPEED_LIMIT = 31.29
DT = 0.5
CLOSING_LANE = 2
THROUGH_LANES = (0, 1)

# Zones (WM frame, metres) - fixed before any result (WM_QUAL_GATE.md section 3)
APPROACH_ZONE_S = (DROP_S - 300.0, DROP_S)          # breakdown speed zone, all lanes
QUEUE_ZONE_S = (DROP_S - 1000.0, DROP_S)            # queue count zone, all lanes
QUEUE_EXTENT_RANGE_S = (DROP_S - 2000.0, DROP_S)    # queue tail search range
QUEUE_SPEED = 5.0                                   # m/s: "queued" (nm2 closing_lane_slow rule)
DETECTOR_S = DROP_S + 100.0                         # discharge detector on 'wz' (x = 2796)
SECTIONS_S = {"upstream": (ENTRY_S, DROP_S - 300.0), "drop": (DROP_S - 300.0, DROP_S),
              "downstream": (DROP_S, ROAD_END_S)}

_UP = {0: 2, 1: 1, 2: 0}
_WZ = {0: 1, 1: 0}


def wm_lane(edge: str, sumo_lane: int) -> int:
    """WM lane of a SUMO (edge, lane index); junction lanes follow the downstream 'wz' lanes."""
    if edge == "up":
        return _UP[int(sumo_lane)]
    if edge == "wz" or edge.startswith(":n1"):
        return _WZ.get(int(sumo_lane), 0)
    raise ValueError(f"edge {edge!r} is not on the wz3to2 route")


def x_to_s(x: float) -> float:
    return float(x) - ANCHOR_X


def s_to_x(s: float) -> float:
    return float(s) + ANCHOR_X


_CELL = re.compile(r"^D(\d+)_p(\d+)$")


def cell_key(demand, penetration) -> str:
    return f"D{int(round(float(demand)))}_p{int(round(float(penetration) * 100)):02d}"


def parse_cell(cell: str) -> tuple[int, float]:
    m = _CELL.match(str(cell))
    if not m:
        raise ValueError(f"bad cell key {cell!r}")
    return int(m.group(1)), int(m.group(2)) / 100.0


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
