"""Section geometry for the WD-01 total-delay accounting (part 4).

PREDICT_CONTRACT_v1.md §4.1 (authoritative copy,
RD_WT_ROOT/wd01/output/wd01/PREDICT_CONTRACT_v1.md, sha256 74a1de8e...):
total delay = sum over vehicles present at t0 or arriving in
[t0, t0+horizon] of (time on section - free-flow time on the traversed
part of the section) + time waiting to enter at the mainline entrance and
at the ramp, every term truncated at t0+horizon.

The section is the WM-30 physical corridor: 3 physical lanes
(0 main_left, 1 main_right, 2 accel) made of SUMO lanes chained by
junction connections.  Each physical lane carries the continuous
coordinate s from its upstream end (hybrid_state_v1/topology.py,
read-only).  Free-flow speed is the SUMO lane speed limit of the segment
containing the vehicle (per-lane limits differ: 31.29 m/s mainline,
22.22 m/s ramp) -- this is the CONTRACT DEFAULT (free_flow_ref,
vType-agnostic).  The oracle variant divides by the limit times the
vType speedFactor (diagnostic only; ideal sensing does not reveal
speedFactor).

Zero SUMO: the network is read as (gzipped) XML only.
"""
from __future__ import annotations

import gzip
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import wdpaths  # noqa: E402

JUNCTION = -1          # lane id of junction records (zero physical length)
ENTRANCE_NONE = -1
ENTRANCE_MAINLINE = 0
ENTRANCE_RAMP = 1


@dataclass(frozen=True)
class Segment:
    """One SUMO lane as part of a physical lane chain."""
    s0: float           # physical-lane s at the segment's upstream end
    s1: float           # s0 + SUMO lane length
    speed_limit: float  # SUMO lane speed limit, m/s
    edge: str           # SUMO edge id (lane changes happen within an edge)


@dataclass(frozen=True)
class SectionGeometry:
    """Per-physical-lane segment chains with speed limits."""
    segments: dict[int, tuple[Segment, ...]]
    names: dict[int, str]
    default_speed: float   # speed limit for pairs with no physical lane

    def length(self, lane: int) -> float:
        return self.segments[lane][-1].s1

    def segment_at(self, lane: int, s: float) -> Segment:
        segs = self.segments[lane]
        s = min(max(float(s), segs[0].s0), segs[-1].s1)
        for seg in segs:
            if seg.s0 <= s < seg.s1 or (seg is segs[-1] and s == seg.s1):
                return seg
        raise AssertionError("unreachable")  # pragma: no cover

    def segments_near(self, lane: int, s: float) -> tuple[Segment, ...]:
        """Segments adjacent to s, boundaries included (a record exactly on
        a segment boundary may belong to either side, e.g. a lane change
        at the very end of the merge edge)."""
        segs = self.segments[lane]
        s = min(max(float(s), segs[0].s0), segs[-1].s1)
        out = []
        for i, seg in enumerate(segs):
            if seg.s0 <= s < seg.s1 or (seg is segs[-1] and s == seg.s1):
                out.append(seg)
                if s == seg.s1 and i + 1 < len(segs):
                    out.append(segs[i + 1])
                if s == seg.s0 and i > 0:
                    out.append(segs[i - 1])
                break
        return tuple(out)

    def change_segments(self, lane_a: int, s_a: float, lane_b: int,
                        s_b: float):
        """(seg_a, seg_b) for a lane-change step: the candidate pair whose
        edges match (a lane change in SUMO happens within one edge)."""
        for sa in self.segments_near(lane_a, s_a):
            for sb in self.segments_near(lane_b, s_b):
                if sa.edge == sb.edge:
                    return sa, sb
        return None

    def speed_at(self, lane: int, s: float) -> float:
        return self.segment_at(lane, s).speed_limit


def make_geometry(spec: dict[int, list[tuple]], names=None,
                  default_speed: float | None = None) -> SectionGeometry:
    """Geometry from a plain spec: {lane: [(s0, s1, speed_limit, edge)]}.

    Tests use round-number synthetic sections; wm30_geometry() builds the
    real one from the frozen network.
    """
    segs = {int(l): tuple(Segment(float(a), float(b), float(v), str(e))
                          for (a, b, v, e) in rows)
            for l, rows in spec.items()}
    if default_speed is None:
        default_speed = max(sg.speed_limit
                            for rows in segs.values() for sg in rows)
    for l, rows in segs.items():
        for a, b in zip(rows, rows[1:]):
            assert abs(b.s0 - a.s1) < 1e-9, f"gap in chain {l}"
        assert all(sg.s1 > sg.s0 and sg.speed_limit > 0 for sg in rows)
    return SectionGeometry(segments=segs,
                           names=dict(names or {}),
                           default_speed=float(default_speed))


# --------------------------------------------------------------- WM-30 ----

# physical lane -> SUMO lane chain (hybrid_state_v1/topology.py, frozen)
WM30_CHAINS = {
    0: ("main_left", ["1191554636_1", "1487606179_2", "1487606178_1"]),
    1: ("main_right", ["1191554636_0", "1487606179_1", "1487606178_0"]),
    2: ("accel", ["11553048#1_0", "1487606179_0"]),
}
WM30_NET = str(Path(wdpaths.RESEARCH_DENSE_ROOT) / "assets" / "research" /
               "highway_merge_shepherd_long" / "osm.net.xml.gz")


def _net_lane_table(net_path) -> dict[str, tuple[float, float]]:
    """{SUMO lane id: (length, speed limit)} read from the net XML only."""
    table = {}
    opener = gzip.open if str(net_path).endswith(".gz") else open
    with opener(net_path, "rb") as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == "edge" and el.get("function") != "internal":
                for lane in el.findall("lane"):
                    length, speed = lane.get("length"), lane.get("speed")
                    if length is None or speed is None:
                        continue         # e.g. crossings; not our corridors
                    table[lane.get("id")] = (float(length), float(speed))
                el.clear()               # lanes are read; free the edge
    return table


def wm30_geometry(net_path: str | None = None,
                  lane_table: dict[str, tuple[float, float]] | None = None
                  ) -> SectionGeometry:
    """Real WM-30 section geometry from the frozen network (read-only).

    Segment boundaries are taken from the topology's documented chains
    (the s0 continuity is checked); lengths and speed limits come from the
    same net XML.  Passing `lane_table` lets tests inject a parsed table.
    """
    if lane_table is None:
        lane_table = _net_lane_table(net_path or WM30_NET)
    spec, names = {}, {}
    for lane, (name, chain) in WM30_CHAINS.items():
        rows, s0 = [], 0.0
        for lane_id in chain:
            length, speed = lane_table[lane_id]
            rows.append((s0, s0 + length, speed, lane_id.rsplit("_", 1)[0]))
            s0 += length
        spec[lane] = rows
        names[lane] = name
    return make_geometry(spec, names=names)
