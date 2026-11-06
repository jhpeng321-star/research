"""Road geometry of the WM-S model frame: one map = one :class:`RoadGeometry` (WM-A, 2026-11-06).

Every WM-S module used to hard-wire the on-ramp merge map (lane starts -2645.67/-2644.22/-378.66,
through-lane end 553.33, accel-lane end 75.44, "lane 2 = ramp", the RSU zones).  The model frame
itself is map independent (N1 ``tools/coopmwm/map_descriptor.FrameContract``): WM lanes 0/1 are the
through lanes, WM lane 2 is the CLOSING lane (merge map: on-ramp + acceleration lane; lane drop:
the dropped SUMO lane) and ``s`` is anchored so the closing lane ends at ``merge_zone_s_m[1]``
(75.44 m on every map).  What differs between maps is collected here:

* ``lane_extents_s_m`` / ``road_end_m`` / ``entry_s_m`` / ``entry_lane_speed_mps``: where lanes
  start (vehicles are inserted) and end (through-lane exit);
* ``closing_change_zone_s_m``: where a closing-lane car can physically leave its lane into
  ``target_lane`` (merge map: only along the acceleration lane = the merge zone; lane drop: the
  whole approach), and ``into_closing_allowed`` (a lane-drop closing lane can also be entered);
* the RSU zones (frozen frame values; the RSU *source* is N1's ``RsuDescriptor``).

:data:`MERGE_ORIGINAL` carries exactly the frozen constants: with it every module computes what it
computed before the port (``tests/wz/test_merge_golden.py`` proves bit-identity end to end).

Selection (one process = one map, mirroring N1): :func:`current_road` is the bound road, else the
map named by ``WMS_MAP_SPEC`` (or N1's ``COOPMWM_MAP_SPEC``) -> :func:`load_road`, else
:data:`MERGE_ORIGINAL`.  :func:`bind_road` refuses rebinding to another map; :func:`road_scope`
temporarily selects a road (tests, offline tools).  Lane-drop roads come from N1's
``MapDescriptor`` via :func:`road_from_descriptor` (duck-typed: no fork of N1's code); until
c4/nm-abstract is merged, ``load_road`` falls back to :func:`stub_descriptor` (same field names,
same numbers, ``record()['descriptor'] == 'stub'``).  No torch / SUMO import here.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

SCHEMA = "wms.road/1"
ENV_VARS = ("WMS_MAP_SPEC", "COOPMWM_MAP_SPEC")
ORIGINAL_NAME = "merge_original"
KINDS = ("onramp_merge", "lane_drop")


@dataclass(frozen=True)
class RoadGeometry:
    """One map's geometry in the WM-S model frame (``s`` metres, WM lanes 0/1/2)."""

    name: str = ORIGINAL_NAME
    kind: str = "onramp_merge"
    n_lanes: int = 3
    through_lanes: tuple = (0, 1)
    closing_lane: int = 2                      # was "ramp lane" / physics.merge_lane
    target_lane: int = 1                       # physics.merge_target_lane
    merge_zone_s_m: tuple = (0.0, 75.44)       # mandatory-merge zone; [1] = closing-lane end
    lane_extents_s_m: tuple = ((-2645.67, 553.33), (-2644.22, 553.33), (-378.66, 75.44))
    entry_s_m: tuple = (-2645.67, -2644.22, -378.66)
    entry_lane_speed_mps: tuple = (31.29, 31.29, 22.22)
    #: end of each lane's entrance edge (SUMO departLane occupancy key of the entry process)
    entry_edge_end_s_m: tuple = (0.0, 0.0, 0.0)
    road_end_m: float = 553.33                 # through-lane exit (model frame)
    closing_change_zone_s_m: tuple = (0.0, 75.44)
    into_closing_allowed: bool = False
    speed_limit_mps: float = 31.29
    rsu_mainline_s_m: tuple = (-800.0, 225.44)
    rsu_closing_s_m: tuple = (-378.66, 75.44)
    rsu_cell_m: float = 50.0
    #: entrance -> WM lanes (counter channel 0 = mainline/through, 1 = ramp/closing lane)
    entrances: tuple = ((0, 1), (2,))
    k_slots: int = 192
    descriptor: Optional[str] = None           # 'n1' | 'stub' | None (original constants)
    descriptor_sha256: Optional[str] = None    # N1 MapDescriptor.digest

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"road kind must be one of {KINDS}: {self.kind!r}")
        if self.n_lanes != 3 or tuple(self.through_lanes) != (0, 1) or self.closing_lane != 2:
            raise ValueError("the WM-S frame has through lanes 0/1 and closing lane 2")
        z0, z1 = self.merge_zone_s_m
        c0, c1 = self.closing_change_zone_s_m
        if not (z0 < z1 and c0 <= z0 and c1 == z1):
            raise ValueError("closing_change_zone_s_m must contain the merge zone and end with it")
        if self.lane_extents_s_m[self.closing_lane][1] != z1:
            raise ValueError("the closing lane must end at merge_zone_s_m[1]")

    # ------------------------------------------------------------------ identity
    @property
    def is_original(self) -> bool:
        return self.name == ORIGINAL_NAME

    @property
    def closing_lane_end_m(self) -> float:
        return float(self.merge_zone_s_m[1])

    @property
    def closing_change_restricted(self) -> bool:
        """True when a closing-lane car can change only inside the merge zone (on-ramp map)."""
        return tuple(self.closing_change_zone_s_m) == tuple(self.merge_zone_s_m)

    def record(self) -> dict:
        out = {"schema": SCHEMA}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = _plain(value)
        return out

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()


def _plain(value):
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


MERGE_ORIGINAL = RoadGeometry()


# ---------------------------------------------------------------------- descriptor adapter
def road_from_descriptor(desc) -> RoadGeometry:
    """:class:`RoadGeometry` of an N1 ``MapDescriptor`` (duck-typed; the original map ->
    :data:`MERGE_ORIGINAL`).  Lane-drop maps: through lanes run from the approach start to the
    downstream end, the closing lane from the approach start to the drop (``merge_zone_s_m[1]``)."""
    if getattr(desc, "is_original", False) or getattr(desc, "name", None) == ORIGINAL_NAME:
        return MERGE_ORIGINAL
    if getattr(desc, "kind", None) != "lane_drop":
        raise ValueError(f"unsupported map kind {getattr(desc, 'kind', None)!r}")
    frame = desc.frame
    drop_s = float(frame.merge_zone_s_m[1])
    closing, main = int(frame.merge_lane), tuple(int(x) for x in frame.main_lanes)
    lanes = {wm: [] for wm in range(int(frame.n_lanes))}
    for (edge, idx), (wm, start) in desc.lane_table.items():
        lane_id = f"{edge}_{idx}"
        s0 = float(start) - float(desc.here_offsets[wm])
        s1 = drop_s - float(desc.lane_end_distance[lane_id])
        lanes[int(wm)].append((s0, s1, edge))
    extents = tuple((round(min(r[0] for r in rows), 6), round(max(r[1] for r in rows), 6))
                    for _wm, rows in sorted(lanes.items()))
    entry_edge = desc.edges["mainline_entry"]
    entry = tuple(round(min(r[0] for r in lanes[wm] if r[2] == entry_edge), 6)
                  for wm in range(int(frame.n_lanes)))
    entry_end = tuple(round(max(r[1] for r in lanes[wm] if r[2] == entry_edge), 6)
                      for wm in range(int(frame.n_lanes)))
    speeds = _lane_speeds(desc, entry_edge) or (float(frame.speed_limit_mps),) * int(frame.n_lanes)
    road_end = extents[main[0]][1]
    if extents[main[1]][1] != road_end or extents[closing][1] != drop_s:
        raise ValueError("lane-drop road: through lanes must share their end; the closing lane "
                         "must end at the drop")
    rsu = desc.rsu
    return RoadGeometry(
        name=str(desc.name), kind="lane_drop", merge_zone_s_m=tuple(float(x) for x in frame.merge_zone_s_m),
        lane_extents_s_m=extents, entry_s_m=entry, entry_lane_speed_mps=tuple(speeds),
        entry_edge_end_s_m=entry_end,
        road_end_m=float(road_end), closing_change_zone_s_m=(float(extents[closing][0]), drop_s),
        into_closing_allowed=True, speed_limit_mps=float(frame.speed_limit_mps),
        rsu_mainline_s_m=tuple(float(x) for x in rsu.field_mainline_s_m),
        rsu_closing_s_m=tuple(float(x) for x in rsu.field_ramp_s_m), rsu_cell_m=float(rsu.field_cell_m),
        entrances=(main, (closing,)), k_slots=int(desc.k_slots),
        descriptor="stub" if getattr(desc, "is_stub", False) else "n1",
        descriptor_sha256=getattr(desc, "digest", None))


def _lane_speeds(desc, edge) -> Optional[tuple]:
    path = getattr(desc, "network_path", None)
    if not path or not Path(path).is_file():
        return None
    from xml.etree import ElementTree as ET

    speeds = {}
    for e in ET.parse(path).getroot().iter("edge"):
        if e.get("id") == edge:
            for lane in e.findall("lane"):
                speeds[int(lane.get("index"))] = float(lane.get("speed"))
    out = []
    for wm in range(int(desc.frame.n_lanes)):
        idx = desc.sumo_index(edge, wm) if hasattr(desc, "sumo_index") else None
        if idx is None or idx not in speeds:
            return None
        out.append(speeds[idx])
    return tuple(out)


#: wz3to2 numbers of N1's ``map_descriptor.load`` (MAP_SPEC sha256 ac98e73b..., net 946594c5...):
#: drop at x = 2696 m (end of up_0) -> s = 75.44, anchor 2620.56 m.  Used only until c4/nm-abstract
#: is merged; ``tests/wz`` checks that the real descriptor yields the identical road.
_WZ3TO2_STUB = dict(
    name="wz3to2", spec_sha256="ac98e73be605a460ff0c14288bc43a7ce2187d0a26d0b49cc42003f9871f2a8f",
    lane_table={("up", 2): (0, 0.0), ("up", 1): (1, 0.0), ("up", 0): (2, 0.0),
                ("wz", 1): (0, 2704.0), ("wz", 0): (1, 2704.0)},
    drop_x=2696.0, wz_end_x=4000.0, k_slots=512)


class _StubFrame(SimpleNamespace):
    pass


def stub_descriptor(name: str = "wz3to2"):
    """N1-shaped descriptor for a registered lane-drop map (pre-N1-merge stand-in)."""
    if name != "wz3to2":
        raise ValueError(f"no stub descriptor for map {name!r}")
    s = _WZ3TO2_STUB
    anchor = s["drop_x"] - 75.44
    ends = {"up_0": s["drop_x"], "up_1": s["drop_x"], "up_2": s["drop_x"],
            "wz_0": s["wz_end_x"], "wz_1": s["wz_end_x"]}
    frame = _StubFrame(n_lanes=3, main_lanes=(0, 1), merge_lane=2, merge_zone_s_m=(0.0, 75.44),
                       speed_limit_mps=31.29, rsu_mainline_s_m=(-800.0, 225.44),
                       rsu_ramp_s_m=(-378.66, 75.44), rsu_cell_m=50.0)
    rsu = SimpleNamespace(source="cav_reports", field_mainline_s_m=(-800.0, 225.44),
                          field_ramp_s_m=(-378.66, 75.44), field_cell_m=50.0)
    desc = SimpleNamespace(
        name=s["name"], kind="lane_drop", is_original=False, is_stub=True, frame=frame, rsu=rsu,
        lane_table=dict(s["lane_table"]), here_offsets=(anchor, anchor, anchor),
        lane_end_distance={k: s["drop_x"] - v for k, v in ends.items()}, k_slots=s["k_slots"],
        edges={"mainline_entry": "up", "mainline_exit": "wz", "ramp_entry": None},
        network_path=None, spec_sha256=s["spec_sha256"], digest=None)
    desc.sumo_index = lambda edge, wm: next(
        (i for (e, i), (lane, _) in desc.lane_table.items() if e == edge and lane == wm), None)
    return desc


def load_road(path) -> RoadGeometry:
    """Road of a MAP_SPEC.json (or its directory) via N1's ``map_descriptor.load``; the stub
    descriptor only when c4/nm-abstract is not importable yet."""
    path = Path(path)
    spec_path = path / "MAP_SPEC.json" if path.is_dir() else path
    try:
        from tools.coopmwm import map_descriptor as md
    except ImportError:
        md = None
    if md is not None and hasattr(md, "load"):
        return road_from_descriptor(md.load(spec_path))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return road_from_descriptor(stub_descriptor(spec.get("name")))


def road_by_name(name: Optional[str]) -> RoadGeometry:
    """Road for a recorded map name (manifests, checkpoints): the original or the bound road."""
    if name in (None, ORIGINAL_NAME):
        return MERGE_ORIGINAL
    road = current_road()
    if road.name != name:
        raise ValueError(f"map {name!r} is not the process road {road.name!r}; set "
                         f"{ENV_VARS[0]} to its MAP_SPEC.json")
    return road


# ---------------------------------------------------------------------- process selection
_BOUND: Optional[RoadGeometry] = None
_SCOPED: list = []


def current_road() -> RoadGeometry:
    """The process road: a :func:`road_scope` override, the bound road, the env map, the original."""
    global _BOUND
    if _SCOPED:
        return _SCOPED[-1]
    if _BOUND is None:
        path = next((os.environ[v] for v in ENV_VARS if os.environ.get(v)), None)
        _BOUND = load_road(path) if path else MERGE_ORIGINAL
    return _BOUND


def bind_road(road: RoadGeometry) -> RoadGeometry:
    """Bind the process road (idempotent); refuses a different map once bound."""
    global _BOUND
    if _BOUND is not None and _BOUND.digest != road.digest:
        raise ValueError(f"process already bound to road {_BOUND.name}; one map per process")
    _BOUND = road
    return road


@contextlib.contextmanager
def road_scope(road: RoadGeometry) -> Iterator[RoadGeometry]:
    """Temporarily select ``road`` (tests / offline tools); nests; restores on exit."""
    _SCOPED.append(road)
    try:
        yield road
    finally:
        _SCOPED.pop()


def _reset_for_tests() -> None:
    global _BOUND
    _BOUND = None
    _SCOPED.clear()


def with_overrides(road: RoadGeometry, **changes) -> RoadGeometry:
    """A modified copy (validation re-run)."""
    return replace(road, **changes)


def map_default(value, merge_value, road_value):
    """A geometry setting whose configured value is the merge-map default follows the road.

    Config dataclasses keep their merge-map defaults (serialised configs stay byte-identical);
    on a non-original road an UNCHANGED merge default (``value == merge_value``) is replaced by
    the road's value, an explicitly configured value is kept.
    """
    road = current_road()
    if road.is_original or value != merge_value:
        return value
    return road_value
