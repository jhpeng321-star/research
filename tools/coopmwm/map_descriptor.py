"""Map descriptors: one SUMO map's projection into the fixed WM-S model frame (N1, 2026-11-06).

The clean3 runtime (frozen e3final env + frozen speed-only MAPPO + the residual/V2V policy)
observes the road in ONE model frame (``FrameContract``): three WM lanes (0 = main left,
1 = main right, 2 = the MERGE lane, i.e. the lane whose vehicles must leave it), a merge zone
``[0, 75.44]`` m whose end is where the merge lane ends, the frozen RSU field cells and the
observable local zone ``[-100, 75.44]``. The frozen MAPPO u0 and the residual u0 were trained
in that frame, so every map that should run them AS-IS must project into it. A descriptor
therefore carries:

* the projection: ``lane_table`` ((edge, sumo lane) -> (WM lane, segment start)),
  ``here_offsets`` and ``lane_and_s`` (identical arithmetic to ``traffic_shepherd.c4.geometry``);
* the RSU descriptor (``RsuDescriptor``): where the RSU aggregates and from which source;
* the SUMO facts: network, boundary edges, lane-end distances (TableSensor), slot pool size,
  bottleneck location, demand streams.

``ORIGINAL`` is the preserved merge map with exactly the frozen constants (golden tests:
``test_map_golden_cpu.py``). ``load(MAP_SPEC.json)`` builds a lane-drop descriptor from the
probe's MAP_SPEC + its net file: the dropped (rightmost) lane becomes WM lane 2 ("ramp = lane 2"
becomes "closing lane = lane 2") and ``s`` is anchored so the lane end is at ``merge_zone_s_m[1]``.
On wz3to2 that maps x = 2696 m (end of up_0) to s = 75.44 m.

Process selection: ``current()`` is ``ORIGINAL`` unless ``COOPMWM_MAP_SPEC`` names a MAP_SPEC.json
(or ``bind`` was called). One process = one map: rebinding to a different map raises.
No SUMO, torch or TraCI import here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from xml.etree import ElementTree as ET

ENV = 'COOPMWM_MAP_SPEC'
ORIGINAL_NAME = 'merge_original'
SCHEMA = 'coopmwm.map_descriptor.v1'


@dataclass(frozen=True)
class FrameContract:
    """The single WM-S model frame (values = wms.control.contract / c4.geometry / c4.state)."""

    n_lanes: int = 3
    main_lanes: tuple = (0, 1)
    merge_lane: int = 2                         # ramp (merge map) / closing lane (lane drop)
    merge_zone_s_m: tuple = (0.0, 75.44)        # merge lane ends at merge_zone_s_m[1]
    local_zone_s_m: tuple = (-100., 75.44)      # observable local zone (coopfree/variety/proposer)
    rsu_mainline_s_m: tuple = (-800.0, 225.44)  # frozen MAPPO RSU field cells, lanes 0/1
    rsu_ramp_s_m: tuple = (-378.66, 75.44)      # frozen MAPPO RSU field cells, lane 2
    rsu_cell_m: float = 50.0
    speed_limit_mps: float = 31.29
    nominal_length_m: float = 5.0
    sensor_range_m: float = 100.0


FRAME = FrameContract()


@dataclass(frozen=True)
class RsuDescriptor:
    """Where and from what the RSU aggregates.

    ``source``: ``'rsu_detector'`` = the original map's frozen behaviour (CAV self telemetry +
    the RSU's own detections in its zones, ``traffic_shepherd.c4.state.rsu_detections``), kept
    byte-identical; ``'cav_reports'`` = only CAV-reported observations: timestamped CAV
    self-states plus neighbours a CAV detects locally (|ds| <= ``report_range_m``,
    |dlane| <= ``report_lane_reach``), with receipt age, missingness and coverage
    (``map_observations.report_summary``); no true penetration or global ground truth.
    ``block_zone_s_m``: per WM lane the zone of the 12-feature RSU block (coopnative.encode,
    layout unchanged so MAPPO u0 stays loadable). ``field_*``: the frozen MAPPO RSU-field cells
    (must equal the frame; the cell count is a frozen network input size).
    """

    source: str = 'rsu_detector'
    block_zone_s_m: tuple = ((-100., 75.44), (-100., 75.44), (-100., 75.44))
    field_mainline_s_m: tuple = (-800.0, 225.44)
    field_ramp_s_m: tuple = (-378.66, 75.44)
    field_cell_m: float = 50.0
    report_range_m: float = 100.0
    report_lane_reach: int = 1
    receipt_age_s: float = 0.0
    max_receipt_age_s: float = 1.0

    def __post_init__(self):
        if self.source not in ('rsu_detector', 'cav_reports'):
            raise ValueError(f'unknown RSU source {self.source!r}')
        if len(self.block_zone_s_m) != FRAME.n_lanes or any(
                len(z) != 2 or not (math.isfinite(z[0]) and math.isfinite(z[1]) and z[0] < z[1])
                for z in self.block_zone_s_m):
            raise ValueError('block_zone_s_m must be one ordered finite (s0, s1) per WM lane')
        if not 0. <= self.receipt_age_s <= self.max_receipt_age_s:
            raise ValueError('receipt age outside the accepted window')


@dataclass(frozen=True)
class MapDescriptor:
    name: str
    kind: str                                   # 'onramp_merge' | 'lane_drop'
    lane_table: dict
    here_offsets: tuple
    k_slots: int
    edges: dict                                 # mainline_entry / mainline_exit / ramp_entry (None)
    frame: FrameContract = FRAME
    rsu: RsuDescriptor = field(default_factory=RsuDescriptor)
    lane_end_distance: dict | None = None       # TableSensor lane map; None = frozen observation_map
    network_path: str | None = None             # None = the frozen adapter config network
    network_sha256: str | None = None           # semantic digest (lanes + connections)
    spec_sha256: str | None = None              # semantic digest of MAP_SPEC.json
    spec_path: str | None = None
    bottleneck: dict | None = None
    streams: dict | None = None
    routes: tuple = ()

    @property
    def is_original(self) -> bool:
        return self.name == ORIGINAL_NAME

    # -- projection (same arithmetic as traffic_shepherd.c4.geometry.lane_and_s) ----------
    def lane_and_s(self, lane_id: str, lane_position_m: float):
        edge, sep, idx = str(lane_id).rpartition('_')
        if not sep or not idx.isdigit():
            raise ValueError(f'unrecognised SUMO lane id {lane_id!r}')
        row = self.lane_table.get((edge, int(idx)))
        if row is None:
            raise ValueError(f'lane id {lane_id!r} not in the {self.name} lane table')
        lane, start = row
        return lane, start + float(lane_position_m) - self.here_offsets[lane]

    def sumo_index(self, edge: str, wm_lane: int):
        return next((i for (e, i), (lane, _) in self.lane_table.items() if e == edge and lane == wm_lane), None)

    def boundary_edges(self) -> dict:
        """``sumo_io.boundary_edges`` shape; a map without a ramp uses a never-matching id."""
        return {'mainline_entry': str(self.edges['mainline_entry']),
                'mainline_exit': str(self.edges['mainline_exit']),
                'ramp_entry': str(self.edges.get('ramp_entry') or '__no_ramp_entry__')}

    def record(self) -> dict:
        """JSON-able identity (READY files, receipts, job bindings)."""
        return dict(schema=SCHEMA, name=self.name, kind=self.kind,
                    lane_table={f'{e}|{i}': list(v) for (e, i), v in sorted(self.lane_table.items())},
                    here_offsets=list(self.here_offsets), k_slots=self.k_slots, edges=dict(self.edges),
                    frame=_dc(self.frame), rsu=_dc(self.rsu), network_sha256=self.network_sha256,
                    spec_sha256=self.spec_sha256, bottleneck=self.bottleneck, streams=self.streams,
                    routes=[list(r) for r in self.routes])

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()


def _dc(obj):
    out = {}
    for k in obj.__dataclass_fields__:
        v = getattr(obj, k)
        out[k] = [list(x) if isinstance(x, tuple) else x for x in v] if isinstance(v, tuple) else v
    return out


ORIGINAL = MapDescriptor(
    name=ORIGINAL_NAME, kind='onramp_merge',
    # traffic_shepherd.c4.geometry (frozen WM-30 table) -- golden-tested for equality.
    lane_table={
        ("1191554636", 0): (1, 0.0),
        ("1191554636", 1): (0, 0.0),
        ("1487606179", 0): (2, 378.66),
        ("1487606179", 1): (1, 2644.22),
        ("1487606179", 2): (0, 2645.67),
        ("1487606178", 0): (1, 2719.66),
        ("1487606178", 1): (0, 2721.11),
        ("11553048#1", 0): (2, 0.0),
    },
    here_offsets=(2645.67, 2644.22, 378.66),
    k_slots=192,
    edges={'mainline_entry': '1191554636', 'mainline_exit': '1487606178', 'ramp_entry': '11553048#1'},
)


# -------------------------------------------------------------------------- registry
#: Registered non-original maps. Semantic digests pin the map so a changed copy fails closed
#: (CRLF/LF and netconvert path comments do not matter; geometry, lanes and connections do).
#: k_slots: the lane-drop queue can exceed the merge map's 192 present vehicles (3 lanes x
#: 2.7 km upstream); a full slot pool would drop vehicles, which the plant treats as an anomaly.
REGISTERED = {
    'wz3to2': dict(
        spec_sha256='ac98e73be605a460ff0c14288bc43a7ce2187d0a26d0b49cc42003f9871f2a8f',
        network_sha256='946594c52ac96c1e4bb44401b972d7095fdf5188f934af3be73f8b36fb615a9c',
        k_slots=512,
        rsu=dict(source='cav_reports', block_upstream_s_m=-100.0, block_downstream_m=100.0),
    ),
}


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def spec_digest(spec: dict) -> str:
    return hashlib.sha256(_canonical(spec).encode()).hexdigest()


def parse_network(path) -> dict:
    """Plain lanes (id, edge, index, speed, length, shape) + connections of a SUMO net file."""
    root = ET.parse(path).getroot()
    lanes, internal = {}, {}
    for edge in root.iter('edge'):
        target = internal if edge.get('function') == 'internal' else lanes
        for lane in edge.findall('lane'):
            shape = [tuple(float(c) for c in p.split(',')) for p in lane.get('shape').split()]
            target[lane.get('id')] = dict(edge=edge.get('id'), index=int(lane.get('index')),
                                          speed=float(lane.get('speed')), length=float(lane.get('length')),
                                          shape=shape)
    connections = sorted((c.get('from'), c.get('to'), int(c.get('fromLane')), int(c.get('toLane')), c.get('via') or '')
                         for c in root.iter('connection') if not c.get('from', '').startswith(':'))
    return dict(lanes=lanes, internal=internal, connections=connections)


def network_digest(net: dict) -> str:
    body = dict(lanes={k: [v['edge'], v['index'], v['speed'], v['length'], v['shape']] for k, v in net['lanes'].items()},
                internal={k: [v['edge'], v['index'], v['speed'], v['length'], v['shape']] for k, v in net['internal'].items()},
                connections=net['connections'])
    return hashlib.sha256(_canonical(body).encode()).hexdigest()


def _straight_x(lane: dict):
    xs, ys = [p[0] for p in lane['shape']], [p[1] for p in lane['shape']]
    if len(set(ys)) != 1 or xs != sorted(xs) or xs[0] == xs[-1]:
        raise ValueError(f'lane-drop projection needs straight west->east lanes: {lane}')
    if abs((xs[-1]-xs[0])-lane['length']) > 1e-6:
        raise ValueError('lane length differs from its straight shape')
    return xs[0], xs[-1]


def load(path, *, frame: FrameContract = FRAME) -> MapDescriptor:
    """Lane-drop descriptor from a probe MAP_SPEC.json (or its directory) + its net file."""
    path = Path(path)
    spec_path = path/'MAP_SPEC.json' if path.is_dir() else path
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    name = spec.get('name')
    if name not in REGISTERED:
        raise ValueError(f'map {name!r} is not registered for the clean3 runtime')
    reg = REGISTERED[name]
    if reg.get('spec_sha256') and spec_digest(spec) != reg['spec_sha256']:
        raise ValueError(f'{name}: MAP_SPEC differs from the registered map')
    neck = spec.get('bottleneck') or {}
    if 'dropped_lane' not in neck:
        raise ValueError(f'{name}: only lane-drop maps project into the merge frame')
    net_path = spec_path.parent/spec['net_file']
    net = parse_network(net_path)
    digest = network_digest(net)
    if reg.get('network_sha256') and digest != reg['network_sha256']:
        raise ValueError(f'{name}: network differs from the registered map')
    up = neck['edge']
    up_lanes = sorted((v for v in net['lanes'].values() if v['edge'] == up), key=lambda v: v['index'])
    if len(up_lanes) != frame.n_lanes or int(neck['dropped_lane']) != 0:
        raise ValueError('the frame needs a 3-lane approach whose RIGHTMOST lane (index 0) drops')
    if sorted(int(x) for x in neck['through_lanes']) != [1, 2]:
        raise ValueError('through lanes must be SUMO 1 and 2')
    # SUMO index 0 is rightmost: leftmost -> WM 0, middle -> WM 1, dropped -> WM merge lane.
    wm_of = {2: frame.main_lanes[0], 1: frame.main_lanes[1], 0: frame.merge_lane}
    table, ends = {}, {}
    for lane in up_lanes:
        x0, x1 = _straight_x(lane)
        table[(up, lane['index'])] = (wm_of[lane['index']], x0)
        ends[f"{up}_{lane['index']}"] = x1
    if any(c[0] == up and c[2] == 0 for c in net['connections']):
        raise ValueError('the dropped lane must have no downstream connection')
    for frm, to, fl, tl, _via in net['connections']:
        if frm != up:
            raise ValueError('only the single approach -> downstream junction is supported')
        lane = net['lanes'][f'{to}_{tl}']
        x0, x1 = _straight_x(lane)
        table[(to, tl)] = (wm_of[fl], x0)
        ends[f'{to}_{tl}'] = x1
    drop_x = ends[f'{up}_0']
    anchor = drop_x-frame.merge_zone_s_m[1]
    for v in net['internal'].values():
        ends[f"{v['edge']}_{v['index']}"] = _straight_x(v)[1]
    # TableSensor: distance = value + lane_length - position = drop_x - x (signed, + before the drop).
    lane_end = {lane_id: drop_x-x1 for lane_id, x1 in sorted(ends.items())}
    route_edges = spec['routes'][0][1].split()
    rsu_reg = reg['rsu']
    drop_s = frame.merge_zone_s_m[1]
    through = (float(rsu_reg['block_upstream_s_m']), drop_s+float(rsu_reg['block_downstream_m']))
    block = tuple(((float(rsu_reg['block_upstream_s_m']), drop_s) if lane == frame.merge_lane else through)
                  for lane in range(frame.n_lanes))
    rsu = RsuDescriptor(source=rsu_reg['source'], block_zone_s_m=block,
                        field_mainline_s_m=frame.rsu_mainline_s_m, field_ramp_s_m=frame.rsu_ramp_s_m,
                        field_cell_m=frame.rsu_cell_m, report_range_m=frame.sensor_range_m)
    return MapDescriptor(
        name=name, kind='lane_drop', lane_table=table, here_offsets=(anchor,)*frame.n_lanes,
        k_slots=int(reg['k_slots']), frame=frame, rsu=rsu,
        edges=dict(mainline_entry=route_edges[0], mainline_exit=route_edges[-1], ramp_entry=None),
        lane_end_distance=lane_end, network_path=str(net_path.resolve()), network_sha256=digest,
        spec_sha256=spec_digest(spec), spec_path=str(spec_path.resolve()),
        bottleneck=dict(edge=up, sumo_dropped_lane=0, drop_x_m=drop_x, drop_s_m=drop_s,
                        closing_lane=frame.merge_lane, through_lanes=list(frame.main_lanes),
                        adjacent_lane=frame.main_lanes[1], anchor_x_m=anchor, node_x_m=float(neck['x_m'])),
        streams=spec['streams'], routes=tuple(tuple(r) for r in spec['routes']))


# -------------------------------------------------------------------------- selection
_BOUND: MapDescriptor | None = None


def current() -> MapDescriptor:
    """The process map: bound descriptor, else COOPMWM_MAP_SPEC, else ORIGINAL."""
    global _BOUND
    if _BOUND is None:
        path = os.environ.get(ENV)
        _BOUND = load(path) if path else ORIGINAL
    return _BOUND


def bind(desc: MapDescriptor) -> MapDescriptor:
    global _BOUND
    if _BOUND is not None and _BOUND.digest != desc.digest:
        raise ValueError(f'process already bound to map {_BOUND.name}; one map per process')
    _BOUND = desc
    return desc


def for_world(world: dict) -> MapDescriptor:
    """Descriptor a world requires; a world without 'map' is the original map.

    The world key and the process selection must agree (fail closed both ways)."""
    name = world.get('map') if isinstance(world, dict) else None
    desc = current()
    if (name or ORIGINAL_NAME) != desc.name:
        raise ValueError(f"world map {name or ORIGINAL_NAME!r} != process map {desc.name!r} "
                         f"(set {ENV} to the map's MAP_SPEC.json for map worlds)")
    return desc


def _reset_for_tests():
    global _BOUND
    _BOUND = None
