"""wz3to2 scenario builder + world registry for the clean3/clean4 runtime (N1, 2026-11-06).

Bit-for-bit reuse of the newmap probe / NEWMAP2 generator (MAIN 15:0x, N4 ingestion blocker):
``requests``/``write_scenario`` below are the verbatim scenario code of
``runs/active/C4-V7FT-20261017/newmap_probe/probe_core.py`` (== ``newmap2/code/nm2_core.py`` for
wz3to2): Poisson departures per stream from ``random.Random('newmap_requests_v1:{map}:{seed}:{stream}')``,
CAV identity ``random.Random('newmap_cav_v1:{map}:{seed}:{id}').random() < penetration``, one textbook
HDV vType per vehicle from ``traffic_shepherd.c4.population.sample_driver('{map}:{seed}', id)``
(the probe's population.py is a verbatim copy, sha 6af5ee23), CAVs on ``cav_typical``, departLane
from MAP_SPEC (``free``), departPos ``base``, departSpeed ``max``, peak profile (0.8 D 0-300 s,
D 300-1500 s, 0.6 D 1500-1800 s). The same (seed, demand, penetration) therefore writes a
byte-identical ``requests.rou.xml`` (test_map_scenarios_cpu compares against the vendored
generators). No driver-truth file is written (the probe wrote none).

Episodes (MAIN decision): 1200 s like the merge-map plant: native warmup 0-300 s, control
300-1200 s (leverprobe Backend: warmup_s=300, horizon_s=901, stop=1200). The demand file is left
identical to the probe (requests to 1800 s); departures after 1200 s never enter an episode.

World registry (map-namespaced; never aliases the merge-map registry, whose world dicts have no
'map' key): cells = NEWMAP2's grid, demand (2400, 2800, 3200) veh/h x CAV share (0.2, 0.4, 0.6),
CROSSED with seeds (every seed exists in every cell, as in NEWMAP2: an admitted NEWMAP2 moment
(seed, cell, t0) maps one-to-one onto world(seed, demand, penetration)).
TRAIN seeds 989200-989599; HELD-OUT evaluation seeds 989600-989699. 964000-964199, 962xxx and
977xxx are refused explicitly.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path

MAP = 'wz3to2'
DEMANDS = (2400, 2800, 3200)
SHARES = (0.2, 0.4, 0.6)
TRAIN_RANGE = (989200, 989599)
HELDOUT_RANGE = (989600, 989699)
FORBIDDEN_RANGES = ((964000, 964199), (962000, 962999), (977000, 977999))
EPISODE_S = 1200
WARMUP_S = 300
HORIZON_S = 901            # leverprobe Backend: warmup 300 + 901 -> SUMO end 1201, plant stop 1200
CONTROL_WINDOW_S = (300, 1200)
DEMAND_END = 1800.0        # probe/NEWMAP2 demand file (bit-identity); episodes stop at 1200 s
TRAIN_ROLES = frozenset({'train', 'check', 'dev', 'smoke', 'mechanics', 'calibration', 'discovery',
                         'practice', 'gate', 'sanity', 'mining'})
HELDOUT_ROLES = frozenset({'final_test', 'final', 'heldout', 'replication_test', 'random_guide_test'})
SCHEMA = 'coopmwm.map_scenario.v1'


# ------------------------------------------------------------------- cells and worlds
def cell_key(demand, penetration):
    """NEWMAP2 ``nm2_plan.cell_key``: 'D2800_p40'."""
    return f'D{demand}_p{int(round(penetration * 100)):02d}'


def cells():
    return [dict(cell=i*len(SHARES)+j, demand=d, penetration=p, cell_key=cell_key(d, p))
            for i, d in enumerate(DEMANDS) for j, p in enumerate(SHARES)]


def _forbidden(seed):
    return any(lo <= seed <= hi for lo, hi in FORBIDDEN_RANGES)


def seed_scope(seed):
    if type(seed) is not int or _forbidden(seed):
        return None
    if TRAIN_RANGE[0] <= seed <= TRAIN_RANGE[1]:
        return 'train'
    if HELDOUT_RANGE[0] <= seed <= HELDOUT_RANGE[1]:
        return 'heldout'
    return None


def world(seed, demand, penetration):
    """The registered world dict for (seed, demand, penetration); raises when unregistered."""
    scope = seed_scope(seed)
    if scope is None:
        raise ValueError(f'seed {seed!r} outside the registered {MAP} TRAIN/HELD-OUT ranges')
    cell = next((c for c in cells() if c['demand'] == demand and c['penetration'] == penetration), None)
    if cell is None or type(demand) is not int:
        raise ValueError(f'({demand!r}, {penetration!r}) is not a registered {MAP} cell')
    start = TRAIN_RANGE[0] if scope == 'train' else HELDOUT_RANGE[0]
    return dict(seed=seed, cell=cell['cell'], repeat=seed-start, penetration=penetration, demand=demand,
                map=MAP, cell_key=cell['cell_key'])


def worlds(role='train', *, seeds=None, cell_ids=None):
    """Registered worlds of a role; ``seeds``/``cell_ids`` select subsets (order: seed, cell)."""
    lo, hi = TRAIN_RANGE if role in TRAIN_ROLES else HELDOUT_RANGE if role in HELDOUT_ROLES else (None, None)
    if lo is None:
        raise ValueError(f'unknown {MAP} world role {role!r}')
    pool = range(lo, hi+1) if seeds is None else seeds
    out = []
    for s in pool:
        if not lo <= s <= hi:
            raise ValueError(f'seed {s} is not in the {role} range {lo}-{hi}')
        for c in cells():
            if cell_ids is None or c['cell'] in cell_ids:
                out.append(world(s, c['demand'], c['penetration']))
    return out


def guard(w, role):
    """Exact registered map world for this role (TRAIN roles <-> TRAIN seeds, final <-> HELD-OUT)."""
    if not isinstance(w, dict) or w.get('map') != MAP:
        raise ValueError(f'not a {MAP} world')
    expected = world(w.get('seed'), w.get('demand'), w.get('penetration'))
    if w != expected:
        raise ValueError(f'world differs from the registered {MAP} world {expected}')
    scope = seed_scope(w['seed'])
    if role in TRAIN_ROLES and scope == 'train':
        return expected
    if role in HELDOUT_ROLES and scope == 'heldout':
        return expected
    raise ValueError(f'{MAP} seed {w["seed"]} ({scope}) is not allowed for role {role!r}')


def check_case(seed, demand_vph):
    """env.check_case replacement on the map (exact ints, registered seeds and demands)."""
    if type(seed) is not int or type(demand_vph) is not int:
        raise ValueError('seed and demand must be exact integers')
    if seed_scope(seed) is None:
        raise ValueError(f'seed {seed} is not a registered {MAP} seed; refused')
    if demand_vph not in DEMANDS:
        raise ValueError(f'demand {demand_vph} is not a registered {MAP} cell demand')


def world_for_moment(moment):
    """NEWMAP2 moment/row (seed + demand + penetration, or seed + cell 'D2800_p40') -> world."""
    seed = int(moment['seed'])
    if 'demand' in moment and 'penetration' in moment:
        demand = moment['demand']['main'] if isinstance(moment['demand'], dict) else moment['demand']
        return world(seed, int(demand), float(moment['penetration']))
    cell = next(c for c in cells() if c['cell_key'] == moment['cell'])
    return world(seed, cell['demand'], cell['penetration'])


# ------------------------------------------------- verbatim probe / NEWMAP2 scenario code
def rate_profile(kind: str, d: float):
    if kind == "flat":
        return [(0.0, DEMAND_END, d)]
    # "peak": warm 0-300 s at 0.8 D, peak 300-1500 s at D, 1500-1800 s at 0.6 D
    return [(0.0, 300.0, 0.8 * d), (300.0, 1500.0, d), (1500.0, DEMAND_END, 0.6 * d)]


def requests(map_name: str, spec: dict, seed: int, demand: dict, penetration: float):
    """Poisson departures per stream (project scheme, map-namespaced streams)."""
    out = []
    for stream, sdef in spec["streams"].items():
        rng = random.Random(f"newmap_requests_v1:{map_name}:{seed}:{stream}")
        ordinal = 0
        for begin, end, rate in rate_profile(sdef["profile"], float(demand[stream])):
            t = begin
            while True:
                t += rng.expovariate(rate / 3600.0)
                if t >= end:
                    break
                ident = f"{stream}.{ordinal}"
                route = sdef["route"]
                if isinstance(route, list):
                    u = random.Random(f"newmap_route_v1:{map_name}:{seed}:{ident}").random()
                    route = route[1] if u < float(sdef.get("exit_share", 0.0)) else route[0]
                cav = random.Random(f"newmap_cav_v1:{map_name}:{seed}:{ident}").random() < penetration
                out.append(dict(id=ident, depart=round(t, 6), route=route, cav=cav,
                                lane=sdef.get("depart_lane", "best"),
                                pos=sdef.get("depart_pos", "base")))
                ordinal += 1
    return sorted(out, key=lambda r: (r["depart"], r["id"]))


def write_scenario(root: Path, map_dir: Path, spec: dict, map_name: str, seed: int, demand: dict,
                   penetration: float):
    import xml.etree.ElementTree as ET
    from traffic_shepherd.c4 import population as pop
    reqs = requests(map_name, spec, seed, demand, penetration)
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", pop.vtype_attrs(pop.TEXTBOOK_CAV_VTYPE,
                                                    dict(pop.TEXTBOOK_CAV_PARAMS)))
    for r in reqs:
        if not r["cav"]:
            # fresh per-vehicle draw for a NEW (map-namespaced) seed; written only into the route
            # file SUMO reads -- no truth table is produced or read.
            ET.SubElement(routes, "vType", pop.vtype_attrs(
                pop.hdv_vtype_id(r["id"]), pop.sample_driver(f"{map_name}:{seed}", r["id"])))
    for rid, edges in spec["routes"]:
        ET.SubElement(routes, "route", {"id": rid, "edges": edges})
    for r in reqs:
        ET.SubElement(routes, "vehicle", {
            "id": r["id"], "type": pop.TEXTBOOK_CAV_VTYPE if r["cav"] else pop.hdv_vtype_id(r["id"]),
            "route": r["route"], "depart": str(r["depart"]), "departLane": r["lane"],
            "departPos": r["pos"], "departSpeed": "max"})
    ET.indent(routes)
    root.mkdir(parents=True, exist_ok=True)
    rou = root / "requests.rou.xml"
    rou.write_text(ET.tostring(routes, encoding="unicode"), encoding="utf-8")
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value=str((map_dir / spec["net_file"]).resolve()))
    ET.SubElement(inp, "route-files", value=rou.name)
    proc = ET.SubElement(cfg, "processing")
    ET.SubElement(proc, "time-to-teleport", value="-1")
    ET.SubElement(proc, "collision.action", value="warn")
    ET.indent(cfg)
    scene = root / "scenario.sumocfg"
    scene.write_text(ET.tostring(cfg, encoding="unicode"), encoding="utf-8")
    return scene, reqs


# ------------------------------------------------------------- env-compatible builder
def build_scenario(desc, root, seed, demand_vph, *, penetration, role='exposed', config=None,
                   cav_vtype=None, population='textbook'):
    """``traffic_shepherd.scenarios.build_scenario`` signature for ``CavSumoEnv.reset`` on the map.

    Returns ``(scene_path, [TripRequest], manifest)``. Refuses a non-textbook population, an
    unregistered (seed, demand, penetration) world, an existing differing scenario and any
    descriptor/spec mismatch."""
    from traffic_shepherd.c4.population import TEXTBOOK_CAV_VTYPE, hdv_vtype_id
    from traffic_shepherd.scenarios import TripRequest
    if population != 'textbook':
        raise ValueError(f'{MAP} uses the textbook population only')
    if cav_vtype not in (None, TEXTBOOK_CAV_VTYPE):
        raise ValueError(f'textbook population: CAVs use {TEXTBOOK_CAV_VTYPE!r}')
    if desc.name != MAP:
        raise ValueError(f'descriptor {desc.name!r} is not {MAP}')
    check_case(seed, demand_vph)
    world(seed, demand_vph, float(penetration))          # registered cell (penetration included)
    spec_path = Path(desc.spec_path)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    from .map_descriptor import spec_digest
    if spec_digest(spec) != desc.spec_sha256:
        raise ValueError('MAP_SPEC changed after the descriptor was loaded')
    root = Path(root)
    for name in ('requests.rou.xml', 'scenario.sumocfg'):
        if (root/name).exists():
            raise ValueError('refusing to change an existing scenario')
    scene, reqs = write_scenario(root, spec_path.parent, spec, MAP, seed, {'main': demand_vph}, float(penetration))
    trips = [TripRequest(r['id'], r['depart'], r['route'],
                         TEXTBOOK_CAV_VTYPE if r['cav'] else hdv_vtype_id(r['id']), bool(r['cav'])) for r in reqs]
    text = (root/'requests.rou.xml').read_bytes()
    manifest = {
        'schema': SCHEMA, 'map': MAP, 'seed': seed, 'mainline_vph': demand_vph, 'penetration': penetration,
        'cell_key': cell_key(demand_vph, float(penetration)),
        'episode_seconds': EPISODE_S, 'warmup_s': WARMUP_S, 'control_window_s': list(CONTROL_WINDOW_S),
        'demand_end_seconds': DEMAND_END, 'population': len(trips), 'cav_population': sum(t.is_cav for t in trips),
        'routes_sha256': hashlib.sha256(text).hexdigest(),
        'network_sha256': desc.network_sha256, 'map_spec_sha256': desc.spec_sha256,
        'requests_sha256': hashlib.sha256(json.dumps([asdict(t) for t in trips], sort_keys=True).encode()).hexdigest(),
        'generator': 'newmap_probe/probe_core.py write_scenario (== NEWMAP2 nm2_core.py), verbatim',
        'assignment': 'fixed_explicit_driver_type_and_cav_identity_no_runtime_type_change',
        'cav_vtype': TEXTBOOK_CAV_VTYPE, 'driver_population': 'textbook', 'driver_truth_file': None,
        'role': role,
    }
    return scene, trips, manifest


def scenario_builder(desc):
    """Bound replacement for ``traffic_shepherd.c4.env.build_scenario`` on the map."""
    def build(root, seed, demand_vph, *, penetration, role='exposed', config=None, cav_vtype=None,
              population='textbook'):
        return build_scenario(desc, root, seed, demand_vph, penetration=penetration, role=role,
                              config=config, cav_vtype=cav_vtype, population=population)
    build.map_name = MAP
    return build


def registry_record():
    return dict(schema=SCHEMA, map=MAP, demands=list(DEMANDS), shares=list(SHARES), cells=cells(),
                train_range=list(TRAIN_RANGE), heldout_range=list(HELDOUT_RANGE),
                forbidden_ranges=[list(r) for r in FORBIDDEN_RANGES], crossed=True,
                episode_s=EPISODE_S, warmup_s=WARMUP_S, horizon_s=HORIZON_S,
                control_window_s=list(CONTROL_WINDOW_S), demand_end_s=DEMAND_END,
                train_roles=sorted(TRAIN_ROLES), heldout_roles=sorted(HELDOUT_ROLES))
