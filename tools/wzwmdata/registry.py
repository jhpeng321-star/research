"""wz3to2 world registry for the WM data phase ``wm_wz_data`` (WM-B, 2026-11-06).

Two registered blocks (map-namespaced streams, same verbatim generator as the probe / NEWMAP2 /
N1 ``map_scenarios``; only the seed/demand registry differs):

* WMDATA (WM train/val raw data): seeds 989100-989199, each seed in exactly ONE cell (no crossed
  cells: the same seed in two cells gives scaled arrivals, nested CAV sets and identical driver
  draws, i.e. correlated worlds). Cells = demand {2400, 2800, 3200, 3600} veh/h x CAV share
  {0.2, 0.4, 0.6}. Assignment is a fixed seeded permutation, registered before any SUMO.
* WMQUAL (WM-D qualification set (a), MAIN relay 2026-11-06): native tapes on seeds 989580-989599
  x {3200x0.6, 3200x0.4, 2800x0.6, 2800x0.4}, 0-1500 s, split tag WMQUAL. NEVER WM train/val.

Refused everywhere: 989000-989099 (PLANGEN2), 989200-989579 (probes, NEWMAP2, COOPKINDS, PLANGEN,
clean4 TRAIN pool), 989600+ (held-out), 964000-964199, 962xxx, 977xxx. WM-D reserved
qualification worlds (989200-989216, 989350-989353, 989400-989402, 989580-989599) never enter the
WMDATA block (checked).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

MAP = 'wz3to2'
PHASE = 'wm_wz_data'
WMDATA_RANGE = (989100, 989199)
WMQUAL_RANGE = (989580, 989599)
WMQUAL_CELLS = ((3200, 0.6), (3200, 0.4), (2800, 0.6), (2800, 0.4))
DEMANDS = (2400, 2800, 3200, 3600)
SHARES = (0.2, 0.4, 0.6)
#: cell order of the seed assignment: the first 100 mod 12 = 4 cells get one extra world
#: (cooperation-relevant high-demand / high-share cells first).
CELL_ORDER = ((3200, 0.6), (3200, 0.4), (2800, 0.6), (3600, 0.6), (3600, 0.4), (2800, 0.4),
              (2400, 0.6), (3200, 0.2), (3600, 0.2), (2400, 0.4), (2800, 0.2), (2400, 0.2))
ASSIGN_KEY = 'wzwmdata_cells_v1'
FORBIDDEN_RANGES = ((989000, 989099), (989200, 989579), (989600, 989999),
                    (964000, 964199), (962000, 962999), (977000, 977999))
WMD_RESERVED = ((989200, 989216), (989350, 989353), (989400, 989402), (989580, 989599))
WARMUP_S = 300
WMDATA_END_S = 1200            # native warmup 0-300 s, control 300-1200 s (N1 plant episode)
WMQUAL_END_S = 1500            # WM-D set (a): 0-1500 s native
SCHEMA = 'wzwmdata.registry.v1'


def cell_key(demand, penetration):
    return f'D{demand}_p{int(round(penetration * 100)):02d}'


def _in(seed, ranges):
    return any(lo <= seed <= hi for lo, hi in ranges)


def wmdata_assignment():
    """{seed: (demand, penetration)} for the 100 WMDATA seeds (fixed seeded permutation)."""
    if len(set(CELL_ORDER)) != len(DEMANDS) * len(SHARES):
        raise AssertionError('CELL_ORDER must list every cell once')
    seeds = list(range(WMDATA_RANGE[0], WMDATA_RANGE[1] + 1))
    random.Random(ASSIGN_KEY).shuffle(seeds)
    return {s: CELL_ORDER[i % len(CELL_ORDER)] for i, s in enumerate(seeds)}


def world(seed, demand, penetration, *, block):
    """Registered world dict; raises when (seed, demand, penetration) is not in ``block``."""
    if type(seed) is not int or type(demand) is not int:
        raise ValueError('seed and demand must be exact integers')
    if block == 'WMDATA':
        if not WMDATA_RANGE[0] <= seed <= WMDATA_RANGE[1] or _in(seed, WMD_RESERVED):
            raise ValueError(f'seed {seed} is not a WMDATA seed')
        if wmdata_assignment()[seed] != (demand, float(penetration)):
            raise ValueError(f'({seed}, {demand}, {penetration}) is not the registered WMDATA cell')
    elif block == 'WMQUAL':
        if not WMQUAL_RANGE[0] <= seed <= WMQUAL_RANGE[1]:
            raise ValueError(f'seed {seed} is not a WMQUAL seed')
        if (demand, float(penetration)) not in WMQUAL_CELLS:
            raise ValueError(f'({demand}, {penetration}) is not a WMQUAL cell')
    else:
        raise ValueError(f'unknown block {block!r}')
    if _in(seed, FORBIDDEN_RANGES) and block != 'WMQUAL':
        raise ValueError(f'seed {seed} is in a forbidden range')
    return dict(seed=seed, demand=demand, penetration=float(penetration), map=MAP,
                cell_key=cell_key(demand, penetration), block=block)


def wmdata_worlds():
    return [world(s, d, p, block='WMDATA') for s, (d, p) in sorted(wmdata_assignment().items())]


def wmqual_worlds():
    return [world(s, d, p, block='WMQUAL') for s in range(WMQUAL_RANGE[0], WMQUAL_RANGE[1] + 1)
            for d, p in WMQUAL_CELLS]


def check_case(seed, demand_vph):
    """``traffic_shepherd.c4.env.check_case`` replacement (registered seeds/demands only)."""
    if type(seed) is not int or type(demand_vph) is not int:
        raise ValueError('seed and demand must be exact integers')
    ok = (WMDATA_RANGE[0] <= seed <= WMDATA_RANGE[1] or WMQUAL_RANGE[0] <= seed <= WMQUAL_RANGE[1])
    if not ok or _in(seed, ((964000, 964199), (962000, 962999), (977000, 977999))):
        raise ValueError(f'seed {seed} is not a registered {PHASE} seed; refused')
    if demand_vph not in DEMANDS:
        raise ValueError(f'demand {demand_vph} is not a registered {PHASE} demand')


def _block_of(seed, demand, penetration):
    if WMQUAL_RANGE[0] <= seed <= WMQUAL_RANGE[1]:
        return world(seed, demand, penetration, block='WMQUAL')
    return world(seed, demand, penetration, block='WMDATA')


def build_scenario(desc, root, seed, demand_vph, *, penetration, role='exposed', config=None,
                   cav_vtype=None, population='textbook'):
    """``CavSumoEnv.reset`` scenario builder: the verbatim N1/probe generator, this registry.

    Writes only the SUMO input files (``requests.rou.xml`` holds the per-HDV vTypes as SUMO
    input; nothing here parses them) plus ``requests.json`` (ids, departures, routes, CAV flags;
    no driver parameters). No driver-truth file is written."""
    from tools.coopmwm import map_scenarios as ms
    from tools.coopmwm.map_descriptor import spec_digest
    from traffic_shepherd.c4.population import TEXTBOOK_CAV_VTYPE, hdv_vtype_id
    from traffic_shepherd.scenarios import TripRequest
    if population != 'textbook':
        raise ValueError(f'{MAP} uses the textbook population only')
    if cav_vtype not in (None, TEXTBOOK_CAV_VTYPE):
        raise ValueError(f'textbook population: CAVs use {TEXTBOOK_CAV_VTYPE!r}')
    if desc.name != MAP:
        raise ValueError(f'descriptor {desc.name!r} is not {MAP}')
    check_case(seed, demand_vph)
    w = _block_of(seed, demand_vph, float(penetration))
    spec_path = Path(desc.spec_path)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if spec_digest(spec) != desc.spec_sha256:
        raise ValueError('MAP_SPEC changed after the descriptor was loaded')
    root = Path(root)
    for name in ('requests.rou.xml', 'scenario.sumocfg', 'requests.json'):
        if (root / name).exists():
            raise ValueError('refusing to change an existing scenario')
    scene, reqs = ms.write_scenario(root, spec_path.parent, spec, MAP, seed, {'main': demand_vph},
                                    float(penetration))
    (root / 'requests.json').write_text(json.dumps(
        [dict(id=r['id'], depart=r['depart'], route=r['route'], cav=bool(r['cav'])) for r in reqs]),
        encoding='utf-8')
    trips = [TripRequest(r['id'], r['depart'], r['route'],
                         TEXTBOOK_CAV_VTYPE if r['cav'] else hdv_vtype_id(r['id']), bool(r['cav']))
             for r in reqs]
    text = (root / 'requests.rou.xml').read_bytes()
    manifest = {
        'schema': ms.SCHEMA, 'map': MAP, 'seed': seed, 'mainline_vph': demand_vph,
        'penetration': penetration, 'cell_key': w['cell_key'], 'block': w['block'],
        'registry': SCHEMA, 'phase': PHASE,
        'demand_end_seconds': ms.DEMAND_END, 'population': len(trips),
        'cav_population': sum(t.is_cav for t in trips),
        'routes_sha256': hashlib.sha256(text).hexdigest(),
        'network_sha256': desc.network_sha256, 'map_spec_sha256': desc.spec_sha256,
        'requests_sha256': hashlib.sha256(json.dumps([asdict(t) for t in trips],
                                                     sort_keys=True).encode()).hexdigest(),
        'generator': 'tools.coopmwm.map_scenarios.write_scenario (== newmap probe / NEWMAP2), verbatim',
        'assignment': 'fixed_explicit_driver_type_and_cav_identity_no_runtime_type_change',
        'cav_vtype': TEXTBOOK_CAV_VTYPE, 'driver_population': 'textbook', 'driver_truth_file': None,
        'role': role,
    }
    return scene, trips, manifest


def scenario_builder(desc):
    def build(root, seed, demand_vph, *, penetration, role='exposed', config=None, cav_vtype=None,
              population='textbook'):
        return build_scenario(desc, root, seed, demand_vph, penetration=penetration, role=role,
                              config=config, cav_vtype=cav_vtype, population=population)
    build.map_name = MAP
    return build


def record():
    assign = wmdata_assignment()
    per_cell = {}
    for s, (d, p) in sorted(assign.items()):
        per_cell.setdefault(cell_key(d, p), []).append(s)
    body = dict(schema=SCHEMA, map=MAP, phase=PHASE, wmdata_range=list(WMDATA_RANGE),
                wmqual_range=list(WMQUAL_RANGE), wmqual_cells=[list(c) for c in WMQUAL_CELLS],
                demands=list(DEMANDS), shares=list(SHARES), cell_order=[list(c) for c in CELL_ORDER],
                assign_key=ASSIGN_KEY, wmdata_cells=per_cell,
                forbidden_ranges=[list(r) for r in FORBIDDEN_RANGES],
                wmd_reserved=[list(r) for r in WMD_RESERVED], warmup_s=WARMUP_S,
                wmdata_end_s=WMDATA_END_S, wmqual_end_s=WMQUAL_END_S)
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body
