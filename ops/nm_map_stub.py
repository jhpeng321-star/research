"""STUB of the N1 map-descriptor API (branch c4/nm-abstract, still in progress on 2026-11-06).

The wz3to2 final evaluation (ops/nm_finaleval.py) codes against this interface. When N1 lands,
point the launcher at the real module (``--map-api <module>`` or env ``NM_MAP_API``); the
launcher then compares the real API with the registered panel (seed ranges, net SHA, every panel
world) and refuses physics while this stub is bound (``STUB = True``).

Interface expected from N1 (names may be adapted in ``nm_finaleval.bind_map_api`` only):
  STUB                      False in the real module
  seed_ranges(name)         {'train': (lo, hi), 'eval': (lo, hi)} inclusive, registered ranges
  descriptor(name)          dict with at least name, net_sha256, bottleneck, lanes_upstream,
                            speed_limit_mps, stream_namespace
  world(name, seed, cell, repeat)  the world dict the scenario builder consumes
  check_world(world, role)  raises ValueError unless ``world`` is a registered ``role`` world

Values here come only from the map files (MAP_SPEC.json, DEMAND_CHOSEN.json, wz3to2.net.xml)
and MAIN's newmap_spec.md seed registration. Stdlib only; never reads a truth file.
"""
import hashlib
import json
import os
from pathlib import Path

STUB = True
MAPS = ('wz3to2',)
#: newmap_spec.md (MAIN 2026-11-06): TRAIN 989200-989599, HELD-OUT eval 989600-989699.
SEED_RANGES = {'wz3to2': {'train': (989200, 989599), 'eval': (989600, 989699)}}
#: never usable on any map (newmap_spec.md)
EXCLUDED = ((964000, 964199), (962000, 962999), (977000, 977999))
DEFAULT_ASSETS = 'E:/research-dense/assets/research/newmap_candidates'
UPSTREAM_LANES = {'wz3to2': 3}


def _name(name):
    if name not in MAPS:
        raise ValueError(f'unknown map {name!r}')
    return name


def asset_dir(name):
    return Path(os.environ.get('NM_MAP_ASSETS', DEFAULT_ASSETS))/_name(name)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seed_ranges(name):
    return dict(SEED_RANGES[_name(name)])


def descriptor(name):
    root = asset_dir(name)
    spec = _json(root/'MAP_SPEC.json')
    net = root/spec['net_file']
    return dict(name=spec['name'], net_file=spec['net_file'],
                net_sha256=hashlib.sha256(net.read_bytes()).hexdigest(),
                bottleneck=dict(spec['bottleneck']), lanes_upstream=UPSTREAM_LANES[name],
                speed_limit_mps=spec['speed_limit_mps'], streams=dict(spec['streams']),
                stream_namespace=name, stub=True)


def world(name, seed, cell, repeat):
    """``cell`` = {'demand_vph': {'main': D}, 'penetration': p, 'profile': ...} (single stream)."""
    _name(name)
    demand = cell['demand_vph']
    if set(demand) != {'main'}:
        raise ValueError('wz3to2 has one stream: main')
    return dict(map=name, seed=int(seed), cell=0, repeat=int(repeat), penetration=float(cell['penetration']),
                demand=int(demand['main']), profile=str(cell['profile']))


def check_world(world, role):
    ranges = seed_ranges(world.get('map'))
    if role not in ranges:
        raise ValueError(f'unknown role {role!r}')
    seed = world.get('seed')
    lo, hi = ranges[role]
    if type(seed) is not int or not lo <= seed <= hi:
        raise ValueError(f'seed {seed!r} outside the registered {role} range {lo}-{hi}')
    if any(a <= seed <= b for a, b in EXCLUDED):
        raise ValueError('excluded seed')
    return True
