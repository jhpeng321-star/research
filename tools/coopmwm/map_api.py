"""Small wz3to2 adapter API for consumers outside the training runtime (N6 final evaluation, N3/N4 tools).

    from tools.coopmwm import map_api
    map_api.seed_ranges()                    # {'train': [989200, 989599], 'heldout': [989600, 989699], 'forbidden': [...]}
    map_api.descriptor()                     # MapDescriptor of this process (COOPMWM_MAP_SPEC) or descriptor(path)
    map_api.world(989650, 'D3200_p60')       # registered world dict (map-namespaced, never a merge-map world)
    map_api.check_world(world, 'final_test') # raises unless the exact registered world for that role
    map_api.final_worlds('D3200_p60')        # HELD-OUT panel of a cell (seeds 989600-989699)

Pure Python, no SUMO/TraCI/torch. The canonical clean4 cell is ``CANONICAL_CELL_KEY`` (D3200_p60).
"""
from __future__ import annotations

from . import map_descriptor as md
from . import map_scenarios as ms

MAP = ms.MAP
CANONICAL_CELL_KEY = ms.CANONICAL_CELL_KEY
CELLS = tuple(c['cell_key'] for c in ms.cells())


def seed_ranges():
    return dict(train=list(ms.TRAIN_RANGE), heldout=list(ms.HELDOUT_RANGE),
                forbidden=[list(r) for r in ms.FORBIDDEN_RANGES])


def descriptor(spec_path=None):
    """The wz3to2 descriptor: from ``spec_path`` (MAP_SPEC.json or its directory) or the process map."""
    desc = md.load(spec_path) if spec_path is not None else md.current()
    if desc.name != MAP:
        raise ValueError(f'process map is {desc.name!r}; set {md.ENV} to the wz3to2 MAP_SPEC.json')
    return desc


def world(seed, cell_key=CANONICAL_CELL_KEY):
    return ms.world_from_key(int(seed), cell_key)


def check_world(world, role='final_test'):
    """The exact registered world for the role (TRAIN roles <-> TRAIN seeds, final roles <-> HELD-OUT)."""
    return ms.guard(world, role)


def final_worlds(cell_key=CANONICAL_CELL_KEY, seeds=None):
    cell = next(c for c in ms.cells() if c['cell_key'] == cell_key)
    return ms.worlds('final_test', seeds=seeds, cell_ids=[cell['cell']])


def train_worlds(cell_key=CANONICAL_CELL_KEY, seeds=None):
    cell = next(c for c in ms.cells() if c['cell_key'] == cell_key)
    return ms.worlds('train', seeds=seeds, cell_ids=[cell['cell']])
