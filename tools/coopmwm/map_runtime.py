"""Bind a registered non-original map onto the frozen clean3 runtime of THIS process (N1).

Selection: ``COOPMWM_MAP_SPEC=<.../wz3to2/MAP_SPEC.json>`` in every process of the run (chain,
owners, spawned workers; like ``COOPMWM_G5_RUNTIME=1``) AND ``world['map'] == 'wz3to2'`` in every
job. Both must agree (``map_descriptor.for_world``); a merge-map world (no 'map' key) in a map
process, or a map world in a merge process, is refused.

``install(job)`` runs after the frozen-runtime binding (``accel_plant.bind_runtime``: leverprobe
setup -> seed scope -> guard -> G5 overlay -> this; the native-reference plant calls it from its
setup). For the ORIGINAL map it returns None without importing or touching anything, so the
frozen runtime is byte-identical. For a lane-drop map it rebinds, process-wide and idempotently:

* ``traffic_shepherd.c4.geometry`` LANE_TABLE / HERE_OFFSETS (every module that imported the
  frozen objects is rebound) -> FrameBuilder s/lane projection and G5 lane commands;
* ``traffic_shepherd.c4.env``: build_scenario (map_scenarios, bit-identical to the probe),
  observation_map (descriptor lane-end distances), boundary_edges, check_case (map seed
  registry) and CavSumoEnv (slot pool k_slots = descriptor value when the caller asks for the
  default 192);
* RSU ('cav_reports'): ``tools.leverprobe.backend.observations`` and
  ``tools.coopnative.plant.observations`` -> CAV self states + locally detected neighbours;
  ``wms.control.rsu_field.build_rsu_field`` -> frozen MAPPO field over CAV-detected vehicles only.

Before patching it verifies that the frozen frame constants (geometry, state, wms contract,
env speed limit) equal the descriptor's frame: frozen MAPPO u0 and the residual u0 run AS-IS
only in that frame (fail closed otherwise). Refuses rebinding a process to another map.
"""
from __future__ import annotations

import sys

from . import map_descriptor as md

_INSTALLED = None
_RECORD = {}


def _world(job):
    return job.get('world', job) if isinstance(job, dict) else job


def guard_world(world, role):
    """Registered map world for a role; the process map must be the world's map."""
    desc = md.for_world(world)
    if desc.name == 'wz3to2':
        from . import map_scenarios
        return map_scenarios.guard(world, role)
    raise ValueError(f'no world registry for map {desc.name!r}')


def check_frame(desc, *, geometry, state, contract, env):
    f = desc.frame
    checks = {
        'geometry.merge': ((geometry.MERGE_S0_M, geometry.MERGE_S1_M), f.merge_zone_s_m),
        'state.rsu': ((state.RSU_MAINLINE_S_M, state.RSU_RAMP_S_M, state.RSU_CELL_M),
                      (f.rsu_mainline_s_m, f.rsu_ramp_s_m, f.rsu_cell_m)),
        'contract.merge': ((contract.MERGE_S0_M, contract.MERGE_S1_M), f.merge_zone_s_m),
        'contract.rsu': ((contract.RSU_MAINLINE_S_M, contract.RSU_RAMP_S_M, contract.RSU_CELL_M),
                         (f.rsu_mainline_s_m, f.rsu_ramp_s_m, f.rsu_cell_m)),
        'contract.lanes': ((contract.N_LANES, contract.SPEED_LIMIT_MPS), (f.n_lanes, f.speed_limit_mps)),
        'rsu.field': ((desc.rsu.field_mainline_s_m, desc.rsu.field_ramp_s_m, desc.rsu.field_cell_m),
                      (f.rsu_mainline_s_m, f.rsu_ramp_s_m, f.rsu_cell_m)),
        'env.speed_limit': (env.SPEED_LIMIT_MPS, f.speed_limit_mps),
        'geometry.length': (geometry.NOMINAL_LENGTH_M, f.nominal_length_m),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f'frozen runtime frame differs from the {desc.name} descriptor frame: {bad}')


def _rebind(old, new, attribute, prefixes=('traffic_shepherd.', 'wms.', 'tools.')):
    hits = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(prefixes):
            continue
        if getattr(module, attribute, None) is old:
            setattr(module, attribute, new)
            hits.append(name)
    return sorted(hits)


def _map_env_class(env, desc):
    base = env.CavSumoEnv
    if getattr(base, 'MAP_NAME', None) is not None:
        return base

    class CavSumoEnv(base):
        MAP_NAME = desc.name

        def __init__(self, *args, k_slots=env.DEFAULT_K, **kwargs):
            if int(k_slots) not in (int(env.DEFAULT_K), int(desc.k_slots)):
                raise ValueError(f'{desc.name}: k_slots {k_slots} is neither the default nor the map pool')
            super().__init__(*args, k_slots=desc.k_slots, **kwargs)

    CavSumoEnv.__module__ = base.__module__
    CavSumoEnv.__qualname__ = base.__qualname__
    return CavSumoEnv


def install(job, *, leverprobe=True):
    """Bind the job's map in this process; None (no-op) for the original map."""
    global _INSTALLED
    desc = md.for_world(_world(job))
    if desc.is_original:
        return None
    if _INSTALLED is not None:
        if _INSTALLED.digest != desc.digest:
            raise ValueError(f'process runtime already bound to map {_INSTALLED.name}')
        return _INSTALLED
    md.bind(desc)
    import importlib
    env = importlib.import_module('traffic_shepherd.c4.env')
    geometry = importlib.import_module('traffic_shepherd.c4.geometry')
    state = importlib.import_module('traffic_shepherd.c4.state')
    contract = importlib.import_module('wms.control.contract')
    rsu_field = importlib.import_module('wms.control.rsu_field')
    check_frame(desc, geometry=geometry, state=state, contract=contract, env=env)
    from . import map_scenarios
    from .map_observations import cav_report_rows, cav_report_rows_with_length, masked_field_builder
    record = dict(map=desc.name, descriptor_sha256=desc.digest, env_module=getattr(env, '__file__', None),
                  g5_overlay=getattr(env, 'G5_RUNTIME_OVERLAY_SHA256', None))
    table, offsets = dict(desc.lane_table), tuple(desc.here_offsets)
    record['lane_table_rebound'] = _rebind(geometry.LANE_TABLE, table, 'LANE_TABLE')
    record['here_offsets_rebound'] = _rebind(geometry.HERE_OFFSETS, offsets, 'HERE_OFFSETS')
    env.build_scenario = map_scenarios.scenario_builder(desc)
    env.observation_map = lambda api, config=None: dict(desc.lane_end_distance)
    env.boundary_edges = lambda config=None: desc.boundary_edges()
    env.check_case = map_scenarios.check_case
    old_cls = env.CavSumoEnv
    new_cls = _map_env_class(env, desc)
    record['env_class_rebound'] = _rebind(old_cls, new_cls, 'CavSumoEnv')
    if desc.rsu.source == 'cav_reports':
        original = rsu_field.build_rsu_field
        record['rsu_field_rebound'] = _rebind(original, masked_field_builder(original, desc), 'build_rsu_field')
        rows = lambda e: cav_report_rows(e, desc)  # noqa: E731
        if leverprobe:
            lp = importlib.import_module('tools.leverprobe.backend')
            lp.observations = rows
        cn = importlib.import_module('tools.coopnative.plant')
        cn.observations = lambda e: cav_report_rows_with_length(e, desc)
        record['rows'] = 'cav_reports'
    env.MAP_RUNTIME = desc.name
    _RECORD.clear()
    _RECORD.update(record)
    _INSTALLED = desc
    return desc


def patch_globals(namespace, job):
    """Native-reference plant: its FunctionType copied coopnative.plant globals before setup ran."""
    desc = md.for_world(_world(job))
    if desc.is_original or desc.rsu.source != 'cav_reports':
        return
    from .map_observations import cav_report_rows_with_length
    namespace['observations'] = lambda e: cav_report_rows_with_length(e, desc)


def installed_record():
    return dict(_RECORD)


def primary_train_world(world):
    """Registered primary TRAIN world: merge map 986000-986071 (no 'map' key, the literal checks
    formerly inlined in bank_breadth / policy_synergy / replay_bank / launch_preparation);
    a map world: its own TRAIN registry (process map must match)."""
    if isinstance(world, dict) and 'map' in world:
        try:
            guard_world(world, 'train')
        except ValueError:
            return False
        return True
    return 986000 <= world['seed'] <= 986071


def normalised_world(world):
    """N4 (N11 F1/F7): probe/mining rows carry map worlds as NEWMAP2 moments (cell key string,
    'split', no cell_key); map them onto the exact registered world dict. Merge worlds unchanged."""
    if isinstance(world, dict) and world.get('map'):
        from . import map_scenarios
        if world['map'] != map_scenarios.MAP:
            raise ValueError(f'unregistered map {world["map"]!r}')
        return map_scenarios.world_for_moment({k: world[k] for k in ('seed', 'demand', 'penetration', 'cell')
                                               if k in world})
    return world


def registered_train_world(world):
    """Fail-closed TRAIN predicate for every former 986000-986071 literal on the B/R bank path:
    merge map 986000-986071 exactly as before; a map world only when it normalises to a registered
    TRAIN world of the PROCESS map (989600+ held-out, 964000-964199, 962xxx, 977xxx refused)."""
    if not isinstance(world, dict) or type(world.get('seed')) is not int:
        return False
    if world.get('map'):
        try:
            return primary_train_world(normalised_world(world))
        except (ValueError, KeyError, TypeError, StopIteration):
            return False
    return 986000 <= world['seed'] <= 986071


def primary_train_seed(seed):
    """Seed-only rows (track2_monitor registration): merge 986000-986071, or a TRAIN seed of the
    process map (never held-out/forbidden)."""
    if type(seed) is not int:
        return False
    if 986000 <= seed <= 986071:
        return True
    desc = md.current()
    if desc.is_original:
        return False
    from . import map_scenarios
    return desc.name == map_scenarios.MAP and map_scenarios.seed_scope(seed) == 'train'


def map_replacement_pool(failed, used, taken_ids, arm, ident):
    """accel_runner.replacement_world for map worlds: next unused TRAIN seed of the same cell."""
    from . import map_scenarios
    md.for_world(failed)
    pool = [w for w in map_scenarios.worlds('train', cell_ids=[failed['cell']])
            if w['seed'] not in used and f'{ident}_{arm}_{w["seed"]}' not in taken_ids]
    n = map_scenarios.TRAIN_RANGE[1]-map_scenarios.TRAIN_RANGE[0]+1
    pool.sort(key=lambda w: ((w['repeat']-failed['repeat']) % n, w['seed']))
    return pool[0] if pool else None
