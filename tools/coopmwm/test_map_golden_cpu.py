"""N1 golden identity: the original merge map stays byte-identical after the map abstraction.

``fixtures/n1_original_golden.json`` was written by ``python -m tools.coopmwm.test_map_golden_cpu``
on the UNCHANGED clean3 base (c4/clean3-launch 3e5bc9f9 + c4/clean3-g4fix b7d6cde7) in its own
commit, BEFORE any N1 code change. Every value is serialised exactly (floats as ``float.hex``,
dict/tuple structure and Python types kept), so equality is bit-level, not approximate.

Covered (training/runtime path observation builders and frame constants):
coopnative.policy.encode (29 per-CAV features, 12-dim RSU block, anchors), coopfree.selection.features,
coopfree.protocol.ZONE, variety.VARIETY_REGISTERED, moves zone/lane constants, coop_search.RAMP_LANE,
merge_features constants + an enabled feature block, plan_proposer.MERGE_ZONE, and the frozen
frame constants (geometry LANE_TABLE / HERE_OFFSETS / lane_and_s, state RSU zones,
wms.control.contract merge/RSU constants).
"""
import json
import math
import random
from pathlib import Path

FIXTURE = Path(__file__).with_name('fixtures')/'n1_original_golden.json'


def exact(value):
    """Bit-exact, type-preserving JSON encoding."""
    if isinstance(value, bool):
        return dict(bool=value)
    if isinstance(value, float):
        return dict(float=value.hex())
    if isinstance(value, int):
        return dict(int=value)
    if isinstance(value, str):
        return dict(str=value)
    if value is None:
        return dict(none=True)
    if isinstance(value, tuple):
        return dict(tuple=[exact(v) for v in value])
    if isinstance(value, list):
        return dict(list=[exact(v) for v in value])
    if isinstance(value, dict):
        return dict(dict=[[exact(k), exact(v)] for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))])
    import numpy as np
    if isinstance(value, np.ndarray):
        return dict(ndarray=dict(dtype=str(value.dtype), shape=list(value.shape), hex=value.tobytes().hex()))
    if isinstance(value, np.generic):
        return dict(np=dict(dtype=str(value.dtype), hex=value.tobytes().hex()))
    raise TypeError(type(value))


def fixture_rows(seed, n=40, boundary=False):
    """Deterministic observation rows in the original WM frame (coopnative FIELDS)."""
    rng = random.Random(f'n1_golden_rows:{seed}')
    rows = []
    for i in range(n):
        lane = rng.randrange(3)
        s = rng.uniform(-400., 300.)
        rows.append(dict(vid=f'{"cav" if rng.random() < .35 else "hdv"}.{seed}.{i}', s_m=s,
                         v_mps=rng.uniform(0., 33.), lane=lane, is_cav=None, length_m=rng.choice((5., 4.3, 7.1))))
    for r in rows:
        r['is_cav'] = r['vid'].startswith('cav')
    if boundary:
        for k, s in enumerate((-100., 75.44, -100.0000001, 75.4400001, 0., -800., 225.44, -378.66)):
            rows.append(dict(vid=f'cav.edge.{k}', s_m=s, v_mps=10.+k, lane=k % 3, is_cav=bool(k % 2), length_m=5.))
    return rows


def compute():
    from tools.coopnative import policy as cn
    from tools.coopfree import selection, protocol as cfp
    from tools.coopmwm import variety, moves, coop_search, merge_features as mf, plan_proposer
    out = {}
    cases = [fixture_rows(s, boundary=(s % 2 == 0)) for s in range(6)]
    out['encode'] = [exact(cn.encode(rows)) for rows in cases]
    out['selection_features'] = [exact(selection.features(rows)) for rows in cases]
    out['coopfree_zone'] = exact(cfp.ZONE)
    out['variety_registered'] = exact(dict(variety.VARIETY_REGISTERED))
    out['moves_constants'] = exact(dict(ramp=moves._RAMP_LANE, main=moves._MAIN_LANE,
                                        zone=(moves._ZONE_S0, moves._ZONE_S1), wall_min=moves._WALL_MIN_LANES))
    out['coop_search_ramp_lane'] = exact(coop_search.RAMP_LANE)
    out['merge_features_constants'] = exact(dict(s0=mf.MERGE_S0_M, s1=mf.MERGE_S1_M, n=mf.N_LANES,
        sensor=mf.SENSOR_RANGE_M, queue=mf.QUEUE_SPEED_MPS, features=mf.MERGE_FEATURES))
    blocks = []
    for seed in range(4):
        rng = random.Random(f'n1_golden_merge:{seed}')
        def veh(name, lane=None):
            return mf.ObservedVehicle(name, rng.uniform(-200., 150.), rng.randrange(3) if lane is None else lane,
                                      rng.uniform(0., 30.), 5.)
        own_src = veh('ego')
        own = mf.SensorReport(10., own_src, (0, 1, 2), tuple(veh(f'o{i}') for i in range(8)))
        partner_src = mf.ObservedVehicle('p1', own_src.s_m+rng.uniform(-50., 50.), 2, 12., 5.)
        partner = mf.SensorReport(9., partner_src, (1, 2), tuple(veh(f'q{i}') for i in range(6)))
        block = mf.merge_features(own, 'ego', {'p1': partner}, now_s=10., partner_ids=('p1',),
                                  cfg=mf.MergeFeaturesConfig(merge_features=True))
        blocks.append(exact(block))
    out['merge_features_blocks'] = blocks
    out['plan_proposer_merge_zone'] = exact(plan_proposer.MERGE_ZONE)
    return out


def frozen_frame():
    """Frame constants of the frozen runtime modules as importable in this checkout."""
    from traffic_shepherd.c4 import geometry, state
    out = dict(lane_table=exact({f'{e}|{i}': v for (e, i), v in geometry.LANE_TABLE.items()}),
               here_offsets=exact(geometry.HERE_OFFSETS),
               merge=exact((geometry.MERGE_S0_M, geometry.MERGE_S1_M)),
               rsu=exact((state.RSU_MAINLINE_S_M, state.RSU_RAMP_S_M, state.RSU_CELL_M)))
    grid = []
    for (edge, idx) in sorted(geometry.LANE_TABLE):
        for pos in (0., 1.5, 37.25, 75.44, 200., 1000.123):
            grid.append(exact(geometry.lane_and_s(f'{edge}_{idx}', pos)))
    out['lane_and_s'] = grid
    return out


def test_original_observation_builders_byte_identical():
    golden = json.loads(FIXTURE.read_text(encoding='utf-8'))
    current = compute()
    for key, value in golden['tools'].items():
        assert current[key] == value, key


def test_original_frozen_frame_constants_byte_identical():
    golden = json.loads(FIXTURE.read_text(encoding='utf-8'))
    assert frozen_frame() == golden['frozen_frame']


def test_original_descriptor_reproduces_frozen_frame():
    """The default descriptor IS the original map: same table, offsets, lane_and_s bits."""
    from tools.coopmwm import map_descriptor as md
    from traffic_shepherd.c4 import geometry, state
    d = md.ORIGINAL
    assert d.lane_table == geometry.LANE_TABLE and d.here_offsets == geometry.HERE_OFFSETS
    assert (d.frame.merge_zone_s_m, d.frame.rsu_mainline_s_m, d.frame.rsu_ramp_s_m, d.frame.rsu_cell_m) == (
        (geometry.MERGE_S0_M, geometry.MERGE_S1_M), state.RSU_MAINLINE_S_M, state.RSU_RAMP_S_M, state.RSU_CELL_M)
    for (edge, idx) in sorted(geometry.LANE_TABLE):
        for pos in (0., 1.5, 37.25, 75.44, 200., 1000.123, 2644.5):
            a, b = d.lane_and_s(f'{edge}_{idx}', pos), geometry.lane_and_s(f'{edge}_{idx}', pos)
            assert a[0] == b[0] and a[1].hex() == b[1].hex()
    assert d.k_slots == 192 and d.rsu.source == 'rsu_detector'
    assert all(z == (-100., 75.44) for z in d.rsu.block_zone_s_m)


def test_original_contract_constants_match_frame():
    from wms.control import contract
    from tools.coopmwm import map_descriptor as md
    f = md.ORIGINAL.frame
    assert (contract.MERGE_S0_M, contract.MERGE_S1_M) == f.merge_zone_s_m
    assert contract.RSU_MAINLINE_S_M == f.rsu_mainline_s_m and contract.RSU_RAMP_S_M == f.rsu_ramp_s_m
    assert contract.RSU_CELL_M == f.rsu_cell_m and contract.N_LANES == f.n_lanes
    assert contract.SPEED_LIMIT_MPS == f.speed_limit_mps


if __name__ == '__main__':
    FIXTURE.parent.mkdir(exist_ok=True)
    if FIXTURE.exists():
        raise SystemExit('golden fixture exists; never overwrite')
    FIXTURE.write_text(json.dumps(dict(
        note='written on the unchanged clean3 base (3e5bc9f9 + g4fix b7d6cde7) before any N1 change',
        tools=compute(), frozen_frame=frozen_frame()), sort_keys=True)+'\n', encoding='utf-8')
    print('wrote', FIXTURE)
