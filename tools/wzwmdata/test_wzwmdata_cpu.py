"""CPU tests for the WM-B wz3to2 WM-data collector (no SUMO, no torch)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.wzwmdata import coop, registry as reg

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'


def test_wmdata_assignment_one_cell_per_seed_and_balanced():
    a = reg.wmdata_assignment()
    assert sorted(a) == list(range(989100, 989200))
    counts = Counter(a.values())
    assert len(counts) == 12 and sorted(counts.values()) == [8] * 8 + [9] * 4
    assert {c for c, n in counts.items() if n == 9} == set(reg.CELL_ORDER[:4])
    assert a == reg.wmdata_assignment()                       # deterministic


def test_registry_refusals_and_disjoint_from_reserved():
    seed = min(reg.wmdata_assignment())
    d, p = reg.wmdata_assignment()[seed]
    assert reg.world(seed, d, p, block='WMDATA')['cell_key'] == reg.cell_key(d, p)
    other = next(c for c in reg.CELL_ORDER if c != (d, p))
    with pytest.raises(ValueError):
        reg.world(seed, other[0], other[1], block='WMDATA')
    for bad in (989000, 989099, 989200, 989216, 989579, 989580, 989600, 964100, 962500, 977001):
        with pytest.raises(ValueError):
            reg.world(bad, 3200, 0.6, block='WMDATA')
    for s in range(989100, 989200):
        assert not any(lo <= s <= hi for lo, hi in reg.WMD_RESERVED)
    q = reg.wmqual_worlds()
    assert len(q) == 80 and {w['seed'] for w in q} == set(range(989580, 989600))
    with pytest.raises(ValueError):
        reg.world(989590, 3600, 0.6, block='WMQUAL')
    reg.check_case(989150, 3600)
    with pytest.raises(ValueError):
        reg.check_case(989250, 3200)
    with pytest.raises(ValueError):
        reg.check_case(989150, 3000)


def _frame(rows):
    """rows: (vid, lane, s, v, is_cav, junction)."""
    k = 16
    f = SimpleNamespace(present=np.zeros(k, bool), is_cav=np.zeros(k, bool), junction=np.zeros(k, bool),
                        s=np.zeros(k), v=np.zeros(k), lane=np.full(k, -1), vid_index=np.full(k, -1))
    ids = []
    for i, (vid, lane, s, v, cav, jn) in enumerate(rows):
        f.present[i], f.is_cav[i], f.junction[i] = True, cav, jn
        f.s[i], f.v[i], f.lane[i], f.vid_index[i] = s, v, lane, i
        ids.append(vid)
    return f, ids


def test_pick_pair_wall_and_hdvs_never_selected():
    drop = 75.44
    f, ids = _frame([('c0', 2, -1000., 25., True, False), ('c1', 1, -950., 25., True, False),
                     ('c2', 0, -1100., 25., True, False), ('h0', 2, -990., 25., False, False),
                     ('c3', 1, -200., 25., True, False)])
    cands = coop.zone_cavs(f, ids, drop)
    assert [c[0] for c in cands] == ['c0', 'c1', 'c2']           # HDV and out-of-zone CAV excluded
    assert [c[0] for c in coop.pick_pair(cands, 2, 1)] == ['c0', 'c1']
    assert sorted(c[0] for c in coop.pick_wall(cands, (0, 1, 2))) == ['c0', 'c1', 'c2']


def test_schedule_commands_and_release():
    drop = 75.44
    f, ids = _frame([('c0', 2, -1000., 30., True, False), ('c1', 1, -980., 30., True, False),
                     ('c2', 0, -1050., 30., True, False)])
    sch = coop.CoopSchedule('t:1', drop_s=drop, start_s=300., end_s=1200.)
    t = 300.
    accel = {}
    while not accel and t < 400:
        accel, hold = sch.commands(t, f, ids)
        t += 1
    assert accel, 'a move must start within the first 60 s'
    rec = sch.moves[0]
    p = rec['params']
    for slot, a in accel.items():
        assert -p['dmax'] - 1e-9 <= a <= coop.AMAX + 1e-9
        assert a == pytest.approx(max(-p['dmax'], min(coop.AMAX, p['v_w'] - 30.)))
        assert hold[slot] == f.lane[slot]
    t_end = rec['t_start'] + p['hold_s']
    for tt in np.arange(t, t_end + 2):
        sch.commands(float(tt), f, ids)
    assert sch.active is None and set(rec['releases']) == {m['id'] for m in rec['members']}
    assert all(r['why'] in ('hold_end', 'anchor') for r in rec['releases'].values())
    assert sch.commands(1200., f, ids) == ({}, {})


def test_plan_budget(tmp_path, monkeypatch):
    if not SPEC.exists():
        pytest.skip('map asset missing')
    from tools.coopmwm import map_descriptor as md
    from tools.wzwmdata import collect
    monkeypatch.setenv(md.ENV, str(SPEC))
    md._reset_for_tests()
    try:
        body = collect.plan(tmp_path / 'run')
    finally:
        md._reset_for_tests()
    b = body['budget']
    assert b['wmdata_sim_s'] == 100 * 4 * 1200 and b['wmqual_sim_s'] == 80 * 1500
    assert b['total_sim_s'] == 600_000 <= collect.TASK_CAP_SIM_S
    assert {j['arm'] for j in body['jobs']} == {'native', 'mappo_u0', 'coop_mappo_u0', 'coop_native'}
    ids = [j['id'] for j in body['jobs']]
    assert len(ids) == len(set(ids)) == 480
    canary = [j for j in body['jobs'] if j['wave'] == 'canary']
    assert len(canary) == 5 and sum(j['block'] == 'WMQUAL' for j in canary) == 1
    assert all(j['split'] == 'WMQUAL' for j in body['jobs'] if j['block'] == 'WMQUAL')
    assert all('wmqual' in j['path'] for j in body['jobs'] if j['block'] == 'WMQUAL')
    with pytest.raises(ValueError):
        collect.plan(tmp_path / 'run')


def test_every_variant_runs_without_anchor_errors():
    drop = 75.44
    f, ids = _frame([('c0', 2, -1000., 30., True, False), ('c1', 1, -980., 30., True, False),
                     ('c2', 0, -1050., 30., True, False)])
    seen = set()
    for i in range(60):
        sch = coop.CoopSchedule(f't:{i}', drop_s=drop, start_s=300., end_s=1200.)
        for tt in range(300, 700):
            f.s[:3] += 0.0
            sch.commands(float(tt), f, ids)
        seen.update(m['variant'] for m in sch.moves)
        for m in sch.moves:
            assert (m['anchor'] is not None) == (m['variant'] == 'pair_t')
    assert {'pair_g', 'pair_h', 'pair_t', 'wall_g', 'wall_h', 'wall_l', 'R_random'} <= seen


def test_plan_skip_and_prior(tmp_path, monkeypatch):
    if not SPEC.exists():
        pytest.skip('map asset missing')
    from tools.coopmwm import map_descriptor as md
    from tools.wzwmdata import collect
    monkeypatch.setenv(md.ENV, str(SPEC))
    md._reset_for_tests()
    try:
        skip = ['D3200_p60_s989103_native', 'D3200_p60_s989103_mappo', 'WMQUAL_D3200_p60_s989580_native']
        body = collect.plan(tmp_path / 'run', data_root=tmp_path / 'data', skip=skip,
                            prior_charged_sim_s=7500, host='gpuA')
    finally:
        md._reset_for_tests()
    ids = {j['id'] for j in body['jobs']}
    assert 'D3200_p60_s989103_native' not in ids and 'D3200_p60_s989103_mappo_u0' not in ids
    assert 'WMQUAL_D3200_p60_s989580_native' not in ids and 'D3200_p60_s989103_coop_native' in ids
    assert all(str(tmp_path / 'data') in j['path'] for j in body['jobs'])
    assert body['budget']['total_sim_s'] == 600_000 - 2 * 1200 - 1500 and body['host'] == 'gpuA'


def test_val_split_one_world_per_cell_and_holdouts_refused():
    from tools.wzwmdata import shards
    v = shards.val_seeds()
    assert len(v) == 12
    assert len({reg.wmdata_assignment()[x] for x in v}) == 12
    assert sum(shards.split_of(x) == 'val' for x in range(989100, 989200)) == 12
    for bad in (989200, 989210, 989351, 989401, 989585):
        with pytest.raises(ValueError):
            shards.split_of(bad)


def test_job_order_round_robin_over_cells(tmp_path, monkeypatch):
    if not SPEC.exists():
        pytest.skip('map asset missing')
    from tools.coopmwm import map_descriptor as md
    from tools.wzwmdata import collect
    monkeypatch.setenv(md.ENV, str(SPEC))
    md._reset_for_tests()
    try:
        jobs = collect.make_jobs(tmp_path, runtime='r', checkpoint='c')
    finally:
        md._reset_for_tests()
    first48 = [j['world']['cell_key'] for j in jobs[:48]]
    assert len(set(first48)) == 12
    assert jobs[0]['world']['seed'] == 989103 and jobs[0]['wave'] == 'canary'
