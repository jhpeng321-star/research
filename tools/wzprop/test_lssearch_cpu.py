"""CPU tests for S-LOWSHARE-SEARCH (tools.wzprop.lssearch): specs, selection, core parity, per-vehicle logger."""
from __future__ import annotations

import types

from tools.wzprop import lssearch as L
from tools.wzprop import strategy_hook as SH
from tools.wzprop import strategy_search as SS


def _rows(cavs, hdvs=()):
    rows = [dict(vid=v, s_m=x - SH.S_OFF, v_mps=sp, lane=2 - ln, is_cav=True) for v, x, ln, sp in cavs]
    rows += [dict(vid=v, s_m=x - SH.S_OFF, v_mps=sp, lane=2 - ln, is_cav=False) for v, x, ln, sp in hdvs]
    return rows


WALL = [('c0', 1500.0, 0, 25.0), ('c1', 1550.0, 1, 26.0), ('c2', 1600.0, 2, 27.0), ('c2b', 1400.0, 2, 27.5),
        ('c2far', 1100.0, 2, 28.0)]


def test_specs_registered_and_hold_only():
    assert len(L.GROUPS) == 10 and set(L.SPECS) == {'wall_h_solo2', 'wall_h'} | {s['name'] for s in L.GROUPS}
    for s in L.SPECS.values():
        assert all(r['lc_target'] is None for r in s['roles']) and s['cooldown_s'] == 90
    inc = dict(L.INCUMBENT)
    assert {k: v for k, v in inc.items() if k != 'name'} == {k: v for k, v in SS.SOLO2.items() if k != 'name'}
    assert len(L.SCREEN_ARMS) == 13


def test_lscore_identical_to_strategycore_for_wall_specs():
    for spec in (L.INCUMBENT, L.WALL_H, L.SPECS['pw2_h2g1'], L.SPECS['gap1_m_o20']):
        a, b = L.LSCore(spec), SH.StrategyCore(dict(spec))
        rows = _rows(WALL)
        cmds = {r['vid']: 0.0 for r in rows}
        for t in range(300, 520, 7):
            for r in rows:
                r['s_m'] += 7 * r['v_mps'] * 0.5
            assert a.step(rows, cmds, t) == b.step(rows, cmds, t)
        assert [x['members'] for x in a.acts] == [x['members'] for x in b.acts]


def test_relay_helper_upstream_within_spread_and_optional():
    cavs = [SH.to_sumo_row(r) for r in _rows(WALL)]
    mem = L.select(L.SPECS['relay2_o60'], cavs)
    assert [m['id'] for m in mem] == ['c2', 'c2b']           # c2far is 500 m upstream (> 300 m)
    mem = L.select(L.SPECS['relay2_o60'], [c for c in cavs if c['id'] != 'c2b'])
    assert mem[0]['id'] == 'c2' and mem[1] is None
    core = L.LSCore(L.SPECS['relay2_o60'])
    rows = [r for r in _rows(WALL) if r['vid'] != 'c2b']
    acc, lanes = core.step(rows, {r['vid']: 0.0 for r in rows}, 300)
    assert set(acc) == {'c2'} and core.acts[0]['missing'] == [1] and core.acts[0]['members'] == ['c2']
    mem = L.select(L.SPECS['stagger3'], cavs)
    assert [m['id'] for m in mem] == ['c2', 'c1', 'c2b']
    assert L.select(L.SPECS['relay2_o40'], [c for c in cavs if c['lane'] != 0]) is None   # no wall -> infeasible


def test_onset_and_busy_blocking():
    core = L.LSCore(L.SPECS['relay2_o60'])
    rows = _rows(WALL)
    cmds = {r['vid']: 0.0 for r in rows}
    acc, _ = core.step(rows, cmds, 300)
    assert set(acc) == {'c2'}
    acc, _ = core.step(rows, cmds, 359)
    assert set(acc) == {'c2'}
    acc, _ = core.step(rows, cmds, 360)
    assert set(acc) == {'c2', 'c2b'}
    acc, _ = core.step(rows, cmds, 400)                       # t >= next_ok but the helper is still busy
    assert len(core.acts) == 1 and set(acc) == {'c2b'}


class _Ledger:
    def __init__(self, reqs):
        from traffic_shepherd.metrics import RequestLedger
        from traffic_shepherd.scenarios import TripRequest
        self.inner = RequestLedger([TripRequest(v, d, 'main', 'x', c) for v, d, c in reqs])

    def __getattr__(self, k):
        return getattr(self.inner, k)


def test_pervehicle_reconciles_with_ledger_and_deficit():
    import sys
    mod = types.ModuleType('fake_env_mod')
    mod.SPEED_LIMIT_MPS = 31.29
    sys.modules['fake_env_mod'] = mod

    class Env:
        pass
    Env.__module__ = 'fake_env_mod'
    env = Env()
    reqs = [('a', 0.0, True), ('b', 299.0, False), ('c', 301.0, False), ('d', 1199.0, True), ('e', 1250.0, False)]
    env.ledger = _Ledger(reqs).inner
    env._table = {}
    env.now = 0.0
    total = dict(road=0.0, src=0.0)
    lanes = dict(a=2, b=0, c=1, d=1)
    insert_at = dict(a=0.5, b=302.0, c=310.0)
    present = set()

    def physics(after):
        departed = [v for v, t in insert_at.items() if t == after]
        present.update(departed)
        before = env.ledger.source_vehicle_seconds
        env.ledger.advance(after, departed, [])
        env._table = {v: dict(road_id='up', lane_index=lanes[v], speed_mps=20.0 if v != 'c' else 35.0)
                      for v in sorted(present)}
        if 300 < after <= 1200:
            total['road'] += 0.5 * sum(max(0.0, 1.0 - t['speed_mps'] / 31.29) for t in env._table.values())
            total['src'] += env.ledger.source_vehicle_seconds - before
        env.now = after

    env._after_physics = lambda after, *a: physics(after)
    pv = L.PerVehicle()
    env._after_physics(0.5)
    env._after_physics(1.0)
    pv.install(env)
    t = 1.0
    while t < 1200.0:
        t += 0.5
        env._after_physics(t)
    out = pv.finish()
    assert out['error'] is None
    assert abs(out['sum_road_vs'] - total['road']) < 1e-6 and abs(out['sum_source_vs'] - total['src']) < 1e-6
    by = {r[0]: r for r in out['rows']}
    assert set(by) == {'a', 'b', 'c', 'd'}
    assert by['a'][2] == 2 and by['c'][2] == 1 and by['d'][2] == -1 and by['c'][3] == 0.0
    assert abs(by['b'][4] - 2.0) < 1e-9 and abs(by['c'][4] - 9.0) < 1e-9 and abs(by['d'][4] - 1.0) < 1e-9


def test_ineq_and_fairness_rule():
    q = L.ineq([0, 0, 0, 10])
    assert abs(q['gini'] - 0.75) < 1e-12 and abs(q['jain'] - 0.25) < 1e-12 and q['worst10'] == 10
    rows = []
    for s in range(6):
        for arm, d0, d1 in (('u0', 10.0, 10.0), ('x', 8.0, 10.5), ('y', 7.0, 9.0)):
            g = {k: dict(n=1, total_vs=v, mean_vs=v) for k, v in
                 (('lane0', d0), ('lane1', d1), ('lane2', d1), ('hdv', d0), ('cav', d1), ('uninserted', 0.0))}
            g['uninserted'] = dict(n=0, total_vs=0.0, mean_vs=None)
            rows.append(dict(seed=s, arm=arm, cell='c', status='ok', J_300_1200_vs=d0 + d1 + s * 0.01 * (arm == 'x'),
                             fair=dict(groups=g, gini=0.1, jain=0.9, worst10=d1)))
    fx, fy = L.fairness_stats(rows, 'c', 'x', 'u0'), L.fairness_stats(rows, 'c', 'y', 'u0')
    assert fx['FAIR'] is False and 'lane1' in fx['worse_groups_sig']
    assert fy['FAIR'] is True and fy['worse_groups_point'] == []
