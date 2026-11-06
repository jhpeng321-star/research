"""LCEXEC exports for the N7 lane MAPPO (S-MAPPOWZL): lane rule table, per-ego gate, lane writer.

CPU only (fake TraCI via lc_writer_harness, no SUMO). The table must reproduce lc_families.lane_gate
exactly (so the MAPPO action set equals the free_program policy's), and the writer must equal the
frozen ``_issue_lane_commands`` everywhere except closing-lane entries on the lane-drop map.
"""
import math

import numpy as np
import pytest
import torch

from tools.coopmwm import lc_families as lf, lc_program as lp, lc_runtime, map_descriptor as md
from tools.coopmwm.lc_writer_harness import Harness
from tools.coopmwm.test_lc_runtime_cpu import (MAP_DIR, ROSTER, S, current_env, merge_states, wz,  # noqa: F401
                                               wz_states, _unbound)


def _allows(table, source, target, s, v):
    from wms.control.lanes import LaneRules
    guard = LaneRules().edge_guard_m
    for t in table['transitions']:
        if t['source'] == source and t['target'] == target and t['s_lo'] <= s <= t['s_hi']:
            if t['guard'] and any(0 <= p-s < v*1.0+guard for p in table['guard_points_s_m']):
                continue
            return True
    return False


def _gap_ok(rows, ego, target):
    from wms.control.lanes import LaneRules, _target_gap_ok
    tensor = lambda k, kind=float: torch.tensor([[kind(r[k]) for r in rows]])  # noqa: E731
    lane = tensor('lane', int)
    goal = lane.clone()
    goal[0, rows.index(ego)] = target
    return bool(_target_gap_ok(tensor('s_m'), tensor('v_mps'), tensor('length_m'),
                               torch.ones_like(lane, dtype=torch.bool), lane, goal, LaneRules())[0, rows.index(ego)])


def test_rule_table_is_the_lane_gate_on_random_own_sensing_states():
    desc = wz()
    frame = lp.frame_from_descriptor(desc)
    table = lc_runtime.lane_rule_table(desc)
    assert table['schema'] == lf.LANE_RULES_SCHEMA and table['map'] == 'wz3to2'
    assert {(t['source'], t['target'], t['guard']) for t in table['transitions']} == {
        (0, 1, True), (1, 0, True), (1, 2, True), (2, 1, False)}
    assert table['guard_points_s_m'] == [75.44] and table['sensor_range_m'] == 100.
    rng = np.random.default_rng(20261106)
    checked = agree = 0
    for _ in range(300):
        n = int(rng.integers(2, 9))
        rows = []
        for i in range(n):
            lane = int(rng.integers(0, 3))
            s = float(rng.uniform(-400., 74.)) if lane == 2 else float(rng.uniform(-400., 300.))
            rows.append(dict(vid=f'v{i}', s_m=round(s, 2), v_mps=round(float(rng.uniform(0., 30.)), 3), lane=lane,
                             is_cav=bool(rng.random() < .6), length_m=5.))
        cavs = [r for r in rows if r['is_cav']]
        local = {r['vid']: dict(self=r['vid'], rows=[x for x in rows if abs(x['s_m']-r['s_m']) <= 100.])
                 for r in cavs}
        for delta in (-1, 1):
            requests = {r['vid']: r['lane']+delta for r in cavs}
            safe, blocked = lf.mappo_lane_gate(requests, local, {r['vid'] for r in cavs}, frame=frame, n_lanes=3)
            for r in cavs:
                target = r['lane']+delta
                own_rows = local[r['vid']]['rows']
                want = (target in range(3) and _allows(table, r['lane'], target, r['s_m'], r['v_mps'])
                        and _gap_ok(own_rows, r, target))
                checked += 1
                agree += int((r['vid'] in safe) == want)
    assert checked > 500 and agree == checked


def test_gate_fails_closed_on_junction_rows_and_missing_sensing():
    frame = lp.frame_from_descriptor(wz())
    rows = [dict(vid='a', s_m=-100., v_mps=20., lane=1, is_cav=True, length_m=5.),
            dict(vid='j', s_m=-60., v_mps=20., lane=-1, is_cav=False, length_m=5.)]
    safe, blocked = lf.mappo_lane_gate({'a': 0}, {'a': dict(self='a', rows=rows)}, {'a'}, frame=frame, n_lanes=3)
    assert safe == {} and blocked['a'] == 'missing_observable_gap_or_cav'
    safe, blocked = lf.mappo_lane_gate({'a': 0, 'b': 1}, {'a': dict(self='a', rows=rows[:1])}, {'a'},
                                       frame=frame, n_lanes=3)
    assert safe == {'a': 0} and blocked == {'b': 'missing_observable_gap_or_cav'}


def test_mappo_rules_need_a_lane_drop_map_and_fall_back_to_the_table():
    with pytest.raises(ValueError, match='lane-drop'):
        lc_runtime.lane_rule_table(md.ORIGINAL)
    rules = lc_runtime.mappo_lane_rules(wz())
    try:
        from tools.nm_mappo.lane_actions import GateRules
    except ImportError:
        assert rules == lc_runtime.lane_rule_table(wz())
    else:
        assert isinstance(rules, GateRules) and len(rules.transitions) == 4


def _writer_env(mod):
    class WriterEnv(mod.CavSumoEnv):
        def _issue_lane_commands(self, targets, policy_cavs, statuses=None):
            return lc_runtime.mappo_lane_writer(self, targets, policy_cavs, statuses)
    return WriterEnv


def _pair(desc, **kw):
    mod = current_env()
    frozen = Harness(None, ROSTER, desc=desc, env_module=mod, env_class=mod.CavSumoEnv, **kw)
    writer = Harness(None, ROSTER, desc=desc, env_module=mod, env_class=_writer_env(mod), **kw)
    writer.env_globals = mod.CavSumoEnv._issue_lane_commands.__globals__
    return frozen, writer


BASE = dict(c1=0., c2=0., c3=0.)
ACTORS = frozenset({'c1', 'c2', 'c3'})


@pytest.mark.parametrize('states,lanes', [
    (dict(), dict(c1=0)), (dict(), dict(c1=1)), (dict(), dict(c3=1)), (dict(), dict(c2=1)),
    (dict(c1=S('up', 1, 2690., 25.)), dict(c1=0)), (dict(c1=S('wz', 0, 2800., 20.)), dict(c1=0)),
    (dict(c1=S(':n1_0', 0, 2700., 20.)), dict(c1=0)), (dict(), dict(c3=2)), (dict(), dict(c1=3))])
def test_writer_equals_the_frozen_writer_off_the_closing_lane(monkeypatch, states, lanes):
    monkeypatch.setattr(md, '_BOUND', wz())
    frozen, writer = _pair(wz())
    a = frozen.tick(0, wz_states(**states), base=BASE, lanes=lanes, lane_actors=ACTORS)
    b = writer.tick(0, wz_states(**states), base=BASE, lanes=lanes, lane_actors=ACTORS)
    assert a == b and frozen.last_info['lane_vehicles'] == writer.last_info['lane_vehicles']
    assert frozen.last_info['lane_commands'] == writer.last_info['lane_commands']


def test_writer_enters_the_closing_lane_on_wz3to2_only(monkeypatch):
    monkeypatch.setattr(md, '_BOUND', wz())
    frozen, writer = _pair(wz())
    lanes = dict(c1=2)                                     # c1 on SUMO up_1 = WM 1 -> WM 2 (SUMO 0)
    a = frozen.tick(0, wz_states(), base=BASE, lanes=lanes, lane_actors=ACTORS)
    b = writer.tick(0, wz_states(), base=BASE, lanes=lanes, lane_actors=ACTORS,
                    after=wz_states(c1=S('up', 0, 2000., 20.)))
    assert 'changeLane' not in [c[2] for c in a]
    assert frozen.last_info['lane_vehicles']['c1']['status'] == 'refused_not_allowed'
    assert (0, 'c1', 'changeLane', (0, 1.0)) in b
    assert writer.last_info['lane_vehicles']['c1'] == dict(target=2, status='issued', completed=True)
    # Closing lane ending within v*1 s + 10 m ahead: deferred (2696 - 2680 = 16 <= 20 + 10).
    _, writer = _pair(wz())
    b = writer.tick(0, wz_states(c1=S('up', 1, 2680., 20.)), base=BASE, lanes=lanes, lane_actors=ACTORS)
    assert 'changeLane' not in [c[2] for c in b]
    assert writer.last_info['lane_vehicles']['c1']['status'] == 'deferred_edge_end'
    # Downstream edge has no closing lane.
    _, writer = _pair(wz())
    writer.tick(0, wz_states(c1=S('wz', 0, 2800., 20.)), base=BASE, lanes=lanes, lane_actors=ACTORS)
    assert writer.last_info['lane_vehicles']['c1']['status'] == 'refused_no_lane'


@pytest.mark.parametrize('lanes', [dict(c1=1), dict(c1=2), dict(c2=1), dict(c2=0), dict(c3=0), dict(c3=2)])
def test_writer_is_the_frozen_rule_on_the_merge_map(lanes):
    assert md.current().is_original
    frozen, writer = _pair(md.ORIGINAL, default_lane_length=1000.)
    a = frozen.tick(0, merge_states(), base=BASE, lanes=lanes, lane_actors=ACTORS)
    b = writer.tick(0, merge_states(), base=BASE, lanes=lanes, lane_actors=ACTORS)
    assert a == b and frozen.last_info['lane_vehicles'] == writer.last_info['lane_vehicles']


def test_table_entry_into_the_closing_lane_ends_strictly_before_the_drop():
    table = lc_runtime.lane_rule_table(wz())
    into = next(t for t in table['transitions'] if t['target'] == 2)
    assert into['s_hi'] < 75.44000000000005 and math.isinf(into['s_lo'])
