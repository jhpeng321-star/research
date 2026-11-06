"""S-LCEXEC-SMOKE zero-physics parts (CPU only; the episode runner needs SUMO and is never run here)."""
from types import SimpleNamespace

import pytest

from tools.coopmwm import lc_smoke as S, map_descriptor as md
from tools.coopmwm.test_map_descriptor_cpu import MAP_DIR


@pytest.fixture
def wz(monkeypatch):
    monkeypatch.delenv(md.ENV, raising=False)
    md._reset_for_tests()
    yield md.bind(md.load(MAP_DIR))
    md._reset_for_tests()


def test_plan_is_registered_train_worlds_inside_the_cap(wz, tmp_path):
    body = S.plan(tmp_path/'a1', runtime='/rt', checkpoint='/ck.pt', commit='x', attempt='a1')
    assert body['budget']['total_sim_s'] == 13 * S.END_S <= S.PHASE_CAP_SIM_S
    seeds = [j['world']['seed'] for j in body['jobs']]
    assert len(set(seeds)) == 13 and all(989000 <= s <= 989099 for s in seeds)
    assert all(j['id'].endswith('_a1') for j in body['jobs']) and (tmp_path/'ledger').is_dir()
    assert {k.split('/')[0] for k in body['probe_fixtures']} == {'lcmoves'}
    with pytest.raises(ValueError, match='never overwrite'):
        S.plan(tmp_path/'a1')
    assert S.family_order(0, 0)[0] == 'wall_fill' and S.family_order(1, 0)[0] == 'lane_redistribute'
    assert sorted(S.family_order(3, 2)) == sorted(S.FAMILIES)


def test_charge_is_one_phase_ledger_and_refuses_repeats(wz, tmp_path):
    body = S.plan(tmp_path/'a1', runtime='/rt', checkpoint='/ck.pt')
    job = body['jobs'][0]
    S.charge(tmp_path/'a1', job)
    with pytest.raises(ValueError, match='never retried'):
        S.charge(tmp_path/'a1', job)
    big = dict(job, id='x', sim_s=S.PHASE_CAP_SIM_S)
    with pytest.raises(ValueError, match='phase cap'):
        S.charge(tmp_path/'a1', big)


def test_shadow_records_writes_and_passes_reads():
    calls = []
    real = SimpleNamespace(vehicle=SimpleNamespace(getSpeed=lambda v: 12.5, slowDown=lambda *a: calls.append(a)),
                           lane=SimpleNamespace(), simulation=SimpleNamespace())
    sh = S.ShadowApi(real)
    assert sh.vehicle.getSpeed('a') == 12.5
    sh.vehicle.slowDown('a', 13.0, 1.0)
    sh.vehicle.changeLane('a', 0, 2.0)
    sh.vehicle.setLaneChangeMode('a', 513)
    sh.vehicle.setTau('a', 2.0)
    sh.vehicle.setSpeed('a', -1)
    assert calls == [] and sh.take() == [('slowDown', 'a', (13.0, 1.0)), ('changeLane', 'a', (0, 2.0)),
                                         ('setLaneChangeMode', 'a', 513), ('setTau', 'a', 2.0),
                                         ('setSpeed', 'a', -1.0)]
    assert sh.take() == []


def test_spy_passes_through_and_records():
    done = []
    vehicle = SimpleNamespace(**{n: (lambda *a, _n=n: done.append((_n,)+a)) for n in S.WRITES})
    real = SimpleNamespace(vehicle=vehicle)
    spy = S.WriteSpy(real).install()
    real.vehicle.slowDown('a', 10., 1.)
    real.vehicle.changeLane('a', 0, 2.)
    spy.uninstall()
    real.vehicle.setTau('a', 2.)
    assert done == [('slowDown', 'a', 10., 1.), ('changeLane', 'a', 0, 2.), ('setTau', 'a', 2.)]
    assert spy.take() == [('slowDown', 'a', (10., 1.)), ('changeLane', 'a', (0, 2.))]


def test_compare_member_normalisations_and_mismatches():
    probe = {0: [('setLaneChangeMode', 513), ('slowDown', (20., 1.))], 1: [('slowDown', (21., 1.))],
             2: [('setSpeed', -1.), ('setLaneChangeMode', 1621)]}
    ours = {0: [('setLaneChangeMode', 513), ('slowDown', (20.0004, 1.))], 1: [('slowDown', (21., 1.))],
            2: [('slowDown', (18., 1.)), ('setLaneChangeMode', 1621)]}           # base MAPPO after release
    res = S.compare_member(probe, ours, {0, 1})
    assert res['mismatches'] == [] and res['matched'] == 4 and res['max_target_dev'] < 1e-3
    assert res['normalised'] == dict(base_slowdown_dropped=1, N3_release_handback=1)
    # A native lane-only member: probe never slowed it -> its release setSpeed(-1) is a no-op (N1);
    # our owned->native setSpeed(-1) at the first second is the base hand-off (N2).
    probe = {0: [('setLaneChangeMode', 513), ('changeLane', (0, 2.))], 5: [('setSpeed', -1.)]}
    ours = {0: [('setSpeed', -1.), ('setLaneChangeMode', 513)], 5: []}
    res = S.compare_member(probe, ours, set())
    assert [(m['el'], m['side'], m['kind']) for m in res['mismatches']] == [(0, 'probe_only', 'changeLane')]
    assert S.classify(res['mismatches'][0], {0}, set()) == 'gate'
    assert S.classify(res['mismatches'][0], set(), {0}) == 'guard'
    assert res['normalised'] == dict(N1_probe_noop_release=1, N2_ours_release_equivalent=1)
    # Value differences are mismatches.
    res = S.compare_member({0: [('slowDown', (20., 1.))]}, {0: [('slowDown', (20.5, 1.))]}, {0})
    assert len(res['mismatches']) == 2
