"""LCEXEC env-side knobs (lc_runtime) and the writer harness: CPU only (fake TraCI, no SUMO).

* install: inert without COOPMWM_LC_RUNTIME, fails closed on the merge map, binds once on wz3to2;
* golden: without a stash the LCEXEC subclass step is byte-identical to the base env step
  (repo env and the committed G5 overlay = frozen env + G5 hunks; merge map and wz3to2);
* knobs: exact accel, native, lane modes 513/512, closing-lane targets, durations, tau, HDV refusal;
* parity: probe Arm (sha-pinned verbatim copies of lcm_core / pg2_core) vs lc_program executor +
  requests_from_commands + the real env writer, same states, same TraCI writes.
"""
import ast
import functools
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.coopmwm import g5_runtime, lc_program as lp, lc_roles, lc_runtime, map_descriptor as md
from tools.coopmwm.lc_writer_harness import FakeTraci, Harness, requests_from_commands

REPO = Path(__file__).resolve().parents[2]
MAP_DIR = REPO/'assets/research/newmap_candidates/wz3to2'
FIX = Path(__file__).with_name('fixtures')/'lcexec_rt'
LIMIT = 31.29


@pytest.fixture(autouse=True)
def _unbound(monkeypatch):
    monkeypatch.delenv(md.ENV, raising=False)
    monkeypatch.delenv(lc_runtime.FLAG, raising=False)
    md._reset_for_tests()
    yield
    md._reset_for_tests()


@functools.lru_cache(maxsize=None)
def wz():
    return md.load(MAP_DIR)


@functools.lru_cache(maxsize=None)
def overlay_env():
    """The committed G5 overlay (frozen env 59aecd0b + G5 hunks) as a module of the c4 package."""
    src = REPO/'tools/coopmwm/g5_runtime_env.pysrc'
    assert g5_runtime.normalized_sha(src) == g5_runtime.OVERLAY_SHA256
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader('traffic_shepherd.c4._lcexec_rt_g5', None))
    module.__package__ = 'traffic_shepherd.c4'
    exec(compile(src.read_text(encoding='utf-8'), str(src), 'exec'), module.__dict__)
    return module


def env_module(which):
    return importlib.import_module('traffic_shepherd.c4.env') if which == 'repo' else overlay_env()


def S(edge, lane, x, v, cav=True, allowed=LIMIT, **kw):
    return dict(edge=edge, sumo_lane=lane, x=float(x), v=float(v), allowed=float(allowed), is_cav=cav, **kw)


ROSTER = {'c1': True, 'c2': True, 'c3': True, 'h1': False}


def wz_states(**over):
    st = {'c1': S('up', 1, 2000., 20.), 'c2': S('up', 0, 2100., 15.), 'c3': S('up', 2, 1800., 22.),
          'h1': S('up', 1, 1900., 25., cav=False)}
    st.update(over)
    return st


def merge_states(**over):
    st = {'c1': S('1191554636', 0, 0., 20., pos=100.), 'c2': S('1487606179', 0, 0., 18., pos=20.),
          'c3': S('1191554636', 1, 0., 22., pos=300.), 'h1': S('1191554636', 1, 0., 25., cav=False, pos=50.)}
    st.update(over)
    return st


_CURRENT = {'which': 'repo'}


@pytest.fixture(params=['repo', 'g5_overlay'])
def both_envs(request):
    """Run a writer test on the repo env AND on the committed G5 overlay (frozen env + G5 hunks)."""
    _CURRENT['which'] = request.param
    yield request.param
    _CURRENT['which'] = 'repo'


def current_env():
    return env_module(_CURRENT['which'])


def harness(**kw):
    return Harness(None, kw.pop('roster', ROSTER), desc=kw.pop('desc', wz()), env_module=current_env(), **kw)


# ------------------------------------------------------------------ install
def run_py(code, extra_env=None, map_spec=True):
    from tools.coopmwm.test_map_runtime_cpu import run_py as _run
    return _run(code, map_spec=map_spec, extra_env=extra_env)


def test_install_is_inert_without_the_flag_and_imports_nothing():
    out = run_py(r'''
import json, sys
import tools.coopmwm
before = set(sys.modules)
from tools.coopmwm import lc_runtime
r = lc_runtime.install(dict(world=dict(seed=986000, cell=0, repeat=0, penetration=.05, demand=1900), role='train'))
import os; os.environ['COOPMWM_LC_RUNTIME'] = '0'
r0 = lc_runtime.install(dict(world=dict(map='wz3to2', seed=989201), role='train'))
print(json.dumps(dict(r=r, r0=r0, new=sorted(set(sys.modules)-before), rec=lc_runtime.installed_record())))
''', map_spec=False)
    assert out['r'] is None and out['r0'] is None and out['rec'] == {}
    project = [m for m in out['new'] if m.startswith(('traffic_shepherd', 'wms', 'tools', 'torch', 'numpy', 'traci'))]
    assert project == ['tools.coopmwm.lc_runtime'], out['new']           # stdlib only besides itself


def test_flag_on_the_merge_map_fails_closed(monkeypatch):
    env = importlib.import_module('traffic_shepherd.c4.env')
    cls = env.CavSumoEnv
    monkeypatch.setenv(lc_runtime.FLAG, '1')
    merge_world = dict(seed=986000, cell=0, repeat=0, penetration=.05, demand=1900)
    with pytest.raises(ValueError, match='lane-drop'):
        lc_runtime.install(dict(world=merge_world, role='train'))
    with pytest.raises(ValueError):                              # map world in a merge-map process
        lc_runtime.install(dict(world=dict(merge_world, map='wz3to2'), role='train'))
    assert env.CavSumoEnv is cls and lc_runtime.installed_record() == {}


def test_flag_on_wz3to2_binds_once_after_the_map_runtime():
    from tools.coopmwm.test_map_runtime_cpu import PRELUDE
    out = run_py(PRELUDE+r'''
from tools.coopmwm import lc_runtime
world = ms.world(989201, 3200, 0.6)
job = dict(world=world, role='train')
try:
    lc_runtime.install(job); early = 'no error'
except ValueError as e:
    early = str(e)
map_runtime.install(job)
mapcls = env.CavSumoEnv
cls = lc_runtime.install(job)
again = lc_runtime.install(job)
e = env.CavSumoEnv(k_slots=192, sumo_binary='sumo-not-started')
print(json.dumps(dict(early=early, same=cls is again, bound=env.CavSumoEnv is cls, parent=cls.__mro__[1] is mapcls,
    k=e.k, lc=[cls.LC_RUNTIME, cls.LC_MAP, cls.LC_CLOSING_LANE], name=[cls.__module__, cls.__qualname__],
    stash=hasattr(e, 'lc_tick'), rec=lc_runtime.installed_record(), env_flag=getattr(env, 'LC_RUNTIME', None))))
''', extra_env={lc_runtime.FLAG: '1'})
    assert 'map_runtime.install' in out['early']
    assert out['same'] and out['bound'] and out['parent'] and out['k'] == 512 and out['stash']
    assert out['lc'] == [lc_runtime.SCHEMA, 'wz3to2', 2] and out['name'] == ['traffic_shepherd.c4.env', 'CavSumoEnv']
    rec = out['rec']
    assert rec['map'] == 'wz3to2' and rec['closing_lane'] == 2 and 'traffic_shepherd.c4.env' in rec['env_class_rebound']
    assert out['env_flag'] == lc_runtime.SCHEMA


def test_frozen_runtime_files_untouched():
    """The subclass lives in repo code: the G5 overlay bytes keep their registered SHA."""
    assert g5_runtime.normalized_sha(g5_runtime.OVERLAY) == g5_runtime.OVERLAY_SHA256
    assert g5_runtime.BASE_SHA256.startswith('59aecd0b')
    src = (REPO/'tools/coopmwm/accel_plant.py').read_text(encoding='utf-8')
    assert src.count('from .lc_runtime import') == 1 and src.index('_map_runtime(job)') < src.index('_lc_runtime(job)')


# ------------------------------------------------------------------ golden (no stash = base env)
NAN = math.nan
GOLDEN = dict(
    no_lanes=[dict(base={'c1': .5, 'c2': -1., 'c3': 0.}), dict(base={'c1': .5}), dict(base={})],
    lanes=[dict(base={'c1': .5, 'c2': 0., 'c3': 0.}, lanes={'c1': 0, 'c2': 1, 'c3': 2}, after='c1'),
           dict(base={'c1': .2, 'c3': 0.}, lanes={'c3': 1}), dict(base={'c2': 0.})],
    lane_actors=[dict(base={'c1': .5, 'c2': 0., 'c3': 0.}, lanes={'c1': 0, 'c2': 1}, lane_actors={'c1'}, after='c1'),
                 dict(base={'c1': .5, 'c2': 0.}, lanes={'c2': 1, 'c3': 1}, lane_actors=set()),
                 dict(base={'c1': 0.})])


def _golden_trace(env_class, which, mapname, script, requests=None):
    mod = env_module(which)
    if mapname == 'merge':
        h = Harness(None, ROSTER, desc=md.ORIGINAL, env_module=mod, env_class=env_class, default_lane_length=1000.)
        states, moved = merge_states(), dict(c1=S('1191554636', 1, 0., 20., pos=100.))
    else:
        h = Harness(None, ROSTER, desc=wz(), env_module=mod, env_class=env_class)
        states, moved = wz_states(), dict(c1=S('up', 2, 2000., 20.))
    trace = []
    for el, tick in enumerate(GOLDEN[script]):
        after = dict(states, **moved) if tick.get('after') else None
        calls = h.tick(el, states, requests, base=tick['base'], lanes=tick.get('lanes'),
                       lane_actors=tick.get('lane_actors'), after=after)
        info = {k: v for k, v in h.last_info.items() if k not in ('lc_lane_vehicles', 'lc_lane_commands')}
        trace.append((calls, json.dumps(info, sort_keys=True, default=str), dict(h.env._lc_original),
                      dict(h.env.lane_command_counts), sorted(h.env.ownership._owned.items())))
        states = after or states
    return trace, dict(h.api.lc_mode), list(h.api.raw)


@pytest.mark.parametrize('which', ['repo', 'g5_overlay'])
@pytest.mark.parametrize('mapname', ['merge', 'wz3to2'])
@pytest.mark.parametrize('script', sorted(GOLDEN))
def test_no_stash_step_is_byte_identical_to_the_base_env(which, mapname, script):
    base = env_module(which).CavSumoEnv
    desc = md.ORIGINAL if mapname == 'merge' else wz()
    lc_class = lc_runtime.lc_env_class(base, desc)
    assert lc_class is not base and issubclass(lc_class, base)
    reference = _golden_trace(base, which, mapname, script)
    candidate = _golden_trace(lc_class, which, mapname, script)
    assert candidate == reference
    assert any(t[0] for t in reference[0])                    # the script really commands something
    if script != 'no_lanes':
        assert any('changeLane' == c[2] for t in reference[0] for c in t[0])
    # an EMPTY LCEXEC request every tick issues the identical TraCI writes (only lc_* info added)
    assert _golden_trace(lc_class, which, mapname, script, requests={}) == reference


@pytest.mark.usefixtures('both_envs')
def test_torch_base_command_keeps_other_entries_exactly():
    import torch
    h = harness()
    h.tick(0, wz_states())
    base = torch.full((1, h.env.k), float('nan'), dtype=torch.float32)
    for vid, a in (('c1', .3), ('c2', -.7), ('c3', .123456789)):
        base[0, h.env.builder.pool.slot_of(vid)] = a
    stash = lc_runtime.validate(dict(accel={'c1': -1.5}, native=['c3']), h.env._lc_live_cavs())
    out = h.env._lc_accel(base, stash)
    ref = np.asarray(base.numpy(), dtype=np.float64).reshape(-1)
    slot = h.env.builder.pool.slot_of
    assert out.shape == (1, h.env.k) and out[0, slot('c1')] == -1.5 and math.isnan(out[0, slot('c3')])
    keep = [i for i in range(h.env.k) if i not in (slot('c1'), slot('c3'))]
    assert np.array_equal(out[0, keep], ref[keep], equal_nan=True)
    assert h.env._lc_accel(base, lc_runtime.empty_request()) is base       # nothing to replace: same object


# ------------------------------------------------------------------ knobs
def probe_target(v, a, allowed=LIMIT):
    return min(max(0.0, v + a * 1.0), allowed)


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('v,a', [(20., 1.0), (20., -1.5), (1., -1.5), (30.8, 1.0), (0., -1.5), (31.29, 1.0)])
def test_exact_accel_is_the_probe_slowdown(v, a):
    h = harness()
    calls = h.tick(0, wz_states(c1=S('up', 1, 2000., v)), dict(accel={'c1': a}))
    assert [c[:3] for c in calls] == [(0, 'c1', 'slowDown')]
    assert abs(calls[0][3] - probe_target(v, a)) <= 1e-9
    assert h.api.raw[-1][2][2] == 1.0                         # slowDown duration 1 s
    assert h.last_info['actions']['c1'] == 'direct' and h.last_info['lc_lane_vehicles'] == {}


@pytest.mark.usefixtures('both_envs')
def test_accel_overrides_base_command_and_native_releases_owned_member():
    h = harness()
    base = {'c1': .3, 'c2': .4}
    calls = h.tick(0, wz_states(), dict(accel={'c1': -1.5}), base=base)
    assert sorted((c[1], c[2], round(c[3], 9)) for c in calls) == [('c1', 'slowDown', 18.5), ('c2', 'slowDown', 15.4)]
    calls = h.tick(1, wz_states(), dict(native=['c1']), base=base)
    assert ('c1', 'setSpeed', -1.0) in [c[1:] for c in calls] and not [c for c in calls if c[1] == 'c1' and c[2] == 'slowDown']
    assert [c for c in calls if c[1] == 'c2'] == [(1, 'c2', 'slowDown', 15.4)]   # base command kept
    calls = h.tick(2, wz_states(), dict(native=['c1']), base=base)
    assert [c for c in calls if c[1] == 'c1'] == []           # native, not owned: no TraCI write
    assert h.last_info['actions']['c1'] == 'native'


@pytest.mark.usefixtures('both_envs')
def test_native_member_never_goes_to_a_damping_fallback():
    h = harness()
    h.tick(0, wz_states(), dict(accel={'c1': 1.0}), base={'c2': 0., 'c3': 0.})
    # every other CAV has a base command; the harness damping stub raises if c1 were handed to it
    calls = h.tick(1, wz_states(), dict(native=['c1']), base={'c2': 0., 'c3': 0.}, fallback='damping')
    assert (1, 'c1', 'setSpeed', -1.0) in calls
    with pytest.raises(RuntimeError, match='damping'):
        h.tick(2, wz_states(), None, base={'c2': 0.}, fallback='damping')


@pytest.mark.usefixtures('both_envs')
def test_lane_modes_513_512_switch_and_restore_without_a_command():
    h = harness(settings={'c1': dict(lc_mode=1557), 'c2': dict(lc_mode=1621)})
    st = wz_states()
    calls = h.tick(0, st, dict(lane_mode={'c1': 513, 'c2': 512}))
    assert sorted(c[1:] for c in calls) == [('c1', 'setLaneChangeMode', 513), ('c2', 'setLaneChangeMode', 512)]
    assert h.env._lc_original == {'c1': 1557, 'c2': 1621}
    assert h.tick(1, st, dict(lane_mode={'c1': 513, 'c2': 512})) == []          # set once
    calls = h.tick(2, st, dict(lane_mode={'c1': 512, 'c2': 512}))
    assert calls == [(2, 'c1', 'setLaneChangeMode', 512)]                      # value switch
    calls = h.tick(3, st, dict(lane_mode={'c2': 512}))
    assert calls == [(3, 'c1', 'setLaneChangeMode', 1557)]                     # c1 left: original
    calls = h.tick(4, st)                                                      # no stash: restore rest
    assert calls == [(4, 'c2', 'setLaneChangeMode', 1621)]
    assert h.env._lc_original == {} and not h.env._lc_outstanding()
    assert h.tick(5, st) == [] and 'lc_lane_vehicles' not in h.last_info       # frozen path again
    assert h.api.lc_mode['h1'] == 1621 and all(c[1] != 'h1' for c in h.api.calls)


@pytest.mark.usefixtures('both_envs')
def test_lane_only_actor_is_policy_lane_control_and_gets_its_target():
    h = harness()
    calls = h.tick(0, wz_states(), dict(native=['c1'], lane_mode={'c1': 513}, lanes={'c1': [0, 2.0]}))
    assert sorted(c[1:] for c in calls) == [('c1', 'changeLane', (2, 2.0)), ('c1', 'setLaneChangeMode', 513)]
    assert h.last_info['lc_lane_vehicles'] == {'c1': dict(target=0, status='issued', completed=False)}
    assert h.last_info['lane_commands'] == {}                  # frozen G5 counters untouched
    # without a lane mode (not policy lane control) the request is refused, never issued
    h2 = harness()
    calls = h2.tick(0, wz_states(), dict(lanes={'c1': [0, 2.0]}))
    assert calls == [] and h2.last_info['lc_lane_vehicles']['c1']['status'] == 'refused_not_policy'


@pytest.mark.usefixtures('both_envs')
def test_closing_lane_target_issued_deferred_and_refused():
    h = harness()
    req = dict(lane_mode={'c1': 513}, lanes={'c1': [2, 2.0]})
    calls = h.tick(0, wz_states(), req, after=wz_states(c1=S('up', 0, 2000., 20.)))
    assert calls == [(0, 'c1', 'setLaneChangeMode', 513), (0, 'c1', 'changeLane', (0, 2.0))]
    assert h.last_info['lc_lane_vehicles']['c1'] == dict(target=2, status='issued', completed=True)
    assert h.last_info['lc_lane_commands'] == dict(issued=1, completed=1)
    # near the closing-lane end: 2696 - 2680 = 16 m <= v*d + 10 = 30 m -> deferred
    h = harness()
    calls = h.tick(0, wz_states(c1=S('up', 1, 2680., 10.)), req)
    assert [c[2] for c in calls] == ['setLaneChangeMode']
    assert h.last_info['lc_lane_vehicles']['c1']['status'] == 'deferred_edge_end'
    # 25 m left, slower car (v*d + 10 = 20 m): issued
    h = harness()
    calls = h.tick(0, wz_states(c1=S('up', 1, 2671., 5.)), req)
    assert (0, 'c1', 'changeLane', (0, 2.0)) in calls
    # downstream edge 'wz' has no closing lane
    h = harness()
    calls = h.tick(0, wz_states(c1=S('wz', 0, 2800., 20.)), req)
    assert h.last_info['lc_lane_vehicles']['c1']['status'] == 'refused_no_lane' and 'changeLane' not in [c[2] for c in calls]
    # merge map: WM 2 is the ramp lane -> frozen refusal
    h = Harness(None, ROSTER, desc=md.ORIGINAL, env_module=current_env(), default_lane_length=1000.)
    calls = h.tick(0, merge_states(c1=S('1487606179', 1, 0., 20., pos=100.)), req)
    assert h.last_info['lc_lane_vehicles']['c1']['status'] == 'refused_not_allowed'
    assert 'changeLane' not in [c[2] for c in calls]


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('state,target,status', [
    (S('up', 2, 2000., 20.), 2, 'refused_not_allowed'),      # WM 0 -> WM 2: not adjacent
    (S('up', 0, 2000., 20.), 0, 'refused_not_allowed'),      # WM 2 -> WM 0
    (S(':n1_0', 0, 2700., 20.), 0, 'refused_junction'),
    (S('up', 1, 2000., 20.), 3, 'refused_not_allowed')])
def test_lane_rules(state, target, status):
    h = harness()
    calls = h.tick(0, wz_states(c1=state), dict(lane_mode={'c1': 513}, lanes={'c1': [target, 2.0]}))
    assert h.last_info['lc_lane_vehicles']['c1']['status'] == status
    assert 'changeLane' not in [c[2] for c in calls]


@pytest.mark.usefixtures('both_envs')
def test_lcexec_current_lane_request_is_a_held_change_lane_g5_keeps_noop():
    h = harness()
    st = wz_states()
    g5 = dict(base={'c3': 0.}, lanes={'c3': 0}, lane_actors={'c3'})          # c3 already in WM 0
    req = dict(lane_mode={'c1': 513, 'c2': 512}, lanes={'c1': [1, 2.0], 'c2': [2, 1.5]})
    calls = h.tick(0, st, req, **g5)
    assert (0, 'c1', 'changeLane', (1, 2.0)) in calls and (0, 'c2', 'changeLane', (0, 1.5)) in calls
    assert not [c for c in calls if c[1] == 'c3' and c[2] == 'changeLane']   # frozen G5: noop
    assert h.last_info['lane_vehicles'] == {'c3': dict(target=0, status='noop', completed=False)}
    assert h.last_info['lc_lane_vehicles'] == {'c1': dict(target=1, status='held', completed=True),
                                               'c2': dict(target=2, status='held', completed=True)}
    assert h.last_info['lc_lane_commands'] == dict(held=2)
    # pushed out during the step: the hold is reported not kept
    h.tick(1, st, req, after=wz_states(c1=S('up', 2, 2000., 20.)))
    assert h.last_info['lc_lane_vehicles']['c1'] == dict(target=1, status='held', completed=False)
    # hold on the downstream edge: current SUMO index of that edge
    h = harness()
    calls = h.tick(0, wz_states(c1=S('wz', 0, 2800., 20.)), dict(lane_mode={'c1': 513}, lanes={'c1': [1, 2.0]}))
    assert (0, 'c1', 'changeLane', (0, 2.0)) in calls
    # closing-lane hold near the drop: deferred; junction: refused
    for state, status in ((S('up', 0, 2680., 10.), 'deferred_edge_end'),
                          (S(':n1_0', 0, 2700., 10.), 'refused_junction')):
        h = harness()
        calls = h.tick(0, wz_states(c2=state), dict(lane_mode={'c2': 513}, lanes={'c2': [2, 2.0]}))
        assert h.last_info['lc_lane_vehicles']['c2']['status'] == status
        assert 'changeLane' not in [c[2] for c in calls]
    # merge map: holding a CAV in the ramp lane is refused; a main-lane hold is issued
    h = Harness(None, ROSTER, desc=md.ORIGINAL, env_module=current_env(), default_lane_length=1000.)
    calls = h.tick(0, merge_states(), dict(lane_mode={'c1': 513, 'c2': 513},
                                           lanes={'c1': [1, 2.0], 'c2': [2, 2.0]}))
    assert h.last_info['lc_lane_vehicles']['c2']['status'] == 'refused_not_allowed'
    assert h.last_info['lc_lane_vehicles']['c1']['status'] == 'held' and (0, 'c1', 'changeLane', (0, 2.0)) in calls


@pytest.mark.usefixtures('both_envs')
def test_every_requested_vehicle_gets_a_receipt_status():
    h = harness(roster=dict(ROSTER, c4=True, c5=True))
    st = wz_states(c2=S('up', 0, 2680., 10.), c4=S('up', 2, 1500., 20.), c5=S('up', 1, 1200., 20.))
    req = dict(lane_mode={'c1': 513, 'c2': 512, 'c4': 513}, native=['c5'],
               lanes={'c1': [0, 2.0], 'c2': [1, 2.0], 'c4': [2, 2.0], 'c5': [0, 2.0]})
    h.tick(0, st, req)
    assert {v: r['status'] for v, r in h.last_info['lc_lane_vehicles'].items()} == dict(
        c1='issued', c2='deferred_edge_end', c4='refused_not_allowed', c5='refused_not_policy')
    assert h.last_info['lc_lane_commands'] == dict(issued=1, not_completed=1, deferred_edge_end=1,
                                                   refused_not_allowed=1, refused_not_policy=1)
    assert h.env.lc_lane_command_counts['deferred_edge_end'] == 1


@pytest.mark.usefixtures('both_envs')
def test_known_difference_very_fast_vehicle_is_released_to_native():
    """v > allowed + 3: direct_target is infeasible -> native (the probe would slowDown to allowed)."""
    h = harness()
    assert h.tick(0, wz_states(), dict(accel={'c1': -1.0}))[0][2] == 'slowDown'
    calls = h.tick(1, wz_states(c1=S('up', 1, 2020., 25., allowed=20.)), dict(accel={'c1': -1.0}))
    assert calls == [(1, 'c1', 'setSpeed', -1.0)] and h.last_info['actions']['c1'] == 'native'
    assert probe_target(25., -1.0, 20.) == 20.0


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('duration', [2.0, 1.0, 3.5])
def test_lane_duration_passes_through(duration):
    h = harness()
    calls = h.tick(0, wz_states(), dict(lane_mode={'c2': 513}, lanes={'c2': [1, duration]}))
    assert (0, 'c2', 'changeLane', (1, duration)) in calls


@pytest.mark.usefixtures('both_envs')
def test_tau_set_change_restore_and_left_vehicle():
    h = harness(settings={'c1': dict(tau=1.3), 'c2': dict(tau=1.1)})
    st = wz_states()
    assert h.tick(0, st, dict(tau={'c1': .6})) == [(0, 'c1', 'setTau', .6)]
    assert h.env._lc_tau_saved == {'c1': 1.3}
    assert h.tick(1, st, dict(tau={'c1': .6})) == []
    assert h.tick(2, st, dict(tau={'c1': 2.0})) == [(2, 'c1', 'setTau', 2.0)]
    assert h.tick(3, st, {}) == [(3, 'c1', 'setTau', 1.3)]
    assert h.tick(4, st, {}) == [] and h.env._lc_tau_saved == {}
    assert h.tick(5, st, dict(tau={'c2': .8})) == [(5, 'c2', 'setTau', .8)]
    gone = {v: s for v, s in st.items() if v != 'c2'}
    assert h.tick(6, gone, {}) == [] and h.env._lc_tau_saved == {}             # left: no call
    assert h.tick(7, gone) == [] and 'lc_lane_vehicles' not in h.last_info


@pytest.mark.parametrize('request_', [
    dict(accel={'h1': 1.0}), dict(native=['h1']), dict(lanes={'h1': [0, 2.0]}), dict(lane_mode={'h1': 513}),
    dict(tau={'h1': .6}), dict(accel={'ghost': 1.0}), dict(accel={'c1': NAN}), dict(accel={'c1': True}),
    dict(lane_mode={'c1': 514}), dict(tau={'c1': .7}), dict(tau={'c1': math.inf}), dict(lanes={'c1': [0, 0.]}),
    dict(lanes={'c1': [1.0, 2.0]}), dict(lanes={'c1': [-1, 2.0]}), dict(lanes={'c1': [0]}),
    dict(accel={'c1': 1.}, native=['c1']), dict(native=['c1', 'c1']), dict(speed={'c1': 1.})])
def test_invalid_requests_raise_and_touch_nothing(request_):
    h = harness()
    with pytest.raises(ValueError):
        h.tick(0, wz_states(), request_)
    assert h.api.calls == [] and h.env.__dict__.get('_lc_stash') is None


@pytest.mark.usefixtures('both_envs')
def test_stash_is_single_use_and_needs_direct_mode():
    h = harness()
    st = wz_states()
    assert h.tick(0, st) == []
    h.env.lc_tick(dict(accel={'c1': 1.}))
    with pytest.raises(ValueError, match='one lc_tick'):
        h.env.lc_tick(dict(accel={'c1': 1.}))
    h.tick(1, st)                                              # consumes the stash
    assert h.api.calls[-1][1:3] == ('c1', 'slowDown') and h.env.__dict__.get('_lc_stash') is None
    calls = h.tick(2, st)                                      # next step: no stash -> native release
    assert calls == [(2, 'c1', 'setSpeed', -1.0)] and 'lc_lane_vehicles' not in h.last_info
    h.env.mode = 'native'
    with pytest.raises(ValueError, match='direct'):
        h.env.lc_tick({})


@pytest.mark.usefixtures('both_envs')
def test_g5_lanes_tensor_targets_unchanged_alongside_lcexec_requests():
    st = wz_states()
    g5 = dict(base={'c1': .5, 'c3': 0.}, lanes={'c1': 0}, lane_actors={'c1'})
    ref = Harness(None, ROSTER, desc=wz(), env_module=current_env(), env_class=current_env().CavSumoEnv)
    ref_calls = ref.tick(0, st, **g5)
    h = harness()
    calls = h.tick(0, st, dict(native=['c2'], lane_mode={'c2': 512}, lanes={'c2': [1, 2.0]}), **g5)
    assert [c for c in calls if c[1] == 'c1'] == [c for c in ref_calls if c[1] == 'c1']
    assert (0, 'c1', 'changeLane', (2, 1.0)) in calls                          # frozen G5: LANE_REQUEST_S
    assert (0, 'c2', 'changeLane', (1, 2.0)) in calls and (0, 'c2', 'setLaneChangeMode', 512) in calls
    for key in ('lane_commands', 'lane_vehicles', 'actions'):
        if key == 'actions':
            assert {k: v for k, v in h.last_info[key].items() if k != 'c2'} == \
                   {k: v for k, v in ref.last_info[key].items() if k != 'c2'}
        else:
            assert h.last_info[key] == ref.last_info[key]
    assert h.last_info['lc_lane_vehicles'] == {'c2': dict(target=1, status='issued', completed=False)}
    # one lateral writer per vehicle: an LCEXEC lane request for a G5 lane target raises
    h = harness()
    with pytest.raises(ValueError, match='both LCEXEC'):
        h.tick(0, st, dict(lane_mode={'c1': 513}, lanes={'c1': [0, 2.0]}), **g5)


def test_requests_from_commands_follows_the_executor_contract():
    frame = lp.frame_from_descriptor(wz())
    hold = lp.MemberExecutor('c1', lc_roles.member_program('hold', dict(v_w=22., amax=1., dmax=1., hold_s=2), frame))
    tau = lp.MemberExecutor('c2', lc_roles.member_program('tau', dict(tau=.6, lc_mode=512), frame))
    pulse = lp.MemberExecutor('c3', lc_roles.member_program('pulse', dict(a=-1., dur_s=1), frame))
    ex = dict(c1=hold, c2=tau, c3=pulse)
    own = dict(c1=lp.View('c1', -600., 1, 20., True, True), c2=lp.View('c2', -500., 2, 15., True, True),
               c3=lp.View('c3', -800., 0, 22., True, True))
    cmds = {v: e.step(lp.Obs(0, own[v])) for v, e in ex.items()}
    req = requests_from_commands(cmds, ex)
    assert req == dict(accel={'c1': 1.0, 'c3': -1.0}, native=('c2',), lanes={}, lane_mode={'c1': 513, 'c2': 512},
                       tau={'c2': .6})
    cmds = {v: e.step(lp.Obs(1, own[v])) for v, e in ex.items()}
    req = requests_from_commands(cmds, ex)
    assert pulse.released and req['accel'] == {'c1': 1.0} and 'c3' not in req['lane_mode']


# ------------------------------------------------------------------ probe parity
@functools.lru_cache(maxsize=None)
def probe(name):
    """Class Arm, _positions, xglob and the constants of a sha-pinned verbatim probe core."""
    meta = json.loads((FIX/'MANIFEST.json').read_text(encoding='utf-8'))['files'][name]
    raw = (FIX/name).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(raw).hexdigest() == meta['lf_sha256']
    text = raw.decode()
    tree = ast.parse(text)
    keep = [n for n in tree.body
            if (isinstance(n, ast.ClassDef) and n.name == 'Arm')
            or (isinstance(n, ast.FunctionDef) and n.name in ('_positions', 'xglob'))
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
                'X_B', 'LC_MODE_POLICY', 'LC_MODE_TRACI_ONLY', 'EDGE_OFFSET') for t in n.targets))]
    arm = next(n for n in keep if isinstance(n, ast.ClassDef))
    assert [arm.lineno, arm.end_lineno] == meta['arm_lines']
    segment = '\n'.join(text.splitlines()[arm.lineno-1:arm.end_lineno])+'\n'
    assert hashlib.sha256(segment.encode()).hexdigest() == meta['arm_lf_sha256']
    ns = {'__name__': f'lcexec_rt_{name}'}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(FIX/name), 'exec'), ns)
    return SimpleNamespace(**ns)


def persistent_hold(ex, cmds, own):
    """LCMOVES persistent lane request (lcm_core ``_prog``): changeLane(target, d) EVERY second while
    on the approach before the give-up point, also after reaching the target (hold) or after being
    pushed out. lc_program's until_reached rule stops requesting after lane_done (LCEXEC-owned
    executor, reported); this adds the probe's request only where the executor emitted none."""
    q = ex.p.lane
    if (ex.released or q is None or q.rule != 'until_reached' or ex.lane_done is None or own is None
            or any(k == 'changeLane' for k, _ in cmds)):
        return cmds
    if own.on_approach and own.s_m < (q.giveup_s if q.giveup_s is not None else math.inf):
        return list(cmds)+[('changeLane', (int(q.target), float(q.duration_s)))]
    return cmds


def run_parity(core, members, seq, settings=None, persistent=False):
    roster = {}
    for st in seq:
        roster.update({v: s['is_cav'] for v, s in st.items()})
    h = Harness(None, roster, desc=wz(), env_module=current_env(), settings=settings)
    api = FakeTraci(roster, lane_lengths=h.api.lane_lengths, settings=settings)
    p = probe(core)
    arm = p.Arm(api, members)
    execs = {m['id']: lp.MemberExecutor(m['id'], lc_roles.member_program(m['role'], m['params'], h.frame))
             for m in members}
    probe_calls, env_calls = [], []
    for el, st in enumerate(seq):
        api.load(el, h.states(st))
        live = set(st)
        n = len(api.calls)
        arm.control(el, live, p._positions(api, sorted(roster), live))
        probe_calls += api.calls[n:]
        cmds = {v: ex.step(h.obs(el, v, st)) for v, ex in execs.items()}
        if persistent:
            cmds = {v: persistent_hold(execs[v], c, h.obs(el, v, st).own) for v, c in cmds.items()}
        env_calls += h.tick(el, st, requests_from_commands(cmds, execs))
    return probe_calls, env_calls


def effective(calls):
    """Drop the SUMO no-op ``setSpeed(vid, -1)``: no slowDown to that vid in the previous control
    second (an older slowDown(target, 1 s) speed timeline has expired; the probe's release issues
    setSpeed(-1) also for members it never slowed, the env only for members its owner held)."""
    slowed = {(el, vid) for el, vid, kind, _ in calls if kind == 'slowDown'}
    out = [c for c in calls if not (c[2] == 'setSpeed' and c[3] == -1 and (c[0]-1, c[1]) not in slowed)]
    return sorted(out, key=lambda c: (c[0], c[1], c[2]))


def assert_parity(probe_calls, env_calls):
    a, b = effective(probe_calls), effective(env_calls)
    assert [c[:3] for c in a] == [c[:3] for c in b], (a, b)
    for x, y in zip(a, b):
        if isinstance(x[3], tuple):
            assert x[3][0] == y[3][0] and abs(x[3][1]-y[3][1]) <= 1e-9, (x, y)
        else:
            assert abs(float(x[3])-float(y[3])) <= 1e-9, (x, y)
    return a


CORES = ['lcm_core.pysrc', 'pg2_core.pysrc']
HDV = S('up', 1, 1900., 25., cav=False)


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('core', CORES)
@pytest.mark.parametrize('v_w,vs', [(22., [18., 19., 21.5, 22.3, 22.3]), (15., [25., 24., 23.4, 22., 21.])])
def test_parity_hold_law_lcmode_and_release(core, v_w, vs):
    xs = [2000., 2018.5, 2038., 2060., 2082.]
    seq = [{'c1': S('up', 1, x, v), 'h1': HDV} for x, v in zip(xs, vs)]
    members = [dict(id='c1', role='hold', params=dict(v_w=v_w, amax=1., dmax=1., hold_s=3))]
    calls = assert_parity(*run_parity(core, members, seq))
    kinds = [(c[0], c[2]) for c in calls]
    assert kinds[:2] == [(0, 'setLaneChangeMode'), (0, 'slowDown')] and (3, 'setSpeed') in kinds
    assert [c[3] for c in calls if c[2] == 'setLaneChangeMode'] == [513, 1621]


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('core', CORES)
def test_parity_pulse(core):
    seq = [{'c1': S('up', 2, 1500.+20*i, 20.-i), 'h1': HDV} for i in range(4)]
    members = [dict(id='c1', role='pulse', params=dict(a=-1.0, dur_s=2))]
    calls = assert_parity(*run_parity(core, members, seq))
    assert [(c[0], c[2], c[3]) for c in calls] == [(0, 'slowDown', 19.), (1, 'slowDown', 18.), (2, 'setSpeed', -1.)]


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('core', CORES)
@pytest.mark.parametrize('role,params,lanes,expect', [
    ('lane_to', dict(target_lane=2, giveup_x=2500., hold_s=4), [1, 1, 2, 2, 2, 2], (2, 2.0)),
    ('merge_early', dict(trigger_x=2030., target_lane=1, hold_s=4), [0, 0, 0, 1, 1, 1], (1, 2.0)),
])
def test_parity_change_lane_and_release(core, role, params, lanes, expect):
    seq = [{'c1': S('up', lane, 2000.+20*i, 20.), 'h1': HDV} for i, lane in enumerate(lanes)]
    calls = assert_parity(*run_parity(core, [dict(id='c1', role=role, params=params)], seq))
    assert ('changeLane', expect) in [(c[2], c[3]) for c in calls]
    assert [c[3] for c in calls if c[2] == 'setLaneChangeMode'] == [513, 1621]


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('core', CORES)
def test_parity_zip_merge_lcmode_512(core):
    xs, lanes = [2300., 2360., 2380., 2400.], [0, 0, 1, 1]
    seq = [{'c1': S('up', lane, x, 10.), 'h1': HDV} for x, lane in zip(xs, lanes)]
    members = [dict(id='c1', role='zip_merge', params=dict(merge_x=2350., fallback_x=2600.))]
    calls = assert_parity(*run_parity(core, members, seq, settings={'c1': dict(lc_mode=1557)}))
    assert [(c[0], c[2], c[3]) for c in calls] == [(0, 'setLaneChangeMode', 512), (1, 'changeLane', (1, 2.0)),
                                                   (2, 'setLaneChangeMode', 1557)]


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('core', CORES)
@pytest.mark.parametrize('tau,mode', [(.6, 513), (2.0, 512), (.5, 513)])
def test_parity_tau_zone_and_restore(core, tau, mode):
    xs = [1950., 2010., 2050., 2110., 2130.]
    seq = [{'c1': S('up', 1, x, 20.), 'h1': HDV} for x in xs]
    members = [dict(id='c1', role='tau', params=dict(tau=tau, lc_mode=mode, zone=[2000., 2100.], hold_s=30))]
    calls = assert_parity(*run_parity(core, members, seq, settings={'c1': dict(tau=1.2)}))
    assert [(c[0], c[2], c[3]) for c in calls] == [(1, 'setLaneChangeMode', mode), (1, 'setTau', tau),
                                                   (3, 'setLaneChangeMode', 1621), (3, 'setTau', 1.2)]


@pytest.mark.usefixtures('both_envs')
@pytest.mark.parametrize('core', CORES)
def test_parity_zip_yield_track_and_partner_release(core):
    seq = []
    for i, (xp, lp_, xm, vm) in enumerate([(2200., 0, 2150., 20.), (2215., 0, 2170., 20.7), (2232., 0, 2191., 21.2),
                                           (2250., 1, 2213., 21.6), (2268., 1, 2235., 21.6)]):
        seq.append({'p1': S('up', lp_, xp, 18.), 'c1': S('up', 1, xm, vm), 'h1': HDV})
    members = [dict(id='p1', role='zip_merge', params=dict(merge_x=2240., fallback_x=2600.)),
               dict(id='c1', role='zip_yield', params=dict(partner='p1', gap=25., kp=.15, kv=.6, dmax=1.5, amax=1.))]
    calls = assert_parity(*run_parity(core, members, seq))
    assert sum(c[2] == 'slowDown' for c in calls) == 3 and (3, 'c1', 'setSpeed', -1.0) in calls


@pytest.mark.usefixtures('both_envs')
def test_parity_prog_closing_lane_target_first_second():
    """LCMOVES prog: lane request INTO the closing lane (SUMO 0 = WM 2) + hold speed law."""
    seq = [{'c1': S('up', 1, 2000., 18.), 'h1': HDV}]
    members = [dict(id='c1', role='prog', params=dict(lc_target=0, speed='hold', v_w=15., dmax=1.))]
    calls = assert_parity(*run_parity('lcm_core.pysrc', members, seq))
    assert [(c[2], c[3]) for c in calls] == [('changeLane', (0, 2.0)), ('setLaneChangeMode', 513), ('slowDown', 17.0)]


@pytest.mark.usefixtures('both_envs')
@pytest.mark.usefixtures('both_envs')
def test_parity_prog_persistent_lane_hold_into_the_closing_lane():
    """LCMOVES prog into the closing lane: change (el 0), HOLD seconds (el 1, 2, 4: changeLane to the
    lane it is in), re-request after being pushed out (el 3); speed hold law throughout."""
    lanes = [1, 0, 0, 1, 0]
    seq = [{'c1': S('up', lane, 2000.+16*i, 18.-i), 'h1': HDV} for i, lane in enumerate(lanes)]
    members = [dict(id='c1', role='prog', params=dict(lc_target=0, speed='hold', v_w=15., dmax=1.))]
    probe_calls, env_calls = run_parity('lcm_core.pysrc', members, seq, persistent=True)
    calls = assert_parity(probe_calls, env_calls)
    assert [c[0] for c in calls if c[2] == 'changeLane'] == [0, 1, 2, 3, 4]
    assert all(c[3] == (0, 2.0) for c in calls if c[2] == 'changeLane')
    assert sum(c[2] == 'slowDown' for c in calls) == 5


def test_harness_rejects_a_frame_from_another_descriptor():
    other = lp.Frame('x', 'up', 0., 75.44, {}, {0: 2, 1: 1, 2: 0}, 2, 1, 0)
    with pytest.raises(ValueError, match='frame'):
        Harness(other, ROSTER, desc=wz())
    assert np.isfinite(lp.frame_from_descriptor(wz()).anchor_x_m)
