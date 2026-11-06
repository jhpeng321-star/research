"""LCEXEC free_program runtime (lc_families): CPU contracts (no simulator, GPU or network).

Real EpisodePolicy / TriggerPolicy / MoveExecutor / feasible slates / v2v_runtime controller /
lc_program executors / lc_runtime env subclass (lc_writer_harness on a fake TraCI). Fakes: a
test-only trigger scoring head and FULLPOL-style template generators (fixtures below).
Geometry: the REAL N1 wz3to2 descriptor (WM lanes 0/1 through, 2 = closing lane ending at
s = 75.44 m).
"""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from . import lc_families as lf, lc_program as lp, map_descriptor as md, track2_backend as tb
from .acceleration import Controller
from .merge_features import ObservedVehicle, SensorReport
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)
from .v2v_intent import IntentPolicy
from .v2v_runtime import controller_class

PROV = 'LCEXEC-FAM CPU fixture (2026-11-06 free_program runtime)'
REPO = Path(__file__).resolve().parents[2]
MAP_DIR = REPO/'assets/research/newmap_candidates/wz3to2'
BASE_COMMIT = 'aafdb8d2'
HERE = __name__                     # generator fixtures resolve to THIS loaded module
DROP = 75.44


# ------------------------------------------------------------------ FULLPOL-style generator fixtures

def _front_of_lane(state, own, lane):
    if int(own['lane']) != lane or not state['frame'].on_approach(own['s_m'], own['lane']):
        return False
    return not any(r['is_cav'] and r['vid'] != own['vid'] and int(r['lane']) == lane and r['s_m'] > own['s_m']
                   for r in state['sensed'])


HOLD = lp.MemberProgram(lon=lp.Lon('speed', v_w=18., amax=1., dmax=1.), hold_s=8, releases=('hold_end',),
                        role='hold')
FILL = lp.MemberProgram(lon=lp.Lon('track', v_w=18., amax=1., dmax=1., refs=('@0', '@1'), off=-25., kp=.15,
                                   kv=.6, settle_gap=25.),
                        lane=lp.LaneReq(2, 'hold', giveup_s=DROP-300.), hold_s=8, tau=2.0, mode_on='on_action',
                        releases=('hold_end',), role='fill')
WALL_FILL = lp.FreeProgram(('@0', '@1', '@2'), (HOLD, HOLD, FILL), 'wz3to2', (0, 1, 1), 10,
                           dict(fixture='wall_fill'),
                           (None, dict(lane=0, ds=(-25., 25.)), dict(lane=1, ds=(-75., -15.))))


def wall_fill(state, own, rng):
    """wall_fill-like: A (lane 1, front) + B (lane 0 abreast) hold v_w; filler F (lane 1 behind)
    changes into the closing lane, tracks the A/B slot (off -25 m), then holds v_w."""
    return [WALL_FILL] if _front_of_lane(state, own, 1) else []


CHAIN = lp.FreeProgram(tuple(f'@{k}' for k in range(6)), (HOLD,)*6, 'wz3to2', (0,)*6, 20, dict(fixture='chain6'),
                       (None,)+tuple(dict(lane=1, ds=(-40.*k-20., -40.*k+20.)) for k in range(1, 6)))


def chain6(state, own, rng):
    return [CHAIN] if _front_of_lane(state, own, 1) else []


def empty(state, own, rng):
    return []


def nothing(state, own, rng):
    return None


_FLAKY = {'calls': 0}


def flaky(state, own, rng):
    """A template at the pool draw, none at the slate redraw (N11 F8 guard)."""
    _FLAKY['calls'] += 1
    return [WALL_FILL] if _FLAKY['calls'] % 2 == 1 and _front_of_lane(state, own, 1) else []


def unregistered(state, own, rng):
    bad = lp.MemberProgram(lon=lp.Lon('speed', v_w=40., amax=1., dmax=1.), hold_s=8, releases=('hold_end',))
    return [lp.FreeProgram(('@0',), (bad,), 'wz3to2', (0,), 10, {}, (None,))]


# ------------------------------------------------------------------ helpers

@pytest.fixture
def wz(monkeypatch):
    monkeypatch.delenv(md.ENV, raising=False)
    md._reset_for_tests()
    desc = md.bind(md.load(MAP_DIR))
    yield desc
    md._reset_for_tests()


def lc_cfg(cfg, *, cap=6, generator='wall_fill', range_m=None, **extra):
    out = dict(cfg, trigger_partner_slots=4, trigger_candidates=4, group_lane_moves=True)
    out.update(lf.config_keys(provenance=PROV, cap=cap, range_m=range_m,
                              generator=None if generator is None else f'{HERE}:{generator}'))
    out.update(extra)
    return out


def make(cfg, arm='B'):
    torch.manual_seed(7)
    return tb.EpisodePolicy(IntentPolicy(), cfg, arm)


class Prefer(nn.Module):
    """Test-only trigger head: initiate > join > none from the lc_families descriptor columns."""

    def __init__(self, trigger, initiate=50., join=40.):
        super().__init__()
        hidden = trigger.base_policy.actor[-1].in_features
        base = len(trigger.families)+trigger.partner_slots+3
        ext = trigger.extension
        parts = getattr(ext, 'parts', (ext,))
        offset = hidden+base
        for part in parts:
            if isinstance(part, lf.TriggerExtension):
                break
            offset += part.descriptor_width
        self.cols = (offset+lf.DESCRIPTOR.index('is_program_initiate'), offset+lf.DESCRIPTOR.index('is_program_join'))
        self.w = (initiate, join)

    def forward(self, x):
        return (self.w[0]*x[..., self.cols[0]]+self.w[1]*x[..., self.cols[1]])[..., None]


def controller(p):
    return controller_class(Controller)(p, torch.Generator().manual_seed(9), True)


def row(vid, s, lane, cav=True, v=20., **extra):
    return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=lane, is_cav=cav, length_m=5., **extra)


class Capture:
    """env stand-in for Controller.lane_request: records the lc_tick stash."""

    def __init__(self, ids=()):
        self.ticks, self.ids = [], list(ids)

    def lc_tick(self, request):
        self.ticks.append(json.loads(json.dumps(request)))

    def slot_vehicle_ids(self):
        return list(self.ids)


def step(p, c, rows, now, env=None):
    p.observe(rows, now)
    requests = c.apply(rows, {r['vid']: 0. for r in rows if r['is_cav']}, now)
    if env is not None:
        c.lane_request(env)
    return c.events[-1]['sample'], requests


def offered(sample, ego):
    trig = sample['track2_trigger']
    i = trig['context']['members'].index(ego)
    return [m['family'] for m in trig['tables'][i] if m]


def canon(value):
    return json.dumps(value, sort_keys=True, default=repr)


def report(vid, s, lane, v, t):
    return SensorReport(float(t), ObservedVehicle(vid, float(s), int(lane), float(v), 5.), (0, 1, 2), ())


# ------------------------------------------------------------------ settings / registration

def test_settings_default_off_and_registration(cfg, wz):
    assert lf.settings(cfg) is None and lf.extension(cfg) is None
    assert lf.settings(dict(cfg, lc_families=False)) is None
    on = lc_cfg(cfg, cap=4, range_m=150.)
    got = lf.settings(on)
    assert (got['cap'], got['range_m'], got['map'], got['propose']) == (4, 150., 'wz3to2', wall_fill)
    assert got['frame'] == lp.frame_from_descriptor(wz) and got['frame'].closing_lane == 2
    assert lf.settings(lc_cfg(cfg, generator=None))['propose'] is lf.default_generator
    assert lf.settings(lc_cfg(cfg))['range_m'] == 100.
    raw = on['lc_families']
    bad = [dict(raw, rule='x'), dict(raw, provenance=' '), dict(raw, enabled=1), dict(raw, extra=1),
           dict(raw, cap=1), dict(raw, cap=7), dict(raw, cap=True), dict(raw, cap=3.), dict(raw, map='merge'),
           dict(raw, range_m=float('nan')), dict(raw, range_m=0), dict(raw, range_m=True),
           dict(raw, generator='nowhere:fn'), dict(raw, generator=f'{HERE}:missing'), dict(raw, generator='x'),
           {k: v for k, v in raw.items() if k != 'cap'}]
    for value in bad:
        with pytest.raises(ValueError):
            lf.settings(dict(on, lc_families=value))
    with pytest.raises(ValueError, match='group_lane_moves'):
        lf.settings(dict(on, group_lane_moves=False))
    with pytest.raises(ValueError):
        tb._check(dict(on, lc_families=bad[0]), 'B')


def test_merge_process_map_fails_closed(cfg, monkeypatch):
    monkeypatch.delenv(md.ENV, raising=False)
    md._reset_for_tests()
    try:
        with pytest.raises(ValueError, match='lane-drop process map'):
            lf.settings(lc_cfg(cfg))
    finally:
        md._reset_for_tests()


def test_off_keys_absent_or_false_are_identical(cfg):
    for lanes in (False, True):
        base = dict(cfg, trigger_partner_slots=4, group_lane_moves=lanes)
        outs = []
        for value in (base, dict(base, lc_families=False)):
            p = make(value)
            c = controller(p)
            rows = lambda dt: [row('a', 20+2*dt, 2, v=2.), row('b', -30+2*dt, 1, v=2.),  # noqa: E731
                               row('h', 2*dt, 1, cav=False, v=2.)]
            samples = [step(p, c, rows(t-300), t, Capture())[0] for t in range(300, 304)]
            outs.append(canon(samples)+canon({k: v.tolist() for k, v in p.worker_state()['trigger'].items()}))
            assert p.lc_families is None and c.lc_requests is None
            assert 'lc_families' not in p.local_states['a'] and 'lc_families' not in samples[0]['track2']
            assert all('lc_tick' not in s for s in samples)
        assert outs[0] == outs[1]


SCENARIO = r'''
import hashlib, json, sys, torch
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1]+'/src')
import tools.coopmwm as pkg
if sys.argv[2] != '-':
    pkg.__path__.insert(0, sys.argv[2])
from tools.coopmwm import exploration, track2_backend as tb
from tools.coopmwm.acceleration import Controller
from tools.coopmwm.v2v_intent import IntentPolicy
from tools.coopmwm.v2v_runtime import controller_class
exploration.regime = lambda rows: 'ordinary'
torch.set_num_threads(1)
cfg = json.loads(sys.argv[3])
torch.manual_seed(7)
p = tb.EpisodePolicy(IntentPolicy(), cfg, 'B')
controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
def rows(dt):
    out = [('a', 20., 2, True), ('b', -30., 1, True), ('c', -60., 0, True), ('h1', 0., 1, False)]
    return [dict(vid=v, s_m=s+2.*dt, v_mps=2., lane=l, is_cav=c, length_m=5.) for v, s, l, c in out]
lanes = []
for now in range(300, 306):
    r = rows(now-300)
    p.observe(r, now)
    controller.apply(r, {x['vid']: 0.1 for x in r if x['is_cav']}, now)
    t = controller.lane_request(type('E', (), {'slot_vehicle_ids': lambda self: [x['vid'] for x in r]})())
    lanes.append(None if t is None else t.tolist())
    controller.observe_lane(dict(lane_commands={}, lane_vehicles={}))
digest = hashlib.sha256()
digest.update(json.dumps([[e['sample'], e['requests'], sorted(e)] for e in controller.events],
                         sort_keys=True, default=repr).encode())
digest.update(json.dumps([lanes, sorted(controller.lane_actors())]).encode())
for part in ('layer', 'trigger'):
    for k, v in sorted(p.worker_state()[part].items()):
        digest.update(k.encode()); digest.update(v.detach().numpy().tobytes())
print(digest.hexdigest())
'''

TOUCHED = ('track2_backend.py', 'feasible_proposals.py', 'v2v_runtime.py', 'acceleration.py',
           'sparse_synergy_credit.py')


@pytest.mark.parametrize('lanes', [False, True])
def test_off_is_byte_identical_to_the_base_commit(cfg, tmp_path, lanes):
    """Default-off samples, requests, lane tensors and initial weights equal the aafdb8d2 sources."""
    overlay = tmp_path/'overlay'
    overlay.mkdir()
    for name in TOUCHED:
        try:
            text = subprocess.run(['git', '-C', str(REPO), 'show', f'{BASE_COMMIT}:tools/coopmwm/{name}'],
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip('base commit source unavailable in this checkout')
        (overlay/name).write_bytes(text)
    value = json.dumps(dict(cfg, run_root=str(tmp_path), group_lane_moves=lanes, trigger_partner_slots=4))
    digests = []
    for path in ('-', str(overlay)):
        run = subprocess.run([sys.executable, '-c', SCENARIO, str(REPO), path, value], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]
        digests.append(run.stdout.strip())
    assert len(digests[0]) == 64 and digests[0] == digests[1]


# ------------------------------------------------------------------ options / templates

def test_program_option_round_trip_validation_and_registered_templates(cfg, wz):
    frame = lp.frame_from_descriptor(wz)
    t = lf.template_of(WALL_FILL, cap=6, frame=frame, n_lanes=3)
    option = lf.ProgramOption('free_program_initiate', ('a',), 'free_program:a@300', 'a', 0, t.to_dict(), 0, 10,
                              0., 1, 6, 0., 100., 1)
    assert lf.ProgramOption.from_dict(json.loads(json.dumps(option.to_dict()))) == option
    assert lf.option_from_dict(option.to_dict()) == option and hash(option) == hash(lf.ProgramOption(**option.__dict__))
    assert option.name == 'free_program' and option.kind == 'initiate' and option.actors == ('a',)
    assert option.self_validated_windows and (option.onset_s, option.hold_s) == (0, 10)
    bad = [dict(slot=1), dict(hold=9), dict(onset=1), dict(depth=.5), dict(cap=2), dict(cap=7), dict(initiator='b'),
           dict(members=('a', 'b')), dict(family='free_program_x'), dict(member_count=0)]
    for change in bad:
        with pytest.raises(ValueError):
            lf.ProgramOption(**dict(option.__dict__, **change))
    join = lf.ProgramOption('free_program_join', ('b',), 'free_program:a@300', 'a', 2, t.to_dict(), 0, 5, 0., 2, 6,
                            -40., 100., 1)
    assert join.kind == 'join'
    for change in (dict(slot=0), dict(slot=3), dict(member_count=3), dict(hold=11)):
        with pytest.raises(ValueError):
            lf.ProgramOption(**dict(join.__dict__, **change))
    # Generator contract: '@k' slots in order, registered ranges, map, references.
    with pytest.raises(ValueError, match='registered'):
        lf.template_of(unregistered(None, None, None)[0], cap=6, frame=frame, n_lanes=3)
    with pytest.raises(ValueError, match='@0'):
        lf.template_of(lp.FreeProgram(('@0', '@2'), (HOLD, HOLD), 'wz3to2', (0, 0), 10, {},
                                      (None, dict(lane=1, ds=(0., 1.)))), cap=6, frame=frame, n_lanes=3)
    with pytest.raises(ValueError, match='@0'):
        lf.template_of(WALL_FILL, cap=2, frame=frame, n_lanes=3)
    ghost = lp.MemberProgram(lon=lp.Lon('track', v_w=18., refs=('@5',), kp=.15, kv=.6, dmax=1.), hold_s=8)
    with pytest.raises(ValueError, match='names no slot'):
        lf.template_of(lp.FreeProgram(('@0',), (ghost,), 'wz3to2', (0,), 10, {}, (None,)), cap=6, frame=frame,
                       n_lanes=3)
    # Offline template (FULLPOL bank BC): ds windows +- 25 m containing the t0 offsets, refs -> '@k'.
    concrete = lp.FreeProgram(('x', 'y', 'z'), (HOLD, HOLD, lp.MemberProgram(
        lon=lp.Lon('track', v_w=18., refs=('x', 'y'), off=-25., kp=.15, kv=.6, dmax=1., settle_gap=25.),
        lane=lp.LaneReq(2, 'hold'), partner='y', hold_s=8)), 'wz3to2', (0, 2, 5), 10, dict(row='r1'))
    rows = [row('x', -1000, 1), row('y', -1003, 0), row('z', -1050, 1)]
    tmpl = lf.template_from_program(concrete, frame, rows)
    assert tmpl.members == ('@0', '@1', '@2') and tmpl.join_el == (0, 2, 5) and tmpl.source['row'] == 'r1'
    assert tmpl.join_rules == (None, dict(lane=0, ds=(-28., 22.)), dict(lane=1, ds=(-75., -25.)))
    assert tmpl.programs[2].lon.refs == ('@0', '@1') and tmpl.programs[2].partner == '@1'
    assert lf.template_of(tmpl, cap=6, frame=frame, n_lanes=3) == tmpl
    resolved = lf.resolve(tmpl.programs[2], {0: 'p', 2: 'r'})
    assert resolved.lon.refs == ('p', '@1') and resolved.partner == '@1'
    # Built-in sampler draws only registered values (test-fixture default).
    state = dict(frame=frame, cap=6, sensed=())
    for seed in range(20):
        for t in lf.default_generator(state, row('a', -1000, 1), np.random.default_rng(seed)):
            assert lf.template_of(t, cap=6, frame=frame, n_lanes=3) == t
    assert lf.default_generator(state, row('a', 100, 1), np.random.default_rng(0)) == []   # past the drop


# ------------------------------------------------------------------ E2E: offer -> initiate -> join -> execute -> release

ROSTER = {'c0': True, 'c1': True, 'c2': True, 'h1': False, 'h2': False}


def e2e_rows(t):
    dt = t-300
    fill_lane = 2 if t >= 303 else 1
    return [row('c0', -1000+20*dt, 1), row('c1', -1005+20*dt, 0), row('c2', -1040+20*dt, fill_lane),
            row('h1', -1110+20*dt, 2, cav=False), row('h2', -900+20*dt, 0, cav=False)]


def harness_states(rows, frame):
    return {r['vid']: dict(edge='up', sumo_lane=frame.sumo(r['lane']), x=r['s_m']+frame.anchor_x_m, v=r['v_mps'],
                           allowed=31.29, is_cav=r['is_cav']) for r in rows}


def test_end_to_end_wall_fill_program_on_the_real_env_writer(cfg, wz):
    from .lc_writer_harness import Harness
    p = make(lc_cfg(cfg, cap=3))
    p.trigger.trigger_head = Prefer(p.trigger)
    c = controller(p)
    frame = lp.frame_from_descriptor(wz)
    h = Harness(None, ROSTER, desc=wz)
    samples, writes, requests, snap = {}, {}, {}, {}
    for t in range(300, 312):
        rows = e2e_rows(t)
        env = Capture()
        s, _ = step(p, c, rows, t, env)
        samples[t] = s
        requests[t] = env.ticks[-1] if env.ticks else None
        assert len(env.ticks) <= 1 and (requests[t] or None) == s.get('lc_tick')
        session = p.lc_families.sessions.get('free_program:c0@300')
        if session is not None:
            snap[t] = dict(slots=dict(session['slots']), refs=session['executors'].get('c2') and
                           session['executors']['c2'].p.lon.refs, lane_done=session['executors'].get('c2') and
                           session['executors']['c2'].lane_done, actors=c.lane_actors())
        writes[t] = h.tick(t-300, harness_states(rows, frame), requests[t])
        c.observe_lane(h.last_info)
    # 300: c0 (lane 1, front) initiates from its own state; nobody has received anything yet.
    s = samples[300]
    assert s['track2_trigger']['accepted'] == dict(c0=True, c1=False, c2=False)
    assert 'free_program_join' not in offered(s, 'c1') and 'free_program_initiate' not in offered(s, 'c1')
    assert s['track2']['lc_families']['offers'] == {}
    assert requests[300] == dict(accel=dict(c0=-1.0), lane_mode=dict(c0=513))
    assert sorted(writes[300]) == [(0, 'c0', 'setLaneChangeMode', 513), (0, 'c0', 'slowDown', 19.0)]
    assert [m['broadcaster'] for m in s['track2']['lc_families']['published']] == ['c0']
    # 301: intents received over V2V (t-1, in range): first unfilled slot whose join rule holds.
    s = samples[301]
    stats = s['track2_trigger']['proposals']['lc_families']
    assert stats['event'] and stats['event_cavs'] == ['c1', 'c2'] and stats['accepted_join'] == 2
    assert s['track2']['lc_families']['offers'] == dict(c1=[['free_program:c0@300', 1]],
                                                       c2=[['free_program:c0@300', 2]])
    assert snap[300]['slots'] == {0: 'c0'} and snap[300]['actors'] == frozenset({'c0'})
    assert snap[301]['slots'] == {0: 'c0', 1: 'c1', 2: 'c2'}
    assert snap[301]['refs'] == ('c0', 'c1')                                # '@k' resolved as slots filled
    assert requests[301]['lanes'] == dict(c2=[2, 2.0]) and requests[301]['tau'] == dict(c2=2.0)
    assert requests[301]['lane_mode'] == dict(c0=513, c1=513, c2=513)
    w = writes[301]
    assert (1, 'c2', 'changeLane', (0, 2.0)) in w and (1, 'c2', 'setTau', 2.0) in w   # into SUMO lane 0
    assert (1, 'c1', 'setLaneChangeMode', 513) in w and (1, 'c2', 'setLaneChangeMode', 513) in w
    assert snap[301]['actors'] == frozenset({'c0', 'c1', 'c2'})
    assert c.events[301-300]['lc_lane_outcome']['vehicles']['c2'] == dict(target=2, status='issued',
                                                                            completed=False)
    # The filler tracks the A/B slot (accel within [-dmax, amax]); it reaches the closing lane at 303.
    accel = {t: requests[t].get('accel', {}) for t in range(301, 308)}
    assert all(-1. <= a <= 1. for a in accel[301].values()) and accel[301]['c2'] == 1.0
    assert snap[302]['lane_done'] is None and snap[303]['lane_done'] == 3
    # 308: hold_end release -> no request; the env restores lcMode / tau and hands back to native.
    assert requests[307] and requests[308] is None and 'lc_tick' not in samples[308]
    w = writes[308]
    for vid in ('c0', 'c1', 'c2'):
        assert (8, vid, 'setLaneChangeMode', 1621) in w and (8, vid, 'setSpeed', -1.0) in w
    assert (8, 'c2', 'setTau', 1.0) in w
    assert h.env._lc_original == {} and not h.env._lc_outstanding()
    assert writes[309] == [] and writes[310] == [] and writes[311] == []
    # HDVs: never offered, members or commanded; TraCI writes only for CAV members.
    assert all(c[1] in ('c0', 'c1', 'c2') for t in writes for c in writes[t])
    for s in samples.values():
        receipt = s['track2']['lc_families']
        assert not {'h1', 'h2'} & set(receipt['offers']) and not {'h1', 'h2'} & set(receipt['commands'])
        assert not {'h1', 'h2'} & {v for k in (s.get('lc_tick') or {}).values() for v in k}
    # Per member-second labels (schema lcexec-tick-label-v1) and executor receipts.
    labels = [lab for t in range(300, 310) for lab in samples[t]['track2']['lc_families']['labels']]
    assert {(lab['vid'], lab['t']) for lab in labels} == {('c0', 300)} | {
        (v, t) for v in ('c0', 'c1', 'c2') for t in range(301, 310)}
    assert all(lab['schema'] == lp.LABEL_SCHEMA and lab['session_id'] == 'free_program:c0@300' for lab in labels)
    by = {(lab['vid'], lab['t']): lab for lab in labels}
    assert by[('c2', 301)]['lane_target'] == 2 and by[('c2', 301)]['tau'] == 2.0
    assert by[('c0', 308)]['event'] == 'release' and by[('c0', 308)]['release_reason'] == 'hold_end'
    assert by[('c0', 300)]['event'] == 'activate' and by[('c0', 300)]['accel_mps2'] == -1.0
    final = samples[309]['track2']['lc_families']['sessions'][0]['receipts']
    assert final['c2']['lane_outcome'] == 'reached' and final['c0']['release_reason'] == 'hold_end'
    expired = samples[310]['track2']['lc_families']['expired']
    assert expired and expired[0]['session_id'] == 'free_program:c0@300'
    assert p.lc_families.membership == {} and p.lc_families.sessions == {}
    # Recorded program decisions replay exactly for trigger PPO.
    record = json.loads(json.dumps(samples[301]['track2_trigger']))
    batch = tb._batch(record)
    assert any(isinstance(m, lf.ProgramOption) for table in batch.tables for m in table)
    logp, _, _ = p.trigger.evaluate(batch)
    assert logp.tolist() == pytest.approx(record['logp'])
    from . import sparse_synergy_credit as g4
    assert g4.opportunities([dict(id='ep', steps=[dict(time_s=301, sample=samples[301])])]) == []


def test_b_and_r_run_the_identical_protocol(cfg, wz):
    out = {}
    for arm in ('B', 'R'):
        p = make(lc_cfg(cfg, cap=3), arm)
        p.trigger.trigger_head = Prefer(p.trigger)
        c = controller(p)
        events = []
        for t in range(300, 306):
            env = Capture()
            s, req = step(p, c, e2e_rows(t), t, env)
            assert s['track2'].pop('arm') == arm
            events.append([s, req, env.ticks])
        out[arm] = canon(events)
    assert out['B'] == out['R'] and '"free_program_join"' in out['B']


# ------------------------------------------------------------------ offers: range, t+1, slots, cap, ownership, relay

def chain_rows(t, extra=()):
    dt = t-300
    out = [row(f'c{k}', -1000-40*k+2*dt, 1, v=2.) for k in range(6)]
    out.append(row('h1', -1020+2*dt, 1, cav=False, v=2.))           # an HDV inside slot 1's window
    return out+list(extra)


def test_relay_chaining_fills_a_six_member_group_beyond_the_initiator_range(cfg, wz):
    p = make(lc_cfg(cfg, cap=6, generator='chain6'))
    p.trigger.trigger_head = Prefer(p.trigger)
    c = controller(p)
    joins = {}
    for t in range(300, 307):
        s, _ = step(p, c, chain_rows(t), t)
        for j in s['track2']['lc_families']['joins']:
            joins[j['vid']] = (j['time_s'], j['slot'])
    sid = 'free_program:c0@300'
    assert joins == dict(c1=(301, 1), c2=(301, 2), c3=(303, 3), c4=(303, 4), c5=(305, 5))
    session = p.lc_families.sessions[sid]
    assert session['slots'] == {k: f'c{k}' for k in range(6)}
    # c4 / c5 were never within the initiator's radio range (100 m); they joined through relays.
    for vid, t in (('c4', 303), ('c5', 305)):
        rows = {r['vid']: r for r in chain_rows(t)}
        assert abs(rows[vid]['s_m']-rows['c0']['s_m']) > 100.
    assert 'h1' not in p.lc_families.membership and 'h1' not in p.lc_families.rows
    # Full group: nobody is offered the session any more.
    s, _ = step(p, c, chain_rows(307, [row('c6', -1240+14, 1, v=2.)]), 307)
    assert 'c6' not in s['track2']['lc_families']['offers']


def test_offers_need_radio_range_t_plus_one_and_respect_the_cap(cfg, wz):
    p = make(lc_cfg(cfg, cap=3, generator='chain6'))
    with pytest.raises(ValueError, match='@0'):        # a 6-slot template exceeds cap 3 (fail closed)
        p.trigger.trigger_head = Prefer(p.trigger)
        c = controller(p)
        step(p, c, chain_rows(300), 300)
    p = make(lc_cfg(cfg, cap=3, range_m=50.))
    p.trigger.trigger_head = Prefer(p.trigger)
    c = controller(p)

    def rows(t):
        dt = 2.*(t-300)
        return [row('c0', -1000+dt, 1, v=2.), row('c1', -1005+dt, 0, v=2.), row('c2', -1060+dt, 1, v=2.),
                row('c3', -1040+dt, 1, v=2.), row('h1', -1020+dt, 0, cav=False, v=2.)]
    s, _ = step(p, c, rows(300), 300)
    assert p.lc_families.offers == {} and s['track2_trigger']['accepted']['c0']     # nothing before t+1
    s, _ = step(p, c, rows(301), 301)
    # c2 is 58 m from c0's message position (> 50 m radio range): no offer although its rule holds;
    # c3 (42 m) takes slot 2; the HDV satisfying slot 1's lane/ds window is never offered.
    assert s['track2']['lc_families']['offers'] == dict(c1=[['free_program:c0@300', 1]],
                                                       c3=[['free_program:c0@300', 2]])
    assert p.lc_families.sessions['free_program:c0@300']['slots'] == {0: 'c0', 1: 'c1', 2: 'c3'}
    # Full (3 = template length <= cap): c2, now within range of the relays, is never offered.
    for t in (302, 303, 304):
        s, _ = step(p, c, rows(t), t)
        assert s['track2']['lc_families']['offers'] == {}
    assert len(s['track2']['lc_families']['published']) == 3                    # initiator + 2 relays


def test_one_session_per_cav_and_ownership(cfg, wz):
    p = make(lc_cfg(cfg, cap=3))
    p.trigger.trigger_head = Prefer(p.trigger)
    c = controller(p)
    step(p, c, e2e_rows(300), 300)
    step(p, c, e2e_rows(301), 301)
    proto = p.lc_families
    state = proto.state('c1', 301)
    again = lf.pool('free_program_initiate', dict(state, propose=lambda *a: [WALL_FILL]), proto.rows['c1'],
                    np.random.default_rng(0))[0]
    assert not proto.start(tb.TriggerMove(again), ego='c1', executor=p.executor, observable_cav_ids=['c1'], now=301)
    # A join naming a filled slot / a full session is refused; a mismatched schedule raises.
    msg = next(iter(proto.board.values()))
    late = lf.ProgramOption('free_program_join', ('c9',), msg.session_id, 'c0', 2, msg.template, 0, 5, 0., 1, 3)
    proto.rows['c9'] = row('c9', -1040, 1)
    assert not proto.start(late, ego='c9', executor=p.executor, observable_cav_ids=['c9'], now=301)
    proto.sessions[msg.session_id]['slots'].pop(2)
    proto.sessions[msg.session_id]['members'].pop('c2')
    with pytest.raises(ValueError, match='schedule'):
        proto.start(late, ego='c9', executor=p.executor, observable_cav_ids=['c9'], now=301)


# ------------------------------------------------------------------ redraw None guard (N11 F8)

@pytest.mark.parametrize('generator', ['empty', 'nothing', 'flaky'])
def test_generator_without_templates_never_crashes_the_slate(cfg, wz, generator):
    _FLAKY['calls'] = 0
    p = make(lc_cfg(cfg, cap=3, generator=generator))
    p.trigger.trigger_head = Prefer(p.trigger)
    c = controller(p)
    s, _ = step(p, c, e2e_rows(300), 300)
    assert 'free_program_initiate' not in offered(s, 'c0')
    assert s['track2_trigger']['accepted'] == dict(c0=False, c1=False, c2=False)
    assert all(m is None or m['family'] for table in s['track2_trigger']['tables'] for m in table)
    if generator == 'flaky':
        assert _FLAKY['calls'] >= 2                                        # pool drew, redraw had none


def test_unregistered_generator_template_fails_closed(cfg, wz):
    p = make(lc_cfg(cfg, cap=3, generator='unregistered'))
    c = controller(p)
    with pytest.raises(ValueError, match='registered'):
        step(p, c, e2e_rows(300), 300)


# ------------------------------------------------------------------ executor inputs / gate / reservations

def _proto(cfg, wz, generator='wall_fill', cap=3):
    proto = lf.Protocol(lf.settings(lc_cfg(cfg, cap=cap, generator=generator)))
    return proto


class _Exec:
    def __init__(self):
        self.calls = []

    def start(self, move, *, ego, observable_cav_ids, now, until):
        self.calls.append((ego, now, until))
        return True


def _session(proto, rows, now=300, joins=(('c1', 1), ('c2', 2))):
    table = {r['vid']: r for r in rows}
    sensing = {v: [r for r in rows if abs(r['s_m']-table[v]['s_m']) <= 100.] for v in table}
    proto.receive(now, {v: r for v, r in table.items() if r['is_cav']}, sensing, {})
    state = proto.state('c0', now)
    option = lf.pool('free_program_initiate', state, table['c0'], np.random.default_rng(0))[0]
    assert proto.start(option, ego='c0', executor=_Exec(), observable_cav_ids=['c0'], now=now)
    sid = option.session_id
    for vid, slot in joins:
        join = lf.ProgramOption('free_program_join', (vid,), sid, 'c0', slot, option.template, 0, 10, 0.,
                                len(proto.sessions[sid]['members']), proto.cap)
        assert proto.start(join, ego=vid, executor=_Exec(), observable_cav_ids=[vid], now=now)
    return sid


def test_member_observation_uses_only_own_row_sensing_and_teammate_packets(cfg, wz):
    outs = []
    for noisy in (False, True):
        proto = _proto(cfg, wz)
        rows = e2e_rows(301)
        if noisy:
            rows = [dict(r, driver_type='aggressive', tau_truth=.3) for r in rows]
            rows += [row('far_hdv', -600, 2, cav=False), row('far_cav', -1500, 2)]   # outside every 100 m sensing
        sid = _session(proto, rows, joins=(('c2', 2),))                              # slot 1 still unfilled
        proto.prev_reports = {'c0': report('c0', -1000, 1, 20., 300)}
        assert proto.sessions[sid]['executors']['c2'].p.lon.refs == ('c0', '@1')    # '@1' unfilled: missing ref
        obs = proto.observation(sid, 'c2', 301)
        assert obs.own == lp.View('c2', -1020., 1, 20., True, True) and obs.ref('@1') is None
        assert {v.vid for v in obs.sensed} == {'c0', 'c1', 'h1'}                    # h2: 140 m, unsensed
        active = [(v, lf.ProgramOption('free_program_join', (v,), sid, 'c0', 2,
                                       proto.sessions[sid]['template_dict'], 0, 9, 0., 1, 3)) for v in ('c2',)]
        tick = proto.commands(301, active)
        outs.append(canon([tick, proto.last_commands, proto.last_labels, proto.last_gate]))
    assert outs[0] == outs[1]
    # A reference outside own sensing comes from the teammate's OWN t-1 packet (within radio range).
    proto = _proto(cfg, wz)
    rows = [row('c0', -1000, 1), row('c1', -1005, 0), row('c2', -1110, 1)]
    sid = _session(proto, rows)
    proto.prev_reports = {'c0': report('c0', -1020, 1, 20., 298), 'c1': report('c1', -1025, 0, 19., 299)}
    obs = proto.observation(sid, 'c2', 300)
    assert obs.sensed == () and obs.refs == {'c1': lp.View('c1', -1025., 0, 19., True, True)}   # c0: stale
    proto.prev_reports['c0'] = report('c0', -1020, 1, 20., 299)
    assert proto.observation(sid, 'c2', 300).refs['c0'] == lp.View('c0', -1020., 1, 20., True, True)
    proto.prev_reports['c0'] = report('c0', -1005, 1, 20., 299)                     # 105 m: beyond radio range
    assert 'c0' not in proto.observation(sid, 'c2', 300).refs


def test_lane_gate_own_sensing_rules(cfg, wz):
    frame = lp.frame_from_descriptor(wz)
    gate = lambda rows, target, vid='c': lf.lane_gate(vid, target, rows, frame, 3)  # noqa: E731
    ego = row('c', -1000, 1)
    assert gate([ego], 2) == (True, 'ok')                                          # closing lane upstream
    assert gate([ego], 0) == (True, 'ok')
    assert gate([ego, row('h', -1003, 2, cav=False)], 2) == (False, 'unsafe_gap_or_edge')   # alongside
    assert gate([ego, row('h', -1003, 1, cav=False)], 1) == (True, 'hold')        # same lane: no gap check
    assert gate([row('c', -1000, 0)], 2) == (False, 'not_adjacent')
    assert gate([row('c', 80, 1)], 2) == (False, 'no_target_lane')               # past the drop
    assert gate([row('c', 60, 1)], 2) == (False, 'unsafe_gap_or_edge')           # inside the edge guard
    assert gate([row('c', 60, 2)], 1) == (True, 'ok')                            # leaving the closing lane
    assert gate([ego], 3) == (False, 'not_adjacent') and gate([row('c', -1000, 2)], 3) == (False, 'no_target_lane')
    assert gate([row('c', -1000, 1, cav=False)], 2) == (False, 'missing_observable_gap_or_cav')
    assert gate([ego, row('x', -800, 1)], 2) == (False, 'missing_observable_gap_or_cav')   # beyond own sensing


def test_unsafe_lane_request_is_dropped_with_a_receipt(cfg, wz):
    proto = _proto(cfg, wz)
    rows = e2e_rows(300)[:3]+[row('h3', -1042, 2, cav=False)]
    sid = _session(proto, rows)
    template = proto.sessions[sid]['template_dict']
    active = [(v, lf.ProgramOption('free_program_join', (v,), sid, 'c0', k, template, 0, 10, 0., 1, 3))
              for v, k in (('c1', 1), ('c2', 2))]+[('c0', lf.ProgramOption(
                  'free_program_initiate', ('c0',), sid, 'c0', 0, template, 0, 10, 0., 1, 3))]
    tick = proto.commands(300, active)
    assert 'c2' not in tick.get('lanes', {}) and proto.last_gate == dict(c2=dict(target=2, reason='unsafe_gap_or_edge'))
    assert proto.last_commands['c2']['gate'] == dict(target=2, reason='unsafe_gap_or_edge')
    label = next(lab for lab in proto.last_labels if lab['vid'] == 'c2')
    assert label['lane_target'] == -1 and proto.sessions[sid]['executors']['c2'].lane_requests == 1
    # Controller-reserved members get nothing (no command, no lane mode, no tau).
    proto2 = _proto(cfg, wz)
    sid2 = _session(proto2, e2e_rows(300)[:3])
    tick = proto2.commands(300, [(v, o) for v, o in active], reserved={'c2'})
    assert proto2.last_commands['c2']['status'] == 'controller_reserved'
    assert all('c2' not in value for value in tick.values()) and sid2 == sid


def test_hdv_rows_never_enter_the_protocol(cfg, wz):
    proto = _proto(cfg, wz)
    rows = e2e_rows(300)
    table = {r['vid']: r for r in rows}
    proto.receive(300, table, {v: rows for v in table}, {})
    assert set(proto.rows) == {'c0', 'c1', 'c2'}
    assert lf.pool('free_program_initiate', proto.state('c0', 300), table['h1'], np.random.default_rng(0)) == []


def test_trigger_columns_zero_initialised_and_compose_with_n9(cfg, wz):
    from . import nm_families as nf
    off, on = make(dict(cfg, trigger_partner_slots=4, trigger_candidates=4, group_lane_moves=True)), make(lc_cfg(cfg))
    t_off, t_on = off.trigger.trigger_head[0].weight, on.trigger.trigger_head[0].weight
    assert t_on.shape[1] == t_off.shape[1]+len(lf.DESCRIPTOR)
    assert torch.equal(t_on[:, :t_off.shape[1]], t_off) and not bool(t_on[:, t_off.shape[1]:].any())
    assert on.trigger.option_families[-2:] == lf.PROGRAM_FAMILIES
    assert set(lf.PROGRAM_FAMILIES) <= set(on.executor.moves_api.EXECUTABLE_FAMILIES)
    both = make(dict(lc_cfg(cfg), **nf.config_keys(provenance=PROV, map_name='wz3to2', families=dict(early_merge=2))))
    ext = both.trigger.extension
    assert isinstance(ext, nf.Composite) and ext.nm is not None and ext.context_width == len(nf.GEOMETRY)
    assert both.trigger.option_families[-4:] == ('early_merge_initiate', 'early_merge_join')+lf.PROGRAM_FAMILIES
    frame = lp.frame_from_descriptor(wz)
    option = lf.ProgramOption('free_program_join', ('c2',), 'free_program:c0@300', 'c0', 2,
                              lf.template_of(WALL_FILL, cap=6, frame=frame, n_lanes=3).to_dict(), 0, 9, 0., 2, 3,
                              40., 100., 1)
    d = dict(zip(lf.DESCRIPTOR, lf.TriggerExtension(None).descriptor(option)))
    assert d['is_program_join'] == 1. and d['slot_2'] == 1. and d['lon_track'] == 1. and d['tau_present'] == 1.
    assert d['lane_change_sign'] == 1. and d['v_w_norm'] == pytest.approx(3/7) and d['gap_over_range'] == .4
    assert lf.TriggerExtension(None).descriptor(option.to_dict()) == lf.TriggerExtension(None).descriptor(option)
    assert lf.TriggerExtension(None).descriptor(None) == [0.]*len(lf.DESCRIPTOR)


def test_controller_requires_the_lc_runtime_env_and_applies_branch_release(cfg, wz):
    p = make(lc_cfg(cfg, cap=3))
    p.trigger.trigger_head = Prefer(p.trigger)
    c = controller(p)
    step(p, c, e2e_rows(300), 300)
    assert c.lc_requests == dict(accel=dict(c0=-1.0), lane_mode=dict(c0=513))
    with pytest.raises(ValueError, match='lc_runtime'):
        c.lane_request(SimpleNamespace(slot_vehicle_ids=lambda: []))
    c.branch = dict(time_s=300, revert=['c0'], group=['c0'], until=400)
    assert lf.controller_requests(c, 300) is None

    class Stub:
        def sample(self, context, generator, deterministic=False, memory=None):
            return dict(context=context, action=[[0.] for _ in context['members']],
                        lc_tick=dict(native=['c0'])), {}
    plain = Controller(Stub(), torch.Generator().manual_seed(9), True)               # no V2V lane runtime
    with pytest.raises(ValueError, match='single-writer'):
        plain.apply(e2e_rows(300), {r['vid']: 0. for r in e2e_rows(300) if r['is_cav']}, 300)


def test_offline_open_and_admit_reconstruct_a_session_without_executors(cfg, wz):
    """FULLPOL bank BC helpers: open_offline / admit_offline mirror start() without commitments."""
    proto = _proto(cfg, wz)
    rows = e2e_rows(300)
    table = {r['vid']: r for r in rows}
    proto.receive(300, {v: r for v, r in table.items() if r['is_cav']}, {v: rows for v in table}, {})
    option = lf.pool('free_program_initiate', proto.state('c0', 300), table['c0'], np.random.default_rng(0))[0]
    proto.open_offline(option, 300)
    sid = option.session_id
    proto.publish(300, {'c0': ObservedVehicle('c0', -1000., 1, 20., 5.)})
    offers = proto.receive(301, {v: dict(r, s_m=r['s_m']+20) for v, r in table.items() if r['is_cav']},
                           {v: rows for v in table}, {})
    assert {v: [(o.session_id, o.slot) for o in opts] for v, opts in offers.items()} == dict(
        c1=[(sid, 1)], c2=[(sid, 2)])
    assert proto.admit_offline(offers['c2'][0], 301) and not proto.admit_offline(offers['c2'][0], 301)
    assert proto.admit_offline(offers['c1'][0], 301)
    session = proto.sessions[sid]
    assert session['slots'] == {0: 'c0', 1: 'c1', 2: 'c2'} and session['executors']['c2'].p.lon.refs == ('c0', 'c1')
    assert proto.membership == dict(c0=sid, c1=sid, c2=sid)
