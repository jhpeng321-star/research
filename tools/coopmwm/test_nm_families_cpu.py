"""N9 lane-drop move families: CPU contracts (no simulator, GPU or network).

Real EpisodePolicy / TriggerPolicy / MoveExecutor / feasible slates / v2v_runtime
controller / G5 group_lane gate + lanes_tensor code paths. Fakes: a test-only trigger
scoring head, fixed registered draws and a slot-id env stub. Geometry: the nm_map
stubs ('merge': the ramp lane as the closing lane, so the unchanged G5 envelope
applies; 'wz3to2' for geometry-only checks).
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from . import bank_breadth as breadth, group_lane, moves, nm_families as nf, nm_map
from . import track2_backend as tb, v2v_wall
from .acceleration import Controller
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)
from .v2v_intent import IntentPolicy
from .v2v_runtime import controller_class
from tools.coopnative.policy import encode

PROV = 'N9 CPU fixture (owner 2026-11-06 new-map build)'
REPO = Path(__file__).resolve().parents[2]
BASE_COMMIT = '3e5bc9f9'
FIXED = dict(
    platoon_headway=dict(time_headway_s=1.2),
    early_merge=dict(upstream_m=800., sequence_gap_s=2, attempt_s=5),
    zipper_order=dict(zipper_zone_m=50., order_headway_s=1.5, attempt_s=5, order='through_first'),
    lane_balance=dict(sequence_gap_s=2, attempt_s=5),
    discharge_sync=dict(queue_speed_mps=5., zone_m=100., release_speed_mps=15., sync_lead_s=2),
    harmonize_rows=dict(row_speed_mps=18., row_spacing_m=60.))


def fam_cfg(cfg, families, map_name='merge', range_m=None, **extra):
    out = dict(cfg, trigger_partner_slots=4, trigger_candidates=8, group_lane_moves=True)
    out.update(nf.config_keys(provenance=PROV, map_name=map_name, families=families, range_m=range_m))
    out.update(extra)
    return out


def make(cfg, arm='B'):
    torch.manual_seed(7)
    return tb.EpisodePolicy(IntentPolicy(), cfg, arm)


def row(vid, s, lane, cav=True, v=2.):
    return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=lane, is_cav=cav, length_m=5.)


def controller(p):
    return controller_class(Controller)(p, torch.Generator().manual_seed(9), True)


def step(p, c, rows, now, native=None):
    """One real control second: observe -> v2v_runtime apply (policy sample, G5 gate)."""
    p.observe(rows, now)
    commands = {r['vid']: 0. for r in rows if r['is_cav']}
    commands.update(native or {})
    requests = c.apply(rows, commands, now)
    return c.events[-1]['sample'], requests


def env_for(rows):
    ids = [r['vid'] for r in rows]
    return SimpleNamespace(slot_vehicle_ids=lambda: list(ids))


class Prefer(nn.Module):
    """Test-only trigger head: sum of weight * product of named N9 columns."""

    def __init__(self, trigger, terms):
        super().__init__()
        hidden = trigger.base_policy.actor[-1].in_features
        base = len(trigger.families)+trigger.partner_slots+3
        ext = trigger.extension
        parts = ext.parts if isinstance(ext, nf.Composite) else (ext,)
        offset_desc = hidden+base
        offset_ctx = hidden+trigger.descriptor_width
        for part in parts:
            if isinstance(part, nf.TriggerExtension):
                break
            offset_desc += part.descriptor_width
            offset_ctx += part.context_width
        cols = {name: offset_desc+i for i, name in enumerate(nf.DESCRIPTOR)}
        cols.update({name: offset_ctx+i for i, name in enumerate(nf.GEOMETRY)})
        self.terms = [(float(w), [cols[n] for n in names]) for w, names in terms]

    def forward(self, x):
        out = torch.zeros(x.shape[:-1])
        for weight, cols in self.terms:
            term = torch.ones(x.shape[:-1])
            for col in cols:
                term = term*x[..., col]
            out = out+weight*term
        return out[..., None]


def initiate_then_join(family, role):
    return [(50., [f'is_{family}', 'is_initiate', f'role_{role}']), (40., ['is_join'])]


@pytest.fixture
def fixed(monkeypatch):
    def common(spec, rng):
        depth = {'pinned_lo': nf.DEPTH[0], 'pinned_hi': nf.DEPTH[1]}.get(spec.depth, .6)
        return dict(FIXED[spec.name]), 0, 40, depth
    monkeypatch.setattr(nf, '_sample_common', common)


def canon(value):
    return json.dumps(value, sort_keys=True, default=repr)


def families_offered(sample, ego):
    trig = sample['track2_trigger']
    i = trig['context']['members'].index(ego)
    return [m['family'] for m in trig['tables'][i] if m]


# ------------------------------------------------------------------ registration

def test_settings_default_off_and_registered_values(cfg):
    assert nf.settings(cfg) is None and nf.extension(cfg) is None
    assert nf.settings(dict(cfg, nm_families=False)) is None
    on = fam_cfg(cfg, dict(platoon_headway=4, early_merge=3, zipper_order=4, lane_balance=1,
                           discharge_sync=2, harmonize_rows=6), map_name='wz3to2', range_m=200.)
    got = nf.settings(on)
    assert got['families'] == nf.FAMILY_NAMES and got['range_m'] == 200.
    assert got['geometry'].drop_s_m == 2700. and got['geometry'].g5_executable
    raw = on['nm_families']
    bad = [dict(raw, rule='x'), dict(raw, provenance=' '), dict(raw, enabled=1), dict(raw, extra=1),
           dict(raw, families={}), dict(raw, families=dict(walls=dict(member_cap=2))),
           dict(raw, families=dict(platoon_headway=dict(member_cap=5))),
           dict(raw, families=dict(lane_balance=dict(member_cap=0))),
           dict(raw, families=dict(harmonize_rows=dict(member_cap=7))),
           dict(raw, families=dict(early_merge=dict(member_cap=2, x=1))),
           dict(raw, map='nowhere'), dict(raw, range_m=float('nan')), dict(raw, range_m=True)]
    for value in bad:
        with pytest.raises(ValueError):
            nf.settings(dict(on, nm_families=value))
    with pytest.raises(ValueError):
        nf.settings(dict(on, group_lane_moves=False))     # G5 slate + lane path required
    with pytest.raises(ValueError):
        tb._check(dict(on, nm_families=bad[0]), 'B')


def test_map_descriptor_binding_guards_g5_executability(cfg):
    on = fam_cfg(cfg, dict(early_merge=2, platoon_headway=2), map_name='wz3to2')
    assert nm_map.lane_drop('merge').drop_s_m == pytest.approx(75.44)
    flipped = dict(nm_map.lane_drop('wz3to2').to_dict(), closing_lane=0, adjacent_lane=1, far_lane=2)
    previous = nm_map.bind(lambda name: flipped)
    try:
        with pytest.raises(ValueError, match='G5-executable'):
            nf.settings(on)                               # lane changes would be lane+1
        assert nf.settings(fam_cfg(cfg, dict(platoon_headway=2), map_name='wz3to2')) is not None
        nm_map.bind(lambda name: dict(flipped, n_lanes=2, closing_lane=1, adjacent_lane=0, far_lane=None))
        with pytest.raises(ValueError, match='far through lane'):
            nf.settings(fam_cfg(cfg, dict(lane_balance=1), map_name='wz2to1'))
    finally:
        nm_map.bind(previous)
    assert nm_map.lane_drop('wz3to2').closing_lane == 2


def test_registry_options_and_pinned_depth():
    table = nf.registry()
    assert set(table) == set(nf.FAMILY_NAMES)
    for name, spec in nf.SPECS.items():
        assert table[name]['provenance'] and table[name]['member_cap'] == list(spec.cap)
        params = {p.key: (p.choices[0] if p.kind == 'choice' else (int(p.lo) if p.kind == 'int' else p.lo))
                  for p in spec.params}
        depth = nf.DEPTH[1] if spec.depth == 'pinned_hi' else nf.DEPTH[0]
        role = spec.roles[0]
        option = nf.FamilyOption(f'{name}_initiate', ('a',), 0, 20, depth, f'{name}:a@1', 'a', role,
                                 tuple(params.items()), cap=spec.cap[1])
        assert nf.FamilyOption.from_dict(json.loads(json.dumps(option.to_dict()))) == option
        bad = [dict(depth=.5 if spec.depth != 'sampled' else 1.5), dict(onset=31), dict(hold=19),
               dict(cap=spec.cap[1]+1), dict(role='nobody'), dict(initiator='b')]
        for change in bad:
            kwargs = dict(option.__dict__, **change)
            with pytest.raises(ValueError):
                nf.FamilyOption(**kwargs)
        for p in spec.params:
            wrong = dict(params, **{p.key: 'x' if p.kind == 'choice' else (p.hi+1 if p.kind == 'int' else p.hi*2+1)})
            with pytest.raises(ValueError):
                nf.FamilyOption(**dict(option.__dict__, params=tuple(wrong.items())))


def _platoon_speeds(kp, kd, kff, h, n=6, v0=20., length=5., s0=2.5):
    """0.5 s physics, 1 s control, slowDown-style linear speed change, Euler positions."""
    dt = .5
    x = np.array([-(i)*(length+s0+h*v0) for i in range(n)])
    v, target, rate, prev = np.full(n, v0), np.full(n, v0), np.zeros(n), np.full(n, v0)
    peak = np.zeros(n)
    for k in range(int(200/dt)):
        t = k*dt
        if float(t).is_integer():
            for i in range(n):
                if i == 0:
                    a = -1.5 if 10 <= t < 14 else (1. if 14 <= t < 20 else 0.)
                else:
                    a = nf.cacc_accel(x[i-1]-x[i]-length, v[i], v[i-1], prev[i] if kff else None,
                                      time_headway_s=h, standstill_gap_m=s0, kp=kp, kd=kd, kff=kff)
                    prev[i] = v[i-1]
                    a = max(-4.5, min(1.2, a))
                target[i] = max(0., v[i]+a)
                rate[i] = target[i]-v[i]
        v = np.where(rate >= 0, np.minimum(v+rate*dt, target), np.maximum(v+rate*dt, target))
        x = x+v*dt
        peak = np.maximum(peak, np.abs(v-v0))
    return peak


def test_platoon_cacc_registration_is_string_stable_and_needs_v2v_feedforward():
    spec = nf.SPECS['platoon_headway']
    gains = {k: spec.constant(k) for k in ('kp', 'kd', 'kff')}
    lo, hi = spec.param('time_headway_s').lo, spec.param('time_headway_s').hi
    for h in np.linspace(lo, hi, 13):
        assert nf.string_gain(h, **gains) <= 1+1e-9
        peak = _platoon_speeds(gains['kp'], gains['kd'], gains['kff'], float(h))
        assert all(peak[i+1] <= peak[i]+1e-9 for i in range(len(peak)-1))
    # Without the V2V feed-forward (plain ACC) the same headway amplifies.
    assert nf.string_gain(1.2, kp=gains['kp'], kd=gains['kd'], kff=0.) > 1.01
    peak = _platoon_speeds(gains['kp'], gains['kd'], 0., 1.2)
    assert any(peak[i+1] > peak[i]+1e-6 for i in range(len(peak)-1))
    assert lo < 1.5      # tighter than the IDM textbook default headway


# ------------------------------------------------------------------ off = byte-identical

def test_off_keys_absent_or_false_are_identical(cfg):
    base = dict(cfg, trigger_partner_slots=4, group_lane_moves=True)
    outs = []
    for value in (base, dict(base, nm_families=False)):
        p = make(value)
        c = controller(p)
        rows = lambda dt: [row('a', 20+2*dt, 2), row('b', -30+2*dt, 1), row('h', 0+2*dt, 1, cav=False)]  # noqa: E731
        samples = [step(p, c, rows(t-300), t)[0] for t in range(300, 304)]
        outs.append(canon(samples)+canon({k: v.tolist() for k, v in p.worker_state()['trigger'].items()}))
        assert p.nm_families is None and p.trigger.extension is None
        assert 'nm_families' not in p.local_states['a'] and 'nm_families' not in samples[0]['track2']
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
for now in range(300, 306):
    r = rows(now-300)
    p.observe(r, now)
    controller.apply(r, {x['vid']: 0.1 for x in r if x['is_cav']}, now)
    controller.lane_request(type('E', (), {'slot_vehicle_ids': lambda self: [x['vid'] for x in r]})())
digest = hashlib.sha256()
digest.update(json.dumps([[e['sample'], e['requests']] for e in controller.events],
                         sort_keys=True, default=repr).encode())
for part in ('layer', 'trigger'):
    for k, v in sorted(p.worker_state()[part].items()):
        digest.update(k.encode()); digest.update(v.detach().numpy().tobytes())
print(digest.hexdigest())
'''

TOUCHED = ('track2_backend.py', 'trigger_policy.py', 'feasible_proposals.py', 'merge_features.py',
           'bank_breadth.py', 'sparse_synergy_credit.py', 'v2v_runtime.py', 'group_lane.py',
           'acceleration.py', 'moves.py')


@pytest.mark.parametrize('lanes', [False, True])
def test_off_is_byte_identical_to_the_base_commit(cfg, tmp_path, lanes):
    """Default-off samples, requests and initial weights equal the pre-W1/N9 source (3e5bc9f9)."""
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


# ------------------------------------------------------------------ early_merge: G5 path + sequencing

def early_rows(t, m1_lane=2, with_m2=True):
    dt = t-300
    out = [row('m1', 40+2*dt, m1_lane), row('y1', -30+2*dt, 1), row('h1', -90+2*dt, 1, cav=False)]
    if with_m2:
        out.append(row('m2', 20+2*dt, 2))
    return out


def test_early_merge_sequenced_lane_changes_go_through_the_g5_path(cfg, fixed):
    p = make(fam_cfg(cfg, dict(early_merge=4)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('early_merge', 'merge'))
    c = controller(p)
    # 300: m1 (closing lane, front) initiates; its own lane request passes the G5 gate.
    rows = early_rows(300, with_m2=False)
    s, _ = step(p, c, rows, 300)
    assert s['track2_trigger']['accepted'] == dict(m1=True, y1=False)
    assert 'early_merge_join' not in families_offered(s, 'y1')        # t-1 rule: not yet visible
    assert s['g5_lane_requests'] == dict(m1=1)
    assert s['g5_lane_gate'] == dict(safe=dict(m1=1), blocked={})
    assert c.g5_lane_targets == dict(m1=1) and c.lane_actors() == frozenset({'m1'})
    assert c.lane_request(env_for(rows)).tolist() == [[1., -1., -1.]]
    # 301: m1 completed; m2 (merge) and y1 (yield) receive and join on the event tick.
    rows = early_rows(301, m1_lane=1)
    s, _ = step(p, c, rows, 301)
    stats = s['track2_trigger']['proposals']['nm_families']
    assert stats['event'] and stats['event_cavs'] == ['m2', 'y1'] and stats['accepted_join'] == 2
    assert p.nm_families.membership == dict(m1='early_merge:m1@300', m2='early_merge:m1@300',
                                            y1='early_merge:m1@300')
    commands = s['track2']['nm_families']['commands']
    assert commands['m2']['status'] == 'sequence_wait'                # m1 still ahead in lane 2 at t-1
    assert commands['m1']['status'] == 'lane_done'
    assert commands['y1']['status'] == 'yield' and s['context']['move_offsets']['y1'] == -.6
    assert 'g5_lane_requests' not in s and c.g5_lane_targets == {}
    # 302: m1 completion seen on its own t-1 packet; registered 2 s sequence gap not yet over.
    s, _ = step(p, c, early_rows(302, m1_lane=1), 302)
    assert s['track2']['nm_families']['commands']['m2']['status'] == 'sequence_gap_wait'
    assert p.nm_families.sessions['early_merge:m1@300']['completed'] == dict(m1=301)
    # 303: m2's turn; the request passes the SAME G5 gate (own sensed gaps) and actuator ABI.
    rows = early_rows(303, m1_lane=1)
    s, _ = step(p, c, rows, 303)
    assert s['g5_lane_requests'] == dict(m2=1) and c.g5_lane_targets == dict(m2=1)
    assert c.lane_request(env_for(rows)).tolist() == [[-1., -1., -1., 1.]]
    assert c.lane_actors() == frozenset({'m2'})
    c.observe_lane(dict(lane_commands=dict(issued=1, completed=1),
                        lane_vehicles=dict(m2=dict(target=1, status='issued', completed=True))))
    assert c.events[-1]['g5_lane_outcome']['vehicles'] == dict(m2=dict(target=1, status='issued',
                                                                         completed=True))
    assert p.nm_families.sessions['early_merge:m1@300']['attempts'] == dict(m1=1, m2=1)


def test_g5_gate_refuses_an_unsafe_family_lane_change(cfg, fixed):
    p = make(fam_cfg(cfg, dict(early_merge=2)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('early_merge', 'merge'))
    c = controller(p)
    rows = early_rows(300, with_m2=False)+[row('h2', 41, 1, cav=False)]   # HDV alongside in the target lane
    s, _ = step(p, c, rows, 300)
    assert s['g5_lane_requests'] == dict(m1=1)
    assert s['g5_lane_gate']['safe'] == {} and s['g5_lane_gate']['blocked'] == dict(m1='unsafe_gap_or_edge')
    assert c.lane_request(env_for(rows)) is None and c.lane_actors() == frozenset()
    # Attempts are bounded by the registered attempt_s (seconds of requests).
    for now in range(301, 306):
        s, _ = step(p, c, early_rows(now, with_m2=False)+[row('h2', 41+2*(now-300), 1, cav=False)], now)
    assert p.nm_families.sessions['early_merge:m1@300']['attempts'] == dict(m1=5)
    assert s['track2']['nm_families']['commands']['m1']['status'] == 'attempts_exhausted'
    assert 'g5_lane_requests' not in s


# ------------------------------------------------------------------ the other lane families

def test_lane_balance_front_first_to_the_far_lane(cfg, fixed):
    p = make(fam_cfg(cfg, dict(lane_balance=3)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('lane_balance', 'mover'))
    c = controller(p)

    def rows(t, b1_lane=1):
        dt = t-300
        out = [row('b1', -500+2*dt, b1_lane), row('h1', -560+2*dt, 1, cav=False)]
        return out+([row('b2', -520+2*dt, 1)] if t > 300 else [])
    s, _ = step(p, c, rows(300), 300)
    assert s['g5_lane_requests'] == dict(b1=0) and c.g5_lane_targets == dict(b1=0)
    s, _ = step(p, c, rows(301, b1_lane=0), 301)
    assert s['track2_trigger']['accepted']['b2'] is True
    assert s['track2']['nm_families']['commands']['b2']['status'] == 'sequence_wait'
    s, _ = step(p, c, rows(302, b1_lane=0), 302)
    assert s['track2']['nm_families']['commands']['b2']['status'] == 'sequence_gap_wait'
    s, _ = step(p, c, rows(303, b1_lane=0), 303)
    assert s['g5_lane_requests'] == dict(b2=0) and c.g5_lane_targets == dict(b2=0)
    assert all(v == 0. for v in s['context']['move_offsets'].values())    # lane-only family


def test_zipper_order_alternation_ease_then_merge_in_the_zone(cfg, fixed):
    p = make(fam_cfg(cfg, dict(zipper_order=2)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('zipper_order', 'through'))
    c = controller(p)

    def rows(t, c1):
        return [row('t1', 15+10*(t-300), 1, v=10.), row('c1', c1, 2)]
    s, _ = step(p, c, rows(300, 12), 300)                     # through member initiates (through_first)
    assert s['track2_trigger']['accepted'] == dict(c1=False, t1=True)
    s, _ = step(p, c, rows(301, 22), 301)
    assert s['track2_trigger']['accepted']['c1'] is True
    commands = s['track2']['nm_families']['commands']
    # c1 is not yet behind its designated predecessor t1 by s0 + h*v: it eases, no lane request.
    assert commands['c1']['status'] == 'order_ease' and s['context']['move_offsets']['c1'] == -.6
    assert commands['t1']['status'] == 'in_order' and 'g5_lane_requests' not in s
    s, _ = step(p, c, rows(302, 24), 302)                     # behind t1, still upstream of the zone
    assert s['track2']['nm_families']['commands']['c1']['status'] == 'in_order'
    assert 'g5_lane_requests' not in s
    rows3 = rows(303, 26)
    s, _ = step(p, c, rows3, 303)                             # inside the 50 m zipper zone, in order
    assert s['track2']['nm_families']['commands']['c1']['status'] == 'lane_request'
    assert s['g5_lane_requests'] == dict(c1=1) and c.g5_lane_targets == dict(c1=1)
    assert c.lane_request(env_for(rows3)).tolist() == [[-1., 1.]]


def test_zipper_role_cap_alternation_and_closing_first(cfg, fixed, monkeypatch):
    FIXED_ZIP = dict(FIXED['zipper_order'], order='closing_first')
    monkeypatch.setitem(FIXED, 'zipper_order', FIXED_ZIP)
    geometry = nm_map.lane_drop('merge')
    proto = nf.Protocol(nf.settings(fam_cfg({}, dict(zipper_order=4))))
    proto.receive(300, {'c1': row('c1', 10, 2), 't1': row('t1', 12, 1)}, {'c1': [], 't1': []}, {})

    class Exec:
        def start(self, *a, **k):
            return True
    option = nf.initiate('zipper_order', row('c1', 10, 2), [], geometry, now=300, cap=4, range_m=100.,
                         rng=np.random.default_rng(0))
    assert option.role == 'closing' and option.param('order') == 'closing_first'
    assert proto.start(tb.TriggerMove(option), ego='c1', executor=Exec(), observable_cav_ids=['c1'], now=300)
    session = proto.sessions[option.session_id]
    join = nf.FamilyOption('zipper_order_join', ('c2',), 0, 40, .6, option.session_id, 'c1', 'closing',
                           option.params, member_count=1, cap=4)
    proto.rows['c2'] = row('c2', 10, 2)
    assert proto.start(tb.TriggerMove(join), ego='c2', executor=Exec(), observable_cav_ids=['c2'], now=300)
    proto.rows['c3'] = row('c3', 0, 2)
    third = nf.FamilyOption('zipper_order_join', ('c3',), 0, 40, .6, option.session_id, 'c1', 'closing',
                            option.params, member_count=2, cap=4)
    assert not proto.start(tb.TriggerMove(third), ego='c3', executor=Exec(), observable_cav_ids=['c3'], now=300)
    assert set(session['members']) == {'c1', 'c2'}


# ------------------------------------------------------------------ absolute-law families: single writer

def test_platoon_cacc_is_the_issued_request_via_the_single_writer(cfg, fixed):
    p = make(fam_cfg(cfg, dict(platoon_headway=4)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('platoon_headway', 'head'))
    c = controller(p)
    native = dict(p1=.3, p2=-.2)

    def rows(t, v1=20., cut_in=False):
        dt = t-300
        out = [row('p1', -300+20*dt, 1, v=v1), row('p2', -330+20*dt, 1, v=20.)]
        return out+([row('hx', -315+20*dt, 1, cav=False, v=20.)] if cut_in else [])
    s, _ = step(p, c, rows(300), 300, native)
    assert s['track2_trigger']['accepted'] == dict(p1=True, p2=False)    # head initiates
    s, req = step(p, c, rows(301), 301, native)
    assert s['track2_trigger']['accepted']['p2'] is True                 # follower joins on receipt
    gap, h = 30.-5., FIXED['platoon_headway']['time_headway_s']
    spec = nf.SPECS['platoon_headway']
    k = {key: spec.constant(key) for key in ('standstill_gap_m', 'kp', 'kd', 'kff')}
    target = nf.cacc_accel(gap, 20., 20., 20., time_headway_s=h, **k)
    commands = s['track2']['nm_families']['commands']
    assert commands['p2']['status'] == 'cacc' and commands['p2']['target'] == pytest.approx(target)
    assert commands['p1']['status'] == 'no_member_predecessor' and s['context']['move_offsets']['p1'] == 0.
    assert req['p2']['requested_raw'] == pytest.approx(target, abs=1e-5)  # native + base + offset
    # V2V feed-forward: the predecessor's own t-1 packet speed vs its sensed speed now.
    s, req = step(p, c, rows(302, v1=18.), 302, native)
    target = nf.cacc_accel(gap, 20., 18., 20., time_headway_s=h, **k)
    assert req['p2']['requested_raw'] == pytest.approx(target, abs=1e-5)
    # A cut-in HDV is not a platoon member: the follower returns to the base policy.
    s, _ = step(p, c, rows(303, cut_in=True), 303, native)
    assert s['track2']['nm_families']['commands']['p2']['status'] == 'no_member_predecessor'
    assert s['context']['move_offsets']['p2'] == 0. and 'hx' not in s['track2']['nm_families']['commands']


def test_platoon_without_native_command_does_not_guess(cfg, fixed):
    p = make(fam_cfg(cfg, dict(platoon_headway=2)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('platoon_headway', 'head'))
    rows = lambda dt: [row('p1', -300+20*dt, 1, v=20.), row('p2', -330+20*dt, 1, v=20.)]  # noqa: E731
    for t in (300, 301):
        p.observe(rows(t-300), t)
        context = encode(rows(t-300))
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['track2']['nm_families']['commands']['p2']['status'] == 'native_unavailable'
    assert s['context']['move_offsets']['p2'] == 0.


def test_discharge_sync_common_start_and_gap_bounded_acceleration(cfg, fixed):
    p = make(fam_cfg(cfg, dict(discharge_sync=4)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('discharge_sync', 'sync'))
    c = controller(p)
    native = dict(d1=0., d2=0.)

    def rows(t, v=1.):
        dt = t-300
        return [row('d1', 60+v*dt, 1, v=v), row('d2', 50+v*dt, 1, v=v),
                row('hq', 80+5*dt, 1, cav=False, v=5.)]
    s, _ = step(p, c, rows(300), 300, native)
    assert s['track2_trigger']['accepted'] == dict(d1=True, d2=False)   # only the queue front initiates
    s, _ = step(p, c, rows(301), 301, native)
    assert s['track2_trigger']['accepted']['d2'] is True
    assert {v: m['status'] for v, m in s['track2']['nm_families']['commands'].items()} == dict(
        d1='outside_window', d2='outside_window')                       # common start at 300 + lead 2 s
    s, req = step(p, c, rows(302), 302, native)
    spec = nf.SPECS['discharge_sync']
    law = {k: spec.constant(k) for k in ('time_headway_s', 'standstill_gap_m', 'kp', 'kd')}
    r = {x['vid']: x for x in rows(302)}
    want = {}
    for vid, lead in (('d1', 'hq'), ('d2', 'd1')):
        gap = r[lead]['s_m']-r[vid]['s_m']-5.
        want[vid] = min(.6, nf.cacc_accel(gap, 1., r[lead]['v_mps'], None, kff=0., **law))
        assert req[vid]['requested_raw'] == pytest.approx(want[vid], abs=1e-5)
    assert want['d2'] < .6                                              # bounded by its own gap law
    s, _ = step(p, c, rows(303, v=15.), 303, native)                    # recovered: released
    assert {m['status'] for m in s['track2']['nm_families']['commands'].values()} == {'released'}


def test_harmonize_rows_slots_tracking_and_release(cfg, fixed):
    p = make(fam_cfg(cfg, dict(harmonize_rows=6)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('harmonize_rows', 'row0'))
    c = controller(p)
    native = {}

    def rows(t, extra=True, r1=-1000.):
        dt = t-300
        out = [row('r1', r1+20*dt, 1, v=20.)]
        if extra:
            out += [row('r2', -1005+20*dt, 0, v=20.), row('r3', -1060+20*dt, 1, v=20.),
                    row('r4', -1003+20*dt, 1, v=20.), row('hh', -1002+20*dt, 0, cav=False, v=20.)]
        return out
    s, _ = step(p, c, rows(300, extra=False), 300, native)
    assert s['track2_trigger']['accepted'] == dict(r1=True)
    s, req = step(p, c, rows(301), 301, native)
    offers = {v: [(o.role, o.slot) for o in opts] for v, opts in p.nm_families.offers.items()}
    assert offers == dict(r2=[('row0', 0)], r3=[('row1', 1)])          # r4: (row0, lane 1) slot taken
    assert {v for v, a in s['track2_trigger']['accepted'].items() if a} == {'r2', 'r3'}
    for vid in ('r1', 'r2', 'r3'):
        assert req[vid]['requested_raw'] == pytest.approx(-.6, abs=1e-5)  # clip((18-20)/1, -.6, .6)
    assert 'hh' not in s['track2']['nm_families']['commands'] and 'r4' not in s['track2']['nm_families']['commands']
    s, _ = step(p, c, rows(302, r1=-340.), 302, native)                  # initiator inside the release distance
    assert s['track2']['nm_families']['commands']['r1']['status'] == 'released'
    assert s['track2']['nm_families']['published'][0]['released'] is True


# ------------------------------------------------------------------ ownership / HDVs / range

def test_ownership_cap_and_hdvs_never_offered_or_commanded(cfg, fixed):
    p = make(fam_cfg(cfg, dict(early_merge=2)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('early_merge', 'merge'))
    busy = tb.TriggerMove(moves.Move('ease_off', ('o1',), 0, 30, .5))
    rows0 = early_rows(300, with_m2=False)+[row('o1', 30, 2), row('far', -300, 1)]
    p.observe(rows0, 300)
    assert p.executor.start(busy, ego='o1', observable_cav_ids=p.visible['o1'], now=300, until=330)
    p.last_decision = None
    c = controller(p)
    s, _ = step(p, c, rows0, 300)
    assert families_offered(s, 'o1') == []                               # owned CAV: none only
    assert s['track2_trigger']['accepted']['m1'] is True
    rows1 = early_rows(301, m1_lane=2)+[row('o1', 32, 2), row('far', -298, 1)]
    s, _ = step(p, c, rows1, 301)
    stats = s['track2_trigger']['proposals']['nm_families']
    assert stats['event_cavs'] == ['m2', 'y1']                           # o1 owned; far out of radio range
    assert stats['selected_join'] == 2 and stats['accepted_join'] == 1   # registered cap 2
    assert s['track2_trigger']['accepted'] == dict(far=False, m1=False, m2=True, o1=False, y1=False)
    assert 'h1' not in p.nm_families.offers and 'h1' not in p.nm_families.rows
    # One session per CAV: a member cannot initiate or join again.
    again = nf.initiate('early_merge', p.nm_families.rows['m2'], [], p.nm_families.geometry, now=301,
                        cap=2, range_m=100., rng=np.random.default_rng(1))
    assert not p.nm_families.start(tb.TriggerMove(again), ego='m2', executor=p.executor,
                                   observable_cav_ids=p.visible['m2'], now=301)
    step(p, c, early_rows(302, m1_lane=2)+[row('o1', 34, 2), row('far', -296, 1)], 302)
    # The t-1 message acknowledges m2 (joined at 301) from 302 on: the full session is no longer offered.
    s, _ = step(p, c, early_rows(303, m1_lane=2)+[row('o1', 36, 2), row('far', -294, 1)], 303)
    assert s['track2']['nm_families']['published'][0]['roster'] == [['m1', 'merge', None], ['m2', 'merge', None]]
    assert not any(o.session_id == 'early_merge:m1@300' for opts in p.nm_families.offers.values() for o in opts)
    for vid, command in s['track2']['nm_families']['commands'].items():
        assert vid in ('m1', 'm2')
    assert set(s.get('g5_lane_requests', {})) <= {'m1', 'm2'}


def test_controller_reserved_member_gets_no_offset_or_lane(cfg, fixed):
    p = make(fam_cfg(cfg, dict(early_merge=2)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('early_merge', 'merge'))
    rows = early_rows(300, with_m2=False)
    p.observe(rows, 300)
    p.command_reservations = dict(m1=dict(m1=999), y1={})
    context = encode(rows)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    p.executor.start(tb.TriggerMove(nf.initiate('early_merge', rows[0], [], p.nm_families.geometry, now=300,
                     cap=2, range_m=100., rng=np.random.default_rng(0))), ego='m1',
                     observable_cav_ids=['m1'], now=300, until=340)
    p.nm_families.sessions['early_merge:m1@300'] = dict(
        family='early_merge', initiator='m1', start_s=300, onset=0, hold=40, depth=.6,
        params=tuple(sorted(FIXED['early_merge'].items())), cap=2, anchor_lane=2,
        members=dict(m1=dict(t=300, role='merge', slot=None, target=1)), released=False,
        completed={}, attempts={}, latched=set())
    p.nm_families.membership['m1'] = 'early_merge:m1@300'
    p.last_decision = 300
    s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['track2']['nm_families']['commands']['m1']['status'] == 'controller_reserved'
    assert 'g5_lane_requests' not in s


def test_join_only_within_radio_range_and_after_one_second(cfg, fixed):
    p = make(fam_cfg(cfg, dict(lane_balance=3), range_m=50.))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('lane_balance', 'mover'))
    rows = lambda t: [row('b1', -500+2*(t-300), 1), row('b2', -540+2*(t-300), 1),  # noqa: E731
                      row('b3', -580+2*(t-300), 1)]
    p.observe(rows(300)[:1], 300)
    context = encode(rows(300)[:1])
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    p.sample(context, torch.Generator().manual_seed(1), True)
    assert p.nm_families.offers == {}
    p.observe(rows(301), 301)
    assert set(p.nm_families.offers) == {'b2'}                            # b3 is 78 m away > 50 m


# ------------------------------------------------------------------ B = R

def test_b_and_r_run_the_identical_protocol(cfg, fixed):
    out = {}
    for arm in ('B', 'R'):
        p = make(fam_cfg(cfg, dict(early_merge=4, platoon_headway=4, lane_balance=2)), arm)
        p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('early_merge', 'merge')
                                        + initiate_then_join('platoon_headway', 'head'))
        c = controller(p)
        events = []
        for t in range(300, 305):
            rows = early_rows(t, m1_lane=2 if t == 300 else 1)+[
                row('p1', -400+20*(t-300), 0, v=20.), row('p2', -430+20*(t-300), 0, v=20.)]
            s, req = step(p, c, rows, t, dict(p1=.1, p2=-.1))
            assert s['track2'].pop('arm') == arm
            events.append([s, req, c.g5_lane_targets])
        out[arm] = canon(events)+canon(p.nm_families.membership)
    assert out['B'] == out['R']
    assert '"early_merge_join"' in out['B'] and '"platoon_headway_initiate"' in out['B']


# ------------------------------------------------------------------ trigger inputs / PPO replay

def test_trigger_columns_zero_initialised_and_composite_with_w1(cfg):
    base_cfg = dict(cfg, trigger_partner_slots=4, group_lane_moves=True)
    on_cfg = fam_cfg(cfg, dict(early_merge=2, harmonize_rows=3))
    off, on = make(base_cfg), make(on_cfg)
    t_off, t_on = off.trigger.trigger_head[0].weight, on.trigger.trigger_head[0].weight
    assert t_on.shape[1] == t_off.shape[1]+len(nf.DESCRIPTOR)+len(nf.GEOMETRY)
    assert torch.equal(t_on[:, :t_off.shape[1]], t_off) and not bool(t_on[:, t_off.shape[1]:].any())
    assert all(torch.equal(v, on.layer.state_dict()[k]) for k, v in off.layer.state_dict().items())
    both = make(dict(on_cfg, **v2v_wall.config_keys(provenance=PROV, member_cap=3)))
    w1 = make(dict(base_cfg, **v2v_wall.config_keys(provenance=PROV, member_cap=3)))
    ext = both.trigger.extension
    assert isinstance(ext, nf.Composite) and ext.wall is not None and ext.nm is not None
    assert both.trigger.option_families == w1.trigger.option_families+tuple(
        f'{n}_{k}' for n in ('early_merge', 'harmonize_rows') for k in nf.KINDS)
    assert both.trigger.descriptor_width == w1.trigger.descriptor_width+len(nf.DESCRIPTOR)
    rows = early_rows(300)
    for p in (both, w1):
        p.observe(rows, 300)
        context = encode(rows)
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
        assert 'track2_trigger' in s
    block = nf.geometry_block(encode(rows)['x'][0], nm_map.lane_drop('merge'))
    assert block == pytest.approx([(75.44-40)/2000, 1., 0., 0., 2./30])   # m1: own row only


def test_recorded_family_decisions_replay_for_trigger_ppo(cfg, fixed):
    p = make(fam_cfg(cfg, dict(early_merge=4)))
    p.trigger.trigger_head = Prefer(p.trigger, initiate_then_join('early_merge', 'merge'))
    c = controller(p)
    step(p, c, early_rows(300, with_m2=False), 300)
    s, _ = step(p, c, early_rows(301, m1_lane=1), 301)
    record = json.loads(json.dumps(s['track2_trigger']))
    batch = tb._batch(record)
    assert any(isinstance(m, nf.FamilyOption) for table in batch.tables for m in table)
    logp, _, _ = p.trigger.evaluate(batch)
    assert logp.tolist() == pytest.approx(record['logp'])
    from . import sparse_synergy_credit as g4
    assert g4.opportunities([dict(id='ep', steps=[dict(time_s=301, sample=s)])]) == []


# ------------------------------------------------------------------ executable fields (G5 convention)

def em_move(t0=300):
    return dict(schema=nf.EXECUTABLE_SCHEMA, rule=nf.RULE, family='early_merge', map='merge',
                initiator='m1', members=['m1', 'm2', 'y1'], onset=0, hold=40, depth=.6,
                params=dict(FIXED['early_merge']), cap=4, anchor_lane=2,
                roles=dict(m1='merge', m2='merge', y1='yield'), slots=dict(m1=None, m2=None, y1=None),
                join_s=dict(m1=0, m2=1, y1=1))


def em_plan(duration=40):
    move = em_move()
    base = [[.01*i+.001*t for t in range(duration)] for i in range(3)]
    executed = [[base[i][t]+(-.6 if i == 2 and 1 <= t < 4 else 0.) for t in range(duration)] for i in range(3)]
    lanes = [[1 if t == 0 else -1 for t in range(duration)], [1 if t == 3 else -1 for t in range(duration)],
             [-1]*duration]
    return nf.executable_fields(move, duration, policy_profile=base, profile=executed, lane_targets=lanes)


@pytest.mark.parametrize('mutation', ['ok', 'profile', 'perturbation', 'lane_wrong_target', 'lane_yield',
                                      'lane_before_join', 'attempts', 'role', 'cap', 'params', 'knots',
                                      'uses_flag', 'order'])
def test_executable_fields_bind_exactly(mutation):
    plan = em_plan()
    if mutation == 'ok':
        assert nf.from_plan(plan)['family'] == 'early_merge'
        assert nf.lane_requests(plan, 0) == dict(m1=1) and nf.lane_requests(plan, 3) == dict(m2=1)
        with pytest.raises(ValueError, match='nm_move'):
            moves.from_plan(plan)                                       # never projected to a box move
        return
    if mutation == 'profile':
        plan['profile'][0][5] += .1
    elif mutation == 'perturbation':
        plan['perturbation_profile'][2][1] = 0.
    elif mutation == 'lane_wrong_target':
        plan['lane_targets'][0][0] = 0
    elif mutation == 'lane_yield':
        plan['lane_targets'][2][5] = 1
    elif mutation == 'lane_before_join':
        plan['lane_targets'][1][0] = 1
    elif mutation == 'attempts':
        plan['lane_targets'][1] = [1]*10+[-1]*30
    elif mutation == 'role':
        plan['nm_move']['roles']['y1'] = 'head'
    elif mutation == 'cap':
        plan['nm_move']['cap'] = 2
    elif mutation == 'params':
        plan['nm_move']['params']['upstream_m'] = 5000.
    elif mutation == 'knots':
        plan['knots'] = [[0., 0.]]*3
    elif mutation == 'uses_flag':
        plan['uses_lane_actions'] = False
    else:
        plan['members'] = ['m2', 'm1', 'y1']
    with pytest.raises(ValueError):
        nf.from_plan(plan)


def test_plan_replay_lane_requests_use_the_g5_gate(cfg):
    p = make(fam_cfg(cfg, dict(early_merge=4)))
    c = controller(p)
    plan = dict(em_plan(), id='N9_EXEC', family='profile')
    c.active_plans = [dict(plan=plan, start_s=300)]
    rows = early_rows(300)
    p.observe(rows, 300)
    p.last_decision = 300
    c.apply(rows, {r['vid']: 0. for r in rows if r['is_cav']}, 300)
    assert c.events[-1]['sample']['absolute_accels'] == dict(m1=0., m2=.01, y1=.02)
    assert c.g5_lane_targets == dict(m1=1) and c.lane_actors() == frozenset({'m1'})
    c.g5_lane_runtime = False
    with pytest.raises(ValueError, match='single-writer'):
        c.apply(rows, {r['vid']: 0. for r in rows if r['is_cav']}, 301)


def test_verify_tape_requires_executed_profile_and_lane_effect():
    plan = em_plan(duration=40)

    def tape(m2_done=True):
        steps = []
        for t in range(40):
            lanes = dict(m1=2 if t == 0 else 1, m2=2 if t <= 3 or not m2_done else 1, y1=1)
            sample = dict(absolute_accels={v: plan['profile'][i][t] for i, v in enumerate(plan['members'])})
            step_ = dict(time_s=300+t, sample=sample,
                         rows=[dict(vid=v, lane=lanes[v], is_cav=True) for v in plan['members']])
            requests = nf.lane_requests(plan, t)
            if requests:
                sample.update(g5_lane_requests=requests, g5_lane_gate=dict(safe=requests, blocked={}))
                step_['g5_lane_outcome'] = dict(vehicles={v: dict(target=g, status='issued', completed=True)
                                                          for v, g in requests.items()})
            steps.append(step_)
        steps.append(dict(time_s=340, sample={}, rows=steps[-1]['rows']))
        return dict(steps=steps)
    assert nf.verify_tape(plan, tape(), 300)['family'] == 'early_merge'
    with pytest.raises(ValueError, match='lane effect'):
        nf.verify_tape(plan, tape(m2_done=False), 300)
    bad = tape()
    bad['steps'][7]['sample']['absolute_accels']['y1'] += .1
    with pytest.raises(ValueError, match='profile'):
        nf.verify_tape(plan, bad, 300)


# ------------------------------------------------------------------ bank ingestion (initiate/join labels)

def em_frames(t0=300, seconds=4):
    return {t: dict(vehicles=early_rows(t, m1_lane=2 if t <= t0 else 1)) for t in range(t0-1, t0+seconds)}


def em_pairs(frames, t0=300):
    pairs = []
    for t in sorted(frames):
        if t < t0:
            continue
        context = encode(frames[t]['vehicles'])
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        pairs.append((dict(context=context, intent_link=dict(t=t)), 1.))
    return pairs


def test_verified_family_example_is_taught_through_initiate_and_join(cfg):
    frames = em_frames()
    plan = em_plan()
    p = make(fam_cfg(cfg, dict(early_merge=4)))
    got = list(breadth.observable_contexts(p, plan, em_pairs(frames), frames, t0=300))
    labels = [(o.family, o.members[0], o.role, t) for _, o, _, _, t in got]
    assert labels == [('early_merge_initiate', 'm1', 'merge', 300), ('early_merge_join', 'm2', 'merge', 301),
                      ('early_merge_join', 'y1', 'yield', 301)]
    for context, option, reader, _, _ in got:
        assert context['members'] == [option.members[0]]
        assert option.params == tuple(sorted(FIXED['early_merge'].items()))
    off = make(dict(cfg, trigger_partner_slots=4, group_lane_moves=True))
    with pytest.raises(ValueError, match='nm_families enabled'):
        list(breadth.observable_contexts(off, plan, em_pairs(frames), frames, t0=300))


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_bank_supervision_imitates_family_options_identically_for_b_and_r(cfg, monkeypatch, arm):
    from . import bank_v2v, launch_production as launch
    frames = em_frames()
    plan = em_plan()
    on = fam_cfg(cfg, dict(early_merge=4))

    def visit(config, requested_arm, consume):
        consume(cfg=config, receipt=dict(verification_id='nm_989201_t300', t0=300),
                example=dict(example_id='nm_989201_t300#s1'), plan=deepcopy(plan),
                pairs=em_pairs(frames), trace='CPU')
    monkeypatch.setattr(breadth, 'visit', visit)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path: frames)
    results = {}
    for value in ('B', 'R'):
        policy = make(on, value)
        results[value] = launch.bank_move_supervision(SimpleNamespace(cfg=on, policy=policy, arm=value))
    result = results[arm]
    assert result['family_counts'] == {'early_merge_initiate': 1, 'early_merge_join': 2}
    assert result['final_ce'] < result['initial_ce']
    strip = lambda r: canon({k: v for k, v in r.items() if k != 'passes'})  # noqa: E731
    assert strip(results['B']) == strip(results['R'])
