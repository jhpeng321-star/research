"""W1 V2V wall initiate/join protocol + wider coop awareness: CPU contracts with fakes.

Real EpisodePolicy / TriggerPolicy / MoveExecutor / bank supervision code paths; the
only fakes are a test-only trigger scoring head, fixed variety draws and bank visitors.
No simulator, GPU or network.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import bank_breadth as breadth, launch_production as launch, moves
from . import track2_backend as tb, v2v_wall
from .merge_features import MERGE_FEATURES
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)
from .v2v_intent import IntentPolicy
from tools.coopnative.policy import encode

PROV = 'W1 CPU fixture (owner 2026-11-06 wall V2V build)'
REPO = Path(__file__).resolve().parents[2]
BASE_COMMIT = '3e5bc9f9'


def wall_cfg(cfg, cap=6, range_m=None, **aware):
    out = dict(cfg, trigger_partner_slots=4)
    out.update(v2v_wall.config_keys(member_cap=cap, provenance=PROV, range_m=range_m, **aware))
    return out


def make(cfg, arm='B'):
    torch.manual_seed(7)
    return tb.EpisodePolicy(IntentPolicy(), cfg, arm)


def row(vid, s, lane, cav=True, v=2.):
    return dict(vid=vid, s_m=float(s), v_mps=v, lane=lane, is_cav=cav, length_m=5.)


def scene(dt=0, extra=()):
    """a initiates (no lane-0 rear); b 70 m away (in range); c 150 m away; h1 HDV."""
    out = [row('a', -400, 0), row('b', -330, 1), row('c', -250, 0), row('h1', -300, 0, cav=False)]
    out += list(extra)
    return [dict(r, s_m=r['s_m']+2.*dt) for r in out]


def tick(p, vehicles, now, deterministic=True, seed=1):
    p.observe(vehicles, now)
    context = encode(vehicles)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    sample, _ = p.sample(context, torch.Generator().manual_seed(seed), deterministic)
    return sample


class Prefer(nn.Module):
    """Test-only trigger head: initiate when the ego's own lane-0 rear is unobserved, join always."""

    def __init__(self, trigger, initiate=50., join=40.):
        super().__init__()
        hidden = trigger.base_policy.actor[-1].in_features
        base = len(trigger.families)+trigger.partner_slots+3
        self.init_col, self.join_col = hidden+base, hidden+base+1
        self.rear_col = hidden+trigger.descriptor_width+MERGE_FEATURES.index('lane_0_rear_missing')
        self.initiate, self.join = initiate, join

    def forward(self, x):
        return (self.initiate*x[..., self.init_col]*x[..., self.rear_col]
                + self.join*x[..., self.join_col])[..., None]


@pytest.fixture
def fixed_draws(monkeypatch):
    monkeypatch.setattr(moves, '_sampled', lambda rng, onset=None: (2, 40, .6))


def families(sample, ego):
    trig = sample['track2_trigger']
    i = trig['context']['members'].index(ego)
    return [m and m['family'] for m in trig['tables'][i] if m]


def canon(value):
    return json.dumps(value, sort_keys=True, default=repr)


# ------------------------------------------------------------------ settings

def test_settings_default_off_and_registered_values(cfg):
    assert v2v_wall.settings(cfg) is None and v2v_wall.awareness_settings(cfg) is None
    assert v2v_wall.extension(cfg) is None
    assert v2v_wall.settings(dict(cfg, v2v_wall=False)) is None
    on = wall_cfg(cfg, cap=6, awareness_slots=6, awareness_range_m=250.)
    assert v2v_wall.settings(on)['member_cap'] == 6
    assert v2v_wall.awareness_settings(on)['slots'] == 6
    bad = [dict(on, v2v_wall=dict(on['v2v_wall'], member_cap=7)),
           dict(on, v2v_wall=dict(on['v2v_wall'], member_cap=1)),
           dict(on, v2v_wall=dict(on['v2v_wall'], provenance=' ')),
           dict(on, v2v_wall=dict(on['v2v_wall'], rule='other')),
           dict(on, v2v_wall=dict(on['v2v_wall'], range_m=float('inf'))),
           dict(on, coop_awareness=dict(on['coop_awareness'], slots=5)),
           dict(on, coop_awareness=dict(on['coop_awareness'], range_m=99.)),
           dict(on, coop_awareness=dict(on['coop_awareness'], extra=1))]
    for value in bad:
        with pytest.raises(ValueError):
            v2v_wall.settings(value)
            v2v_wall.awareness_settings(value)
    with pytest.raises(ValueError):
        tb._check(dict(bad[0]), 'B')


# ------------------------------------------------------------------ off = byte-identical

def test_off_keys_absent_or_false_are_identical(cfg):
    outs = []
    for value in (cfg, dict(cfg, v2v_wall=False, coop_awareness=False)):
        p = make(value)
        samples = [tick(p, scene(t-300), t, deterministic=False, seed=t) for t in range(300, 304)]
        outs.append(canon(samples) + canon({k: v.tolist() for k, v in p.worker_state()['layer'].items()})
                    + canon({k: v.tolist() for k, v in p.worker_state()['trigger'].items()}))
        assert p.wall is None and p.awareness is None and p.trigger.extension is None
        assert 'v2v_wall' not in p.local_states['a'] and 'v2v_wall' not in samples[0]['track2']
    assert outs[0] == outs[1]


SCENARIO = r'''
import hashlib, json, sys, torch
sys.path.insert(0, sys.argv[1])
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
    out = [('a', -400., 0, True), ('b', -330., 1, True), ('c', -250., 0, True), ('h1', -300., 0, False)]
    return [dict(vid=v, s_m=s+2.*dt, v_mps=2., lane=l, is_cav=c, length_m=5.) for v, s, l, c in out]
for now in range(300, 306):
    r = rows(now-300)
    p.observe(r, now)
    controller.apply(r, {x['vid']: 0. for x in r if x['is_cav']}, now)
digest = hashlib.sha256()
digest.update(json.dumps([e['sample'] for e in controller.events], sort_keys=True, default=repr).encode())
for part in ('layer', 'trigger'):
    for k, v in sorted(p.worker_state()[part].items()):
        digest.update(k.encode()); digest.update(v.detach().numpy().tobytes())
print(digest.hexdigest())
'''


def test_off_is_byte_identical_to_the_base_commit(cfg, tmp_path):
    """Default-off samples and initial weights equal the pre-W1 source (3e5bc9f9)."""
    touched = ('track2_backend.py', 'trigger_policy.py', 'feasible_proposals.py', 'merge_features.py',
               'bank_breadth.py', 'sparse_synergy_credit.py')
    overlay = tmp_path/'overlay'
    overlay.mkdir()
    for name in touched:
        try:
            text = subprocess.run(['git', '-C', str(REPO), 'show', f'{BASE_COMMIT}:tools/coopmwm/{name}'],
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip('base commit source unavailable in this checkout')
        (overlay/name).write_bytes(text)
    value = json.dumps(dict(cfg, run_root=str(tmp_path)))
    digests = []
    for path in ('-', str(overlay)):
        run = subprocess.run([sys.executable, '-c', SCENARIO, str(REPO), path, value], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]
        digests.append(run.stdout.strip())
    assert len(digests[0]) == 64 and digests[0] == digests[1]


# ------------------------------------------------------------------ protocol timing / range / single writer

def test_message_visible_only_at_t_plus_one_within_range_and_join_executes(cfg, fixed_draws):
    p = make(wall_cfg(cfg))
    p.trigger.trigger_head = Prefer(p.trigger)
    s = tick(p, scene(0), 300)
    assert s['track2_trigger']['accepted'] == dict(a=True, b=False, c=False)
    assert 'wall_join' not in families(s, 'b')                 # same-tick: not yet visible
    published = s['track2']['v2v_wall']['published']
    assert [(m['wall_id'], m['timestamp_s'], m['member_count']) for m in published] == [('a@300', 300, 1)]
    assert p.wall.offers == {}
    s = tick(p, scene(1), 301)
    stats = s['track2_trigger']['proposals']['v2v_wall']
    assert stats['event'] and stats['event_cavs'] == ['b']
    assert set(p.wall.offers) == {'b'}                         # c is 150 m away, h1 is an HDV
    assert families(s, 'b') and set(families(s, 'b')) == {'wall_join'}
    assert families(s, 'c') == [] and families(s, 'a') == []  # non-event / owned: none only
    assert s['track2_trigger']['accepted'] == dict(a=False, b=True, c=False)
    offer = p.wall.offers['b'][0]
    assert (offer.onset, offer.hold, offer.gap_m, offer.member_count) == (1, 40, -400.-(-328.), 1)
    assert stats['accepted_join'] == 1 and stats['selected_initiate'] == 0
    s = tick(p, scene(2), 302)
    # Same absolute window [302, 342) for initiator and joiner via the committed offsets.
    assert s['context']['move_offsets'] == dict(a=-.6, b=-.6, c=0.)
    expected, _ = p.layer.base.sample(s['context'], torch.Generator().manual_seed(1), True, {})
    offsets = torch.tensor([s['context']['move_offsets'][v] for v in s['context']['members']])[:, None]
    assert torch.tensor(s['action']) == pytest.approx(torch.tensor(expected['action'])+offsets)
    active = {tuple(m['move']['members']): m['until'] for m in s['track2']['active_moves']}
    assert active == {('a',): 342, ('b',): 342}
    assert 'h1' not in s['context']['move_offsets']
    assert s['track2']['v2v_wall']['published'][0]['member_count'] == 2   # b joined at 301 <= t-1


def test_skipped_publication_is_not_received_and_hdvs_never_get_options(cfg, fixed_draws):
    p = make(wall_cfg(cfg, range_m=1000.))
    p.trigger.trigger_head = Prefer(p.trigger)
    tick(p, scene(0), 300)
    p.observe(scene(2), 302)                                   # no publication at 301
    assert p.wall.offers == {}
    assert all(not s['offers'] for s in [p.local_states[v]['v2v_wall'] for v in ('b', 'c')])
    p.wall.publish(302, {v: r.source for v, r in p.reports.items()})
    p.observe(scene(3), 303)
    assert set(p.wall.offers) == {'b', 'c'}                    # every CAV in the 1 km range, never h1
    assert 'h1' not in p.local_states


def test_ownership_and_member_cap(cfg, fixed_draws):
    p = make(wall_cfg(cfg, cap=2))
    p.trigger.trigger_head = Prefer(p.trigger)
    busy = tb.TriggerMove(moves.Move('ease_off', ('d',), 0, 30, .5))
    extra = (row('b2', -350, 1), row('d', -345, 1))

    def tick_rows(dt):
        return scene(dt, extra)
    p.observe(tick_rows(0), 300)
    assert p.executor.start(busy, ego='d', observable_cav_ids=p.visible['d'], now=300, until=330)
    p.last_decision = None
    s = tick(p, tick_rows(0), 300)
    assert s['track2_trigger']['accepted']['a'] is True
    assert families(s, 'd') == []                              # owned CAV: none only
    s = tick(p, tick_rows(1), 301)
    stats = s['track2_trigger']['proposals']['v2v_wall']
    assert stats['event_cavs'] == ['b', 'b2']                  # d is owned: no event
    assert stats['selected_join'] == 2 and stats['accepted_join'] == 1
    assert s['track2_trigger']['accepted'] == dict(a=False, b=True, b2=False, c=False, d=False)
    members = {v for e in p.executor.active for v in e['move'].members}
    assert members == {'a', 'b', 'd'}
    assert p.wall.membership == {'a': 'a@300', 'b': 'a@300'}
    # A member is exclusively owned: no new offers, no second initiation or join.
    s = tick(p, tick_rows(2), 302)
    assert 'b' not in p.wall.offers and 'a' not in p.wall.offers
    join = v2v_wall.WallOption('wall_join', ('b2',), 0, 39, .6, wall_id='a@300', initiator='a',
                               member_count=1, cap=2)
    assert not p.wall.start(tb.TriggerMove(join), ego='b2', executor=p.executor,
                            observable_cav_ids=p.visible['b2'], now=302)           # cap
    again = v2v_wall.WallOption('wall_initiate', ('b',), 0, 30, .6, wall_id='b@302', initiator='b', cap=2)
    assert not p.wall.start(tb.TriggerMove(again), ego='b', executor=p.executor,
                            observable_cav_ids=p.visible['b'], now=302)            # one move per CAV
    tick(p, tick_rows(3), 303)
    assert p.wall.board['a@300'].member_count == 2 and 'b2' not in p.wall.offers  # full: no offers


def test_release_message_and_zone_release(cfg, fixed_draws):
    p = make(wall_cfg(cfg, range_m=300.))
    p.trigger.trigger_head = Prefer(p.trigger)

    def two(a_s, b_s):
        return [row('a', a_s, 0), row('b', b_s, 1), row('h1', b_s-30, 0, cav=False)]
    tick(p, two(-400, -330), 300)
    s = tick(p, two(-398, -328), 301)
    assert s['track2_trigger']['accepted']['b'] is True
    s = tick(p, two(-396, -326), 302)
    assert s['context']['move_offsets'] == dict(a=-.6, b=-.6)
    s = tick(p, two(-99, -324), 303)                            # initiator reaches the merge zone
    assert s['context']['move_offsets']['a'] == 0. and s['context']['move_offsets']['b'] == -.6
    assert s['track2']['v2v_wall']['published'][0]['released'] is True
    s = tick(p, two(-97, -322), 304)                            # b is 225 m away: not sensed, received
    assert 'a' not in p.execution_positions['b']
    assert s['context']['move_offsets'] == dict(a=0., b=0.)
    assert s['track2']['v2v_wall']['published'] == []          # released session stops broadcasting
    assert {e['until'] for e in p.executor.active} == {342}     # ownership held to expiry


# ------------------------------------------------------------------ base observation / zero-init

def test_mappo_observation_and_base_untouched(cfg):
    off = make(dict(cfg, trigger_partner_slots=4))
    on = make(wall_cfg(cfg, awareness_slots=6, awareness_range_m=250.))
    base_off = {k: v.clone() for k, v in off.layer.base.state_dict().items()}
    assert all(torch.equal(v, on.layer.base.state_dict()[k]) for k, v in base_off.items())
    for t in (300, 301):
        a, b = tick(off, scene(t-300), t), tick(on, scene(t-300), t)
        for key in ('members', 'x', 'v2v', 'rsu', 'anchors'):
            assert a['context'][key] == b['context'][key]
        with torch.no_grad():
            assert torch.equal(off.layer.base(a['context'])[0], on.layer.base(b['context'])[0])
            assert torch.equal(off.trigger._hidden(a['context']), on.trigger._hidden(b['context']))
    assert all(torch.equal(v, on.layer.base.state_dict()[k]) for k, v in base_off.items())
    assert all(not p.requires_grad and p.grad is None for p in on.layer.base.parameters())
    assert not on.layer.base.training


@pytest.mark.parametrize('slots,range_m', [(6, 250.), (4, 200.), (6, 100.)])
def test_awareness_zero_init_equals_off(cfg, slots, range_m):
    base_cfg = dict(cfg, trigger_partner_slots=4)
    aware_cfg = dict(base_cfg, **v2v_wall.config_keys(provenance=PROV, awareness_slots=slots,
                                                      awareness_range_m=range_m))
    assert set(aware_cfg)-set(base_cfg) == {'coop_awareness'}
    off, on = make(base_cfg), make(aware_cfg)
    width = v2v_wall.awareness_width(on.awareness)
    w_off, w_on = off.layer.residual[0].weight, on.layer.residual[0].weight
    assert w_on.shape[1] == w_off.shape[1]+width
    assert torch.equal(w_on[:, :w_off.shape[1]-1], w_off[:, :-1]) and torch.equal(w_on[:, -1], w_off[:, -1])
    assert not bool(w_on[:, w_off.shape[1]-1:-1].any())
    t_off, t_on = off.trigger.trigger_head[0].weight, on.trigger.trigger_head[0].weight
    assert torch.equal(t_on[:, :t_off.shape[1]], t_off) and not bool(t_on[:, t_off.shape[1]:].any())
    far = (row('e', -50, 1), row('f', -480, 1))                # beyond 100 m: only awareness sees them
    for t in range(300, 304):
        a = tick(off, scene(t-300, far), t, deterministic=False, seed=t)
        b = tick(on, scene(t-300, far), t, deterministic=False, seed=t)
        assert a['action'] == b['action'] and a['logp'] == b['logp']
        for key in ('old_residual', 'old_logits', 'choices', 'offsets'):
            assert a['coop_layer'][key] == b['coop_layer'][key]
        if 'track2_trigger' in a:
            for key in ('tables', 'actions', 'logp', 'values', 'mask'):
                assert a['track2_trigger'][key] == b['track2_trigger'][key]
        assert len(b['context']['coop_awareness']['a']) == width
    if range_m > 100.:
        assert any(b['context']['coop_awareness']['a'][4*i+3] for i in range(slots))
    assert off.layer.config.feature_dim+width == on.layer.config.feature_dim


# ------------------------------------------------------------------ B = R

def test_b_and_r_run_the_identical_protocol(cfg, fixed_draws):
    out = {}
    for arm in ('B', 'R'):
        p = make(wall_cfg(cfg, cap=3), arm)
        p.trigger.trigger_head = Prefer(p.trigger)
        samples = [tick(p, scene(t-300, (row('b2', -350, 1),)), t, deterministic=False, seed=t)
                   for t in range(300, 305)]
        for s in samples:
            assert s['track2'].pop('arm') == arm
        out[arm] = canon(samples)+canon(p.wall.membership)
    assert out['B'] == out['R']
    assert '"wall_join"' in out['B'] and '"wall_initiate"' in out['B']


# ------------------------------------------------------------------ bank ingestion

MEMBERS = ('m1', 'm2', 'm3', 'm4')


def wall_frames(t0=300, seconds=4):
    """Four mainline CAVs 60 m apart: every CAV senses only its neighbours."""
    frames = {}
    for t in range(t0-1, t0+seconds):
        dt = t-t0
        frames[t] = dict(vehicles=[row('m1', -400+2*dt, 0), row('m2', -340+2*dt, 1),
                                   row('m3', -280+2*dt, 0), row('m4', -220+2*dt, 1),
                                   row('h1', -310+2*dt, 0, cav=False)])
    return frames


def wall_bank(frames, t0=300):
    plan = dict(members=list(MEMBERS), duration_s=40, perturbation_profile=[[-1.]*30+[0.]*10]*4)
    pairs = []
    for t in sorted(frames):
        if t < t0:
            continue
        context = encode(frames[t]['vehicles'])
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        pairs.append((dict(context=context, intent_link=dict(t=t)), 1.))
    return plan, pairs


def test_four_member_wall_unobservable_by_any_member_is_taught_through_the_protocol(cfg):
    frames = wall_frames()
    plan, pairs = wall_bank(frames)
    assert moves.from_plan(plan) == moves.Move('wall', MEMBERS, 0, 30, 1.)
    legacy = make(dict(cfg, trigger_partner_slots=4))
    reader = deepcopy(legacy)
    for t in sorted(frames):
        reader.observe(frames[t]['vehicles'], t)
        assert not any(set(MEMBERS) <= set(reader.visible[v]) for v in MEMBERS)   # precondition
    assert list(breadth.observable_contexts(legacy, plan, pairs, frames, t0=300)) == []
    p = make(wall_cfg(cfg, cap=6))
    got = list(breadth.observable_contexts(p, plan, pairs, frames, t0=300))
    labels = [(o.family, o.members[0], t) for _, o, _, _, t in got]
    # Range 100 m: m2 reaches m1 and m3 (ties broken by plan order); m4 (120 m) cannot receive.
    assert labels == [('wall_initiate', 'm2', 300), ('wall_join', 'm1', 301), ('wall_join', 'm3', 301)]
    initiate, join = got[0][1], got[1][1]
    assert (initiate.onset, initiate.hold, initiate.depth, initiate.wall_id) == (0, 30, 1., 'm2@300')
    assert (join.wall_id, join.onset, join.hold, join.member_count) == ('m2@300', 0, 29, 1)
    assert join.gap_m == -340.-(-398.)                         # t-1 message vs own position at t
    for context, option, reader, _, t in got:
        ego = option.members[0]
        assert context['members'] == [ego] and set(context['trigger_v2v']) == {ego}
        assert context['merge_features'] == {ego: reader.blocks[ego]}
        assert not set(MEMBERS) <= set(reader.visible[ego])
    # Longer registered message range: the plan-order initiator reaches all three members.
    p = make(wall_cfg(cfg, cap=6, range_m=200.))
    labels = [(o.family, o.members[0]) for _, o, _, _, _ in
              breadth.observable_contexts(p, plan, pairs, frames, t0=300)]
    assert labels == [('wall_initiate', 'm1'), ('wall_join', 'm2'), ('wall_join', 'm3'), ('wall_join', 'm4')]
    # Registered cap: initiator + 2 joiners, admitted in stable order.
    p = make(wall_cfg(cfg, cap=3, range_m=200.))
    labels = [o.members[0] for _, o, _, _, _ in breadth.observable_contexts(p, plan, pairs, frames, t0=300)]
    assert labels == ['m1', 'm2', 'm3']


def test_ingestion_uses_only_own_observation_and_received_message(cfg):
    frames = wall_frames()
    plan, pairs = wall_bank(frames)
    p = make(wall_cfg(cfg, cap=6))
    base = list(breadth.observable_contexts(p, plan, pairs, frames, t0=300))
    far = deepcopy(frames)
    for t, frame in far.items():                               # outside every member's sensing / radio
        frame['vehicles'].append(row('zz', 600+2*(t-300), 1, cav=False))
    moved = list(breadth.observable_contexts(p, plan, pairs, far, t0=300))
    assert [(o, c['merge_features'], c['trigger_v2v']) for c, o, *_ in base] == \
           [(o, c['merge_features'], c['trigger_v2v']) for c, o, *_ in moved]


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_bank_supervision_imitates_initiate_and_join_identically_for_b_and_r(cfg, monkeypatch, arm):
    from . import bank_v2v
    frames = wall_frames()
    plan, pairs = wall_bank(frames)
    on = wall_cfg(cfg, cap=6)

    def visit(config, requested_arm, consume):
        consume(cfg=config, receipt=dict(verification_id='mined_986061_t455', t0=300),
                example=dict(example_id='mined_986061_t455#s1'), plan=plan, pairs=deepcopy(pairs),
                trace='CPU')
    monkeypatch.setattr(breadth, 'visit', visit)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path: frames)
    results = {}
    for value in ('B', 'R'):
        policy = make(on, value)
        before = deepcopy(policy.layer.base.state_dict())
        results[value] = launch.bank_move_supervision(SimpleNamespace(cfg=on, policy=policy, arm=value))
        assert all(torch.equal(v, policy.layer.base.state_dict()[k]) for k, v in before.items())
    result = results[arm]
    assert result['family_counts'] == {'wall_initiate': 1, 'wall_join': 2}
    assert [(c['family'], c['members'], c['time_s']) for c in result['contexts']] == [
        ('wall_initiate', ['m2'], 300), ('wall_join', ['m1'], 301), ('wall_join', ['m3'], 301)]
    assert result['final_ce'] < result['initial_ce']

    def strip(r):
        return canon({k: v for k, v in r.items() if k != 'passes'})
    assert strip(results['B']) == strip(results['R'])


def test_recorded_wall_decisions_replay_for_trigger_ppo(cfg, fixed_draws):
    p = make(wall_cfg(cfg))
    p.trigger.trigger_head = Prefer(p.trigger)
    tick(p, scene(0), 300)
    s = tick(p, scene(1), 301, deterministic=False, seed=5)
    record = json.loads(json.dumps(s['track2_trigger']))
    batch = tb._batch(record)
    assert any(isinstance(m, v2v_wall.WallOption) for table in batch.tables for m in table)
    logp, values, _ = p.trigger.evaluate(batch)
    assert logp.tolist() == pytest.approx(record['logp'])
    # G4 sparse credit ignores accepted per-CAV protocol options instead of failing on them.
    from . import sparse_synergy_credit as g4
    assert record['accepted']['b'] is True
    assert g4.opportunities([dict(id='ep', steps=[dict(time_s=301, sample=s)])]) == []
