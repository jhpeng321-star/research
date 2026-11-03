"""CPU gates for the optional V2V cooperation signal (option ``v2v_coop_signal``,
tools/coopmwm/v2v_intent.py + v2v_obs.py, DEFAULT OFF and byte-identical when off),
real torch, no SUMO."""
import copy
import gzip
import json
import math

import numpy as np
import pytest
import torch

from tools.coopnative.storage import canonical
from .acceleration import Controller, encode
from .accel_learner import save
from . import bank_v2v, bank_imitation, v2v_intent as vi
from .v2v_obs import (V2VConfig, coop_columns, published_coop, v2v_coop_dim, v2v_features,
                      v2v_intent_dim, v2v_intent_features, v2v_partner_ids)
from .v2v_runtime import controller_class, from_weights, load_checkpoint

H = 5
W = v2v_coop_dim(4, H)                      # 56
CFG = vi.config_keys(coop=True)             # intent + coop keys, v2v_dim 56
CFGI = vi.config_keys()                     # intent only, v2v_dim 48
STRIDE = 14                                 # 6 + H + 1 + 2 per partner slot


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def rows(t=0, drop=(), speed=None):
    spec = (('c0', 0., 10., 0, True), ('c1', 20., 10.5, 0, True), ('c2', -30., 11., 1, True),
            ('c3', 50., 9., 1, True), ('h0', 10., 9.5, 2, False))
    out = []
    for vid, s, v, lane, cav in spec:
        if vid in drop:
            continue
        v = (speed or {}).get(vid, v)
        out.append(dict(vid=vid, s_m=s+10.*t, v_mps=v, lane=lane, is_cav=cav, length_m=5.))
    return out


def mappo_u0(path, seed=7):
    """Base 33-wide MAPPO-shaped checkpoint with a NONZERO action head and warmed Adam."""
    from .accel_learner import save as _save
    from .acceleration import Policy as BasePolicy
    torch.manual_seed(seed)
    policy = BasePolicy()
    torch.nn.init.normal_(policy.actor[-1].weight, std=.3)
    torch.nn.init.normal_(policy.actor[-1].bias, std=.1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    context = encode(rows())
    mean, value = policy(context)
    (mean.sum()+value).backward()
    optimizer.step()
    _save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, 0)
    return policy


def coop_policy(tmp_path, seed=7, nonzero=False):
    path = tmp_path/'A_checkpoint_000.pt'
    mappo_u0(path, seed)
    policy, optimizer, generator, _ = load_checkpoint(path, config=dict(CFG))
    if nonzero:
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            policy.intent_head.weight.copy_(torch.randn(policy.intent_head.weight.shape, generator=g))
            policy.coop_head.weight.copy_(torch.randn(policy.coop_head.weight.shape, generator=g))
            policy.coop_head.bias.copy_(torch.randn(policy.coop_head.bias.shape, generator=g))
    return policy, optimizer, generator


def intent_policy(tmp_path, seed=7):
    """v2v_intent ON, coop OFF (option off): plain 48-wide IntentPolicy, no coop head."""
    path = tmp_path/'A_checkpoint_000_intent.pt'
    mappo_u0(path, seed)
    policy, optimizer, generator, _ = load_checkpoint(path, config=dict(CFGI))
    return policy, optimizer, generator


def runtime_controller(policy, seed=4, deterministic=True):
    return controller_class(Controller)(policy, torch.Generator().manual_seed(seed), deterministic)


# ---------------------------------------------------------------- 1. layout / dims

def test_coop_block_layout_and_columns():
    assert W == 56 and v2v_intent_dim() == 48 and CFG['v2v_dim'] == 56 and CFGI['v2v_dim'] == 48
    assert coop_columns() == [s*STRIDE+12+j for s in range(4) for j in range(2)]
    state = {r['vid']: dict(r, a_mps2=.1) for r in rows()}
    cfg = V2VConfig(v2v=True)
    board = {'c1': (300, [1., 2., 3., 4., 5.])}
    coop_board = {'c1': (300, .7, 'c0'), 'c2': (300, .9, None), 'c3': (299, .5, 'c2')}
    wide = v2v_intent_features(state, 'c0', {'c1': .5}, board, 301, H, cfg, coop=dict(board=coop_board))
    narrow = v2v_intent_features(state, 'c0', {'c1': .5}, board, 301, H, cfg)
    assert wide.shape == (W,) and wide.dtype == np.float32
    # the base 6 + H + 1 per-slot features are exactly the 48-wide intent block
    for slot in range(4):
        assert np.array_equal(wide[slot*STRIDE:slot*STRIDE+12], narrow[slot*12:slot*12+12])
    # coop columns: c1 fresh -> p=.7 and c0 IS the yield target; c2 fresh, yields to nobody;
    # c3 stale publication -> zeros; empty slot -> zeros
    assert list(wide[12:14]) == [.7, 1.]
    assert list(wide[26:28]) == [.9, 0.]
    assert list(wide[40:42]) == [0., 0.]
    assert list(wide[54:56]) == [0., 0.]
    assert not v2v_intent_features(state, 'c0', {}, board, 301, H, V2VConfig(),
                                   coop=dict(board=coop_board)).any()          # flag off -> zeros


def test_published_coop_validity_rules():
    assert published_coop({'z': (302, .5, 'y')}, 'z', 303) == (.5, 'y')
    assert published_coop({'z': (303, .5, 'y')}, 'z', 303) is None            # never same-step
    assert published_coop({'z': (302, .5, 'y')}, 'z', 304) is None            # stale
    assert published_coop({'z': (302, 1.5, 'y')}, 'z', 303) is None           # not a probability
    assert published_coop({'z': (302, float('nan'), None)}, 'z', 303) is None
    assert published_coop({'z': (302, .5, None)}, 'z', 303) == (.5, None)     # none is a valid target
    assert published_coop({}, 'z', 303) is None


# ---------------------------------------------------------------- 2. option off is unchanged

def test_option_off_settings_keys_and_policy_are_unchanged():
    assert vi.settings(dict(CFGI))['coop'] is False and 'v2v_coop_signal' not in CFGI
    assert vi.settings({}) is None and vi.config_keys() == CFGI
    with pytest.raises(ValueError, match='requires v2v_intent'):
        vi.settings(dict(v2v_coop_signal=True))
    with pytest.raises(ValueError, match='requires v2v_intent'):
        from . import v2v_runner
        v2v_runner.prepare(dict(budget={}), v2v_dim=W, v2v_coop_signal=True, root='/r',
                           legacy_root='/r/legacy', source_root='/r/src',
                           common_checkpoint_sha256='a'*64)


def test_option_off_policy_ignores_stray_coop_targets(tmp_path):
    """Coop-off IntentPolicy: no coop head, coop targets in pairs never reach the loss."""
    policy, optimizer, _ = intent_policy(tmp_path)
    assert policy.coop_enabled is False and not hasattr(policy, 'coop_head')
    context = encode(rows())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    policy.intent_head.bias.detach().fill_(.1)
    pairs = [(dict(context=copy.deepcopy(context),
                   intent_target=[[0.]*H]*len(context['members']),
                   intent_mask=[[1.]*H]*len(context['members']),
                   coop_targets={v: [1., 0] for v in context['members']}), 1.)]
    from .v2v_intent import bank_term, coop_bank_stats, weight_norms
    (bank_term(policy, pairs)-policy.intent_loss_weight*vi.mean_loss(
        policy, [pairs[0][0]['context']], [pairs[0][0]['intent_target']],
        [pairs[0][0]['intent_mask']])[0]).backward()
    assert float(policy.intent_head.bias.grad.sum()) == 0.     # the two terms cancel exactly
    assert coop_bank_stats(policy, pairs) == dict(coop_loss=None, coop_members_supervised=0, coop_pairs=0)
    assert set(weight_norms(policy)) == {'actor', 'critic'}


def test_option_off_runtime_publishes_no_coop(tmp_path):
    policy, _, _ = intent_policy(tmp_path)
    c = runtime_controller(policy)
    commands = {r['vid']: .2 for r in rows() if r['is_cav']}
    c.apply(rows(0), commands, 301)
    c.apply(rows(1), commands, 302)
    sample = c.events[-1]['sample']
    assert 'published_coop' not in sample
    assert all(len(v) == 48 for v in sample['context']['v2v'].values())


# ---------------------------------------------------------------- 3. zero-init identity at load

def test_zero_init_identity_at_load_from_u0_and_intent_checkpoints(tmp_path):
    u0 = tmp_path/'u0.pt'
    base = mappo_u0(u0)
    coop, opt, _, _ = load_checkpoint(u0, config=dict(CFG))
    assert isinstance(coop, vi.IntentPolicy) and coop.coop_enabled and coop.v2v_width == W
    assert {id(p) for p in coop.parameters()} == {id(p) for g in opt.param_groups for p in g['params']}
    context = encode(rows())
    table = {r['vid']: dict(r, a_mps2=.3) for r in rows()}
    board = {v: (300, [1.5, -.5, .25, 2., -3.]) for v in table if v.startswith('c')}
    narrow = dict(context, v2v={v: v2v_features(table, v, {'c1': .5}, V2VConfig(v2v=True)).tolist()
                                for v in context['members']})
    wide = dict(context, v2v={v: v2v_intent_features(table, v, {'c1': .5}, board, 301, H,
                               V2VConfig(v2v=True), coop=dict(board={})).tolist()
                              for v in context['members']})
    assert any(x != 0. for b in wide['v2v'].values() for x in b[6:12])
    assert all(b[s*STRIDE+12:s*STRIDE+14] == [0., 0.]
               for b in wide['v2v'].values() for s in range(4))      # empty coop board -> zeros
    for deterministic in (True, False):
        a, ma = base.sample(narrow, torch.Generator().manual_seed(9), deterministic, {'c0': .1})
        b, mb = coop.sample(wide, torch.Generator().manual_seed(9), deterministic, {'c0': .1})
        assert a['action'] == b['action'] and a['logp'] == b['logp'] and ma == mb
        assert a['value'] == pytest.approx(b['value'], abs=1e-7)
    assert torch.equal(base.score(a)[0], coop.score(b)[0])
    published = coop.publish_coop(wide)
    assert all(v[0] == pytest.approx(.5, abs=1e-7) and v[1] == pytest.approx([.2]*5, abs=1e-7)
               for v in published.values())                    # zero-init head: p=.5, uniform
    assert vi.weight_norms(coop) == dict(actor=0., critic=0., coop_actor=0., coop_critic=0.)
    # an intent-only checkpoint loads into the coop policy with identical actions
    intent_path = tmp_path/'intent.pt'
    intent, iopt, _, _ = load_checkpoint(u0, config=dict(CFGI))
    with torch.no_grad():
        intent.intent_head.weight.normal_(std=.1)
        intent.actor[0].weight[:, 29:29+48].normal_(std=.05)
        intent.critic[0].weight[:, 29:29+48].normal_(std=.05)
    save(intent_path, intent, iopt, torch.Generator().manual_seed(3), 1, 0)
    from_intent, iopt2, _, _ = load_checkpoint(intent_path, config=dict(CFG))
    ai, _ = intent.sample(dict(context, v2v={v: v2v_intent_features(table, v, {'c1': .5}, board, 301, H,
                                 V2VConfig(v2v=True)).tolist() for v in context['members']}),
                          torch.Generator().manual_seed(5), False, {})
    ac, _ = from_intent.sample(wide, torch.Generator().manual_seed(5), False, {})
    assert ai['action'] == ac['action'] and ai['logp'] == ac['logp']
    assert torch.equal(intent.intent_head.weight, from_intent.intent_head.weight)
    assert float(from_intent.coop(wide).abs().max()) == 0.       # coop head starts at zero
    assert any(v['exp_avg'].shape == (64, 33+W) for v in iopt2.state_dict()['state'].values())
    # a trained v2v-only (block 24) checkpoint loads into the coop policy: zero coop columns,
    # identical actions, re-gridded Adam moments
    from .v2v_obs import V2V_DIM as V24
    v2v_only, vopt, gen, _ = load_checkpoint(u0, config=dict(v2v=True, v2v_dim=V24))
    with torch.no_grad():
        v2v_only.actor[0].weight[:, 29:29+V24].normal_(std=.2)
        v2v_only.critic[0].weight[:, 29:29+V24].normal_(std=.2)
    mean, value = v2v_only(dict(context, v2v={v: v2v_features(table, v, {'c1': .5},
                                 V2VConfig(v2v=True)).tolist() for v in context['members']}))
    (mean.sum()+value).backward()
    vopt.step()
    v2v_path = tmp_path/'v2v_only.pt'
    save(v2v_path, v2v_only, vopt, gen, 2, 0)
    from_v2v, vopt2, _, _ = load_checkpoint(v2v_path, config=dict(CFG))
    av, _ = v2v_only.sample(dict(context, v2v={v: v2v_features(table, v, {'c1': .5},
                             V2VConfig(v2v=True)).tolist() for v in context['members']}),
                            torch.Generator().manual_seed(5), False, {})
    bv, _ = from_v2v.sample(wide, torch.Generator().manual_seed(5), False, {})
    assert np.allclose(av['action'], bv['action'], rtol=0., atol=1e-6)
    assert abs(av['logp']-bv['logp']) < 1e-5
    assert vi.weight_norms(from_v2v)['coop_actor'] == 0.
    assert any(v['exp_avg'].shape == (64, 33+W) for v in vopt2.state_dict()['state'].values())


def test_checkpoint_roundtrip_from_weights_and_rejections(tmp_path):
    policy, optimizer, generator = coop_policy(tmp_path, nonzero=True)
    context = encode(rows())
    table = {r['vid']: r for r in rows()}
    coop_board = {'c1': (300, .7, 'c0')}
    context['v2v'] = {v: v2v_intent_features(table, v, {}, {'c1': (300, [.1]*5)}, 301, H,
                       V2VConfig(v2v=True), coop=dict(board=coop_board)).tolist() for v in context['members']}
    (policy.intent(context).sum()+policy.coop(context).square().sum()+policy(context)[1]).backward()
    optimizer.step()
    path = tmp_path/'A_checkpoint_001.pt'
    save(path, policy, optimizer, generator, 1, 0)
    restored, opt2, _, payload = load_checkpoint(path, config=dict(CFG))
    assert payload['update'] == 1 and isinstance(restored, vi.IntentPolicy) and restored.coop_enabled
    for (k, a), (k2, b) in zip(policy.state_dict().items(), restored.state_dict().items()):
        assert k == k2 and torch.equal(a, b)
    assert torch.equal(restored.intent(context), policy.intent(context))
    assert torch.equal(restored.coop(context), policy.coop(context))
    assert restored.publish_coop(context) == policy.publish_coop(context)
    assert len(opt2.state) == len(optimizer.state) > 0
    for a, b in zip(opt2.state_dict()['state'].values(), optimizer.state_dict()['state'].values()):
        assert all(torch.equal(a[k], b[k]) for k in a)
    worker = from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True)
    assert isinstance(worker, vi.IntentPolicy) and worker.coop_enabled
    assert worker.publish_coop(context) == policy.publish_coop(context)
    with pytest.raises(ValueError, match='explicit V2V'):
        from_weights(policy.state_dict())
    with pytest.raises(ValueError, match='horizon'):
        load_checkpoint(path, config=dict(vi.config_keys(coop=True, horizon=3)))
    with pytest.raises(ValueError, match='v2v_coop_signal'):
        load_checkpoint(path, config=dict(CFGI))                 # coop checkpoint, coop OFF
    u0b = tmp_path/'u0b.pt'
    mappo_u0(u0b)
    plain, _, _, _ = load_checkpoint(u0b, config=dict(CFGI))
    assert plain.coop_enabled is False and not hasattr(plain, 'coop_head')


# ---------------------------------------------------------------- 4. broadcast t-1 latency

def coop_broadcast_head(policy):
    """Deterministic coop head: p_coop ~ .88, yield argmax = partner slot 0."""
    with torch.no_grad():
        policy.coop_head.bias.zero_()
        policy.coop_head.bias[0] = 2.
        policy.coop_head.bias[1] = 4.                           # slot 0 beats none and the rest
    return policy


def test_partner_coop_signal_has_one_step_latency(tmp_path):
    policy = coop_policy(tmp_path, nonzero=True)[0]   # random head: publications move across steps
    c = runtime_controller(policy)
    commands = {r['vid']: .2 for r in rows() if r['is_cav']}
    c.apply(rows(0), commands, 301)
    first = c.events[-1]['sample']
    assert set(first['published_coop']) == set(first['context']['members'])
    assert all(len(v) == W for v in first['context']['v2v'].values())
    assert all(v[s*STRIDE+12:s*STRIDE+14] == [0., 0.]
               for v in first['context']['v2v'].values() for s in range(4))   # nothing yet
    assert all(t == 301 for t, _p, _y in c.coop_board.values())              # published at 301
    c.apply(rows(1), commands, 302)
    second = c.events[-1]['sample']
    table = {r['vid']: dict(r) for r in rows(1)}
    slot = v2v_partner_ids(table, 'c0', V2VConfig(v2v=True)).index('c1')
    seen = second['context']['v2v']['c0'][slot*STRIDE+12:slot*STRIDE+14]
    expect = first['published_coop']['c1']
    assert seen[0] == pytest.approx(expect[0], abs=1e-6)
    assert seen[1] == (1. if expect[1] == 'c0' else 0.)
    assert seen[0] != pytest.approx(second['published_coop']['c1'][0], abs=1e-6)   # never same-step
    # a skipped second makes every received coop signal stale
    c.apply(rows(3), commands, 304)
    assert all(v[s*STRIDE+12:s*STRIDE+14] == [0., 0.]
               for v in c.events[-1]['sample']['context']['v2v'].values() for s in range(4))


def test_yield_target_bit_marks_only_the_named_partner(tmp_path):
    policy = coop_broadcast_head(coop_policy(tmp_path)[0])
    c = runtime_controller(policy)
    commands = {r['vid']: .2 for r in rows() if r['is_cav']}
    c.apply(rows(0), commands, 301)
    c.apply(rows(1), commands, 302)
    resolved = c.events[-1]['sample']['published_coop']
    # c1's slot-0 partner is c0 (gap -20, nearest): c1 yields to c0
    table = {r['vid']: r for r in rows(1)}
    assert resolved['c1'][1] == 'c0'
    blocks = c.events[-1]['sample']['context']['v2v']
    for ego in blocks:
        slots = v2v_partner_ids({r['vid']: r for r in rows(1)}, ego, V2VConfig(v2v=True))
        for s, vid in enumerate(slots):
            got = blocks[ego][s*STRIDE+12:s*STRIDE+14]
            assert got[1] == (1. if resolved[vid][1] == ego else 0.)
    assert blocks['c0'][v2v_partner_ids(table, 'c0', V2VConfig(v2v=True)).index('c1')*STRIDE+13] == 1.
    assert blocks['c2'][v2v_partner_ids(table, 'c2', V2VConfig(v2v=True)).index('c1')*STRIDE+13] == 0.


# ---------------------------------------------------------------- 5. coop targets (masked supervision)

def plan_command(vid, t):
    return 1.5*math.sin(.7*t) if vid in ('c0', 'c1') else 0.


def coop_bank(tmp_path, t0=420, duration=12, command=None, cooperative=True):
    """Plan c0+c1 over the intent-fixture geometry; pairs carry 56-wide blocks with EMPTY coop
    inputs, intent links, and (cooperative plans only) coop targets from the plan geometry."""
    from .exact_grade import imitation_pairs
    command = command or plan_command
    tmp_path.mkdir(parents=True, exist_ok=True)
    steps, frames = [], []
    for t in range(t0-1, t0+duration+H):
        sensor = rows(t-t0)
        context = encode(sensor)
        raw = {v: command(v, t) for v in ('c0', 'c1')}
        commands = {r['vid']: 0. for r in sensor if r['is_cav']}
        steps.append(dict(time_s=t, commands=commands,
                          requests={v: dict(requested_raw=x) for v, x in raw.items()},
                          sample=dict(context=context,
                                      action=[[raw.get(v, 0.)] for v in context['members']],
                                      absolute_accels=raw,
                                      behaviour=dict(executed=True, plan_id='p',
                                                     proposal_members=['c0', 'c1']),
                                      plan_executions=[dict(plan_id='p', proposal_members=['c0', 'c1'])])))
        frames.append(dict(time_s=t, vehicles=[dict(r, issued_acceleration_mps2=raw.get(r['vid'], 0.)
                                                    if r['is_cav'] else None) for r in sensor]))
    tape = dict(steps=steps)
    trace = tmp_path/'observable.jsonl.gz'
    with gzip.open(trace, 'wt', encoding='utf-8') as f:
        for frame in frames:
            f.write(canonical(frame)+'\n')
    entry = dict(start_s=t0, duration_s=duration, plan_id='p', weight=1.)
    pairs = imitation_pairs(dict(steps=[s for s in steps if t0 <= s['time_s'] < t0+duration]), entry)
    pairs = vi.attach_bank_targets(pairs, tape, entry, H)
    kept, counts = bank_v2v.attach_pairs(pairs, trace, t0,
                                         intent=dict(horizon=H, entry='v/b', coop=dict(board={})))
    assert counts['kept'] == duration
    return vi.attach_coop_targets(kept, tape, entry, trace, cooperative), tape, entry, trace


def test_coop_targets_only_for_cooperative_plans_and_members(tmp_path):
    pairs, tape, entry, trace = coop_bank(tmp_path)
    for sample, _w in pairs:
        targets = sample['coop_targets']
        assert targets['c0'] == [1., None]          # nobody behind c0 in lane 0 -> none
        assert targets['c1'] == [1., 0]             # c0 behind in lane 0, nearest -> slot 0
        assert set(targets) == {'c0', 'c1'}         # context members outside the plan stay masked
    # c1's ranking really puts c0 in slot 0
    assert pairs[3][0]['intent_link']['slots']['c1'][0] == 'c0'
    # a non-decelerating plan opens no gap: every yield target is none
    pairs2, _, _, _ = coop_bank(tmp_path/'flat', command=lambda v, t: .5)
    assert all(t == [1., None] for s, _w in pairs2 for t in s['coop_targets'].values())
    # non-cooperative entries: no targets at all
    pairs3, _, _, _ = coop_bank(tmp_path/'nocoop', cooperative=False)
    assert all('coop_targets' not in s for s, _w in pairs3)
    # blocks stay 56 wide with EMPTY coop inputs until the refresh
    assert all(b[s*STRIDE+12:s*STRIDE+14] == [0., 0.] for s, _w in pairs
               for b in s['context']['v2v'].values() for s in range(4))


def test_coop_terms_and_grad_reach_only_the_coop_head(tmp_path):
    policy, _, _ = coop_policy(tmp_path)
    pairs, _, _, _ = coop_bank(tmp_path)
    contexts = [s['context'] for s, _w in pairs]
    targets = [s['coop_targets'] for s, _w in pairs]
    total, count = vi.coop_terms(policy, contexts, targets)
    assert count == 2*len(pairs)
    policy.zero_grad()
    total.backward()
    assert float(policy.coop_head.weight.grad.abs().sum()) > 0
    trunk = sum(float(p.grad.abs().sum()) for n, p in policy.named_parameters()
                if not n.startswith('coop_head') and p.grad is not None)
    assert trunk == 0.                                              # detach_trunk default
    assert policy.intent_head.weight.grad is None or \
        float(policy.intent_head.weight.grad.abs().sum()) == 0.     # intent head untouched
    assert vi.grad_share(policy) == 0.                              # it counts the intent head only
    # out-of-range slots and missing members are label-none / masked
    crafted = [dict(contexts[0]), dict(contexts[1])]
    crafted_targets = [{v: [1., 7] for v in ('c0', 'c1')}, {}]
    total2, count2 = vi.coop_terms(policy, crafted, crafted_targets)
    assert count2 == 2                                              # only the first context supervised
    # a coop term with no targets is exactly zero and keeps the graph
    policy.zero_grad()
    zero, count0 = vi.coop_terms(policy, [crafted[1]], [crafted_targets[1]])
    zero.backward()
    assert float(policy.coop_head.weight.grad.abs().sum()) == 0.


def test_ppo_rollouts_get_no_coop_supervision(tmp_path):
    policy, _, _ = coop_policy(tmp_path)
    steps = []
    for i in range(6):
        context = encode(rows(i))
        steps.append(dict(time_s=300+i,
                          sample=dict(context=dict(context, v2v={v: [0.]*W for v in context['members']})),
                          commands={r['vid']: 0. for r in rows(i) if r['is_cav']},
                          requests={}, reward=-1.))
    episodes = [dict(steps=steps)]
    helper = vi.ppo_setup(policy, episodes)
    policy.zero_grad()
    helper.term(range(len(helper.contexts))).backward()
    assert policy.coop_head.weight.grad is None or float(policy.coop_head.weight.grad.abs().sum()) == 0.
    assert helper.report.get('coop_loss') is None                   # never in the PPO report


# ---------------------------------------------------------------- 6. refresh fills coop inputs

def test_refresh_bank_coop_inputs_from_current_head(tmp_path):
    policy, _, _ = coop_policy(tmp_path, nonzero=True)
    pairs, _, _, _ = coop_bank(tmp_path)
    report = vi.refresh_bank_intents(policy, pairs)
    assert report['coop_slots'] > 0 and report['coop_slots_valid'] > 0
    assert report['coop_slots_valid'] < report['coop_slots']         # plan start has no t-1 context
    for i, (sample, _w) in enumerate(pairs):
        prev = pairs[i-1][0] if i else None
        for ego, slots in sample['intent_link']['slots'].items():
            for slot, vid in enumerate(slots[:4]):
                got = sample['context']['v2v'][ego][slot*STRIDE+12:slot*STRIDE+14]
                if prev is None or vid not in prev['context']['members']:
                    assert got == [0., 0.]
                    continue
                expect = vi.resolve_coop(policy.publish_coop(prev['context']),
                                         prev['intent_link']['slots'])[vid]
                assert got[0] == pytest.approx(expect[0], abs=1e-6)
                assert got[1] == (1. if expect[1] == ego else 0.)
    # recomputed with the CURRENT head on every refresh
    with torch.no_grad():
        policy.coop_head.bias.add_(.3)
    before = [list(b[s*STRIDE+12:s*STRIDE+14]) for s, _w in pairs
              for b in s['context']['v2v'].values() for s in range(4)]
    vi.refresh_bank_intents(policy, pairs)
    after = [list(b[s*STRIDE+12:s*STRIDE+14]) for s, _w in pairs
             for b in s['context']['v2v'].values() for s in range(4)]
    assert before != after


# ---------------------------------------------------------------- 7. one cooperative bank update

def rollout(policy, seconds=9, seed=5):
    c = runtime_controller(policy, seed, deterministic=False)
    steps = []
    for i in range(seconds):  # stochastic steps avoid multiples of 10 (no regime/leverprobe lookup)
        sensor = rows(i)
        commands = {r['vid']: .1*math.sin(i+k) for k, r in enumerate(sensor) if r['is_cav']}
        c.apply(sensor, commands, 311+i)
        steps.append(dict(c.events[-1], reward=-1.-.1*i))
    return dict(steps=steps, world=dict(seed=1, cell='x'))


def test_cooperative_bank_update_moves_coop_columns_and_lowers_coop_loss(tmp_path, monkeypatch):
    from .exact_grade import exact_update
    policy, optimizer, generator = coop_policy(tmp_path)
    pairs, _, _, _ = coop_bank(tmp_path)
    vi.refresh_bank_intents(policy, pairs)
    assert any(any(x != 0. for x in b[s*STRIDE+12:s*STRIDE+14]) for s, _w in pairs[1:]
               for b in s['context']['v2v'].values() for s in range(4))
    worker = from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True)
    episodes = [rollout(worker)]
    assert all('published_coop' in st['sample'] for st in episodes[0]['steps'])
    monkeypatch.setattr(bank_imitation, 'run', bank_imitation.run_separate)
    torch.manual_seed(0)
    norms_before = vi.weight_norms(policy)
    metrics = exact_update(policy, optimizer, episodes, generator, pairs, 1., epochs=1, imitation_epochs=5)
    report = metrics[-1]['bank_imitation']
    intent = report['v2v_intent']
    assert intent['before']['coop_members_supervised'] == 2*len(pairs)
    assert intent['after']['coop_loss'] < intent['before']['coop_loss']
    assert intent['coop_target_rule'] == vi.COOP_TARGET_RULE
    norms_after = vi.weight_norms(policy)
    assert norms_before['coop_actor'] == 0. and norms_after['coop_actor'] > 0.
    ppo = next(m['v2v_intent'] for m in metrics if 'v2v_intent' in m)
    json.dumps(metrics)  # update receipts stay JSON-serialisable


# ---------------------------------------------------------------- 8. registered bank wiring

from .test_replay_bank_confirm_cpu import FakeRunner
from .test_replay_bank_v2v_cpu import V2VTapeRunner


class CoopV2VTapeRunner(V2VTapeRunner):
    """Group plan beats every solo (sum-profile J) so the entry is labelled cooperative; the
    plan tape carries a deceleration for c2, so c2's yield target is its follower c1."""

    def J(self, p):
        if p is None:
            return self.J_policy
        return self.J_policy-float(sum(x for row in p['profile'] for x in row))

    def run(self, spec, snap):
        r = FakeRunner.run(self, spec, snap)
        p = spec['plan']
        if p is None:
            return r
        from tools.coopmwm.accel_storage import write as awrite
        from tools.coopmwm.protocol import digest
        d = self.work/'branches'/spec['verification_id']/spec['branch']
        tape = dict(steps=[dict(time_s=t, sample=dict(context=dict(members=list(p['members'])),
            action=[0.5, -0.5],
            absolute_accels={v: (.5 if v == 'c1' else -.5) for v in p['members']},
            plan_executions=[dict(plan_id=p['id'], proposal_members=list(p['members']))]))
            for t in range(spec['window']['start_s'], spec['window']['start_s']+p['duration_s'])])
        awrite(d/'RESULT.json', tape)
        r['result_sha256'] = digest(tape)
        self.t0 = spec['window']['start_s']
        first = self.t0-1 if self.prev_second else self.t0
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            for t in range(first, self.t0+p['duration_s']):
                f.write(canonical(self.frame(t))+'\n')
        return r


@pytest.fixture
def bank_budget(tmp_path):
    from .warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (tmp_path/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (tmp_path/'AUTHORITY.md').write_text('test-bank-budget')


def test_bank_pairs_carry_coop_targets_only_through_cooperative_entries(tmp_path, bank_budget):
    from . import replay_bank as rb
    from .test_replay_bank_v2v_cpu import V2VTapeRunner, _bank
    root = tmp_path
    bank = rb.bank_root(root, 'B')
    bank.mkdir(parents=True)
    _bank(root, bank, lambda work: CoopV2VTapeRunner(work))
    plain, s1 = rb.bank_pairs(root, 'B', 12345, v2v=True, v2v_intent=dict(CFGI))
    pairs, s2 = rb.bank_pairs(root, 'B', 12345, v2v=True, v2v_intent=dict(CFG))
    assert len(pairs) == len(plain) == 120
    assert 'v2v_coop_signal' not in s1
    assert all('coop_targets' not in p for p, _ in plain)
    assert all(len(b) == 48 for p, _ in plain for b in p['context']['v2v'].values())
    supervised = s2['v2v_coop_signal']
    assert supervised['supervised_pairs'] == 120 and supervised['unsupervised_pairs'] == 0
    assert all(len(b) == W for p, _ in pairs for b in p['context']['v2v'].values())
    for sample, _w in pairs[:4]:
        targets = sample['coop_targets']
        assert targets['c1'] == [1., None]          # nobody behind c1 -> none
        assert targets['c2'] == [1., 0]             # c1 behind in lane 0 -> slot 0
    # a NON-cooperative bank (plain runner: gain 0) supervises nothing
    root2 = tmp_path/'plain'
    bank2 = rb.bank_root(root2, 'B')
    bank2.mkdir(parents=True)
    recon = json.loads((root/'B_REPLAY_PHASE_RECONCILIATION_0627.json').read_text())
    (root2/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(recon))
    (root2/'AUTHORITY.md').write_text('test-bank-budget')
    _bank(root2, bank2, lambda work: V2VTapeRunner(work))
    pairs2, s3 = rb.bank_pairs(root2, 'B', 12345, v2v=True, v2v_intent=dict(CFG))
    assert all('coop_targets' not in p for p, _ in pairs2)
    assert s3['v2v_coop_signal']['supervised_pairs'] == 0
    assert s3['v2v_coop_signal']['unsupervised_pairs'] == len(pairs2)
    json.dumps([p for p, _ in pairs])               # targets stay JSON-safe


# ---------------------------------------------------------------- 9. bindings and hooks

def test_runner_and_live_hooks_wire_the_coop_keys(tmp_path):
    # NOTE: relative imports on purpose. pytest imports this module under its own package name
    # (coopmwm.*), so `from . import ...` below and inside v2v_live.hooks resolve to the SAME
    # module objects; absolute tools.* imports would be a second module identity on Windows and
    # the restore would miss the wrapped attributes.
    from . import bank_imitation as bi, v2v_live, v2v_runner
    from . import replay_bank as rb
    # hooks() installs process-global adapters (rb.confirmations boot_v2, bank_imitation.run);
    # restore both afterwards so other bank tests in this process stay untouched.
    plain_confirmations, plain_run = rb.confirmations, bi.run
    try:
        common = dict(root='/r/v2v', legacy_root='/r/legacy', source_root='/r/src',
                      common_checkpoint_sha256='a'*64)
        on = v2v_runner.prepare(dict(budget={}), v2v_dim=W, v2v_intent=True, v2v_coop_signal=True, **common)
        assert {k: on[k] for k in CFG} == CFG
        assert on['v2v_coop_signal'] is True and on['v2v_coop_signal_loss_weight'] == 1.
        with pytest.raises(ValueError, match='coop signal block width'):
            v2v_runner.prepare(dict(budget={}), v2v_dim=48, v2v_intent=True, v2v_coop_signal=True, **common)
        with pytest.raises(ValueError, match='requires v2v_intent'):
            v2v_runner.prepare(dict(budget={}), v2v_dim=W, v2v_coop_signal=True, **common)
        off = v2v_runner.prepare(dict(budget={}), v2v_dim=48, v2v_intent=True, **common)
        assert 'v2v_coop_signal' not in off and off['v2v_dim'] == 48
        hooks = v2v_live.hooks(tmp_path, dict(on, accel_bounds=[-3., 2.6]))
        policy = hooks['policy_factory']()
        assert isinstance(policy, vi.IntentPolicy) and policy.coop_enabled
        assert policy.v2v_width == W and policy.coop_loss_weight == 1.
        plain = v2v_live.hooks(tmp_path, dict(off, accel_bounds=[-3., 2.6]))
        p = plain['policy_factory']()
        assert isinstance(p, vi.IntentPolicy) and not p.coop_enabled and not hasattr(p, 'coop_head')
        u0 = tmp_path/'u0.pt'
        mappo_u0(u0)
        loaded, opt, _, _ = hooks['load_checkpoint'](u0, config=on)
        assert loaded.coop_enabled and {id(q) for q in loaded.parameters()} == \
            {id(q) for g in opt.param_groups for q in g['params']}
        assert not hooks['load_checkpoint'](u0, config=off)[0].coop_enabled
    finally:
        rb.confirmations = getattr(plain_confirmations, 'original', plain_confirmations)
        bi.run = plain_run
    assert getattr(rb.confirmations, 'bank_rule', None) is None and bi.run is plain_run
