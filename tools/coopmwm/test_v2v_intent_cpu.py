"""CPU gates for planned-command intent sharing (tools/coopmwm/v2v_intent.py), real torch, no SUMO."""
import copy
import gzip
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.coopnative.storage import canonical
from .acceleration import Controller, Policy as BasePolicy, encode
from .accel_learner import save
from . import bank_imitation, bank_v2v, v2v_intent as vi
from .v2v_obs import (V2V_DIM, V2VConfig, intent_columns, published_intent, v2v_features,
                      v2v_intent_dim, v2v_intent_features, v2v_partner_ids)
from .v2v_runtime import controller_class, from_weights, load_checkpoint

H = 5
W = v2v_intent_dim(4, H)
CFG = vi.config_keys()
BASE = '56c37cb0'  # branch point: pre-intent sources for the option-off identity gate


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def report_keys(run):
    """Base-commit run_separate omits last_loss/last_gradient_norm (fixed on c4/imit-fix2); the
    shim only fills ABSENT keys so exact_update can build its metric row in this branch."""
    def wrapped(*args, **kwargs):
        report = run(*args, **kwargs)
        for key in ('last_loss', 'last_gradient_norm'):
            report.setdefault(key, 0.)
        return report
    return wrapped


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
    torch.manual_seed(seed)
    policy = BasePolicy()
    torch.nn.init.normal_(policy.actor[-1].weight, std=.3)
    torch.nn.init.normal_(policy.actor[-1].bias, std=.1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    context = encode(rows())
    mean, value = policy(context)
    (mean.sum()+value).backward()
    optimizer.step()
    save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, 0)
    return policy


def intent_policy(tmp_path, seed=7):
    path = tmp_path/'A_checkpoint_000.pt'
    mappo_u0(path, seed)
    policy, optimizer, generator, _ = load_checkpoint(path, config=dict(CFG))
    return policy, optimizer, generator


# ---------------------------------------------------------------- 1. shapes / order

def test_block_shapes_and_partner_order():
    assert W == 48 and V2V_DIM == 24 and CFG['v2v_dim'] == 48
    state = {r['vid']: dict(r, a_mps2=.1) for r in rows()}
    cfg = V2VConfig(v2v=True)
    ids = v2v_partner_ids(state, 'c0', cfg)
    assert ids == ['c1', 'c2', 'c3']  # nearest first; the HDV h0 is never a partner
    board = {'c1': (300, [1., 2., 3., 4., 5.]), 'c2': (299, [9.]*5), 'c3': (300, [1., 2.])}
    out = v2v_intent_features(state, 'c0', {'c1': .5}, board, 301, H, cfg)
    base = v2v_features(state, 'c0', {'c1': .5}, cfg)
    assert out.shape == (W,) and out.dtype == np.float32
    for slot in range(4):
        assert np.array_equal(out[slot*12:slot*12+6], base[slot*6:slot*6+6])
    # slot order = base order: gaps ascending by |gap|
    assert [float(base[s*6]) for s in range(3)] == [20., -30., 50.]
    assert list(out[6:12]) == [1., 2., 3., 4., 5., 1.]          # c1 fresh (published t-1)
    assert list(out[18:24]) == [0.]*6                          # c2 stale (t-2)
    assert list(out[30:36]) == [0.]*6                          # c3 malformed length
    assert list(out[42:48]) == [0.]*6                          # empty 4th slot
    assert intent_columns() == [s*12+6+j for s in range(4) for j in range(6)]
    assert not v2v_intent_features(state, 'c0', {}, board, 301, H, V2VConfig()).any()  # flag off


# ---------------------------------------------------------------- 2. latency t-1 (runtime)

def runtime_controller(policy, seed=4, deterministic=True):
    return controller_class(Controller)(policy, torch.Generator().manual_seed(seed), deterministic)


def nonzero_head(policy, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        policy.intent_head.weight.copy_(torch.randn(policy.intent_head.weight.shape, generator=g))
    return policy


def test_partner_intent_has_one_step_latency(tmp_path):
    policy = nonzero_head(intent_policy(tmp_path)[0])
    c = runtime_controller(policy)
    commands = {r['vid']: .2 for r in rows() if r['is_cav']}
    c.apply(rows(0), commands, 301)
    first = c.events[-1]['sample']
    assert set(first['published_intent']) == set(first['context']['members'])
    assert all(len(v) == W for v in first['context']['v2v'].values())
    assert all(v[s*12+11] == 0. for v in first['context']['v2v'].values() for s in range(4))  # nothing yet
    c.apply(rows(1), commands, 302)
    second = c.events[-1]['sample']
    table = {r['vid']: dict(r) for r in rows(1)}
    slot = v2v_partner_ids(table, 'c0', V2VConfig(v2v=True)).index('c1')
    seen = second['context']['v2v']['c0'][slot*12+6:slot*12+12]
    assert seen[-1] == 1.
    assert seen[:5] == pytest.approx(first['published_intent']['c1'], abs=1e-6)        # t-1
    assert seen[:5] != pytest.approx(second['published_intent']['c1'], abs=1e-6)       # never same-step
    lo, hi = vi.ACCEL_BOUNDS
    assert all(lo <= x <= hi for v in second['published_intent'].values() for x in v)
    # a skipped second makes every received intent stale
    c.apply(rows(3), commands, 304)
    assert all(v[s*12+11] == 0. for v in c.events[-1]['sample']['context']['v2v'].values() for s in range(4))


# ---------------------------------------------------------------- 3. masking / reset / despawn

def test_targets_mask_episode_end_despawn_and_gaps():
    def step(t, members, raws):
        return dict(time_s=t, sample=dict(context=dict(members=members)), commands={v: 0. for v in members},
                    requests={v: dict(requested_raw=r) for v, r in raws.items()})
    steps = [step(300, ['a', 'b'], {'a': 1., 'b': 5.}), step(301, ['a', 'b'], {'a': 1.1, 'b': -9.}),
             step(302, ['a'], {'a': 1.2}), step(304, ['a'], {})]  # b despawned; 303 missing
    out = vi.episode_targets(steps, 3)
    target, mask = out[0]
    assert target[0] == pytest.approx([1.1, 1.2, 0.]) and mask[0] == [1., 1., 0.]   # 303 missing
    assert target[1] == pytest.approx([-3., 0., 0.]) and mask[1] == [1., 0., 0.]   # clip + despawn
    target, mask = out[2]
    assert mask[0] == [0., 1., 0.] and target[0][1] == 0.                           # 304 native 0
    assert out[3][1] == [[0., 0., 0.]]                                              # episode end
    # step_commands: no request -> native base; non-finite dropped; plan tapes use absolute_accels
    assert vi.step_commands(dict(commands={'x': .4, 'y': float('nan')}, requests={})) == {'x': .4}
    assert vi.step_commands(dict(sample=dict(absolute_accels={'p': 9.}))) == {'p': 2.6}


def test_runtime_reset_and_despawn(tmp_path):
    policy = nonzero_head(intent_policy(tmp_path)[0])
    commands = {r['vid']: 0. for r in rows() if r['is_cav']}
    c = runtime_controller(policy)
    c.apply(rows(0), commands, 301)
    c.apply(rows(1, drop=('c1',)), {k: v for k, v in commands.items() if k != 'c1'}, 302)
    sample = c.events[-1]['sample']
    assert 'c1' not in sample['published_intent'] and 'c1' not in sample['context']['members']
    table = {r['vid']: r for r in rows(1, drop=('c1',))}
    assert 'c1' not in v2v_partner_ids(table, 'c0', V2VConfig(v2v=True))
    # c1 respawns at 303 without having published at 302: its slot is present but intent-invalid
    c.apply(rows(2), commands, 303)
    table = {r['vid']: r for r in rows(2)}
    slot = v2v_partner_ids(table, 'c0', V2VConfig(v2v=True)).index('c1')
    block = c.events[-1]['sample']['context']['v2v']['c0']
    assert block[slot*12+5] == 1. and block[slot*12+11] == 0.
    # a new controller (new episode) starts with an empty board
    fresh = runtime_controller(policy)
    fresh.apply(rows(2), commands, 303)
    assert all(v[s*12+11] == 0. for v in fresh.events[-1]['sample']['context']['v2v'].values() for s in range(4))
    assert published_intent({'z': (302, [0.]*5)}, 'z', 303, 5) is not None
    assert published_intent({'z': (303, [0.]*5)}, 'z', 303, 5) is None


# ---------------------------------------------------------------- 4. actions identical at load

def test_actions_identical_to_current_v2v_policy_at_load(tmp_path):
    path = tmp_path/'u0.pt'
    mappo_u0(path)
    current, _, _, _ = load_checkpoint(path, config=dict(v2v=True, v2v_dim=V2V_DIM))
    intent, opt, _, _ = load_checkpoint(path, config=dict(CFG))
    assert isinstance(intent, vi.IntentPolicy) and not isinstance(current, vi.IntentPolicy)
    assert {id(p) for p in intent.parameters()} == {id(p) for g in opt.param_groups for p in g['params']}
    context = encode(rows())
    table = {r['vid']: dict(r, a_mps2=.3) for r in rows()}
    last = {'c0': .5, 'c1': -1., 'c2': .2, 'c3': 0.}
    board = {v: (300, [1.5, -.5, .25, 2., -3.]) for v in last}
    narrow = dict(context, v2v={v: v2v_features(table, v, last, V2VConfig(v2v=True)).tolist()
                                for v in context['members']})
    wide = dict(context, v2v={v: v2v_intent_features(table, v, last, board, 301, H, V2VConfig(v2v=True)).tolist()
                              for v in context['members']})
    assert any(x != 0. for b in wide['v2v'].values() for x in b[6:12])
    for deterministic in (True, False):
        a, ma = current.sample(narrow, torch.Generator().manual_seed(9), deterministic, {'c0': .1})
        b, mb = intent.sample(wide, torch.Generator().manual_seed(9), deterministic, {'c0': .1})
        assert a['action'] == b['action'] and a['logp'] == b['logp'] and ma == mb
        assert a['value'] == pytest.approx(b['value'], abs=1e-7)
    assert torch.equal(current.score(a)[0], intent.score(b)[0])
    assert float(intent.intent(wide).detach().abs().max()) == 0.       # zero-init head
    assert vi.weight_norms(intent) == dict(actor=0., critic=0.)


# ---------------------------------------------------------------- 5. head learns a synthetic sequence

def synthetic_episode(n=16):
    """Constant per-vehicle commands that depend on observable speed/lane."""
    steps = []
    for i in range(n):
        speed = {'c0': 8.+.1*i, 'c1': 12., 'c2': 10.-.05*i, 'c3': 9.}
        sensor = rows(0, speed=speed)
        context = encode(sensor)
        table = {r['vid']: dict(r, a_mps2=0.) for r in sensor}
        context['v2v'] = {v: v2v_intent_features(table, v, {}, {}, 300+i, H, V2VConfig(v2v=True)).tolist()
                          for v in context['members']}
        commands = {r['vid']: .25*(r['v_mps']-10.)-.5*r['lane'] for r in sensor if r['is_cav']}
        steps.append(dict(time_s=300+i, sample=dict(context=context), commands=commands, requests={}))
    return dict(steps=steps)


def test_intent_head_learns_synthetic_sequence(tmp_path):
    policy, _, _ = intent_policy(tmp_path)
    helper = vi.ppo_setup(policy, [synthetic_episode()])
    before = helper.stats()['intent_loss']
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(80):
        loss = helper.term(range(len(helper.contexts)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after = helper.stats()['intent_loss']
    assert before > .05 and after < .1*before


# ---------------------------------------------------------------- 6. one cooperative bank update

def rollout(policy, seconds=9, seed=5):
    c = runtime_controller(policy, seed, deterministic=False)
    steps = []
    for i in range(seconds):  # stochastic steps avoid multiples of 10 (no regime/leverprobe lookup)
        sensor = rows(i)
        commands = {r['vid']: .1*math.sin(i+k) for k, r in enumerate(sensor) if r['is_cav']}
        c.apply(sensor, commands, 311+i)
        steps.append(dict(c.events[-1], reward=-1.-.1*i))
    return dict(steps=steps, world=dict(seed=1, cell='x'))


def plan_command(vid, t):
    return 1.5*math.sin(.7*t) if vid in ('c0', 'c1') else 0.


def cooperative_bank(tmp_path, t0=420, duration=12):
    """Plan c0+c1: c1 mirrors c0's plan -> partner intent predicts c1's action."""
    from .exact_grade import imitation_pairs
    steps, frames = [], []
    for t in range(t0-1, t0+duration+H):
        sensor = rows(t-t0)
        context = encode(sensor)
        raw = {v: plan_command(v, t) for v in ('c0', 'c1')}
        commands = {r['vid']: 0. for r in sensor if r['is_cav']}
        steps.append(dict(time_s=t, commands=commands, requests={v: dict(requested_raw=x) for v, x in raw.items()},
            sample=dict(context=context, action=[[raw.get(v, 0.)] for v in context['members']],
                        absolute_accels=raw, behaviour=dict(executed=True, plan_id='p'),
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
    kept, counts = bank_v2v.attach_pairs(pairs, trace, t0, intent=dict(horizon=H, board=vi.tape_board(tape, H)))
    assert counts['kept'] == duration
    return kept


# Legacy run() bounds KL against the PRE-PPO policy; with this cold-Adam fixture one PPO step
# already exceeds .045, so the legacy variant runs the imitation phase with zero PPO epochs.
@pytest.mark.parametrize('runner,epochs', [('run_separate', 1), ('run', 0)])
def test_cooperative_bank_update_lowers_action_and_intent_error(tmp_path, monkeypatch, runner, epochs):
    """End to end from MAPPO u0: intent loader -> wired runtime rollout (worker weights) ->
    bank pairs with partner intents -> ONE exact_update (PPO + imitation, run or run_separate)."""
    from .exact_grade import exact_update
    from .bank_imitation import action_error
    policy, optimizer, generator = intent_policy(tmp_path)
    pairs = cooperative_bank(tmp_path)
    block = pairs[0][0]['context']['v2v']['c1']
    assert block[11] == 1. and block[6] == pytest.approx(plan_command('c0', 420))    # c0 intent seen by c1
    mask = pairs[-1][0]['intent_mask']
    members = pairs[-1][0]['context']['members']
    assert mask[members.index('c0')] == [0.]*H and mask[members.index('c3')] == [0.]*H  # plan end / non-member
    assert pairs[0][0]['intent_target'][members.index('c0')][0] == pytest.approx(plan_command('c0', 421))
    worker = from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True)
    episodes = [rollout(worker)]   # spawned-worker reconstruction, as accel_plant.episode does
    assert sum(b[s*12+11] for st in episodes[0]['steps'][1:] for b in st['sample']['context']['v2v'].values()
               for s in range(4)) > 0                      # partner intents broadcast during the rollout
    monkeypatch.setattr(bank_imitation, 'run', report_keys(getattr(bank_imitation, runner)))
    torch.manual_seed(0)
    before = action_error(policy, pairs)
    metrics = exact_update(policy, optimizer, episodes, generator, pairs, 1., epochs=epochs, imitation_epochs=5)
    report = metrics[-1]['bank_imitation']
    assert report['action_error_after_mps2'] < report['action_error_before_mps2'] and action_error(policy, pairs) < before
    intent = report['v2v_intent']
    assert intent['after']['intent_loss'] < intent['before']['intent_loss']
    assert 0. < intent['partner_intent_coverage'] <= 1.
    assert intent['weight_norm_after']['actor'] > 0. and vi.weight_norms(policy)['actor'] > 0.
    ppo = next(m['v2v_intent'] for m in metrics if 'v2v_intent' in m)
    assert ppo['intent_loss_after'] < ppo['intent_loss_before'] or not epochs
    assert ppo['partner_intent_coverage'] is not None and 0. < ppo['partner_intent_coverage'] <= 1.
    assert ppo['partner_intent_weight_norm_before'] == dict(actor=0., critic=0.)
    json.dumps(metrics)  # update receipts stay JSON-serialisable


# ---------------------------------------------------------------- 7. checkpoint round trip

def test_checkpoint_save_load_roundtrip(tmp_path):
    policy, optimizer, generator = intent_policy(tmp_path)
    nonzero_head(policy)
    context = encode(rows())
    table = {r['vid']: r for r in rows()}
    context['v2v'] = {v: v2v_intent_features(table, v, {}, {'c1': (300, [.1]*5)}, 301, H,
                                             V2VConfig(v2v=True)).tolist() for v in context['members']}
    (policy.intent(context).sum()+policy(context)[1]).backward()
    optimizer.step()
    path = tmp_path/'A_checkpoint_001.pt'
    save(path, policy, optimizer, generator, 1, 0)
    restored, opt2, _, payload = load_checkpoint(path, config=dict(CFG))
    assert payload['update'] == 1 and isinstance(restored, vi.IntentPolicy)
    for (k, a), (k2, b) in zip(policy.state_dict().items(), restored.state_dict().items()):
        assert k == k2 and torch.equal(a, b)
    assert torch.equal(restored.intent(context), policy.intent(context))
    assert len(opt2.state) == len(optimizer.state) > 0
    for a, b in zip(opt2.state_dict()['state'].values(), optimizer.state_dict()['state'].values()):
        assert all(torch.equal(a[k], b[k]) for k in a)
    worker = from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True)
    assert isinstance(worker, vi.IntentPolicy) and worker.publish(context) == policy.publish(context)
    with pytest.raises(ValueError, match='explicit V2V'):
        from_weights(policy.state_dict())
    with pytest.raises(ValueError, match='horizon'):
        load_checkpoint(path, config=dict(vi.config_keys(horizon=3)))


# ---------------------------------------------------------------- 8. option off is byte-identical

def _base_module(name, tmp_path):
    """Import the pre-intent source of tools/coopmwm/<name>.py at the branch point."""
    try:
        source = subprocess.run(['git', 'show', f'{BASE}:tools/coopmwm/{name}.py'], capture_output=True,
                                text=True, check=True, cwd=Path(__file__).resolve().parents[2]).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('branch-point source unavailable')
    path = tmp_path/f'base_{name}.py'
    path.write_text(source, encoding='utf-8')
    package = __name__.rpartition('.')[0]  # the package THIS test imported (pytest may root it)
    spec = importlib.util.spec_from_file_location(f'{package}._base_{name}', path,
                                                  submodule_search_locations=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plain_episode(policy):
    from .v2v_runtime import Policy as V2VPolicy
    assert not isinstance(policy, vi.IntentPolicy)
    c = controller_class(Controller)(policy, torch.Generator().manual_seed(5), False) if isinstance(
        policy, V2VPolicy) else Controller(policy, torch.Generator().manual_seed(5), False)
    steps = []
    for i in range(9):
        sensor = rows(i)
        c.apply(sensor, {r['vid']: .1*i for r in sensor if r['is_cav']}, 311+i)
        steps.append(dict(c.events[-1], reward=-1.-.2*i))
    return dict(steps=steps, world=dict(seed=1, cell='x'))


def _states(policy, optimizer):
    return [v.clone() for v in policy.state_dict().values()], json.dumps(
        {k: v for k, v in optimizer.state_dict()['param_groups'][0].items() if k != 'params'})


def test_option_off_is_byte_identical_to_branch_point(tmp_path):
    from . import accel_learner, exact_grade
    from .v2v_runtime import Policy as V2VPolicy
    old_learner = _base_module('accel_learner', tmp_path)
    old_grade = _base_module('exact_grade', tmp_path)
    old_imit = _base_module('bank_imitation', tmp_path)
    old_obs = _base_module('v2v_obs', tmp_path)
    # observation contract unchanged
    rng = np.random.default_rng(1)
    for _ in range(20):
        state = {f'v{i}': dict(s_m=float(rng.uniform(-150, 150)), v_mps=float(rng.uniform(0, 30)),
                               a_mps2=float(rng.normal()), lane=int(rng.integers(0, 3)),
                               is_cav=bool(rng.random() < .7)) for i in range(8)}
        last = {k: float(rng.normal()) for k in state}
        for ego in state:
            assert np.array_equal(v2v_features(state, ego, last, V2VConfig(v2v=True)),
                                  old_obs.v2v_features(state, ego, last, old_obs.V2VConfig(v2v=True)))
    for factory in (BasePolicy, V2VPolicy):
        torch.manual_seed(11)
        p0 = factory()
        torch.nn.init.normal_(p0.actor[-1].weight, std=.3)
        episodes = [_plain_episode(p0)]
        assert all('published_intent' not in s['sample'] for s in episodes[0]['steps'])
        outs = []
        for learner in (accel_learner, old_learner):
            p = copy.deepcopy(p0)
            opt = torch.optim.Adam(p.parameters(), lr=3e-4)
            metrics = learner.update(p, opt, copy.deepcopy(episodes), torch.Generator().manual_seed(3), epochs=2)
            outs.append((json.dumps(metrics), *_states(p, opt)))
        assert outs[0][0] == outs[1][0] and outs[0][2] == outs[1][2]
        assert all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))
        if factory is V2VPolicy:
            continue
        pairs = [(dict(s['sample'], behaviour=dict(executed=True, proposal_members=s['sample']['context']['members'])),
                  1.) for s in episodes[0]['steps']]
        outs = []
        for grade, imit in ((exact_grade, bank_imitation), (old_grade, old_imit)):
            p = copy.deepcopy(p0)
            opt = torch.optim.Adam(p.parameters(), lr=3e-4)
            saved = bank_imitation.run
            bank_imitation.run = report_keys(imit.run_separate)
            try:
                metrics = grade.exact_update(p, opt, copy.deepcopy(episodes), torch.Generator().manual_seed(3),
                                             copy.deepcopy(pairs), 1., epochs=1, imitation_epochs=5)
            finally:
                bank_imitation.run = saved
            outs.append((json.dumps(metrics), *_states(p, opt)))
        new, old = json.loads(outs[0][0]), json.loads(outs[1][0])
        assert outs[0][0] == outs[1][0], [(k, new[-1]['bank_imitation'].get(k), old[-1]['bank_imitation'].get(k))
                                          for k in new[-1]['bank_imitation']
                                          if new[-1]['bank_imitation'].get(k) != old[-1]['bank_imitation'].get(k)]
        assert all(torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))


# ---------------------------------------------------------------- 9. loaders and wiring

def test_loads_u0_v2v_only_and_intent_checkpoints_with_identical_actions(tmp_path):
    u0 = tmp_path/'u0.pt'
    base = mappo_u0(u0)
    # a trained v2v-only (block 24) checkpoint with NONZERO partner weights and warmed Adam
    v2v, opt, gen, _ = load_checkpoint(u0, config=dict(v2v=True, v2v_dim=V2V_DIM))
    with torch.no_grad():
        v2v.actor[0].weight[:, 29:29+V2V_DIM].normal_(std=.2)
        v2v.critic[0].weight[:, 29:29+V2V_DIM].normal_(std=.2)
    context = encode(rows())
    table = {r['vid']: dict(r, a_mps2=.3) for r in rows()}
    last = {'c0': .5, 'c1': -1., 'c2': .2, 'c3': 0.}
    narrow = dict(context, v2v={v: v2v_features(table, v, last, V2VConfig(v2v=True)).tolist()
                                for v in context['members']})
    board = {v: (300, [1.]*H) for v in last}
    wide = dict(context, v2v={v: v2v_intent_features(table, v, last, board, 301, H, V2VConfig(v2v=True)).tolist()
                              for v in context['members']})
    mean, value = v2v(narrow)
    (mean.sum()+value).backward()
    opt.step()
    path = tmp_path/'v2v_only.pt'
    save(path, v2v, opt, gen, 2, 0)
    for source, reference, ctx in ((u0, base, context), (path, v2v, narrow)):
        policy, optimizer, _, _ = load_checkpoint(source, config=dict(CFG))
        assert isinstance(policy, vi.IntentPolicy) and policy.actor[0].weight.shape[1] == 33+W
        assert {id(p) for p in policy.parameters()} == {id(p) for g in optimizer.param_groups for p in g['params']}
        a, _ = reference.sample(ctx, torch.Generator().manual_seed(9), False, {})
        b, _ = policy.sample(wide, torch.Generator().manual_seed(9), False, {})
        assert np.allclose(a['action'], b['action'], rtol=0., atol=1e-6) and a['logp'] == pytest.approx(b['logp'], abs=1e-5)
        assert vi.weight_norms(policy) == dict(actor=0., critic=0.)
        mean, value = policy(wide)
        (policy.intent(wide).square().sum()+mean.sum()+value).backward()
        optimizer.step()      # widened Adam moments line up with the widened parameters
    # v2v-only moments were re-gridded, not dropped
    assert any(v['exp_avg'].shape == (64, 33+W) for v in optimizer.state_dict()['state'].values())
    # head-bearing checkpoint round trip
    head = tmp_path/'intent.pt'
    save(head, policy, optimizer, torch.Generator().manual_seed(1), 3, 0)
    again, _, _, _ = load_checkpoint(head, config=dict(CFG))
    assert torch.equal(again.intent(wide), policy.intent(wide))
    # intent OFF keeps the current v2v loader and class
    off, _, _, _ = load_checkpoint(u0, config=dict(v2v=True, v2v_dim=V2V_DIM))
    assert not isinstance(off, vi.IntentPolicy) and off.actor[0].weight.shape[1] == 33+V2V_DIM


def test_live_hooks_and_runner_binding_are_width_aware(tmp_path, monkeypatch):
    from . import replay_bank as rb, v2v_live, v2v_runner
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations)    # restored after hooks() installs
    monkeypatch.setattr(bank_imitation, 'run', bank_imitation.run)
    common = dict(root='/r/v2v', legacy_root='/r/legacy', source_root='/r/src', common_checkpoint_sha256='a'*64)
    off = v2v_runner.prepare(dict(budget={}), v2v_dim=V2V_DIM, **common)
    assert 'v2v_intent' not in off and off['v2v_dim'] == V2V_DIM
    on = v2v_runner.prepare(dict(budget={}), v2v_dim=W, v2v_intent=True, **common)
    assert {k: on[k] for k in CFG} == CFG
    assert {k: v for k, v in on.items() if not k.startswith('v2v')} == {
        k: v for k, v in off.items() if not k.startswith('v2v')}
    with pytest.raises(ValueError, match='intent block width'):
        v2v_runner.prepare(dict(budget={}), v2v_dim=V2V_DIM, v2v_intent=True, **common)
    hooks = v2v_live.hooks(tmp_path, dict(on, accel_bounds=[-3., 2.6]))
    policy = hooks['policy_factory']()
    assert isinstance(policy, vi.IntentPolicy) and policy.v2v_width == W and policy.intent_horizon == H
    assert hooks['bank_pairs'].func is rb.bank_pairs
    assert hooks['bank_pairs'].keywords['v2v_intent']['v2v_intent'] is True
    u0 = tmp_path/'u0.pt'
    mappo_u0(u0)
    assert isinstance(hooks['load_checkpoint'](u0, config=on)[0], vi.IntentPolicy)
    plain = v2v_live.hooks(tmp_path, off)
    assert plain['bank_pairs'] is rb.bank_pairs
    assert not isinstance(plain['policy_factory'](), vi.IntentPolicy)
    assert not isinstance(plain['load_checkpoint'](u0, config=off)[0], vi.IntentPolicy)


@pytest.fixture
def bank_budget(tmp_path):
    """Same additive bank binding as test_replay_bank_confirm_cpu.bank_budget (not autouse here)."""
    from .warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (tmp_path/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (tmp_path/'AUTHORITY.md').write_text('test-bank-budget')


def test_bank_pairs_carry_intents_through_the_registered_bank(tmp_path, bank_budget):
    from . import replay_bank as rb
    from .test_replay_bank_v2v_cpu import V2VTapeRunner, _bank
    root = tmp_path
    bank = rb.bank_root(root, 'B')
    bank.mkdir(parents=True)
    _bank(root, bank, lambda work: V2VTapeRunner(work))
    plain, s1 = rb.bank_pairs(root, 'B', 12345, v2v=True)
    pairs, s2 = rb.bank_pairs(root, 'B', 12345, v2v=True, v2v_intent=dict(CFG))
    assert json.dumps(s1) == json.dumps(s2) and len(pairs) == len(plain) == 120
    first, last = pairs[0][0], pairs[-1][0]
    members = first['context']['members']
    assert all(len(b) == W for b in first['context']['v2v'].values())
    assert first['context']['v2v']['c1'][6:12] == pytest.approx([.5]*H+[1.])   # c2 plan intent at t-1
    assert first['intent_target'][members.index('c1')] == [.5]*H and first['intent_mask'][0] == [1.]*H
    assert last['intent_mask'][0] == [0.]*H                                   # plan end
    assert last['context']['v2v']['c1'][11] == 0.                             # tape ends: no broadcast
    # the base 6 per-partner features are exactly the v2v-only bank features
    for (a, _), (b, _) in zip(plain[:3], pairs[:3]):
        for vid, block in a['context']['v2v'].items():
            for slot in range(4):
                assert b['context']['v2v'][vid][slot*12:slot*12+6] == block[slot*6:slot*6+6]
