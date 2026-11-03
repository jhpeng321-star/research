"""CPU gates for planned-command intent sharing (tools/coopmwm/v2v_intent.py), real torch, no SUMO."""
import copy
import gzip
import importlib.util
import json
import math
import os
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
BASE = 'cf3efc9e'  # merge base on the live code: pre-intent sources for the option-off identity gate


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def report_keys(run):
    """Historical shim (pre-imit-fix2 run_separate lacked last_loss/last_gradient_norm); on the
    merged base the keys exist, so it only fills ABSENT keys and is a no-op here."""
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


def cooperative_bank(tmp_path, t0=420, duration=12, drop_context=None, command=None):
    """Plan c0+c1: c1 mirrors c0's plan -> partner intent predicts c1's action.

    drop_context: {second: vid} removes that CAV from the recorded policy context of that second
    (still a partner in the trace frame).  command: override for plan_command."""
    from .exact_grade import imitation_pairs
    command = command or plan_command
    tmp_path.mkdir(parents=True, exist_ok=True)
    steps, frames = [], []
    for t in range(t0-1, t0+duration+H):
        sensor = rows(t-t0)
        context = encode([r for r in sensor if r['vid'] != (drop_context or {}).get(t)])
        raw = {v: command(v, t) for v in ('c0', 'c1')}
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
    kept, counts = bank_v2v.attach_pairs(pairs, trace, t0, intent=dict(horizon=H, entry='v/b'))
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
    assert all(b[s*12+11] == 0. for p, _ in pairs for b in p['context']['v2v'].values() for s in range(4))
    vi.refresh_bank_intents(policy, pairs)
    assert pairs[0][0]['context']['v2v']['c1'][11] == 0.                       # plan start: no t-1 context
    block = pairs[1][0]['context']['v2v']['c1']
    assert block[11] == 1. and block[6:11] == [0.]*H                           # zero-init head at load
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
    fixture_dir = os.environ.get('COOPMWM_INTENT_BASELINE_DIR')
    if fixture_dir:
        # Immutable Linux releases pin these exact git blobs in LS_SOURCE_HASHES.
        # A configured but incomplete fixture is a gate failure, never a skip.
        assert name in ('accel_learner', 'exact_grade', 'bank_imitation', 'v2v_obs')
        baseline = Path(fixture_dir)
        provenance = json.loads((baseline/'PROVENANCE.json').read_text(encoding='utf-8'))
        assert provenance['commit'].startswith(BASE)
        source = (baseline/(name+'.py')).read_text(encoding='utf-8')
    else:
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
    # inputs are built EMPTY: no recorded (hindsight) command ever reaches a partner-intent input
    assert all(b[s*12+6:s*12+12] == [0.]*(H+1) for p, _ in pairs for b in p['context']['v2v'].values()
               for s in range(4))
    assert first['intent_target'][members.index('c1')] == [.5]*H and first['intent_mask'][0] == [1.]*H
    assert last['intent_mask'][0] == [0.]*H                                   # plan end
    json.dumps([p for p, _ in pairs])                                         # links stay JSON-safe
    class Publisher(torch.nn.Module):  # this fixture's contexts carry members only (no x/rsu)
        intent_horizon, intent_k = H, 4

        def publish(self, context):
            return {v: [.1*len(context['members'])+.01*i]*H for i, v in enumerate(context['members'])}
    policy = Publisher()
    report = vi.refresh_bank_intents(policy, pairs)
    assert report['partner_slots_valid'] > 0 and report['partner_slots_no_prev_context'] > 0
    by_key = {(p['intent_link']['entry'], p['intent_link']['t']): p for p, _ in pairs}
    checked = 0
    for p, _ in pairs:
        prev = by_key.get((p['intent_link']['entry'], p['intent_link']['t']-1))
        for ego, slots in p['intent_link']['slots'].items():
            for slot, vid in enumerate(slots):
                got = p['context']['v2v'][ego][slot*12+6:slot*12+12]
                if prev is None:
                    assert got == [0.]*(H+1)
                elif vid in prev['context']['members']:
                    assert got[:H] == pytest.approx(policy.publish(prev['context'])[vid], abs=1e-6) and got[H] == 1.
                    checked += 1
    assert checked > 0
    # the base 6 per-partner features are exactly the v2v-only bank features
    for (a, _), (b, _) in zip(plain[:3], pairs[:3]):
        for vid, block in a['context']['v2v'].items():
            for slot in range(4):
                assert b['context']['v2v'][vid][slot*12:slot*12+6] == block[slot*6:slot*6+6]


# ---------------------------------------------------------------- 10. review fixes (merged branch)

def _intent_inputs(pairs):
    return [{ego: [list(b[s*12+6:s*12+12]) for s in range(4)] for ego, b in p['context']['v2v'].items()}
            for p, _ in pairs]


def test_bank_partner_intents_come_from_current_head_on_t_minus_1_context(tmp_path):
    """Fix 1: no hindsight leak - partner intents = current head on the partner's own t-1 context."""
    policy = nonzero_head(intent_policy(tmp_path)[0])
    pairs = cooperative_bank(tmp_path)
    vi.refresh_bank_intents(policy, pairs)
    checked = 0
    for i, (p, _) in enumerate(pairs):
        for ego, slots in p['intent_link']['slots'].items():
            assert slots == v2v_partner_ids({r['vid']: r for r in rows(i)}, ego, V2VConfig(v2v=True))
            for slot, vid in enumerate(slots):
                got = p['context']['v2v'][ego][slot*12+6:slot*12+12]
                if i == 0:
                    assert got == [0.]*(H+1)
                    continue
                expect = policy.publish(pairs[i-1][0]['context'])[vid]
                assert got[:H] == pytest.approx(expect, abs=1e-6) and got[H] == 1.
                recorded = [plan_command(vid, 420+i+k) for k in range(H)]
                assert got[:H] != pytest.approx(recorded, abs=1e-3)          # never recorded commands
                checked += 1
    assert checked > 0
    # recorded FUTURE commands never reach the inputs: a bank whose plan commands differ only
    # from second 422 on yields identical partner-intent inputs at pairs 420 and 421
    other = cooperative_bank(tmp_path/'o', command=lambda v, t: plan_command(v, t)+(1. if t >= 422 else 0.))
    vi.refresh_bank_intents(policy, other)
    assert _intent_inputs(other)[:2] == _intent_inputs(pairs)[:2]
    # recomputed with the CURRENT head on every refresh
    before = _intent_inputs(pairs)
    with torch.no_grad():
        policy.intent_head.bias.add_(.3)
    vi.refresh_bank_intents(policy, pairs)
    after = _intent_inputs(pairs)
    assert after[0] == before[0] and after[1] != before[1]


def test_partner_validity_is_per_partner_and_targets_per_element(tmp_path):
    """Fix 2: a partner absent from the t-1 context is invalid alone; a missing t-1 pair blanks
    only its successor; own targets mask only the missing element."""
    policy = nonzero_head(intent_policy(tmp_path)[0])
    pairs = cooperative_bank(tmp_path, drop_context={423: 'c3'})
    vi.refresh_bank_intents(policy, pairs)
    p = pairs[4][0]                                                    # second 424, t-1 = 423
    slots = p['intent_link']['slots']['c0']
    block = p['context']['v2v']['c0']
    assert 'c3' in slots and 'c1' in slots
    j = slots.index('c3')*12
    assert block[j+11] == 0. and block[j+6:j+11] == [0.]*H
    assert block[slots.index('c1')*12+11] == 1.
    # an excluded second (pair 6 removed) blanks pair 7 only; pair 8 is valid again
    fresh = cooperative_bank(tmp_path/'gap')
    kept = fresh[:6]+fresh[7:]
    vi.refresh_bank_intents(policy, kept)
    assert all(b[s*12+11] == 0. for b in kept[6][0]['context']['v2v'].values() for s in range(4))
    assert any(b[s*12+11] == 1. for b in kept[7][0]['context']['v2v'].values() for s in range(4))
    # own targets: a missing tape second masks that element only, not the whole horizon
    steps = [dict(time_s=t, commands={'a': 0.}, requests={'a': dict(requested_raw=.1*t)})
             for t in (10, 11, 13, 14, 15, 16)]
    target, mask = vi.own_targets(['a'], 10, vi.commands_by_time(steps, (-3., 2.6)), H)
    assert mask == [[1., 0., 1., 1., 1.]] and target[0][2] == pytest.approx(1.3)
    assert not hasattr(vi, 'tape_board')


def test_intent_loss_detaches_trunk_by_default_and_logs_grad_share(tmp_path):
    """Fix 3: default intent loss trains the head only; option False restores the shared trunk."""
    assert CFG['v2v_intent_detach_trunk'] is True and vi.settings(dict(CFG))['detach_trunk'] is True
    with pytest.raises(ValueError, match='bool'):
        vi.settings(dict(CFG, v2v_intent_detach_trunk='yes'))
    helper_eps = [synthetic_episode()]
    for detach in (True, False):
        path = tmp_path/f'u0_{detach}.pt'
        mappo_u0(path)
        policy = load_checkpoint(path, config=dict(CFG, v2v_intent_detach_trunk=detach))[0]
        nonzero_head(policy)
        assert policy.intent_detach_trunk is detach
        helper = vi.ppo_setup(policy, helper_eps)
        policy.zero_grad()
        helper.term(range(len(helper.contexts))).backward()
        trunk = sum(float(p.grad.abs().sum()) for n, p in policy.named_parameters()
                    if not n.startswith('intent_head') and p.grad is not None)
        head = float(policy.intent_head.weight.grad.abs().sum())
        assert head > 0 and ((trunk == 0.) if detach else (trunk > 0.))
        share = vi.grad_share(policy)
        assert (share == pytest.approx(1.)) if detach else (0. < share < 1.)
    assert vi.IntentPolicy().intent_detach_trunk is True


def test_e2e_u0_intent_exact_update_through_run_separate(tmp_path, monkeypatch):
    """MAPPO-u0 checkpoint, intent ON -> ONE exact_update through the merged run_separate on a
    cooperative bank: pass1_effective, intent loss falls, intent input columns move, refreshed
    partner intents, grad share logged, and every backtrack restores the exact Adam snapshot."""
    from .exact_grade import exact_update
    from .test_imitfix2_cpu import same_adam
    policy, optimizer, generator = intent_policy(tmp_path)
    pairs = cooperative_bank(tmp_path)
    episodes = [rollout(from_weights({k: v.clone() for k, v in policy.state_dict().items()}, enabled=True))]
    norms_before = vi.weight_norms(policy)
    captured = dict(restores=[])
    original_load = optimizer.load_state_dict

    def load(sd):
        original_load(sd)
        captured['restores'].append(copy.deepcopy(optimizer.state_dict()))
    monkeypatch.setattr(optimizer, 'load_state_dict', load)

    def separate(policy_, optimizer_, pairs_, generator_, reference, coefficient=1., passes=5):
        captured['snapshot'] = copy.deepcopy(optimizer_.state_dict())
        captured['restores'] = []
        captured['pair_inputs'] = _intent_inputs(pairs_)
        lrs = [g['lr'] for g in optimizer_.param_groups]
        for g in optimizer_.param_groups:
            g['lr'] = .05                                     # force pass-1 backtracking
        try:
            return bank_imitation.run_separate(policy_, optimizer_, pairs_, generator_, reference,
                                               coefficient, passes)
        finally:
            for g, lr in zip(optimizer_.param_groups, lrs):
                g['lr'] = lr
    monkeypatch.setattr(bank_imitation, 'run', separate)
    torch.manual_seed(0)
    metrics = exact_update(policy, optimizer, episodes, generator, pairs, 1., epochs=1, imitation_epochs=5)
    report = metrics[-1]['bank_imitation']
    assert report['pass1_effective'] is True and report['epochs_completed'] >= 1
    assert report['imitation_failed'] is False and report['kl'] <= .045
    intent = report['v2v_intent']
    assert intent['after']['intent_loss'] < intent['before']['intent_loss']
    assert intent['detach_trunk'] is True and intent['intent_grad_share']['steps'] > 0
    assert 0. < intent['intent_grad_share']['mean'] < 1.
    assert intent['partner_intents_pre_imitation']['partner_slots_valid'] > 0
    norms_after = vi.weight_norms(policy)
    assert norms_before == dict(actor=0., critic=0.) and norms_after['actor'] > 0.   # intent columns moved
    # the imitation phase saw the post-PPO head's (nonzero) intents, not recorded commands
    assert any(any(x != 0. for x in slot[:H]) for row in captured['pair_inputs'][1:] for blocks in row.values()
               for slot in blocks)
    # Adam rollback exactness: every pass-1 backtrack restores the pre-pass Adam snapshot exactly
    attempts = report['epochs'][0]['attempts']
    assert attempts >= 2 and report['backtracks'] >= 1
    for sd in captured['restores'][:attempts-1]:
        same_adam(sd, captured['snapshot'])
    ppo = next(m['v2v_intent'] for m in metrics if 'v2v_intent' in m)
    assert ppo['intent_grad_share']['steps'] > 0 and ppo['detach_trunk'] is True
    json.dumps(metrics)
