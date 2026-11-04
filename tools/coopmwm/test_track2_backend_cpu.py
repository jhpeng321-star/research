"""P1 production path contract tests; real models/controllers, no simulator."""
from copy import deepcopy
import json
import sys
import types

import numpy as np
import pytest
import torch

from . import track2_backend as tb
from .acceleration import Controller
from .v2v_intent import IntentPolicy
from .v2v_runtime import controller_class


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    from . import exploration
    # Deployment supplies leverprobe; no imported simulator in these CPU tests.
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def cfg(tmp_path):
    return dict(track2_backend=True, run_root=str(tmp_path), training_seed=123,
        v2v=True, v2v_dim=48, v2v_intent=True, merge_features=True, host='cpuV',
        layer_config=dict(hidden_dim=8, learning_rate=.0003, kl_budget=.045,
            clip_ratio=.2, max_grad_norm=.5), trigger_candidates=4, trigger_partner_slots=5,
        decision_interval_s=60, trigger_lr=.0003, layer_epochs=1,
        trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=.5),
        worlds_by_update={'1': [dict(seed=1), dict(seed=2)]}, practice_starts=False,
        runtime='unused', checkpoint='unused', accel_bounds=[-3, 2.6], gpu_deadline=9999999999,
        workers=2, load_max=40, budget=dict(total=5000000, phases=dict(track2_train=4000000)),
        safety_reference=dict(by_world={}, unmatched_zero_event_benchmark=True, source='CPU fixture'),
        track2_reward=dict(weight=.5, provenance='owner registration; CPU fixture'),
        imitation_epochs=5)


def rows(extra=False):
    out = [dict(vid='c1', s_m=-200., v_mps=2., lane=0, is_cav=True, length_m=5.),
           dict(vid='c2', s_m=-205., v_mps=2., lane=1, is_cav=True, length_m=5.),
           dict(vid='h1', s_m=-210., v_mps=0., lane=2, is_cav=False, length_m=5.)]
    if extra:
        out.append(dict(vid='far', s_m=500., v_mps=30., lane=2, is_cav=False, length_m=5.))
    return out


def policy(cfg):
    torch.manual_seed(7)
    return tb.EpisodePolicy(IntentPolicy(), cfg, 'B')


def sample(cfg, now=300):
    p = policy(cfg)
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    p.observe(rows(), now)
    controller.apply(rows(), {r['vid']: 0. for r in rows() if r['is_cav']}, now)
    return p, controller.events[-1]['sample'], controller


def fake_load(monkeypatch):
    def load(cfg, arm, update):
        torch.manual_seed(7)
        base = IntentPolicy()
        return base, torch.optim.Adam(base.parameters(), lr=.0003), torch.Generator().manual_seed(3), {'update': update}
    monkeypatch.setattr(tb, '_load_base', load)


def reward_result(cfg, sample, arm='B', *, outcomes=None):
    """Test-only P2 output from real-outcome inputs, never WM estimates."""
    outcomes = outcomes or dict(c1=1., c2=-2., h1=3.)
    registration = cfg['track2_reward']
    rows = []
    for vid in reversed(sample['context']['members']):
        neighbours = {v: outcomes[v] for v in sample['track2']['reward_neighbourhoods'][vid]}
        reward = outcomes[vid]+registration['weight']*(sum(neighbours.values())/len(neighbours)
                                                       if neighbours else 0.)
        rows.append(dict(episode_id='ep1', time_s=300, cav_id=vid, source='sumo',
            own_reward=outcomes[vid], neighbour_rewards=neighbours, reward=reward,
            weight=registration['weight'], weight_provenance=registration['provenance']))
    return dict(schema=tb.REWARD_SCHEMA, arm=arm, update=1, rows=rows)


def test_default_off_exact_objects():
    base = object()
    assert tb.episode_components({}, base, Controller) == (base, Controller)


def test_live_zero_init_identity_rng_and_local_receipts(cfg):
    p, s, controller = sample(cfg)
    context = s['context']
    expected, _ = p.layer.base.sample(context, torch.Generator().manual_seed(9), False, {})
    assert s['action'] == expected['action']
    assert torch.equal(controller.generator.get_state(), torch.Generator().manual_seed(9).get_state()) is False
    # Same generator advancement, including stochastic trigger draws on a separate stream.
    g = torch.Generator().manual_seed(9)
    p.layer.base.sample(context, g, False, {})
    assert torch.equal(g.get_state(), controller.generator.get_state())
    assert s['track2_trigger']['tables'][0][1] is not None  # real T1A -> T2A adapter
    assert set(context['merge_features']) == {'c1', 'c2'}
    assert all(len(x) == tb.MERGE_DIM for x in context['merge_features'].values())
    assert float(p.layer.score(s)[0].detach()) == pytest.approx(s['logp'])
    assert set(controller.events[-1]['requests']) <= {'c1', 'c2'}
    assert s['track2']['schema'] == tb.SCHEMA


def test_sensor_privacy_and_t_minus_one_reports(cfg):
    p = policy(cfg)
    p.observe(rows(), 300)
    first = deepcopy(p.blocks)
    p.observe(rows(extra=True), 301)
    assert p.blocks['c1'][tb.MERGE_FEATURES.index('partner_report_count')] == 1
    assert p.blocks['c1'][tb.MERGE_FEATURES.index('partner_report_age_s')] == 1
    assert 'far' not in {r['vid'] for r in p.local_states['c1']['rows']}
    assert first['c1'][tb.MERGE_FEATURES.index('partner_reports_missing')] == 1
    p.observe(rows(extra=True), 303)
    assert p.blocks['c1'][tb.MERGE_FEATURES.index('partner_reports_missing')] == 1
    from tools.coopnative.policy import encode
    context = encode(rows())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    a, _ = p.sample(context, torch.Generator().manual_seed(1))
    context['rsu'] = [1e6]*12
    b, _ = p.sample(context, torch.Generator().manual_seed(1))
    assert tb._features(a['context']).tolist() == tb._features(b['context']).tolist()
    assert p.trigger._hidden(a['context']).tolist() == p.trigger._hidden(b['context']).tolist()


def test_real_move_commitment_hold_release_and_no_hdv(cfg):
    p = policy(cfg)
    move = tb.TriggerMove(tb.moves.Move('wall', ('c1', 'c2'), 0, 20, .3))
    assert p.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=300, until=320)
    assert not p.executor.start(move, ego='c2', observable_cav_ids=['c1', 'c2'], now=300, until=320)
    from tools.coopnative.policy import encode
    context = encode(rows())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    p.last_decision = 300
    p.layer.trigger_gain.data.fill_(.5)
    p.observe(rows(), 301)
    s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['context']['move_offsets'] == dict(c1=-.3, c2=-.3)
    assert np.asarray(s['action']).ravel().tolist() == pytest.approx([-.3*np.tanh(.5)]*2)
    p.observe(rows(), 320)
    s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['context']['move_offsets'] == dict(c1=0., c2=0.)
    # Walls release at the observable merge boundary even before expiry.
    p.executor = tb.MoveExecutor(moves_api=tb.MoveAPI())
    p.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=320, until=340)
    near = rows()
    near[0]['s_m'] = -100.
    near[1]['s_m'] = -105.
    p.observe(near, 321)
    s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['context']['move_offsets'] == dict(c1=0., c2=0.)


def test_production_adapter_unobserved_wall_partner_crossing_is_invariant(cfg):
    p = policy(cfg)
    for parameter in p.layer.base.parameters():
        parameter.data.zero_()
    p.layer.trigger_gain.data.fill_(.5)
    job = dict(track2_backend=True, track2_config=cfg, arm='B',
               track2_state=p.worker_state(), noise_seed=81)
    results = []
    for partner_position in (-101., -100.):
        out, factory = tb.episode_components(job, p.layer.base, controller_class(Controller))
        controller = factory(out, torch.Generator().manual_seed(2), True)
        move = tb.TriggerMove(tb.moves.Move('wall', ('c1', 'c2'), 0, 20, .3))
        assert out.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=300, until=320)
        out.last_decision = 300
        separated = rows()[:2]
        separated[0]['s_m'], separated[1]['s_m'] = -300., partner_position
        controller.apply(separated, {'c1': 0., 'c2': 0.}, 319)
        sample = controller.events[-1]['sample']
        assert out.visible['c1'] == ['c1']
        assert sample['context']['move_offsets']['c1'] == -.3
        assert sample['track2']['active_moves'][0]['until'] == 320
        results.append((sample['action'][0], out.blocks['c1'], out.local_states['c1']))
        controller.apply(separated, {'c1': 0., 'c2': 0.}, 320)
        assert controller.events[-1]['sample']['context']['move_offsets']['c1'] == 0.
    assert results[0] == results[1]


@pytest.mark.parametrize('report_time,expected', [(318, 0.), (317, -.3)])
def test_wall_release_accepts_only_authorized_t_minus_one_report(cfg, report_time, expected):
    p = policy(cfg)
    local = rows()[:2]
    local[0]['s_m'], local[1]['s_m'] = -300., -100.
    local.append(dict(local[0], vid='c3', s_m=-200.))
    p.observe(local, report_time)
    move = tb.TriggerMove(tb.moves.Move('wall', ('c1', 'c2'), 0, 20, .3))
    p.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=300, until=320)
    p.last_decision = 300
    p.observe(local, 319)
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(2), True)
    controller.apply(local, {'c1': 0., 'c2': 0., 'c3': 0.}, 319)
    assert controller.events[-1]['sample']['context']['move_offsets']['c1'] == expected


def test_spawn_state_and_practice_factory_wrapper(cfg):
    p = policy(cfg)
    job = dict(track2_backend=True, track2_config=cfg, arm='B', track2_state=p.worker_state(), noise_seed=81)
    # Practice supplies a callable controller factory, not a subclassable type.
    out, factory = tb.episode_components(job, p.layer.base,
        lambda *a, **k: controller_class(Controller)(*a, **k))
    controller = factory(out, torch.Generator().manual_seed(2), False)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    json.dumps(s)
    json.dumps(tb._batch_dict(tb._batch(s['track2_trigger'])))
    assert out.now == 300
    assert out.layer.base.state_dict().keys() == p.layer.base.state_dict().keys()
    for key, value in p.layer.base.state_dict().items():
        assert torch.equal(out.layer.base.state_dict()[key], value)
    deepcopy(out)  # bank imitation's transactional reference must be copyable


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_observable_sumo_rewards_update_layer_and_trigger_freeze_base(cfg, monkeypatch, arm):
    fake_load(monkeypatch)
    hooks = tb.build_arm_runtime(cfg, arm)
    runtime = hooks['runtime']
    runtime.safety_checked = True  # safety's own paired adapter is tested separately
    p = runtime.policy
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    p.observe(rows(), 300)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    episodes = [dict(id='ep1', arm=arm, steps=[dict(time_s=300, sample=s, reward=-999999.)])]
    before = deepcopy(p.layer.base.state_dict())
    weights = deepcopy(p.layer.residual.state_dict())
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {'pairs': 0}))
    targets = []
    layer_targets = []
    layer_update = p.layer.update_rewards
    def capture_layer(records, **kwargs):
        layer_targets.extend(r['returns'] for r in records)
        return layer_update(records, **kwargs)
    monkeypatch.setattr(p.layer, 'update_rewards', capture_layer)
    update = tb.ppo_update
    def capture(model, optimizer, batch, advantages, returns, **kw):
        targets.append((advantages, returns))
        return update(model, optimizer, batch, advantages, returns, **kw)
    monkeypatch.setattr(tb, 'ppo_update', capture)
    result = hooks['train_update'](episodes, rewards=reward_result(cfg, s, arm))
    assert result['layer'][0]['accepted']
    assert result['trigger'][0]['cav_count'] == 2
    expected_returns = [1.25/tb.UNIT, -1./tb.UNIT]
    expected_advantages = [r-v for r, v in zip(expected_returns, s['track2_trigger']['values'])]
    assert targets == [(expected_advantages, expected_returns)]
    assert layer_targets == [expected_returns]
    assert result['dense'] == result['wm_credit'] == dict(enabled=False)
    assert result['reward']['registration'] == cfg['track2_reward']
    assert any(not torch.equal(weights[k], v) for k, v in p.layer.residual.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in p.layer.base.state_dict().items())
    assert all(v.grad is None and not v.requires_grad for v in p.layer.base.parameters())


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_production_delayed_onset_full_hold_outcomes_train_originating_trigger(cfg, monkeypatch, arm):
    """Same real rollout/actions, opposite later outcomes must teach both heads."""
    fake_load(monkeypatch)
    cfg['trigger_ppo']['entropy_weight'] = 0.  # isolate the reward-dependent update
    def candidates(state, rng, n):
        # Only the candidate boundary is fake: T1A legal registered parameters.
        return [tb.TriggerMove(tb.moves.Move('ease_off', (state['self'],), 2, 20, .3))
                for _ in range(n)]
    monkeypatch.setattr(tb.MoveAPI, 'candidates', staticmethod(candidates))
    snapshots, tapes, learning_targets = [], [], []
    for sign in (-1., 1.):
        runtime = tb.build_arm_runtime(cfg, arm)['runtime']
        runtime.safety_checked = True
        p = runtime.policy
        with torch.no_grad():
            p.layer.trigger_gain.fill_(.5)
            p.trigger.trigger_value.weight.zero_()
            p.trigger.trigger_value.bias.zero_()
        frozen = deepcopy(p.layer.base.state_dict())
        job = dict(track2_backend=True, track2_config=cfg, arm=arm,
                   track2_state=p.worker_state(), noise_seed=123)
        worker, factory = tb.episode_components(job, p.layer.base, controller_class(Controller))
        controller = factory(worker, torch.Generator().manual_seed(9), False)
        steps, reward_rows = [], []
        for t in range(300, 364):  # includes onset, complete hold, and next decision
            controller.apply(rows(), {'c1': 0., 'c2': 0.}, t)
            s = controller.events[-1]['sample']
            steps.append(dict(time_s=t, sample=s))
            outcome = sign if 302 <= t < 322 or t == 363 else 0.
            envelope = reward_result(cfg, s, arm,
                outcomes={v: outcome for v in ('c1', 'c2', 'h1')})
            for row in envelope['rows']:
                row['time_s'] = t
            reward_rows.extend(envelope['rows'])
        first = steps[0]['sample']['track2_trigger']
        assert all(action > 0 for action in first['actions'])
        assert all(first['accepted'].values())
        assert steps[0]['sample']['context']['move_offsets'] == dict(c1=0., c2=0.)
        assert steps[1]['sample']['context']['move_offsets'] == dict(c1=0., c2=0.)
        for step in steps[2:22]:
            assert step['sample']['context']['move_offsets'] == dict(c1=-.3, c2=-.3)
        assert steps[22]['sample']['context']['move_offsets'] == dict(c1=0., c2=0.)
        assert 'track2_trigger' in steps[60]['sample']
        tapes.append(json.dumps(steps, sort_keys=True))
        monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
        captured = []
        update = tb.ppo_update
        def capture(model, optimizer, batch, advantages, returns, **kw):
            captured.append((list(advantages), list(returns)))
            return update(model, optimizer, batch, advantages, returns, **kw)
        with monkeypatch.context() as patch:
            patch.setattr(tb, 'ppo_update', capture)
            report = runtime.train([dict(id='ep1', arm=arm, steps=steps, actual_sim_s=364)],
                rewards=dict(schema=tb.REWARD_SCHEMA, arm=arm, update=1, rows=reward_rows))
        expected = sign*1.5/tb.UNIT*(sum(tb.GAMMA**j for j in range(2, 22))+tb.GAMMA**63)
        assert captured[0][0] == pytest.approx([expected]*2)
        assert captured[0][1] == pytest.approx([expected]*2)
        assert all(metric['accepted'] for metric in report['trigger'])
        assert report['layer'][0]['accepted']
        assert report['reward']['temporal']['gamma'] == tb.GAMMA
        assert all(torch.equal(v, p.layer.base.state_dict()[k]) for k, v in frozen.items())
        assert all(v.grad is None and not v.requires_grad for v in p.layer.base.parameters())
        learning_targets.append(captured)
        snapshots.append((deepcopy(p.trigger.trigger_head.state_dict()),
                          deepcopy(p.trigger.trigger_value.state_dict()),
                          deepcopy(p.layer.residual.state_dict())))
    assert tapes[0] == tapes[1]
    assert learning_targets[0] != learning_targets[1]
    for minus, plus in zip(snapshots[0], snapshots[1]):
        assert any(not torch.equal(v, plus[k]) for k, v in minus.items())


def test_temporal_targets_preserve_cav_lifetimes_order_terminals_and_episode_boundaries(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    def step(t, members, *, terminal=False, values=None):
        s = dict(context=dict(members=members))
        if values is not None:
            s['track2_trigger'] = dict(values=values)
        return dict(time_s=t, sample=s, terminated=terminal)
    episodes = [dict(id='one', steps=[
        step(300, ['c1', 'c2'], values=[.1, -.2]),
        step(301, ['c2', 'c1']),  # changing order must not swap identity
        step(302, ['c2']),        # c1 departs
        step(303, ['c1', 'c2'], terminal=True),  # a new c1 lifetime
        step(304, ['c1', 'c2'])]), dict(id='two', steps=[step(300, ['c1', 'c2'])])]
    rewards = {('one', 300): [1., 10.], ('one', 301): [20., 2.],
        ('one', 302): [30.], ('one', 303): [100., 40.], ('one', 304): [1000., 1000.],
        ('two', 300): [-500., -500.]}
    returns, advantages = runtime._temporal_targets(episodes, rewards)
    assert returns['one', 300] == pytest.approx([
        (1+tb.GAMMA*2)/tb.UNIT,
        (10+tb.GAMMA*20+tb.GAMMA**2*30+tb.GAMMA**3*40)/tb.UNIT])
    assert returns['one', 301][1] == 2/tb.UNIT
    assert returns['one', 303] == [100/tb.UNIT, 40/tb.UNIT]
    assert returns['two', 300] == [-500/tb.UNIT]*2
    assert advantages['one', 300] == pytest.approx([
        returns['one', 300][0]-.1, returns['one', 300][1]+.2])


@pytest.mark.parametrize('boundary', ['executed', 'all_departed'])
def test_nonoptimizable_tick_outcomes_and_departure_boundaries(cfg, monkeypatch, boundary):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    _, s, _ = sample(cfg)
    later = deepcopy(s)
    later.pop('track2_trigger')
    if boundary == 'executed':
        later['behaviour'] = dict(executed=True)
    else:
        later['context']['members'] = []
    episode = dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s), dict(time_s=301, sample=later)])
    envelope = reward_result(cfg, s, outcomes=dict(c1=0., c2=0., h1=0.))
    delayed = reward_result(cfg, later, outcomes=dict(c1=2., c2=2., h1=2.))
    for row in delayed['rows']:
        row['time_s'] = 301
    envelope['rows'] += delayed['rows']
    captures = []
    update = runtime.policy.layer.update_rewards
    def capture(records, **kwargs):
        captures.extend(records)
        return update(records, **kwargs)
    monkeypatch.setattr(runtime.policy.layer, 'update_rewards', capture)
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    result = runtime.train([episode], rewards=envelope)
    assert len(captures) == 1
    assert captures[0]['returns'] == pytest.approx([
        tb.GAMMA*3/tb.UNIT if boundary == 'executed' else 0.]*2)
    assert result['reward']['cav_rows'] == (4 if boundary == 'executed' else 2)


@pytest.mark.parametrize('failure', ['gap', 'suffix', 'truncated_episode', 'truncated_step',
    'value_count', 'value_nonfinite', 'duplicate_members', 'duplicate_episode', 'missing_later_reward'])
def test_invalid_temporal_tapes_refuse_before_optimizer(cfg, monkeypatch, failure):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    _, s, _ = sample(cfg)
    episode = dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s)])
    envelope = reward_result(cfg, s)
    if failure == 'gap':
        later = deepcopy(episode['steps'][0])
        later['time_s'] = 302
        episode['steps'].append(later)
        extra = deepcopy(envelope['rows'])
        for row in extra:
            row['time_s'] = 302
        envelope['rows'] += extra
    elif failure == 'suffix':
        episode['actual_sim_s'] = 1200
    elif failure == 'truncated_episode':
        episode['truncated'] = True
    elif failure == 'truncated_step':
        episode['steps'][0]['truncated'] = True
    elif failure == 'value_count':
        s['track2_trigger']['values'].pop()
    elif failure == 'value_nonfinite':
        s['track2_trigger']['values'][0] = float('nan')
    elif failure == 'missing_later_reward':
        later = deepcopy(episode['steps'][0])
        later['time_s'] = 301
        later['sample']['behaviour'] = dict(executed=True)
        episode['steps'].append(later)
    elif failure == 'duplicate_episode':
        pass  # checked below at the train boundary
    else:
        s['context']['members'].append('c1')
    monkeypatch.setattr(runtime.policy.layer, 'update_rewards', lambda *a, **k: pytest.fail('optimized'))
    with pytest.raises(ValueError):
        runtime.train([episode]* (2 if failure == 'duplicate_episode' else 1), rewards=envelope)


@pytest.mark.parametrize('kwargs', [dict(labels=[dict(calibrated_probability=1.)]),
    dict(labels=[dict(sumo_verified=True)]), dict(credit=[dict(group_advantage=1.)])])
def test_retired_dense_labels_and_wm_credit_refused(cfg, monkeypatch, kwargs):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'R')['runtime']
    with pytest.raises(ValueError, match='OFF'):
        runtime.train([], **kwargs)


def test_stochastic_samples_required(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    with pytest.raises(ValueError, match='stochastic own-action'):
        runtime.train([])


def test_p2_lazy_dependency_uses_episode_outcomes_and_common_registration(cfg, monkeypatch):
    fake_load(monkeypatch)
    calls = []
    def reward_update(config, arm, update, episodes):
        calls.append((config['track2_reward'], arm, update))
        step = episodes[0]['steps'][0]
        return reward_result(config, step['sample'], arm, outcomes=step['per_vehicle_rewards'])
    fake = types.ModuleType(tb.__package__+'.track2_reward')
    fake.reward_update = reward_update
    monkeypatch.setitem(sys.modules, tb.__package__+'.track2_reward', fake)
    for arm in ('B', 'R'):
        runtime = tb.build_arm_runtime(cfg, arm)['runtime']
        runtime.safety_checked = True
        _, s, _ = sample(cfg)
        monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
        report = runtime.train([dict(id='ep1', arm=arm, steps=[dict(time_s=300, sample=s,
            per_vehicle_rewards=dict(c1=1., c2=-2., h1=3.))])])
        assert report['reward']['cav_rows'] == 2
    assert calls == [(cfg['track2_reward'], 'B', 1), (cfg['track2_reward'], 'R', 1)]


def test_same_mean_rewards_with_different_cav_alignment_learn_differently(cfg, monkeypatch):
    """A scalar broadcast/mean reduction must not erase which CAV earned reward."""
    fake_load(monkeypatch)
    _, s, _ = sample(cfg)
    results = []
    totals = []
    for outcomes in (dict(c1=1., c2=-2., h1=3.), dict(c1=-2., c2=1., h1=3.)):
        runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
        runtime.safety_checked = True
        monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
        envelope = reward_result(cfg, s, outcomes=outcomes)
        totals.append(sum(row['reward'] for row in envelope['rows']))
        runtime.train([dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=deepcopy(s))])],
                      rewards=envelope)
        results.append((deepcopy(runtime.policy.layer.residual.state_dict()),
                        deepcopy(runtime.policy.trigger.state_dict())))
    assert totals[0] == totals[1]
    assert any(not torch.equal(v, results[1][0][k]) for k, v in results[0][0].items())
    assert any(not torch.equal(v, results[1][1][k]) for k, v in results[0][1].items())


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'unknown_cav', 'episode', 'time',
    'arm', 'update', 'source', 'provenance', 'weight', 'neighbour', 'missing_neighbour',
    'formula', 'nonfinite', 'trigger_order'])
def test_invalid_reward_alignment_or_provenance_refuses_before_optimizer(cfg, monkeypatch, failure):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    _, s, _ = sample(cfg)
    envelope = reward_result(cfg, s)
    row = envelope['rows'][0]
    if failure == 'missing':
        envelope['rows'].pop()
    elif failure == 'duplicate':
        envelope['rows'].append(deepcopy(row))
    elif failure == 'unknown_cav':
        row['cav_id'] = 'other'
    elif failure in ('arm', 'update'):
        envelope[failure] = 'A' if failure == 'arm' else 2
    elif failure == 'episode':
        row['episode_id'] = 'unknown'
    elif failure == 'time':
        row['time_s'] = 301
    elif failure == 'source':
        row['source'] = 'wm'
    elif failure == 'provenance':
        row['weight_provenance'] = 'unregistered'
    elif failure == 'weight':
        row['weight'] = .7
    elif failure == 'neighbour':
        row['neighbour_rewards']['unobserved'] = 0.
    elif failure == 'missing_neighbour':
        row['neighbour_rewards'].pop('h1')
    elif failure == 'formula':
        row['reward'] += 1.
    elif failure == 'trigger_order':
        s['track2_trigger']['context'] = dict(s['track2_trigger']['context'], members=['c2', 'c1'])
    else:
        row['own_reward'] = float('nan')
    monkeypatch.setattr(runtime.policy.layer, 'update_rewards', lambda *a, **k: pytest.fail('optimized'))
    with pytest.raises(ValueError):
        runtime.train([dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s)])], rewards=envelope)


def test_missing_p2_dependency_is_explicit(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    _, s, _ = sample(cfg)
    monkeypatch.setitem(sys.modules, tb.__package__+'.track2_reward', None)
    with pytest.raises(RuntimeError, match='P2 track2_reward.reward_update integration dependency'):
        runtime.train([dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s)])])


def test_collect_uses_real_wave_contract_and_refuses_without_approval(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import accel_protocol
    monkeypatch.setattr(accel_protocol, 'guard', lambda w, role: None)
    calls = []
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: calls.append((a, k)) or [])
    with pytest.raises(FileNotFoundError):
        tb.collect_update(cfg, 'B', 1)
    assert not calls
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    tb.collect_update(cfg, 'B', 1)
    args, kw = calls[0]
    assert args[2] == cfg['worlds_by_update']['1']
    assert args[4]['track2_backend'] is True and args[4]['b_replay'] is True
    assert set(args[1]['B'].state_dict()) == set(IntentPolicy().state_dict())
    calls.clear()
    tb.collect_update(cfg, 'A', 1)
    assert 'track2_backend' not in calls[0][0][4]
    assert calls[0][0][4]['b_replay'] is False


def test_bank_reader_is_real_per_seed_effective_and_restored(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import replay_bank as rb
    original = rb.confirmations
    calls = []
    def pairs(*a, **k):
        assert rb.confirmations.bank_rule == 'per_seed_v1'
        calls.append((a, k))
        return [], {'pairs': 0}
    monkeypatch.setattr(rb, 'bank_pairs', pairs)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.bank_pairs()
    assert calls[0][1]['imitation_target_space'] == 'effective'
    assert calls[0][1]['imitation_bank_filter'] == 'cooperative_only'
    assert rb.confirmations is original


def test_save_resume_includes_trainable_weights_and_optimizers(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.policy.layer.trigger_gain.data.fill_(.123)
    target = runtime.save()
    saved = torch.load(target, map_location='cpu', weights_only=False)
    assert saved['charged'] == 960
    with pytest.raises(ValueError, match='already exists'):
        runtime.save()
    restored = tb.build_arm_runtime(dict(cfg, update=2), 'B')['runtime']
    assert restored.policy.layer.trigger_gain.item() == pytest.approx(.123)


@pytest.mark.parametrize('change', [dict(track2_backend=False), dict(coop_reward=True),
    dict(host='cpuS'), dict(device='cuda'), dict(v2v=False), dict(merge_features=False),
    dict(dense_teacher=True), dict(group_credit=True), dict(track2_reward={}),
    dict(track2_reward=dict(weight=.5, provenance=''))])
def test_config_refusals_before_loading(cfg, monkeypatch, change):
    monkeypatch.setattr(tb, '_load_base', lambda *a: pytest.fail('loaded before validation'))
    with pytest.raises(ValueError):
        tb.build_arm_runtime(dict(cfg, **change), 'B')


def test_nonempty_effective_bank_trains_layer_only(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    p = runtime.policy
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(2), False)
    p.observe(rows(), 300)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    bank = deepcopy(s)
    bank['behaviour'] = dict(executed=True, proposal_members=['c1', 'c2'])
    bank['action'] = [[.5], [.5]]
    bank['binding_limits'] = dict(c1=.3, c2=.3)
    before = deepcopy(p.layer.base.state_dict())
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([(bank, 1.)], {'pairs': 1}))
    report = runtime.train([dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s, reward=-100)])],
        rewards=reward_result(cfg, s))
    assert report['bank']['imitation_target_space'] == 'effective'
    assert report['bank']['optimizer_steps'] >= 5
    assert report['bank']['effective']['binding_share'] == 1.
    assert report['bank']['kl'] <= .045
    assert all(torch.equal(before[k], v) for k, v in p.layer.base.state_dict().items())


def test_bank_cap_explicit_no_truth_lookup(cfg, monkeypatch):
    from . import effective_imitation as ei
    captured = []
    original = lambda *a, **k: captured.append(k['cap'])
    monkeypatch.setattr(ei, 'attach_limits', original)
    with tb._bank_view(cfg):
        ei.attach_limits()
    assert captured == [1.2]
    assert ei.attach_limits is original


def test_weighted_practice_uses_existing_frozen_plan_and_batch(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import practice_starts as ps, accel_protocol, practice_weighting as pw
    cfg.update(practice_starts=True, practice_weighting=True, practice_fraction=.25,
               practice_weighted_fraction=.5, practice_repeat_cap=2)
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    monkeypatch.setattr(accel_protocol, 'guard', lambda *a: None)
    monkeypatch.setattr(pw, 'effective_fraction', lambda c: .5)
    requests = []
    monkeypatch.setattr(ps, 'plan_for_update', lambda *a, **k: requests.append((a, k)) or
                        dict(rows=[dict(example_id='x')]))
    monkeypatch.setattr(ps, 'wave_call', lambda *a: dict(worlds=[dict(seed=5)],
        extras=dict(practice=dict(slot='clean1_warm_u001_p01', t0=400), branch={'test': True})))
    calls = []
    def wave(*a, **k):
        calls.append((a, k))
        if a[3].endswith('_practice'):
            return [dict(id='practice', arm='B', world=dict(seed=5), practice=dict(t0=400),
                         steps=[dict(time_s=400), dict(time_s=401)])]
        return [dict(id='normal', arm='B', world=dict(seed=1), steps=[])]
    monkeypatch.setattr(tb, '_wave', wave)
    episodes = tb.collect_update(cfg, 'B', 1)
    assert requests[0][1]['weighting'] is True
    assert requests[0][1]['fraction'] == .5
    assert calls[0][0][2] == [cfg['worlds_by_update']['1'][0]]
    assert calls[1][1]['used_seeds'] == {1}
    assert calls[1][0][4]['practice_batch'][0]['branch'] == {'test': True}
    assert episodes[1]['steps'] == [dict(time_s=401)]


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_production_weighted_practice_requires_compliant_dependency(cfg, monkeypatch, arm):
    fake_load(monkeypatch)
    from . import practice_weighting as pw, accel_protocol
    cfg.update(practice_starts=True, practice_weighting=True, practice_repeat_cap=2)
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    monkeypatch.setattr(accel_protocol, 'guard', lambda *a: None)
    monkeypatch.setattr(pw, 'VERIFIED_CHILDREN', False)
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: pytest.fail('dispatched before child gate'))
    with pytest.raises(ValueError, match='SUMO-verified augmentation dependency'):
        tb.collect_update(cfg, arm, 1)


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_production_frozen_child_cannot_bypass_registered_prelaunch(cfg, monkeypatch, arm):
    fake_load(monkeypatch)
    from . import practice_starts as ps, accel_protocol
    cfg.update(practice_starts=True, practice_weighting=True, practice_repeat_cap=2)
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    monkeypatch.setattr(accel_protocol, 'guard', lambda *a: None)
    monkeypatch.setattr(ps, 'realised_count', lambda *a: 1)
    monkeypatch.setattr(ps, 'plan_for_update', lambda *a, **k: dict(rows=[dict(
        source='augmented', prelaunch=dict(provenance='unbound'))]))
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: pytest.fail('dispatched unbound child'))
    with pytest.raises(ValueError, match='registered prelaunch provenance'):
        tb.collect_update(cfg, arm, 1)


def test_r_practice_uses_b_realised_count(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import practice_starts as ps, accel_protocol, practice_weighting as pw
    cfg.update(practice_starts=True, practice_fraction=.25, practice_repeat_cap=2)
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    monkeypatch.setattr(accel_protocol, 'guard', lambda *a: None)
    monkeypatch.setattr(pw, 'effective_fraction', lambda c: .25)
    monkeypatch.setattr(ps, 'realised_count', lambda r, arm, u: 0)
    captured = []
    monkeypatch.setattr(ps, 'plan_for_update', lambda *a, **k: captured.append(k['count']) or dict(rows=[]))
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: [])
    tb.collect_update(cfg, 'R', 1)
    assert captured == [0]


def test_practice_safety_uses_pre_update_a_separate_pairs(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import practice_starts as ps, warm_health, accel_storage, practice_weighting as pw
    cfg.update(practice_starts=True, practice_fraction=.25, practice_repeat_cap=2)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    monkeypatch.setattr(pw, 'effective_fraction', lambda c: .25)
    monkeypatch.setattr(ps, 'plan_for_update', lambda *a, **k: dict(rows=[dict(example_id='x')]))
    monkeypatch.setattr(ps, 'a_reference_call', lambda *a: dict(ident='ref_B_pa01', worlds=[dict(seed=1)],
        extras=dict(practice={}, branch={})))
    reference = dict(id='normal-A', world=dict(seed=1))
    monkeypatch.setattr(accel_storage, 'read', lambda p: reference)
    matched = dict(id='ref_B_pa01_A_1', world=dict(seed=1))
    waves = []
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: waves.append(a) or [matched])
    loads = []
    original_load = tb._load_base
    monkeypatch.setattr(tb, '_load_base', lambda *a: loads.append(a) or original_load(*a))
    checks = []
    monkeypatch.setattr(warm_health, 'braking', lambda *a: checks.append(a))
    report = runtime.safety([dict(id='normal-B', world=dict(seed=1)), dict(id='practice-B',
        world=dict(seed=1), practice=dict(slot='clean1_warm_u001_p01', example_id='x'))])
    assert loads[0][1:] == ('A', 0)
    assert checks[0][4] == [reference]
    assert checks[0][5] == {'practice-B': matched}
    assert 'track2_backend' not in waves[0][4]
    assert waves[0][3].endswith('_B_practiceref')
    assert report['new_reference_sim_s'] == 1200


def test_train_refuses_before_paired_safety(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    with pytest.raises(ValueError, match='paired safety'):
        runtime.train([])


def test_real_wave_job_serializes_base_and_track2_state(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import accel_runner
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    monkeypatch.setattr(accel_runner, 'ledger', lambda r: [])
    extra = dict(update=1, v2v=True, v2v_dim=48, b_replay=True, track2_backend=True,
                 track2_config=cfg, track2_state=runtime.policy.worker_state())
    jobs, receipts = accel_runner.wave_jobs(tb._path(cfg['run_root']), cfg, {'B': runtime.base},
        cfg['worlds_by_update']['1'], 'train', 'clean1_warm_u001', 'track2_train', False, extra)
    assert not receipts and len(jobs) == 2
    assert jobs[0]['sim_s'] == 1200
    assert jobs[0]['policy_state']  # real WM base-actor whitelist, not wrapper-prefixed weights
    assert jobs[0]['track2_state']['schema'] == tb.SCHEMA
    assert 'actor.0.weight' in jobs[0]['weights']


def test_reservation_own_cap_prior_charge_phase_and_no_double_charge(cfg, monkeypatch):
    monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace(flock=lambda *a: None, LOCK_EX=1))
    cfg['budget'] = dict(total=5000000, phases=dict(track2_train=2400))
    tb._reserve(cfg, [dict(id='ep1', sim_s=1200)], 'track2_train')
    with pytest.raises(ValueError, match='duplicate'):
        tb._reserve(cfg, [dict(id='ep1', sim_s=1200)], 'track2_train')
    with pytest.raises(ValueError, match='budget'):
        tb._reserve(cfg, [dict(id='ep2', sim_s=2400)], 'track2_train')
    cfg['prior_charged_sim_s'] = 4998000
    with pytest.raises(ValueError, match='budget'):
        tb._reserve(cfg, [dict(id='ep3', sim_s=1200)], 'track2_train')


def test_real_accel_plant_full_episode_with_fake_physics(cfg, monkeypatch):
    """Run the actual 900-tick worker, V2V, layer, receipts and compression path."""
    from . import accel_plant
    p = policy(cfg)
    calls = []
    leverprobe = types.ModuleType('tools.leverprobe')
    exact = types.ModuleType('tools.leverprobe.backend')
    free = types.ModuleType('tools.coopfree.backend')
    class Backend:
        def __init__(self, path, world, sim_s, checkpoint):
            self.actual = 300
            self.totals = dict(team_delay_vs=0., collisions=0, teleports=0, hard_brake=0, ttc_below_3_vs=0)
        def fingerprint(self):
            return str(self.actual)
        def close(self):
            calls.append('closed')
    exact.Backend = Backend
    exact.setup = lambda *a: calls.append('fake_setup')
    def tick(backend, controller, trace):
        t = backend.actual
        sensed = rows()
        commands = dict(c1=0., c2=0.)
        requests = controller.apply(sensed, commands, t)
        assert set(requests) <= {'c1', 'c2'}
        trace.write(json.dumps(dict(time_s=t, vehicles=sensed, actions=[]))+'\n')
        backend.actual += 1
        return [dict(clamped=False) for v in requests]
    free.tick = tick
    free.observations = lambda backend: rows()
    monkeypatch.setitem(sys.modules, 'tools.leverprobe', leverprobe)
    monkeypatch.setitem(sys.modules, 'tools.leverprobe.backend', exact)
    monkeypatch.setitem(sys.modules, 'tools.coopfree.backend', free)
    monkeypatch.setattr(accel_plant, 'install_seed_scope', lambda: None)
    monkeypatch.setattr(accel_plant, 'guard', lambda *a: None)
    job = dict(id='fake-full-episode', path=str(tb._path(cfg['run_root'])/'episode'),
        runtime='unused', checkpoint='unused', arm='B', world=dict(seed=1), role='train', sim_s=1200,
        weights=p.layer.base.state_dict(), noise_seed=81, deterministic=False, v2v=True,
        b_replay=True, track2_backend=True, track2_config=cfg, track2_state=p.worker_state())
    result = accel_plant.episode(job)
    assert len(result['steps']) == 900
    assert result['actual_sim_s'] == 1200 and result['charged_sim_s'] == 1200
    assert sum('track2_trigger' in s['sample'] for s in result['steps']) == 15
    assert all('track2' in s['sample'] for s in result['steps'])
    assert set(result['metrics']) == {'600', '900'}
    assert calls == ['fake_setup', 'closed']
    assert (tb._path(job['path'])/'RESULT.json.gz').is_file()
    # A last-tick outcome must reach the first decision across the full existing
    # 900-tick production receipt, rather than a shortened move/decision horizon.
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    outcomes = {(result['id'], s['time_s']): [1. if s['time_s'] == 1199 else 0.]*2
                for s in result['steps']}
    returns, _ = runtime._temporal_targets([result], outcomes)
    assert returns[result['id'], 300] == pytest.approx([tb.GAMMA**899/tb.UNIT]*2)


def test_a_uses_real_ppo_without_bank_or_coop_layer(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'A')['runtime']
    monkeypatch.setitem(sys.modules, tb.__package__+'.track2_reward', None)
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: pytest.fail('A bank accessed'))
    controller = controller_class(Controller)(runtime.policy, torch.Generator().manual_seed(9), False)
    steps = []
    for t, reward in [(300, -10.), (301, 20.)]:
        controller.apply(rows(), {'c1': 0., 'c2': 0.}, t)
        steps.append(dict(controller.events[-1], reward=reward))
    before = deepcopy(runtime.policy.state_dict())
    report = runtime.train([dict(arm='A', steps=steps)])
    assert report['ppo'] and report['bank']['status'] == 'A_no_bank'
    assert any(not torch.equal(before[k], v) for k, v in runtime.policy.state_dict().items())
    assert not isinstance(runtime.policy, tb.EpisodePolicy)


def test_real_checkpoint_round_trip_no_fake_loader(cfg):
    from .accel_learner import save
    base = IntentPolicy()
    path = tb._path(cfg['run_root'])/'common_u0.pt'
    save(path, base, torch.optim.Adam(base.parameters(), lr=.0003), torch.Generator().manual_seed(12), 0, 960)
    cfg['initial_checkpoint'] = str(path)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.policy.layer.trigger_gain.data.fill_(.12)
    runtime.save()
    restored = tb.build_arm_runtime(dict(cfg, update=2), 'B')['runtime']
    assert restored.policy.layer.trigger_gain.item() == pytest.approx(.12)
    assert all(torch.equal(v, restored.base.state_dict()[k]) for k, v in base.state_dict().items())


def test_trigger_kl_rejection_restore_adam_and_weights(cfg, monkeypatch):
    cfg['layer_config']['kl_budget'] = 1e-12
    cfg['trigger_lr'] = 1.
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    p = runtime.policy
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(2), False)
    p.observe(rows(), 300)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    before = deepcopy(p.trigger.state_dict())
    layer_before = deepcopy(p.layer.state_dict())
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    report = runtime.train([dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s, reward=0.)])],
        rewards=reward_result(cfg, s))
    assert report['trigger'][0]['accepted'] is False
    assert all(torch.equal(v, p.trigger.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, p.layer.state_dict()[k]) for k, v in layer_before.items())
    assert p.layer.optimizer.state_dict()['state'] == {}
    assert p.trigger_optimizer.state_dict()['state'] == {}


def test_trigger_kl_is_joint_against_whole_update_reference(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    _, s, _ = sample(cfg)
    batch = tb._batch(s['track2_trigger'])
    with torch.no_grad():
        logits = runtime.policy.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0]
        reference = [(batch, logits.clone())]
        runtime.policy.trigger.trigger_head[-1].weight.add_(.2)
        after = runtime.policy.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0]
        expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=logits),
            torch.distributions.Categorical(logits=after)).sum().item()
    assert runtime._trigger_kl(reference) == pytest.approx(expected)


def test_bad_base_checkpoint_update_refuses(cfg, monkeypatch):
    fake_load(monkeypatch)
    load = tb._load_base
    def wrong(*a):
        p, o, g, payload = load(*a)
        return p, o, g, dict(update=999)
    monkeypatch.setattr(tb, '_load_base', wrong)
    with pytest.raises(ValueError, match='checkpoint update mismatch'):
        tb.build_arm_runtime(cfg, 'B')
