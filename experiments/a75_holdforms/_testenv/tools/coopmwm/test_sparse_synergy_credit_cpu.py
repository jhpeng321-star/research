"""G4 CPU-only contracts. All 'sumo' receipts below are explicit test doubles."""
from copy import deepcopy

import pytest
import torch

from . import sparse_synergy_credit as g4
from . import track2_backend as tb
from .test_track2_backend_cpu import (cfg, cpu_threads, fake_load, rows, reward_result)  # noqa: F401


def config():
    return dict(training_seed=123, worlds_by_update={'1': [dict(seed=7)]},
        g4_synergy_credit=dict(enabled=True, registration=g4.registration(),
                               phase_id='g4-successor-test', first_update=1))


def episode(arm='B', arity=2, *, count=1):
    members = [f'c{i}' for i in range(arity)]
    move = tb.moves.Move('wall', tuple(members), 2, 20, .3).to_dict()
    steps = []
    candidate_times = [t for t in range(1100) if g4._sampled(
        dict(time_s=t, proposer=members[0]), dict(world=dict(seed=7)), config())]
    for t in candidate_times[:count]:
        trigger = dict(context=dict(members=members), tables=[[None, move]]*arity,
            actions=[1]+[0]*(arity-1), mask=[[True, True]]*arity,
            accepted={members[0]: True}, values=[0.]*arity, logp=[-.5]*arity)
        steps.append(dict(time_s=t, sample=dict(context=dict(members=members),
                                              track2_trigger=trigger)))
    return dict(id=arm+'_ep', arm=arm, world=dict(seed=7), g4_source='ordinary_training',
                actual_sim_s=1200, steps=steps)


def captures(episodes):
    return {a: {r['opportunity_id']: dict(stage='pre_action', time_s=r['time_s'],
        **{k: g4.digest([a, r['time_s'], k]) for k in g4.CAPTURE_KEYS})
        for r in g4.opportunities(es)} for a, es in episodes.items()}


def planned(arity=2, count=1):
    c = config()
    episodes = {a: [episode(a, arity, count=count)] for a in ('B', 'R')}
    plan, state = g4.plan_update(c, 1, episodes, captures(episodes), g4.initial_state(c))
    return c, episodes, plan, state


def measured(job, costs=None):
    costs = costs or dict(base=100., group=80., solo0=90., solo1=95., solo2=97., solo3=98.)
    return dict(job_sha256=job['job_sha256'], source='sumo', status='graded',
        J_veh_s=costs[job['branch']], actual_sim_s=job['sim_s'],
        **{k: deepcopy(job[k]) for k in ('window', 'verification_seed', 'retained_members')},
        capture=deepcopy(job['start']['capture']),
        safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0))


def envelope(plan, costs=None):
    return g4.measured_envelope(plan, [measured(j, costs) for j in plan['jobs']])


@pytest.mark.parametrize('arity,quote', [(2, 153600), (3, 192000), (4, 230400)])
def test_full_prefix_budget_all_solos_and_pair_effort(arity, quote):
    c, episodes, plan, state = planned(arity, count=10)
    assert len(plan['starts']) == 16
    assert plan['quote_sim_s'] == state['reserved_sim_s'] == quote
    assert len(plan['jobs']) == 16*2*(arity+2)
    assert all(j['sim_s'] == 1200 and j['window']['end_s'] == 1200 for j in plan['jobs'])
    assert {j['branch'] for j in plan['jobs']} == {'base', 'group', *[f'solo{i}' for i in range(arity)]}
    assert state['starts'] == {'B': 8, 'R': 8}
    g4.trigger_terms(c, 'B', 1, episodes['B'], envelope(plan))
    # A later update cannot top up completed/failed reservations.
    c['worlds_by_update']['2'] = c['worlds_by_update']['1']
    later, final = g4.plan_update(c, 2, episodes, captures(episodes), state)
    assert later['jobs'] == [] and final['reserved_sim_s'] == quote


def test_sampling_ignores_outcomes_arm_and_input_order():
    c, episodes, plan, _ = planned(count=3)
    altered = deepcopy(episodes)
    for es in altered.values():
        for e in es:
            e['steps'].reverse()
            for s in e['steps']:
                s.update(reward=1e9, J_veh_s=-1e9)
    other, _ = g4.plan_update(c, 1, altered, captures(altered), g4.initial_state(c))
    assert other == plan
    assert [s['time_s'] for s in plan['starts'] if s['arm'] == 'B'] == [
        s['time_s'] for s in plan['starts'] if s['arm'] == 'R']


def test_scarcity_and_arity_are_matched_without_extra_verification():
    c = config()
    episodes = {'B': [episode('B', 4, count=8)], 'R': [episode('R', 2, count=1)]}
    plan, state = g4.plan_update(c, 1, episodes, captures(episodes), g4.initial_state(c))
    assert plan['jobs'] == [] and state['reserved_sim_s'] == 0
    episodes['B'] = [episode('B', 2, count=8)]
    plan, _ = g4.plan_update(c, 1, episodes, captures(episodes), g4.initial_state(c))
    assert len(plan['starts']) == 2 and plan['quote_sim_s'] == 19200


@pytest.mark.parametrize('mutation', ['practice', 'deterministic', 'replay', 'rejected', 'gap_open'])
def test_non_joint_non_own_and_non_ordinary_boundaries_excluded(mutation):
    ep = episode()
    s = ep['steps'][0]['sample']
    if mutation == 'practice':
        ep['practice'] = dict(t0=0)
    elif mutation == 'deterministic':
        s['deterministic'] = True
    elif mutation == 'replay':
        s['behaviour'] = dict(executed=True)
    elif mutation == 'rejected':
        s['track2_trigger']['accepted']['c0'] = False
    else:
        s['track2_trigger']['tables'][0][1]['family'] = 'gap_open'
    assert g4.opportunities([ep]) == []


@pytest.mark.parametrize('costs,expected', [
    (dict(base=100., group=80., solo0=90., solo1=95.), 10.),
    # Harmful solos do NOT get floored to base: S=120-110=10, T=-10.
    (dict(base=100., group=110., solo0=120., solo1=130.), 10.),
    (dict(base=100., group=110., solo0=90., solo1=95.), -20.),
    (dict(base=100., group=90., solo0=90., solo1=95.), 0.),
])
def test_exact_best_solo_signed_credit_only_for_initiator(costs, expected):
    c, episodes, plan, _ = planned()
    env = envelope(plan, costs)
    terms, receipt = g4.trigger_terms(c, 'B', 1, episodes['B'], env)
    start = plan['starts'][0]
    assert terms == {(start['episode_id'], start['time_s'], 'c0'): expected/tb.UNIT}
    assert receipt['rows'][0]['synergy_veh_s'] == expected
    assert receipt['rows'][0]['seeds'][0]['total_benefit_veh_s'] == costs['base']-costs['group']
    assert all(k[-1] != 'c1' for k in terms)


def test_seedwise_best_solo_not_best_of_seed_mean_and_bounded():
    c, episodes, plan, _ = planned()
    results = []
    for j in plan['jobs']:
        solos = (0., 20000.) if j['verification_seed'] == 1 else (20000., 0.)
        results.append(measured(j, dict(base=20000., group=20000., solo0=solos[0], solo1=solos[1])))
    env = g4.measured_envelope(plan, results)
    terms, _ = g4.trigger_terms(c, 'B', 1, episodes['B'], env)
    assert env['rows'][0]['synergy_veh_s'] == -20000.
    assert list(terms.values()) == [-1.]


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wm', 'window', 'prefix',
                                    'seed', 'mask', 'nan', 'cost', 'ungraded', 'safety'])
def test_refuse_invalid_counterfactuals_before_credit(mutation):
    _, _, plan, _ = planned()
    results = [measured(j) for j in plan['jobs']]
    if mutation == 'missing':
        results.pop()
    elif mutation == 'duplicate':
        results.append(deepcopy(results[0]))
    elif mutation == 'wm':
        results[0]['source'] = 'wm'
    elif mutation == 'window':
        results[0]['window']['end_s'] = 1199
    elif mutation == 'prefix':
        results[0]['capture']['prefix_sha256'] = 'f'*64
    elif mutation == 'seed':
        results[0]['verification_seed'] = 3
    elif mutation == 'mask':
        results[0]['retained_members'] = []
    elif mutation == 'nan':
        results[0]['J_veh_s'] = float('nan')
    elif mutation == 'cost':
        results[0]['actual_sim_s'] = 1199
    elif mutation == 'ungraded':
        results[0]['status'] = 'failed'
    else:
        results[0]['safety'] = {}
    with pytest.raises(ValueError):
        g4.measured_envelope(plan, results)


def test_empty_sample_has_no_credit_and_stale_labels_refuse():
    c, episodes, plan, _ = planned()
    altered = deepcopy(episodes)
    altered['B'][0]['steps'][0]['sample']['track2_trigger']['logp'][0] -= .1
    with pytest.raises(ValueError, match='stale'):
        g4.trigger_terms(c, 'B', 1, altered['B'], envelope(plan))
    with pytest.raises(ValueError, match='envelope'):
        g4.trigger_terms(c, 'B', 1, episodes['B'], None)
    episodes['R'][0]['steps'] = []
    empty, _ = g4.plan_update(c, 1, episodes, captures(episodes), g4.initial_state(c))
    terms, receipt = g4.trigger_terms(c, 'B', 1, episodes['B'], envelope(empty))
    assert terms == {} and receipt['measured_starts'] == 0 and receipt['unsampled_starts'] == 1
    assert g4.trigger_terms({}, 'B', 1, episodes['B'], None) == ({}, None)
    assert g4.trigger_terms(c, 'A', 1, [], None) == ({}, None)


@pytest.mark.parametrize('field,value', [('weight', 2.), ('sample_denominator', 1),
                                       ('budget_cap_sim_s', 999999), ('arms', ['B'])])
def test_registration_is_fixed_common_and_not_arm_tuned(field, value):
    c = config()
    c['g4_synergy_credit']['registration'][field] = value
    with pytest.raises(ValueError, match='registration'):
        g4.settings(c)


@pytest.mark.parametrize('mutation', ['phase', 'diagnostic', 'truncated', 'capture', 'repeat', 'budget'])
def test_planning_refuses_wrong_source_boundary_capture_or_budget(mutation):
    c, episodes, _, _ = planned()
    state = g4.initial_state(c)
    caps = captures(episodes)
    if mutation == 'phase':
        c['g4_synergy_credit']['first_update'] = 2
    elif mutation == 'diagnostic':
        episodes['B'][0]['g4_source'] = 'diagnostic'
    elif mutation == 'truncated':
        episodes['B'][0]['truncated'] = True
    elif mutation == 'capture':
        next(iter(caps['B'].values()))['stage'] = 'post_action'
    elif mutation == 'repeat':
        state['last_update'] = 1
    else:
        state['reserved_sim_s'] = g4.REGISTRATION['budget_cap_sim_s']
    with pytest.raises(ValueError):
        g4.plan_update(c, 1, episodes, caps, state)


@pytest.mark.parametrize('mutation', ['all_solos', 'wrong_mask', 'quote', 'cap', 'asymmetric'])
def test_executor_rejects_resigned_inconsistent_plan_before_reservation(mutation):
    _, _, plan, _ = planned()
    if mutation == 'all_solos':
        plan['jobs'].pop()
    elif mutation == 'wrong_mask':
        plan['jobs'][0]['retained_members'] = []
    elif mutation == 'quote':
        plan['quote_sim_s'] -= 1200
    elif mutation == 'cap':
        plan['reservation_after']['reserved_sim_s'] = 230401
    else:
        plan['starts'] = plan['starts'][:1]
        plan['jobs'] = [j for j in plan['jobs'] if j['start']['arm'] == 'B']
    plan['plan_sha256'] = g4.digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    with pytest.raises(ValueError):
        g4.execute_plan(plan, reserve=lambda p: pytest.fail('must refuse before reserve'),
                        run_branch=lambda j: pytest.fail('must refuse before dispatch'))


def test_executor_atomic_reservation_no_retry_or_partial_label():
    _, _, plan, _ = planned()
    order = []
    def reserve(p):
        order.append(('reserve', p['quote_sim_s']))
    def worker(j):
        order.append(('branch', j['job_sha256']))
        return measured(j)
    assert g4.execute_plan(plan, reserve=reserve, run_branch=worker)['rows']
    assert order[0] == ('reserve', 19200)
    order.clear()
    def failed(j):
        order.append(('failed', j['job_sha256']))
        raise RuntimeError('test worker failure')
    with pytest.raises(RuntimeError, match='worker failure'):
        g4.execute_plan(plan, reserve=reserve, run_branch=failed)
    assert [x[0] for x in order] == ['reserve', 'failed']
    order.clear()
    def blocked(p):
        raise ValueError('test ledger duplicate/budget refusal')
    with pytest.raises(ValueError, match='ledger'):
        g4.execute_plan(plan, reserve=blocked, run_branch=worker)
    assert order == []


def test_worker_mask_preserves_original_move_and_release():
    _, _, plan, _ = planned()
    jobs = {j['branch']: j for j in plan['jobs']}
    assert g4.command_offsets(jobs['group'], 2) == dict(c0=-.3, c1=-.3)
    assert g4.command_offsets(jobs['solo0'], 2) == dict(c0=-.3, c1=0.)
    assert g4.command_offsets(jobs['solo1'], 2) == dict(c0=0., c1=-.3)
    assert g4.command_offsets(jobs['base'], 2) == dict(c0=0., c1=0.)
    assert g4.command_offsets(jobs['group'], 22) == dict(c0=0., c1=0.)
    assert g4.command_offsets(jobs['solo0'], 2, observable_state={'s_m': {'c0': -100., 'c1': -105.}}) == dict(c0=0., c1=0.)


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_real_ppo_edge_changes_only_proposer_advantage(cfg, monkeypatch, arm):
    """Real model/PPO and frozen-base guard, fake counterfactual physics only."""
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    fake_load(monkeypatch)
    cfg['g4_synergy_credit'] = config()['g4_synergy_credit']
    seed = next(seed for seed in range(10000) if g4._sampled(
        dict(time_s=300, proposer='c1'), dict(world=dict(seed=seed)), cfg))
    cfg['worlds_by_update']['1'] = [dict(seed=seed)]
    monkeypatch.setattr(tb.MoveAPI, 'candidates', staticmethod(lambda state, rng, n: [
        tb.TriggerMove(tb.moves.Move('wall', ('c1', 'c2'), 2, 20, .3)) for _ in range(n)]))
    runtime = tb.build_arm_runtime(cfg, arm)['runtime']
    runtime.safety_checked = True
    p = runtime.policy
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    p.observe(rows(), 300)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    assert s['track2_trigger']['accepted']['c1']
    episodes = {a: [dict(id='ep1' if a == arm else a+'_ep', arm=a,
        steps=[dict(time_s=300, sample=deepcopy(s))], g4_source='ordinary_training',
        actual_sim_s=1200, world=dict(seed=seed))] for a in ('B', 'R')}
    plan, _ = g4.plan_update(cfg, 1, episodes, captures(episodes), g4.initial_state(cfg))
    env = envelope(plan)
    returns = {('ep1', 300): [.25, -.5]}
    advantages = {('ep1', 300): [.15, -.7]}
    monkeypatch.setattr(runtime, '_temporal_targets', lambda *a: (returns, advantages))
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    base = deepcopy(p.layer.base.state_dict())
    captured = []
    original = tb.ppo_update
    def capture(*args, **kw):
        captured.append((args[3], args[4]))
        return original(*args, **kw)
    monkeypatch.setattr(tb, 'ppo_update', capture)
    invalid = deepcopy(env)
    invalid['results'][0]['source'] = 'wm'
    trigger_before = deepcopy(p.trigger.state_dict())
    with pytest.raises(ValueError, match='measured source'):
        runtime.train(episodes[arm], rewards=reward_result(cfg, s, arm), synergy=invalid)
    assert captured == []
    assert all(torch.equal(trigger_before[k], v) for k, v in p.trigger.state_dict().items())
    result = runtime.train(episodes[arm], rewards=reward_result(cfg, s, arm), synergy=env)
    assert captured[0][0] == pytest.approx([.151, -.7])
    assert captured[0][1] == returns[('ep1', 300)]
    assert result['g4_synergy_credit']['measured_starts'] == 1
    assert all(torch.equal(base[k], v) for k, v in p.layer.base.state_dict().items())
    assert advantages[('ep1', 300)] == [.15, -.7]
