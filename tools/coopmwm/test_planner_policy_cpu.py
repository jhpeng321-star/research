"""Live/imagined V2V parity, with learned radio columns and no simulator/data."""
import copy
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import pytest
import torch

from . import plan_wm, planner_online, planner_policy
from .acceleration import Controller, Policy as PlainPolicy
from .policy_wm import load_actor, residuals
from .test_planner_online_cpu import Moves, row
from .v2v_runtime import Policy, controller_class
from .v2v_intent import IntentPolicy


@pytest.fixture(autouse=True)
def single_thread(monkeypatch):
    from . import exploration
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def learned_policy(intent=False, gap_only=False, plain=False):
    policy = PlainPolicy() if plain else IntentPolicy(horizon=2) if intent else Policy()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.actor[0].weight[0, 1 if plain else 29] = 1.
        if not gap_only and not plain:
            # Sensed acceleration and last issued (not raw) partner command.
            policy.actor[0].weight[0, 32] = .2
            policy.actor[0].weight[0, 33] = .4
        policy.actor[2].weight[0, 0] = 1.
        policy.actor[4].weight[0, 0] = 1.
        if intent:
            policy.actor[0].weight[0, 35] = .7
            policy.actor[0].weight[0, 37] = .1
            policy.intent_head.weight[:, 0] = .3
            policy.intent_head.bias[:] = torch.tensor([.2, -.4])
    return policy


def controller(policy, deterministic=True, planner=None):
    c = controller_class(Controller)(policy, torch.Generator().manual_seed(7), deterministic,
                                    planner_online=planner)
    c.observe_issued(dict(time_s=59, vehicles=[dict(row(), v_mps=14.5,
                         issued_acceleration_mps2=-.8),
                         dict(row('b', 20.), v_mps=15.25, issued_acceleration_mps2=.6)]))
    if isinstance(policy, IntentPolicy):
        c.intent_board = {'a': (59, [.9, -.4]), 'b': (59, [-.7, .5])}
    return c


def capture(policy, deterministic=True):
    records = []

    def rollout(rec, actions, K, H):
        records.append(rec)
        return {a['id']: [10.]*K for a in actions}

    planner = planner_online.OnlinePlanner(rollout=rollout, calibration={}, rng=None,
                    accel_bounds=(-3., 1.2), calibrate=lambda p, c: 0., move_api=Moves())
    c = controller(policy, deterministic, planner)
    c.apply([row(), row('b', 20.)], {'a': 0., 'b': 0.}, 60)
    assert records, c.events[-1]['sample']['planner_online']
    return c, records[0]


def tensor_state(rows, batch=1):
    result = {k: torch.tensor([[r[k] for r in rows]]) for k in ('s_m', 'v_mps', 'lane', 'is_cav')}
    result['present'] = torch.ones_like(result['is_cav'], dtype=torch.bool)
    return {k: v.expand(batch, -1).clone() for k, v in result.items()}


def safe_accel(state, raw, **kw):
    # Observable synthetic safety bound deliberately differs from raw request.
    return raw.clamp(-.25, .15)


def option_controller(mp, candidate_actions, ids):
    """Test-local frozen executor ABI; the production AST decoder is exercised."""
    fixed = None

    def injected(state, accel, tick, **kw):
        return safe_accel(state, accel, **kw)

    return injected


def executor_modules(monkeypatch):
    runtime = SimpleNamespace(option_controller=option_controller)
    monkeypatch.setitem(sys.modules, 'tools.wmteach', SimpleNamespace(runtime=runtime))
    monkeypatch.setitem(sys.modules, 'wms.control.sumo_idm', SimpleNamespace(
        allowed_speed=lambda lane, s: torch.full_like(s, 31.29)))
    monkeypatch.setitem(sys.modules, 'wms.control.alwayscmd', SimpleNamespace(
        inferred_cap=lambda state: torch.zeros_like(state['s_m'])))
    return runtime


def test_review_gap_weight_reproduction_and_absent_state_refusal():
    c, rec = capture(learned_policy(gap_only=True))
    live = torch.tensor(c.events[-1]['sample']['own_action']).flatten()
    assert torch.allclose(live, torch.tensor([.19485158, -.19485158]), atol=1e-7)
    state = tensor_state(rec['rows'])
    # Legacy bank forecasts remain unchanged; the online adapter must never use
    # this missing-partner approximation for the live V2V policy.
    legacy = residuals(load_actor(rec['policy_state'], 'cpu'), state, ['a', 'b'])
    assert torch.equal(legacy, torch.zeros_like(legacy))
    missing = copy.deepcopy(rec)
    del missing['policy_radio']
    with pytest.raises(ValueError, match='policy_radio'):
        planner_online.validate_observation(missing)
    runtime = SimpleNamespace(model=torch.nn.Linear(1, 1),
                              mp=SimpleNamespace(policy=torch.nn.Linear(1, 1)))
    with pytest.raises(ValueError, match='policy_radio'):
        planner_online.WMRollout(runtime)(missing, [dict(id='policy')], 2, 600)
    imagined = planner_policy.Continuation(rec)
    assert torch.equal(imagined.residuals(state, ['a', 'b'], 0)[0], live)
    imagined.observe_issued(state, torch.zeros(1, 2), 0)
    c.observe_issued(dict(time_s=60, vehicles=[dict(r, issued_acceleration_mps2=0.) for r in rec['rows']]))
    c.apply(rec['rows'], rec['commands'], 61)
    expected = torch.tensor(c.events[-1]['sample']['own_action']).flatten()
    assert torch.allclose(imagined.residuals(state, ['a', 'b'], 2)[0], expected, atol=1e-7)
    assert expected.abs().min() > .19  # would catch the old all-zero continuation


@pytest.mark.parametrize('kind', ['plain', 'v2v', 'intent'])
@pytest.mark.parametrize('deterministic', [False, True])
def test_live_imagined_policy_offset_release_and_causal_radio(monkeypatch, kind, deterministic):
    intent = kind == 'intent'
    c, rec = capture(learned_policy(intent, plain=kind == 'plain'), deterministic)
    rt = executor_modules(monkeypatch)
    policy = load_actor(rec['policy_state'], 'cpu')
    mp = SimpleNamespace(policy=SimpleNamespace(accel_min=-3., accel_max=1.2,
                                               parameters=policy.parameters))
    actions = [dict(id='policy'), dict(id='move:0', offset_profile=[{'a': -.8, 'b': .7}])]
    imagined = planner_policy.Continuation(rec, policy)
    references = [c, controller(copy.deepcopy(c.policy), deterministic)]
    references[1].apply(rec['rows'], rec['commands'], 60)
    rows = rec['rows']
    state = tensor_state(rows, 2)
    trace = []
    with plan_wm.plan_executor(600, policy, trace, final_chain=True, policy_radio=imagined):
        callback = rt.option_controller(mp, actions, ['a', 'b'])
        for second in range(4):
            expected = []
            for branch, reference in enumerate(references):
                branch_rows = [dict(r, s_m=float(state['s_m'][branch, i]),
                                    v_mps=float(state['v_mps'][branch, i])) for i, r in enumerate(rows)]
                if second:
                    reference.apply(branch_rows, rec['commands'], 60+second)
                base = torch.tensor(reference.events[-1]['sample']['own_action']).flatten()
                if branch == 1 and second == 0:
                    base = (base+torch.tensor([-.8, .7])).clamp(-3., 1.2)
                expected.append(base)
            expected = torch.stack(expected)
            native = torch.zeros_like(state['s_m'])
            issued = safe_accel(state, expected)
            acceleration = callback(state, native, 2*second)
            assert torch.allclose(trace[-1], expected, atol=2e-7, rtol=0)
            assert torch.allclose(acceleration, issued, atol=2e-6, rtol=0)
            # Board after prediction contains only this step's head publication;
            # the next live step receives those same t-1 reports.
            for b, reference in enumerate(references):
                assert torch.equal(imagined.branches[b]['generator'].get_state(), reference.generator.get_state())
                assert imagined.branches[b]['noise'] == reference.memory
                if intent:
                    assert imagined.branches[b]['intent']['board'] == reference.intent_board
                observed = [dict(r, v_mps=float(state['v_mps'][b, i]),
                                 s_m=float(state['s_m'][b, i]),
                                 issued_acceleration_mps2=float(issued[b, i])) for i, r in enumerate(rows)]
                reference.observe_issued(dict(time_s=60+second, vehicles=observed))
                assert imagined.branches[b]['previous_commands'] == {
                    v: float(issued[b, i]) for i, v in enumerate(('a', 'b'))}
            # Synthetic observed speeds/positions; branch differences persist
            # after the offset ends, so release must use each branch's new base.
            state['s_m'] += state['v_mps']+.5*issued
            state['v_mps'] += issued
            callback(state, native, 2*second+1)
    assert len(trace) == 4
    assert not torch.equal(trace[1][0], trace[1][1])


def test_review_plain_stochastic_current_sample_through_real_decoder(monkeypatch):
    c, rec = capture(PlainPolicy(), deterministic=False)
    live = torch.tensor(c.events[-1]['sample']['own_action']).flatten()
    assert torch.allclose(live, torch.tensor([.1297888308763504, .16240733861923218]), atol=1e-8)
    planner_online.validate_observation(rec)
    rt = executor_modules(monkeypatch)
    policy = load_actor(rec['policy_state'], 'cpu')
    mp = SimpleNamespace(policy=SimpleNamespace(accel_min=-3., accel_max=1.2,
                                               parameters=policy.parameters))
    trace = []
    state = tensor_state(rec['rows'], 2)
    with plan_wm.plan_executor(600, policy, trace, policy_radio=planner_policy.Continuation(rec, policy)):
        callback = rt.option_controller(mp, [dict(id='policy'),
                       dict(id='move:0', offset_profile=[{'a': -.4}])], ['a', 'b'])
        callback(state, torch.zeros_like(state['s_m']), 0)
    assert torch.equal(trace[0][0], live)
    assert torch.allclose(trace[0][1], live+torch.tensor([-.4, 0.]), atol=1e-8)


@pytest.mark.parametrize('deterministic', [False, True])
@pytest.mark.parametrize('missing', ['policy_radio', 'noise', 'generator_state', 'log_std', 'sample'])
def test_plain_actor_missing_sampling_state_fails_closed(deterministic, missing):
    _, rec = capture(PlainPolicy(), deterministic)
    if missing == 'policy_radio':
        del rec['policy_radio']
    elif missing == 'sample':
        del rec['policy_residuals']['b']
    elif missing == 'noise':
        del rec['policy_radio']['noise']['b']
    else:
        del rec['policy_radio'][missing]
    with pytest.raises(ValueError):
        planner_online.validate_observation(rec)
    runtime = SimpleNamespace(model=torch.nn.Linear(1, 1),
                              mp=SimpleNamespace(policy=torch.nn.Linear(1, 1)))
    with pytest.raises(ValueError):
        planner_online.WMRollout(runtime)(rec, [dict(id='policy')], 2, 600)


@pytest.mark.parametrize('mutate', [
    lambda r: r['policy_radio']['frame']['vehicles'][0].update(secret=1),
    lambda r: r['policy_radio']['previous_commands'].update(far=1.),
    lambda r: r['policy_radio']['intent']['board'].update(b=(60, [0., 0.])),
    lambda r: r['policy_radio']['intent']['board'].update(far=(59, [0., 0.])),
    lambda r: r['policy_radio'].update(intent=None),
    lambda r: r['policy_residuals'].pop('b'),
])
def test_incomplete_nonlocal_or_noncausal_radio_is_refused(mutate):
    _, rec = capture(learned_policy(True))
    mutate(rec)
    with pytest.raises(ValueError):
        planner_online.validate_observation(rec)


def test_radio_snapshot_is_ego_local_and_uses_previous_intent_board():
    c, rec = capture(learned_policy(True))
    assert rec['policy_radio']['intent']['board']['b'] == (59, [-.7, .5])
    assert c.intent_board['b'][0] == 60
    local = planner_policy.local_snapshot(rec['policy_radio'], [row()])
    assert set(local['previous_commands']) == set(local['noise']) == {'a'}
    assert set(local['intent']['board']) == {'a'}
    assert [r['vid'] for r in local['frame']['vehicles']] == ['a']


@pytest.mark.parametrize('plain', [False, True])
def test_production_adapter_full_horizon_chunks_and_multiple_samples(monkeypatch, plain):
    """Keep WMRollout/evaluate/decoder/continuation real; fake only WM dynamics."""
    _, rec = capture(learned_policy(intent=not plain, plain=plain), deterministic=not plain)
    rt_module = executor_modules(monkeypatch)
    monkeypatch.setattr(plan_wm, 'executor_adapter', lambda: nullcontext())
    traces = []

    def adapted_futures(horizons):
        namespace = {}

        def futures(rt, record, indices):
            actions = namespace['_actions']
            state = tensor_state(record['rows'], len(actions)*len(indices))
            callback = rt_module.option_controller(rt.mp, actions, ['a', 'b'])
            # The inherited batching has one row per action in these synthetic
            # callbacks; execute each indexed future with the same CRN state.
            result = {a['id']: [] for a in actions}
            for _ in indices:
                state = tensor_state(record['rows'], len(actions))
                raw_trace = []
                for tick in range(2*max(horizons)):
                    acceleration = callback(state, torch.zeros_like(state['s_m']), tick)
                    state['s_m'] += .5*state['v_mps']
                    state['v_mps'] += .5*acceleration
                    if tick % 2 == 0:
                        raw_trace.append(state['v_mps'].clone())
                traces.append(raw_trace)
                for i, a in enumerate(actions):
                    result[a['id']].append(dict(team_delay=float(state['s_m'][i].sum())))
            return dict(population_audit=dict(finite=True, conserved=True),
                        futures={str(max(horizons)): result})

        namespace['futures'] = futures
        return namespace

    monkeypatch.setattr(plan_wm, 'adapted_futures', adapted_futures)
    base = torch.nn.Linear(1, 1)
    base.accel_min, base.accel_max = -3., 1.2
    runtime = SimpleNamespace(model=torch.nn.Linear(1, 1), mp=SimpleNamespace(policy=base))
    memory = SimpleNamespace(context=lambda rt, record: nullcontext())
    actions = [dict(id='policy'), dict(id='move:0', offset_profile=[{'a': -.8}])]
    before = torch.get_rng_state().clone()
    # Actor loading in evaluate must also preserve the live global RNG.
    joint = planner_online.WMRollout(runtime, memory=memory, chunk_actions=2)(rec, actions, 2, 600)
    split = planner_online.WMRollout(runtime, memory=memory, chunk_actions=1)(rec, actions, 2, 600)
    assert joint == split
    assert all(len(t) == 600 for t in traces)
    assert len(joint['policy']) == len(joint['move:0']) == 2
    assert joint['policy'] != joint['move:0']
    assert torch.equal(before, torch.get_rng_state())
