"""Meaningful CPU checks: real ownership handback, information boundary and oracle."""
import copy
import json
from types import SimpleNamespace

import pytest

from tools.c1switch.core import Belief, Handover, branch_all, choose, medians
from tools.c1switch.design import (BUDGET, CONTROLLERS, checked, worlds, measurement_indices,
                                    require_main_budget, STEP0_SEEDS, STEP0_TIMES)
from tools.c1switch.rpc import FilePredictor
from tools.c1switch.storage import Ledger, canonical, write
from tools.c1switch.model import select_model
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.config import load_config
from traffic_shepherd.damping import DampingDecision


def test_real_commandownership_change_release_native_resume_and_identity():
    calls = []
    class Fixed:
        config = load_config()
        def decide(self, frame, *, now_s):
            return DampingDecision(command_acceleration_mps2=-1., target_speed_mps=14.)
    api = SimpleNamespace(vehicle=SimpleNamespace(
        getIDList=lambda:['c', 'h'], getSpeedMode=lambda v:31,
        slowDown=lambda *a:calls.append(('slowDown',)+a),
        setSpeed=lambda *a:calls.append(('setSpeed',)+a)))
    owner = CommandOwnership(api, {'c'}, Fixed())
    env = SimpleNamespace(ownership=owner, api=api, _circles=owner.controller,
                          pi_session=object(), _direct_pi=True)
    h = Handover(lambda n:SimpleNamespace(name=n))
    a = h.select('pi', env)
    owner.step({}, now_s=0, live_ids={'c', 'h'})
    assert h.select('pi', env) is a
    assert calls == [('slowDown', 'c', 14., 1.)]
    h.select('native', env)
    assert calls[-1] == ('setSpeed', 'c', -1)
    assert not owner._owned and env.pi_session is None and not env._direct_pi
    assert owner.controller is env._circles
    h.select('circles', env)
    owner.step({}, now_s=1, live_ids={'c', 'h'})
    assert calls[-1] == ('slowDown', 'c', 14., 1.)
    assert all('h' not in row for row in calls)


def receipt(vid='x', **extra):
    return dict(vehicle_id=vid, timestamp_s=299., s_m=-10., lane=1,
                v_mps=12., a_mps2=0., **extra)


def test_belief_noninterference_hidden_inputs_and_immutable_payload():
    b = Belief()
    raw = receipt(hidden_type='aggressive', true_penetration=.4, vehicle_params={'secret':1},
                  length_m=9, slot=99, future_presence=[1])
    x = b.observe(300, self_reports=[receipt('c')], detections=[raw], coverage=[])
    raw['hidden_type'] = 'other'
    y = Belief().observe(300, self_reports=[receipt('c')], detections=[raw], coverage=[])
    assert x == y
    assert x['history'][0]['detections'][0]['age_s'] == 1
    text = json.dumps(x)
    assert all(k not in text for k in ('hidden_type', 'penetration', 'vehicle_params', 'length_m', 'future_presence'))
    x['history'][0]['detections'][0]['s_m'] = 999
    assert b.payload()['history'][0]['detections'][0]['s_m'] == -10
    with pytest.raises(ValueError):
        b.observe(298, self_reports=[], detections=[receipt()], coverage=[])


def test_preregistered_tie_rule_and_nonfinite_stop():
    values = dict.fromkeys(CONTROLLERS, 10.)
    assert choose(values, 'pi') == 'pi'
    values['native'] = 9.
    values['pi'] = 9.+.9e-6
    assert choose(values, 'pi') == 'pi'
    values['pi'] = 10.
    assert choose(values, 'pi') == 'native'
    values['native'] = float('nan')
    with pytest.raises(ValueError):
        choose(values)
    with pytest.raises(ValueError):
        medians({c:[1.]*7 for c in CONTROLLERS})


class Toy:
    def __init__(self):
        self.time, self.total, self.name, self.memory, self.spent = 300, 0., 'native', 0, 0
    def save(self, label):
        return copy.deepcopy((self.time, self.total, self.name, self.memory))
    def load(self, state):
        self.time, self.total, self.name, self.memory = copy.deepcopy(state)
    def select(self, name):
        if self.name != name:
            self.memory = 0
        self.name = name
    def advance(self, seconds):
        # PI uniquely wins before t360, MAPPO afterwards, not a static winner.
        best = 'pi' if self.time < 360 else 'mappo'
        self.total += seconds * (1 if self.name == best else 5)
        self.time += seconds
        self.memory += seconds
        self.spent += seconds
    def metrics(self):
        return {'team_delay_vs':self.total}


def test_oracle_same_state_no_branch_leakage_and_selected_trunk_reuse():
    toy = Toy()
    rows, states = branch_all(toy, horizon=60)
    pick = choose({c:rows[c]['60']['team_delay_vs'] for c in CONTROLLERS})
    assert pick == 'pi'
    assert toy.spent == 5*60
    toy.load(states[pick])
    assert (toy.time, toy.total, toy.name, toy.memory) == (360, 60., 'pi', 60)
    rows, states = branch_all(toy, horizon=90, repeat_native=True)
    assert choose({c:rows[c]['60']['team_delay_vs'] for c in CONTROLLERS}, 'pi') == 'mappo'
    assert rows['native_repeat']['team_delay_vs'] == rows['native']['90']['team_delay_vs']
    toy.load(states['mappo'])
    assert toy.time == 420 and toy.total == 120 and toy.spent == 300+6*90


def test_budget_and_world_disjointness():
    assert sum(BUDGET.values()) == 478410
    assert sum(v for k,v in BUDGET.items() if k.startswith('step0')) == 47010
    assert sum(5*(300+60*j+90) for w in worlds('main')
               for j in measurement_indices(w['seed'])) == 158100
    assert len(STEP0_SEEDS) == 8 and 969100 not in STEP0_SEEDS
    assert len(STEP0_SEEDS)*6*sum(t+90 for t in STEP0_TIMES) == 46080
    require_main_budget()
    a, b = worlds('step0'), worlds('main')
    assert len(a) == 12 and len(b) == 16
    assert not {w['seed'] for w in a} & {w['seed'] for w in b}
    assert len({(w['penetration'],w['demand']) for w in a}) == 12
    with pytest.raises(ValueError):
        checked('E:/sealed/example.json')


def test_durable_precharge_refuses_duplicate_and_step0_overflow(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.charge('a', 59160, 'step0')
    with pytest.raises(ValueError):
        ledger.charge('a', 1, 'step0')
    with pytest.raises(ValueError):
        ledger.charge('b', 841, 'step0')
    assert len(ledger.rows()) == 1


class ReplayToy:
    """A new t0..300 run, with recurrent controller state and full-prefix tape."""
    def __init__(self, *, corrupt=False):
        self.time, self.total, self.name, self.memory, self.actual = 300, 0., 'native', 0, 300
        self.closed, self.selected, self.corrupt = False, [], corrupt
        self.belief = SimpleNamespace(payload=lambda:{'observed_speed':7})
    def select(self, name):
        self.selected.append(name)
        if name != self.name:
            self.memory = 0
        self.name = name
    def advance(self, seconds):
        best = 'pi' if self.time < 360 else 'mappo'
        self.total += seconds*(1 if self.name == best else 5)+self.memory*.01
        self.memory += seconds
        self.time += seconds
        self.actual += seconds
    def metrics(self):
        return {'team_delay_vs':self.total}
    def fingerprint(self):
        return dict(time=self.time, total=self.total, name=self.name,
                    memory=self.memory, tape='bad' if self.corrupt else 'complete')
    def close(self):
        self.closed = True


def test_full_rerun_oracle_rebuilds_trunk_and_closes_every_instance():
    from tools.c1switch.rerun import candidates
    made = []
    def factory_for(name):
        def create():
            b = ReplayToy()
            made.append(b)
            return b
        return create
    rows = candidates(factory_for, ['pi', 'pi', 'native'])
    assert len(made) == 6 and all(b.closed for b in made)
    assert sum(b.actual for b in made) == 6*(480+90)
    assert all(b.selected[:3] == ['pi', 'pi', 'native'] for b in made)
    assert rows['native']['prefix']['memory'] == 60
    assert choose({c:rows[c]['60']['team_delay_vs'] for c in CONTROLLERS}) == 'mappo'
    assert rows['native']['90'] == rows['native_repeat']['90']


def test_full_rerun_rejects_prefix_before_any_alternative_tick():
    from tools.c1switch.rerun import replay
    reference = replay(ReplayToy, [], 'native')
    broken = ReplayToy(corrupt=True)
    with pytest.raises(ValueError, match='trunk bitwise'):
        replay(lambda:broken, [], 'pi', expected_prefix=reference['prefix'])
    assert broken.closed and broken.actual == 300 and broken.selected == []


def test_full_rerun_native_continuation_gate_precedes_nonnative():
    from tools.c1switch.rerun import candidates
    made = []
    def factory_for(name):
        def create():
            b = ReplayToy()
            if name == 'native_repeat':
                original = b.advance
                def changed(seconds):
                    original(seconds)
                    b.total += 1
                b.advance = changed
            made.append(name)
            return b
        return create
    with pytest.raises(ValueError, match='continuation bitwise'):
        candidates(factory_for, [])
    assert made == ['native', 'native_repeat']


def test_retired_saved_state_runner_cannot_simulate(tmp_path):
    from tools.c1switch.runner import run
    with pytest.raises(ValueError, match='retired'):
        run(tmp_path, 'step0', worlds('step0')[0])
    with pytest.raises(ValueError, match='retired'):
        run(tmp_path, 'main', worlds('main')[0], 'oracle')


def test_file_handshake_accepts_only_matching_k8_team_binding(tmp_path):
    binding = dict(team_source_entries=True, checkpoint_sha256='frozen')
    predictor = FilePredictor(tmp_path, binding, timeout_s=.01)
    payload = Belief().payload()
    request = dict(id='toy', belief=payload, binding=binding, K=8, horizon_s=60)
    key = canonical(request)
    result = dict(request_sha256=key, binding=binding,
                  samples={c:[float(i)]*8 for i,c in enumerate(CONTROLLERS)})
    write(tmp_path/(key+'.result.json'), result)
    assert predictor.predict(payload, 'toy')['native'] == 0
    result['binding'] = dict(binding, checkpoint_sha256='wrong')
    write(tmp_path/(key+'.result.json'), result)
    with pytest.raises(ValueError):
        predictor.predict(payload, 'toy')
    with pytest.raises(ValueError):
        FilePredictor(tmp_path, {})


def test_model_rule_cannot_select_road_only_or_mismatched_panel():
    rows = [dict(checkpoint=c, cohort=cohort, world=cohort, start=300,
                 panel='original', components='original', history='longest_available',
                 settings='inferred', teacher_forced=True, clock_s=60,
                 hdv_abs_endpoint_error_m=1 if c == '87ed02e0' else 2)
            for c in ('87ed02e0','b8797242') for cohort in ('pi','wall')]
    support = {c:dict(team_source_entries=True) for c in ('87ed02e0','b8797242')}
    assert select_model(rows, support)['checkpoint'] == '87ed02e0'
    support['87ed02e0']['team_source_entries'] = False
    assert select_model(rows, support)['checkpoint'] == 'b8797242'
    with pytest.raises(ValueError):
        select_model([], support)
    support['87ed02e0']['team_source_entries'] = True
    with pytest.raises(ValueError):
        select_model(rows[:-1], support)


def test_ctrl0_snapshot_keeps_raw_executor_and_independent_memory(monkeypatch):
    import sys
    from tools.c1switch.sumo import Controller
    # Exact failure mechanism: a transparent wrapper's delegated __getattr__
    # cannot be reconstructed before its executor attribute is assigned.
    class Proxy:
        def __init__(self, executor):
            self.executor = executor
        def __getattr__(self, name):
            return getattr(self.executor, name)
        def __call__(self, state):
            self.executor.memory.append(state)
            return 'typed_zero'
    raw = SimpleNamespace(memory=[])
    with pytest.raises(RecursionError):
        copy.deepcopy(Proxy(raw))
    monkeypatch.setitem(sys.modules, 'zero', SimpleNamespace(ZeroResidual=Proxy))
    controller = Controller.__new__(Controller)
    controller.name, controller.exe, controller.previous = 'ctrl0', raw, None
    assert controller.command(SimpleNamespace(state=lambda:'tick')) == ('typed_zero', {})
    restored = copy.deepcopy(controller)
    assert restored.exe is not raw and restored.exe.memory == ['tick']
    restored.command(SimpleNamespace(state=lambda:'next'))
    assert raw.memory == ['tick'] and restored.exe.memory == ['tick','next']
