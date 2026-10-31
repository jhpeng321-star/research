"""Meaningful CPU checks: real ownership handback, information boundary and oracle."""
import copy
import json
from types import SimpleNamespace

import pytest

from tools.c1switch.core import Belief, Handover, branch_all, choose, medians
from tools.c1switch.design import BUDGET, CONTROLLERS, checked, worlds
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
    assert sum(BUDGET.values()) == 498360
    assert sum(v for k,v in BUDGET.items() if k.startswith('step0')) == 59160
    a, b = worlds('step0'), worlds('main')
    assert len(a) == 12 and len(b) == 24
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
