"""CPU gates for the c4/lane-plans x c4/lane-policy merge (LANEPOLICY_READY c25218ca).

The pinned integration: a committed plan member's lane action supersedes the policy lane
choice and enters the SAME safe changeLane tick actuator (SUMO's own lane-change safety
model); the tick outcome on the event is the single record of truth - the apply-time
placeholder never claims execution and is never counted as blocked in the branch audit.
Real torch, real controller/apply/audit code; only the SUMO engine itself is faked (the
house CPU pattern). No SUMO, no GPU.
"""
import copy
import hashlib
from unittest.mock import patch

import pytest
import torch

from tools.coopmwm.acceleration import Controller, Policy as BasePolicy
from tools.coopmwm import lane_plans as lp, v2v_lane as vl
from tools.coopmwm.sumo_verify import verify_controller_class
from tools.coopmwm.test_lane_plans_cpu import rows as plan_rows
from tools.coopmwm.test_v2v_lane_cpu import wake
from tools.coopmwm.v2v_runtime import controller_class


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def runtime_controller(policy, seed=4):
    return controller_class(Controller)(policy, torch.Generator().manual_seed(seed), False)


def lane_plan(members, acts, lane_by_vid):
    """A committed-plan-shaped dict (60 s profile plan + one lane attempt per member)."""
    return dict(id='Mplan', members=list(members), duration_s=60,
                profile=[[0.]*300 for _ in members], knot_times_s=[0., 60.],
                origin='sampled', regime='ordinary', family='MPPI',
                lane_actions=[list(a) for a in acts],
                lane_envelope={v: dict(lane=int(lane_by_vid[v]),
                                       left=bool(lane_by_vid[v] > 0),
                                       right=bool(lane_by_vid[v] == 0)) for v in members},
                lane_predictions=[], uses_lane_actions=True)


def test_plan_lane_targets_map_clip_and_skip_despawned():
    sensor = [dict(vid='c0', s_m=10., v_mps=10., lane=0, is_cav=True, length_m=5.),
              dict(vid='c1', s_m=20., v_mps=10., lane=1, is_cav=True, length_m=5.)]
    got = vl.plan_lane_targets({'c0': lp.LANE_LEFT, 'c1': lp.LANE_RIGHT, 'c9': lp.LANE_LEFT}, sensor)
    assert got == {'c0': 0, 'c1': 2}          # clip into 0..2; despawned c9 skipped
    assert vl.plan_lane_targets({'c0': lp.LANE_KEEP}, sensor) == {}
    assert vl.plan_lane_targets(None, sensor) == {}


def test_plan_requests_supersede_policy_and_reach_the_actuator():
    torch.manual_seed(5)
    policy = vl.LaneIntentPolicy()            # exact-neutral: every policy choice is keep
    c = runtime_controller(policy)
    members = ['c0', 'c1']                    # c0 lane 1 left; c1 lane 0 right (envelope-legal)
    envelope = lp.decision_envelope(
        dict(rows=plan_rows(), state_id='s', policy_residuals={v: 0. for v in members}), members)
    assert envelope['c0']['left'] and envelope['c1']['right']
    plan = lane_plan(members, [[lp.LANE_LEFT]+[0]*59, [lp.LANE_RIGHT]+[0]*59],
                     {'c0': 1, 'c1': 0})
    c.active_plans = [dict(plan=copy.deepcopy(plan), start_s=301)]
    data = plan_rows()
    with patch('tools.coopmwm.exploration.regime', lambda r: 'ordinary'):
        c.apply(data, {v: 0. for v in members}, 301)
    sample = c.events[-1]['sample']
    assert sample['lane_requests'] == {'c0': lp.LANE_LEFT, 'c1': lp.LANE_RIGHT}
    assert 'lane_events' not in sample        # the tick actuator owns the execution record
    assert c.lane_targets == {'c0': 0, 'c1': 1}

    class SlotEnv:
        def slot_vehicle_ids(self):
            return [r['vid'] for r in data if r['is_cav']]
    tensor = c.lane_request(SlotEnv())
    ids = SlotEnv().slot_vehicle_ids()
    assert tensor.shape == (1, len(ids))
    assert tensor[0, ids.index('c0')].item() == 0. and tensor[0, ids.index('c1')].item() == 1.
    c.observe_lane(dict(lane_commands=dict(issued=2, completed=2)))
    assert c.events[-1]['lane_outcome'] == dict(requested=2, issued=2, blocked=0, executed=2,
                                                counts=dict(issued=2, completed=2))


def test_woken_policy_choice_is_superseded_on_plan_members():
    torch.manual_seed(5)
    policy = wake(vl.LaneIntentPolicy(), bias=(8., -8.))      # left almost surely
    c = runtime_controller(policy)
    members = ['c0', 'c1']                    # plan: c0 left -> 0, c1 right -> 1 (envelope-legal)
    plan = lane_plan(members, [[lp.LANE_LEFT]+[0]*59, [lp.LANE_RIGHT]+[0]*59],
                     {'c0': 1, 'c1': 0})
    c.active_plans = [dict(plan=copy.deepcopy(plan), start_s=301)]
    data = plan_rows()
    with patch('tools.coopmwm.exploration.regime', lambda r: 'ordinary'):
        c.apply(data, {v: 0. for v in members}, 301)
    assert c.events[-1]['sample']['lane_requests'] == {'c0': lp.LANE_LEFT, 'c1': lp.LANE_RIGHT}
    assert c.lane_targets['c1'] == 1          # plan right wins over the policy's left (clip 0)
    assert c.lane_targets['c0'] == 0
    assert 'lane_events' not in c.events[-1]['sample']


def _snapshot(members):
    from tools.coopmwm.test_sumo_verify_cpu import snapshot
    snap = snapshot(t0=400)
    gen = torch.Generator().manual_seed(9)
    snap['generator_state'] = gen.get_state().clone()
    snap['rng']['generator_state_sha256'] = hashlib.sha256(
        snap['generator_state'].numpy().tobytes()).hexdigest()
    snap['rng']['memory_after_t0'] = {}
    snap['samples'] = {str(t): dict(action=[[0.], [0.]], context=dict(members=list(members)),
                                    behaviour={}, memory={}) for t in range(300, 401)}
    return snap


def test_branch_audit_skips_placeholder_and_folds_the_tick_outcome():
    members = ['c0', 'c1']                    # c0 lane 1 left -> 0; c1 lane 0 right -> 1
    plan = lane_plan(members, [[lp.LANE_LEFT]+[0]*59, [lp.LANE_RIGHT]+[0]*59], {'c0': 1, 'c1': 0})
    cls = controller_class(verify_controller_class())
    c = cls(BasePolicy(), torch.Generator().manual_seed(9), _snapshot(members), plan)
    data = plan_rows()
    with patch('tools.coopmwm.exploration.regime', lambda r: 'ordinary'):
        c.apply(data, {v: 0. for v in members}, 400)
    sample = c.events[-1]['sample']
    assert sample['lane_requests'] == {'c0': lp.LANE_LEFT, 'c1': lp.LANE_RIGHT}
    assert 'lane_events' not in sample        # popped by the runtime wrapper
    assert c.lane_targets == {'c0': 0, 'c1': 1}
    assert 'lane_actions' not in c.audit      # the placeholder was never counted as blocked
    c.observe_lane(dict(lane_commands=dict(issued=1, noop=1)))
    assert c.events[-1]['lane_outcome']['requested'] == 2
    assert c.audit['lane_actions'] == dict(requests=2, issued=1, blocked=1, executed=0)


def test_plain_unbound_verify_controller_records_never_counts():
    members = ['c0', 'c1']
    plan = lane_plan(members, [[lp.LANE_LEFT]+[0]*59, [lp.LANE_RIGHT]+[0]*59], {'c0': 1, 'c1': 0})
    c = verify_controller_class()(BasePolicy(), torch.Generator().manual_seed(9),
                                  _snapshot(members), plan)
    data = plan_rows()
    with patch('tools.coopmwm.exploration.regime', lambda r: 'ordinary'):
        c.apply(data, {v: 0. for v in members}, 400)
    events = c.events[-1]['sample']['lane_events']        # plain path: the per-vid record stays
    for e in events.values():
        assert e['blocked'] == 'unbound' and not e['issued'] and not e['executed']
    assert 'lane_actions' not in c.audit      # no executor verdict -> nothing claimed


def test_imitation_pairs_use_the_tick_outcome_when_the_placeholder_is_popped():
    from tools.coopmwm.exact_grade import imitation_pairs
    steps = [dict(time_s=300, lane_outcome=dict(requested=2, issued=2, blocked=0, executed=2),
                  sample=dict(action=[[0.], [0.]], context=dict(members=['c0', 'c1']),
                              behaviour=dict(executed=True, proposal_members=['c0', 'c1']),
                              plan_executions=[dict(plan_id='lane1', proposal_members=['c0', 'c1'])],
                              lane_requests={'c0': lp.LANE_LEFT, 'c1': lp.LANE_RIGHT}))]
    entry = dict(start_s=300, duration_s=1, plan_id='lane1', weight=1., lane_actions=[[1], [2]])
    sample, _w = imitation_pairs(dict(steps=steps), entry)[0]
    assert sample['lane_target'] == {'c0': 1, 'c1': 2}
    assert sample['lane_target_outcome_summary'] == dict(requested=2, issued=2, blocked=0,
                                                         executed=2)
    assert 'lane_target_outcome' not in sample
