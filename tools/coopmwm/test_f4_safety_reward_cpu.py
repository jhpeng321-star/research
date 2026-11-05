"""F4 physical-observation/reward contracts; no simulator, remote or GPU work."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from . import track2_reward as reward, track2_safety_reward as safety
from .merge_features import ObservedVehicle as Vehicle, SensorReport as Report


CFG = dict(track2_reward=dict(weight=1., provenance='F4 CPU fixture'))


def vehicle(vid, s, v, lane=0):
    return Vehicle(vid, s, lane, v, 5.)


def report(time=301., source=None, others=None):
    return Report(time, source or vehicle('c1', 0., 20.), (0, 1, 2),
                  tuple(others if others is not None else [vehicle('h1', 15., 10.)]))


def step(time=300, *, offset=-.3, active=True, packet=None):
    sample = dict(context=dict(members=['c1'], move_offsets=dict(c1=offset)),
        track2=dict(reward_neighbourhoods=dict(c1=['h1']),
                    outcome_neighbourhoods=dict(c1=['h1']), active_moves=[]))
    if active:
        sample['track2']['active_moves'] = [dict(start=300, until=304,
            move=dict(family='wall', members=['c1'], onset=0, hold=4, depth=.3))]
    return dict(time_s=time, reward=-999., sample=sample,
        reward_observations=dict(c1=packet or dict(own=asdict(report(time+1)),
                                                   partner_ids=[], partner_reports={})))


def episode(*steps, arm='B'):
    return dict(id='episode', arm=arm, steps=list(steps))


def envelope(*steps, arm='B'):
    # Exercise the production primitive JSON boundary.
    return reward.reward_update(CFG, arm, 1, json.loads(json.dumps([episode(*steps, arm=arm)])))


def test_registered_penalty_post_action_and_b_r_identity_without_mutation():
    original = [episode(step())]
    before = deepcopy(original)
    b = reward.reward_update(CFG, 'B', 1, original)
    r = reward.reward_update(CFG, 'R', 1, original)
    assert original == before
    assert b['rows'] == r['rows']
    assert {k:v for k,v in b['receipt'].items() if k != 'arm'} == {
        k:v for k,v in r['receipt'].items() if k != 'arm'}
    row, = b['rows']
    assert row['reward'] == pytest.approx(row['own_reward'] + row['neighbour_rewards']['h1'] - 1.)
    assert row['close_call']['events'][0]['ttc_s'] == 1.
    assert row['close_call']['events'][0]['timestamp_s'] == 301.
    assert b['receipt']['close_call_events'] == 1
    assert b['receipt']['close_call_penalty_sum'] == -1.
    assert b['receipt']['close_call'] == safety.REGISTRATION
    assert safety.REGISTRATION['weight'] == 1.
    assert 'no tuning' in safety.REGISTRATION['provenance']


@pytest.mark.parametrize('offset,active,expected', [(0., True, 0), (-.3, False, 0), (-.3, True, 1)])
def test_noop_delayed_onset_rejected_proposal_are_not_move_exposure(offset, active, expected):
    s = step(offset=offset, active=active)
    s['sample']['track2_trigger'] = dict(accepted=dict(c1=False)) if not active else {}
    assert envelope(s)['rows'][0]['close_call']['event_count'] == expected


def test_attribution_retains_zero_tail_expires_and_does_not_cross_episodes():
    out = envelope(step(offset=0.), step(301), step(302, offset=0.), step(303, offset=0.), step(304))
    assert [r['close_call']['event_count'] for r in out['rows']] == [0, 1, 1, 1, 0]
    assert [r['close_call']['first_actuation_s'] for r in out['rows']] == [None, 301, 301, 301, None]
    out = reward.reward_update(CFG, 'B', 1, [episode(step()),
        dict(id='different', steps=[step(301, offset=0.)])])
    assert [r['close_call']['event_count'] for r in out['rows']] == [1, 0]


@pytest.mark.parametrize('gap,closing,expected', [(15., 10., 0), (14.999, 10., 1),
    (0., 10., 1), (-1., 10., 1), (10., 0., 0), (10., -2., 0), (0., 1e-6, 0)])
def test_ttc_strict_threshold_overlap_and_nonclosing(gap, closing, expected):
    own = report(source=vehicle('c1', 0., 10.+closing),
                 others=[vehicle('h1', gap+5., 10.)])
    result = reward.neighbourhood_reward(own, 'c1', now_s=301., weight=1., first_actuation_s=300.)
    assert result['close_call']['event_count'] == expected


def test_rear_end_and_neighbour_neighbour_events_are_both_penalized_once():
    own = report(source=vehicle('c1', 0., 20.), others=[vehicle('rear', -15., 30.),
        vehicle('h1', 15., 10.), vehicle('h2', 25., 0.), vehicle('other_lane', 1., 0., 1)])
    out = reward.neighbourhood_reward(own, 'c1', now_s=301., weight=1., first_actuation_s=300.)
    pairs = {(e['follower_id'], e['leader_id']) for e in out['close_call']['events']}
    assert pairs == {('rear', 'c1'), ('c1', 'h1'), ('h1', 'h2')}
    assert out['close_call_penalty'] == -3.


def test_atomic_reports_never_splice_geometry_and_own_sensing_wins():
    own = report(302., others=[vehicle('p', 90., 20.), vehicle('h1', 15., 10.)])
    remote = Report(301., vehicle('p', 90., 20.), (0,),
        (vehicle('c1', 1., 40.), vehicle('h1', 2., 0.),
         vehicle('remote_follower', 150., 20.), vehicle('remote_leader', 165., 10.)))
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
                                      partner_ids=['p'], first_actuation_s=300.)
    pairs = {(e['follower_id'], e['leader_id']) for e in out['close_call']['events']}
    assert pairs == {('c1', 'h1'), ('remote_follower', 'remote_leader')}
    assert out['close_call']['event_count'] == 2
    # A newly started move at 301 cannot be charged for a t-1 packet from 301.
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
                                      partner_ids=['p'], first_actuation_s=301.)
    assert out['close_call']['event_count'] == 1


@pytest.mark.parametrize('timestamp,position,selected,expected', [
    (301., 90., True, 1), (302., 90., True, 0), (300., 90., True, 0),
    (303., 90., True, 0), (301., 101., True, 0), (301., 90., False, 0)])
def test_remote_timestamp_range_and_authorization(timestamp, position, selected, expected):
    own = report(302., others=[])
    remote = Report(timestamp, vehicle('p', position, 20.), (0,),
                    (vehicle('rear', position-15., 30.),))
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
        partner_ids=['p'] if selected else [], first_actuation_s=300.)
    assert out['close_call']['event_count'] == expected


def test_range_clipping_and_duplicate_packets_do_not_add_events():
    own = report(302., others=[vehicle('p', 50., 20.), vehicle('h1', 15., 10.),
                             vehicle('far', 200., 0.)])
    remote = Report(301., vehicle('p', 50., 20.), (0,), own.vehicles[:2])
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
        partner_ids=['p', 'p'], first_actuation_s=300.)
    assert out['close_call']['event_count'] == 1
    assert 'far' not in out['neighbour_ids']


def test_a_unaffected_even_if_reward_config_or_move_evidence_invalid():
    original = [episode(dict(time_s='invalid', sample=None))]
    cfg = dict(track2_reward='invalid')
    assert reward.training_episodes(cfg, 'A', original) == (original, None)
    assert reward.training_episodes(cfg, 'A', original)[0] is original
    assert reward.layer_records(cfg, 'A', original) == ([], None)
    # A production bridge also bypasses B/R registration validation.
    out = reward.reward_update(cfg, 'A', 1, original)
    assert out['rows'] == [] and out['receipt'] is None


def test_terminal_exit_zero_and_post_observation_required():
    s = step()
    s['reward_observations'] = dict(c1=dict(terminal_exit=dict(cav_id='c1', time_s=301.,
        before_time_s=300, present_before=True, present_after=False,
        source='sumo_observation_transition')))
    row, = envelope(s)['rows']
    assert row['reward'] == row['close_call']['penalty'] == 0.
    assert row['close_call']['first_actuation_s'] == 300
    s = step()
    del s['reward_observations']
    with pytest.raises(ValueError, match='terminal post-action'):
        envelope(s)


def test_production_target_validation_and_return_use_safety_penalty():
    from . import track2_backend as backend
    from .accel_learner import UNIT
    runtime = backend.ArmRuntime.__new__(backend.ArmRuntime)
    runtime.cfg, runtime.arm, runtime.update = CFG, 'B', 1
    s = step()
    by_key = {('episode', 300):s['sample']}
    out = envelope(s)
    targets = runtime._reward_targets(out, by_key, set(by_key))
    assert targets[('episode', 300)] == [out['rows'][0]['reward']]
    returns, _ = runtime._temporal_targets([episode(s)], targets)
    assert returns[('episode', 300)] == [out['rows'][0]['reward']/UNIT]
    for mutation in ('missing', 'weight', 'count', 'time', 'domain', 'formula'):
        bad = deepcopy(out)
        row = bad['rows'][0]
        if mutation == 'missing':
            del row['close_call']
        elif mutation == 'weight':
            row['close_call']['registration']['weight'] = 2.
        elif mutation == 'count':
            row['close_call']['event_count'] = 0
        elif mutation == 'time':
            row['close_call']['events'][0]['timestamp_s'] = 300.
        elif mutation == 'domain':
            row['close_call']['events'][0]['leader_id'] = 'unobserved'
        else:
            row['reward'] += 1.
        with pytest.raises(ValueError, match='F4|formula'):
            runtime._reward_targets(bad, by_key, set(by_key))


def test_delayed_safety_cost_reaches_originating_trigger_without_changing_horizon():
    from . import track2_backend as backend
    from .accel_learner import GAMMA, UNIT
    runtime = backend.ArmRuntime.__new__(backend.ArmRuntime)
    runtime.cfg, runtime.arm, runtime.update = CFG, 'B', 1
    tapes = [[step(offset=0.), step(301)],
             [step(offset=0., active=False), step(301, active=False)]]
    targets = []
    for steps in tapes:
        steps[0]['sample']['track2_trigger'] = dict(context=dict(members=['c1']), values=[.25])
        by_key = {('episode', s['time_s']):s['sample'] for s in steps}
        rewards = runtime._reward_targets(envelope(*steps), by_key, set(by_key))
        targets.append(runtime._temporal_targets([episode(*steps)], rewards))
    for field in (0, 1):  # return AND trigger advantage receive delayed close-call cost
        assert targets[0][field][('episode', 300)][0] - targets[1][field][('episode', 300)][0] == pytest.approx(-GAMMA/UNIT)


def test_registration_returns_independent_copies():
    value = safety.registration()
    value['weight'] = 20.
    value['arms'].append('A')
    assert safety.registration()['weight'] == 1.
    assert safety.registration()['arms'] == ['B', 'R']


def test_real_production_layer_trains_from_close_call_outcomes_and_base_stays_frozen(monkeypatch, tmp_path):
    import torch
    from . import track2_backend as backend
    from .test_track2_backend_cpu import fake_load, rows
    from .launch_production import OPTIMIZATION
    from .v2v_runtime import controller_class
    from .acceleration import Controller
    cfg = dict(deepcopy(OPTIMIZATION), track2_backend=True, run_root=str(tmp_path),
        training_seed=123, v2v=True, v2v_dim=48, v2v_intent=True,
        merge_features=True, **CFG)
    from . import exploration
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')
    fake_load(monkeypatch)
    runtime = backend.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    p = runtime.policy
    frozen = deepcopy(p.layer.base.state_dict())
    before = deepcopy(p.layer.residual.state_dict())
    move = backend.TriggerMove(backend.moves.Move('ease_off', ('c1',), 0, 20, .3))
    assert p.executor.start(move, ego='c1', observable_cav_ids=['c1'], now=300, until=320)
    p.last_decision = 300
    p.observe(rows(), 300)
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    controller.apply(rows(), {'c1':0., 'c2':0.}, 300)
    event = controller.events[-1]
    event['reward'] = -999.
    post = rows()
    post[0].update(v_mps=20., lane=0)
    post[1].update(s_m=-185., v_mps=10., lane=0)
    backend.record_outcomes(p, event, post)
    e = dict(id='episode', arm='B', actual_sim_s=301, steps=[event])
    out = reward.reward_update(cfg, 'B', 1, [e])
    assert out['receipt']['close_call_events'] == 1
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
    learning = runtime.train([e], rewards=out)
    assert learning['reward']['close_call'] == safety.REGISTRATION
    assert all(torch.equal(value, p.layer.base.state_dict()[k]) for k,value in frozen.items())
    assert any(not torch.equal(value, p.layer.residual.state_dict()[k]) for k,value in before.items())
