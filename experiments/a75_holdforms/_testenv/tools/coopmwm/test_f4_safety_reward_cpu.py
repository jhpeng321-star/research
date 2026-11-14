"""F4 physical-observation/reward contracts; no simulator, remote or GPU work."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from . import track2_reward as reward, track2_safety_reward as safety
from .merge_features import ObservedVehicle as Vehicle, SensorReport as Report


CFG = dict(track2_reward=dict(weight=1., provenance='F4 CPU fixture'),
           f4_close_call=dict(enabled=True, registration=safety.registration()))


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
    result = reward.neighbourhood_reward(own, 'c1', now_s=301., weight=1., first_actuation_s=300.,
                                         close_call_enabled=True)
    assert result['close_call']['event_count'] == expected


def test_rear_end_and_neighbour_neighbour_events_are_both_penalized_once():
    own = report(source=vehicle('c1', 0., 20.), others=[vehicle('rear', -15., 30.),
        vehicle('h1', 15., 10.), vehicle('h2', 25., 0.), vehicle('other_lane', 1., 0., 1)])
    out = reward.neighbourhood_reward(own, 'c1', now_s=301., weight=1., first_actuation_s=300.,
                                      close_call_enabled=True)
    pairs = {(e['follower_id'], e['leader_id']) for e in out['close_call']['events']}
    assert pairs == {('rear', 'c1'), ('c1', 'h1'), ('h1', 'h2')}
    assert out['close_call_penalty'] == -3.


def test_atomic_reports_never_splice_geometry_and_own_sensing_wins():
    own = report(302., others=[vehicle('p', 90., 20.), vehicle('h1', 15., 10.)])
    remote = Report(301., vehicle('p', 90., 20.), (0,),
        (vehicle('c1', 1., 40.), vehicle('h1', 2., 0.),
         vehicle('remote_follower', 150., 20.), vehicle('remote_leader', 165., 10.)))
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
                                      partner_ids=['p'], first_actuation_s=300., close_call_enabled=True)
    pairs = {(e['follower_id'], e['leader_id']) for e in out['close_call']['events']}
    assert pairs == {('c1', 'h1'), ('remote_follower', 'remote_leader')}
    assert out['close_call']['event_count'] == 2
    # A newly started move at 301 cannot be charged for a t-1 packet from 301.
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
                                      partner_ids=['p'], first_actuation_s=301., close_call_enabled=True)
    assert out['close_call']['event_count'] == 1


@pytest.mark.parametrize('timestamp,position,selected,expected', [
    (301., 90., True, 1), (302., 90., True, 0), (300., 90., True, 0),
    (303., 90., True, 0), (301., 101., True, 0), (301., 90., False, 0)])
def test_remote_timestamp_range_and_authorization(timestamp, position, selected, expected):
    own = report(302., others=[])
    remote = Report(timestamp, vehicle('p', position, 20.), (0,),
                    (vehicle('rear', position-15., 30.),))
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
        partner_ids=['p'] if selected else [], first_actuation_s=300., close_call_enabled=True)
    assert out['close_call']['event_count'] == expected


def test_range_clipping_and_duplicate_packets_do_not_add_events():
    own = report(302., others=[vehicle('p', 50., 20.), vehicle('h1', 15., 10.),
                             vehicle('far', 200., 0.)])
    remote = Report(301., vehicle('p', 50., 20.), (0,), own.vehicles[:2])
    out = reward.neighbourhood_reward(own, 'c1', {'p':remote}, now_s=302., weight=1.,
        partner_ids=['p', 'p'], first_actuation_s=300., close_call_enabled=True)
    assert out['close_call']['event_count'] == 1
    assert 'far' not in out['neighbour_ids']


def test_a_unaffected_even_if_reward_config_or_move_evidence_invalid():
    original = [episode(dict(time_s='invalid', sample=None))]
    cfg = dict(track2_reward='invalid', f4_close_call='invalid')
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


def production_config(root):
    from . import launch_production as production, rerun_cpus as rc
    cfg = rc.runtime_config({}, root=str(root), reference_root=str(root/'reference'),
        source_root=str(root/'source'), common_checkpoint_sha256='a'*64, track2=True)
    cfg.update(launch_optimization=deepcopy(production.OPTIMIZATION),
               worlds_by_update={'1': [dict(seed=986000), dict(seed=986001)]},
               f4_close_call=deepcopy(CFG['f4_close_call']))
    return cfg


def multi_cav_step():
    s = step()
    s['sample']['context']['members'].append('c2')
    s['sample']['track2']['reward_neighbourhoods']['c2'] = []
    s['sample']['track2']['outcome_neighbourhoods']['c2'] = []
    s['reward_observations']['c2'] = dict(own=asdict(report(
        source=vehicle('c2', 500., 20.), others=[])), partner_ids=[], partner_reports={})
    s['sample']['track2_trigger'] = dict(tables=[[None, dict(family='wall')]], actions=[1])
    return s


def test_real_b_r_penalties_publish_via_production_monitor_and_reload_history(tmp_path, monkeypatch):
    from . import launch_production as production, launch_training, track2_monitor as monitor
    from .exact_grade import _seal, verify_sealed
    cfg = production_config(tmp_path)
    native = production.backend_config(cfg)
    assert native['f4_close_call'] == CFG['f4_close_call']
    receipts, rows = {}, []
    for u in (1, 2):
        for arm in ('A', 'B', 'R'):
            e = episode(multi_cav_step(), arm=arm)
            e.update(totals=dict(collisions=0., teleports=0., hard_brake=0.), ttc_events=1)
            out = reward.reward_update(native, arm, u, [e])
            learning = dict(layer=[dict(accepted=True, kl=.002)], bank_move_ce=.7)
            metrics = production.update_metrics(arm, [e], learning, out, augmented_examples=0)
            if arm != 'A':
                assert out['receipt']['close_call_events'] == 1
                assert out['receipt']['move_exposed_cav_ticks'] == 1
                # Two CAV outcomes, only one event: penalty is the mean, not raw count.
                assert out['receipt']['close_call_penalty_sum'] == -.5
                assert metrics['track2_reward'] == out['receipt']
            receipts[f'{arm}_UPDATE_{u:03}.json'] = _seal(dict(metrics=metrics), 'update_sha256')
        monkeypatch.setattr(launch_training, 'wait_for', lambda path, root: receipts[path.name])
        row = launch_training.monitor_update(cfg, u)
        verify_sealed(row, 'receipt_sha256')
        assert (tmp_path/f'LEARNING_MONITOR_u{u:03}.json').is_file()
        rows.append(row)
    status = monitor.read_monitor(tmp_path)
    assert status['updates'] == dict(A=2, B=2, R=2)
    for arm in ('B', 'R'):
        history = monitor.monitor_history(tmp_path, arm)
        assert history == [next(r for r in row['arms'] if r['arm'] == arm) for row in rows]
        assert history[-1]['components']['f4_close_call'] == CFG['f4_close_call']
        assert history[-1]['metrics']['track2_reward']['close_call_penalty_sum'] == -.5
    summary = (tmp_path/'CLEAN1_STATUS.md').read_text()
    assert 'close_call_penalty_sum=-0.5' in summary
    assert not (tmp_path/'FAULT.json').exists()


@pytest.mark.parametrize('change', [
    dict(close_call=None), dict(close_call=dict(safety.REGISTRATION, weight=2.)),
    dict(close_call_events=-1), dict(close_call_events=True), dict(close_call_events=0),
    dict(move_exposed_cav_ticks=0), dict(move_exposed_cav_ticks=3),
    dict(move_exposed_cav_ticks=1.5), dict(close_call_penalty_sum=float('nan')),
    dict(close_call_penalty_sum=float('-inf')), dict(close_call_penalty_sum=.5),
    dict(close_call_penalty_sum=0.), dict(close_call_penalty_sum=-2.),
    dict(close_call_penalty_sum=-.1), dict(reward_sum=0.),
])
def test_enabled_monitor_rejects_invalid_safety_contract_before_publication(tmp_path, change):
    from . import launch_production as production, track2_monitor as monitor
    from .test_track2_monitor_cpu import real_metrics
    cfg = production_config(tmp_path)
    out = reward.reward_update(production.backend_config(cfg), 'B', 1, [episode(multi_cav_step())])
    metrics = real_metrics('B', track2_reward=dict(out['receipt'], **change))
    with pytest.raises(ValueError, match='track2_reward'):
        monitor.monitor_update(production.monitor_config(cfg), 'B', 1, metrics)
    assert not (tmp_path/'learning_monitor/B/u001.json').exists()


@pytest.fixture(scope='module')
def pre_f4_modules():
    """Differential oracle: the declared launch parent's actual implementations."""
    from pathlib import Path
    import subprocess
    import os
    import hashlib
    import types
    modules = {}
    for name in ('track2_reward', 'track2_backend', 'track2_monitor', 'launch_production'):
        frozen = os.environ.get('COOPMWM_PRE_F4_BASELINE_DIR')
        raw = ((Path(frozen)/(name+'.py')).read_bytes() if frozen else subprocess.check_output(
            ['git', 'show', f'8bf7f533:tools/coopmwm/{name}.py'], cwd=Path(__file__).resolve().parents[2]))
        assert hashlib.sha256(raw).hexdigest() == {
            'launch_production':'abf898930f31b84a8ff7880308e6ff08c7dc11c83985f9d8800250981f341586',
            'track2_backend':'c819aab35014c156f864522224300f36a85596953a74896218f06621c9cd1eed',
            'track2_monitor':'5ac1b22bf951e2097089be4d97d2bea7b98ccf659d954ad0639667ae7404a055',
            'track2_reward':'6aeca5065b0bbc43e4736c3ce6f014621176524131d57c6f7022cf934b39b48b'}[name]
        source = raw.decode('utf-8')
        module = types.ModuleType('tools.coopmwm._pre_f4_'+name)
        module.__package__ = 'tools.coopmwm'
        exec(compile(source, f'<pre-F4 {name}>', 'exec'), module.__dict__)
        modules[name] = module
    return modules


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('off', ['absent', False])
@pytest.mark.parametrize('context', ['active', 'idle', 'unparsed', 'unordered', 'terminal', 'empty'])
def test_absent_off_reward_bytes_and_historical_targets_match_launch_parent(pre_f4_modules, arm, off, context):
    from . import track2_backend as backend
    cfg = dict(track2_reward=deepcopy(CFG['track2_reward']))
    if off is False:
        cfg['f4_close_call'] = False
    s = step(active=context != 'idle')
    steps = [s]
    if context == 'unparsed':
        s['sample']['context']['move_offsets']['c1'] = 'historical unparsed offset'
    elif context == 'unordered':
        steps.append(step(299))  # Historically accepted by the reward adapter.
    elif context == 'terminal':
        s['sample']['track2']['outcome_neighbourhoods']['c1'] = []
        s['reward_observations']['c1'] = dict(terminal_exit=dict(cav_id='c1', time_s=301.,
            before_time_s=300, present_before=True, present_after=False,
            source='sumo_observation_transition'))
    elif context == 'empty':
        s['sample']['context']['members'] = []
    episodes = [episode(*steps, arm=arm)]
    before = deepcopy(episodes)
    old = pre_f4_modules['track2_reward'].reward_update(cfg, arm, 1, episodes)
    new = reward.reward_update(cfg, arm, 1, episodes)
    assert json.dumps(new, allow_nan=False) == json.dumps(old, allow_nan=False)
    assert episodes == before
    native = dict(cfg, track2_reward=True, track2_reward_weight=1.,
                  track2_reward_provenance=CFG['track2_reward']['provenance'])
    assert json.dumps(reward.training_episodes(native, arm, episodes)) == json.dumps(
        pre_f4_modules['track2_reward'].training_episodes(native, arm, episodes))
    by_key = {('episode', s['time_s']): s['sample'] for s in steps}
    targets = []
    for module in (pre_f4_modules['track2_backend'], backend):
        runtime = module.ArmRuntime.__new__(module.ArmRuntime)
        runtime.cfg, runtime.arm, runtime.update = cfg, arm, 1
        targets.append(runtime._reward_targets(old, by_key, set(by_key)))
    assert targets[0] == targets[1]


@pytest.mark.parametrize('off', ['absent', False])
def test_default_off_monitor_bytes_and_history_match_launch_parent(tmp_path, pre_f4_modules, off):
    from . import launch_production as production, track2_monitor as monitor
    from .test_track2_monitor_cpu import real_metrics
    cfg = production_config(tmp_path)
    cfg.pop('f4_close_call')
    if off is False:
        cfg['f4_close_call'] = False
    assert production.backend_config(cfg) == pre_f4_modules['launch_production'].backend_config(cfg)
    native = production.monitor_config(cfg)
    old_monitor = pre_f4_modules['track2_monitor']
    rows = []
    for u in (1, 2):
        receipt = pre_f4_modules['track2_reward'].reward_update(
            production.backend_config(cfg), 'B', u, [episode(step())])['receipt']
        metrics = real_metrics('B', track2_reward=receipt)
        expected = old_monitor._monitor_row(native, 'B', u, metrics, rows, {})
        actual = monitor.monitor_update(native, 'B', u, metrics)
        assert json.dumps(actual) == json.dumps(expected)
        rows.append(actual)
    assert monitor.monitor_history(tmp_path, 'B') == old_monitor.monitor_history(tmp_path, 'B')


@pytest.mark.parametrize('slot', [True, None, 0, 'true', {}, dict(enabled=True),
    dict(enabled=True, registration=dict(safety.REGISTRATION, weight=2.)),
    dict(enabled=True, registration=dict(safety.REGISTRATION, provenance='unregistered')),
    dict(enabled=True, registration=dict(safety.REGISTRATION, weight=True)),
    dict(enabled=True, registration=dict(safety.REGISTRATION, arms=['B']))])
def test_f4_opt_in_requires_exact_common_registration(tmp_path, slot):
    from . import launch_production as production, track2_backend as backend
    cfg = production_config(tmp_path)
    cfg['f4_close_call'] = slot
    with pytest.raises(ValueError, match='F4'):
        production.backend_config(cfg)
    with pytest.raises(ValueError, match='F4'):
        production.monitor_config(cfg)
    native = dict(CFG, f4_close_call=slot)
    for arm in ('B', 'R'):
        with pytest.raises(ValueError, match='F4'):
            reward.reward_update(native, arm, 1, [episode(step())])
        runtime = backend.ArmRuntime.__new__(backend.ArmRuntime)
        runtime.cfg, runtime.arm, runtime.update = native, arm, 1
        with pytest.raises(ValueError, match='F4'):
            runtime._reward_targets(envelope(step(), arm=arm), {}, set())
    assert reward.reward_update(native, 'A', 1, None)['receipt'] is None


def test_continuation_recipe_explicitly_binds_fixed_b_r_weight_and_provenance(tmp_path):
    from pathlib import Path
    from . import launch_production as production, track2_monitor as monitor
    recipe = json.loads((Path(__file__).resolve().parents[2]/'F4_CONTINUATION_RECIPE.json').read_text())
    cfg = production_config(tmp_path)
    cfg.update(recipe['config_patch'])
    assert recipe['config_patch'] == dict(f4_close_call=CFG['f4_close_call'])
    native = production.backend_config(cfg)
    assert safety.settings(native) == safety.registration()
    assert monitor._components(production.monitor_config(cfg))['f4_close_call'] == CFG['f4_close_call']
    outputs = [reward.reward_update(native, arm, 1, [episode(step())]) for arm in ('B', 'R')]
    assert outputs[0]['rows'] == outputs[1]['rows']
    assert outputs[0]['receipt']['close_call_penalty_sum'] == -1.


@pytest.mark.parametrize('off', ['absent', False])
@pytest.mark.parametrize('arm', ['B', 'R'])
def test_default_off_real_optimizer_and_learning_receipt_match_parent(
        tmp_path, monkeypatch, pre_f4_modules, off, arm):
    import torch
    from . import track2_backend as backend, exploration
    from .test_track2_backend_cpu import fake_load, rows
    from .launch_production import OPTIMIZATION
    from .v2v_runtime import controller_class
    from .acceleration import Controller
    cfg = dict(deepcopy(OPTIMIZATION), track2_backend=True, run_root=str(tmp_path),
        training_seed=123, v2v=True, v2v_dim=48, v2v_intent=True, merge_features=True,
        track2_reward=deepcopy(CFG['track2_reward']))
    if off is False:
        cfg['f4_close_call'] = False
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')
    fake_load(monkeypatch)
    parent = pre_f4_modules['track2_backend']
    monkeypatch.setattr(parent, '_load_base', backend._load_base)
    # F1/F2 intentionally change execution/proposals independently of F4.
    # Hold the actual integrated policy class common while differentially
    # exercising the parent's and current F4-off reward/optimizer code paths.
    monkeypatch.setattr(parent, 'EpisodePolicy', backend.EpisodePolicy)
    runtimes, learnings = [], []
    for module in (parent, backend):
        runtime = module.build_arm_runtime(cfg, arm)['runtime']
        runtime.safety_checked = True
        p = runtime.policy
        move = backend.TriggerMove(backend.moves.Move('ease_off', ('c1',), 0, 20, .3))
        assert p.executor.start(move, ego='c1', observable_cav_ids=['c1'], now=300, until=320)
        p.last_decision = 300
        p.observe(rows(), 300)
        controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
        controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
        event = controller.events[-1]
        event['reward'] = -999.
        post = rows()
        post[0].update(v_mps=20., lane=0)
        post[1].update(s_m=-185., v_mps=10., lane=0)
        module.record_outcomes(p, event, post)
        e = dict(id='episode', arm=arm, actual_sim_s=301, steps=[event])
        out = reward.reward_update(cfg, arm, 1, [e])
        assert 'close_call' not in out['rows'][0]
        monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {}))
        learnings.append(runtime.train([e], rewards=out))
        runtimes.append(runtime)
    assert json.dumps(learnings[0]) == json.dumps(learnings[1])

    def same(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                same(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                same(x, y)
        else:
            assert a == b

    same(runtimes[0].policy.worker_state(), runtimes[1].policy.worker_state())
    same(runtimes[0].policy.layer.optimizer.state_dict(), runtimes[1].policy.layer.optimizer.state_dict())
    same(runtimes[0].policy.trigger_optimizer.state_dict(), runtimes[1].policy.trigger_optimizer.state_dict())
