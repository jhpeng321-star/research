"""T1B synthetic CPU contracts: no SUMO, model files, or seed artifacts."""
import copy
from dataclasses import dataclass
import sys
from types import SimpleNamespace

import pytest
import torch

from . import planner_online as po


@dataclass(frozen=True)
class Move:
    family: str = 'ease_off'
    members: tuple = ('a', 'b')
    depth: float = -.5
    onset: int = 0
    hold: int = 2


class Moves:
    FAMILIES = ('none', 'ease_off', 'wall', 'gap_open', 'ramp_assist')

    def __init__(self, values=None):
        self.values = [Move()] if values is None else values
        self.seen = []

    def candidates(self, obs_state, rng, n):
        self.seen.append(copy.deepcopy(obs_state))
        return self.values[:n]

    def offsets(self, move, t):
        return {v: move.depth for v in move.members} if move.onset <= t < move.onset+move.hold else {}


def row(vid='a', s=0., cav=True):
    return dict(vid=vid, s_m=s, v_mps=15., lane=1, is_cav=cav, length_m=5.)


def record():
    return dict(ego_id='a', group=['a', 'b'], rows=[row(), row('b', 20.), row('h', 10., False)],
                time_s=60, commands={'a': .1, 'b': .2}, policy_residuals={'a': 0., 'b': 0.},
                policy_state={}, history=[])


def install_calibration(monkeypatch, probability):
    calls = []

    def calibrate_probability(p_raw, calibration):
        calls.append((p_raw, calibration))
        return probability(p_raw) if callable(probability) else probability

    # Cross-track production API stub lives only in this test.
    monkeypatch.setitem(sys.modules, po.__package__+'.planner_calibration',
                        SimpleNamespace(calibrate_probability=calibrate_probability))
    return calls


def score(monkeypatch, costs, *, probability=.8, moves=None, rec=None, **config):
    calls = install_calibration(monkeypatch, probability)
    seen = []

    def rollout(r, actions, K, H):
        seen.append((r, actions, K, H))
        return costs

    result = po.score_moves(rec or record(), None, rollout=rollout, calibration={'dev': True},
                            config=po.Config(**config), move_api=moves or Moves())
    return result, calls, seen


def test_paired_probability_and_gain_are_distinct_from_probability_of_best(monkeypatch):
    # Three wins, a tie, and four losses; pairing, not mean-only or P(argmax).
    costs = {'policy': [10., 20., 30., 40., 50., 60., 70., 80.],
             'move:0': [8., 18., 28., 40., 51., 61., 71., 81.]}
    result, calls, seen = score(monkeypatch, costs)
    assert calls == [(.375, {'dev': True})]
    assert result['scores'][0]['paired_gains'] == [2., 2., 2., 0., -1., -1., -1., -1.]
    assert result['expected_gain'] == .25
    assert result['accepted']  # calibrated .8, despite raw .375
    assert seen[0][2:] == (8, 600)
    assert seen[0][1][0] == dict(id='policy', residuals={}, window=0)


def test_calibration_can_close_raw_unanimous_gate(monkeypatch):
    result, calls, _ = score(monkeypatch, {'policy': [10.]*8, 'move:0': [5.]*8}, probability=.79)
    assert calls[0][0] == 1.
    assert not result['accepted'] and result['selected_id'] == 'policy'


def test_choose_best_expected_gain_among_calibrated_eligible_candidates(monkeypatch):
    moves = Moves([Move(depth=-.1), Move(depth=-.2), Move(depth=-.3)])
    result, _, _ = score(monkeypatch, {'policy': [10.]*8, 'move:0': [8.]*8,
                                     'move:1': [4.]*8, 'move:2': [6.]*8}, moves=moves)
    assert result['selected_move'] == moves.values[1]
    assert result['selected_id'] == 'move:1' and result['expected_gain'] == 6.


def test_highest_mean_with_failed_calibrated_gate_cannot_displace_eligible_move(monkeypatch):
    result, _, _ = score(monkeypatch, {'policy': [10.]*8, 'move:0': [9.]*8,
                                     'move:1': [0.]*4+[11.]*4},
                         moves=Moves([Move(depth=-.1), Move(depth=-.2)]),
                         probability=lambda p: .9 if p == 1. else .7)
    assert result['scores'][1]['expected_gain'] > result['scores'][0]['expected_gain']
    assert result['selected_id'] == 'move:0'


def test_high_confidence_negative_expected_gain_does_not_fire(monkeypatch):
    result, _, _ = score(monkeypatch, {'policy': [10.]*8,
                                     'move:0': [9.]*7+[100.]}, probability=1.)
    assert result['scores'][0]['p_raw'] == 7/8
    assert not result['accepted']


def test_short_move_still_scored_on_full_requested_horizon(monkeypatch):
    result, _, seen = score(monkeypatch, {'policy': [10.]*8, 'move:0': [9.]*8})
    action = seen[0][1][1]
    assert seen[0][3] == result['horizon_s'] == len(action['offset_profile']) == 600
    assert action['offset_profile'][:3] == [{'a': -.5, 'b': -.5}, {'a': -.5, 'b': -.5}, {}]
    assert not any(action['offset_profile'][2:])


@pytest.mark.parametrize('kwargs', [dict(horizon_s=300), dict(wm_samples=1), dict(cadence_s=0),
                                    dict(horizon_s=720), dict(n_candidates=-1), dict(p_star=float('nan'))])
def test_invalid_or_shortened_registration_refused(kwargs):
    with pytest.raises(ValueError):
        po.Config(**kwargs)


@pytest.mark.parametrize('probability', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_calibration_refused(monkeypatch, probability):
    with pytest.raises(ValueError, match='calibration API'):
        score(monkeypatch, {'policy': [10.]*8, 'move:0': [9.]*8}, probability=probability)


@pytest.mark.parametrize('costs', [{'policy': [10.]*7, 'move:0': [9.]*8},
                                 {'policy': [10.]*8},
                                 {'policy': [1e308]*8, 'move:0': [-1e308]*8},
                                 {'policy': [10.]*8, 'move:0': [float('nan')]*8}])
def test_incomplete_or_nonfinite_rollouts_refused(monkeypatch, costs):
    with pytest.raises(ValueError):
        score(monkeypatch, costs)


def test_missing_calibration_fails_before_rollout():
    with pytest.raises(ValueError, match='calibration is required'):
        po.score_moves(record(), None, rollout=lambda *a: pytest.fail('must not roll'),
                       calibration=None, move_api=Moves())


def test_production_move_and_calibration_api_imports_use_only_test_local_stubs(monkeypatch):
    install_calibration(monkeypatch, .9)
    stub = Moves()
    monkeypatch.setitem(sys.modules, po.__package__+'.moves', stub)
    result = po.score_moves(record(), None, calibration={'dev': True},
                            rollout=lambda r, a, K, H: {'policy': [10.]*K, 'move:0': [9.]*K})
    assert result['accepted'] and len(stub.seen) == 1


def test_none_only_and_empty_candidate_sets(monkeypatch):
    for moves in (Moves([]), Moves([Move(family='none')])):
        result, calls, seen = score(monkeypatch, {'policy': [10.]*8}, moves=moves)
        assert not result['accepted'] and not result['scores'] and not calls
        assert len(seen[0][1]) == 1


@pytest.mark.parametrize('mutate', [lambda r: r.update(truth={}),
                                    lambda r: r['rows'][0].update(hidden_type='x'),
                                    lambda r: r['rows'].append(row('far', 101.)),
                                    lambda r: r.update(group=['a', 'h']),
                                    lambda r: r['commands'].update(h=1.)])
def test_truth_global_view_and_hdv_commands_rejected(mutate):
    r = record()
    mutate(r)
    with pytest.raises(ValueError):
        po.validate_observation(r)


@pytest.mark.parametrize('time, rows', [(61., [row(), row('b', 20.), row('h', 10., False)]),
                                      (60., [row(), row('far', 101.)])])
def test_noncausal_or_nonlocal_history_rejected(time, rows):
    r = record()
    r['history'] = [dict(time_s=time, rows=rows, entrance_counts=None)]
    with pytest.raises(ValueError):
        po.validate_observation(r)


@pytest.mark.parametrize('members', [('a', 'h'), ('b',), ('a', 'far')])
def test_move_cannot_command_hdv_unobservable_or_exclude_self(monkeypatch, members):
    with pytest.raises(ValueError, match='move members'):
        score(monkeypatch, {}, moves=Moves([Move(members=members)]))


def test_offset_decoder_adds_to_base_policy_and_clamps_only_through_existing_bounds():
    from .plan_wm import profile_table, profile_apply
    actions = [dict(id='policy'), dict(id='move:0', offset_profile=[{'a': .7, 'b': -.5}])]
    table = profile_table(actions, ['a', 'b', 'h'], 'cpu', horizon=600, dynamic_policy=True)
    base = torch.tensor([[.9, .2, .3], [.9, .2, .3]])
    offset_actions = torch.tensor([[False], [True]])
    result = profile_apply(base, 0, table, (-3., 2.6), offset_actions=offset_actions)
    assert torch.equal(result[0], base[0])
    assert torch.allclose(result[1], torch.tensor([1.2, -.3, .3]))
    assert torch.equal(profile_apply(base, 2, table, (-3., 1.2)), base)


def test_legacy_absolute_and_static_residual_profile_decoding_remains_identical():
    from .plan_wm import profile_table, profile_apply
    actions = [dict(id='absolute', members=['a'], duration_s=2, profile=[[2., 2.]]),
               dict(id='residual', window=2, residuals={'a': .5})]
    base = torch.tensor([[1., .3], [1., .3]])
    table = profile_table(actions, ['a', 'h'], 'cpu', horizon=600)
    expected = torch.tensor([[2., .3], [1.5, .3]])
    assert torch.equal(profile_apply(base, 0, table, (-3., 2.6)), expected)
    assert torch.equal(profile_apply(base, 4, table, (-3., 2.6)), base)


def test_wm_adapter_uses_existing_evaluator_and_final_ownership_chain(monkeypatch):
    from . import plan_wm
    calls = []
    monkeypatch.setattr(plan_wm, 'evaluate', lambda *args, **kw: calls.append((args, kw)) or {'ok': True})
    runtime = SimpleNamespace(model=torch.nn.Linear(1, 1), mp=SimpleNamespace(policy=torch.nn.Linear(1, 1)))
    adapter = po.WMRollout(runtime, memory='belief', chunk_actions=2)
    assert adapter(record(), [], 8, 600) == {'ok': True}
    assert calls[0][0][0] is runtime and calls[0][0][3:] == (8, 600)
    assert calls[0][1] == dict(trim=False, chunk_actions=2, memory='belief', final_chain=True)


def test_wm_adapter_rejects_non_cpu_model_without_allocating_gpu():
    runtime = SimpleNamespace(model=torch.nn.Linear(1, 1, device='meta'),
                              mp=SimpleNamespace(policy=torch.nn.Linear(1, 1)))
    with pytest.raises(ValueError, match='CPU WM/runtime'):
        po.WMRollout(runtime)


class GroupMoves(Moves):
    def __init__(self, depth=-.5, onset=0, hold=2):
        super().__init__()
        self.depth, self.onset, self.hold = depth, onset, hold

    def candidates(self, obs_state, rng, n):
        self.seen.append(copy.deepcopy(obs_state))
        return [Move(members=tuple(obs_state['group']), depth=self.depth,
                     onset=self.onset, hold=self.hold)] if n else []


def runtime(monkeypatch, *, probability=.9, config=None, moves=None, failure=None, depth=-.5):
    from .acceleration import Controller
    from .v2v_runtime import Policy, controller_class
    calls = []
    install_calibration(monkeypatch, probability)

    def rollout(r, actions, K, H):
        calls.append((copy.deepcopy(r), copy.deepcopy(actions), K, H))
        if failure:
            raise failure
        return {a['id']: [10. if a['id'] == 'policy' else 9.]*K for a in actions}

    move_api = moves or GroupMoves(depth=depth)
    planner = po.OnlinePlanner(rollout=rollout, calibration={'dev': True}, rng=object(),
                               accel_bounds=(-3., 1.2), config=config, move_api=move_api)
    policy = Policy()
    controller = controller_class(Controller)(policy, torch.Generator().manual_seed(3), True,
                                               planner_online=planner)
    return controller, planner, calls, move_api


def test_runtime_cadence_commitment_release_and_base_action_preservation(monkeypatch):
    c, planner, calls, _ = runtime(monkeypatch)
    rows = record()['rows']
    commands = {'a': .2, 'b': .3}
    assert c.apply(rows, commands, 59) == {}
    first = c.apply(rows, commands, 60)
    assert first['a']['requested_raw'] == pytest.approx(-.3)
    assert first['b']['requested_raw'] == pytest.approx(-.2)
    sample = c.events[-1]['sample']
    assert sample['own_action'] == [[0.], [0.]]
    assert sample['action'] == [[-.5], [-.5]]
    assert c.events[-1]['requests'] is first
    assert c.apply(rows, commands, 61) == first
    assert c.apply(rows, commands, 62) == {}
    assert not planner.active
    c.apply(rows, commands, 119)
    c.apply(rows, commands, 120)
    assert [x[0]['time_s'] for x in calls] == [60, 120]
    assert all(x[2:] == (8, 600) for x in calls)


def test_registered_custom_cadence_replans_long_moves_without_shortening_forecast(monkeypatch):
    c, _, calls, _ = runtime(monkeypatch, config=po.Config(cadence_s=10), moves=GroupMoves(hold=25))
    for now in (0, 10, 20, 25, 30):
        c.apply(record()['rows'], record()['commands'], now)
    assert [x[0]['time_s'] for x in calls] == [0, 10, 20, 30]
    assert all(x[3] == 600 for x in calls)


def test_next_cadence_failed_gate_cancels_ongoing_move_immediately(monkeypatch):
    c, planner, calls, _ = runtime(monkeypatch, moves=GroupMoves(hold=300))
    assert c.apply(record()['rows'], record()['commands'], 60)
    assert c.apply(record()['rows'], record()['commands'], 119)
    install_calibration(monkeypatch, .79)
    assert c.apply(record()['rows'], record()['commands'], 120) == {}
    assert len(calls) == 2 and not planner.active
    assert c.events[-1]['sample']['planner_online']['decisions'][0]['reason'] == 'base_policy'


def test_best_move_residual_clamp_and_return_to_current_policy(monkeypatch):
    c, _, _, _ = runtime(monkeypatch, depth=.5)
    with torch.no_grad():
        c.policy.actor[-1].bias.fill_(.9)
    rows, commands = record()['rows'], record()['commands']
    request = c.apply(rows, commands, 60)
    assert request['b']['requested_raw'] == 1.2
    sample = c.events[-1]['sample']
    assert sample['own_action'][0][0] == pytest.approx(.9)
    assert sample['action'][0][0] == pytest.approx(1.1)  # 1.2 - base command .1
    released = c.apply(rows, commands, 62)
    assert released['b']['requested_raw'] == pytest.approx(1.1)


@pytest.mark.parametrize('failure', [None, RuntimeError('WM offline')])
def test_failed_confidence_or_wm_retains_exact_base_policy(monkeypatch, failure):
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    c, _, _, _ = runtime(monkeypatch, probability=.79, failure=failure)
    reference = controller_class(Controller)(copy.deepcopy(c.policy), torch.Generator().manual_seed(3), True)
    rows, commands = record()['rows'], record()['commands']
    before = c.generator.get_state().clone()
    assert c.apply(rows, commands, 60) == reference.apply(rows, commands, 60)
    assert torch.equal(before, c.generator.get_state()) is False  # base sample owns its noise draw
    assert torch.equal(c.generator.get_state(), reference.generator.get_state())
    for key in ('action', 'own_action', 'logp', 'value'):
        assert c.events[-1]['sample'][key] == reference.events[-1]['sample'][key]
    reason = c.events[-1]['sample']['planner_online']['decisions'][0]['reason']
    assert reason == ('planner_unavailable' if failure else 'base_policy')


def test_runtime_default_off_is_byte_identical_and_does_not_load_planner_dependencies(monkeypatch):
    import json
    from .acceleration import Controller
    from .v2v_runtime import Policy, controller_class
    cls = controller_class(Controller)
    policy = Policy()
    a = cls(policy, torch.Generator().manual_seed(5))
    b = cls(copy.deepcopy(policy), torch.Generator().manual_seed(5), planner_online=None)
    for name in ('moves', 'planner_calibration'):
        monkeypatch.setitem(sys.modules, po.__package__+'.'+name, None)
    for now in (61, 62):
        assert a.apply(record()['rows'], record()['commands'], now) == b.apply(record()['rows'], record()['commands'], now)
    assert json.dumps(a.events, sort_keys=True) == json.dumps(b.events, sort_keys=True)
    assert not hasattr(a, 'online_planner') and not hasattr(b, 'online_planner')
    assert 'planner_online' not in a.events[-1]['sample']
    assert torch.equal(a.generator.get_state(), b.generator.get_state())


def test_default_off_event_bytes_and_rng_match_track_base_source():
    import json
    from pathlib import Path
    import subprocess
    from .acceleration import Controller
    from .v2v_runtime import Policy, controller_class
    source = subprocess.run(['git', 'show', '13f184b5:tools/coopmwm/v2v_runtime.py'],
                            cwd=Path(__file__).resolve().parents[2], check=True,
                            capture_output=True, text=True).stdout
    namespace = {'__name__': po.__package__+'._track_base_v2v_runtime', '__package__': po.__package__}
    exec(compile(source, '<track-base-v2v-runtime>', 'exec'), namespace)
    for deterministic in (False, True):
        policy = Policy()
        current = controller_class(Controller)(policy, torch.Generator().manual_seed(5), deterministic)
        baseline = namespace['controller_class'](Controller)(copy.deepcopy(policy),
                                                             torch.Generator().manual_seed(5), deterministic)
        for now in (61, 62):
            assert current.apply(record()['rows'], record()['commands'], now) == baseline.apply(
                record()['rows'], record()['commands'], now)
        assert json.dumps(current.events, sort_keys=True) == json.dumps(baseline.events, sort_keys=True)
        assert torch.equal(current.generator.get_state(), baseline.generator.get_state())


@pytest.mark.parametrize('conflict', ['guide', 'replay', 'branch'])
def test_conflicting_request_producers_refused(conflict):
    from .acceleration import Controller
    from .v2v_runtime import Policy, controller_class
    with pytest.raises(ValueError, match='sole runtime request ownership'):
        controller_class(Controller)(Policy(), torch.Generator(), planner_online=object(), **{conflict: {}})


def test_each_disjoint_group_scored_and_previous_commitments_shared_in_both_wm_arms(monkeypatch):
    c, planner, calls, _ = runtime(monkeypatch)
    rows = [row(v, i*10.) for i, v in enumerate('abcdef')]+[row('h', 15., False)]
    desired = c.apply(rows, {r['vid']: .2 for r in rows}, 60)
    assert len(calls) == 2 and len(planner.active) == 2
    assert calls[0][0]['group'] == list('abcd') and calls[1][0]['group'] == list('ef')
    assert set(desired) == set('abcdef')  # HDV excluded even when commands contains it
    background = calls[1][0]['committed_offsets'][0]
    assert background == {v: -.5 for v in 'abcd'}
    baseline, move = calls[1][1]
    assert baseline['offset_profile'][0] == background
    assert move['offset_profile'][0] == {v: -.5 for v in 'abcdef'}
    assert planner.active[1]['offset_profile'][0] == {v: -.5 for v in 'ef'}


def test_only_ego_local_rows_commands_and_history_reach_move_generator_or_wm(monkeypatch):
    c, _, calls, moves = runtime(monkeypatch)
    rows = record()['rows']+[row('z', 1000.), row('far_h', 1010., False)]
    commands = {'a': .1, 'b': .2, 'z': .3}
    c.apply(rows, commands, 59)
    c.apply(rows, commands, 60)
    assert len(calls) == len(moves.seen) == 2
    near, far = [x[0] for x in calls]
    assert {r['vid'] for r in near['rows']} == {'a', 'b', 'h'}
    assert set(near['commands']) == {'a', 'b'}
    assert {r['vid'] for r in far['rows']} == {'z', 'far_h'}
    assert far['group'] == ['z']
    for rec in (near, far):
        assert [h['time_s'] for h in rec['history']] == [59., 60.]
        assert all(h['rows'] == rec['rows'] for h in rec['history'])
        po.validate_observation(rec)


def test_lost_radio_contact_or_departure_releases_whole_move(monkeypatch):
    for departed in (False, True):
        c, planner, _, _ = runtime(monkeypatch, moves=GroupMoves(hold=120))
        c.apply(record()['rows'], record()['commands'], 60)
        rows = [row(), row('h', 10., False)]
        if not departed:
            rows.append(row('b', 101.))
        assert c.apply(rows, record()['commands'], 61) == {}
        assert not planner.active


def test_delayed_onset_keeps_commitment_but_base_drives_until_onset(monkeypatch):
    c, planner, _, _ = runtime(monkeypatch, moves=GroupMoves(onset=3, hold=2))
    assert c.apply(record()['rows'], record()['commands'], 60) == {}
    assert planner.active
    assert c.apply(record()['rows'], record()['commands'], 63)['a']['requested_raw'] == -.4
    assert c.apply(record()['rows'], record()['commands'], 65) == {}


def test_repeated_tick_refused_before_redispatch(monkeypatch):
    c, _, calls, _ = runtime(monkeypatch)
    c.apply(record()['rows'], record()['commands'], 60)
    with pytest.raises(ValueError, match='increasing whole-second clock'):
        c.apply(record()['rows'], record()['commands'], 60)
    assert len(calls) == 1


def test_real_command_ownership_is_the_only_writer_and_keeps_safe_speedmode(monkeypatch):
    from traffic_shepherd.ownership import CommandOwnership
    c, _, _, _ = runtime(monkeypatch, depth=.5)
    desired = c.apply(record()['rows'], record()['commands'], 60)
    writes = []
    api = SimpleNamespace(vehicle=SimpleNamespace(getSpeedMode=lambda v: 27,
                          slowDown=lambda *args: writes.append(args)))
    cfg = SimpleNamespace(control_dt_s=1., acceleration_min_mps2=-3.,
                          acceleration_max_mps2=1.2, sensor_range_m=100.)
    owner = CommandOwnership(api, ['a', 'b'], SimpleNamespace(config=cfg), cav_speed_mode=27)
    owner.wall_accel_max_mps2 = 1.2
    frames = {v: SimpleNamespace(vehicle_id=v, timestamp_s=60., speed_mps=15.,
                                allowed_speed_mps=31.29, front_coverage_m=100., rear_coverage_m=100.,
                                front=None, rear=None) for v in ('a', 'b')}
    assert writes == []  # planner/controller have no SUMO writer
    issued = owner.step(frames, now_s=60., live_ids=['a', 'b', 'h'],
                        wall_commands={v: d['requested_raw'] for v, d in desired.items()})
    assert sorted(x[0] for x in writes) == ['a', 'b']
    assert len(issued) == 2 and all(x.issued_acceleration_mps2 <= 1.2 for x in issued)
