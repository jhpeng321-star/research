from copy import deepcopy
import types

import pytest

from tools.coopmwm import rerun_cpus as rc, rerun_tracks_backend as prod
from tools.coopmwm.test_t2c_runtime_cpu import install_moves, Move


def exercise(monkeypatch, selected, reverse=False):
    moves = install_moves(monkeypatch)
    moves.candidates = lambda obs, rng, n: [Move('none', (obs['own_sensing']['ego_id'],))]
    seen = []
    class Features:
        def encode(self, obs):
            seen.append(deepcopy(obs))
            return dict(missing_share=0.)
    class Trigger:
        def choose(self, obs, features, proposals, rng):
            return proposals[0]
    class Layer:
        def residual(self, obs, features):
            return {v: .3 for v in obs['cav_ids']}
    class Controller:
        previous_frame = None
        def __init__(self): self.events = []
        def apply(self, rows, commands, now):
            self.events.append(dict(sample={}))
            return {}
    sensor = [dict(vid=v, is_cav=True, s_m=s, v_mps=10., lane='m1', length_m=5.)
              for v, s in [('near', 0.), ('far1', 10000.), ('far2', 10020.)]]
    if reverse: sensor.reverse()
    c = Controller()
    comp = dict(merge_features=Features(), trigger_head=Trigger(), coop_layer=Layer())
    hook = prod.EpisodeHook(dict(trigger_head=dict(decision_interval_s=60),
        dense_teacher=dict(candidates=8)), 'B', None, comp, selected=selected)
    hook(c, dict(id='test', accel_bounds=[-3., 1.2]), {})
    result = c.apply(sensor, {r['vid']: .2 for r in sensor}, 300)
    return hook, seen, result


@pytest.mark.parametrize('selected', [None, ['far1', 'far2'], ['far1'], ['far2'], []])
def test_disconnected_actors_and_isolated_all_solo_masks(monkeypatch, selected):
    hook, seen, commands = exercise(monkeypatch, selected)
    expected = {'near', 'far1', 'far2'} if selected is None else set(selected)
    assert {o['own_sensing']['ego_id'] for o in seen} == expected
    assert set(commands) == expected
    assert all(v['requested_raw'] == pytest.approx(.5) for v in commands.values())  # never double-added
    for obs in seen:
        assert ('near' in obs['cav_ids']) == (obs['own_sensing']['ego_id'] == 'near')
    assert set(hook.actors) == expected


def test_input_row_order_does_not_change_contexts_decisions_or_commands(monkeypatch):
    a, seen_a, commands_a = exercise(monkeypatch, ['far1', 'far2'])
    b, seen_b, commands_b = exercise(monkeypatch, ['far1', 'far2'], reverse=True)
    assert seen_a == seen_b and commands_a == commands_b
    assert {v: (s['move'], s['onset']) for v, s in a.actors.items()} == {
        v: (s['move'], s['onset']) for v, s in b.actors.items()}


def turnover_frame(now, kind):
    sensor = [dict(vid=v, is_cav=True, s_m=s, v_mps=10., lane='m1', length_m=5.)
              for v, s in [('c1', 0.), ('c2', 20.), ('c3', 40.)]]
    sensor.append(dict(vid='h1', is_cav=False, s_m=10., v_mps=10., lane='m1', length_m=5.))
    if now in (301, 302):
        if kind == 'range':
            sensor[1]['s_m'] = 10000.  # still on-road/commandable, outside c1/c3 sensing
        elif kind == 'exit':
            sensor = [r for r in sensor if r['vid'] != 'c2']
    commands = {r['vid']: .2 for r in sensor if r['is_cav']}
    if now in (301, 302) and kind == 'uncommandable':
        commands.pop('c2')
    return sensor, commands


class TurnoverReward:
    def realised_rewards(self, before, after, *, elapsed_s):
        assert elapsed_s == 1
        # The sibling adapter owes rewards for the *prior* local neighbourhood,
        # including exits. A constant observable per-step reward isolates routing.
        return {r['vid']: 1. for r in before['own_sensing']['vehicles']}


def turnover_hook(monkeypatch, selected=None, residual=.0):
    moves = install_moves(monkeypatch)
    moves.candidates = lambda obs, rng, n: [Move('none', (obs['own_sensing']['ego_id'],)),
        Move('gap_open', tuple(obs['cav_ids']), .2)]
    choices = []
    class Features:
        def encode(self, obs):
            return dict(values=[2.], missing_share=float(not obs['v2v_reports']))
    class Trigger:
        def choose(self, obs, features, proposals, rng):
            choices.append((obs['time_s'], obs['own_sensing']['ego_id']))
            return proposals[-1]
    class Layer:
        def residual(self, obs, features):
            return {v: residual for v in obs['cav_ids']} if residual else {}
    class Controller:
        previous_frame = None
        def __init__(self): self.events = []
        def apply(self, rows, commands, now):
            self.events.append(dict(sample={}))
            return {v: dict(requested_raw=a, acceleration_residual=0.) for v, a in commands.items()}
    cfg = dict(trigger_head=dict(decision_interval_s=60), dense_teacher=dict(candidates=8),
               track2_reward=dict(weight=1., provenance='test-only registered equal weighting'))
    comp = dict(merge_features=Features(), trigger_head=Trigger(), coop_layer=Layer(),
                track2_reward=TurnoverReward())
    hook = prod.EpisodeHook(cfg, 'B', None, comp, selected=selected)
    controller = Controller()
    hook(controller, dict(id='turnover', accel_bounds=[-3., 1.2]), {})
    return hook, controller, choices, moves


@pytest.mark.parametrize('kind', ['range', 'exit', 'uncommandable'])
@pytest.mark.parametrize('selected', [None, prod.AllMembers(), ['c1', 'c2', 'c3'], ['c1'], ['c2'], []])
def test_partner_turnover_completes_episode_with_cadence_ownership_and_receipts(monkeypatch, kind, selected):
    hook, controller, choices, _ = turnover_hook(monkeypatch, selected)
    allowed = {'c1', 'c2', 'c3'} if selected is None or isinstance(selected, prod.AllMembers) else set(selected)
    steps = 0
    for now in range(300, 1200):
        sensor, commands = turnover_frame(now, kind)
        desired = controller.apply(sensor, commands, now)
        acting = set(commands) & allowed
        assert set(desired) == set(commands)  # no departed, HDV or absent actuator write
        assert set(hook.actors) == set(hook.pending) == acting
        steps += len(acting)
        if now in (300, 301, 302, 303, 359, 360):
            for vid in commands:
                active_move = now in (300, 360) or (kind != 'range' and vid == 'c2' and now >= 303)
                expected = .4 if vid in allowed and active_move else .2
                assert desired[vid]['requested_raw'] == pytest.approx(expected)  # one owner, no double add
        if now == 301 and selected is None:
            for ego in acting:
                row = hook.rows[hook.reward_rows[ego]]
                assert row['move_cancelled_at_s'] == 301
                assert row['move_unavailable_members'] == (['c1', 'c3'] if ego == 'c2' else ['c2'])
        hook.after_step(now, turnover_frame(now+1, kind)[0])
        controller.previous_frame = dict(time_s=now, vehicles=sensor)
    expected_choices = [(t, v) for t in range(300, 1200, 60) for v in sorted(allowed)]
    if kind != 'range' and 'c2' in allowed:
        expected_choices.append((303, 'c2'))  # returning actor gets a new context
    assert sorted(choices) == sorted(expected_choices)  # cancellation/visibility return is not a new decision
    assert len(controller.events) == 900
    if selected is None:
        assert len(hook.rows) == len(choices)
        assert len({r['row_id'] for r in hook.rows}) == len(choices)
        assert sum(r['reward_steps'] for r in hook.rows) == steps
        assert all(r['reward_steps'] > 0 for r in hook.rows)
        # c2 has no observable neighbours on the two out-of-range ticks.
        assert sum(r['realised_team_reward'] for r in hook.rows) == pytest.approx(2.*steps-(2 if kind == 'range' else 0))
        assert all(r['family'] == r['move'].family == 'gap_open' for r in hook.rows)
    else:
        assert not hook.rows and not hook.reward_rows  # evaluation has no learning rows


def test_cancelled_move_keeps_valid_residual_and_does_not_resurrect(monkeypatch):
    hook, controller, choices, _ = turnover_hook(monkeypatch, residual=.1)
    for now, expected in [(300, .5), (301, .3), (303, .3), (360, .5)]:
        sensor, commands = turnover_frame(now, 'range')
        desired = controller.apply(sensor, commands, now)
        assert all(r['requested_raw'] == pytest.approx(expected) for r in desired.values())
    assert choices == [(t, v) for t in (300, 360) for v in ('c1', 'c2', 'c3')]


@pytest.mark.parametrize('invalid', ['remote', 'h1', 'c2'])
def test_fresh_nonobservable_or_uncommandable_proposal_still_refused(monkeypatch, invalid):
    hook, controller, _, moves = turnover_hook(monkeypatch)
    moves.candidates = lambda obs, rng, n: [Move('none', ('c1',)), Move('gap_open', ('c1', invalid), .2)]
    sensor, commands = turnover_frame(301, 'uncommandable')
    with pytest.raises(ValueError, match='trigger selected nonobservable CAV'):
        controller.apply(sensor, commands, 301)


@pytest.mark.parametrize('invalid', ['remote', 'h1', 'c2'])
def test_cancelled_move_does_not_mask_invalid_residual(monkeypatch, invalid):
    hook, controller, _, _ = turnover_hook(monkeypatch)
    sensor, commands = turnover_frame(300, 'exit')
    controller.apply(sensor, commands, 300)
    hook.components['coop_layer'].residual = lambda obs, features: {invalid: .1}
    sensor, commands = turnover_frame(301, 'uncommandable')
    with pytest.raises(ValueError, match='HDV/nonobservable production command'):
        controller.apply(sensor, commands, 301)
