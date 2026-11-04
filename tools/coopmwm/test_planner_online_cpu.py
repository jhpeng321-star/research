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


def test_high_confidence_negative_expected_gain_does_not_fire(monkeypatch):
    result, _, _ = score(monkeypatch, {'policy': [10.]*8,
                                     'move:0': [9.]*7+[100.]}, probability=1.)
    assert result['scores'][0]['p_raw'] == 7/8
    assert not result['accepted']


def test_short_move_still_scored_on_full_requested_horizon(monkeypatch):
    result, _, seen = score(monkeypatch, {'policy': [10.]*8, 'move:0': [9.]*8}, horizon_s=720)
    action = seen[0][1][1]
    assert seen[0][3] == result['horizon_s'] == len(action['offset_profile']) == 720
    assert action['offset_profile'][:3] == [{'a': -.5, 'b': -.5}, {'a': -.5, 'b': -.5}, {}]
    assert not any(action['offset_profile'][2:])


@pytest.mark.parametrize('kwargs', [dict(horizon_s=300), dict(wm_samples=1), dict(cadence_s=0),
                                    dict(n_candidates=-1), dict(p_star=float('nan'))])
def test_invalid_or_shortened_registration_refused(kwargs):
    with pytest.raises(ValueError):
        po.Config(**kwargs)


@pytest.mark.parametrize('probability', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_calibration_refused(monkeypatch, probability):
    with pytest.raises(ValueError, match='calibration API'):
        score(monkeypatch, {'policy': [10.]*8, 'move:0': [9.]*8}, probability=probability)


@pytest.mark.parametrize('costs', [{'policy': [10.]*7, 'move:0': [9.]*8},
                                 {'policy': [10.]*8},
                                 {'policy': [10.]*8, 'move:0': [float('nan')]*8}])
def test_incomplete_or_nonfinite_rollouts_refused(monkeypatch, costs):
    with pytest.raises(ValueError):
        score(monkeypatch, costs)


def test_missing_calibration_fails_before_rollout():
    with pytest.raises(ValueError, match='calibration is required'):
        po.score_moves(record(), None, rollout=lambda *a: pytest.fail('must not roll'),
                       calibration=None, move_api=Moves())


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


@pytest.mark.parametrize('members', [('a', 'h'), ('b',), ('a', 'far')])
def test_move_cannot_command_hdv_unobservable_or_exclude_self(monkeypatch, members):
    with pytest.raises(ValueError, match='move members'):
        score(monkeypatch, {}, moves=Moves([Move(members=members)]))


def test_offset_decoder_adds_to_base_policy_and_clamps_only_through_existing_bounds():
    from .plan_wm import profile_table, profile_apply
    actions = [dict(id='policy'), dict(id='move:0', offset_profile=[{'a': .7, 'b': -.5}])]
    table = profile_table(actions, ['a', 'b', 'h'], 'cpu', horizon=600, dynamic_policy=True)
    base = torch.tensor([[.9, .2, .3], [.9, .2, .3]])
    result = profile_apply(base, 0, table, (-3., 1.2))
    assert torch.equal(result[0], base[0])
    assert torch.allclose(result[1], torch.tensor([1.2, -.3, .3]))
    assert torch.equal(profile_apply(base, 2, table, (-3., 1.2)), base)


def test_wm_adapter_uses_existing_evaluator_and_final_ownership_chain(monkeypatch):
    from . import plan_wm
    calls = []
    monkeypatch.setattr(plan_wm, 'evaluate', lambda *args, **kw: calls.append((args, kw)) or {'ok': True})
    runtime = SimpleNamespace(model=torch.nn.Linear(1, 1), mp=SimpleNamespace(policy=torch.nn.Linear(1, 1)))
    adapter = po.WMRollout(runtime, memory='belief', chunk_actions=2)
    assert adapter(record(), [], 8, 600) == {'ok': True}
    assert calls[0][0][0] is runtime and calls[0][0][3:] == (8, 600)
    assert calls[0][1] == dict(trim=False, chunk_actions=2, memory='belief', final_chain=True)
