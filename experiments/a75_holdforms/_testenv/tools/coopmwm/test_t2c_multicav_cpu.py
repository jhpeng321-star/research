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
