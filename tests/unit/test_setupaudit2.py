"""CPU mechanism checks for crossed lanes and independent physical-J accounting."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.unit.test_loop14c import ScriptedLanes, group, identity_safety, ramp_scene
from wms.control.bridge import make_cav_controller
from wms.control.lanes import LaneRules, LANE_LEFT
from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY


def test_native_lane_intervention_delegates_ramp_without_changing_absolute_command():
    st = ramp_scene()
    st[WM_LANE_TARGET_KEY] = st['lane'].clone()
    seen = []
    cb = make_cav_controller(ScriptedLanes([LANE_LEFT], accel=.3), group, identity_safety,
        lane_rules=LaneRules(), native=lambda s: torch.zeros_like(s['s_m']),
        native_lane_change=True, native_ramp_merge='wm_law', policy_ramp_merge=True,
        policy_native_lanes=True, record=seen)
    out = cb(st)
    assert out[0, 0, 1] == 2  # WM first passage, not the MAPPO head, owns crossing.
    assert st[WM_MERGE_KEY][0, 0]
    assert cb.native_lane_mask(st)[0, 0]
    assert float(out[0, 0, 3]) == pytest.approx(.3)
    assert not seen[0]['lane_changed'].any()


def test_native_lane_intervention_uses_current_lane_for_longitudinal_safety():
    st = ramp_scene()
    st['lane'][:] = 1
    st[WM_LANE_TARGET_KEY] = torch.zeros_like(st['lane'])
    seen = []

    def safety(state, accel, **kw):
        seen.append(state['lane'].clone())
        return accel

    cb = make_cav_controller(ScriptedLanes([LANE_LEFT], accel=.3), group, safety,
        lane_rules=LaneRules(), native=lambda s: torch.zeros_like(s['s_m']),
        native_lane_change=True, native_ramp_merge='wm_law', policy_native_lanes=True)
    out = cb(st)
    assert out[0, 0, 1] == 0
    assert all(torch.equal(x, st['lane']) for x in seen)


def test_lane_intervention_default_off_is_bitwise_and_rng_identical():
    st = ramp_scene()
    st[WM_LANE_TARGET_KEY] = st['lane'].clone()
    results, rngs = [], []
    for extra in ({}, {'policy_native_lanes': False}):
        torch.manual_seed(21)
        cb = make_cav_controller(ScriptedLanes([LANE_LEFT], accel=.3), group, identity_safety,
            lane_rules=LaneRules(), native_lane_change=True, native_ramp_merge='wm_law',
            native=lambda s: torch.zeros_like(s['s_m']), policy_ramp_merge=True, **extra)
        results.append(cb(copy.deepcopy(st)))
        rngs.append(torch.get_rng_state())
    assert torch.equal(*results) and torch.equal(*rngs)
    with pytest.raises(ValueError, match='boolean'):
        make_cav_controller(None, None, None, policy_native_lanes=1)
    with pytest.raises(ValueError, match='requires native_lane_change'):
        make_cav_controller(None, None, None, policy_native_lanes=True)


def panel_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/setupaudit'))
    import panel2
    return panel2


def test_independent_cost_conserves_road_partitions_and_matches_authoritative_reward():
    from types import SimpleNamespace
    from wms.control.contract import RewardWeights
    from wms.control.objective14 import authoritative_reward
    mod = panel_module()
    st = ramp_scene()
    st['is_cav'] = torch.tensor([[True, False, True]])
    states = [{k: v.expand(5, *v.shape[1:]).clone() if torch.is_tensor(v) else v
               for k, v in st.items()} for _ in range(180)]
    speeds = torch.full((5, 180, 3), 8., dtype=torch.float64)
    entries = SimpleNamespace(states=states, queues=[np.full(5, 4.) for _ in states],
                              pending=[np.full((5, 2), 2.) for _ in states],
                              new_masks=[torch.zeros(5, 3, dtype=torch.bool) for _ in states],
                              real_source=[np.full(5, 2.) for _ in states])
    scored = mod.costs({'speeds': speeds, 'masks': torch.ones_like(speeds).bool()}, entries, 2)['90']
    before = {**states[0], 'entry_queue': torch.full((5,), 4.)}
    after = {**states[1], 'v_mps': speeds[:, 0]}
    actual = authoritative_reward(before, after, torch.zeros(5, 3), RewardWeights())['J'] * 180
    torch.testing.assert_close(torch.tensor(scored['J']), actual.float())
    for a, b in [('CAV', 'HDV'), ('initial', 'entrant')]:
        np.testing.assert_allclose(np.array(scored[a])+scored[b], scored['road'])
    np.testing.assert_allclose(np.array(scored['ramp'])+scored['mainline']+scored['merge_zone'],
                               scored['road'])


def test_frozen_admission_uses_ctrl0_births_without_resetting_actor_road_state():
    mod = panel_module()

    class Process:
        pending = np.zeros((5, 2))

        def insert(self, state):
            assert torch.equal(state['s_m'][0], state['s_m'][4])
            mask = torch.zeros_like(state['present'])
            mask[:, 1] = True
            state['s_m'][:, 1] = -100.
            state['present'][:, 1] = True
            return mask

        def backlog(self):
            return np.zeros(5)

    state = dict(s_m=torch.arange(10.).reshape(5, 2),
                 present=torch.zeros(5, 2, dtype=torch.bool))
    old = state['s_m'][:, 0].clone()
    entries = mod.Entries(Process(), frozen=True)
    entries.insert(state)
    assert torch.equal(state['s_m'][:, 0], old)
    assert (state['s_m'][:, 1] == -100.).all()


def test_batched_native_reference_keeps_wm_cap_instead_of_actor_injected_cap(monkeypatch):
    from types import SimpleNamespace
    from tools.setupaudit.measure import native_background_motion
    from wms.model.cav import EXEC_CAP_KEY
    mod = panel_module()
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    st = ramp_scene()
    st = {k: v.expand(6, *v.shape[1:]).clone() if torch.is_tensor(v) else v for k, v in st.items()}
    st[WM_LANE_TARGET_KEY] = st['lane'].clone()
    st[EXEC_CAP_KEY] = torch.full_like(st['s_m'], -.7)
    cp = ScriptedLanes(accel=0.)
    cp.residual_on_native = True
    mp = SimpleNamespace(policy=ScriptedLanes(accel=.3), group_encoder=group, lane_rules=LaneRules())
    def native(s):
        return torch.full_like(s['s_m'], .35)
    with torch.no_grad():
        actual = mod.controller((cp, group, None), mp, native)(st)
        expected = native_background_motion(st, actual[..., 1], native)
    assert torch.equal(actual[4], expected[4])


def test_real_metric_birth_exclusion_and_exit_last_speed_are_recomputed():
    from types import SimpleNamespace
    mod = panel_module()
    st = ramp_scene()
    st = {k: v.expand(5, *v.shape[1:]).clone() if torch.is_tensor(v) else v for k, v in st.items()}
    states = [copy.deepcopy(st) for _ in range(180)]
    speeds = torch.full((5, 180, 3), 8., dtype=torch.float64)
    masks = torch.ones_like(speeds).bool()
    masks[:, 0, 0] = False  # exited: real tracker uses input speed (5), not output speed (8).
    births = [torch.zeros(5, 3, dtype=torch.bool) for _ in states]
    births[0][:, 2] = True  # first road tick is excluded by the real tracker.
    source = [np.zeros(5) for _ in states]
    source[0][:] = .5
    entries = SimpleNamespace(states=states, queues=[np.zeros(5) for _ in states],
                              pending=[np.zeros((5, 2)) for _ in states],
                              new_masks=births, real_source=source)
    c = mod.costs({'speeds': speeds, 'masks': masks}, entries, 2)['90']
    expected_delta = .5 - .5*(1-8/31.29) + .5*(8-5)/31.29
    np.testing.assert_allclose(np.array(c['real_J'])-c['J'], expected_delta, atol=1e-8)


def test_exact_due_time_observer_preserves_headway_draw_count():
    mod = panel_module()

    class Renewal:
        def __init__(self):
            self.pending = np.zeros((5, 2))
            self.now = np.zeros(5)
            self.t_next = np.full((5, 2), .25)
            self._started = np.ones(5, dtype=bool)
            self.draws = 0

        def _headway(self, b, e):
            self.draws += 1
            return 1.

        def insert(self, state):
            for b in range(5):
                self.now[b] = .5
                self.t_next[b, 0] += self._headway(b, 0)
            self.pending[:, 0] = 1
            return torch.zeros_like(state['present'])

        def backlog(self):
            return self.pending.sum(-1)

    process = Renewal()
    entries = mod.Entries(process)
    entries.insert(dict(present=torch.zeros(5, 2, dtype=torch.bool)))
    assert process.draws == 5
    np.testing.assert_array_equal(entries.real_source[0], np.full(5, .25))
    np.testing.assert_array_equal(.5*entries.queues[0], np.full(5, .5))
