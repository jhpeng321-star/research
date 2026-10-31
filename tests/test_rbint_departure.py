import copy
from types import SimpleNamespace

import pytest
import torch

from tools.rbint.departure import rollout_after_movement
from tests.unit.test_rb_merge_departure_phase import setup


def test_multistep_no_arrivals_matches_full_rollout_and_absolute_controller_clock():
    state, model, process = setup([10., 100.], [2, 1], [5., 20.], [True, True])
    process._emit_due = lambda b: []
    ticks = []
    def callback(current):
        ticks.append(current['physics_step_index'])
        return model.hold_teacher(current)
    with torch.no_grad():
        full = model.rollout(copy.deepcopy(state), n_steps=4, k0=12)
        phased = rollout_after_movement(model, state, n_steps=4, entries=process,
                                        cav_controller=callback, k0=12)
    assert ticks == [12, 13, 14, 15]
    for name in ('positions', 'speeds', 'lanes', 'masks', 'accel', 'history', 'history_mask'):
        torch.testing.assert_close(phased[name], full[name], rtol=0, atol=0)


def test_cost_observer_retains_pre_exit_speed_and_newborn_cohort_without_birth_motion():
    state, model, process = setup([553., 0.], [1, 0], [20., 0.], [True, False])
    process._emit_due = lambda b: [0]
    class Entries:
        def __init__(self):
            self.process = process
            self.observer = SimpleNamespace(states=[])
        def __getattr__(self, key):
            return getattr(self.process, key)
        def insert(self, current):
            born = self.process.insert(current)
            self.observer.states.append({k: v.clone() for k, v in current.items() if torch.is_tensor(v)})
            return born
    entries = Entries()
    with torch.no_grad():
        out = rollout_after_movement(model, state, n_steps=2, entries=entries)
    before = entries.observer.states[0]
    assert before['present'].tolist() == [[True, True]]
    assert before['v_mps'][0, 0] == 20.
    assert not out['masks'][0, 0, 0]  # original exit is not replaced by newborn
    assert out['masks'][0, 0, 1]
    assert out['accel'][0, 0, 1] == 0
    assert out['positions'][0, 1, 1] > out['positions'][0, 0, 1]


def test_future_teacher_inputs_fail_before_source_advances():
    state, model, process = setup([10.], [1], [5.], [True])
    with torch.no_grad(), pytest.raises(ValueError, match='unsupported'):
        rollout_after_movement(model, state, n_steps=2, entries=process, present_seq=torch.ones(1, 3, 1))
    assert process.now[0] == 0
