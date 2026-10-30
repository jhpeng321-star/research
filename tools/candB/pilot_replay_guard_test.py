"""Real multi-future segmented backward regression for the observed closure failure."""
import copy
import os
import unittest
from unittest.mock import patch

import torch

from tests.unit.test_longmem import trainer, observe, assert_tree_close
from tools.candB.pilot_replay_guard import vjp_guard


def core_factory(policy, group, safety, lane_record=None, **kwargs):
    def callback(state):
        obs = observe(state)
        action = policy(obs, group(obs, state))[0]
        accel = safety(state, action)
        speed = (state['v_mps'] + .5*accel).clamp_min(0.)
        return torch.stack((state['s_m']+.5*speed, state['lane'], speed, accel), -1)
    return callback


def measured_factory():
    state = dict(calls=0)
    def factory(*args, **kwargs):
        callback = core_factory(*args, **kwargs)
        def measured(frame):
            state['calls'] += 1
            return callback(frame)
        return measured
    return factory, lambda: state


class ReplayGuardTests(unittest.TestCase):
    def test_observed_failure_reproduced_without_guard(self):
        # accel-noise futures: the synthetic world model has no encode_drivers.
        t = trainer(1, 2, True, horizon=3, batch=2, future_samples=2,
            future_driver_scale=0., future_accel_noise=True)
        t.controller_factory, state = measured_factory()
        with patch.dict(os.environ, WMS_SPEED_FUTURE_WORKERS='1', WMS_SPEED_PARALLEL_FORWARD='0'):
            with self.assertRaises(TypeError):
                t.update()
        self.assertIsNone(state())

    def test_two_real_updates_preserve_gradients_adam_rng_and_telemetry_lifetime(self):
        options = dict(future_samples=2, future_driver_scale=0., future_accel_noise=True)
        expected = trainer(1, 2, True, horizon=3, batch=2, **options)
        actual = trainer(1, 2, True, horizon=3, batch=2, **options)
        expected.controller_factory = core_factory
        wrapped, state = measured_factory()
        actual.controller_factory = wrapped
        with patch.dict(os.environ, WMS_SPEED_FUTURE_WORKERS='1', WMS_SPEED_PARALLEL_FORWARD='0'):
            for seed in (31, 32):
                torch.manual_seed(seed)
                left = expected.update()
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                with vjp_guard(core_factory):
                    right = actual.update()
                self.assertIs(actual.controller_factory, wrapped)
                self.assertGreater(state()['calls'], 0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key in ('actor_loss','actor_grad_norm','critic_loss','critic_grad_norm'):
                    assert_tree_close(left[key], right[key], atol=0, rtol=0)
                for a, b in zip(expected.actor_parameters, actual.actor_parameters):
                    assert_tree_close(a, b, atol=0, rtol=0)
                    assert_tree_close(a.grad, b.grad, atol=0, rtol=0)
                assert_tree_close(expected.actor_optimizer.state_dict(),
                    actual.actor_optimizer.state_dict(), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
