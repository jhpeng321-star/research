"""Tests for built-in baseline policies."""

import numpy as np

from policies import ConstantSpeedPolicy, NoOpPolicy, RandomPolicy


class _FakeEnv:
    """Minimal env stand-in for baseline reset()."""

    def __init__(self):
        import gymnasium as gym

        self.action_space = gym.spaces.Discrete(5)
        self.max_speed = 35.0


def test_no_op_policy():
    env = _FakeEnv()
    policy = NoOpPolicy()
    policy.reset(env)
    obs_dict = {"veh_0": np.zeros(24, dtype=np.float32)}
    actions = policy.act(obs_dict)
    assert "veh_0" in actions
    assert actions["veh_0"] == 0
    print("test_no_op_policy passed")


def test_random_policy():
    env = _FakeEnv()
    policy = RandomPolicy(seed=42)
    policy.reset(env)
    obs_dict = {"veh_0": np.zeros(24, dtype=np.float32), "veh_1": np.zeros(24, dtype=np.float32)}
    actions = policy.act(obs_dict)
    assert len(actions) == 2
    for mode in actions.values():
        assert 0 <= mode < 5
    print("test_random_policy passed")


def test_constant_speed_policy():
    env = _FakeEnv()
    policy = ConstantSpeedPolicy()
    policy.reset(env)
    obs = np.zeros(24, dtype=np.float32)
    actions = policy.act({"veh_0": obs})
    # Under shepherd mode the policy is equivalent to no_op: non_intervention
    # hands control back to SUMO (see the policy docstring).
    assert actions["veh_0"] == 0
    print("test_constant_speed_policy passed")


if __name__ == "__main__":
    test_no_op_policy()
    test_random_policy()
    test_constant_speed_policy()
