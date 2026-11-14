"""Unit test: RLlib agent ids must not revive after done."""

from __future__ import annotations


import numpy as np

from env.rllib_wrapper import TrafficShepherdRLlibEnv


class _FakeSumoEnv:
    observation_space = type("S", (), {"shape": (4,)})()
    action_space = None

    def __init__(self, _config=None):
        self._t = 0
        self._script = [
            # t0 reset
            ({"veh_a": np.zeros(4, np.float32)}, {}),
            # t1: a leaves
            (
                {"veh_a": np.zeros(4, np.float32)},
                {"veh_a": 0.0},
                {"veh_a": True, "__all__": False},
                {"veh_a": False, "__all__": False},
                {},
            ),
            # t2: only veh_b (new)
            (
                {"veh_b": np.ones(4, np.float32)},
                {"veh_b": 1.0},
                {"veh_b": False, "__all__": False},
                {"veh_b": False, "__all__": False},
                {},
            ),
            # t3: veh_a "revives" in SUMO — must get a NEW rllib id
            (
                {"veh_a": np.ones(4, np.float32), "veh_b": np.ones(4, np.float32)},
                {"veh_a": 0.5, "veh_b": 0.5},
                {"veh_a": False, "veh_b": False, "__all__": False},
                {"veh_a": False, "veh_b": False, "__all__": False},
                {},
            ),
        ]

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._script[0]

    def step(self, action_dict):
        self._t += 1
        return self._script[self._t]

    def close(self):
        pass


def test_no_agent_id_reuse_after_done(monkeypatch=None):
    import env.rllib_wrapper as mod

    original = mod.TrafficShepherdMultiAgentEnv
    mod.TrafficShepherdMultiAgentEnv = _FakeSumoEnv
    try:
        env = TrafficShepherdRLlibEnv({})
        obs, _ = env.reset()
        assert list(obs.keys()) == ["veh_a__1"]

        obs, rew, term, trunc, _ = env.step({"veh_a__1": 0})
        assert term["veh_a__1"] is True
        assert "veh_a__1" not in env.agents

        obs, rew, term, trunc, _ = env.step({})
        assert "veh_b__2" in obs
        assert "veh_a__1" not in obs

        obs, rew, term, trunc, _ = env.step({aid: 0 for aid in obs})
        # revived SUMO veh_a must not reuse veh_a__1
        assert "veh_a__1" not in obs
        revived = [k for k in obs if k.startswith("veh_a__")]
        assert revived and revived[0] != "veh_a__1"
        assert "veh_b__2" in obs
        print("OK:", sorted(obs.keys()))
    finally:
        mod.TrafficShepherdMultiAgentEnv = original


if __name__ == "__main__":
    test_no_agent_id_reuse_after_done()
