"""Sanity tests for the multi-agent Gymnasium API."""

import gymnasium as gym

from env.core import TrafficShepherdMultiAgentEnv


def test_env_api():
    env = TrafficShepherdMultiAgentEnv(
        {
            "sumo_binary": "sumo",
            "sumo_config": "osm.light.sumocfg",
            "cav_penetration": 0.20,
            "max_episode_steps": 30,
            "simulation_step": 1.0,
        }
    )
    assert env.observation_space is not None
    assert env.action_space is not None
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 5

    obs, info = env.reset(seed=123)
    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    for cav_id, o in obs.items():
        assert env.observation_space.contains(o), f"obs for {cav_id} is not in observation_space"

    # Step once with integer mode actions (0..4) for active CAVs.
    if obs:
        action_dict = {cav_id: 0 for cav_id in obs.keys()}
    else:
        action_dict = {}

    obs2, rewards, terminated, truncated, info2 = env.step(action_dict)
    assert isinstance(obs2, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminated, dict)
    assert isinstance(truncated, dict)
    assert isinstance(info2, dict)
    assert "__all__" in terminated
    assert "__all__" in truncated

    env.close()
    print("test_env_api passed")


if __name__ == "__main__":
    test_env_api()
