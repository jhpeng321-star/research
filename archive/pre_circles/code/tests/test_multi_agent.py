"""Test variable active-agent count and dict outputs."""

from env.core import TrafficShepherdMultiAgentEnv


def test_variable_agent_count():
    env = TrafficShepherdMultiAgentEnv(
        {
            "sumo_binary": "sumo",
            "sumo_config": "osm.dense.sumocfg",
            "cav_penetration": 0.50,
            "max_episode_steps": 100,
            "simulation_step": 1.0,
        }
    )

    obs, _ = env.reset(seed=3)
    max_active = len(obs)
    total_steps = 0

    while total_steps < 100:
        action_dict = {cav_id: 0 for cav_id in obs.keys()}
        obs, rewards, terminated, truncated, _ = env.step(action_dict)
        total_steps += 1
        max_active = max(max_active, len(obs))
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            break

    print(f"test_variable_agent_count: max active CAVs = {max_active}")
    assert max_active > 0, "expected at least one active CAV during the episode"
    env.close()


if __name__ == "__main__":
    test_variable_agent_count()
