"""Integration test for the policy evaluation runner."""

from pathlib import Path

from evaluate_policy import run_policy_evaluation
from policies import ConstantSpeedPolicy, NoOpPolicy, RandomPolicy

SUMO_ROOT = Path(__file__).parent.parent.parent / "research" / "highway_merge_shepherd"


def test_policy_runner():
    policy = NoOpPolicy()
    metrics = run_policy_evaluation(
        policy=policy,
        density="light",
        project_root=SUMO_ROOT,
        max_episode_steps=30,
        cav_penetration=0.1,
        policy_name="no_op",
    )

    assert metrics["policy"] == "NoOpPolicy"
    assert metrics["density"] == "light"
    assert metrics["steps"] > 0
    assert "travel_time" in metrics
    assert "ttc" in metrics
    assert Path(metrics["tripinfo_path"]).exists()
    assert Path(metrics["fcd_path"]).exists()
    print("test_policy_runner passed")


def test_policy_runner_random_baseline():
    # RandomPolicy must run in shepherd mode (Discrete(5)) — under the old
    # continuous default its reset() crashed on action_space.n (Box).
    metrics = run_policy_evaluation(
        policy=RandomPolicy(seed=42),
        density="light",
        project_root=SUMO_ROOT,
        max_episode_steps=30,
        cav_penetration=0.1,
        policy_name="random",
    )
    assert metrics["policy"] == "RandomPolicy"
    assert metrics["steps"] > 0
    assert "travel_time" in metrics
    print("test_policy_runner_random_baseline passed")


def test_policy_runner_constant_speed_baseline():
    # ConstantSpeedPolicy (non_intervention under shepherd mode) must run a
    # short episode without its output being misread as an acceleration.
    metrics = run_policy_evaluation(
        policy=ConstantSpeedPolicy(),
        density="light",
        project_root=SUMO_ROOT,
        max_episode_steps=30,
        cav_penetration=0.1,
        policy_name="constant_speed",
    )
    assert metrics["policy"] == "ConstantSpeedPolicy"
    assert metrics["steps"] > 0
    assert "travel_time" in metrics
    print("test_policy_runner_constant_speed_baseline passed")


if __name__ == "__main__":
    test_policy_runner()
