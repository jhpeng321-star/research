from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch

from env.damping_v2 import FilteredHistory, SelectableActuator, expert_from_observation
from env.native_damping_sumo import SumoLocalSensor
from idm.damping_v2_actor import SelectableActor
from idm.damping_v2_reward import reward_series, prepare_batch
from idm.damping_ppo import TeamCritic, PPOContract, update_ppo
from tests.test_native_damping_sumo import api


def observation():
    x = np.zeros(89, dtype=np.float32)
    x[68:85] = [0.5, 0, 1, 0, 0, 0, 0, 0.5, 0.05, 1, 1, 0.5, 0, 1, 1, 0, 1]
    x[85:] = [0.5, 1, 0, 0]
    return x


def trajectory(t=10):
    interval = dict(
        braking_vehicle_seconds=61.3 / t,
        acceleration_energy=4120.4235963623185 / t,
        requested_not_arrived_vehicle_seconds=100 / t,
        added_collision_events=0,
        added_teleport_events=0,
    )
    return dict(
        summary=dict(
            scenario_identity="same",
            seed=4107,
            population=10,
            global_control_steps=t,
            team_terminal=True,
            system_seconds_per_requested_vehicle=10,
            source_wait_vehicle_seconds=0,
            ttc_below_3_vehicle_seconds=0,
            arrived=10,
            unfinished_requested=0,
            totals=dict(added_collision_events=0, added_teleport_events=0),
        ),
        intervals=[dict(interval) for _ in range(t)],
        intervention_fractions=[0] * t,
        source_intervals=[0] * t,
        ttc_intervals=[0] * t,
        old_values=np.zeros(t + 1),
        actor_observations=np.stack([observation(), observation()]),
        actions=np.array([[0, 0], [1, 0]], dtype=np.float32),
        old_logp=np.zeros(2, dtype=np.float32),
        actor_times=np.array([0, 2]),
        actor_ids=["leaves_at_3", "other"],
        critic_observations=np.zeros((t, 22), dtype=np.float32),
    )


def test_reward_ordering_and_episode_bound():
    native = trajectory()
    unnecessary = deepcopy(native)
    unnecessary["intervention_fractions"] = [1] * 10
    helpful = deepcopy(unnecessary)
    harmful = deepcopy(unnecessary)
    for row in helpful["intervals"]:
        row["braking_vehicle_seconds"] *= 0.5
        row["acceleration_energy"] *= 0.5
    for row in harmful["intervals"]:
        row["braking_vehicle_seconds"] *= 1.5
        row["acceleration_energy"] *= 1.5
    returns = [reward_series(r, native)[0].sum() for r in (helpful, native, unnecessary, harmful)]
    assert returns == pytest.approx([0.03, 0, -0.02, -0.07])
    assert all(a > b for a, b in zip(returns, returns[1:]))
    for t in (1, 10, 2400):
        a = trajectory(t)
        a["intervention_fractions"] = [1] * t
        assert reward_series(a, trajectory(t))[1]["intervention_cost"] == pytest.approx(-0.02)


@pytest.mark.parametrize("kind", ["source", "ttc", "service", "completion", "hard_events"])
def test_delayed_constraint_cost_is_not_erased_by_normalization(kind):
    native = trajectory()
    own = deepcopy(native)
    if kind == "source":
        own["summary"]["source_wait_vehicle_seconds"] = 30
        own["source_intervals"][-1] = 30
    elif kind == "ttc":
        own["summary"]["ttc_below_3_vehicle_seconds"] = 6
        own["ttc_intervals"][-1] = 6
    elif kind == "service":
        own["summary"]["system_seconds_per_requested_vehicle"] = 13
        own["intervals"][-1]["requested_not_arrived_vehicle_seconds"] += 30
    elif kind == "completion":
        own["summary"]["arrived"] = 9
        own["summary"]["unfinished_requested"] = 1
    else:
        own["summary"]["totals"]["added_collision_events"] = 1
        own["intervals"][-1]["added_collision_events"] = 1
    before = prepare_batch(native, native)
    after = prepare_batch(own, native)
    assert after["reward_audit"]["violations"][kind]
    assert (after["actor_advantages"] - before["actor_advantages"] <= -2).all()
    assert after["actor_advantages"][0] == pytest.approx(after["rewards"].sum())


def test_preference_remains_in_actor_credit_and_positive_gain_cannot_hide_violation():
    native = trajectory()
    own = deepcopy(native)
    own["intervention_fractions"][-1] = 1
    assert prepare_batch(own, native)["actor_advantages"] == pytest.approx([-0.002, -0.002])
    own["summary"]["source_wait_vehicle_seconds"] = 30
    own["source_intervals"][-1] = 30
    for r in native["intervals"]:
        r["requested_not_arrived_vehicle_seconds"] += 1000
    native["summary"]["system_seconds_per_requested_vehicle"] += 1000
    rewards, audit = reward_series(own, native)
    assert audit["components"]["service"] > 10
    assert rewards.sum() == pytest.approx(-2.002)


def test_real_discrete_native_mass_conditional_gradient_and_mask():
    torch.manual_seed(5)
    actor = SelectableActor()
    x = torch.tensor(np.stack([observation()] * 1000))
    actions, logp = actor.sample(x)
    assert 100 < (actions[:, 0] == 0).sum() < 900
    torch.testing.assert_close(logp, actor.log_prob(x, actions))
    native = torch.zeros((1000, 2))
    actor.log_prob(x, native).mean().backward()
    assert actor.mean_head.weight.grad.abs().sum() == 0
    assert actor.mode_head.weight.grad.abs().sum() > 0
    actor.zero_grad()
    x[:, 88] = 0.5
    actions, logp = actor.sample(x)
    assert actions.abs().sum() == 0 and logp.abs().sum() == 0
    logp.sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in actor.parameters())


def test_explicit_release_reengagement_and_zero_active_are_distinct():
    connection = api()
    actuator = SelectableActuator(connection)
    frame = replace(SumoLocalSensor(connection).capture("cav", now_s=0), front=None, rear=None)
    x = observation()
    for now, mode in ((0, 0), (1, 1), (2, 0), (3, 1), (7, 1), (8, 0)):
        frame = replace(frame, timestamp_s=now)
        rows = actuator.apply(
            {"cav": frame},
            {"cav": (mode, 0)},
            now_s=now,
            live_cav_ids={"cav"},
            observations={"cav": x},
        )
        if now == 3:
            assert rows[0]["reasons"] == ("native_cooldown",)
    assert [r[0] for r in connection.vehicle.calls] == [
        "slowDown",
        "setSpeed",
        "slowDown",
        "setSpeed",
    ]
    assert connection.vehicle.calls[-1] == ("setSpeed", "cav", -1)
    assert not actuator.owned


def test_ema_is_causal_idempotent_and_expert_abstains_in_steady_flow():
    connection = api()
    h = FilteredHistory()
    a = SelectableActuator(connection)
    f = SumoLocalSensor(connection).capture("cav", now_s=0)
    h.add(f, now_s=0)
    first = h.vector("cav", now_s=0, actuator=a)
    h.add(f, now_s=0)
    np.testing.assert_array_equal(first, h.vector("cav", now_s=0, actuator=a))
    with pytest.raises(ValueError):
        h.add(replace(f, timestamp_s=2), now_s=1)
    x = observation()
    x[76] = 0
    assert not expert_from_observation(x)[1]
    h.drop("cav")
    assert not h.filtered


def test_hybrid_ppo_updates_both_heads_with_signed_credit():
    torch.manual_seed(4)
    actor = SelectableActor()
    critic = TeamCritic(22)
    native = trajectory(10)
    own = deepcopy(native)
    x = torch.tensor(own["actor_observations"])
    own["actions"][1, 1] = 0.05  # an exploratory action, not exactly the Gaussian mean
    own["old_logp"] = actor.log_prob(x, torch.tensor(own["actions"])).detach().numpy()
    own["intervals"][-1]["acceleration_energy"] *= 0.1
    batch = prepare_batch(own, native)
    result = update_ppo(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-4),
        torch.optim.Adam(critic.parameters(), lr=1e-4),
        batch,
        rng=np.random.default_rng(4),
        contract=PPOContract(time_minibatch=10),
    )
    assert result["actor_parameter_change_l2"] > 0
    assert actor.mode_head.weight.abs().sum() > 0 and actor.mean_head.weight.abs().sum() > 0
