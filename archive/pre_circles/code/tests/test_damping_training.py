from copy import deepcopy

import numpy as np
import pytest
import torch

from env.damping_rollout import CRITIC_DIM, RequestLedger
from idm.damping_actor import SharedDampingActor
from idm.damping_credit import TeamCostScales
from idm.damping_ppo import (
    PPOContract,
    TeamCritic,
    centered_reward_series,
    prepare_batch,
    update_ppo,
)
from idm.damping_scenario import TripRequest, build_scenario, request_schedule


def test_explicit_population_is_fixed_before_insertion(tmp_path):
    first = request_schedule(4107, 2000)
    assert first == request_schedule(4107, 2000)
    assert first != request_schedule(4109, 2000)
    assert all(0 <= r.depart_s < 1800 for r in first)
    assert len({r.vehicle_id for r in first}) == len(first)
    scene, requests, manifest = build_scenario(tmp_path, 4107, 2000)
    assert requests == first
    assert build_scenario(tmp_path, 4107, 2000)[2] == manifest
    assert 'speedDev="0"' in (scene.parent / "requests.rou.xml").read_text()
    with pytest.raises(ValueError, match="existing scenario"):
        build_scenario(tmp_path, 4107, 2200)


@pytest.mark.parametrize("seed", [4201, 4601, 5001, 9001])
def test_reserved_or_unregistered_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="exposed"):
        request_schedule(seed, 2000)


def test_request_ledger_counts_blocked_source_and_late_arrival():
    requests = [
        TripRequest("a", 0.25, "r", "normal_car", True),
        TripRequest("b", 0.75, "r", "normal_car", False),
    ]
    ledger = RequestLedger(requests)
    assert ledger.advance(0.5, ["a"], []) == 0.25
    assert ledger.advance(1, [], []) == 0.75
    assert ledger.advance(1.5, [], ["a"]) == 1
    assert ledger.advance(2, ["b"], []) == 0.5
    assert ledger.system_vehicle_seconds == 2.5
    assert ledger.source_vehicle_seconds == 1.5
    assert ledger.due(2) - len(ledger.arrived) == 1
    with pytest.raises(ValueError, match="duplicate"):
        ledger.advance(2.5, ["b"], [])


def sample_rollout(actor, length=6):
    torch.manual_seed(18)
    # Every actor has left before the last two reward intervals.
    times = np.array([0, 1, 1, 2, 3], dtype=np.int64)
    obs = torch.randn(len(times), actor.contract.observation_dim)
    with torch.no_grad():
        actions, logp = actor.sample(obs)
    intervals = [
        dict(
            braking_vehicle_seconds=0.0,
            acceleration_energy=0.0,
            requested_not_arrived_vehicle_seconds=0.0,
            added_collision_events=0,
            added_teleport_events=0,
        )
        for _ in range(length)
    ]
    return dict(
        summary=dict(
            scenario_identity="fixture",
            seed=4107,
            population=2,
            global_control_steps=length,
            team_terminal=True,
            arrived=2,
            system_seconds_per_requested_vehicle=0.0,
        ),
        intervals=intervals,
        old_values=np.zeros(length + 1, dtype=np.float32),
        actor_observations=obs.numpy(),
        actions=actions.numpy(),
        old_logp=logp.numpy(),
        actor_times=times,
        actor_ids=["a", "a", "b", "b", "b"],
        critic_observations=np.zeros((length, CRITIC_DIM), dtype=np.float32),
    )


def test_real_batch_keeps_post_departure_credit_and_ppo_updates():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    actor, critic = SharedDampingActor(), TeamCritic(CRITIC_DIM)
    rollout = sample_rollout(actor)
    native = deepcopy(rollout)
    rollout["intervals"][-1]["acceleration_energy"] = 2.0
    batch = prepare_batch(rollout, native, TeamCostScales(1, 1, 2))
    np.testing.assert_allclose(batch["team_advantages"], -1)
    np.testing.assert_allclose(batch["actor_advantages"], -1)
    metrics = update_ppo(
        actor,
        critic,
        torch.optim.Adam(actor.parameters(), lr=1e-4),
        torch.optim.Adam(critic.parameters(), lr=1e-4),
        batch,
        rng=np.random.default_rng(91),
        contract=PPOContract(time_minibatch=2),
    )
    assert metrics["actor_parameter_change_l2"] > 0
    assert metrics["critic_grad_norm"] > 0
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics["global_times"] == 6
    assert metrics["actor_samples"] == 5


def test_reward_audit_and_pair_identity():
    rollout = sample_rollout(SharedDampingActor())
    native = deepcopy(rollout)
    rollout["intervals"][-1]["requested_not_arrived_vehicle_seconds"] = 6
    rollout["summary"]["system_seconds_per_requested_vehicle"] = 3
    rewards, audit = centered_reward_series(rollout, native, TeamCostScales(1, 1, 2))
    assert rewards.sum() == pytest.approx(-2.3)
    assert audit["service_violation"]
    native["summary"]["scenario_identity"] = "different"
    with pytest.raises(ValueError, match="scenario_identity"):
        prepare_batch(rollout, native, TeamCostScales(1, 1, 2))
    rollout["summary"]["team_terminal"] = False
    with pytest.raises(ValueError, match="complete fixed-clock"):
        prepare_batch(rollout, rollout, TeamCostScales(1, 1, 2))


def test_terminal_service_penalty_reaches_actor_with_variable_earlier_costs():
    actor = SharedDampingActor()
    rollout = sample_rollout(actor)
    native = deepcopy(rollout)
    rollout["intervals"][2]["acceleration_energy"] = 1.0
    rollout["intervals"][4]["requested_not_arrived_vehicle_seconds"] = 6.0
    rollout["summary"]["system_seconds_per_requested_vehicle"] = 3.0
    strict = prepare_batch(
        rollout, native, TeamCostScales(1, 1, 2), PPOContract(service_budget_s_per_vehicle=2.0)
    )
    permissive = prepare_batch(
        rollout, native, TeamCostScales(1, 1, 2), PPOContract(service_budget_s_per_vehicle=100.0)
    )
    # Holding physical outcomes fixed isolates the terminal constraint signal.
    # With undiscounted, uncentered advantages it must lower EVERY actor credit.
    np.testing.assert_allclose(strict["actor_advantages"] - permissive["actor_advantages"], -2.0)


def test_repair_reference_reuse_cannot_expand_original_budget(tmp_path):
    import json
    from experiments.idm.train_native_damping import import_native_references

    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            dict(
                status="stopped_service_penalty_erased_by_episode_centering",
                remaining_original_19200_step_budget=7200,
            )
        )
    )
    with pytest.raises(ValueError, match="original allocated training budget"):
        import_native_references(tmp_path, tmp_path, {}, 5)


def test_run_identity_survives_json_roundtrip():
    import json
    from experiments.idm.train_native_damping import run_identity

    identity = run_identity(__file__)
    assert identity == json.loads(json.dumps(identity))


def test_collected_rollout_and_optimizer_checkpoint_roundtrip(tmp_path):
    from experiments.idm.train_native_damping import (
        exclusive_run,
        load_rollout,
        save_checkpoint,
        save_rollout,
    )

    actor, critic = SharedDampingActor(), TeamCritic(CRITIC_DIM)
    raw = sample_rollout(actor)
    path = tmp_path / "rollout.json"
    save_rollout(path, raw)
    restored = load_rollout(path)
    for key in ("actor_observations", "actions", "old_logp", "actor_times", "old_values"):
        np.testing.assert_array_equal(raw[key], restored[key])
    assert restored["intervals"] == raw["intervals"]
    opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-4)
    rng = np.random.default_rng(71)
    checkpoint = tmp_path / "trainer.pt"
    save_checkpoint(
        checkpoint,
        actor,
        critic,
        opt,
        copt,
        rng,
        identity={"test": "frozen"},
        iteration=0,
        history=[],
    )
    saved = torch.load(checkpoint, weights_only=True)
    recovered_rng = np.random.default_rng()
    recovered_rng.bit_generator.state = saved["numpy_rng"]
    np.testing.assert_array_equal(rng.permutation(200), recovered_rng.permutation(200))
    recovered_actor = SharedDampingActor()
    recovered_actor.load_state_dict(saved["actor"])
    probe = torch.from_numpy(raw["actor_observations"])
    assert torch.equal(actor(probe), recovered_actor(probe))
    with exclusive_run(tmp_path):
        with pytest.raises(OSError):
            with exclusive_run(tmp_path):
                pass


def test_development_report_does_not_hide_service_or_safety_regression():
    from experiments.idm.evaluate_native_damping_check import compare_summaries

    native = dict(
        scenario_identity="same",
        seed=4107,
        population=100,
        global_control_steps=2400,
        team_terminal=True,
        arrived=100,
        system_seconds_per_requested_vehicle=50.0,
        source_wait_vehicle_seconds=0.0,
        ttc_below_3_vehicle_seconds=10.0,
        totals=dict(
            braking_vehicle_seconds=100.0,
            acceleration_energy=200.0,
            added_collision_events=0,
            added_teleport_events=0,
        ),
    )
    policy = deepcopy(native)
    policy["totals"]["braking_vehicle_seconds"] = 50.0
    policy["system_seconds_per_requested_vehicle"] = 53.0
    policy["ttc_below_3_vehicle_seconds"] = 15.0
    policy["arrived"] = 98
    outcome = compare_summaries(policy, native)
    assert outcome["braking_vehicle_seconds_improvement_percent"] == 50
    assert not outcome["within_existing_service_margins"]
    assert outcome["ttc_below_3_delta_vehicle_seconds"] == 5
    assert outcome["arrived_delta"] == -2
    policy["team_terminal"] = native["team_terminal"] = False
    with pytest.raises(ValueError, match="full episodes"):
        compare_summaries(policy, native)
