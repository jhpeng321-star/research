from copy import deepcopy

import pytest
import torch

from idm import contract as historical
from idm.damping_campaign import (
    ACCEPTANCE_SEEDS,
    CampaignBudget,
    DEVELOPMENT_CASES,
    DEVELOPMENT_SEEDS,
    TRAINING_CASES,
    TRAINING_SEEDS,
    campaign_manifest,
    validate_case,
)
from idm.damping_scenario import request_schedule
from idm.damping_selection import assess_candidate, select_candidate
from policies.velocity_matching_damping import VelocityMatchingDamping


def test_partitions_budget_and_no_acceptance_generation(monkeypatch):
    monkeypatch.setattr("idm.damping_campaign.validate_seed_audit", lambda: "fixture")
    groups = [set(TRAINING_SEEDS), set(DEVELOPMENT_SEEDS), set(ACCEPTANCE_SEEDS)]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert set.union(*groups).issubset(historical.PROSPECTIVE_EVALUATION_EXCLUDED_SEEDS)
    budget = CampaignBudget().manifest()
    assert budget["allocated_policy_seconds"] == 729600
    assert budget["new_training_reference_seconds"] == 38400
    assert budget["development_checkpoint_rollouts"] == 72
    assert campaign_manifest()["checkpoint_optimized_seconds"] == [48000, 148800, 240000]
    with pytest.raises(ValueError, match="sealed"):
        request_schedule(ACCEPTANCE_SEEDS[0], 2000, role="acceptance")
    with pytest.raises(ValueError, match="exposed"):
        request_schedule(TRAINING_SEEDS[0], TRAINING_CASES[0][1])
    with pytest.raises(ValueError, match="registered"):
        validate_case(TRAINING_SEEDS[0], 2200, "training")
    assert request_schedule(*TRAINING_CASES[0], role="training") == request_schedule(
        *TRAINING_CASES[0], role="training"
    )


def test_velocity_matching_uses_local_current_closing_and_respects_masks():
    policy = VelocityMatchingDamping()
    obs = torch.zeros(4, policy.contract.observation_dim)
    obs[:, -1] = 1
    obs[:, -8] = 1  # Front detection, field 9 of the current 17-field frame.
    obs[:, -9] = torch.tensor([1.0, -1.0, 0.01, 0.01])
    obs[2, -8] = 0
    obs[3, -2] = 1  # Stale current measurement.
    result = policy(obs).flatten()
    torch.testing.assert_close(result, torch.tensor([-0.5, 0.3, 0.0, 0.0]))


def rows_with_improvement():
    rows = []
    for seed, loading in DEVELOPMENT_CASES:
        native = dict(
            scenario_identity=str(seed),
            seed=seed,
            population=100,
            global_control_steps=2400,
            team_terminal=True,
            arrived=100,
            uninserted_due=0,
            unfinished_requested=0,
            system_seconds_per_requested_vehicle=50.0,
            source_wait_vehicle_seconds=0.0,
            ttc_below_3_vehicle_seconds=100.0,
            totals=dict(
                braking_vehicle_seconds=100.0,
                acceleration_energy=100.0,
                added_collision_events=0,
                added_teleport_events=0,
            ),
        )
        policy = deepcopy(native)
        policy["totals"]["braking_vehicle_seconds"] = 90.0
        policy["totals"]["acceleration_energy"] = 90.0
        rows.append(dict(case=[seed, loading], policy=policy, native=native))
    return rows


def test_single_harm_cannot_hide_in_good_mean_and_noop_cannot_win():
    rows = rows_with_improvement()
    assert assess_candidate(rows)["eligible"]
    rows[-1]["policy"]["system_seconds_per_requested_vehicle"] = 53.0
    report = assess_candidate(rows)
    assert not report["eligible"]
    assert any(row["constraint"] == "system_time" for row in report["failures"])
    assert select_candidate([dict(assessment=report)]) is None
    for row in rows:
        row["policy"] = deepcopy(row["native"])
    assert not assess_candidate(rows)["eligible"]


def test_selection_rejects_incomplete_or_duplicated_cohorts():
    rows = rows_with_improvement()
    with pytest.raises(ValueError, match="exactly once"):
        assess_candidate(rows[:-1])
    rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="exactly once"):
        assess_candidate(rows)


def test_old_exposed_request_population_is_unchanged():
    import hashlib
    import json
    from dataclasses import asdict

    # Frozen original fixture identity, before registration expanded the API.
    original = [
        (4107, 2000, "55b428e3dad8c610d03386b2e262f81942ea10f5c4d6f5b23a30e34de241228f"),
        (4109, 2000, "0aed210f3e0be3f820f0a078ef3adc2926251573bd510082bf9cdada07eff4bd"),
        (4107, 2200, "2994be3264577bee51d7a6dbd9966308380145f883f295415e5c016d5364798d"),
        (4109, 2200, "61c24c9303b39ce8f4e04147acfc0a9cc0b23234069f511808c0fd4e9dbc25a0"),
    ]
    for seed, loading, expected in original:
        requests = request_schedule(seed, loading)
        actual = hashlib.sha256(
            json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
        ).hexdigest()
        assert actual == expected


def test_optimizer_restore_keeps_action_distribution_and_rng():
    import numpy as np
    from experiments.idm.train_damping_campaign import models_from_state

    actor, critic, aopt, copt, rng = models_from_state(None, 72)
    obs = torch.randn(7, actor.contract.observation_dim)
    state = dict(
        actor=actor.state_dict(),
        critic=critic.state_dict(),
        actor_optimizer=aopt.state_dict(),
        critic_optimizer=copt.state_dict(),
        numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
    )
    expected_actions, expected_logp = actor.sample(obs)
    restored, _, _, _, restored_rng = models_from_state(state, 72)
    actions, logp = restored.sample(obs)
    assert torch.equal(actions, expected_actions)
    assert torch.equal(logp, expected_logp)
    np.testing.assert_array_equal(rng.permutation(50), restored_rng.permutation(50))
