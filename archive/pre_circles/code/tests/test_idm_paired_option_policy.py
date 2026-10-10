import numpy as np
import torch

from idm.paired_option_dataset import PHYSICAL_OBSERVATION_DIM, PairedOptionRecord
from idm.paired_option_policy import (
    BinaryOptionActorCritic,
    evaluate_binary_policy,
    normalization,
    select_deployment_threshold,
)


def _row(sample, signal, reward, constraints=True, penetration=0.1):
    observation = np.zeros(PHYSICAL_OBSERVATION_DIM, dtype=np.float32)
    observation[0] = signal
    return PairedOptionRecord(
        sample_id=sample,
        source="fixture",
        demand=2000,
        seed=1,
        penetration=penetration,
        decision_index=1,
        decision_step=1,
        vehicle_id=sample,
        observation=tuple(observation),
        activate_reward=reward if constraints else -2.0,
        physical_utility=reward,
        constraints_pass=constraints,
        classification="helpful" if reward > 0.02 else "harmful",
        dose_steps=20,
    )


def test_actor_starts_exactly_half_and_consumes_every_field():
    rows = [_row("a", -1.0, -0.2), _row("b", 1.0, 0.3)]
    mean, scale = normalization(rows)
    model = BinaryOptionActorCritic(mean, scale, hidden=16)
    observations = np.asarray([row.observation for row in rows], dtype=np.float32)
    assert np.allclose(model.activation_probability(observations), 0.5)
    logits, values = model(torch.tensor(observations, requires_grad=True))
    (logits[:, 1].sum() + values.sum()).backward()
    assert model.encoder[0].weight.grad is not None
    assert model.encoder[0].weight.grad.shape[1] == PHYSICAL_OBSERVATION_DIM


def test_policy_evaluation_counts_constraint_failure_as_harm():
    rows = [_row("good", 1.0, 0.4), _row("unsafe", -1.0, 0.3, constraints=False)]
    mean, scale = normalization(rows)
    model = BinaryOptionActorCritic(mean, scale, hidden=8)
    result = evaluate_binary_policy(model, rows, threshold=0.5)
    assert result["activation_count"] == 2
    assert result["helpful_activation_count"] == 1
    assert result["harmful_activation_count"] == 1
    assert result["constraint_failure_activation_count"] == 1


def test_threshold_selection_never_hides_gate_fallback():
    rows = [_row(str(index), float(index), -0.2) for index in range(4)]
    mean, scale = normalization(rows)
    model = BinaryOptionActorCritic(mean, scale, hidden=8)
    threshold, diagnostic = select_deployment_threshold(model, rows)
    assert 0.0 <= threshold <= 1.0
    assert diagnostic["frozen_gate_feasible"] is False
