from dataclasses import replace

import pytest
import torch
from torch import nn

from env.native_damping import DampingContract, LocalFrame, OwnIDMParameters, project_residual
from experiments.idm.probe_damping_magnitude_ablation import EPSILON, MagnitudeAblationActor


class FixedActor(nn.Module):
    contract = DampingContract()

    def forward(self, observations):
        return observations[:, :1]


def test_ablation_retains_sign_and_native_zero_decisions_without_amplification():
    values = torch.tensor([[-0.005], [-1e-7], [-1e-9], [0.0], [1e-9], [1e-7], [0.007]])
    actor = MagnitudeAblationActor(FixedActor())
    result = actor(values)
    assert torch.equal(result.sign(), values.sign())
    assert torch.equal(result.abs() <= actor.contract.zero_tolerance, values.abs() <= actor.contract.zero_tolerance)
    assert (result.abs() <= values.abs()).all()
    assert result.abs().max().item() == pytest.approx(EPSILON)
    assert actor.log_prob(values, result).shape == (7,)


def test_ablation_rejects_epsilon_that_would_be_native_passthrough():
    base = FixedActor()
    base.contract = replace(base.contract, zero_tolerance=EPSILON)
    with pytest.raises(ValueError, match="above native-zero"):
        MagnitudeAblationActor(base)


def test_tiny_signed_residual_still_engages_nominal_path_in_feasible_state():
    frame = LocalFrame("cav", 0, 20, 0, 30, 1, front_coverage_m=100, rear_coverage_m=100)
    own = OwnIDMParameters(0.2, 4.5, 1.5, 2.5, 9)
    zero = project_residual(frame, own, 0, now_s=0)
    assert zero.target_acceleration is None
    for residual in (-EPSILON, EPSILON):
        small = project_residual(frame, own, residual, now_s=0)
        assert small.target_acceleration is not None
        assert small.target_acceleration > 0.1
        assert small.target_acceleration-small.nominal_acceleration == pytest.approx(residual)
