"""Integrator gates must reject plausible-looking but invalid smoke outputs."""
import pytest
import torch
from torch import nn

from tools.rbint.audit import audit_closed_cohort, forbid_learned_corrections


def fixture():
    return dict(masks=torch.tensor([[[True, True], [True, False]]]),
                positions=torch.tensor([[[10., 90.], [20., 101.]]]),
                speeds=torch.full((1, 2, 2), 10.), accel=torch.zeros(1, 2, 2),
                lanes=torch.zeros(1, 2, 2, dtype=torch.long),
                raw_residual_mps2=torch.zeros(1, 2, 2))


def test_accounts_for_boundary_exit():
    receipt = audit_closed_cohort(torch.ones(1, 2, dtype=torch.bool), fixture(), exit_s_m=100.)
    assert receipt["initial"] == [2] and receipt["exited"] == [1] and receipt["final"] == [1]


@pytest.mark.parametrize("defect", ["early_exit", "resurrection", "nan", "negative_speed",
                                    "invalid_lane", "residual", "no_boundary"])
def test_rejects_corrupt_rollouts(defect):
    result = fixture()
    boundary = 100.
    if defect == "early_exit":
        result["positions"][0, 1, 1] = 99.
    elif defect == "resurrection":
        result["masks"][0, 0, 0] = False
        result["positions"][0, 0, 0] = 101.
    elif defect == "nan":
        result["speeds"][0, 1, 1] = float("nan")
    elif defect == "negative_speed":
        result["speeds"][0, 0, 0] = -1.
    elif defect == "invalid_lane":
        result["lanes"][0, 0, 0] = 3
    elif defect == "residual":
        result["raw_residual_mps2"][0, 0, 0] = 0.01
    else:
        boundary = None
    with pytest.raises(AssertionError):
        audit_closed_cohort(torch.ones(1, 2, dtype=torch.bool), result, exit_s_m=boundary)


def test_guard_allows_encoder_rejects_learned_lane_and_removes_hooks():
    model = nn.Module()
    model.driver_encoder = nn.Linear(2, 2)
    model.lane_v5 = nn.Linear(2, 2)
    value = torch.ones(1, 2)
    with forbid_learned_corrections(model) as calls:
        model.driver_encoder(value)
        assert calls == ["driver_encoder"]
        with pytest.raises(AssertionError, match="lane_v5"):
            model.lane_v5(value)
    assert model.lane_v5(value).shape == value.shape
