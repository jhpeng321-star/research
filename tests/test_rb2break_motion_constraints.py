import pytest
import torch

from wms.physics.motion_constraints import constrain_acceleration


def test_shrinkage_cannot_resurrect_a_physically_impossible_acceleration():
    weights = torch.tensor([.4, .6], dtype=torch.float64)
    support = torch.tensor([[30., 1., .8, 2., 2.], [31., 1.2, 1.2, 2.2, 2.5]], dtype=torch.float64)
    result, receipt = constrain_acceleration(weights, support, torch.tensor([0.]),
                                             torch.tensor([.5]), torch.tensor([.5]))
    torch.testing.assert_close(result, torch.tensor([0., 1.], dtype=torch.float64))
    assert receipt['discarded_weight'] == .4
    assert torch.equal(weights, torch.tensor([.4, .6], dtype=torch.float64))


def test_no_evidence_is_exact_population_and_unsupported_driver_fails_closed():
    weights = torch.tensor([1.], dtype=torch.float64)
    support = torch.tensor([[30., 1., .8, 2., 2.]])
    empty = torch.empty(0)
    result, _ = constrain_acceleration(weights, support, empty, empty, empty)
    assert result is weights
    with pytest.raises(ValueError, match='no positive-weight'):
        constrain_acceleration(weights, support, torch.tensor([0.]), torch.tensor([1.]), torch.tensor([.5]))
