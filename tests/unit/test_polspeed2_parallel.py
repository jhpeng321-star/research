"""Cross-process parity incl. RNG, full-K quantile VJP and optimizer state."""
import copy

import pytest
import torch

from tests.unit.test_longmem import trainer, assert_tree_close


@pytest.mark.parametrize('forward', [False, True])
def test_parallel_update_matches_serial_bitwise(tmp_path, monkeypatch, forward):
    pytest.importorskip('cloudpickle')
    serial = trainer(1, 2, True, horizon=2, batch=1, counterfactual=True, real_world=True,
                     future_samples=8, pessimistic_credit=True, uncertainty_weight=0.)
    from wms.model import WmSModel, ModelSpec
    serial.world_models = [WmSModel(ModelSpec(rollout_mode='driver_params',
                                             history_hidden=8, k_hidden=8)).eval().requires_grad_(False)]
    parallel = copy.deepcopy(serial)
    parallel.output_dir = tmp_path
    torch.manual_seed(561)
    expected = serial.update()
    rng = torch.get_rng_state()
    monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', '2')
    monkeypatch.setenv('WMS_SPEED_PARALLEL_FORWARD', '1' if forward else '0')
    torch.manual_seed(561)
    actual = parallel.update()
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('actor_loss', 'returns', 'actor_grad_norm', 'critic_loss'):
        assert_tree_close(expected[key], actual[key], atol=0, rtol=0)
    for x, y in zip(serial.actor_parameters, parallel.actor_parameters):
        torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert_tree_close(serial.actor_optimizer.state_dict(), parallel.actor_optimizer.state_dict(),
                      atol=0, rtol=0)
    pools = list((tmp_path / 'future_workers').iterdir())
    assert pools and all((p / 'STOP').exists() for p in pools)


def test_invalid_worker_count_rejected(monkeypatch):
    from wms.control.parallel_futures import worker_count
    for value in ('0', '9', '-1', '2.5', 'abc'):
        monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', value)
        with pytest.raises(ValueError):
            worker_count()
