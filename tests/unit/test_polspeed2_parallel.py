"""Cross-process parity incl. RNG, full-K quantile VJP and optimizer state."""
import copy

import pytest
import torch

from tests.unit.test_longmem import trainer, assert_tree_close


@pytest.mark.parametrize('forward', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_parallel_update_matches_serial_bitwise(tmp_path, monkeypatch, forward, device):
    _check_parallel(tmp_path, monkeypatch, forward, device)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_gated_vjps_preserve_full_update(tmp_path, monkeypatch, device):
    _check_parallel(tmp_path, monkeypatch, True, device, workers=4, slots=2)


def _check_parallel(tmp_path, monkeypatch, forward, device, workers=2, slots=None):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    pytest.importorskip('cloudpickle')
    serial = trainer(1, 2, True, horizon=2, batch=1, counterfactual=True, real_world=True,
                     future_samples=8, pessimistic_credit=True, uncertainty_weight=0.)
    from wms.model import WmSModel, ModelSpec
    serial.world_models = [WmSModel(ModelSpec(rollout_mode='driver_params',
                                             history_hidden=8, k_hidden=8)).eval().requires_grad_(False)]
    if device == 'cuda':
        serial.device = torch.device('cuda')
        for module in [*serial.world_models, serial.policy, serial.group_encoder,
                       serial.critic, serial.target_critic]:
            module.to(device)
        serial.start_states.batch.state = {k: v.to(device) for k, v in serial.start_states.batch.state.items()}
    parallel = copy.deepcopy(serial)
    parallel.output_dir = tmp_path
    torch.manual_seed(561)
    expected = serial.update()
    rng = torch.get_rng_state()
    monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', str(workers))
    if slots is not None:
        monkeypatch.setenv('WMS_SPEED_FUTURE_VJP_SLOTS', str(slots))
    monkeypatch.setenv('WMS_SPEED_PARALLEL_FORWARD', '1' if forward else '0')
    torch.manual_seed(561)
    actual = parallel.update()
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('actor_loss', 'returns', 'actor_grad_norm', 'critic_loss', 'memory_spill'):
        assert_tree_close(expected[key], actual[key], atol=0, rtol=0)
    for x, y in zip(serial.actor_parameters, parallel.actor_parameters):
        torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    assert_tree_close(serial.actor_optimizer.state_dict(), parallel.actor_optimizer.state_dict(),
                      atol=0, rtol=0)
    pools = list((tmp_path / 'future_workers').iterdir())
    assert pools and all((p / 'STOP').exists() for p in pools)


def test_invalid_worker_count_rejected(monkeypatch):
    from wms.control.parallel_futures import worker_count, vjp_slots
    for value in ('0', '9', '-1', '2.5', 'abc'):
        monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', value)
        with pytest.raises(ValueError):
            worker_count()
    monkeypatch.setenv('WMS_SPEED_FUTURE_WORKERS', '4')
    for value in ('0', '5', '2.5', 'abc'):
        monkeypatch.setenv('WMS_SPEED_FUTURE_VJP_SLOTS', value)
        with pytest.raises(ValueError):
            vjp_slots()


def test_orphan_workers_request_owned_stop(tmp_path, monkeypatch):
    from wms.control.parallel_futures import parent_alive
    monkeypatch.setattr('wms.control.parallel_futures._REPARENTING', True)
    monkeypatch.setenv('WMS_SPEED_FUTURE_PARENT_PID', '98765')
    monkeypatch.setattr('os.getppid', lambda: 1)
    assert not parent_alive(tmp_path)
    assert (tmp_path/'STOP').exists()
