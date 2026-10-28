"""S-POLSPEED: WMS_SPEED_ASYNC_SPILL keeps LONGMEM saved-tensor spill bit-identical."""
from types import SimpleNamespace

import pytest
import torch

from wms.control.imagination_memory import _saved_tensor_spill


def _grads(device, monkeypatch, flag):
    monkeypatch.setenv('WMS_SPEED_ASYNC_SPILL', flag)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 1)).to(device)
    empty = torch.nn.Linear(1, 1).to(device)
    trainer = SimpleNamespace(world_models=[net], policy=empty, group_encoder=empty,
                              critic=empty, device=torch.device(device))
    x = torch.randn(32, 16, device=device)
    with _saved_tensor_spill(trainer):
        h = x
        for _ in range(30):              # many saved tensors, freed/reused blocks
            h = net[0](h).tanh()
            h = net[2](h).tanh()
            h = net[0].weight.new_ones(()) * h[:, :16]
        loss = net[4](net[2](net[0](h).tanh()).tanh()).square().mean()
    grads = torch.autograd.grad(loss, list(net.parameters()))
    return loss.detach(), grads, trainer.memory_spill_stats


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_async_spill_bit_identical(monkeypatch, device):
    loss0, g0, s0 = _grads(device, monkeypatch, '0')
    loss1, g1, s1 = _grads(device, monkeypatch, '1')
    assert torch.equal(loss0, loss1)
    assert all(torch.equal(a, b) for a, b in zip(g0, g1))
    assert s0 == s1 and (device == 'cpu' or s1['copies'] > 50)
