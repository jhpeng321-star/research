"""S-POLSPEED: WMS_SPEED_GRAPH_CVAR must be bit-identical to the eager CVaR bisection."""
import time

import pytest
import torch

from wms.control import reward


def _inputs(device, k, generator, weighted):
    values = torch.rand(1, k, generator=generator).to(device) * 30.0
    present = (torch.rand(1, k, generator=generator) > 0.3).to(device)
    if weighted:
        weight = torch.where(present, torch.rand(1, k, generator=generator).to(device) * 3,
                             torch.zeros(1, k, device=device))
        present = present & (weight > 0)
    else:
        weight = present.to(values.dtype)
    values = torch.where(present, values, torch.full_like(values, float('nan')))
    count = weight.sum(-1).clamp_min(1)
    return values, present, weight, count


def test_flag_off_and_cpu_use_eager(monkeypatch):
    monkeypatch.setenv('WMS_SPEED_GRAPH_CVAR', '1')
    g = torch.Generator().manual_seed(0)
    args = _inputs('cpu', 12, g, False)
    with torch.no_grad():
        assert torch.equal(reward._cvar_bisection(*args), reward._cvar_bisection_eager(*args))
    assert not reward._CVAR_GRAPHS


def test_soft_cvar_unchanged_with_grad_cpu(monkeypatch):
    g = torch.Generator().manual_seed(1)
    delay = (torch.rand(3, 9, generator=g) * 20).requires_grad_()
    present = torch.rand(3, 9, generator=g) > 0.2
    base = reward._soft_cvar(delay, present)
    (gb,) = torch.autograd.grad(base.sum(), delay)
    monkeypatch.setenv('WMS_SPEED_GRAPH_CVAR', '1')
    fast = reward._soft_cvar(delay, present)
    (gf,) = torch.autograd.grad(fast.sum(), delay)
    assert torch.equal(base, fast) and torch.equal(gb, gf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph path')
@pytest.mark.parametrize('k,weighted', [(64, False), (65, True), (7, False)])
def test_graph_bit_identical_cuda(monkeypatch, k, weighted):
    monkeypatch.setenv('WMS_SPEED_GRAPH_CVAR', '1')
    g = torch.Generator().manual_seed(k)
    for _ in range(20):          # replays with fresh inputs reuse the same captured graph
        args = _inputs('cuda', k, g, weighted)
        with torch.no_grad():
            fast = reward._cvar_bisection(*args)
            eager = reward._cvar_bisection_eager(*args)
        assert torch.equal(fast, eager)
    # the full reward with gradients: identical value and delay gradient
    delay = (torch.rand(1, k, generator=g) * 20).cuda().requires_grad_()
    present = (torch.rand(1, k, generator=g) > 0.2).cuda()
    fast = reward._soft_cvar(delay, present)
    (gf,) = torch.autograd.grad(fast.sum(), delay)
    monkeypatch.setenv('WMS_SPEED_GRAPH_CVAR', '0')
    base = reward._soft_cvar(delay, present)
    (gb,) = torch.autograd.grad(base.sum(), delay)
    assert torch.equal(base, fast) and torch.equal(gb, gf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA timing')
def test_graph_timing_report(monkeypatch, capsys):
    g = torch.Generator().manual_seed(5)
    args = _inputs('cuda', 64, g, False)
    out = {}
    for flag in ('0', '1'):
        monkeypatch.setenv('WMS_SPEED_GRAPH_CVAR', flag)
        with torch.no_grad():
            reward._cvar_bisection(*args)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                reward._cvar_bisection(*args)
            torch.cuda.synchronize()
        out[flag] = (time.perf_counter() - t0) / 200
    with capsys.disabled():
        print(f"\nCVaR bisection per call: eager {out['0']*1e3:.3f} ms, graph {out['1']*1e3:.3f} ms")
