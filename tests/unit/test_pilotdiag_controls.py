"""S-PILOTDIAG controls: per-step cotangent normalisation + acceptance rollback."""
import math

import pytest
import torch

from wms.control.dreamer import DreamerConfig
from wms.control.longgrad import StateTrace


def test_step_cotangent_normalize_rescales_adjoint():
    x = torch.randn(4, 3, requires_grad=True)
    weights = torch.tensor([1., 2., 3.])
    trace = StateTrace(normalize=True)
    trace.watch({"s_m": x}, 0)
    (x * weights).sum().backward()
    # whole-tensor L2 norm: the (4, 3) adjoint tiles weights across 4 rows
    expected = weights / (2.0 * weights.norm())
    assert torch.allclose(x.grad, expected.expand(4, 3), atol=1e-6)
    # telemetry keeps the RAW pre-normalisation adjoint (per-row squared norms)
    row_sq = (weights ** 2).sum().double().expand(4)
    assert torch.allclose(trace.values[(0, "s_m")], row_sq, atol=1e-6)


def test_step_cotangent_off_is_bitwise_passive():
    a = torch.randn(5, 2, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    StateTrace().watch({"v_mps": a}, 3)
    (a * 2).sum().backward()
    (b * 2).sum().backward()
    assert torch.equal(a.grad, b.grad)


def test_normalize_zero_and_nonfinite_adjoints_unchanged():
    x = torch.zeros(1, 3, requires_grad=True)
    StateTrace(normalize=True).watch({"a_mps2": x}, 1)
    (x * 0. + 5.).sum().backward()
    assert torch.equal(x.grad, torch.zeros(1, 3))
    y = torch.tensor([[1., 1.]], requires_grad=True)
    nan = torch.tensor([[1., float("nan")]])
    StateTrace(normalize=True).watch({"a_mps2": y}, 0)
    (y * nan).sum().backward()
    assert torch.isnan(y.grad[0, 1]) and y.grad[0, 0] == 1.


def test_pilotdiag_config_keys_default_off_and_typed():
    cfg = DreamerConfig()
    assert cfg.imagination_step_cotangent_normalize is False
    assert cfg.update_acceptance_panel is False
    with pytest.raises(ValueError):
        DreamerConfig(imagination_step_cotangent_normalize=1)
    with pytest.raises(ValueError):
        DreamerConfig(update_acceptance_panel="yes")
    assert DreamerConfig(imagination_step_cotangent_normalize=True).imagination_step_cotangent_normalize


def test_snapshot_restore_actor_roundtrip():
    from wms.control.policy14_cli import snapshot_actor, restore_actor

    class Trainer:
        pass

    torch.manual_seed(0)
    module = torch.nn.Linear(3, 2)
    trainer = Trainer()
    trainer.actor_parameters = tuple(module.parameters())
    trainer.actor_optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    batch = torch.randn(4, 3)
    module(batch).sum().backward()
    trainer.actor_optimizer.step()
    snapshot = snapshot_actor(trainer)
    before = [p.detach().clone() for p in module.parameters()]
    adam_before = {k: {kk: vv.clone() for kk, vv in v.items() if torch.is_tensor(vv)}
                   for k, v in trainer.actor_optimizer.state_dict()["state"].items()}
    (module(batch) * 2).sum().backward()
    trainer.actor_optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(module.parameters(), before))
    restore_actor(trainer, snapshot)
    assert all(torch.equal(a, b) for a, b in zip(module.parameters(), before))
    state = trainer.actor_optimizer.state_dict()["state"]
    for key, moments in adam_before.items():
        for name, value in moments.items():
            assert torch.equal(state[key][name], value)


def test_panel_improved_monotone_no_threshold():
    from wms.control.policy14_cli import panel_improved

    assert panel_improved([10., 12.], [9., 11.])
    assert panel_improved([10., 10.], [10., 10.])
    assert not panel_improved([9., 9.], [9., 10.])
    # one start worse, mean still better -> accepted (no per-start threshold)
    assert panel_improved([10., 10.], [8., 11.])


def test_normalize_bounds_multistep_adjoint_growth():
    # Two chained boundaries: the upstream normalised adjoint flows onward, and
    # the leaf boundary re-normalises what it receives (not the pre-chain value).
    v = torch.tensor([[3., 4.]], requires_grad=True)
    trace = StateTrace(normalize=True)
    trace.watch({"v_mps": v}, 0)
    middle = v * torch.tensor([1., 2.])
    trace.watch({"s_m": middle}, 1)
    (middle * torch.tensor([5., 5.])).sum().backward()
    # middle receives [5,5] -> unit; v receives [5,5]/sqrt(50) * [1,2] -> re-normalised
    assert torch.allclose(v.grad[0], torch.tensor([1., 2.]) / math.sqrt(5.), atol=1e-6)
    raw_at_v = torch.tensor([1., 2.]) / math.sqrt(2.)
    assert math.isclose(float(trace.values[(0, "v_mps")].sum().sqrt()),
                        float(raw_at_v.norm()), rel_tol=1e-6)
