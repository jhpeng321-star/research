import math
from types import SimpleNamespace

import torch

from tests.unit.test_longmem import trainer as make_trainer, assert_tree_close
from wms.control.looplong_real import direct_gradient_norm, secant_loss


def direct_loss(trainer, scale):
    action = torch.stack((trainer.policy.weight, trainer.group_encoder.weight))[None]
    loss, _ = secant_loss(action, torch.zeros_like(action), action.new_tensor([[1., -2.]]),
                          action.new_tensor([20.]), torch.ones_like(action, dtype=torch.bool), scale)
    return loss, {'direct_real_credit_loss': float(loss.detach())}


def test_direct_norm_matches_secant_and_does_not_mutate_grad_or_rng():
    p = torch.tensor([.2, .1], requires_grad=True)
    loss, _ = secant_loss(p[None], torch.zeros(1, 2), torch.tensor([[1., -2.]]),
                          torch.tensor([20.]), torch.ones(1, 2, dtype=torch.bool), 5.)
    p.grad = torch.tensor([7., 8.])
    rng = torch.get_rng_state().clone()
    norm = direct_gradient_norm(loss, [p])
    assert math.isclose(norm, math.sqrt(.8 ** 2 + 1.6 ** 2), rel_tol=1e-6)
    assert torch.equal(p.grad, torch.tensor([7., 8.]))
    assert torch.equal(torch.get_rng_state(), rng)
    loss.backward()  # graph must remain available for the real update
    torch.testing.assert_close(p.grad, torch.tensor([6.2, 9.6]))


def test_every_update_logs_direct_norm_without_changing_optimizer(monkeypatch):
    outputs = []
    for probe in (True, False):
        t = make_trainer(horizon=2, batch=1)
        t.looplong_real_credit = SimpleNamespace(loss=direct_loss)
        if not probe:
            monkeypatch.setattr('wms.control.looplong_real.direct_gradient_norm', lambda *args: 0.)
        rows = [t.update(), t.update()]
        if probe:
            for row in rows:
                expected = math.sqrt(4. ** 2 + 8. ** 2) / row['return_scale']
                assert math.isclose(row['direct_real_credit_grad_norm'], expected, rel_tol=1e-6)
        outputs.append((t.policy.state_dict(), t.group_encoder.state_dict(),
                        t.actor_optimizer.state_dict(), torch.get_rng_state()))
    assert_tree_close(outputs[0], outputs[1], atol=0, rtol=0)
