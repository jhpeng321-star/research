"""Candidate-B seams: CTRL0 reference/worker binding and residual decode Jacobian."""
from types import SimpleNamespace

import torch

from wms.control import gated300
from wms.control.counterfactual import CounterfactualConfig, build_variants
from wms.control.policy import CavPolicy


def test_ctrl0_hook_install_restore_and_objective_contract():
    trainer = SimpleNamespace(world_model_ids=['wm'],
        config=SimpleNamespace(future_samples=8, entrant_learned_mix=True),
        policy=SimpleNamespace(residual_reference='ctrl0'),
        counterfactual=CounterfactualConfig())
    receipt = gated300.install(trainer, dict(ctrl0_reference=True), ['sha'])
    assert trainer._long_credit_reference is gated300.ctrl0_reference_totals
    contract = gated300.objective_contract(trainer)
    assert contract['reference'] == 'CTRL0' and contract['entrant_learned_mix'] is True
    gated300.checkpoint_restore(trainer, dict(gated300=receipt))
    assert trainer._long_credit_reference is gated300.ctrl0_reference_totals
    gated300.install(trainer, {}, ['sha'])
    assert not hasattr(trainer, '_long_credit_reference')


def test_reference_totals_use_paired_ctrl0_rows():
    state = dict(present=torch.ones(2, 3, dtype=torch.bool),
                 is_cav=torch.ones(2, 3, dtype=torch.bool),
                 s_m=torch.tensor([[0., 10., 20.], [0., 10., 20.]]),
                 lane=torch.zeros(2, 3, dtype=torch.long))
    variants = gated300.add_ctrl0_reference(build_variants(
        state, CounterfactualConfig(enabled=True, max_starts=2)))
    totals = torch.arange(8 * variants.n_rows).reshape(8, 1, -1)
    actual = gated300.ctrl0_reference_totals(None, dict(variants=variants), totals, None)
    assert torch.equal(actual, totals[..., gated300.reference_index(variants)])


def test_caprel_decode_slope_matches_autograd_at_zero_and_saturation():
    policy = CavPolicy(residual_on_native=True, lane_head=False, residual_reference='ctrl0')
    latent = torch.tensor([0., -.2, 1., 8.], requires_grad=True)
    residual = min(-policy.accel_min, policy.accel_max) * latent.tanh()
    expected, = torch.autograd.grad(residual.sum(), latent)
    torch.testing.assert_close(policy.eslong_decode_slope(latent), expected)
    assert expected[0] > 0
