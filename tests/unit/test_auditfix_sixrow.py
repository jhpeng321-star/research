from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tests.unit.test_longmem import trainer as make_trainer
from wms.control.counterfactual import CounterfactualConfig, build_variants
from wms.control.train_cli import _read_config


@pytest.mark.parametrize('pessimistic', [False, True])
def test_preset_loss_and_full_gradient_parity(monkeypatch, pessimistic):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(root))
    registered = CounterfactualConfig(**_read_config(
        root / 'configs/c4_policy14_c4prep_on.toml')['counterfactual'])
    lean = CounterfactualConfig(**_read_config(
        root / 'configs/c4_policy14_c4prep_sixrow.toml')['counterfactual'])
    # Resolved candidate A pins the registered terminal weights from the base.
    registered = replace(registered, curriculum_updates=[0, 1],
                         **{name: [getattr(registered, name)[-1]] * 2 for name in
                            ('w_individual', 'w_group', 'w_coop', 'w_team', 'w_all')})
    assert registered.weights(180) == {'individual': 8.9, 'group': 9.9, 'coop': 8.9,
                                        'team': 10.3, 'all': 1.}
    out = []
    for cf in (registered, lean):
        t = make_trainer(horizon=2, batch=1, slots=4,
                         future_samples=8 if pessimistic else 1,
                         pessimistic_credit=pessimistic, uncertainty_weight=0.)
        if pessimistic:
            from wms.model import WmSModel, ModelSpec
            t.world_models = [WmSModel(ModelSpec(rollout_mode='driver_params',
                                                history_hidden=8, k_hidden=8)).eval().requires_grad_(False)]
        t.counterfactual = replace(cf, max_starts=1)
        torch.manual_seed(5)
        batch = t.start_states.sample(1, 'cpu')
        variants = build_variants(batch.state, t.counterfactual, torch.Generator().manual_seed(1))
        imagined = t.imagine(batch, deterministic=True, variants=variants)
        loss, _ = t.actor_objective(imagined, 100., t.counterfactual.weights(180))
        grads = torch.autograd.grad(loss, [p for p in t.actor_parameters if p.requires_grad])
        out.append((variants.n_rows, loss.detach(), grads))
    assert (out[0][0], out[1][0]) == (9, 6)
    torch.testing.assert_close(out[0][1], out[1][1], rtol=1e-5, atol=1e-7)
    for a, b in zip(out[0][2], out[1][2]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)
