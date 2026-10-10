"""Selectable residual-gate behavior and backward-compatible checkpoint loading."""
from __future__ import annotations

import torch

from wms.model.checkpoint import load_checkpoint
from wms.model.gate import monotone_gate, saturating_gate
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


def test_saturating_gate_shape_monotonicity_and_limit():
    a = torch.logit(torch.tensor(0.30))
    b = torch.log(torch.expm1(torch.tensor(50.0)))
    k = torch.tensor([[0, 10, 20], [60, 120, 240]])
    gate = saturating_gate(a, b, k, 240)

    assert gate.shape == k.shape
    assert gate[0, 0].item() == 0.0
    assert bool((gate[:, 1:] >= gate[:, :-1]).all())
    assert bool((gate[1:, 0] >= gate[:-1, -1]).all())
    assert gate[-1, -1].item() < 0.30
    assert torch.isclose(gate[-1, -1], torch.tensor(0.2975), atol=0.002)


def test_saturating_gate_initial_parameters_and_gradients():
    model = WmSModel(ModelSpec(gate_kind="saturating"))
    values = model.gate(torch.tensor([10, 20, 60, 120, 240]))
    assert torch.isclose(torch.sigmoid(model.gate_g_max_logit), torch.tensor(0.30))
    assert torch.isclose(torch.nn.functional.softplus(model.gate_tau_raw), torch.tensor(50.0))
    assert torch.allclose(values, torch.tensor([0.0544, 0.0989, 0.2096, 0.2727, 0.2975]),
                          atol=0.001)

    values.sum().backward()
    assert model.gate_g_max_logit.grad is not None
    assert model.gate_tau_raw.grad is not None
    assert torch.isfinite(model.gate_g_max_logit.grad)
    assert torch.isfinite(model.gate_tau_raw.grad)
    assert model.gate_g_max_logit.grad.abs() > 0
    assert model.gate_tau_raw.grad.abs() > 0


def test_legacy_checkpoint_defaults_to_identical_cumsum_gate(tmp_path):
    torch.manual_seed(17)
    model = WmSModel(ModelSpec())
    steps = torch.tensor([0, 1, 10, 20, 60, 120, 240])
    expected = model.gate(steps)
    old_cumsum = monotone_gate(model.gate_increments, steps, model.physics.max_gap_steps)
    assert torch.equal(expected, old_cumsum)

    # Legacy payload: no gate selector in its flat config and the original state dict.
    path = tmp_path / "legacy_cumsum.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(), "config": {}}, path)
    loaded = load_checkpoint(path)
    assert loaded.model.gate_kind == "cumsum"
    assert torch.equal(loaded.model.gate(steps), expected)
    state = {
        "s_m": torch.tensor([[0.0, 30.0, 60.0]]),
        "v_mps": torch.tensor([[20.0, 18.0, 22.0]]),
        "a_mps2": torch.zeros(1, 3),
        "lane": torch.tensor([[0, 1, 2]]),
        "length_m": torch.full((1, 3), 5.0),
        "present": torch.ones(1, 3, dtype=torch.bool),
        "is_cav": torch.zeros(1, 3, dtype=torch.bool),
    }
    with torch.no_grad():
        expected_output = model(state, k_steps=steps[2])
        actual_output = loaded.model(state, k_steps=steps[2])
    assert expected_output.keys() == actual_output.keys()
    for key in expected_output:
        if isinstance(expected_output[key], torch.Tensor):
            assert torch.equal(expected_output[key], actual_output[key]), key


def test_saturating_checkpoint_loads_learned_parameters(tmp_path):
    model = WmSModel(ModelSpec(gate_kind="saturating"))
    path = tmp_path / "saturating.pt"
    torch.save({"format": "wm_s/1", "model": model.state_dict(),
                "config": {"gate_kind": "saturating"}}, path)
    loaded = load_checkpoint(path)
    assert loaded.model.gate_kind == "saturating"
    assert torch.equal(loaded.model.gate_g_max_logit, model.gate_g_max_logit)
    assert torch.equal(loaded.model.gate_tau_raw, model.gate_tau_raw)
