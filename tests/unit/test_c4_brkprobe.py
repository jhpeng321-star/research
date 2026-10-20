"""V10BRKPROBE inference-only braking diagnostics (CPU, tiny random WM-S).

Guarantees: every switch defaults off and a toggled-off rollout is bit-identical; zeroing the
temporal output changes the acceleration; the combined-residual bound caps ``|a - physics|``;
the low-pass state is reset between rollouts.
"""
from __future__ import annotations

import torch

from wms.model.residual_net import HdvConfig
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel


def _model():
    torch.manual_seed(11)
    model = WmSModel(ModelSpec(rollout_mode="hybrid", history_hidden=16, k_hidden=8),
                     hdv=HdvConfig(d_model=16, n_heads=2, n_layers=2, ffn=32))
    return model.eval()


def _state(s, v, lane, cav):
    k = len(s)
    return {"s_m": torch.tensor([s], dtype=torch.float32),
            "v_mps": torch.tensor([v], dtype=torch.float32),
            "a_mps2": torch.zeros(1, k), "lane": torch.tensor([lane], dtype=torch.long),
            "present": torch.ones(1, k, dtype=torch.bool),
            "is_cav": torch.tensor([cav]), "length_m": torch.full((1, k), 5.0)}


def test_defaults_off_and_off_rollout_is_bit_identical():
    model = _model()
    assert model.diag_zero_temporal is False
    assert model.diag_residual_bound_mps2 is None
    assert model.diag_temporal_tau_s is None
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0),
               (False, False, False))

    def run():
        return model.rollout(dict(x), n_steps=8)

    first = run()
    model.diag_zero_temporal = True
    _ = run()
    model.diag_zero_temporal = False
    again = run()
    for key in ("positions", "speeds", "accel", "lanes", "masks"):
        assert torch.equal(first[key], again[key]), key


def test_zero_temporal_removes_the_temporal_acceleration_term():
    model = _model()
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0),
               (False, False, False))
    base = model.step(dict(x), k_steps=5)
    model.diag_zero_temporal = True
    zeroed = model.step(dict(x), k_steps=5)
    assert not torch.equal(base["accel_mps2"], zeroed["accel_mps2"])
    # The attention head and physics core are untouched.
    assert torch.equal(base["physics_accel_mps2"], zeroed["physics_accel_mps2"])


def test_combined_residual_bound_caps_the_correction():
    model = _model()
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0),
               (False, False, False))
    model.diag_residual_bound_mps2 = 0.5
    out = model.step(dict(x), k_steps=5)
    correction = out["accel_mps2"] - out["physics_accel_mps2"]
    learned = out["hdv_mask"]
    assert torch.all(correction[learned].abs() <= 0.5 + 1e-5)


def test_temporal_low_pass_is_causal_and_reset():
    model = _model()
    x = _state((-600.0, -560.0, -500.0), (15.0, 10.0, 20.0), (1, 1, 0),
               (False, False, False))
    model.diag_temporal_tau_s = 2.0
    first = model.step(dict(x), k_steps=5)
    second = model.step(dict(x), k_steps=5)
    model.reset_diag_state()
    again = model.step(dict(x), k_steps=5)
    assert torch.equal(first["accel_mps2"], again["accel_mps2"])
    assert not torch.equal(first["accel_mps2"], second["accel_mps2"])
