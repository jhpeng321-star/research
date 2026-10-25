"""LOOP14 / S-TTC ``physics.emergency_onset``: default bit-identical; driver_b plans with learned b."""
from dataclasses import replace

import pytest
import torch

from wms.config.load import load_train_config
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.merge_pivot import advance_longitudinal
from wms.physics.accel import apply_emergency_envelope, baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.emergency import leader_inputs, safe_accel, safe_accel_driver_b

DT, B = 0.5, 9.0


def _params(kin, decel, **over):
    defaults = dict(zip(PARAMETER_NAMES, (31.29, 1.1, 2.5, 2.6, decel, .2, .1, 6., 1.)))
    defaults.update(over)
    return {k: (v.to(kin.dtype) if torch.is_tensor(v) else torch.full_like(kin[..., 4], v))
            for k, v in defaults.items()}


def _scene(n=64, k=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    kin = torch.zeros(n, k, 7, dtype=torch.float64)
    kin[..., 0] = torch.rand(n, k, generator=g, dtype=torch.float64) * 400 - 300
    lane = torch.randint(0, 2, (n, k), generator=g)
    kin[..., 1:4] = torch.nn.functional.one_hot(lane, 3).to(kin.dtype)
    kin[..., 4] = torch.rand(n, k, generator=g, dtype=torch.float64) * 31
    kin[..., 4] *= (torch.rand(n, k, generator=g) > 0.25).to(kin.dtype)    # some stopped vehicles
    kin[..., 6] = 5.0
    present = torch.rand(n, k, generator=g) > 0.1
    nominal = torch.rand(n, k, generator=g, dtype=torch.float64) * 4 - 2
    b = 1.0 + 3.5 * torch.rand(n, k, generator=g, dtype=torch.float64)
    return kin, present, nominal, b


def test_validation_and_default_omitted():
    with pytest.raises(ValueError):
        PhysicsConfig(emergency_onset="driver_b")                 # needs emergency_braking
    with pytest.raises(ValueError):
        PhysicsConfig(emergency_braking=True, emergency_onset="comfort")
    assert PhysicsConfig().emergency_onset == "plant"
    cfg = load_train_config("configs/c4_wm_v10round13e_stage_b.toml")
    assert "emergency_onset" not in cfg.to_legacy_flat()           # old configs/fingerprints unchanged
    ft = load_train_config("configs/c4_wm_loop14_ft.toml", ['model.init_from="x.pt"'])
    assert ft.physics.emergency_onset == "driver_b" and ft.to_legacy_flat()["emergency_onset"] == "driver_b"


def test_plant_default_is_the_r13e_envelope_bit_for_bit():
    kin, present, nominal, b = _scene()
    phys = PhysicsConfig(emergency_braking=True)
    params = _params(kin, b)
    out = apply_emergency_envelope(kin, present, nominal, phys, params)
    gap, lv, has = leader_inputs(kin, present, phys)
    ref = torch.where(has, safe_accel(kin[..., 4], lv, gap, nominal, params["min_gap"], dt=DT,
                                      emergency_decel=B), nominal)
    assert torch.equal(out, ref)
    assert torch.equal(out, apply_emergency_envelope(kin, present, nominal,
                                                     replace(phys, emergency_onset="plant"), params))


def test_driver_b_respects_the_plant_floor_and_is_conservative_for_stopped_leaders():
    kin, present, nominal, b = _scene(seed=3)
    phys = PhysicsConfig(emergency_braking=True)
    params = _params(kin, b)
    plant = apply_emergency_envelope(kin, present, nominal, phys, params)
    onset = apply_emergency_envelope(kin, present, nominal, replace(phys, emergency_onset="driver_b"), params)
    assert torch.isfinite(onset).all()
    assert torch.all(onset >= -B)
    gap, lv, has = leader_inputs(kin, present, phys)
    assert torch.equal(onset[~has], nominal[~has])                 # no leader / obstacle: unchanged
    # A moving leader is assumed to stop with ITS learned b (SUMO predMaxDecel), which can be more
    # permissive than 13E's B; against a stopped leader/obstacle comfortable planning is stricter.
    stopped = has & (lv == 0)
    assert stopped.any()
    assert torch.all(onset[stopped] <= plant[stopped] + 1e-9)
    # Harder than b only where the plant-limit plan itself demands it.
    harsh = onset < -b - 1e-6
    assert torch.all(plant[harsh] < -b[harsh] + 1e-6)


def test_driver_b_gradients_are_finite():
    kin, present, nominal, b = _scene(n=16, seed=5)
    kin.requires_grad_(True)
    nominal.requires_grad_(True)
    b.requires_grad_(True)
    phys = PhysicsConfig(emergency_braking=True, emergency_onset="driver_b")
    out = apply_emergency_envelope(kin, present, nominal, phys, _params(kin, b))
    out.sum().backward()
    for t in (kin.grad, nominal.grad, b.grad):
        assert torch.isfinite(t).all()


def test_leader_comfort_decel_enters_the_leader_stop_term():
    v = torch.tensor([[20.0]], dtype=torch.float64)
    lv = torch.tensor([[15.0]], dtype=torch.float64)
    gap = torch.tensor([[30.0]], dtype=torch.float64)
    nominal = torch.tensor([[1.0]], dtype=torch.float64)
    b = torch.tensor([[3.0]], dtype=torch.float64)
    soft = safe_accel_driver_b(v, lv, gap, nominal, 2.5, dt=DT, emergency_decel=B, comfort_decel=b,
                               leader_comfort_decel=torch.tensor([[1.5]], dtype=torch.float64))
    hard = safe_accel_driver_b(v, lv, gap, nominal, 2.5, dt=DT, emergency_decel=B, comfort_decel=b,
                               leader_comfort_decel=torch.tensor([[6.0]], dtype=torch.float64))
    assert hard.item() < soft.item()        # a leader that can stop shorter leaves less room


@pytest.mark.parametrize("b_comf", [3.0, 4.5])
def test_approach_to_stopped_leader_brakes_comfortably_and_earlier(b_comf):
    """Feasible-with-b approach: driver_b never brakes harder than b; 13E waits and brakes near B."""
    peaks = {}
    for onset in ("plant", "driver_b"):
        phys = PhysicsConfig(emergency_braking=True, stop_time_integration=False,
                             lane_rule="forced_merge_only", emergency_onset=onset)
        v0 = torch.tensor([12.0, 20.0, 28.0], dtype=torch.float64)
        gap = 2.5 + v0.square() / (2 * b_comf) + v0 * DT + 1.0
        kin = torch.zeros(len(v0), 2, 7, dtype=v0.dtype)
        kin[..., 6] = 5
        kin[..., 2] = 1
        kin[:, 0, 4] = v0
        kin[:, 1, 0] = gap + 5
        present = torch.ones(len(v0), 2, dtype=torch.bool)
        params = _params(kin, b_comf, accel=2.6)
        worst = torch.zeros(len(v0), dtype=v0.dtype)
        for _ in range(120):
            accel = baseline_accel(kin, present, phys, params)
            assert torch.isfinite(accel).all() and torch.all(accel >= -B)
            worst = torch.minimum(worst, accel[:, 0])
            commands = torch.stack((accel[:, 0], torch.zeros_like(v0)), -1)
            kin = advance_longitudinal(kin, commands, DT, stop_time_integration=False)
            assert torch.all(kin[:, 1, 0] - 5 - kin[:, 0, 0] >= -1e-6)     # no overlap
        assert kin[:, 0, 4].max() < 0.05
        peaks[onset] = worst
    assert torch.all(peaks["driver_b"] >= -b_comf - 1e-6)
    assert torch.all(peaks["driver_b"] >= peaks["plant"] - 1e-9)
