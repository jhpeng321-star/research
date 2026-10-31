"""S-RB-CF: pure-physics switch (``WmSModel.learned_accel_off``) and preset tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import tensor

from wms.eval.predictor import RolloutFlags, WmsPredictor
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.idm_sumo import PURE_SUMO_PHYSICS


def scene_state() -> dict[str, torch.Tensor]:
    kin = torch.zeros(1, 4, 7)
    for k, lane, s, v in [(0, 1, 260.0, 25.0), (1, 1, 230.0, 22.0),
                          (2, 1, 180.0, 27.0), (3, 2, 40.0, 12.0)]:
        kin[:, k, 0] = s
        kin[:, k, 1 + lane] = 1.0
        kin[:, k, 4] = v
        kin[:, k, 6] = 5.0
    return {"veh_kin": kin, "present": torch.ones(1, 4, dtype=torch.bool)}


def test_learned_accel_off_defaults_off_and_bit_identical():
    model = WmSModel(ModelSpec(), physics=PhysicsConfig())
    assert model.learned_accel_off is False
    state = scene_state()
    torch.manual_seed(0)
    a = model(state)["accel_mps2"]
    torch.manual_seed(0)
    b = model(state)["accel_mps2"]
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_learned_accel_off_returns_pure_physics_core():
    model = WmSModel(ModelSpec(), physics=PhysicsConfig())
    state = scene_state()
    torch.manual_seed(1)
    default = model(state, k_steps=5)
    # the gate is open at k>0 and the random residual actually contributes
    assert not torch.allclose(default["accel_mps2"], default["physics_accel_mps2"])
    model.learned_accel_off = True
    torch.manual_seed(1)
    pure = model(state, k_steps=5)
    torch.testing.assert_close(pure["accel_mps2"], pure["physics_accel_mps2"],
                               rtol=0.0, atol=0.0)
    # the lane logits/gate read-outs are untouched (only acceleration is switched)
    torch.testing.assert_close(default["gate"], pure["gate"], rtol=0.0, atol=0.0)


def test_learned_accel_off_with_sumo_cf_matches_baseline_accel():
    from dataclasses import replace

    from wms.physics.accel import baseline_accel
    from wms.physics.state import state_kin, state_present

    cfg = replace(PhysicsConfig(), **PURE_SUMO_PHYSICS)
    model = WmSModel(ModelSpec(), physics=cfg)
    model.learned_accel_off = True
    state = scene_state()
    out = model(state)
    kin = state_kin(state)
    present = state_present(state, kin)
    ref = baseline_accel(kin, present, cfg)
    torch.testing.assert_close(out["accel_mps2"], ref, rtol=0.0, atol=1e-6)


def test_preset_is_complete_and_valid():
    cfg = PhysicsConfig(**PURE_SUMO_PHYSICS)
    assert cfg.cf_model == "sumo_idm"
    assert cfg.emergency_braking is False
    assert cfg.merge_cal is False and cfg.mainline_merge_yield is False
    assert cfg.merge_end_wait is False and cfg.merge_impatience is False


def test_rollout_flags_pure_physics_sets_model_switch():
    model = WmSModel(ModelSpec(), physics=PhysicsConfig())
    backend = SimpleNamespace(model=model, config={}, tools=SimpleNamespace())
    assert RolloutFlags().pure_physics is False
    WmsPredictor(backend, RolloutFlags(pure_physics=True))
    assert model.learned_accel_off is True
    # default flags leave it off
    model2 = WmSModel(ModelSpec(), physics=PhysicsConfig())
    backend2 = SimpleNamespace(model=model2, config={}, tools=SimpleNamespace())
    WmsPredictor(backend2, RolloutFlags())
    assert model2.learned_accel_off is False
