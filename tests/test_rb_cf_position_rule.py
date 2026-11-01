"""S-RB-CF: ``physics.position_rule`` switch (SUMO Euler position parity).

SUMO 1.27.1 default update (``--step-method.ballistic`` off):
``MSVehicle.cpp:4954-4963`` ``updateState``: with
``MSGlobals::gSemiImplicitEulerUpdate`` the displacement is
``deltaPos = SPEED2DIST(vNext) = vNext * TS`` -- the position advances with the
NEW speed.  The WM's historical integrator used the trapezoid
``s' = s + 0.5 (v + v') dt``; ``position_rule='sumo_euler'`` (default OFF,
bit-identical legacy default) restores SUMO parity.  Evidence: 2,346,379 FIT
motion intervals, Euler residual 3.5e-5 m vs trapezoid 0.039 m
(rb-shared LANE_TO_CF_POSITION_20261101 / LANE receipts).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import tensor

from wms.model.merge_pivot import advance_longitudinal
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import POSITION_RULES, PhysicsConfig
from wms.physics.idm_sumo import PURE_SUMO_PHYSICS, PURE_SUMO_PHYSICS_EULER

TS = 0.5


def kin_batch(rows: list[tuple[float, float, float]]) -> torch.Tensor:
    """[1, K, 7] kin: rows of (s, v, accel)."""
    kin = torch.zeros(1, len(rows), 7)
    for k, (s, v, _a) in enumerate(rows):
        kin[:, k, 0] = s
        kin[:, k, 4] = v
        kin[:, k, 6] = 5.0
    return kin


def test_source_formula_euler_displacement():
    # MSVehicle.cpp:4957-4959: deltaPos = SPEED2DIST(vNext) = vNext * TS.
    # v=20, a=-2, dt=0.5 -> v'=19: euler ds = 9.5, trapezoid ds = 9.75.
    kin = kin_batch([(100.0, 20.0, -2.0)])
    accel = tensor([[-2.0]], dtype=torch.float64)
    euler = advance_longitudinal(kin.double(), accel, TS, position_rule="sumo_euler")
    assert euler[0, 0, 4].item() == pytest.approx(19.0, abs=1e-12)
    assert (euler[0, 0, 0].item() - 100.0) == pytest.approx(19.0 * TS, abs=1e-12)
    trap = advance_longitudinal(kin.double(), accel, TS)
    assert (trap[0, 0, 0].item() - 100.0) == pytest.approx(0.5 * (20.0 + 19.0) * TS, abs=1e-12)


def test_braking_bias_direction_and_size():
    # While braking, trapezoid moves FARTHER than SUMO by 0.5 (v - v') dt
    # (overshoots into the leader / jam); while accelerating it moves LESS far
    # by the same amount (slower queue discharge).  Both biases of the legacy
    # rule point at "too severe, too slow to clear".
    for s0, v, a in [(100.0, 25.0, -4.0), (100.0, 3.0, 2.0), (100.0, 12.0, 0.7)]:
        kin = kin_batch([(s0, v, a)]).double()
        accel = tensor([[a]], dtype=torch.float64)
        d_euler = advance_longitudinal(kin, accel, TS, position_rule="sumo_euler")[0, 0, 0] - s0
        d_trap = advance_longitudinal(kin, accel, TS)[0, 0, 0] - s0
        v_next = max(0.0, v + a * TS)
        assert float(d_euler) == pytest.approx(v_next * TS, abs=1e-12)
        assert float(d_trap - d_euler) == pytest.approx(0.5 * (v - v_next) * TS, abs=1e-12)


def test_clamped_stop_euler_does_not_move_and_skips_stop_fallback():
    # v=1, a=-3, dt=0.5 -> v' clamps to 0.  SUMO Euler: deltaPos = 0 (the car
    # does not move in its stopping step).  The exact stopping-distance fallback
    # is a trapezoid-only correction: under Euler it must stay inactive.
    kin = kin_batch([(50.0, 1.0, -3.0)]).double()
    accel = tensor([[-3.0]], dtype=torch.float64)
    euler = advance_longitudinal(kin, accel, TS, stop_time_integration=True,
                                 position_rule="sumo_euler")
    assert euler[0, 0, 4].item() == 0.0
    assert euler[0, 0, 0].item() == pytest.approx(50.0, abs=1e-12)
    trap = advance_longitudinal(kin, accel, TS, stop_time_integration=True)
    assert trap[0, 0, 0].item() == pytest.approx(50.0 + 1.0 / 6.0, abs=1e-9)


def test_default_rule_is_bit_identical_legacy():
    kin = kin_batch([(100.0, 20.0, -2.0), (60.0, 5.0, 1.5)]).double()
    accel = tensor([[-2.0, 1.5]], dtype=torch.float64)
    a = advance_longitudinal(kin, accel, TS, stop_time_integration=True)
    b = advance_longitudinal(kin, accel, TS, stop_time_integration=True,
                             position_rule="legacy_trapezoid")
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    assert PhysicsConfig().position_rule == "legacy_trapezoid"


def test_config_validation_and_preset():
    assert "sumo_euler" in POSITION_RULES and "legacy_trapezoid" in POSITION_RULES
    with pytest.raises(ValueError):
        PhysicsConfig(position_rule="euler")
    cfg = PhysicsConfig(position_rule="sumo_euler")
    assert cfg.position_rule == "sumo_euler"
    # provenance: the frozen FULL-v4 preset is UNCHANGED (no position key ->
    # legacy trapezoid); the Euler preset adds exactly that one key.
    assert "position_rule" not in PURE_SUMO_PHYSICS
    assert PURE_SUMO_PHYSICS_EULER == {**PURE_SUMO_PHYSICS, "position_rule": "sumo_euler"}
    assert PhysicsConfig(**PURE_SUMO_PHYSICS).position_rule == "legacy_trapezoid"
    assert PhysicsConfig(**PURE_SUMO_PHYSICS_EULER).position_rule == "sumo_euler"


def scene_state() -> dict[str, torch.Tensor]:
    kin = torch.zeros(1, 4, 7)
    for k, lane, s, v in [(0, 1, 260.0, 25.0), (1, 1, 230.0, 22.0),
                          (2, 1, 180.0, 27.0), (3, 2, 40.0, 12.0)]:
        kin[:, k, 0] = s
        kin[:, k, 1 + lane] = 1.0
        kin[:, k, 4] = v
        kin[:, k, 6] = 5.0
    lane = torch.tensor([[1, 1, 1, 2]])
    return {"s_m": kin[..., 0].clone(), "v_mps": kin[..., 4].clone(),
            "a_mps2": torch.zeros(1, 4), "lane": lane,
            "present": torch.ones(1, 4, dtype=torch.bool)}


def test_wm_step_position_rule_composes_with_the_model():
    # End-to-end through WmSModel.step (no pivot: the wms.py integrator branch).
    # Same seeds -> identical accelerations; only the position rule differs and
    # it differs by exactly the source formula on every HDV.
    torch.manual_seed(7)
    model_a = WmSModel(ModelSpec(), physics=PhysicsConfig())
    torch.manual_seed(7)
    model_b = WmSModel(ModelSpec(), physics=replace(PhysicsConfig(),
                                                    position_rule="sumo_euler"))
    state = scene_state()
    torch.manual_seed(2)
    out_a = model_a.step(state, k_steps=3)
    torch.manual_seed(2)
    out_b = model_b.step(state, k_steps=3)
    torch.testing.assert_close(out_a["accel_mps2"], out_b["accel_mps2"],
                               rtol=0.0, atol=0.0)
    v = state["v_mps"]
    s = state["s_m"]
    v_next = (v + out_a["accel_mps2"] * model_a.physics.dt_s).clamp_min(0.0)
    torch.testing.assert_close(out_a["next_s_m"], s + 0.5 * (v + v_next)
                               * model_a.physics.dt_s, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(out_b["next_s_m"], s + v_next
                               * model_b.physics.dt_s, rtol=0.0, atol=1e-6)
    assert not torch.allclose(out_a["next_s_m"], out_b["next_s_m"])
