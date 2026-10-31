"""S-RB-CF switch tests: ``physics.cf_model='sumo_idm'`` dispatch and default-off
legacy bit-identity."""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import tensor

from wms.physics.accel import baseline_accel
from wms.physics.config import PhysicsConfig
from wms.physics.idm import IDMParams, idm_accel
from wms.physics.idm_sumo import (SumoIDMParams, follow_speed, idm_iterations,
                                  stop_speed, sumo_idm_step)
from wms.physics.neighbours import lead_neighbour

TS = 0.5


def scene(batch: int = 1, slots: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """A mainline platoon in lane 1 (descending gaps) plus one ramp car."""
    kin = torch.zeros(batch, slots, 7)
    # slot 0 leader, slot 1 follower, slot 2 far follower, slot 3 ramp car
    rows = [(0, 1, 260.0, 25.0), (1, 1, 230.0, 22.0), (2, 1, 180.0, 27.0),
            (3, 2, 40.0, 12.0)]
    for k, lane, s, v in rows:
        kin[:, k, 0] = s
        kin[:, k, 1 + lane] = 1.0
        kin[:, k, 4] = v
        kin[:, k, 6] = 5.0
    present = torch.ones(batch, slots, dtype=torch.bool)
    return kin, present


def test_default_config_is_legacy_and_bit_identical_path():
    cfg = PhysicsConfig()
    assert cfg.cf_model == "legacy"
    assert cfg.sumo_cf_stepping_s == 0.25
    assert cfg.sumo_cf_emergency_decel_mps2 is None
    assert cfg.sumo_cf_vtype_max_mps is None
    assert cfg.sumo_cf_action_step_s is None
    kin, present = scene()
    out = baseline_accel(kin, present, cfg)
    # manual legacy transcription (idm.py path, no driver params, no extras)
    gap, lead_v = lead_neighbour(kin, present, cfg)
    p = cfg.idm_params()
    ref = idm_accel(kin[..., 4], lead_v, gap, p,
                    accel_min=cfg.accel_min_mps2, accel_max=cfg.accel_max_mps2)
    ref = torch.where(present, ref, torch.zeros_like(ref))
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_switch_replaces_core_with_sumo_step():
    cfg = replace(PhysicsConfig(), cf_model="sumo_idm")
    kin, present = scene()
    out = baseline_accel(kin, present, cfg)
    gap, lead_v = lead_neighbour(kin, present, cfg)
    p = cfg.idm_params()
    sp = SumoIDMParams(accel=tensor(float(p.a_max)), decel=tensor(float(p.b_comf)),
                       headway_time=tensor(float(p.time_headway)),
                       min_gap=tensor(float(p.s0)), des_speed=tensor(float(p.v0)),
                       delta=tensor(float(p.delta)))
    des = sp.des_speed.expand_as(kin[..., 4])
    it = idm_iterations(TS, cfg.sumo_cf_stepping_s)
    _, ref = sumo_idm_step(kin[..., 4], gap, lead_v, des, sp, dt_s=TS, iterations=it)
    # slot 3 (ramp, near lane end 75.44): stopSpeed semantics
    v3 = kin[:, 3, 4]
    end_gap = (cfg.merge_zone_s1_m - kin[:, 3, 0]).clamp_min(0.3)
    stop_next = stop_speed(v3, end_gap, sp, dt_s=TS, iterations=it)
    ref[:, 3] = (stop_next - v3) / TS
    ref = torch.where(present, ref, torch.zeros_like(ref))
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)
    # it is NOT the legacy core
    legacy = baseline_accel(kin, present, PhysicsConfig())
    assert not torch.allclose(out, legacy)


def test_switch_with_per_driver_settings():
    cfg = replace(PhysicsConfig(), cf_model="sumo_idm")
    kin, present = scene()
    dp = {"v0": tensor([[26.0, 24.0, 28.0, 18.0]]),
          "time_headway": tensor([[1.3, 0.9, 1.0, 1.2]]),
          "min_gap": tensor([[3.0, 2.0, 2.2, 2.5]]),
          "accel": tensor([[2.0, 2.4, 2.6, 2.1]]),
          "decel": tensor([[4.0, 4.2, 4.5, 4.1]])}
    out = baseline_accel(kin, present, cfg, dp)
    gap, lead_v = lead_neighbour(kin, present, cfg)
    sp = SumoIDMParams.from_driver_params(dp)
    it = idm_iterations(TS, cfg.sumo_cf_stepping_s)
    _, ref = sumo_idm_step(kin[..., 4], gap, lead_v, sp.des_speed, sp,
                           dt_s=TS, iterations=it)
    v3 = kin[:, 3, 4]
    end_gap = (cfg.merge_zone_s1_m - kin[:, 3, 0]).clamp_min(0.3)
    sp3 = SumoIDMParams.from_driver_params({k: v[:, 3:4] for k, v in dp.items()})
    ref[:, 3] = (stop_speed(v3, end_gap, sp3, dt_s=TS, iterations=it) - v3) / TS
    ref = torch.where(present, ref, torch.zeros_like(ref))
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1e-6)


def test_sumo_core_follow_speed_matches_module_on_platoon():
    # independent cross-check: the dispatched follower accel == (followSpeed - v)/dt
    cfg = replace(PhysicsConfig(), cf_model="sumo_idm")
    kin, present = scene()
    out = baseline_accel(kin, present, cfg)
    gap, lead_v = lead_neighbour(kin, present, cfg)
    p = cfg.idm_params()
    v = kin[0, 1, 4]
    vsafe = follow_speed(tensor(v, dtype=torch.float64),
                         tensor(float(gap[0, 1]), dtype=torch.float64),
                         tensor(float(lead_v[0, 1]), dtype=torch.float64),
                         tensor(float(p.v0), dtype=torch.float64),
                         SumoIDMParams(accel=tensor(float(p.a_max), dtype=torch.float64),
                                       decel=tensor(float(p.b_comf), dtype=torch.float64),
                                       headway_time=tensor(float(p.time_headway), dtype=torch.float64),
                                       min_gap=tensor(float(p.s0), dtype=torch.float64),
                                       des_speed=tensor(float(p.v0), dtype=torch.float64)),
                         dt_s=TS, iterations=2)
    assert out[0, 1].item() == pytest.approx((vsafe.item() - float(v)) / TS, abs=5e-6)


def test_config_validation():
    with pytest.raises(ValueError):
        PhysicsConfig(cf_model="krauss")
    with pytest.raises(ValueError):
        PhysicsConfig(cf_model="sumo_idm", sumo_cf_stepping_s=0.0)
    with pytest.raises(ValueError):
        PhysicsConfig(cf_model="sumo_idm", sumo_cf_emergency_decel_mps2=-1.0)
    with pytest.raises(ValueError):
        PhysicsConfig(cf_model="sumo_idm", sumo_cf_action_step_s=0.25)
    cfg = PhysicsConfig(cf_model="sumo_idm", sumo_cf_action_step_s=1.0,
                        sumo_cf_emergency_decel_mps2=9.0, sumo_cf_vtype_max_mps=36.11)
    assert cfg.cf_model == "sumo_idm"
