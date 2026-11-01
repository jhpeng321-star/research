"""Regression for missing network-local speedFactor mapping in SUMO CF."""
from dataclasses import replace

import pytest
import torch

from wms.physics.config import PhysicsConfig
from wms.physics.idm_sumo import (SumoIDMParams, idm_iterations, sumo_idm_baseline_accel,
                                 sumo_idm_step)


def test_approach_cf_uses_local_speed_factor_once_and_preserves_mainline():
    # Separate rows prevent same-lane interactions. Front positions distinguish
    # ramp approach from acceleration lane; target speeds are NETWORK attributes.
    kin = torch.tensor([[[-200., 0., 0., 1., 25., 0., 5.]],
                        [[30., 0., 0., 1., 25., 0., 5.]],
                        [[-200., 1., 0., 0., 25., 0., 5.]]], dtype=torch.float64)
    mask = torch.ones((3, 1), dtype=torch.bool)
    physics = PhysicsConfig(cf_model='sumo_idm', merge_model='sumo')
    params = {name: torch.full((3, 1), value, dtype=torch.float64) for name, value in
              dict(v0=31.29*1.1, accel=1.2, decel=2., time_headway=1.2, min_gap=2.).items()}
    unchanged = {key:value.clone() for key,value in params.items()}
    legacy = sumo_idm_baseline_accel(kin, mask, physics, params)
    corrected = sumo_idm_baseline_accel(kin, mask,
        replace(physics, sumo_cf_lane_speed_limits=True), params)
    assert torch.equal(corrected[1:], legacy[1:])
    assert corrected[0, 0] < 0 < legacy[0, 0]
    p = replace(SumoIDMParams.from_driver_params({key:value[:1] for key,value in params.items()}),
                des_speed=torch.full((1, 1), 22.22*1.1, dtype=torch.float64))
    _, expected = sumo_idm_step(kin[:1, :, 4], torch.full((1, 1), 1e6, dtype=torch.float64),
        kin[:1, :, 4], p.des_speed, p, dt_s=physics.dt_s,
        iterations=idm_iterations(physics.dt_s))
    torch.testing.assert_close(corrected[:1], expected, rtol=0, atol=1e-12)
    assert all(torch.equal(params[key], value) for key,value in unchanged.items())


def test_network_speed_switch_fails_closed_for_legacy_cf():
    with pytest.raises(ValueError, match='cf_model=sumo_idm'):
        PhysicsConfig(sumo_cf_lane_speed_limits=True)
