"""Tests for the optional per-driver v0 rollout flag (default OFF)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from driver.rollout_flag import apply_desired_speed_v0
from rollout2 import baseline
from rollout2.state import Rollout2Config, Rollout2State, StrategyActions, WorldState
from rollout2.step import step_once


def _world():
    return WorldState(
        s_m=torch.tensor([[0.0, 200.0]]),
        v_mps=torch.tensor([[25.0, 25.0]]),
        a_mps2=torch.zeros(1, 2),
        lane=torch.zeros(1, 2, dtype=torch.int64),
        length_m=torch.full((1, 2), 5.0),
        present=torch.ones(1, 2, dtype=torch.bool),
        is_cav=torch.zeros(1, 2, dtype=torch.bool),
        wall_group=torch.full((1, 2), -1, dtype=torch.int64),
        slot_id=torch.tensor([[0, 1]]),
    )


def _run(desired_speed_mps):
    cfg = Rollout2Config(params=None, integrator="d1", residual_enabled=False,
                         arrivals_enabled=False, lane_changes_enabled=False,
                         t0_s=0.0, desired_speed_mps=desired_speed_mps)
    cfg.state = Rollout2State(1, 2)
    actions = StrategyActions(
        action_code=torch.full((1, 2), -1, dtype=torch.int64))
    bundle = SimpleNamespace(ctrl=None, hdv=None, arrival=None)
    _, aux = step_once(_world(), actions, bundle=bundle, config=cfg, t_s=0.0)
    return aux.baseline_idm_accel_mps2.numpy()[0]


def test_default_flag_is_off():
    assert Rollout2Config().desired_speed_mps is None


def test_flag_changes_idm_but_default_is_reference():
    base = _run(None)
    slow = _run(np.array([5.0, 5.0]))
    assert slow[0] < base[0] - 1.0          # lower desired speed -> brake harder
    assert abs(base[0] - base[0]) < 1e-12


def test_apply_desired_speed_v0_falls_back_and_floors():
    assert apply_desired_speed_v0(None, 20.0) is None
    out = apply_desired_speed_v0(np.array([np.nan, 0.0, 12.0]),
                                 np.array([30.0, 30.0, 30.0]))
    assert out[0] == 30.0                    # missing -> segment-limit fallback
    assert out[1] > 0.0                      # floored positive
    assert out[2] == 12.0


def test_core_accelerations_override_is_optional():
    params = baseline.Params(baseline.load_params())
    core = baseline.V2GCore(params, lane_changes_enabled=False)
    w = baseline.WorldRows(
        params.off, np.array([0, 0]), np.array([0.0, 200.0]),
        np.array([25.0, 25.0]), np.array([5.0, 5.0]), np.array([0, 1]))
    a_default = core.accelerations(w)[0]
    a_override = core.accelerations(w, v0_override=np.array([8.0, 8.0]))[0]
    assert a_override[0] < a_default[0]
