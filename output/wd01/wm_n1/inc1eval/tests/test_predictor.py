"""INC1 predictor: trajectory metric derivation and arm orchestration."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from inc1eval.obstacle import inject_obstacle
from inc1eval.predictor import (HARD_BRAKE_MPS2, Inc1Predictor,
                                trajectory_metrics)
from inc1eval.tests._helpers import make_world
from rollout2 import baseline
from rollout2.engine import IdeaABundle
from rollout2.state import Rollouts


def _rollouts(pos, spd, lanes, acc, masks=None):
    pos = torch.tensor(pos, dtype=torch.float64)
    if pos.ndim == 3:
        pos = pos.unsqueeze(0)
    spd = torch.tensor(spd, dtype=torch.float64)
    if spd.ndim == 3:
        spd = spd.unsqueeze(0)
    lanes = torch.tensor(lanes, dtype=torch.int64)
    if lanes.ndim == 3:
        lanes = lanes.unsqueeze(0)
    t_n, b_n, k_n = spd.shape[1], spd.shape[2], spd.shape[3]
    if masks is None:
        masks = np.ones((t_n, b_n, k_n), dtype=bool)
    masks = torch.tensor(masks).bool()
    if masks.ndim == 3:
        masks = masks.unsqueeze(0)
    acc = torch.tensor(acc, dtype=torch.float64)
    if acc.ndim == 3:
        acc = acc.unsqueeze(0)
    return Rollouts(
        positions=pos, speeds=spd, lanes=lanes, masks=masks, accel=acc,
        arrivals=torch.zeros((1, t_n - 1, b_n, 2), dtype=torch.int64),
        insertions=torch.zeros((1, t_n - 1, b_n, 2), dtype=torch.int64),
        validity=[], cav_commands=None, collisions_by_step=[{}],
        anomalies=[{}])


def test_delay_and_hard_brake_from_synthetic_trajectory():
    params = baseline.Params(baseline.load_params())
    t_n, k_n = 5, 2
    pos = np.zeros((t_n, 1, k_n))
    pos[:, 0, 0] = 100.0
    pos[:, 0, 1] = 60.0
    spd = np.full((t_n, 1, k_n), 10.0)
    lanes = np.full((t_n, 1, k_n), 1)
    acc = np.full((t_n - 1, 1, k_n), 0.0)
    acc[:, 0, 1] = -4.0                     # follower hard-brakes every tick
    out = trajectory_metrics(_rollouts(pos, spd, lanes, acc), params)
    lim0 = params.seg_limit(1, 100.0)
    lim1 = params.seg_limit(1, 60.0)
    expected = t_n * 0.5 * 0.5 * ((1.0 - 10.0 / lim0) + (1.0 - 10.0 / lim1))
    assert out.delay_s_per_veh[0] == pytest.approx(expected, rel=1e-6)
    assert float(out.hard_brake_s[0]) == pytest.approx(0.5 * (t_n - 1))
    assert HARD_BRAKE_MPS2 == -3.0


def test_ttc_exposure_from_synthetic_close_pair():
    params = baseline.Params(baseline.load_params())
    t_n, k_n = 4, 2
    pos = np.zeros((t_n, 1, k_n))
    pos[:, 0, 0] = 25.0
    pos[:, 0, 1] = 10.0
    spd = np.full((t_n, 1, k_n), 5.0)
    spd[:, 0, 1] = 20.0
    lanes = np.full((t_n, 1, k_n), 1)
    acc = np.zeros((t_n - 1, 1, k_n))
    out = trajectory_metrics(_rollouts(pos, spd, lanes, acc), params)
    assert float(out.ttc_lt3_s[0]) > 0.0
    assert float(out.collisions[0]) == 0.0


def test_predictor_orchestrates_all_arms_with_injected_runner():
    params = baseline.Params(baseline.load_params())

    def runner(model, plan, world, n_steps, seed):
        k_n = world.slots
        pos = np.tile(np.linspace(0, 10, n_steps + 1)[:, None, None],
                      (1, 1, k_n))
        spd = np.full((n_steps + 1, 1, k_n), 20.0)
        lanes = np.full((n_steps + 1, 1, k_n), 1)
        acc = np.zeros((n_steps, 1, k_n))
        return _rollouts(pos, spd, lanes, acc)

    world = make_world([(1, 150.0, 10.0, True), (1, 100.0, 8.0, False)])
    world, rec = inject_obstacle(world, lane=1, s_m=200.0)
    pred = Inc1Predictor(
        IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None),
        params=params, n_steps=3, rollout_fn=runner)
    out = pred.predict(world, rec, world_name="w0")
    assert set(out.idea_a) == {"R0", "R1", "R2", "R3", "R4", "R5", "R6"}
    assert set(out.v2g) == {"R0", "R1", "R2", "R3", "R4", "R5", "R6"}
    assert out.obstacle["s_m"] == 200.0
    assert out.idea_a["R0"].delay_s_per_veh.shape == (1,)


def test_build_arm_plan_r1_no_cav_is_r0_with_log():
    from inc1eval.predictor import build_arm_plan

    params = baseline.Params(baseline.load_params())
    world = make_world([(1, 100.0, 8.0, False)])
    world, rec = inject_obstacle(world, lane=1, s_m=200.0)
    plan = build_arm_plan("R1", "idea_a", IdeaABundle(
        ctrl=None, hdv=None, arrival=None, corr=None), world, rec, params)
    assert plan.log.get("r1_no_qualifying_cav") is True
    assert not plan.bundle_is_incident_mover
