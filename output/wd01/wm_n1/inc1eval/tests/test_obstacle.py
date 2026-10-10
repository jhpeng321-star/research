"""INC1 obstacle support: placement, pinning, neighbour perception, v2g."""
from __future__ import annotations

import pytest
import torch

from inc1eval.obstacle import (Obstacle, inject_obstacle, nearest_free_shift,
                               obstacle_mask)
from inc1eval.rollout_baseline import idea_a_rollout, rollout_v2g_baseline
from inc1eval.tests._helpers import make_world
from rollout2.engine import IdeaABundle
from rollout2.state import Rollout2Config, Rollout2State, StrategyActions
from rollout2.step import step_once
from rollout2.validity import project_rollout_step


def test_nearest_free_shift_clear_and_blocked():
    assert nearest_free_shift(100.0, [70.0]) == 0.0
    shift = nearest_free_shift(100.0, [101.0])
    assert shift != 0.0 and abs(shift) <= 20.0
    assert abs((100.0 + shift) - 101.0) >= 15.0


def test_inject_obstacle_sets_slot_and_logs_shift():
    world = make_world([(1, 70.0, 15.0, False)])
    out, record = inject_obstacle(world, lane=1, s_m=95.0)
    assert isinstance(record, Obstacle)
    assert record.slot == 1 and record.shift_m == 0.0
    assert bool(out.present[0, 1])
    assert float(out.v_mps[0, 1]) == 0.0 and float(out.a_mps2[0, 1]) == 0.0
    assert bool(obstacle_mask(out)[0, 1])
    assert int(out.slot_id[0, 1]) >= 0


def test_inject_obstacle_k_full_raises():
    world = make_world([(1, 70.0, 15.0, False)], slots=1)
    with pytest.raises(ValueError):
        inject_obstacle(world, lane=1, s_m=95.0)


def _run_steps(world, n_steps=20, integrator="d1", lane_changes=False):
    bundle = IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None)
    cfg = Rollout2Config(integrator=integrator, t0_s=0.0,
                         lane_changes_enabled=bool(lane_changes))
    cfg.state = Rollout2State(world.batch, world.slots)
    actions = StrategyActions(
        action_code=torch.where(world.is_cav,
                                torch.zeros_like(world.wall_group),
                                torch.full_like(world.wall_group, -1)))
    w = world
    for i in range(n_steps):
        w, _aux = step_once(w, actions, bundle=bundle, config=cfg,
                            t_s=i * 0.5)
    return w


def test_obstacle_is_pinned_and_kept():
    world = make_world([(1, 70.0, 15.0, False)])
    world, rec = inject_obstacle(world, lane=1, s_m=100.0)
    out = _run_steps(world, n_steps=20)
    assert bool(out.present[0, rec.slot])
    assert float(out.s_m[0, rec.slot]) == pytest.approx(100.0)
    assert float(out.v_mps[0, rec.slot]) == 0.0
    assert bool(obstacle_mask(out)[0, rec.slot])


def test_neighbour_sees_stopped_leader_and_does_not_overlap():
    world = make_world([(1, 60.0, 16.0, False)])
    world, rec = inject_obstacle(world, lane=1, s_m=100.0)
    out = _run_steps(world, n_steps=30)
    foll = 0
    assert float(out.v_mps[0, foll]) < 16.0          # IDM braked for it
    gap = float(out.s_m[0, rec.slot]) - 5.0 - float(out.s_m[0, foll])
    assert gap >= -1e-6                              # never overlapped
    assert bool(out.present[0, rec.slot])


def test_validity_layer_holds_obstacle_even_with_accel():
    b, k = 1, 3
    s = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64)
    v = torch.tensor([[0.0, 5.0, 5.0]], dtype=torch.float64)
    length = torch.full((b, k), 5.0, dtype=torch.float64)
    present = torch.ones((b, k), dtype=torch.bool)
    lane = torch.zeros((b, k), dtype=torch.int64)
    accel = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float64)
    lane_class = torch.full((b, k), 1, dtype=torch.int64)   # try to change lane
    obstacle = torch.tensor([[True, False, False]])
    s_new, lane_new, v_new, a_used, _cls, _c = project_rollout_step(
        s, lane, v, length, present, accel, lane_class,
        obstacle=obstacle)
    assert float(s_new[0, 0]) == 10.0
    assert float(v_new[0, 0]) == 0.0
    assert float(a_used[0, 0]) == 0.0
    assert int(lane_new[0, 0]) == 0
    # a non-obstacle slot is free to accelerate / lane-change
    assert float(v_new[0, 1]) > 5.0


def test_v2g_baseline_pins_obstacle_too():
    world = make_world([(1, 60.0, 16.0, False)])
    world, rec = inject_obstacle(world, lane=1, s_m=100.0)
    bundle = IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None)
    out = rollout_v2g_baseline(bundle, world, n_steps=20, seed=0)
    assert float(out.positions[0, -1, 0, rec.slot]) == pytest.approx(100.0)
    assert float(out.speeds[0, -1, 0, rec.slot]) == 0.0
    assert bool(out.masks[0, -1, 0, rec.slot])


def test_idea_a_rollout_runs_with_obstacle():
    world = make_world([(1, 60.0, 16.0, False)])
    world, rec = inject_obstacle(world, lane=1, s_m=100.0)
    bundle = IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None)
    out = idea_a_rollout(bundle, world, n_steps=10)
    assert torch.isfinite(out.positions).all()
    assert float(out.positions[0, 0, 0, rec.slot]) == 100.0
