"""Physics functions, projection, batched physics and wall executor vs legacy."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from wms.legacy.blackbox.traj import BlackBoxTrajModel
from wms.model.lane_head import lane_context
from wms.model.residual_net import HdvConfig
from wms.physics import (FeasibilityProjection, IDMParams, PhysicsConfig, ProjectionConfig,
                         baseline_accel, batched, idm_accel, mobil_merge_logits, state_kin,
                         wall_exec)

from . import inputs
from .legacy import legacy, load_legacy
from .test_model_equivalence import assert_same

DRIVER_NAMES = ("v0", "time_headway", "min_gap", "accel", "decel", "politeness",
                "mobil_threshold", "merge_gap", "merge_urgency")


@pytest.fixture(scope="module", autouse=True)
def _legacy():
    load_legacy()


def _driver_params(b, k, seed=2):
    g = torch.Generator().manual_seed(seed)
    lo = torch.tensor([20., 0.8, 1.0, 1.5, 2.5, 0.1, 0.05, 3.0, 0.5])
    hi = torch.tensor([35., 2.0, 3.0, 3.5, 5.5, 0.5, 0.3, 9.0, 1.5])
    vals = lo + torch.rand(b, k, 9, generator=g) * (hi - lo)
    return {n: vals[..., i] for i, n in enumerate(DRIVER_NAMES)}


@pytest.mark.parametrize("phys", [dict(), dict(sumo_fidelity=True, merge_cal=True),
                                  dict(open_road=False)])
@pytest.mark.parametrize("with_params", [False, True])
def test_baseline_physics_equal(phys, with_params):
    leg = legacy("model")
    old = leg.WmSModel(physics=leg.PhysicsConfig(**phys))
    cfg = PhysicsConfig(**phys)
    state = inputs.make_state(3)
    kin = state_kin(state)
    params = _driver_params(2, inputs.K) if with_params else None
    present = state["present"]
    assert torch.equal(old.physics_accel(dict(state), params),
                       baseline_accel(kin, present, cfg, params))
    assert torch.equal(old.mobil_merge_logits(dict(state), params),
                       mobil_merge_logits(kin, present, cfg, params))


def test_idm_params_shape_and_bound():
    cfg = PhysicsConfig(sumo_fidelity=True)
    v = torch.linspace(0, 30, 8)
    a = idm_accel(v, v * 0.5, torch.full_like(v, 20.0), cfg.idm_params(),
                  accel_min=-9.0, accel_max=3.0)
    assert a.shape == v.shape and bool((a <= 3.0).all())
    assert isinstance(cfg.idm_params(), IDMParams)


def test_lane_context_equal():
    leg = legacy("model")
    old = leg.WmSModel(physics=leg.PhysicsConfig())
    state = inputs.make_state(5)
    kin = state_kin(state)
    cfg = PhysicsConfig()
    base = mobil_merge_logits(kin, state["present"], cfg)
    age = torch.randint(0, 20, kin.shape[:2]).float()
    for params in (None, _driver_params(2, inputs.K)):
        o = old._lane_context(kin, state["present"], base, params, age)
        n = lane_context(kin, state["present"], base, cfg, params, age)
        assert torch.equal(o[0], n[0]) and torch.equal(o[1], n[1])


def test_projection_equal():
    stab = legacy("stability")
    g = torch.Generator().manual_seed(9)
    s = torch.rand(2, 30, generator=g) * 60
    v = torch.rand(2, 30, generator=g) * 40 - 2
    lane = torch.randint(0, 4, (2, 30), generator=g)
    present = torch.rand(2, 30, generator=g) > 0.2
    cav = torch.rand(2, 30, generator=g) > 0.8
    length = torch.full((2, 30), 5.0)
    for project_cavs in (False, True):
        old = stab.FeasibilityProjection(stab.ProjectionConfig(project_cavs=project_cavs))
        new = FeasibilityProjection(ProjectionConfig(project_cavs=project_cavs))
        assert_same(old(s, v, lane, present, length, is_cav=cav),
                    new(s, v, lane, present, length, is_cav=cav), atol=0)
        assert old.stats == new.stats


@pytest.mark.parametrize("conditioned", [True, False])
def test_blackbox_traj_equal(conditioned):
    leg = legacy("blackbox_traj")
    small = dict(d_model=32, n_heads=4, n_layers=2, ffn=64)
    torch.manual_seed(1)
    old = leg.BlackBoxTrajModel(hdv=sys.modules["hdv.config"].HdvConfig(**small),
                                action_conditioned=conditioned)
    torch.manual_seed(1)
    new = BlackBoxTrajModel(hdv=HdvConfig(**small), action_conditioned=conditioned)
    assert list(old.state_dict()) == list(new.state_dict())
    old.eval()
    new.eval()
    state = inputs.make_state()
    hist, mask = inputs.make_history()
    cid = torch.randint(0, 7, (2, inputs.K))
    cmd = torch.randn(2, inputs.K, 2)
    with torch.no_grad():
        kw = dict(history=hist, history_mask=mask, k_steps=4, controller_id=cid,
                  action_command=cmd)
        assert_same(old(dict(state), **kw), new(dict(state), **kw), atol=1e-6)
        teacher = inputs.make_teacher(state, 6)
        assert_same(old.rollout(dict(state), n_steps=6, cav_teacher=teacher,
                                history=hist, history_mask=mask),
                    new.rollout(dict(state), n_steps=6, cav_teacher=teacher,
                                history=hist, history_mask=mask), atol=1e-5)


def test_batched_physics_equal():
    old = legacy("physics_torch")
    g = torch.Generator().manual_seed(11)
    lane = torch.randint(0, 3, (2, 24), generator=g)
    s = torch.rand(2, 24, generator=g).double() * 300 - 100
    s = torch.where(lane == 2, torch.rand(2, 24, generator=g).double() * 70, s)
    v = torch.rand(2, 24, generator=g).double() * 25
    active = torch.rand(2, 24, generator=g) > 0.2
    cav = torch.zeros(2, 24, dtype=torch.bool)
    ro = old.rollout(old.PhysicsState(s, v, lane, active, cav),
                     old.IDMParameters.constant(2, 24), 12)
    rn = batched.rollout(batched.PhysicsState(s, v, lane, active, cav),
                         batched.IDMParameters.constant(2, 24), 12)
    for a, b in zip(ro[0], rn[0]):
        for name in ("s", "v", "lane", "active", "cav"):
            assert torch.equal(getattr(a, name), getattr(b, name))
    assert torch.equal(ro[1][-1].acceleration, rn[1][-1].acceleration)


@pytest.mark.parametrize("cls", ["NumpyWallExecutor", "TorchWallExecutor"])
def test_wall_exec_equal(cls):
    old = legacy("wall_exec")
    mask = np.zeros((1, 6), bool)
    mask[0, :3] = True
    prior = np.ones((1, 6), np.int64)
    pos = np.array([[100., 104., 108., 90., 120., 60.]])
    spd = np.array([[10., 11., 12., 20., 20., 20.]])
    eo = getattr(old, cls)(mask, [10.0], prior_controller=prior)
    en = getattr(wall_exec, cls)(mask, [10.0], prior_controller=prior)
    for t in range(6):
        ro = eo.step(pos, spd, now_s=float(t))
        rn = en.step(pos, spd, now_s=float(t))
        for key in ("owner", "acceleration_mps2", "target_speed_mps", "phase"):
            assert np.array_equal(np.asarray(ro[key]), np.asarray(rn[key]))
        assert ro["events"] == rn["events"]
        pos = pos + spd * 1.0
