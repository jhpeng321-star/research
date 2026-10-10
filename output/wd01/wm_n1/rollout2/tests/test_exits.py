"""Synthetic tests for the flag-gated physical-coordinate road exits.

The rollout world's ``s_m`` is the model ``s_down`` coordinate
(``features.geometry.s_down_of`` = physical s - lane offset), while
``params.llen`` is the PHYSICAL lane end.  ``Rollout2Config.physical_exits``
(default OFF) compares ``s_down > llen - off`` so a vehicle that passes the
recorded "on the road" boundary is masked out, mirroring ``cav_plan_ch``.
"""
from __future__ import annotations

import numpy as np
import torch

from rollout2 import baseline
from rollout2.engine import IdeaABundle
from rollout2.state import Rollout2Config, Rollout2State, WorldState
from rollout2.step import step_once
from rollout2.tests._parity_helpers import NativeMover

#: lane 0 (main_left) physical end and offset from the frozen params.
_P = baseline.Params(None)
_END0 = float(_P.llen[0] - _P.off[0])          # ~553.33 model s_down


def _world(rows):
    """rows: list of (lane, s, v) at slot index order."""
    k = len(rows)
    s = torch.zeros(1, k, dtype=torch.float64)
    v = torch.zeros(1, k, dtype=torch.float64)
    lane = torch.full((1, k), -1, dtype=torch.int64)
    present = torch.zeros(1, k, dtype=torch.bool)
    for i, (ln, ss, vv) in enumerate(rows):
        s[0, i] = ss
        v[0, i] = vv
        lane[0, i] = ln
        present[0, i] = True
    return WorldState(
        s_m=s, v_mps=v, a_mps2=torch.zeros(1, k, dtype=torch.float64),
        lane=lane, length_m=torch.full((1, k), 5.0, dtype=torch.float64),
        present=present, is_cav=torch.zeros(1, k, dtype=torch.bool),
        wall_group=torch.full((1, k), -1, dtype=torch.int64),
        slot_id=torch.arange(k).expand(1, k).contiguous())


def _run(world, *, physical_exits, n_steps, t0=0.0):
    bundle = IdeaABundle(ctrl=NativeMover(), hdv=None, arrival=None, corr=None)
    cfg = Rollout2Config(t0_s=t0, physical_exits=physical_exits)
    cfg.state = Rollout2State(world.batch, world.slots)
    actions = torch.full((world.batch, world.slots), -1, dtype=torch.int64)
    from rollout2.state import StrategyActions

    acts = StrategyActions(action_code=actions)
    w = world
    for i in range(n_steps):
        w, _ = step_once(w, acts, bundle=bundle, config=cfg,
                         t_s=t0 + 0.5 * i)
    return w


def test_boundary_is_physical_end_minus_offset():
    # the frozen geometry: model s_down exit ~553 m, NOT the physical 3199 m.
    assert 550.0 < _END0 < 556.0


def test_physical_exits_drops_vehicle_past_lane_end():
    # 550 + ~15 m/step crosses the ~553.33 m boundary on the first step.
    world = _world([(0, 550.0, 30.0)])
    legacy = _run(world, physical_exits=False, n_steps=1)
    fixed = _run(world, physical_exits=True, n_steps=1)
    assert bool(legacy.present[0, 0]) is True
    assert bool(fixed.present[0, 0]) is False


def test_physical_exits_keeps_vehicle_before_boundary():
    world = _world([(0, 100.0, 10.0)])
    fixed = _run(world, physical_exits=True, n_steps=5)
    assert bool(fixed.present[0, 0]) is True


def test_legacy_exit_still_fires_at_physical_length():
    # a vehicle already past the RAW physical length is dropped by both modes.
    world = _world([(0, 3300.0, 5.0)])
    legacy = _run(world, physical_exits=False, n_steps=1)
    fixed = _run(world, physical_exits=True, n_steps=1)
    assert bool(legacy.present[0, 0]) is False
    assert bool(fixed.present[0, 0]) is False


def test_flag_off_is_byte_identical():
    world = _world([(0, 300.0, 25.0), (1, -200.0, 20.0), (2, 60.0, 15.0)])
    a = _run(world, physical_exits=False, n_steps=8)
    b = _run(world, physical_exits=False, n_steps=8)
    assert torch.equal(a.s_m, b.s_m)
    assert torch.equal(a.v_mps, b.v_mps)
    assert torch.equal(a.present, b.present)


def test_long_plan_cav_count_drops_after_exit():
    """The plan-summary count mirrors a dropped rollout mask (no model)."""
    from closedloop.a11_predictor import BatchRollout, long_plan_from_rollout

    n = 240
    pos = np.zeros((n + 1, 1, 1))
    spd = np.full((n + 1, 1, 1), 25.0)
    acc = np.zeros((n, 1, 1))
    lns = np.zeros((n + 1, 1, 1), np.int64)
    msk = np.ones((n + 1, 1, 1), bool)
    msk[41:] = False                       # slot departs after tick 40
    ids = np.zeros((n + 1, 1, 1), np.int64)
    roll = BatchRollout(
        pos=pos, spd=spd, acc=acc, lns=lns, msk=msk, ids=ids,
        length=np.full((1, 1), 5.0), a0=np.zeros((1, 1)))
    plan = long_plan_from_rollout(roll, np.asarray([True]), horizon_s=120.0)
    plan = plan.detach().cpu().numpy().reshape(12, 5)
    assert plan[0, 4] > 0.9                # bin 0 fully alive
    assert plan[1, 4] > 0.9                # bin 1 fully alive
    assert plan[3, 4] == 0.0               # later bins empty
    assert plan[11, 4] == 0.0


def _synth_live(k=4):
    from closedloop.live_state import LiveState

    kin = np.zeros((k, 7), np.float32)
    mask = np.zeros(k, bool)
    is_cav = np.zeros(k, bool)
    vid = [None] * k
    for i in range(k):
        kin[i, 0] = -500.0 + 40.0 * i
        kin[i, 2] = 1.0
        kin[i, 4] = 20.0
        kin[i, 6] = 5.0
        mask[i] = True
        is_cav[i] = i == 0
        vid[i] = f"cav{i}" if i == 0 else f"hdv{i}"
    return LiveState(time_s=400.0, veh_kin=kin, veh_mask=mask, is_cav=is_cav,
                     slot_id=np.where(mask, np.arange(k), -1).astype(np.int64),
                     counter=np.zeros((2, 30), np.float32), slot_vid=vid)


def test_predictor_forwards_physical_exits_flag():
    from closedloop.a11_predictor import (A11Predictor, BatchRollout,
                                          LiveSample)
    from traffic_shepherd.planner.interface import PlanCandidate

    st = _synth_live()
    seen = []

    def fake_rollout(bundle, state, codes, *, n_steps, t0_s=0.0, **kw):
        seen.append(bool(kw.get("physical_exits", False)))
        B = codes.shape[0]
        K = codes.shape[1]
        T = n_steps
        return BatchRollout(
            pos=np.zeros((T + 1, B, K)), spd=np.full((T + 1, B, K), 20.0),
            acc=np.zeros((T, B, K)), lns=np.ones((T + 1, B, K), np.int64),
            msk=np.repeat(state.rollout_mask[None, None], T + 1, 0).repeat(B, 1),
            ids=np.repeat(state.slot_id[None, None], T + 1, 0).repeat(B, 1),
            length=np.full((B, K), 5.0), a0=np.zeros((B, K)))

    def fake_head(bundle, state, plan):
        from model import constants as MC
        B = plan.shape[0]
        return np.zeros((B, len(MC.TAUS), len(MC.QUANTILE_LEVELS)))

    plan = PlanCandidate({"cav0": "pi-external-u"})
    A11Predictor(object(), rollout_fn=fake_rollout, head_fn=fake_head,
                 fast=False).predict_many([LiveSample(st)], [plan])
    assert seen and seen[0] is False
    A11Predictor(object(), rollout_fn=fake_rollout, head_fn=fake_head,
                 fast=False, physical_exits=True).predict_many(
        [LiveSample(st)], [plan])
    assert seen[-1] is True

