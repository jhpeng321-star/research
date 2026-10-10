"""Synthetic tests for the flag-gated SUMO-fidelity rollout mode.

The P3 world ``s_m`` is the model ``s_down`` coordinate while the frozen v2g
params geometry is in PHYSICAL lane s, so the nominal merge zone
(``s0z..s1z`` 378.66..454.1) lies beyond lane-2 ``s_down`` (max 75.44) and the
mandatory merge never fires.  ``Rollout2Config.sumo_fidelity`` (default OFF =
byte-identical) translates the params into the ``s_down`` frame and matches
the ``hdv_mix`` SUMO vType IDM means.
"""
from __future__ import annotations

import numpy as np
import torch

from rollout2 import baseline
from rollout2.engine import IdeaABundle
from rollout2.state import (Rollout2Config, Rollout2State, StrategyActions,
                            WorldState)
from rollout2.step import step_once


def _world(rows):
    k = len(rows)
    s = torch.zeros(1, k, dtype=torch.float64)
    v = torch.zeros(1, k, dtype=torch.float64)
    lane = torch.full((1, k), -1, dtype=torch.int64)
    present = torch.zeros((1, k), dtype=torch.bool)
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


def _run(world, *, sumo_fidelity, sumo_idm=True, n_steps=1, t0=0.0):
    bundle = IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None)
    cfg = Rollout2Config(t0_s=t0, sumo_fidelity=sumo_fidelity,
                         sumo_idm=sumo_idm)
    cfg.state = Rollout2State(world.batch, world.slots)
    acts = StrategyActions(
        action_code=torch.full((world.batch, world.slots), -1,
                               dtype=torch.int64))
    w = world
    for i in range(n_steps):
        w, _ = step_once(w, acts, bundle=bundle, config=cfg,
                         t_s=t0 + 0.5 * i)
    return w


def test_model_frame_params_translates_geometry():
    p = baseline.model_frame_params()
    assert p["geometry"]["lane_offsets_m"] == [0.0, 0.0, 0.0]
    # merge zone becomes the shared s_down segment [0, 75.44]
    assert abs(p["merge"]["zone_s0_m"] - 0.0) < 1e-9
    assert abs(p["merge"]["zone_s1_m"] - 75.44) < 1e-9
    # lane-2 segment starts shifted by its offset
    assert abs(p["geometry"]["lane_segment_starts_m"][2][0] + 378.66) < 1e-6
    assert abs(p["geometry"]["lane_segment_starts_m"][2][1] - 0.0) < 1e-9
    # model-frame lane ends
    assert abs(p["geometry"]["lane_lengths_m"][0] - (3199.0 - 2645.67)) < 1e-6
    assert abs(p["geometry"]["lane_lengths_m"][2] - 75.44) < 1e-6
    # default is a pure copy (idm untouched)
    base = baseline.load_params()
    assert p["idm"] == base["idm"]


def test_sumo_idm_mean_matches_vtype_mix():
    m = baseline.sumo_idm_mean()
    assert abs(m["T_s"] - 1.165) < 1e-6
    assert abs(m["a_max_mps2"] - 2.52) < 1e-6
    assert abs(m["b_comf_mps2"] - 4.2) < 1e-6
    assert abs(m["s0_m"] - 2.6) < 1e-6
    assert abs(baseline.sumo_speed_factor_mean() - 0.992) < 1e-6
    p = baseline.model_frame_params(sumo_idm=True)
    assert p["idm"]["T_s"] == m["T_s"]
    assert baseline.model_frame_params(
        sumo_idm=True, speed_factor=0.9)["geometry"][
            "lane_segment_limits_mps"][0][0] < 31.29


def test_nominal_geometry_never_reaches_lane2_merge_zone():
    p = baseline.Params(None)
    # local check on the nominal params: lane-2 s_down max is llen-off
    lane2_down_max = p.llen[2] - p.off[2]
    assert lane2_down_max < p.s0z          # merge never a candidate


def test_fidelity_enables_mandatory_merge():
    # lane-2 ego in the merge window with a large lane-1 gap; the nominal
    # mode cannot even be a merge candidate (its zone is 378.66..454.1).
    world = _world([(2, 40.0, 10.0), (1, 300.0, 30.0)])
    nominal = _run(world, sumo_fidelity=False, n_steps=1)
    fixed = _run(world, sumo_fidelity=True, n_steps=1)
    assert int(nominal.lane[0, 0]) == 2
    assert int(fixed.lane[0, 0]) == 1


def test_fidelity_off_is_byte_identical():
    world = _world([(0, 300.0, 25.0), (1, -200.0, 20.0), (2, 60.0, 15.0)])
    a = _run(world, sumo_fidelity=False, n_steps=8)
    b = _run(world, sumo_fidelity=False, n_steps=8)
    assert torch.equal(a.s_m, b.s_m)
    assert torch.equal(a.v_mps, b.v_mps)
    assert torch.equal(a.present, b.present)


def test_fidelity_lane2_end_obstacle_binds_in_model_frame():
    # A lane-2 vehicle near the model-frame lane end is forced to stop by the
    # standing-obstacle rule, which the nominal (physical s1z) never triggers.
    # A same-position lane-1 blocker makes the mandatory merge unsafe so the
    # vehicle stays on lane 2 and meets the end obstacle.
    world = _world([(2, 70.0, 25.0), (1, 70.0, 25.0)])
    fixed = _run(world, sumo_fidelity=True, n_steps=3)
    nominal = _run(world, sumo_fidelity=False, n_steps=3)
    assert float(fixed.v_mps[0, 0]) < 20.0
    assert float(fixed.v_mps[0, 0]) < float(nominal.v_mps[0, 0]) - 5.0


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


def test_predictor_forwards_sumo_fidelity_flag():
    from closedloop.a11_predictor import (A11Predictor, BatchRollout,
                                          LiveSample)
    from traffic_shepherd.planner.interface import PlanCandidate

    st = _synth_live()
    seen = []

    def fake_rollout(bundle, state, codes, *, n_steps, t0_s=0.0, **kw):
        seen.append((bool(kw.get("sumo_fidelity", False)),
                     bool(kw.get("sumo_idm", True))))
        B, K, T = codes.shape[0], codes.shape[1], n_steps
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
    assert seen[0] == (False, True)
    A11Predictor(object(), rollout_fn=fake_rollout, head_fn=fake_head,
                 fast=False, sumo_fidelity=True, sumo_idm=False).predict_many(
        [LiveSample(st)], [plan])
    assert seen[-1] == (True, False)
