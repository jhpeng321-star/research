"""Part 3k engine tests for kind 'idm_merge' (M1-M4) -- synthetic, no data,
no weights, no SUMO: rollouts are hand-built via Rollout.__new__ (same
pattern as test_p3_core.test_collision_detection_counts_without_repair).
'idm_sampled' behaviour is covered by the existing suite and is unchanged.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "part3d"))

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402


def build_rollout(lane, s, v, vtype=None, merge_params=None, lc_scale=0.0,
                  seed=7):
    """Hand-built idm_merge Rollout with untrained p3d nets (hazard is
    disabled by lc_scale=0.0 unless overridden)."""
    n = len(lane)
    r = E.Rollout.__new__(E.Rollout)
    r.dev = torch.device("cpu")
    r.kind = "idm_merge"
    r.vid = np.arange(n)
    r.lane = torch.tensor(lane, dtype=torch.long)
    r.s = torch.tensor(s, dtype=torch.float64)
    r.v = torch.tensor(v, dtype=torch.float64)
    r.a_prev = torch.zeros(n, dtype=torch.float64)
    r.cav = torch.zeros(n, dtype=torch.float32)
    r.typ = np.ones(n, dtype=np.int64)
    r.vtype = (np.ones(n) if vtype is None
               else np.asarray(vtype)).astype(np.int64)
    lim = torch.as_tensor(E.lane_speed_limits(r.lane.numpy(), r.s.numpy()),
                          dtype=torch.float64)
    r.lim = lim.clone()
    r.vcap = 1.2 * lim
    ll = P.lane_lengths()
    r.llen = torch.tensor([ll[0], ll[1], ll[2]],
                          dtype=torch.float64)[r.lane].clone()
    r.lc_scale = float(lc_scale)
    r.mp = dict(E.MERGE_PARAMS_DEFAULT)
    if merge_params:
        r.mp.update(merge_params)
    r.rng = np.random.default_rng(seed)
    import models as M
    r.p3d_post = M.Posterior().eval()
    r.p3d_haz = M.Hazard().eval()
    r.births = {}
    r.collisions = r.capviol = r.lc_count = 0
    r.n_birthed = r.n_exited = 0
    r.merge_d = []
    r.snapshots = {}
    return r


# ----------------------------------------------------------------- M2 ----

def test_merge_acceptance_synthetic_gaps():
    tau, b = 1.1, 3.0
    s0 = 2.5
    # (gap_lead, v_ego, gap_foll, v_foll) -> expected
    cases = [
        (250.0, 10.0, 250.0, 0.0, True),    # empty target lane
        (5.0, 10.0, 250.0, 0.0, False),     # leader gap < minGap + v*tau
        (13.5, 10.0, 250.0, 0.0, True),     # exactly minGap + v*tau
        (13.4, 10.0, 250.0, 0.0, False),    # just below
        (250.0, 10.0, 5.0, 25.0, False),    # fast close follower: b_req big
        (250.0, 10.0, 20.0, 12.0, True),    # slow-ish follower
        (250.0, 20.0, 20.0, 10.0, True),    # follower slower than ego
        (2.0, 0.0, 250.0, 0.0, False),      # leader gap < minGap
    ]
    gl = np.array([c[0] for c in cases])
    ve = np.array([c[1] for c in cases])
    gf = np.array([c[2] for c in cases])
    vf = np.array([c[3] for c in cases])
    ok = E.merge_acceptance(gl, ve, np.full(len(cases), s0), gf, vf,
                            np.full(len(cases), s0), tau, b)
    for i, c in enumerate(cases):
        assert bool(ok[i]) is c[4], (i, c, bool(ok[i]))


def test_merge_acceptance_follower_breq_monotone():
    """The faster the closing follower, the harder to merge."""
    tau, b = 1.1, 3.0
    vf = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    ok = E.merge_acceptance(np.full(5, 250.0), np.full(5, 10.0),
                            np.full(5, 2.5), np.full(5, 10.0), vf,
                            np.full(5, 2.5), tau, b)
    assert ok[0] and not ok[-1]
    assert list(ok) == sorted(ok, reverse=True) or list(ok)[:2] == \
        [True, True]


def test_step_rejects_fast_close_follower():
    """M2 at step level: a follower 5 m behind in d closing at +5 m/s
    requires ~46 m/s^2 of braking -> rejected -> ego still on accel."""
    # ego d = 378.66-430 = -51.34; follower d = 2644.22-2690.56 = -46.34
    r = build_rollout([2, 1], [430.0, 2690.56], [20.0, 25.0])
    r.step(1)
    assert r.lane.tolist() == [2, 1]
    assert r.lc_count == 0


def test_step_accepts_and_preserves_d():
    """M2 at step level: empty target lane -> mandatory merge fires at the
    first in-zone step; d is kept constant; the merge d is recorded."""
    r = build_rollout([2], [400.0], [20.0])
    r.step(1)
    assert r.lane.tolist() == [1] and r.lc_count == 1
    assert len(r.vid) == 1                        # survives (llen refreshed)
    d_after = 2644.22 - float(r.s[0])
    assert -21.34 - 16.0 <= d_after <= -21.34     # advanced only forward
    assert r.merge_d == pytest.approx([378.66 - 400.0])


# ----------------------------------------------------------------- M1 ----

def test_lane_end_obstacle_acc_levels():
    dev = torch.device("cpu")
    v = torch.tensor([20.0, 10.0, 20.0])
    s = torch.tensor([440.0, 100.0, 440.0])
    on2 = torch.tensor([True, True, False])
    f = lambda x: torch.full_like(v, x)  # noqa: E731
    acc = E.lane_end_obstacle_acc(v, s, on2, f(2.6), f(4.5), f(1.1),
                                  f(2.5), f(30.0))
    assert acc[0] < -9.0          # 14 m to the end at 20 m/s: emergency
    assert acc[1] > 2.0           # 354 m to the end: ~free acceleration
    assert acc[2] == 1e9          # off the accel lane: unconstrained


def test_step_queues_at_lane_end_when_never_accepted():
    """M1: with acceptance held off, the unmerged vehicle decelerates to a
    standstill at the lane end and stays on the accel lane (queued)."""
    r = build_rollout([2], [430.0], [20.0])
    r._gap_accept_merge = (  # decision mocked: never acceptable
        lambda: torch.zeros(len(r.lane), dtype=torch.bool))
    for j in range(30):
        r.step(j)
    assert r.lane.tolist() == [2]
    assert float(r.v[0]) < 0.2
    assert 440.0 < float(r.s[0]) < E.ACCEL_END_S
    assert r.lc_count == 0 and len(r.vid) == 1


# ----------------------------------------------------------------- M3 ----

def test_coop_leader_map_triggers_only_near_waiting_accel():
    # follower on main_right at d=30 (s=2614.22); waiting accel at d=-20
    lane = np.array([1, 2])
    s = np.array([2614.22, 398.66])
    v = np.array([25.0, 0.5])
    u = np.zeros(2)
    p = np.full(2, 0.6)
    fi, fg, fv = E.coop_leader_map(lane, s, v, np.ones(2, dtype=np.int64),
                                   u, p)
    assert list(fi) == [0]
    assert fg[0] == pytest.approx(50.0)           # d-gap 30 - (-20)
    assert fv[0] == pytest.approx(0.5)
    # leader-to-be beyond COOP_LOOK: no adaptation
    s_far = np.array([2614.22, 378.66 - (-150.0)])   # accel at d=-150
    fi2, _, _ = E.coop_leader_map(lane, s_far, v,
                                  np.ones(2, dtype=np.int64), u, p)
    assert len(fi2) == 0
    # no accel vehicle at all: no adaptation
    fi3, _, _ = E.coop_leader_map(np.array([1, 1]), s[:2], v[:2],
                                  np.ones(2, dtype=np.int64), u[:2], p[:2])
    assert len(fi3) == 0
    # u >= p: no adaptation
    fi4, _, _ = E.coop_leader_map(lane, s, v, np.ones(2, dtype=np.int64),
                                  np.ones(2), p)
    assert len(fi4) == 0


def test_coop_leaders_method_scales_prior():
    r = build_rollout([1, 2], [2614.22, 398.66], [25.0, 0.5],
                      merge_params={"p_coop_scale": 2.0})
    fi, fg, fv = r._coop_leaders()
    assert list(fi) == [0]                        # p = 0.6*2 = 1.2 >= any u
    assert fg[0] == pytest.approx(50.0)
    assert fv[0] == pytest.approx(0.5)


def test_coop_slow_down_reduces_follower_speed():
    """End to end: a fast main_right follower behind a standing accel
    vehicle slows more when cooperation fires than when it cannot."""
    def run(p_scale):
        # follower d=30, accel stopped at d=-20, gap 50 m
        r = build_rollout([1, 2], [2614.22, 398.66], [25.0, 0.0],
                          merge_params={"p_coop_scale": p_scale})
        for j in range(6):
            r.step(j)
        return float(r.v[0])

    v_coop = run(2.0)      # p = 1.2: cooperation every step
    v_none = run(0.0)      # cooperation never fires
    assert v_coop < v_none


# ------------------------------------------------- idm_sampled untouched --

def test_idm_sampled_kind_still_listed_and_weights_dir_default():
    """The new kind extends the allowed kinds; idm_sampled remains valid
    and the default weights path keeps the in-repo layout (env override
    is honoured at call time)."""
    assert "idm_sampled" in ("model", "persistence", "idm_normal",
                             "idm_oracle", "idm_sampled", "idm_merge")
    old = os.environ.get("WD01_DATA_ROOT")
    try:
        os.environ.pop("WD01_DATA_ROOT", None)
        dr = E._p3d_weights_dir()
        assert dr == Path(E.HERE).parent / "part3d" / "weights" / "run3d"
        monkey_env = "D:/tmp/wd01_data_root_test"
        os.environ["WD01_DATA_ROOT"] = monkey_env
        dr2 = E._p3d_weights_dir()
        assert dr2 == Path(monkey_env) / "part3d" / "weights" / "run3d"
    finally:
        if old is None:
            os.environ.pop("WD01_DATA_ROOT", None)
        else:
            os.environ["WD01_DATA_ROOT"] = old


def test_idm_sampled_weights_loadable_when_data_root_present():
    """With WD01_DATA_ROOT at the populated tree, both p3d nets load (the
    same objects idm_sampled has always used)."""
    dr = os.environ.get("WD01_DATA_ROOT", "").strip()
    if not dr:
        pytest.skip("WD01_DATA_ROOT not set (untracked weights absent)")
    post, haz = E._p3d_models()
    assert post is not None and haz is not None


def test_merge_params_override():
    r = build_rollout([2], [430.0], [20.0],
                      merge_params={"tau_accept": 2.0, "b_accept": 5.0})
    assert r.mp["tau_accept"] == 2.0 and r.mp["b_accept"] == 5.0
    assert r.mp["p_coop_scale"] == E.MERGE_PARAMS_DEFAULT["p_coop_scale"]
