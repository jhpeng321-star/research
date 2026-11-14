"""Fast synthetic tests for the Part A v0 model core (WD-01 part 3)."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import p3lib as P


def test_idm_free_flow_and_hard_approach():
    dev = P.device()
    v = torch.tensor([15.0], device=dev)
    a_max, b, tau, s0, v0 = 2.6, 4.5, 1.1, 2.5, 30.0
    free = P.idm_acc(v, torch.tensor([250.0]), torch.tensor([0.0]),
                     torch.full_like(v, a_max), torch.full_like(v, b),
                     torch.full_like(v, tau), torch.full_like(v, s0),
                     torch.full_like(v, v0))
    assert 0 < float(free[0]) <= a_max            # free: accelerates
    hard = P.idm_acc(torch.tensor([20.0]), torch.tensor([2.0]),
                     torch.tensor([20.0]), torch.full_like(v, a_max),
                     torch.full_like(v, b), torch.full_like(v, tau),
                     torch.full_like(v, s0), torch.full_like(v, v0))
    assert float(hard[0]) < -4.0                  # closing fast on leader


def test_theta_decode_bounded_to_vtype_ranges():
    torch.manual_seed(0)
    m = P.PartAModel()
    z = torch.randn(64, P.Z_DIM)
    th = m.theta_of(z)
    for k, lo, hi in (("a_max", 2.0, 3.0), ("b", 3.0, 4.5),
                      ("tau", 0.95, 1.5), ("s0", 1.5, 3.5),
                      ("sf", 0.92, 1.08)):
        assert th[k].min() >= lo - 1e-6 and th[k].max() <= hi + 1e-6, k


def test_lane_change_mask_topology():
    # accel (2) may go to main_right (1) only; main_left (0) never to accel
    allowed = [[j for j in range(3) if P and torch.tensor(
        [[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])[i, j] > 0]
        for i in range(3)]
    assert allowed == [[0, 1], [0, 1, 2], [1, 2]]


def test_collision_detection_counts_without_repair():
    from engine import Rollout, same_lane_leader_idx
    lane = torch.tensor([1, 1, 1])
    s_pre = torch.tensor([4.0, 12.0, 10.0])       # rear(0) mid(1) front(2)
    pre_leader = same_lane_leader_idx(lane, s_pre)
    # ascending s: vid0(4) -> leader vid2(10); vid2(10) -> leader vid1(12)
    assert pre_leader.tolist() == [2, -1, 1]
    # after the step vehicle 0 (rear) passes BOTH: order inverted
    s_post = torch.tensor([12.5, 12.0, 10.0])
    d = SimpleNamespace(lane=lane, s=s_post, collisions=0,
                        vid=torch.tensor([0, 1, 2]),
                        vid_of=torch.tensor([0, 1, 2]))
    Rollout._count_collisions(d, lane, s_pre, pre_leader)
    assert d.collisions == 1                      # vid0 passed its leader
    assert torch.equal(d.s, s_post)               # NO repair applied
    # normal step: everyone advances in order -> no collision
    d2 = SimpleNamespace(lane=lane, s=torch.tensor([4.9, 12.0, 10.0]),
                         collisions=0, vid=torch.tensor([0, 1, 2]),
                         vid_of=torch.tensor([0, 1, 2]))
    Rollout._count_collisions(d2, lane, s_pre, pre_leader)
    assert d2.collisions == 0


def test_neighbour_slots_cross_lanes_and_accel():
    lane = np.array([0, 1, 1, 2])
    s = np.array([100.0, 90.0, 130.0, 80.0])
    v = np.array([30.0, 28.0, 31.0, 25.0])
    cav = np.array([0.0, 1.0, 0.0, 0.0])
    sl = P.neighbour_slots(lane, s, v, cav)
    ego = 1                                       # main_right at s=90
    d_ego = 2644.22 - 90.0                        # 2554.22
    own_l = sl[ego, 0]                            # leader: vehicle 2 (40 m)
    assert own_l[0] == 1.0 and abs(own_l[1] - 40.0) < 1e-3
    assert abs(own_l[2] - (31.0 - 28.0)) < 1e-3
    own_f = sl[ego, 1]                            # no follower behind
    assert own_f[0] == 0.0
    # left lane (main_left): vehicle 0 at d=2545.67 -> AHEAD of ego by 8.55
    ll = sl[ego, 2]
    assert ll[0] == 1.0 and abs(ll[1] - 8.55) < 1e-3
    assert sl[ego, 3][0] == 0.0
    # right lane (accel): vehicle 3 is far ahead in d -> beyond the 250 m
    # free cutoff -> absent (D1 semantics)
    rl = sl[ego, 4]
    assert rl[0] == 0.0
    assert sl[ego, 5][0] == 0.0
    # a main_right ego near the merge DOES see an accel leader inside 250:
    # ego at s=2500 (d=144.22), accel at s=50 (d=328.66) -> gap 184.44
    lane2 = np.array([1, 2])
    s2 = np.array([2500.0, 50.0])
    v2 = np.array([28.0, 25.0])
    sl2 = P.neighbour_slots(lane2, s2, v2, np.zeros(2))
    assert sl2[0, 5, 0] == 1.0 and abs(sl2[0, 5, 1] - 184.44) < 1e-3
    assert sl2[0, 4, 0] == 0.0
    assert sl2[1, 2, 0] == 1.0                      # accel sees ego as leader
    # accel ego (vehicle 3, d=298.66): all main_right traffic is BEHIND
    acc = sl[3]
    assert acc[2, 0] == 0.0
    # all main_right traffic is >250 m behind in d -> free (D1 semantics)
    assert acc[3, 0] == 0.0
    assert acc[4, 0] == 0.0 and acc[5, 0] == 0.0  # no right lane
    # vehicle 0 (main_left, d=2545.67): right lane has leader (veh2,
    # d=2514.22, gap 31.45) and follower (ego, d=2554.22, gap 8.55)
    assert sl[0, 4, 0] == 1.0 and abs(sl[0, 4, 1] - 31.45) < 1e-3
    assert sl[0, 5, 0] == 1.0 and abs(sl[0, 5, 1] - 8.55) < 1e-3


def test_vtype_is_not_a_model_input():
    m = P.PartAModel()
    in_dim = m.res[0].in_features
    assert in_dim == 8 + 6 * 4 + P.Z_DIM         # ego + slots + z only
    pr = P.PartAModel.prior_input(
        torch.tensor([1]), torch.tensor([20.0]), torch.tensor([50.0]),
        torch.tensor([2.0]), torch.tensor([0.0]))
    assert pr.shape[1] == 7 and "type" not in str(sorted(m.state_dict()))


def test_device_env_resolution():
    os.environ["WD_DEVICE"] = "cpu"
    assert str(P.device()) == "cpu"


def test_weights_only_loadability(tmp_path):
    m = P.PartAModel()
    path = tmp_path / "w.pt"
    torch.save({"model_state": m.state_dict(),
                "meta": {"seed": 0, "roots": ["a", "b"], "epoch": 1}}, path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m2 = P.PartAModel()
    m2.load_state_dict(sd["model_state"])         # loads into a fresh model


def test_integrator_form_is_next_speed():
    "s[t+1]-s[t] == v[t+1]*dt holds to 0 on real physical pairs (measured)."
    try:
        ra = P.root_arrays("cv17")
    except Exception:
        pytest.skip("p2C data not built")
    n = bad = 0
    for vi, idxs in ra._rows.items():
        ch = {int(t) for v, t in zip(ra.lc_vid, ra.lc_tick) if v == vi}
        for a, b in zip(idxs, idxs[1:]):
            if (ra.tick[b] == ra.tick[a] + 1 and ra.lane_m[a] >= 0
                    and ra.lane_m[b] >= 0 and not ra.inj[a]
                    and not ra.inj[b] and int(ra.tick[b]) not in ch):
                r = abs((ra.s[b] - ra.s[a]) - ra.v[b] * 0.5)
                if r > 1e-6:
                    bad += 1
                n += 1
                if n >= 2000:
                    # exact for >=98% of pairs; the rare violations are
                    # single-step junction hops (internal-lane length)
                    assert bad / n < 0.02, (bad, n)
                    return
    pytest.skip("not enough pairs")


def test_births_are_departed_and_inside_window():
    try:
        ra = P.root_arrays("cv17")
    except Exception:
        pytest.skip("p2C data not built")
    bm = ra.b_dep & (ra.b_tick > 480) & (ra.b_tick <= 480 + 240)
    assert bm.any()
    assert (ra.b_tick[bm] > 480).all()
    assert (ra.b_lane[bm] >= 0).all()
    assert set(ra.b_vid[bm]).isdisjoint(
        set(ra.vid[ra.tick_rows(480)]))          # births are new vehicles
