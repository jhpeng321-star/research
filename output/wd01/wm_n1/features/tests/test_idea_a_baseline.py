"""P9 baseline predictions == partA_v2g engine one-step outputs (synthetic).

The acceptance contract: on a synthetic scene, the six ``hdv_base`` channels
and the one-step prediction extras produced by
``features.idea_a.scene_baseline_channels`` equal the frozen
``partA_v2g.engine.PartAV2GEngine`` step output exactly (realised lane
changes, positions, speeds; accelerations equal up to the ``(v'-v)/dt``
roundtrip of the engine snapshot), with ``params_nominal.json`` unchanged.
Scenes keep ``v' > 0`` so the engine's realised ``a_prev`` equals the
commanded baseline acceleration.  Zero SUMO; no data reads.
"""
from __future__ import annotations

import numpy as np
import pytest

from features.idea_a import (HDV_BASE_FIELDS, load_nominal_params,
                             scene_baseline_channels)

K = 8
DT = 0.5


def _scene(rows):
    """rows: (lane, s, v) -> scene arrays on K slots."""
    lane = np.full(K, -1, np.int64)
    s = np.zeros(K)
    v = np.zeros(K)
    length = np.full(K, 5.0)
    present = np.zeros(K, bool)
    for j, (li, sj, vj) in enumerate(rows):
        lane[j], s[j], v[j], present[j] = li, float(sj), float(vj), True
    return lane, s, v, length, present


def _engine_vehicles(rows):
    return {f"v{j}": dict(lane=int(li), s=float(sj), v=float(vj), a=0.0,
                          is_cav=False, length_m=5.0)
            for j, (li, sj, vj) in enumerate(rows)}


def _check_scene(rows, params):
    from partA_v2g.engine import PartAV2GEngine

    lane, s, v, length, present = _scene(rows)
    sc = scene_baseline_channels(lane, s, v, length, present, params=params)
    eng = PartAV2GEngine(params, t0=0.0, vehicles=_engine_vehicles(rows))
    eng.step()
    post = eng.snapshots[1]
    for j, (li, sj, vj) in enumerate(rows):
        vid = f"v{j}"
        e_lane, e_s, e_v, e_a = post[vid]
        assert e_lane == sc["lane_next"][j], (vid, "lane")
        assert e_v == sc["v_next"][j], (vid, "speed")
        assert e_s == sc["s_next"][j], (vid, "pos")
        assert e_a == pytest.approx(sc["idm_accel"][j], abs=1e-9), (vid, "a")
        assert bool(sc["merged"][j]) == (li == 2 and e_lane != 2)
        for name in HDV_BASE_FIELDS:
            assert np.isfinite(sc[name][j]), (vid, name)
    assert np.isfinite(sc["s_next"]).all() and np.isfinite(sc["v_next"]).all()
    return sc


@pytest.fixture(scope="module")
def params():
    return load_nominal_params()


def test_free_road_single_vehicle(params):
    sc = _check_scene([(1, 2500.0, 25.0)], params)
    assert sc["idm_accel"][0] > 0.0                 # free road accelerates


def test_follower_brakes_with_blocked_target(params):
    # lane-1 follower closing on a slow leader; the leader's own left
    # change is refused (lane 0 blocked just ahead) and the follower's
    # left gain is zero (target lane blocked), so the baseline brakes.
    sc = _check_scene([(1, 2000.0, 20.0), (1, 1975.0, 25.0),
                       (0, 2005.0, 20.0)], params)
    assert sc["idm_accel"][1] < 0.0                 # closing on the leader
    assert sc["mobil_gain_left"][1] == 0.0
    assert sc["lc_accept_left"][1] == 0.0
    assert sc["mobil_gain_right"][1] == 0.0         # lane 1: no right change


def test_mandatory_merge_accepted(params):
    sc = _check_scene([(2, 420.0, 20.0)], params)
    assert sc["merged"][0]
    assert sc["lane_next"][0] == 1
    assert sc["merge_urgency"][0] == pytest.approx(
        1.0 - (454.1 - 420.0) / 75.44)


def test_disc_left_accepted(params):
    # front-most-first: the front vehicle takes the open lane-0 gap; the
    # follower then stays (its remaining gain is zero).
    sc = _check_scene([(1, 2500.0, 15.0), (1, 2470.0, 24.0)], params)
    assert sc["lane_next"][0] == 0                  # front moved left
    assert sc["lc_accept_left"][0] == 1.0
    assert sc["mobil_gain_left"][0] >= 0.1 + 0.3    # threshold + bias
    assert sc["lane_next"][1] == 1 and sc["lc_accept_left"][1] == 0.0

def test_accel_end_brake_when_merge_rejected(params):
    # lane-1 follower too close behind the ramp vehicle's target d: the
    # b-safe test refuses the merge; the vehicle brakes at the lane end.
    sc = _check_scene([(2, 440.0, 10.0), (1, 2700.0, 20.0)], params)
    assert not sc["merged"][0]
    assert sc["idm_accel"][0] < 0.0
    assert sc["merge_urgency"][0] == pytest.approx(
        1.0 - (454.1 - 440.0) / 75.44)


def test_mixed_scene_matches_engine(params):
    rows = [(0, 2000.0, 28.0), (0, 1950.0, 25.0), (1, 2500.0, 15.0),
            (1, 2470.0, 24.0), (2, 400.0, 22.0)]
    sc = _check_scene(rows, params)
    assert sc["merge_urgency"][4] > 0.0             # accel lane
    assert sc["merge_urgency"][0] == 0.0            # mainline
    assert sc["idm_accel"][4] != 0.0


def test_absent_and_junction_slots_zero(params):
    lane, s, v, length, present = _scene([(1, 2500.0, 25.0)])
    sc = scene_baseline_channels(lane, s, v, length, present, params=params)
    assert sc["idm_accel"][3] == 0.0 and sc["lane_next"][3] == -1
    present2 = present.copy()
    present2[3] = True                              # present but lane -1
    sc2 = scene_baseline_channels(lane, s, v, length, present2, params=params)
    assert sc2["idm_accel"][3] == 0.0


def test_scene_with_no_vehicles(params):
    sc = scene_baseline_channels(np.full(4, -1, np.int64), np.zeros(4),
                                 np.zeros(4), np.full(4, 5.0),
                                 np.zeros(4, bool), params=params)
    assert sc["idm_accel"].sum() == 0.0


def test_engine_multi_step_matches_scene_replay(params):
    """Three engine steps vs three scene replays from the carried state."""
    from partA_v2g.engine import PartAV2GEngine

    rows = [(1, 2500.0, 20.0), (1, 2460.0, 22.0), (2, 396.0, 21.0)]
    eng = PartAV2GEngine(params, t0=0.0, vehicles=_engine_vehicles(rows))
    cur = [(int(r[0]), float(r[1]), float(r[2])) for r in rows]
    for _ in range(3):
        lane, s, v, length, present = _scene(cur)
        sc = scene_baseline_channels(lane, s, v, length, present,
                                     params=params)
        eng.step()
        post = eng.snapshots[eng.step_i]
        for j in range(len(cur)):
            e_lane, e_s, e_v, _e_a = post[f"v{j}"]
            assert e_lane == sc["lane_next"][j]
            assert e_v == sc["v_next"][j]
            assert e_s == sc["s_next"][j]
        cur = [(int(sc["lane_next"][j]), float(sc["s_next"][j]),
                float(sc["v_next"][j])) for j in range(len(cur))]
