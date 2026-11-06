"""WM-A: LCMOVES 'prog' members in wms.planapi.evaluate (CPU, fresh model, wz3to2 stub road)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

torch = pytest.importorskip("torch")

from wms.core import road as R  # noqa: E402

ANCHOR = 2620.56


@pytest.fixture(scope="module")
def env():
    from wms import planapi as A

    R._reset_for_tests()
    road = R.road_from_descriptor(R.stub_descriptor("wz3to2"))
    with R.road_scope(road):
        h = A.load(None, max_rows=8)
        cavs = [{"id": "m0", "edge": "up", "lane": 0, "x": 2000.0, "v": 20.0},      # closing lane
                {"id": "m1", "edge": "up", "lane": 1, "x": 1900.0, "v": 22.0},
                {"id": "m2", "edge": "up", "lane": 2, "x": 1700.0, "v": 25.0},
                {"id": "blk", "edge": "up", "lane": 1, "x": 2400.0, "v": 20.0}]
        veh = [{"id": "h0", "edge": "up", "lane": 2, "x": 2300.0, "v": 22.0},
               {"id": "h1", "edge": "up", "lane": 0, "x": 2402.0, "v": 20.0},
               {"id": "hd", "edge": "up", "lane": 2, "x": 1500.0, "v": 24.0}]
        st = A.observe(h, {"t0": 0.0, "map": "wz3to2", "cavs": cavs, "vehicles": veh})
        yield A, h, st, road
    R._reset_for_tests()


def run(env, members, H=4.0):
    A, h, st, road = env
    trace = {}
    with R.road_scope(road):
        out = A.evaluate(h, st, [{"id": "a", "kind": "group", "members": members}],
                         K=1, H=H, seed=0, trace=trace)
    ids = trace["ids"]
    speeds = torch.cat(trace["speeds"], 0)[0]          # [T, K] speeds at t0 .. t0 + H - dt
    lanes = torch.cat(trace["lanes"], 0)[0]            # [T, K] lanes after each step
    return out["a"][0], ids, speeds, lanes


def prog(mid, **params):
    return {"id": mid, "role": "prog", "params": params}


def test_supported_roles_are_declared():
    from wms import planapi as A

    assert A.describe()["supported_roles"] == ["hold", "prog"]


def test_lane_change_program_moves_out_of_the_closing_lane_and_holds_speed(env):
    j, ids, speeds, lanes = run(env, [prog("m0", lc_target=1, speed="hold", v_w=15.0, dmax=1.0,
                                           amax=1.0)])
    k = ids.index("m0")
    assert int(lanes[0, k]) == 1                       # WM lane 2 -> 1 on the first tick
    assert torch.allclose(speeds[:4, k], torch.tensor([20.0, 19.5, 19.0, 18.5]))
    assert math.isfinite(j)


def test_blocked_lane_change_waits_for_a_secure_gap(env):
    _j, ids, _speeds, lanes = run(env, [prog("blk", lc_target=0, lc_giveup_x=2650.0)], H=2.0)
    k = ids.index("blk")
    assert (lanes[:, k] == 1).all()                    # target lane 2 (SUMO up_0) occupied by h1


def test_accel_kind_is_capped_by_the_speed_limit(env):
    _j, ids, speeds, _lanes = run(env, [prog("m2", speed="accel", amax=2.0, dmax=3.0)], H=4.0)
    k = ids.index("m2")
    assert torch.allclose(speeds[:8, k], torch.tensor([25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                                                       31.29]))


def test_track_follows_the_reference_and_unobserved_partner_releases(env):
    _j, ids, speeds, _lanes = run(env, [prog("m1", speed="track", track_ref=["m0"], track_off=-30.0,
                                             kp=0.1, kv=0.5, dmax=2.0, amax=2.0)], H=2.0)
    k = ids.index("m1")
    # err = 2000 - 30 - 1900 = 70 -> a = clip(0.1 * 70 + 0.5 * (20 - 22)) = 2.0 for the first second
    assert torch.allclose(speeds[:3, k], torch.tensor([22.0, 23.0, 24.0]))
    _j2, ids2, released, _l2 = run(env, [prog("m1", speed="accel", amax=2.0, dmax=2.0,
                                              release_on_partner_merge="nobody")], H=2.0)
    _j3, _ids3, native, _l3 = run(env, [prog("m1")], H=2.0)
    k = ids2.index("m1")
    assert torch.equal(released[:, k], native[:, k])     # released at once = native motion


def test_merger_releases_after_leaving_the_closing_lane(env):
    kw = dict(lc_target=1, speed="accel", amax=1.0, dmax=1.0)
    _j, ids, merger, lanes = run(env, [prog("m0", merger=True, **kw)], H=3.0)
    _j, _ids, plain, _l = run(env, [prog("m0", **kw)], H=3.0)
    k = ids.index("m0")
    assert int(lanes[0, k]) == 1
    # tick 0: on the closing lane -> +1 m/s^2 for 1 s in both; tick 1: the merger has merged ->
    # released (native) while the plain program keeps accelerating at amax
    assert torch.allclose(merger[:3, k], torch.tensor([20.0, 20.5, 21.0]))
    assert torch.allclose(plain[:5, k], torch.tensor([20.0, 20.5, 21.0, 21.5, 22.0]))
    assert not torch.allclose(merger[3:5, k], plain[3:5, k])


def test_contract_refusals(env):
    A, h, st, road = env
    with R.road_scope(road):
        for members in ([prog("m0", speed="warp", dmax=1.0)], [prog("m0", bogus=1)],
                        [prog("m0", speed="hold", dmax=1.0)], [prog("hd", speed="accel", dmax=1.0)],
                        [prog("m0", lc_target=7)]):
            with pytest.raises(A.WMContractError):
                A.evaluate(h, st, [{"id": "x", "members": members}], K=1, H=1.0)


def test_tau_changes_only_native_motion(env):
    _j, ids, base, _l = run(env, [prog("m1")], H=2.0)
    _j, _ids, tau, _l = run(env, [prog("m1", tau=3.0)], H=2.0)
    k = ids.index("m1")
    assert float(base[0, k]) == float(tau[0, k]) == 22.0
    assert not torch.equal(base[1:, k], tau[1:, k])
