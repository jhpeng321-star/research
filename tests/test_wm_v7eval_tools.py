"""C4 v7 WM-evaluation tools: episode -> state-record conversion and the HDV metric functions
(tools/c4_episode_states.py, tools/wm_eval_common.py).  No SUMO, no checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import c4_episode_states as E  # noqa: E402
import wm_eval_common as C  # noqa: E402


def _episode(t_count=400, k=4):
    """Synthetic collector episode: 0.5 s ticks; slot 0 an HDV upstream, slot 1 a CAV, slot 2 an
    HDV that leaves at tick 250 (slot reused by a new HDV from tick 260), slot 3 a ramp HDV in a
    junction transit at ticks 200-202."""
    t = np.arange(t_count) * 0.5
    s = np.zeros((t_count, k), np.float32)
    lane = np.zeros((t_count, k), np.int8)
    v = np.zeros((t_count, k), np.float32)
    pres = np.zeros((t_count, k), bool)
    vid = np.full((t_count, k), -1, np.int32)
    s[:, 0], v[:, 0], pres[:, 0], vid[:, 0] = -1000 + 10 * t, 10.0, True, 0
    s[:, 1], v[:, 1], pres[:, 1], vid[:, 1], lane[:, 1] = -900 + 12 * t, 12.0, True, 1, 1
    s[:250, 2], v[:250, 2], pres[:250, 2], vid[:250, 2] = -800 + 15 * t[:250], 15.0, True, 2
    s[260:, 2], v[260:, 2], pres[260:, 2], vid[260:, 2] = -2000.0, 20.0, True, 4
    s[:, 3], v[:, 3], pres[:, 3], vid[:, 3], lane[:, 3] = -300 + 5 * t, 5.0, True, 3, 2
    lane[200:203, 3] = -1
    is_cav = np.zeros((t_count, k), bool)
    is_cav[:, 1] = True
    meta = {"schema": "c4_sumo_episode/1", "seed": 969999, "demand_vph": 2400,
            "penetration": 0.4}
    return {"time_s": t, "s_m": s, "lane": lane, "lane_carry": lane.copy(), "v_mps": v,
            "a_mps2": np.zeros_like(v), "present": pres, "is_cav": is_cav, "vid_index": vid,
            "junction": lane < 0, "vehicle_ids": np.array(["h0", "c1", "h2", "r3", "h4"]),
            "cav_vehicle_ids": np.array(["c1", "c9"]), "meta": meta}


def test_episode_record_layout():
    ep = _episode()
    rec = E.episode_record(ep, t0_s=100.0, arm="pi", horizon_steps=120)
    assert rec["world"] == "w969999m2400p40t100"
    assert rec["arms"] == ["pi"] and rec["t0_s"] == 100.0
    ini = rec["initial_state"]
    assert ini["time_s"] == 100.0
    ids = [r["id"] for r in ini["vehicles"]]
    assert sorted(ids) == ["c1", "h0", "h2", "r3"]
    by = {r["id"]: r for r in ini["vehicles"]}
    assert by["c1"]["is_cav"] and not by["h0"]["is_cav"]
    assert by["h0"]["s"] == pytest.approx(-1000 + 10 * 100.0)
    fut = rec["recorded"]["pi"]
    assert len(fut) == 121 and rec["recorded_horizon_steps"]["pi"] == 120
    assert fut[0]["time_s"] == 100.0 and fut[-1]["time_s"] == 160.0
    # h2 leaves at tick 250 (t = 125 s): it drops out of the future and never revives; the
    # newcomer h4 (post-t0 entrant) is never added
    for fr in fut:
        names = {r["id"] for r in fr["vehicles"]}
        assert "h4" not in names
        assert ("h2" in names) == (fr["time_s"] < 125.0)
    # junction transit of r3 at ticks 200-202 is carried as a lane -1 row
    lanes_r3 = [next(r["lane"] for r in fr["vehicles"] if r["id"] == "r3") for fr in fut[:5]]
    assert lanes_r3[:3] == [-1, -1, -1] and lanes_r3[3] == 2
    h = rec["history"]
    kin = np.asarray(h["kin"])
    assert kin.shape[0] == 121 and kin.shape[2] == 7
    assert h["history_s"] == 60.0 and h["dt_s"] == 0.5 and h["available_span_s"] == 60.0
    assert not h["run_start_limited"]
    assert np.asarray(h["target_mask"]).shape == (120, kin.shape[1])
    col = h["ids"].index("h0")
    assert kin[-1, col, 0] == pytest.approx(-1000 + 10 * 100.0)      # last row = t0
    assert kin[0, col, 0] == pytest.approx(-1000 + 10 * 40.0)        # first row = t0 - 60 s
    assert np.asarray(rec["counter"]).shape == (2, 30)


def test_record_feeds_predictor_input_builders():
    from wms.eval.rollout_inputs import t0_vehicles
    rec = E.episode_record(_episode(), t0_s=100.0, arm="native", horizon_steps=40)
    t0, ids, kin, is_cav = t0_vehicles(rec)
    assert t0 == 100.0 and ids == ["c1", "h0", "h2"]      # r3 is in a junction at t0 -> dropped
    assert is_cav.tolist() == [True, False, False]
    assert kin[1, 0] == pytest.approx(0.0) and kin[1, 4] == pytest.approx(10.0)


def test_record_short_episode_and_t0_identity():
    ep = _episode()
    with pytest.raises(ValueError):
        E.episode_record(ep, t0_s=190.0, arm="native", horizon_steps=240)
    a = E.episode_record(ep, t0_s=50.0, arm="native", horizon_steps=10)["initial_state"]
    b = E.episode_record(ep, t0_s=50.0, arm="pi", horizon_steps=10)["initial_state"]
    assert E.frames_identical(a, b)
    ep2 = _episode()
    ep2["v_mps"][100, 0] += 1.0
    c = E.episode_record(ep2, t0_s=50.0, arm="pi", horizon_steps=10)["initial_state"]
    assert not E.frames_identical(a, c)


def test_find_jobs_and_restricted_paths(tmp_path):
    d = tmp_path / "episodes" / "native" / "s969900_m2400_p0400"
    d.mkdir(parents=True)
    (d / "episode.npz").write_bytes(b"")
    (d / "summary.json").write_text("{}")
    (d / "manifest.json").write_text(json.dumps({"status": "complete"}))
    bad = tmp_path / "episodes" / "native" / "s969901_m2400_p0400"
    bad.mkdir()
    (bad / "episode.npz").write_bytes(b"")
    (bad / "manifest.json").write_text(json.dumps({"status": "running"}))
    assert list(E.find_jobs(tmp_path)) == ["s969900_m2400_p0400"]
    with pytest.raises(SystemExit):
        E.find_jobs(tmp_path / "holdout_x")
    with pytest.raises(SystemExit):
        C.guard("E:/x/sealed/y.json")


def test_metric_functions():
    # 3 vehicles, 5 rows (0..2 s); vehicle 2 is a CAV (excluded by sel)
    pos = np.array([[70, 0, 70], [74, 10, 80], [76, 20, 90], [78, 30, 100], [80, 40, 110]], float)
    vel = np.array([[8, 31.29, 20]] * 5, float)
    msk = np.ones_like(pos, bool)
    hdv = np.array([True, True, False])
    assert C.discharge(pos, msk, hdv, 2.0) == 1                # vehicle 0 crosses 75.44 at row 2
    assert C.discharge(pos, msk, hdv, 0.5) == 0
    msk2 = msk.copy()
    msk2[:2, 0] = False                                         # first seen at 76 m: not counted
    assert C.discharge(pos, msk2, hdv, 2.0) == 0
    # delay: vehicle 0 at 8 m/s for 4 steps, vehicle 1 at the limit (0 deficit)
    assert C.delay_proxy(vel, msk, hdv, 2.0) == pytest.approx(4 * 0.5 * (1 - 8 / 31.29))
    assert C.region_speed(pos, vel, msk, hdv, 2.0, 75.44, 600.0) == pytest.approx(8.0)
    assert np.isnan(C.region_speed(pos, vel, msk, hdv, 2.0, 500.0, 600.0))
    assert C.snapshot_speed(pos, vel, msk, hdv, 0.0, 0.0, 75.44) == pytest.approx((8 + 31.29) / 2)


def test_statistics():
    assert C.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert C.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(C.spearman([1, 2], [1, 2]))
    assert np.isnan(C.spearman([1, 1, 1], [1, 2, 3]))
    m, lo, hi = C.bootstrap_mean(np.arange(10.0), 500)
    assert m == pytest.approx(4.5) and lo < m < hi
    m, lo, hi = C.bootstrap_mean([1.0, 1.0, 5.0, 5.0], 500, groups=[0, 0, 1, 1])
    assert m == pytest.approx(3.0) and lo >= 1.0 and hi <= 5.0
    assert C.parse_ckpt("a=/x/y.pt") == ("a", "/x/y.pt", "ON")
    assert C.parse_ckpt("a=E:/x/y.pt@as_is", "OFF") == ("a", "E:/x/y.pt", "AS_IS")
