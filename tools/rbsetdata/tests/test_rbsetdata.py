"""CPU tests for tools/rbsetdata (no SUMO, numpy only).

Run: python -m pytest tools/rbsetdata/tests -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import extract_hdv_histories as ex  # noqa: E402
import fit_likelihood as fl  # noqa: E402
import ref_models as rm  # noqa: E402

DT = 0.5
N_TICKS = 121  # 60 s window inclusive at 0.5 s


def _episode_arrays() -> dict:
    """Synthetic c4_sumo_episode/1 tape: leader/follower pair, an LC driver, a
    ramp->mainline merger (via junction), a CAV, one far background vehicle."""
    t = np.arange(N_TICKS) * DT + 240.0
    K = 6
    present = np.ones((N_TICKS, K), bool)
    lane = np.full((N_TICKS, K), -1, np.int8)
    lane_carry = np.zeros((N_TICKS, K), np.int8)
    junction = np.zeros((N_TICKS, K), bool)
    is_cav = np.zeros((N_TICKS, K), bool)
    vid_index = np.full((N_TICKS, K), -1, np.int32)
    length = np.full((N_TICKS, K), 5.0, np.float32)
    vid_index[:, 0] = 0
    vid_index[:, 1] = 1
    vid_index[:, 2] = 2
    vid_index[:, 3] = 3
    vid_index[:, 4] = 4
    vid_index[:, 5] = 5  # background: lane 0 far behind, sparse
    is_cav[:, 4] = True
    length[:, 0] = 6.0

    s = np.zeros((N_TICKS, K), np.float32)
    v = np.zeros((N_TICKS, K), np.float32)
    a = np.zeros((N_TICKS, K), np.float32)

    # slot 0 "lead1": lane 1, constant 20 m/s from s=100
    lane[:, 0] = 1
    v[:, 0] = 20.0
    s[:, 0] = 100.0 + 20.0 * (t - 240.0)

    # slot 1 "fol1": lane 1 IDM follower
    lane[:, 1] = 1
    s1, v1 = 50.0, 20.0
    for i in range(N_TICKS):
        s[i, 1], v[i, 1] = s1, v1
        gap = s[i, 0] - s1 - 6.0
        v2 = float(rm.idm_sumo_step(v1, gap, 20.0, 28.0, 1.1, 1.2, 1.8, 2.0, dt=DT))
        a[i, 1] = (v2 - v1) / DT
        s1 += v2 * DT
        v1 = v2

    # slot 2 "ramp1": ramp lane 2 from s=-100 at 2 m/s; junction transit at
    # tick 50 (s carried), then mainline lane 1 at 3 m/s
    lane[:50, 2] = 2
    lane_carry[:50, 2] = 2
    junction[50, 2] = True
    lane[51:, 2] = 1
    lane_carry[50:, 2] = 2
    v[:50, 2] = 2.0
    s[:50, 2] = -100.0 + 2.0 * (t[:50] - 240.0)
    s[50:, 2] = s[49, 2] + 3.0 * (t[50:] - t[49])
    v[50:, 2] = 3.0

    # slot 3 "lc1": lane 0 until tick 30, then lane 1 at 22 m/s
    lane[:30, 3] = 0
    lane[30:, 3] = 1
    v[:, 3] = 22.0
    s[:, 3] = 300.0 + 22.0 * (t - 240.0)

    # slot 4 "cav1": lane 0
    lane[:, 4] = 0
    v[:, 4] = 21.0
    s[:, 4] = 500.0 + 21.0 * (t - 240.0)

    # slot 5 background: present only in the first 10 ticks (short -> filtered)
    present[10:, 5] = False
    lane[:10, 5] = 0
    v[:10, 5] = 19.0
    s[:10, 5] = -800.0 + 19.0 * (t[:10] - 240.0)

    meta = {"schema": "c4_sumo_episode/1", "seed": 969400, "demand_vph": 1900,
            "penetration": 0.05, "warmup_s": 300.0, "horizon_s": 600.0,
            "controller": "native", "k_slots": K, "dt_s": 0.5}
    return {
        "time_s": t.astype(np.float64), "s_m": s, "v_mps": v, "a_mps2": a,
        "length_true_m": length, "lane": lane, "lane_carry": lane_carry,
        "present": present, "is_cav": is_cav, "junction": junction,
        "vid_index": vid_index,
        "vehicle_ids": np.array(["lead1", "fol1", "ramp1", "lc1", "cav1", "bg1"]),
        "meta_json": np.array(json.dumps(meta)),
    }


@pytest.fixture()
def episode_npz(tmp_path: Path) -> Path:
    path = tmp_path / "episode.npz"
    np.savez_compressed(path, **_episode_arrays())
    return path


def test_extract_segments_neighbors(episode_npz: Path):
    exr = ex.extract_world(episode_npz, t0_s=300.0, window_s=60.0,
                           min_ticks=20, include_cav=False)
    veh = [s["veh_id"] for s in exr["segs"]]
    assert veh == ["lead1", "fol1", "ramp1", "lc1"]  # CAV + short bg filtered
    ticks = exr["ticks"]
    lead_seg = next(s["seg"] for s in exr["segs"] if s["veh_id"] == "lead1")
    fol_seg = next(s["seg"] for s in exr["segs"] if s["veh_id"] == "fol1")
    fol_rows = ticks[ticks[:, 0] == fol_seg]
    lead_rows = ticks[ticks[:, 0] == lead_seg]
    # follower leader gap = s_lead - s_fol - len_lead(6), DV ~ 0 (steady 20 m/s)
    gap = fol_rows[:, ex.TICK_COLS.index("lead_gap_m")]
    np.testing.assert_allclose(gap, lead_rows[:, 2] - fol_rows[:, 2] - 6.0,
                               rtol=1e-5)
    # follower starts out of equilibrium (44 m gap at T=1.1 -> accelerates)
    assert fol_rows[5, ex.TICK_COLS.index("lead_dv_mps")] < -0.2
    # ...and converges to following the 20 m/s leader (dv -> 0)
    assert np.all(np.abs(fol_rows[-10:, ex.TICK_COLS.index("lead_dv_mps")]) < 0.05)
    # window relative time 0..60
    assert ticks[:, 1].min() == pytest.approx(0.0)
    assert ticks[:, 1].max() == pytest.approx(60.0)
    # lc1: one lane change 0 -> 1 at tick 30, decision features at tick 29
    lc1 = next(s["seg"] for s in exr["segs"] if s["veh_id"] == "lc1")
    lcs = [e for e in exr["lc"] if e["seg"] == lc1]
    assert len(lcs) == 1
    assert (lcs[0]["from"], lcs[0]["to"]) == (0, 1)
    assert lcs[0]["t"] == pytest.approx(30 * DT)
    assert lcs[0]["is_ramp_to_main"] == 0
    # ramp1: junction at tick 50, merge event 2 -> 1 at tick 51
    ramp1 = next(s["seg"] for s in exr["segs"] if s["veh_id"] == "ramp1")
    mgs = [e for e in exr["mg"] if e["seg"] == ramp1]
    assert len(mgs) == 1
    assert mgs[0]["from"] == 2 and mgs[0]["to"] == 1
    assert mgs[0]["is_ramp_merge"] == 1
    assert mgs[0]["t"] == pytest.approx(51 * DT)
    seg_row = next(s for s in exr["segs"] if s["veh_id"] == "ramp1")
    assert seg_row["n_merge"] == 1 and seg_row["n_lc"] == 0
    # lc1 while on lane 0 has no left lane: left gaps NaN at those ticks
    lc1_rows = ticks[ticks[:, 0] == lc1][:30]
    left = lc1_rows[:, ex.TICK_COLS.index("left_lead_gap_m")]
    assert np.all(np.isnan(left))


def test_npz_roundtrip_unique_keys(tmp_path: Path, episode_npz: Path):
    exr = ex.extract_world(episode_npz, 300.0, 60.0, 20, include_cav=True)
    payload, manifest = ex.build_npz([exr], 60.0)
    out = tmp_path / "hist.npz"
    np.savez_compressed(out, **payload)
    hist = fl.load_hist(out)
    # event columns must NOT overwrite tick columns (flat namespace)
    n_ticks = exr["ticks"].shape[0]
    assert hist["t"].size == n_ticks and hist["seg_id"].size == n_ticks
    assert hist["lc_t"].size == len(exr["lc"])
    assert hist["mg_t"].size == len(exr["mg"])
    assert str(hist["schema"]) == ex.SCHEMA
    assert manifest["n_segments"] == 5  # include_cav -> cav1 present


def test_idm_reference_behaviour():
    # near-free road: v = v0, dv = 0, gap >> s* -> ~zero acceleration
    v0, T, a, b, s0 = 30.0, 1.2, 1.0, 1.5, 2.0
    v_next = rm.idm_sumo_step(v0, 10.0 * (v0 * T + s0), v0, v0, T, a, b, s0, dt=DT)
    assert abs(float(v_next) - v0) < 0.01
    # at the interaction scale (gap = s*) the model brakes
    v_star = rm.idm_sumo_step(v0, v0 * T + s0, v0, v0, T, a, b, s0, dt=DT)
    assert float(v_star) < v0 - a * 0.5 * DT
    # free road accelerates towards v0 (capped at the 31.29 m/s road limit)
    v_free = float(rm.idm_sumo_step(10.0, 1e6, 10.0, 40.0, T, a, b, s0, dt=DT))
    assert 10.0 < v_free <= 10.0 + a * DT + 1e-9
    # blocked: small gap, faster than leader -> strong deceleration
    v_blk = float(rm.idm_sumo_step(20.0, 1.0, 10.0, v0, T, a, b, s0, dt=DT))
    assert v_blk < 20.0 - 3.0
    # vectorised == scalar
    vs = np.array([5.0, 20.0, 30.0])
    vec = rm.idm_sumo_step(vs, vs * T + s0, vs, v0, T, a, b, s0, dt=DT)
    np.testing.assert_allclose(vec, [float(rm.idm_sumo_step(x, x * T + s0, x,
                                                            v0, T, a, b, s0,
                                                            dt=DT))
                                     for x in vs], rtol=1e-12)


def _simulated_samples(params=(28.0, 1.1, 1.2, 1.8, 2.0), n=400):
    """Leader with sinusoidal speed, IDM follower -> CF sample arrays."""
    v0, T, a, b, s0 = params
    dt = DT
    s_l, v_l = 0.0, 20.0
    s_f, v_f = -40.0, 20.0
    vs, gaps, lvs, vns = [], [], [], []
    for i in range(n):
        v_lead = 20.0 + 5.0 * np.sin(2.0 * np.pi * i * dt / 30.0)
        gap = s_l - s_f - 6.0
        vs.append(v_f)
        gaps.append(gap)
        lvs.append(v_lead)
        v_next = float(rm.idm_sumo_step(v_f, gap, v_lead, v0, T, a, b, s0, dt=dt))
        vns.append(v_next)
        s_l += v_lead * dt
        s_f += v_next * dt
        v_f = v_next
        gaps[-1] = gap
    return {"v": np.array(vs), "gap": np.array(gaps), "lead_v": np.array(lvs),
            "v_next": np.array(vns), "dt": dt}


def test_fit_recovers_true_params():
    true = (28.0, 1.1, 1.2, 1.8, 2.0)
    samples = _simulated_samples(true)
    cf_fn = rm.idm_default_fn
    params, nll, sigma = fl.fit_cf(samples, cf_fn, sweeps=4)
    assert sigma < 0.05  # the model can reproduce the tape almost exactly
    assert abs(params["v0"] - true[0]) < 2.5
    assert abs(params["T"] - true[1]) < 0.2
    assert abs(params["a"] - true[2]) < 0.3
    assert abs(params["b"] - true[3]) < 0.6
    assert abs(params["s0"] - true[4]) < 0.6
    # a wrong model must score clearly worse
    wrong_pred = rm.idm_default_fn(samples["v"], samples["gap"], samples["lead_v"],
                                   samples["dt"], **rm.IDM_DEFAULTS)
    wrong_nll, _ = fl.gaussian_nll(samples["v_next"] - np.asarray(wrong_pred))
    assert wrong_nll > nll + samples["v"].size * 0.1


def test_lc_loss_separable():
    # target lane always has bigger gaps -> fitted CE below the stay-prior
    rows = []
    for _ in range(40):
        rows.append((-1, 0, 1, {1: (0.0, 50.0, 0.0, 50.0)}))
    params, ce = fl.fit_lc(rows, sweeps=2)
    assert ce < np.log(2.0)
    assert params["w_gap"] > 0.0
    # degenerate: no decisions -> NaN, no crash
    lp, lce = fl.fit_lc([], 2)
    assert lce != lce


def test_cli_end_to_end(tmp_path: Path, episode_npz: Path):
    out = tmp_path / "hist_cli.npz"
    repo = HERE.parents[2]
    r = subprocess.run(
        [sys.executable, str(repo / "tools" / "rbsetdata" / "extract_hdv_histories.py"),
         "--episodes", str(episode_npz.parent), "--out", str(out),
         "--manifest", str(tmp_path / "man.json"), "--include-cav"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    r2 = subprocess.run(
        [sys.executable, str(repo / "tools" / "rbsetdata" / "fit_likelihood.py"),
         "--hist", str(out), "--out", str(tmp_path / "params.csv"),
         "--mode", "both", "--min-cf-ticks", "30", "--sweeps", "2"],
        capture_output=True, text=True, cwd=str(repo / "tools" / "rbsetdata"))
    assert r2.returncode == 0, r2.stderr
    text = (tmp_path / "params.csv").read_text(encoding="utf-8")
    assert "population" in text and "per_driver" in text
    assert (tmp_path / "params_summary.json").exists()
