"""CPU tests of the WM-D qualification suite (fakes only; no SUMO, no WM checkpoint)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.wzqual import breakdown as BD
from tools.wzqual import distrib as D
from tools.wzqual import frame as F
from tools.wzqual import traffic as TR
from tools.wzqual.moves import MoveExecutor, arms_for, sumo_view
from tools.wzqual.traj import concat_steps, from_frames, load_wzqual_tape, start_state

from .fakes import line_traj, write_tape


# ---------------------------------------------------------------- frame
def test_frame_lane_mapping_matches_n1_descriptor():
    assert F.wm_lane("up", 0) == 2 and F.wm_lane("up", 1) == 1 and F.wm_lane("up", 2) == 0
    assert F.wm_lane("wz", 0) == 1 and F.wm_lane("wz", 1) == 0
    assert F.wm_lane(":n1_0", 1) == 0
    assert F.x_to_s(2696.0) == pytest.approx(75.44)
    assert F.ROAD_END_S == pytest.approx(1379.44)
    assert F.parse_cell("D3200_p60") == (3200, 0.6) and F.cell_key(3200, 0.6) == "D3200_p60"
    with pytest.raises(ValueError):
        F.wm_lane("ramp", 0)


def test_sumo_view_inverts_projection():
    assert sumo_view(F.x_to_s(1000.0), 2) == ("up", 0, pytest.approx(1000.0))
    assert sumo_view(F.x_to_s(3000.0), 0) == ("wz", 1, pytest.approx(3000.0))


# ---------------------------------------------------------------- traj / tapes
def test_tape_roundtrip_and_start_state(tmp_path):
    p = write_tape(tmp_path / "a" / "TAPE.npz", t_end=400.0, n_veh=40)
    tr = load_wzqual_tape(p)
    assert tr.meta["cell"] == "D3200_p60" and tr.meta["seed"] == 989201
    assert tr.dt == pytest.approx(0.5)
    st = start_state(tr, 300.0, history_s=60.0)
    assert st.world == "989201:D3200_p60" and st.t0 == 300.0
    assert st.history.time_s[-1] == pytest.approx(300.0) and st.history.time_s[0] >= 240.0 - 1e-9
    assert (st.entry_times_s <= 300.0).all()
    # WM frame: closing-lane vehicles carry WM lane 2, positions shifted by the anchor
    fr = st.frame
    assert set(np.unique(fr["lane"])) <= {0, 1, 2}
    assert fr["s"].max() <= F.ROAD_END_S + 1e-6


def test_window_and_frames_roundtrip():
    tr = line_traj([20.0, 10.0], steps=6)
    w = tr.window(1.0, 2.0)
    assert w.n_steps == 3 and w.frame(0)["time_s"] == pytest.approx(1.0)
    frames = [dict(time_s=tr.time_s[k], vehicles=[dict(id=n, s=s, lane=ln, v=v, is_cav=False)
                                                   for n, s, ln, v in zip(f["ids"], f["s"], f["lane"], f["v"])])
              for k in range(tr.n_steps) for f in [tr.frame(k)]]
    back = from_frames(frames)
    assert back.n_steps == tr.n_steps and np.allclose(back.s, tr.s)
    cat = concat_steps([tr.window(0.0, 1.0), tr.window(1.5, 3.0)])
    assert cat.n_steps == tr.n_steps


# ---------------------------------------------------------------- traffic
def test_team_delay_matches_definition():
    tr = line_traj([31.29, 15.645], steps=4)          # second car loses 0.5 per step
    assert TR.team_delay(tr, 0.0, 2.0) == pytest.approx(4 * 0.5 * 0.5)
    tr.backlog = np.array([0, 1, 1, 0, 2], float)
    assert TR.team_delay(tr, 0.0, 2.0) == pytest.approx(1.0 + 0.5 * (1 + 1 + 0 + 2))
    assert TR.team_delay(tr, 0.0, 2.0, cohort=np.array([0])) == pytest.approx(0.5 * 4)
    series = TR.delay_series(tr, 0.0, 2.0)
    assert series.sum() == pytest.approx(TR.team_delay(tr, 0.0, 2.0))


def test_zone_queue_and_throughput():
    slow = line_traj([3.0, 3.0, 3.0], s0=F.DROP_S - 100.0, steps=40)
    z = TR.zone_speed_bins(slow, 0.0, 20.0)
    assert len(z) == 2 and np.allclose(z, 3.0)
    q = TR.queue_bins(slow, 0.0, 20.0)
    assert np.allclose(q["count"], 3.0) and np.allclose(q["closing"], 0.0)
    assert q["extent_m"][0] > 50.0
    fast = line_traj([20.0], s0=F.DETECTOR_S - 5.0, steps=4)
    assert len(TR.crossings(fast, 0.0, 2.0)) == 1
    assert TR.throughput_vph(fast, 0.0, 2.0) == pytest.approx(3600.0 / 2.0)


# ---------------------------------------------------------------- breakdown
def test_rule_fit_separates_regimes_and_is_deterministic():
    rng = np.random.default_rng(0)
    v = np.concatenate([np.exp(rng.normal(np.log(27), 0.05, 400)), np.exp(rng.normal(np.log(6), 0.3, 200))])
    r1, r2 = BD.fit_rule(v), BD.fit_rule(v)
    assert r1 == r2 and not r1.degenerate
    assert 8.0 < r1.boundary_mps < 25.0
    assert BD.fit_rule(np.full(100, 25.0) + rng.normal(0, 0.01, 100)).degenerate
    assert BD.fit_rule([20.0] * 5).degenerate
    assert len(r1.record()["sha256"]) == 64


def test_event_onset_persistence_and_start_class():
    rule = BD.fit_rule(np.concatenate([np.full(50, 27.0), np.full(50, 5.0)]) + np.linspace(0, 0.5, 100))
    bins = [27, 27, 5, 27, 5, 5, 5, 27]
    ev = BD.event(bins, rule)
    assert ev.onset_s == 40.0 and ev.occurred(70.0) and not ev.occurred(60.0)
    assert BD.event([27, None, 27], rule).onset_s == math.inf
    assert BD.start_class([27, 5, 5, 5], rule) and not BD.start_class([5, 5, 27], rule)
    assert not BD.event(bins, rule, broken_at_start=True).occurred(300.0)
    assert BD.recovered([5, 5, 27, 27, 27], rule, 50.0) and not BD.recovered([5, 5, 27, 27, 27], rule, 40.0)


# ---------------------------------------------------------------- moves
def _view(**kw):
    return {k: (F.x_to_s(x), ln, v) for k, (x, ln, v) in kw.items()}


def test_hold_rule_accel_and_releases():
    spec = [dict(id="a", role="hold", params=dict(v_w=22.0, hold_s=3, dmax=1.0, amax=1.0, release_x=F.X_B))]
    ex = MoveExecutor(spec)
    c0 = ex.control(0, _view(a=(1000.0, 1, 26.0)))
    assert c0["a"] == [("lcmode", 513), ("accel", -1.0)]
    assert ex.control(1, _view(a=(1020.0, 1, 21.5)))["a"] == [("accel", 0.5)]
    assert ex.control(3, _view(a=(1050.0, 1, 22.0)))["a"] == [("release", "hold_end")]
    assert ex.control(4, _view(a=(1080.0, 1, 22.0))) == {}
    ex2 = MoveExecutor(spec)
    assert ex2.control(0, _view(a=(2700.0, 1, 22.0)))["a"] == [("release", "release_x")]
    ex3 = MoveExecutor(spec)
    assert ex3.control(0, {}) == {} and ex3.events[0][2] == "left"


def test_anchor_release_and_pulse_and_lane_roles():
    p = dict(v_w=22.0, hold_s=150, dmax=1.0, amax=1.0, release_x=F.X_B, anchor="c", anchor_release_x=2546.0)
    ex = MoveExecutor([dict(id="a", role="hold", params=p)])
    assert ex.control(0, _view(a=(2000.0, 1, 25.0), c=(2600.0, 2, 20.0)))["a"] == [("release", "anchor")]
    pulse = MoveExecutor([dict(id="d", role="pulse", params=dict(a=-1.0, start_s=1, dur_s=2))])
    assert pulse.control(0, _view(d=(500.0, 0, 25.0))) == {}
    assert pulse.control(1, _view(d=(500.0, 0, 25.0)))["d"] == [("accel", -1.0)]
    assert pulse.control(3, _view(d=(500.0, 0, 25.0)))["d"] == [("release", "pulse_end")]
    lt = MoveExecutor([dict(id="b", role="lane_to", params=dict(target_lane=2, giveup_x=2400.0))])
    assert lt.control(0, _view(b=(1000.0, 2, 25.0)))["b"][-1] == ("lane", 0)   # SUMO lane 2 = WM 0
    zm = MoveExecutor([dict(id="z", role="zip_merge", params=dict(merge_x=2500.0, fallback_x=2650.0))])
    assert zm.control(0, _view(z=(2510.0, 2, 10.0)))["z"] == [("lcmode", 512), ("lane", 1)]
    assert zm.control(1, _view(z=(2520.0, 1, 10.0)))["z"] == [("release", "merged")]
    with pytest.raises(ValueError):
        MoveExecutor([dict(id="x", role="teleport", params={})])


def test_arms_for_solos_and_dither():
    m = [dict(id="a", role="hold", params={}), dict(id="b", role="hold", params={})]
    d = [dict(id="z", role="pulse", params=dict(a=-1.0, dur_s=2))]
    arms = arms_for(m, dither=d)
    assert set(arms) == {"base", "group", "solo:a", "solo:b"}
    assert arms["base"] == d and arms["solo:a"] == [m[0]] + d
    assert set(arms_for(m[:1])) == {"base", "group"}


# ---------------------------------------------------------------- distrib helpers
def test_w1_crps_auc_bootstrap():
    assert D.weighted_w1([0, 1], [1, 1], [0, 1], [1, 1]) == pytest.approx(0.0)
    assert D.weighted_w1([0.0], [1], [2.0], [1]) == pytest.approx(2.0)
    assert D.crps_ensemble([3.0], 1.0) == pytest.approx(2.0)
    assert D.crps_ensemble([0.0, 2.0], 1.0) == pytest.approx(0.5)
    assert D.auc([0.9, 0.1], [True, False]) == 1.0 and D.auc([0.5], [True]) is None
    recs = [dict(seed=s, x=float(s % 3)) for s in range(30)]
    ci1 = D.cluster_bootstrap(recs, lambda rs: float(np.mean([r["x"] for r in rs])), b=200)
    ci2 = D.cluster_bootstrap(recs, lambda rs: float(np.mean([r["x"] for r in rs])), b=200)
    assert ci1 == ci2 and ci1[0] <= 1.0 <= ci1[1]


def test_summarize_future_and_rate_metrics(tmp_path):
    p = write_tape(tmp_path / "b" / "TAPE.npz", t_end=700.0, n_veh=200, slow_after=450.0)
    tr = load_wzqual_tape(p)
    zone = TR.zone_speed_bins(tr, 0.0, 700.0)
    rule = BD.fit_rule([z for z in zone if np.isfinite(z)] + [28.0] * 30 + [3.0] * 30)
    s = D.summarize_future(tr, 300.0, 300.0, rule, broken_at_start=False)
    assert s["occ"]["300"] and s["onset_s"] >= 140.0
    assert s["queue_mean"] > 0 and s["throughput_vph"] > 0
    rec = dict(seed=1, cell="c", broken_at_start=False, truth=s, wm=[s, s], ks=[s])
    assert D.rate_diff([rec]) == 0.0 and D.rel_mean_err([rec], "queue_mean") == pytest.approx(0.0)
    assert D.zone_crps([rec], 120.0) == pytest.approx(0.0)
