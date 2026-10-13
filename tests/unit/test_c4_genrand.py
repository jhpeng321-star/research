"""C4 GENRAND: undirected random CAV-action generator, channels, conversion, paired branches.

Synthetic (no SUMO): the policy on hand-built states, the converter on hand-built episodes.
"""
from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from traffic_shepherd.c4 import genrand as G  # noqa: E402
from traffic_shepherd.c4.recorder import build_episode_arrays  # noqa: E402
from wms.data import c4_episodes as C4  # noqa: E402
from wms.data.action_timeline import CONTROLLERS  # noqa: E402

K = 24
DIRECT = CONTROLLERS.index("direct")


A = functools.partial(G.GenericRandomPolicy, mode="accel")


def make_state(n_cav=12, k=K, lane=None, a=None, v=None):
    present = np.zeros((1, k), bool)
    present[0, :n_cav + 4] = True
    is_cav = np.zeros((1, k), bool)
    is_cav[0, :n_cav] = True
    ln = np.zeros((1, k), np.int64) if lane is None else np.asarray(lane, np.int64)[None]
    acc = np.zeros((1, k)) if a is None else np.asarray(a, np.float64)[None]
    vel = np.full((1, k), 15.0) if v is None else np.broadcast_to(np.asarray(v, float), (k,))[None]
    s_m = (-2000.0 + 7.0 * np.arange(k))[None]      # all within 200 m of each other
    return {"is_cav": is_cav, "present": present, "a_mps2": acc, "lane": ln, "v_mps": vel,
            "s_m": s_m}


def roll(policy, n_ticks, n_cav=12, lane_fn=None, vids=True):
    acc, lanes = [], []
    cur = np.zeros(K, np.int64)
    ids = [f"c{i}" if i < n_cav else None for i in range(K)]
    for _ in range(n_ticks):
        st = make_state(n_cav, lane=cur)
        a, ln = policy(st, with_lanes=True, vehicle_ids=ids if vids else None)
        acc.append(a[0])
        lanes.append(ln[0])
        if lane_fn is not None:
            cur = lane_fn(cur, ln[0])
    return np.asarray(acc), np.asarray(lanes)


# ------------------------------------------------------------------ policy
@pytest.mark.parametrize("mode", ["speed", "accel"])
def test_determinism_per_seed_and_difference_across_seeds(mode):
    a1, l1 = roll(G.GenericRandomPolicy(123, mode=mode), 200)
    a2, l2 = roll(G.GenericRandomPolicy(123, mode=mode), 200)
    a3, _ = roll(G.GenericRandomPolicy(124, mode=mode), 200)
    assert np.array_equal(a1, a2, equal_nan=True) and np.array_equal(l1, l2)
    assert not np.allclose(a1[:, :12], a3[:, :12])
    p = G.GenericRandomPolicy(123, mode=mode)
    roll(p, 50)
    p.reset()
    a4, _ = roll(p, 200)
    assert np.array_equal(a1, a4, equal_nan=True)
    # non-CAV slots are NaN (released / ignored by the env)
    assert np.isnan(a1[:, 12:]).all() and np.isfinite(a1[:, :12]).all()


def test_ranges_holds_and_jerk_and_not_preclipped():
    p = A(7, rho=0.0)
    targets, holds, jerks = [], [], []
    orig = p._new_segment

    def spy(track, target, hold):
        targets.append(target)
        holds.append(hold)
        orig(track, target, hold)
        jerks.append(track["jerk"])
    p._new_segment = spy
    acc, _ = roll(p, 3000, n_cav=12)
    a = acc[:, :12]
    assert a.min() >= G.ACCEL_LOW - 1e-9 and a.max() <= G.ACCEL_HIGH + 1e-9
    # full range is used: commands are not pre-clipped to feasibility (SUMO overrides them)
    assert a.min() < -2.7 and a.max() > 2.3
    assert min(targets) >= -3.0 and max(targets) <= 2.6
    assert min(holds) >= 2 and max(holds) <= 60 and max(holds) >= 40
    # log-uniform: median hold ~ sqrt(2*60) ~ 11 s
    assert 7 <= np.median(holds) <= 16
    assert min(jerks) >= 0.5 and max(jerks) <= 3.0
    # per-tick change bounded by the segment jerk (<= 3 m/s^3 * 1 s)
    assert np.abs(np.diff(a, axis=0)).max() <= 3.0 + 1e-9


def test_new_cav_starts_from_measured_acceleration():
    p = A(5, rho=0.0)
    st = make_state(2, a=np.r_[1.2, -0.7, np.zeros(K - 2)])
    out = p(st, vehicle_ids=["a", "b"] + [None] * (K - 2))
    first_targets = [p.tracks["a"]["target"], p.tracks["b"]["target"]]
    for i, start in enumerate((1.2, -0.7)):
        step = np.clip(first_targets[i] - start, -p.tracks["ab"[i]]["jerk"],
                       p.tracks["ab"[i]]["jerk"])
        assert out[0, i] == pytest.approx(start + step)


def _mean_pair_corr(acc):
    d = np.diff(acc, axis=0)          # changes: removes slow shared drift of levels
    c = np.corrcoef(d.T)
    iu = np.triu_indices(c.shape[0], 1)
    return float(np.nanmean(c[iu]))


def test_shared_latent_correlation_present_only_when_rho_positive():
    a0, _ = roll(A(11, rho=0.0, n_latent=1), 4000)
    a1, _ = roll(A(11, rho=1.0, n_latent=1), 4000)
    c0, c1 = _mean_pair_corr(a0[:, :12]), _mean_pair_corr(a1[:, :12])
    assert abs(c0) < 0.03
    assert c1 > 0.15
    p = A(11, rho=0.0)
    roll(p, 500)
    assert p.events["latent_joins"] == 0


def test_episode_parameters_drawn_in_registered_ranges():
    rhos, nlat, rates = [], [], []
    for s in range(300):
        p = A(s)
        rhos.append(p.rho)
        nlat.append(p.n_latent)
        rates.append(p.lane_rate)
    assert 0.0 <= min(rhos) and max(rhos) <= 1.0 and np.std(rhos) > 0.2
    assert set(nlat) == {1, 2, 3}
    assert min(rates) >= 1 / 600 - 1e-12 and max(rates) <= 1 / 20 + 1e-12
    lr = np.log(rates)
    assert abs(np.median(lr) - 0.5 * (math.log(1 / 600) + math.log(1 / 20))) < 0.4


def test_lane_intent_rate_and_direction():
    rate = 1 / 30
    p = A(3, lane_rate_per_s=rate, rho=0.0)

    n_ticks, n_cav = 6000, 12
    lanes0 = np.ones(K, np.int64)          # all on lane 1: left and right both on-road
    lanes = []
    ids = [f"c{i}" if i < n_cav else None for i in range(K)]
    for _ in range(n_ticks):
        st = make_state(n_cav, lane=lanes0)
        _, ln = p(st, with_lanes=True, vehicle_ids=ids)
        lanes.append(ln[0])
    lanes = np.asarray(lanes)
    expected = n_ticks * n_cav * (1 - math.exp(-rate))
    assert abs(p.events["lane_intents"] - expected) < 4 * math.sqrt(expected)
    req = lanes[:, :n_cav]
    assert set(np.unique(req)) <= {-1, 0, 2}
    assert (req == 0).sum() > 0 and (req == 2).sum() > 0
    assert (lanes[:, n_cav:] == -1).all()
    # intents persist 1-10 s then expire
    runs = []
    for c in range(n_cav):
        r = 0
        for v in req[:, c]:
            if v >= 0:
                r += 1
            elif r:
                runs.append(r)
                r = 0
    # (a new event during an active intent replaces it, so contiguous runs can merge)
    assert min(runs) >= 1 and np.mean(np.asarray(runs) <= 10) > 0.95


def test_lane_intent_clears_when_reached():
    p = A(9, lane_rate_per_s=1 / 20, rho=0.0)
    ids = ["c0"] + [None] * (K - 1)
    cur = np.zeros(K, np.int64)
    seen_reach = False
    for _ in range(2000):
        _, ln = p(make_state(1, lane=cur), with_lanes=True, vehicle_ids=ids)
        if ln[0, 0] >= 0:
            cur = cur.copy()
            cur[0] = ln[0, 0]             # executes immediately
            _, ln2 = p(make_state(1, lane=cur), with_lanes=True, vehicle_ids=ids)
            if p.events["lane_intents"] and ln2[0, 0] == -1:
                seen_reach = True
    assert seen_reach


def test_reseed_branches_differ_but_keep_levels_and_episode_params():
    p = A(42)
    roll(p, 100)
    import copy
    levels = {k: t["cmd"] for k, t in p.tracks.items()}
    b1 = copy.deepcopy(p).reseed(G.arm_seed(42, 400, 1))
    b1b = copy.deepcopy(p).reseed(G.arm_seed(42, 400, 1))
    b2 = copy.deepcopy(p).reseed(G.arm_seed(42, 400, 2))
    assert (b1.rho, b1.n_latent, b1.lane_rate) == (p.rho, p.n_latent, p.lane_rate)
    x1, _ = roll(b1, 60)
    x1b, _ = roll(b1b, 60)
    x2, _ = roll(b2, 60)
    assert np.array_equal(x1, x1b, equal_nan=True)
    assert not np.allclose(x1[:, :12], x2[:, :12])
    # continuity: the first branch command moves from the pre-branch level by <= jerk
    for i in range(12):
        assert abs(x1[0, i] - levels[f"c{i}"]) <= 3.0 + 1e-9
    assert len({G.arm_seed(42, t, j) for t in (360, 420) for j in range(4)}) == 8


def test_slot_keyed_fallback_restarts_reused_slots():
    p = A(1, rho=0.0)
    st = make_state(2)
    p(st)
    assert set(p.tracks) == {0, 1}
    st2 = make_state(1)                    # slot 1 empties
    p(st2)
    assert set(p.tracks) == {0}



# ------------------------------------------------------------- speed mode (default)
def test_speed_mode_is_default_and_draws_mixture_limits_and_targets():
    p = G.GenericRandomPolicy(21, rho=0.0)
    assert p.mode == "speed" and p.params()["mode"] == "speed"
    segs = []
    orig = p._speed_segment

    def spy(track, factor, free, hold):
        orig(track, factor, free, hold)
        segs.append((factor, free, None, hold, track["v_target"], track["alim"], track["dlim"]))
    p._speed_segment = spy
    acc, _ = roll(p, 3000)
    a = acc[:, :12]
    free = np.array([x[1] for x in segs])
    fac = np.array([x[0] for x in segs])[~free]
    assert 0.06 < free.mean() < 0.14
    slow = (fac < 0.8).mean() * (1 - free.mean())
    assert 0.19 < slow < 0.31                                  # ~25 % of all draws
    assert fac.min() >= 0.3 and fac.max() <= 1.15
    # targets scale with the speed limit (not the local flow)
    vt = np.array([x[4] for x in segs])
    assert np.allclose(vt[~free], fac * G.SPEED_LIMIT_MPS)
    assert vt.min() >= 0.3 * G.SPEED_LIMIT_MPS - 1e-9 and vt.max() <= 1.15 * G.SPEED_LIMIT_MPS + 1e-9
    alim, dlim = np.array([x[5] for x in segs]), np.array([x[6] for x in segs])
    assert alim.min() >= 0.5 and alim.max() <= 2.6 and dlim.min() >= 1.0 and dlim.max() <= 3.0
    holds = [x[3] for x in segs]
    assert min(holds) >= 2 and max(holds) <= 60
    # commands inside [-3, 2.6]; at v = 15 m/s both braking and accelerating occur
    assert a.min() >= -3.0 - 1e-9 and a.max() <= 2.6 + 1e-9
    assert a.min() < -1.0 and a.max() > 1.0
    assert np.abs(np.diff(a, axis=0)).max() <= 3.0 + 1e-9


def test_speed_mode_command_tracks_target_within_limits():
    p = G.GenericRandomPolicy(4, rho=0.0)
    ids = ["c0"] + [None] * (K - 1)
    st = make_state(1, v=10.0)
    for _ in range(40):
        out = p(st, vehicle_ids=ids)
        t = p.tracks["c0"]
        if not t["free"]:
            want = np.clip(t["v_target"] - 10.0, -t["dlim"], t["alim"])
            assert t["target"] == pytest.approx(want)
        else:
            assert t["target"] == pytest.approx(2.6)
        assert -3.0 <= out[0, 0] <= 2.6


def test_speed_mode_stuck_cav_redraws_after_20s():
    p = G.GenericRandomPolicy(8, rho=0.0)
    ids = [f"c{i}" for i in range(12)] + [None] * (K - 12)
    for _ in range(25):
        p(make_state(12, v=0.0), vehicle_ids=ids)
    assert p.events.get("stuck_redraws", 0) >= 1
    # no stuck redraws while moving
    q = G.GenericRandomPolicy(8, rho=0.0)
    for _ in range(100):
        q(make_state(12, v=5.0), vehicle_ids=ids)
    assert q.events.get("stuck_redraws", 0) == 0


def roll_kinematic(policy, n_ticks, n_cav=12):
    """Point-mass speeds driven by the commands (v += a * 1 s, clipped to [0, 35])."""
    ids = [f"c{i}" if i < n_cav else None for i in range(K)]
    v = np.full(K, 20.0)
    out = []
    for _ in range(n_ticks):
        a = policy(make_state(n_cav, v=v), vehicle_ids=ids)[0]
        v = np.clip(v + np.nan_to_num(a), 0.0, 35.0)
        out.append(v[:n_cav].copy())
    return np.asarray(out)


def test_speed_mode_shared_latent_correlation():
    v0 = roll_kinematic(G.GenericRandomPolicy(13, rho=0.0, n_latent=1), 3000)
    v1 = roll_kinematic(G.GenericRandomPolicy(13, rho=1.0, n_latent=1), 3000)

    def corr(v):                     # cross-CAV correlation of speeds
        c = np.corrcoef(v.T)
        return float(np.nanmean(c[np.triu_indices(c.shape[0], 1)]))
    assert abs(corr(v0)) < 0.1 and corr(v1) > 0.3


# ----------------------------------------------------- channels and conversion
class F(SimpleNamespace):
    pass


def synth_frames(t_end=100.0, k=8, t_start=0.0, entrants=()):
    """Frames every 0.5 s; vehicle i (0..3) present from the start, CAVs 0 and 1; vehicles in
    ``entrants`` (vid, t_enter, ramp) enter later."""
    frames = []
    for t in np.arange(t_start, t_end + 1e-9, 0.5):
        present = np.zeros(k, bool)
        vid = np.full(k, -1, np.int32)
        lane = np.full(k, -1, np.int8)
        for i in range(4):
            present[i], vid[i], lane[i] = True, i, i % 2
        for slot, (v, te, ramp) in enumerate(entrants, start=4):
            if t >= te:
                present[slot], vid[slot], lane[slot] = True, v, 2 if ramp else 0
        s = np.where(present, -1000.0 + 20.0 * t + 30.0 * np.arange(k), 0.0).astype(np.float32)
        a = np.where(present, 0.3 * np.sin(t + np.arange(k)), 0.0).astype(np.float32)
        frames.append(F(time_s=float(t), s=s, lane=lane, lane_carry=lane.copy(),
                        v=np.where(present, 20.0, 0.0).astype(np.float32), a=a,
                        length_true=np.where(present, 4.5, 0.0).astype(np.float32),
                        present=present, is_cav=present & (np.arange(k) < 2),
                        junction=np.zeros(k, bool), vid_index=vid))
    return frames


def synth_controls(t0, t1, k=8, warmup=10.0):
    controls, lane_c, lane_r = [], [], []
    for t in np.arange(t0, t1, 1.0):
        ctrl = np.zeros(k, np.int8)
        acc = np.zeros(k, np.float32)
        req = np.full(k, np.nan, np.float32)
        lc = np.full(k, -1, np.int8)
        lr = np.full(k, -1, np.int8)
        if t >= warmup:
            ctrl[:2] = DIRECT
            req[:2] = (-2.9, 2.5)                       # requested (policy output)
            acc[:2] = (-1.0, 1.4)                       # issued after the speed-bound clip
            lr[0] = 1
            if int(t) % 3 == 0:
                lc[0] = 1
        controls.append((float(t), ctrl, acc, np.zeros(k, np.float32), req))
        lane_c.append(lc)
        lane_r.append(lr)
    return controls, lane_c, lane_r


def write_ep(root, frames, controls, lane_c, lane_r, meta, manifest_extra=None, extra=None):
    root.mkdir(parents=True)
    vids = [f"v{i}" for i in range(16)]
    arrays = build_episode_arrays(frames, controls, lane_c, lane_r, k=frames[0].present.shape[0],
                                  vehicle_ids=vids, cav_ids={"v0", "v1"}, meta=meta)
    arrays.update(extra or {})
    np.savez_compressed(root / "episode.npz", **arrays)
    (root / "summary.json").write_text("{}", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"status": "complete", **meta,
                                                    **(manifest_extra or {})}),
                                        encoding="utf-8")
    return arrays


BASE_META = dict(seed=961000, demand_vph=2700, penetration=0.2, warmup_s=10.0, horizon_s=90.0,
                 controller="direct", k_slots=8)


def test_command_requested_realized_and_lane_channels_distinct_after_conversion(tmp_path):
    frames = synth_frames(100.0)
    controls, lc, lr = synth_controls(0.0, 100.0)
    write_ep(tmp_path / "col" / "ep", frames, controls, lc, lr, BASE_META,
             {"policy": "generic_random", "genrand": {"rho": 0.3}})
    doc = C4.convert([tmp_path / "col"], tmp_path / "src", val_frac=0.0)
    e = doc["episodes"][0]
    assert e["arm"] == "generic_random" and e["genrand"] == {"rho": 0.3}
    assert set(e["extra_arrays"]) == {"requested_acceleration_mps2", "lane_command_target",
                                      "lane_request_target"}
    d = tmp_path / "src" / e["dir"]
    kin = np.load(d / "veh_kin.npy")
    issued = np.load(d / "issued_acceleration_mps2.npy")
    req = np.load(d / "requested_acceleration_mps2.npy")
    # realized acceleration (states) is NOT the command
    a_real = kin[20::2, :2, 5][:issued[10:].shape[0]]
    assert not np.allclose(a_real, issued[10:, :2])
    assert np.allclose(issued[10:, :2], [-1.0, 1.4]) and np.allclose(req[10:, :2], [-2.9, 2.5])
    assert np.isnan(req[:10]).all()
    assert (np.load(d / "lane_request_target.npy")[10:, 0] == 1).all()
    assert (np.load(d / "lane_command_target.npy")[:, 0] >= 0).sum() < (
        np.load(d / "lane_request_target.npy")[:, 0] >= 0).sum()
    # the window loader's action timeline carries the ISSUED command
    ds = C4.build_c4_datasets(tmp_path / "src", horizon_steps=20, stride_s=10.0)
    rec = ds["train"][0]
    tl = rec["action_timeline"]
    assert np.allclose(np.asarray(tl["acceleration_mps2"])[:, :2], [-1.0, 1.4])
    assert not np.allclose(np.asarray(rec["states"])[1:, :2, 3][::2][:10],
                           np.asarray(tl["acceleration_mps2"])[:, :2])
    assert rec["meta"]["pair_key"] is None
