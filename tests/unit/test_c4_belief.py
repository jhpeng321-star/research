"""C4 RSU belief state (wms.control.belief): association, propagation, drops, shapes.

``python tests/unit/test_c4_belief.py`` (PYTHONPATH=src) prints the synthetic belief-error
table of BELIEF_NOTES.md (2 mainline lanes, heterogeneous IDM, merge slowdown, no SUMO).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from wms.control.belief import (
    BeliefConfig,
    BeliefTracker,
    CavReport,
    Detection,
    RsuDetection,
    RsuZones,
    detection_mask,
    hold_speed_cav_controller,
    idm_propagate,
    oracle_mask_state,
    reports_from_state,
    rsu_detections_from_state,
    wms_step_propagator,
)
from wms.core.constants import HISTORY_F, HISTORY_H, KIN_F

ROAD = (-2645.67, 553.33)


# ----------------------------------------------------------------------- synthetic truth
class TruthSim:
    """Two mainline lanes, heterogeneous IDM drivers, inflow at the upstream end, exit at
    553.33 m, and a temporary slowdown in the merge zone (t in [20, 70) s) that sends a
    wave upstream.  No lane changes."""

    def __init__(self, penetration: float, seed: int = 0, spacing_m: float = 35.0,
                 v_init: float = 22.0, slowdown: bool = True):
        self.rng = np.random.default_rng(seed)
        self.p = penetration
        self.t = 0.0
        self.dt = 0.5
        self.slowdown = slowdown
        self.spacing = spacing_m
        self.next_id = 0
        self.veh: dict[int, dict] = {}
        for lane in (0, 1):
            s = ROAD[1] - 5.0 - (lane * spacing_m / 2.0)
            while s > ROAD[0] + 1.0:
                self._add(s, v_init, lane)
                s -= spacing_m

    def _add(self, s, v, lane):
        r = self.rng
        self.veh[self.next_id] = dict(
            s=s, v=v, a=0.0, lane=lane, length=float(r.uniform(4.5, 5.5)),
            v0=float(r.uniform(27.0, 33.0)), T=float(r.uniform(0.9, 1.7)),
            am=float(r.uniform(1.6, 3.0)), b=float(r.uniform(3.0, 4.5)),
            s0=float(r.uniform(2.0, 3.0)), cav=bool(r.random() < self.p))
        self.next_id += 1

    def step(self):
        for lane in (0, 1):
            ids = sorted((i for i, x in self.veh.items() if x["lane"] == lane),
                         key=lambda i: self.veh[i]["s"])
            for k, i in enumerate(ids):
                x = self.veh[i]
                v0 = x["v0"]
                if self.slowdown and 20.0 <= self.t < 70.0 and 0.0 <= x["s"] <= 75.44:
                    v0 = 8.0
                if k + 1 < len(ids):
                    ld = self.veh[ids[k + 1]]
                    gap, lv = max(ld["s"] - ld["length"] - x["s"], 0.3), ld["v"]
                else:
                    gap, lv = 250.0, x["v"]
                des = x["s0"] + max(0.0, x["v"] * x["T"] + x["v"] * (x["v"] - lv)
                                    / (2 * math.sqrt(x["am"] * x["b"])))
                x["a_new"] = float(np.clip(x["am"] * (1 - (x["v"] / v0) ** 4 - (des / gap) ** 2),
                                           -9.0, 3.0))
        for i in list(self.veh):
            x = self.veh[i]
            nv = max(x["v"] + x["a_new"] * self.dt, 0.0)
            x["s"] += 0.5 * (x["v"] + nv) * self.dt
            x["v"], x["a"] = nv, x["a_new"]
            if x["s"] > ROAD[1]:
                del self.veh[i]
        for lane in (0, 1):
            last = min((x["s"] for x in self.veh.values() if x["lane"] == lane), default=1e9)
            if last - ROAD[0] > self.spacing:
                lv = min((x["v"] for x in self.veh.values()
                          if x["lane"] == lane and x["s"] == last), default=22.0)
                self._add(ROAD[0] + 0.5, min(lv, 22.0), lane)
        self.t += self.dt

    def snapshot(self) -> dict[int, dict]:
        return {i: dict(x) for i, x in self.veh.items()}


def make_reports(snap_now, snap_prev, rng, *, p_miss=0.05, noise=True):
    """CAV self reports + nearest same-lane front/rear within 100 m (bumper gap)."""
    reports = []
    for i, x in snap_now.items():
        if not x["cav"] or rng.random() < p_miss:
            continue
        age = 0.5 if (snap_prev is not None and rng.random() < 0.5 and i in snap_prev) else 0.0
        snap = snap_prev if age > 0 else snap_now
        c = snap[i]
        same = [(j, y) for j, y in snap.items() if y["lane"] == c["lane"] and j != i]
        front = rear = None
        ahead = [(y["s"] - y["length"] - c["s"], j, y) for j, y in same if y["s"] > c["s"]]
        behind = [(c["s"] - c["length"] - y["s"], j, y) for j, y in same if y["s"] < c["s"]]
        nz = (lambda sd: float(rng.normal(0.0, sd))) if noise else (lambda sd: 0.0)
        if ahead:
            g, j, y = min(ahead, key=lambda z: z[0])
            if g <= 100.0:
                front = Detection(g + nz(0.3), y["v"] - c["v"] + nz(0.1), debug_id=j)
        if behind:
            g, j, y = min(behind, key=lambda z: z[0])
            if g <= 100.0:
                rear = Detection(g + nz(0.3), y["v"] - c["v"] + nz(0.1), debug_id=j)
        reports.append(CavReport(f"cav{i}", c["s"] + nz(0.2), c["lane"], c["v"] + nz(0.05),
                                 c["a"], age_s=age, length_m=c["length"], front=front,
                                 rear=rear))
    return reports


def make_rsu(snap, rng, zones: RsuZones, noise=True):
    nz = (lambda sd: float(rng.normal(0.0, sd))) if noise else (lambda sd: 0.0)
    return [RsuDetection(x["s"] + nz(0.3), x["lane"], x["v"] + nz(0.1), x["a"], debug_id=i)
            for i, x in snap.items() if zones.contains(x["lane"], x["s"])]


def belief_metrics(tracker: BeliefTracker, snap, region=ROAD):
    hdv = {i: x for i, x in snap.items() if not x["cav"] and region[0] <= x["s"] <= region[1]}
    tracks = [tr for tr in tracker.tracks.values()
              if not tr.is_cav and region[0] <= tr.s <= region[1]]
    errs, good, calib, rep = [], 0, 0, set()
    for tr in tracks:
        x = snap.get(tr.debug_id)
        if x is None or x["lane"] != tr.lane:
            continue
        e = abs(tr.s - x["s"])
        errs.append(e)
        calib += e <= 2.0 * tr.pos_std + 1.0
        if e < 10.0:
            good += 1
            rep.add(tr.debug_id)
    return {"recall": len(rep & set(hdv)) / max(len(hdv), 1),
            "med_err": float(np.median(errs)) if errs else float("nan"),
            "p90_err": float(np.percentile(errs, 90)) if errs else float("nan"),
            "false_frac": (len(tracks) - good) / max(len(tracks), 1),
            "calib": calib / max(len(errs), 1), "n_hdv": len(hdv), "n_tracks": len(tracks)}


def run_synthetic(penetration, rsu, *, seed=0, duration_s=120.0, eval_at=(30.0, 60.0, 120.0),
                  slowdown=True):
    sim = TruthSim(penetration, seed=seed, slowdown=slowdown)
    rng = np.random.default_rng(seed + 1000)
    cfg = BeliefConfig(rsu_sensors=rsu, lane_extents=(ROAD, ROAD, (-378.66, 75.44)),
                       exit_s_m=ROAD[1], max_slots=320)
    tracker = BeliefTracker(cfg)
    rows = {}
    zone = cfg.rsu_zones.by_lane[0][1]
    while sim.t < duration_s - 1e-9:
        sim.step()
        prev = sim.snapshot()          # 0.5 s before the tick (aged reports)
        sim.step()
        now = sim.snapshot()
        reports = make_reports(now, prev, rng)
        tracker.update(sim.t, reports, make_rsu(now, rng, cfg.rsu_zones) if rsu else None)
        if any(abs(sim.t - t) < 1e-6 for t in eval_at):
            m_all = belief_metrics(tracker, now)
            m_zone = belief_metrics(tracker, now, zone)
            m_up = belief_metrics(tracker, now, (ROAD[0], zone[0]))
            switches = sum(tr.debug_switches for tr in tracker.tracks.values())
            rows[sim.t] = {"all": m_all, "zone": m_zone, "upstream": m_up,
                           "coverage": tracker.coverage(window_s=5.0)["fraction_observed"],
                           "switch_rate": switches / max(tracker.stats["associations"], 1)}
    return rows, tracker


# ----------------------------------------------------------------------- unit tests
def _report(cid, s, lane=0, v=20.0, front=None, rear=None, age=0.0):
    return CavReport(cid, s, lane, v, 0.0, age_s=age, front=front, rear=rear)


def test_association_keeps_identity_and_dedupes_cavs():
    """Two CAVs both see the same HDV between them: one track; identity stable."""
    tr = BeliefTracker(BeliefConfig(rsu_sensors=False))
    for k in range(20):
        t = float(k)
        s_hdv = 100.0 + 20.0 * t
        s_c1, s_c2 = s_hdv - 5.0 - 30.0, s_hdv + 40.0      # c1 behind, c2 ahead
        reps = [_report("c1", s_c1, front=Detection(30.0, 0.0, debug_id="h")),
                _report("c2", s_c2, rear=Detection(40.0 - 5.0, 0.0, debug_id="h"))]
        tr.update(t, reps)
    hdv = [x for x in tr.tracks.values() if not x.is_cav]
    assert len(hdv) == 1 and hdv[0].debug_id == "h" and hdv[0].debug_switches == 0
    assert abs(hdv[0].s - (100.0 + 20.0 * 19)) < 1.0
    assert sum(x.is_cav for x in tr.tracks.values()) == 2


def test_association_two_lanes_platoon_low_switch_rate():
    rows, tracker = run_synthetic(0.4, rsu=False, seed=3, duration_s=30.0, eval_at=(30.0,))
    r = rows[30.0]
    assert r["switch_rate"] < 0.02
    assert r["all"]["med_err"] < 2.0
    assert tracker.stats["slot_overflow"] == 0


def test_cav_detected_as_neighbour_is_not_an_hdv():
    tr = BeliefTracker(BeliefConfig(rsu_sensors=False))
    # c1's front neighbour is CAV c2 (gap 20 m, c2 length 5)
    tr.update(0.0, [_report("c1", 0.0, front=Detection(20.0, 0.0)), _report("c2", 25.0)])
    assert all(x.is_cav for x in tr.tracks.values()) and len(tr.tracks) == 2


def test_propagation_moves_tracks_and_grows_uncertainty():
    tr = BeliefTracker(BeliefConfig(rsu_sensors=False))
    tr.update(0.0, [_report("c1", 0.0, v=20.0, front=Detection(50.0, 2.0, debug_id="h"))])
    h = next(x for x in tr.tracks.values() if not x.is_cav)
    s0, sd0 = h.s, h.pos_std
    stds = []
    for k in range(1, 6):          # CAV keeps reporting far behind (h out of range)
        tr.update(float(k), [_report("c1", -500.0 + 20.0 * k, v=20.0)])
        stds.append(h.pos_std)
    assert h.s > s0 + 5 * 20.0      # moved at ~22 m/s under IDM
    assert sd0 < stds[0] and all(b > a for a, b in zip(stds, stds[1:]))


def test_custom_propagator_is_used():
    calls = []

    def prop(state):
        calls.append(state["s_m"].shape)
        out = dict(state)
        out["s_m"] = state["s_m"] + 1.0
        return out

    tr = BeliefTracker(BeliefConfig(rsu_sensors=False, max_slots=8), propagate=prop)
    tr.update(0.0, [_report("c1", 0.0)])
    tr.update(1.0, [])
    assert len(calls) == 2 and calls[0] == (1, 1)   # compact occupied-track state
    assert abs(next(iter(tr.tracks.values())).s - 2.0) < 1e-6


def test_track_drops_uncertain_exit_negative_and_cav_timeout():
    cfg = BeliefConfig(rsu_sensors=False, max_pos_std_m=5.0, cav_timeout_s=3.0)
    tr = BeliefTracker(cfg)
    tr.update(0.0, [_report("c1", 0.0, front=Detection(50.0, 0.0))])
    for k in range(1, 20):
        tr.update(float(k), [])
    assert tr.stats["drops_uncertain"] == 1 and tr.stats["drops_cav_timeout"] == 1
    assert not tr.tracks
    # exit
    tr = BeliefTracker(BeliefConfig(rsu_sensors=False, exit_s_m=100.0))
    tr.update(0.0, [_report("c1", 90.0, v=30.0)])
    tr.update(1.0, [])
    assert tr.stats["drops_exit"] == 1
    # negative information: CAV sees empty road where the track is predicted
    tr = BeliefTracker(BeliefConfig(rsu_sensors=False))
    tr.update(0.0, [_report("c1", 0.0, v=20.0, front=Detection(30.0, 0.0))])
    tr.update(1.0, [_report("c1", 20.0, v=20.0)])                # nothing ahead now
    assert tr.stats["drops_negative"] == 1
    assert all(x.is_cav for x in tr.tracks.values())


def test_rsu_sensors_add_hdvs_and_ablation_ignores_them():
    zones = RsuZones()
    rsu = [RsuDetection(-100.0, 0, 20.0, debug_id="a"), RsuDetection(-200.0, 2, 15.0),
           RsuDetection(-900.0, 0, 20.0)]   # last one outside the zone: never sent by RSU
    rsu = [d for d in rsu if zones.contains(d.lane, d.s_m)]
    on = BeliefTracker(BeliefConfig(rsu_sensors=True))
    on.update(0.0, [], rsu)
    assert len(on.tracks) == 2
    off = BeliefTracker(BeliefConfig(rsu_sensors=False))
    off.update(0.0, [], rsu)
    assert not off.tracks
    # an RSU frame without a vehicle where a track was predicted drops it
    on.update(1.0, [], [RsuDetection(-185.0, 2, 15.0)])
    assert on.stats["drops_negative"] == 1
    cov = on.coverage(window_s=1.0)
    assert 0.0 < cov["fraction_observed"] < 1.0
    # RSU detection of a reporting CAV is that CAV
    tr = BeliefTracker(BeliefConfig())
    tr.update(0.0, [_report("c1", -50.0)], [RsuDetection(-50.2, 0, 20.0)])
    assert len(tr.tracks) == 1 and next(iter(tr.tracks.values())).is_cav


def test_output_shapes_and_history_features_match_wm():
    from wms.model.history import history_step_features
    from wms.physics.config import PhysicsConfig
    from wms.physics.neighbours import lead_neighbour
    from wms.physics.state import state_kin, state_present

    cfg = BeliefConfig(rsu_sensors=False, max_slots=16)
    tr = BeliefTracker(cfg)
    for k in range(8):
        t = float(k)
        tr.update(t, [_report("c1", 20.0 * t, front=Detection(30.0, 0.0, debug_id="h"),
                              rear=Detection(25.0, 0.0, debug_id="r"))])
    st = tr.state()
    for key in ("s_m", "v_mps", "a_mps2", "lane", "present", "is_cav", "length_m", "exit_s_m"):
        assert st[key].shape == (1, 16)
    assert st["lane"].dtype == torch.long and st["present"].dtype == torch.bool
    assert int(st["present"].sum()) == 3 and int(st["is_cav"].sum()) == 1
    unc = tr.uncertainty()
    assert unc["pos_std_m"].shape == (1, 16)
    hist = tr.history()
    t_len = hist["kin"].shape[0]
    assert hist["kin"].shape == (t_len, 16, KIN_F) and hist["mask"].shape == (t_len, 16)
    assert hist["target_mask"].shape == (t_len - 1, 16) and t_len == 15
    assert hist["available_span_s"] == pytest.approx(7.0)
    feat, mask = tr.driver_history()
    assert feat.shape == (1, 16, HISTORY_H, HISTORY_F) and mask.shape == (1, 16, HISTORY_H)
    kin = state_kin(st)
    gap, _ = lead_neighbour(kin, state_present(st, kin), PhysicsConfig())
    ref = history_step_features(kin, gap.clamp(0.3, 250.0))
    pres = st["present"][0]
    assert torch.allclose(feat[0, pres, -1], ref[0, pres], atol=1e-4)
    assert mask[0, pres, -1].all() and not mask[0, ~pres].any()


def test_state_feeds_wms_rollout_and_step_propagator():
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    model = WmSModel(ModelSpec())
    model.eval()
    tr = BeliefTracker(BeliefConfig(rsu_sensors=False, max_slots=16),
                       propagate=wms_step_propagator(model))   # default hold-speed CAVs
    for k in range(4):
        tr.update(float(k), [_report("c1", 20.0 * k, front=Detection(30.0, 0.0))])
    inp = tr.rollout_inputs()
    with torch.no_grad():
        out = model.rollout(inp["state"], n_steps=4, history=inp["history"],
                            history_mask=inp["history_mask"],
                            cav_controller=hold_speed_cav_controller())
    assert out["positions"].shape[-1] == 16
    assert torch.isfinite(out["positions"]).all()


def test_oracle_mask_and_detection_mask():
    s = torch.tensor([[0.0, 30.0, 60.0, 200.0, 10.0]])
    st = {"s_m": s, "v_mps": torch.full_like(s, 20.0), "a_mps2": torch.zeros_like(s),
          "lane": torch.tensor([[0, 0, 0, 0, 1]]), "present": torch.ones(1, 5, dtype=torch.bool),
          "is_cav": torch.tensor([[False, True, False, False, False]])}
    det = detection_mask(st)
    assert det.tolist() == [[True, False, True, False, False]]
    masked = oracle_mask_state(st, det)
    assert masked["present"].tolist() == [[True, True, True, False, False]]
    reps = reports_from_state(st)
    assert len(reps) == 1 and reps[0].front.debug_id == 2 and reps[0].rear.debug_id == 0
    rsu = rsu_detections_from_state(st)
    assert {d.debug_id for d in rsu} == {0, 1, 2, 3, 4}
    st2 = dict(st, s_m=torch.tensor([[0.0, 30.0, 60.0, 300.0, -900.0]]))
    assert {d.debug_id for d in rsu_detections_from_state(st2)} == {0, 1, 2}


def test_idm_propagate_ramp_merge_and_obstacle():
    s = torch.tensor([[10.0, 70.0]])
    st = {"s_m": s, "v_mps": torch.tensor([[15.0, 0.0]]), "a_mps2": torch.zeros_like(s),
          "lane": torch.tensor([[2, 2]]), "present": torch.ones(1, 2, dtype=torch.bool)}
    out = idm_propagate(st)
    assert out["lane"].tolist() == [[1, 1]]          # free lane 1: both merge
    st["lane"] = torch.tensor([[2, 1]])
    st["s_m"] = torch.tensor([[60.0, 61.0]])
    out = idm_propagate(st)
    assert out["lane"][0, 0] == 2 and out["a_mps2"][0, 0] < 0   # blocked: stays, brakes


def test_synthetic_rsu_improves_recall_and_errors_are_finite():
    lo, _ = run_synthetic(0.05, rsu=False, seed=1, duration_s=30.0, eval_at=(30.0,))
    hi, _ = run_synthetic(0.05, rsu=True, seed=1, duration_s=30.0, eval_at=(30.0,))
    assert hi[30.0]["zone"]["recall"] > 0.95
    assert hi[30.0]["all"]["recall"] > lo[30.0]["all"]["recall"]
    assert math.isfinite(hi[30.0]["all"]["med_err"])


# ----------------------------------------------------------------------- table
def main() -> None:
    print("pen  rsu   t  recall_all recall_zone recall_up med_err p90_err false calib cover "
          "switch")
    for pen in (0.02, 0.05, 0.10, 0.20, 0.40):
        for rsu in (False, True):
            agg: dict = {}
            for seed in (0, 1, 2):
                rows, _ = run_synthetic(pen, rsu, seed=seed)
                for t, r in rows.items():
                    agg.setdefault(t, []).append(r)
            for t, rs in sorted(agg.items()):
                def m(key, sub="all", rs=rs):
                    vals = [r[sub][key] for r in rs if not math.isnan(r[sub][key])]
                    return float(np.mean(vals)) if vals else float("nan")
                print(f"{pen:4.2f} {'on ' if rsu else 'off'} {t:4.0f} "
                      f"{m('recall'):10.3f} {m('recall', 'zone'):11.3f} "
                      f"{m('recall', 'upstream'):9.3f} {m('med_err'):7.2f} {m('p90_err'):7.2f} "
                      f"{m('false_frac'):5.2f} {m('calib'):5.2f} "
                      f"{np.mean([r['coverage'] for r in rs]):5.2f} "
                      f"{np.mean([r['switch_rate'] for r in rs]):6.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"elapsed {time.time() - t0:.0f} s")
