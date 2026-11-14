"""NEWMAP2 (phase newmap2) core for wz3to2: native prefixes with saved states, branch arms, extensions.

One job per process (libsumo):  python nm2_core.py JOB.json  -> JOB out_dir/RESULT.json

Derived from newmap_probe/probe_core.py (same scenario streams, conventions and J definition):
  * 0.5 s physics, 1 s control, time-to-teleport -1, collision.action warn, departSpeed max;
  * HDVs: one textbook vType per vehicle (population.sample_driver, map-namespaced seed), NEVER commanded;
    CAVs: cav_typical, speedMode 27 on insertion / after loadState;
  * CAV commands: slowDown(v, clip(v + a*1 s, 0, allowed), 1 s); release = setSpeed(-1);
    acting members use laneChangeMode 513 (restored on release);
  * J = team delay veh-s per 1 s bin: on-road dt*max(0, 1 - v/31.29) + source waiting dt.
Changes vs probe_core (newmap2 design): control ticks at whole seconds el = t - t0 = 0, 1, ...;
members release at the drop (x >= X_B), at hold end, or when an anchor member passes anchor_release_x;
optional start delay; per-second member tracks and issued commands; state save at el in save_at;
'extend' jobs continue a branch from its saved state without control.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population as pop  # noqa: E402  (verbatim copy of traffic_shepherd/c4/population.py, sha 6af5ee23)

MAP = "wz3to2"
DT = 0.5
SPEED_LIMIT = 31.29
LC_MODE_POLICY = 513
CAV_SPEED_MODE = 27
EPISODE_END = 1800.0
DEMAND_END = 1800.0
X_B = 2696.0                                   # end of lane 0 on edge 'up' (lane drop)
EDGE_OFFSET = {"up": 0.0, ":n1_0": 2696.0, "wz": 2704.0}


def xglob(edge: str, pos: float) -> float:
    return EDGE_OFFSET.get(edge, float("nan")) + pos


# ----------------------------------------------------------------------------- scenario
def rate_profile(kind: str, d: float):
    if kind == "flat":
        return [(0.0, DEMAND_END, d)]
    return [(0.0, 300.0, 0.8 * d), (300.0, 1500.0, d), (1500.0, DEMAND_END, 0.6 * d)]


def requests(spec: dict, seed: int, demand: dict, penetration: float):
    """Poisson departures per stream; identical RNG strings to probe_core (map-namespaced)."""
    out = []
    for stream, sdef in spec["streams"].items():
        rng = random.Random(f"newmap_requests_v1:{MAP}:{seed}:{stream}")
        ordinal = 0
        for begin, end, rate in rate_profile(sdef["profile"], float(demand[stream])):
            t = begin
            while True:
                t += rng.expovariate(rate / 3600.0)
                if t >= end:
                    break
                ident = f"{stream}.{ordinal}"
                cav = random.Random(f"newmap_cav_v1:{MAP}:{seed}:{ident}").random() < penetration
                out.append(dict(id=ident, depart=round(t, 6), route=sdef["route"], cav=cav,
                                lane=sdef.get("depart_lane", "best"), pos=sdef.get("depart_pos", "base")))
                ordinal += 1
    return sorted(out, key=lambda r: (r["depart"], r["id"]))


def write_scenario(root: Path, map_dir: Path, spec: dict, seed: int, demand: dict, penetration: float):
    import xml.etree.ElementTree as ET
    reqs = requests(spec, seed, demand, penetration)
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", pop.vtype_attrs(pop.TEXTBOOK_CAV_VTYPE, dict(pop.TEXTBOOK_CAV_PARAMS)))
    for r in reqs:
        if not r["cav"]:
            # per-vehicle draw written only into the SUMO route file; no truth table produced or read
            ET.SubElement(routes, "vType", pop.vtype_attrs(
                pop.hdv_vtype_id(r["id"]), pop.sample_driver(f"{MAP}:{seed}", r["id"])))
    for rid, edges in spec["routes"]:
        ET.SubElement(routes, "route", {"id": rid, "edges": edges})
    for r in reqs:
        ET.SubElement(routes, "vehicle", {
            "id": r["id"], "type": pop.TEXTBOOK_CAV_VTYPE if r["cav"] else pop.hdv_vtype_id(r["id"]),
            "route": r["route"], "depart": str(r["depart"]), "departLane": r["lane"],
            "departPos": r["pos"], "departSpeed": "max"})
    ET.indent(routes)
    root.mkdir(parents=True, exist_ok=True)
    rou = root / "requests.rou.xml"
    rou.write_text(ET.tostring(routes, encoding="unicode"), encoding="utf-8")
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value=str((map_dir / spec["net_file"]).resolve()))
    ET.SubElement(inp, "route-files", value=rou.name)
    proc = ET.SubElement(cfg, "processing")
    ET.SubElement(proc, "time-to-teleport", value="-1")
    ET.SubElement(proc, "collision.action", value="warn")
    ET.indent(cfg)
    scene = root / "scenario.sumocfg"
    scene.write_text(ET.tostring(cfg, encoding="unicode"), encoding="utf-8")
    return scene, reqs


# ----------------------------------------------------------------------------- sumo
def api_module():
    import libsumo as api
    return api


def start(api, scene: Path, seed: int):
    cmd = ["sumo", "-c", str(scene), "--seed", str(seed), "--step-length", str(DT),
           "--end", str(EPISODE_END + 1), "--no-step-log", "true", "--duration-log.disable", "true",
           "--no-warnings", "true", "--save-state.rng", "true", "--save-state.precision", "17"]
    api.start(cmd)
    return cmd


class Recorder:
    """Per-second team delay, source wait, safety counters (probe_core.Recorder)."""

    def __init__(self, api, reqs, cav_ids, t_start: float, inserted0: int = 0):
        self.api = api
        self.times = sorted(r["depart"] for r in reqs)
        self.cav_ids = cav_ids
        self.inserted = inserted0
        self.road, self.src, self.arr = [], [], []
        self._r = self._s = 0.0
        self._a = 0
        self.collisions = 0
        self.collision_ids = []
        self.teleports = 0
        self.hard_brake_vs = 0.0
        self.min_accel = 0.0
        self.emergency = 0
        self.t = t_start

    def step(self):
        api = self.api
        api.simulationStep()
        self.t = round(api.simulation.getTime(), 3)
        new = list(api.simulation.getDepartedIDList())
        self.inserted += len(new)
        for v in new:
            if v in self.cav_ids:
                api.vehicle.setSpeedMode(v, CAV_SPEED_MODE)
        road = 0.0
        for v in api.vehicle.getIDList():
            sp = api.vehicle.getSpeed(v)
            if sp < SPEED_LIMIT:
                road += 1.0 - sp / SPEED_LIMIT
            a = api.vehicle.getAcceleration(v)
            if a < -3.0:
                self.hard_brake_vs += DT
            if a < -4.5:
                self.emergency += 1
            if a < self.min_accel:
                self.min_accel = a
        due = bisect_right(self.times, self.t)
        backlog = max(0, due - self.inserted)
        self._r += DT * road
        self._s += DT * backlog
        self._a += api.simulation.getArrivedNumber()
        col = list(api.simulation.getCollidingVehiclesIDList())
        if col:
            self.collisions += len(col)
            self.collision_ids.extend(col[:4])
        self.teleports += api.simulation.getStartingTeleportNumber()
        if abs(self.t - round(self.t)) < 1e-6:
            self.road.append(round(self._r, 5))
            self.src.append(round(self._s, 5))
            self.arr.append(self._a)
            self._r = self._s = 0.0
            self._a = 0

    def summary(self):
        return dict(J_road=self.road, J_src=self.src, arrivals=self.arr,
                    J_total=round(sum(self.road) + sum(self.src), 4),
                    collisions=self.collisions, collision_ids=self.collision_ids[:20],
                    teleports=self.teleports, hard_brake_vs=self.hard_brake_vs,
                    min_accel=round(self.min_accel, 3), emergency_steps=self.emergency)


def snapshot(api, cav_ids):
    rows = []
    lane_counts = {}
    sp = []
    q0 = 0
    for v in api.vehicle.getIDList():
        edge = api.vehicle.getRoadID(v)
        lane = api.vehicle.getLaneIndex(v)
        lane_counts[f"{edge}_{lane}"] = lane_counts.get(f"{edge}_{lane}", 0) + 1
        x = xglob(edge, api.vehicle.getLanePosition(v))
        s = api.vehicle.getSpeed(v)
        if v in cav_ids:
            rows.append(dict(id=v, edge=edge, lane=lane, x=round(x, 2), v=round(s, 3)))
        if edge in EDGE_OFFSET and X_B - 300 <= x <= X_B:
            sp.append(s)
        if edge == "up" and lane == 0 and X_B - 500 <= x <= X_B and s < 5.0:
            q0 += 1
    return dict(cavs=rows, lane_counts=lane_counts,
                approach_speed=round(sum(sp) / len(sp), 3) if sp else None, approach_n=len(sp),
                closing_lane_slow=q0)


# ----------------------------------------------------------------------------- jobs
def run_native(job: dict) -> dict:
    api = api_module()
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(job["map_spec"]).read_text())
    scene, reqs = write_scenario(out / "scenario", Path(job["map_spec"]).parent, spec, job["seed"],
                                 job["demand"], job["penetration"])
    cav_ids = frozenset(r["id"] for r in reqs if r["cav"])
    cmd = start(api, scene, job["seed"])
    rec = Recorder(api, reqs, cav_ids, 0.0)
    saves = sorted(float(t) for t in job.get("save_times", []))
    snaps = {}
    states = out / "states"
    states.mkdir(exist_ok=True)
    wall0 = time.time()
    t_end = float(job["t_end"])
    diag = []
    while rec.t < t_end - 1e-9:
        rec.step()
        if abs(rec.t / 10.0 - round(rec.t / 10.0)) < 1e-9:
            s = snapshot(api, frozenset())
            diag.append((rec.t, s["approach_speed"], s["approach_n"], s["closing_lane_slow"]))
        if saves and abs(rec.t - saves[0]) < 1e-6:
            tag = f"t{int(saves[0])}"
            api.simulation.saveState(str(states / f"{tag}.xml.gz"))
            s = snapshot(api, cav_ids)
            s["inserted"] = rec.inserted
            snaps[tag] = s
            saves.pop(0)
    res = rec.summary()
    api.close()
    res.update(kind="native", job=job, sumo_cmd=cmd, n_requests=len(reqs), n_cav=len(cav_ids),
               snapshots=snaps, diag=diag, sim_s=t_end, wall_s=round(time.time() - wall0, 2),
               routes_sha256=hashlib.sha256((out / "scenario" / "requests.rou.xml").read_bytes()).hexdigest())
    return res


class Arm:
    """Rule-based member actions.  roles: hold (wall/pair member), pulse (dither)."""

    def __init__(self, api, members):
        self.api = api
        self.members = [dict(m) for m in members]
        self.lc_saved = {}
        self.events = []
        self.issued = {m["id"]: [] for m in self.members}

    def _cmd(self, vid, a):
        api = self.api
        v = api.vehicle.getSpeed(vid)
        allowed = api.vehicle.getAllowedSpeed(vid)
        target = min(max(0.0, v + a * 1.0), allowed)
        api.vehicle.slowDown(vid, target, 1.0)

    def _release(self, m, el, why):
        vid = m["id"]
        if m.get("released"):
            return
        m["released"] = True
        m["release_el"] = el
        try:
            self.api.vehicle.setSpeed(vid, -1)
            if vid in self.lc_saved:
                self.api.vehicle.setLaneChangeMode(vid, self.lc_saved[vid])
        except Exception:
            pass
        self.events.append((el, vid, "release", why))

    def control(self, el: int, live: set, pos: dict):
        api = self.api
        for m in self.members:
            vid = m["id"]
            if m.get("released"):
                continue
            if vid not in live:
                m["released"] = True
                m["release_el"] = el
                self.events.append((el, vid, "left", ""))
                continue
            p = m["params"]
            st = int(p.get("start_s", 0))
            if el < st:
                continue
            if m["role"] == "pulse":
                if el >= st + int(p["dur_s"]):
                    self._release(m, el, "pulse_end")
                    continue
                self._cmd(vid, p["a"])
                self.issued[vid].append((el, p["a"]))
                continue
            # role 'hold'
            x = pos.get(vid, (None, None, float("nan")))[2]
            if el >= st + int(p["hold_s"]):
                self._release(m, el, "hold_end")
                continue
            if x >= float(p.get("release_x", X_B)):
                self._release(m, el, "at_drop")
                continue
            anc = p.get("anchor")
            if anc is not None:
                ax = pos.get(anc)
                if ax is None or ax[0] != "up" or ax[2] >= float(p["anchor_release_x"]):
                    self._release(m, el, "anchor")
                    continue
            if vid not in self.lc_saved:
                self.lc_saved[vid] = api.vehicle.getLaneChangeMode(vid)
                api.vehicle.setLaneChangeMode(vid, LC_MODE_POLICY)
            v = api.vehicle.getSpeed(vid)
            a = max(-float(p["dmax"]), min(float(p["amax"]), (float(p["v_w"]) - v) / 1.0))
            self._cmd(vid, a)
            self.issued[vid].append((el, round(a, 4)))


def _positions(api, ids, live):
    out = {}
    for vid in ids:
        if vid in live:
            e = api.vehicle.getRoadID(vid)
            out[vid] = (e, api.vehicle.getLaneIndex(vid), round(xglob(e, api.vehicle.getLanePosition(vid)), 2),
                        round(api.vehicle.getSpeed(vid), 4))
    return out


def run_branch(job: dict) -> dict:
    """Load the saved state (native t0 state, or an arm's saved state for 'extend') and run H s."""
    api = api_module()
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    native = Path(job["native_dir"])
    spec = json.loads(Path(job["map_spec"]).read_text())
    scene = native / "scenario" / "scenario.sumocfg"
    reqs = requests(spec, job["seed"], job["demand"], job["penetration"])
    cav_ids = frozenset(r["id"] for r in reqs if r["cav"])
    cmd = start(api, scene, job["seed"])
    t_load = float(job["t_load"])
    api.simulation.loadState(str(job["state_file"]))
    assert abs(api.simulation.getTime() - t_load) < 1e-6, api.simulation.getTime()
    live = set(api.vehicle.getIDList())
    for v in live & cav_ids:
        api.vehicle.setSpeedMode(v, CAV_SPEED_MODE)
    rec = Recorder(api, reqs, cav_ids, t_load, inserted0=int(job["inserted_load"]))
    acting = list(job.get("members") or []) + list(job.get("dither") or [])
    arm = Arm(api, acting)
    anchors = [m["params"]["anchor"] for m in acting if m["params"].get("anchor")]
    track = list(dict.fromkeys(list(job.get("track_ids") or []) + [m["id"] for m in acting] + anchors))
    track_s = int(job.get("track_s", 200))
    t0 = float(job.get("t0", t_load))
    saves = sorted(int(e) for e in job.get("save_at", []))
    saved = {}
    tracks = {vid: [] for vid in track}
    wall0 = time.time()
    t_end = t_load + float(job["H"])
    (out / "states").mkdir(exist_ok=True)
    while rec.t < t_end - 1e-9:
        if abs(rec.t - round(rec.t)) < 1e-6:                    # whole second: observe, act
            el = int(round(rec.t - t0))
            live = set(api.vehicle.getIDList())
            pos = _positions(api, track, live)
            if el <= track_s:
                for vid in track:
                    if vid in pos:
                        tracks[vid].append((el,) + pos[vid])
            if saves and el == saves[0] and rec.t > t_load:
                api.simulation.saveState(str(out / "states" / f"e{el}.xml.gz"))
                saved[f"e{el}"] = dict(t=rec.t, inserted=rec.inserted)
                saves.pop(0)
            if acting:
                arm.control(el, live, pos)
        rec.step()
    if saves and abs(rec.t - (t0 + saves[0])) < 1e-6:            # save at the very end
        api.simulation.saveState(str(out / "states" / f"e{saves[0]}.xml.gz"))
        saved[f"e{saves[0]}"] = dict(t=rec.t, inserted=rec.inserted)
    res = rec.summary()
    api.close()
    res.update(kind=job["kind"], job=job, sim_s=float(job["H"]), wall_s=round(time.time() - wall0, 2),
               arm_events=arm.events, issued=arm.issued,
               release_el={m["id"]: m.get("release_el") for m in arm.members},
               tracks=tracks, saved=saved)
    return res


def main(path: str) -> None:
    job = json.loads(Path(path).read_text())
    t = time.time()
    try:
        res = run_native(job) if job["kind"] == "native" else run_branch(job)
        res["status"] = "ok"
    except Exception as exc:  # failure receipt, never silently retried
        import traceback
        res = dict(status="failed", job=job, error=repr(exc), tb=traceback.format_exc())
    res["elapsed_s"] = round(time.time() - t, 2)
    Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(job["out_dir"]) / "RESULT.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
