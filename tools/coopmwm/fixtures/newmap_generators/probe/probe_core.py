"""NEWMAP probe core: scenario generation, native prefix runs with saved states, and branch arms.

One job per process (libsumo if available, else traci).  Usage:
    python probe_core.py JOB.json            # runs the job, writes JOB's out_dir/RESULT.json

Conventions copied from the project (traffic_shepherd.scenarios / c4.env / c4.branching / ownership):
  * 0.5 s physics, 1 s control, time-to-teleport -1, collision.action warn, departSpeed max;
  * HDVs: one textbook vType per vehicle (c4.population.sample_driver / vtype_attrs, new seeds),
    never commanded;  CAVs: vType cav_typical (population medians), speedMode 27 set on insertion,
    commands as slowDown(v, clip(v + a*1 s, 0, allowed), 1 s) (CommandOwnership direct_target);
    release = setSpeed(v, -1);  lane control during a move = laneChangeMode 513 (LC_MODE_POLICY);
  * J = team delay (veh-s): on-road speed deficit dt*max(0, 1 - v/31.29) + source waiting
    (requested, not yet inserted) dt; recorded per 1 s.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from bisect import bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population as pop  # noqa: E402  (verbatim copy of traffic_shepherd/c4/population.py)

DT = 0.5
SPEED_LIMIT = 31.29
LC_MODE_POLICY = 513
CAV_SPEED_MODE = 27
EPISODE_END = 1800.0
DEMAND_END = 1800.0

EDGE_OFFSET = {
    "wz3to2": {"up": 0.0, ":n1_0": 2696.0, "wz": 2704.0},
    "wz2to1": {"up": 0.0, ":n1_0": 2696.0, "wz": 2704.0},
    "weave": {"up": 0.0, ":j1_0": 2361.17, ":j1_1": 2361.17, "weave": 2364.44,
              "ramp_in": 2045.71, ":j2_0": 2835.56, ":j2_1": 2835.56, "down": 2838.83,
              "ramp_out": 2838.83},
}
BOTTLENECK_X = {"wz3to2": 2696.0, "wz2to1": 2696.0, "weave": 2364.44}


# ----------------------------------------------------------------------------- scenario
def rate_profile(kind: str, d: float):
    if kind == "flat":
        return [(0.0, DEMAND_END, d)]
    # "peak": warm 0-300 s at 0.8 D, peak 300-1500 s at D, 1500-1800 s at 0.6 D
    return [(0.0, 300.0, 0.8 * d), (300.0, 1500.0, d), (1500.0, DEMAND_END, 0.6 * d)]


def requests(map_name: str, spec: dict, seed: int, demand: dict, penetration: float):
    """Poisson departures per stream (project scheme, map-namespaced streams)."""
    out = []
    for stream, sdef in spec["streams"].items():
        rng = random.Random(f"newmap_requests_v1:{map_name}:{seed}:{stream}")
        ordinal = 0
        for begin, end, rate in rate_profile(sdef["profile"], float(demand[stream])):
            t = begin
            while True:
                t += rng.expovariate(rate / 3600.0)
                if t >= end:
                    break
                ident = f"{stream}.{ordinal}"
                route = sdef["route"]
                if isinstance(route, list):
                    u = random.Random(f"newmap_route_v1:{map_name}:{seed}:{ident}").random()
                    route = route[1] if u < float(sdef.get("exit_share", 0.0)) else route[0]
                cav = random.Random(f"newmap_cav_v1:{map_name}:{seed}:{ident}").random() < penetration
                out.append(dict(id=ident, depart=round(t, 6), route=route, cav=cav,
                                lane=sdef.get("depart_lane", "best"),
                                pos=sdef.get("depart_pos", "base")))
                ordinal += 1
    return sorted(out, key=lambda r: (r["depart"], r["id"]))


def write_scenario(root: Path, map_dir: Path, spec: dict, map_name: str, seed: int, demand: dict,
                   penetration: float):
    import xml.etree.ElementTree as ET
    reqs = requests(map_name, spec, seed, demand, penetration)
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", pop.vtype_attrs(pop.TEXTBOOK_CAV_VTYPE,
                                                    dict(pop.TEXTBOOK_CAV_PARAMS)))
    for r in reqs:
        if not r["cav"]:
            # fresh per-vehicle draw for a NEW (map-namespaced) seed; written only into the route
            # file SUMO reads -- no truth table is produced or read by this probe.
            ET.SubElement(routes, "vType", pop.vtype_attrs(
                pop.hdv_vtype_id(r["id"]), pop.sample_driver(f"{map_name}:{seed}", r["id"])))
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
    if os.environ.get("NEWMAP_TRACI", "0") == "1":
        import traci as api
        return api
    try:
        import libsumo as api
    except ImportError:  # pragma: no cover
        import traci as api
    return api


def start(api, scene: Path, seed: int):
    cmd = ["sumo", "-c", str(scene), "--seed", str(seed), "--step-length", str(DT),
           "--end", str(EPISODE_END + 1), "--no-step-log", "true", "--duration-log.disable", "true",
           "--no-warnings", "true", "--save-state.rng", "true", "--save-state.precision", "17"]
    api.start(cmd)
    return cmd


class Recorder:
    """Per-second team delay, source wait, safety counters."""

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
        self.hard_brake_vs = 0.0       # vehicle-seconds with a < -3 m/s^2 (project braking_effort)
        self.min_accel = 0.0
        self.emergency = 0             # a < -4.5 m/s^2 events (vehicle-steps)
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
        if abs(self.t - round(self.t)) < 1e-6:          # close the 1 s bin
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


def snapshot(api, map_name: str, cav_ids):
    off = EDGE_OFFSET[map_name]
    rows = []
    lane_counts = {}
    for v in api.vehicle.getIDList():
        edge = api.vehicle.getRoadID(v)
        lane = api.vehicle.getLaneIndex(v)
        lane_counts[f"{edge}_{lane}"] = lane_counts.get(f"{edge}_{lane}", 0) + 1
        if v in cav_ids:
            rows.append(dict(id=v, edge=edge, lane=lane,
                             x=round(off.get(edge, float("nan")) + api.vehicle.getLanePosition(v), 2),
                             v=round(api.vehicle.getSpeed(v), 3),
                             route=api.vehicle.getRouteID(v)))
    # bottleneck approach speed (last 300 m before the bottleneck, all lanes) -- RSU-detector-like
    xb = BOTTLENECK_X[map_name]
    sp = []
    for v in api.vehicle.getIDList():
        e = api.vehicle.getRoadID(v)
        if e in off:
            x = off[e] + api.vehicle.getLanePosition(v)
            if xb - 300 <= x <= xb:
                sp.append(api.vehicle.getSpeed(v))
    return dict(cavs=rows, lane_counts=lane_counts,
                approach_speed=round(sum(sp) / len(sp), 3) if sp else None, approach_n=len(sp))


# ----------------------------------------------------------------------------- jobs
def run_native(job: dict) -> dict:
    """Native prefix: run 0..t_end, save states + snapshots at job['save_times']."""
    api = api_module()
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(job["map_spec"]).read_text())
    for stream, lane in (job.get("depart_lane") or {}).items():   # job-level override, recorded
        spec["streams"][stream]["depart_lane"] = lane
    for stream, dpos in (job.get("depart_pos") or {}).items():
        spec["streams"][stream]["depart_pos"] = dpos
    scene, reqs = write_scenario(out / "scenario", Path(job["map_spec"]).parent, spec, job["map"],
                                 job["seed"], job["demand"], job["penetration"])
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
            s = snapshot(api, job["map"], frozenset())
            diag.append((rec.t, s["approach_speed"], s["approach_n"], s["lane_counts"]))
        if saves and abs(rec.t - saves[0]) < 1e-6:
            tag = f"t{int(saves[0])}"
            api.simulation.saveState(str(states / f"{tag}.xml.gz"))
            s = snapshot(api, job["map"], cav_ids)
            s["inserted"] = rec.inserted
            snaps[tag] = s
            saves.pop(0)
    res = rec.summary()
    api.close()
    res.update(kind="native", job=job, sumo_cmd=cmd, n_requests=len(reqs), n_cav=len(cav_ids),
               snapshots=snaps, diag=diag, sim_s=t_end, wall_s=round(time.time() - wall0, 2),
               routes_sha256=hashlib.sha256((out / "scenario" / "requests.rou.xml")
                                            .read_bytes()).hexdigest())
    return res


class Arm:
    """Rule-based member actions (one per member)."""

    def __init__(self, api, members, t0: float, map_name: str):
        self.api, self.t0, self.map = api, t0, map_name
        self.members = [dict(m) for m in members]
        self.lc_saved = {}
        self.events = []

    def _cmd(self, vid, a):
        api = self.api
        v = api.vehicle.getSpeed(vid)
        allowed = api.vehicle.getAllowedSpeed(vid)
        target = min(max(0.0, v + a * 1.0), allowed)
        api.vehicle.slowDown(vid, target, 1.0)

    def _lc_on(self, vid):
        if vid not in self.lc_saved:
            self.lc_saved[vid] = self.api.vehicle.getLaneChangeMode(vid)
            self.api.vehicle.setLaneChangeMode(vid, LC_MODE_POLICY)

    def _release(self, m):
        vid = m["id"]
        if m.get("released"):
            return
        m["released"] = True
        try:
            self.api.vehicle.setSpeed(vid, -1)
            if vid in self.lc_saved:
                self.api.vehicle.setLaneChangeMode(vid, self.lc_saved[vid])
        except Exception:  # vehicle left
            pass
        self.events.append((self.t_now, vid, "release"))

    def control(self, t: float, live: set):
        self.t_now = t
        el = t - self.t0           # seconds since t0 (first control tick at el = 1)
        for m in self.members:
            vid = m["id"]
            if m.get("released"):
                continue
            if vid not in live:
                m["released"] = True
                self.events.append((t, vid, "left"))
                continue
            role, p = m["role"], m["params"]
            api = self.api
            if role == "wall":
                if el > p["hold_s"]:
                    self._release(m)
                    continue
                self._lc_on(vid)
                v = api.vehicle.getSpeed(vid)
                a = max(-p["dmax"], min(p["amax"], (p["v_w"] - v) / 1.0))
                self._cmd(vid, a)
            elif role == "gap":
                if el > p["decel_s"] + p["hold_s"]:
                    self._release(m)
                    continue
                self._lc_on(vid)
                self._cmd(vid, -p["decel"] if el <= p["decel_s"] else 0.0)
            elif role == "pulse":                       # noise-floor probe: tiny fixed accel
                if el > p["dur_s"]:
                    self._release(m)
                    continue
                self._cmd(vid, p["a"])
            elif role == "merge":
                lane = api.vehicle.getLaneIndex(vid)
                edge = api.vehicle.getRoadID(vid)
                if el > p["max_s"] or (edge == m["edge"] and lane == p["target_lane"]) \
                        or edge != m["edge"]:
                    self._release(m)
                    continue
                self._lc_on(vid)
                self._cmd(vid, 0.0)                    # hold speed alongside the opening gap
                api.vehicle.changeLane(vid, p["target_lane"], 1.0)
            else:
                raise ValueError(role)


def run_branch(job: dict) -> dict:
    """Load the saved state at t0 and run H s with the arm's members acting."""
    api = api_module()
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    native = Path(job["native_dir"])
    spec = json.loads(Path(job["map_spec"]).read_text())
    scene = native / "scenario" / "scenario.sumocfg"
    from_reqs = requests(job["map"], spec, job["seed"], job["demand"], job["penetration"])
    cav_ids = frozenset(r["id"] for r in from_reqs if r["cav"])
    cmd = start(api, scene, job["seed"])
    t0 = float(job["t0"])
    api.simulation.loadState(str(native / "states" / f"t{int(t0)}.xml.gz"))
    assert abs(api.simulation.getTime() - t0) < 1e-6, api.simulation.getTime()
    live = set(api.vehicle.getIDList())
    for v in live & cav_ids:
        api.vehicle.setSpeedMode(v, CAV_SPEED_MODE)
    rec = Recorder(api, from_reqs, cav_ids, t0, inserted0=int(job["inserted_t0"]))
    # optional common dither (replicate salt): identical extra pulse in EVERY arm of a replicate
    arm = Arm(api, list(job["members"]) + list(job.get("dither") or []), t0, job["map"])
    wall0 = time.time()
    t_end = t0 + float(job["H"])
    member_tr = {m["id"]: [] for m in job["members"]}
    while rec.t < t_end - 1e-9:
        t_next = round(rec.t + DT, 3)
        if abs(t_next - round(t_next)) < 1e-6 and t_next > t0:   # control tick at whole seconds
            live = set(api.vehicle.getIDList())
            arm.control(t_next, live)
        rec.step()
        if abs(rec.t - round(rec.t)) < 1e-6 and rec.t - t0 <= 180:
            live = set(api.vehicle.getIDList())
            for vid in member_tr:
                if vid in live:
                    member_tr[vid].append((rec.t, api.vehicle.getRoadID(vid),
                                           api.vehicle.getLaneIndex(vid),
                                           round(api.vehicle.getLanePosition(vid), 1),
                                           round(api.vehicle.getSpeed(vid), 2)))
    res = rec.summary()
    api.close()
    res.update(kind="branch", job=job, sim_s=float(job["H"]), wall_s=round(time.time() - wall0, 2),
               arm_events=arm.events, member_traj=member_tr)
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
