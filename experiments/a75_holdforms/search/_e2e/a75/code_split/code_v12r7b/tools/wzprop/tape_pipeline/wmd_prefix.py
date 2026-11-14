"""WMQUAL-B native prefix tape (task WM-D, phase wm_wz_data): one job per process (libsumo).

    python wmd_prefix.py JOB.json   ->  JOB out_dir/TAPE.npz + RESULT.json

Re-runs the native (no CAV command) prefix of one labelled wz3to2 world x cell with the
VERBATIM newmap2 core (nm2_core.py sha ad6b4bd6, population.py sha 6af5ee23, copied next to
this file): identical scenario streams, SUMO command, CAV speedMode and J definition.  The only
addition is a READ-ONLY capture after every 0.5 s physics step of every vehicle on the road
(edge, lane, global x, speed, acceleration) and of the source backlog, so the WM can be started
from the exact SUMO state of each labelled moment (with its full observed history).

Identity with the labelled runs is checked afterwards (wmd_check.py): per-second J_road/J_src and
the CAV snapshot rows at the native save times must equal the original native RESULT.json.
HDVs are never commanded; no CAV is commanded either (native prefix only).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nm2_core as C  # noqa: E402  (verbatim newmap2 core)

TAPE_SCHEMA = "wzqual_native_tape/1"


def run_prefix(job: dict) -> dict:
    api = C.api_module()
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(job["map_spec"]).read_text())
    scene, reqs = C.write_scenario(out / "scenario", Path(job["map_spec"]).parent, spec, job["seed"],
                                   job["demand"], job["penetration"])
    cav_ids = frozenset(r["id"] for r in reqs if r["cav"])
    depart = {r["id"]: float(r["depart"]) for r in reqs}
    cmd = C.start(api, scene, job["seed"])
    rec = C.Recorder(api, reqs, cav_ids, 0.0)
    snap_times = sorted(float(t) for t in job.get("snap_times", []))
    snaps = {}
    names: list[str] = []
    index: dict[str, int] = {}
    edges: list[str] = []
    edge_index: dict[str, int] = {}
    cols = {k: [] for k in ("step", "vid", "edge", "lane", "x", "v", "a")}
    times, backlog, inserted = [], [], []
    t_end = float(job["t_end"])
    wall0 = time.time()
    veh = api.vehicle
    while rec.t < t_end - 1e-9:
        rec.step()
        k = len(times)
        times.append(rec.t)
        for vid in veh.getIDList():
            i = index.get(vid)
            if i is None:
                i = index[vid] = len(names)
                names.append(vid)
            edge = veh.getRoadID(vid)
            e = edge_index.get(edge)
            if e is None:
                e = edge_index[edge] = len(edges)
                edges.append(edge)
            cols["step"].append(k)
            cols["vid"].append(i)
            cols["edge"].append(e)
            cols["lane"].append(veh.getLaneIndex(vid))
            cols["x"].append(C.xglob(edge, veh.getLanePosition(vid)))
            cols["v"].append(veh.getSpeed(vid))
            cols["a"].append(veh.getAcceleration(vid))
        backlog.append(max(0, bisect_right(rec.times, rec.t) - rec.inserted))
        inserted.append(rec.inserted)
        if snap_times and abs(rec.t - snap_times[0]) < 1e-6:
            s = C.snapshot(api, cav_ids)
            s["inserted"] = rec.inserted
            snaps[f"t{int(snap_times[0])}"] = s
            snap_times.pop(0)
    res = rec.summary()
    api.close()
    meta = dict(schema=TAPE_SCHEMA, map=C.MAP, x_b=C.X_B, dt=C.DT, speed_limit=C.SPEED_LIMIT,
                seed=int(job["seed"]), demand=job["demand"], penetration=float(job["penetration"]),
                cell=job["cell"], t_end=t_end, edge_offset=C.EDGE_OFFSET, sumo_cmd=cmd,
                core_sha256=hashlib.sha256((HERE / "nm2_core.py").read_bytes()).hexdigest(),
                population_sha256=hashlib.sha256((HERE / "population.py").read_bytes()).hexdigest(),
                job_id=job["job_id"], split="WMQUAL", task="WM-D")
    buf = io.BytesIO()
    np.savez_compressed(
        buf, time_s=np.asarray(times, np.float64), step=np.asarray(cols["step"], np.int32),
        vid=np.asarray(cols["vid"], np.int32), edge=np.asarray(cols["edge"], np.int16),
        lane=np.asarray(cols["lane"], np.int16), x=np.asarray(cols["x"], np.float64),
        v=np.asarray(cols["v"], np.float64), a=np.asarray(cols["a"], np.float64),
        backlog=np.asarray(backlog, np.int32), inserted=np.asarray(inserted, np.int32),
        ids=np.asarray(names), is_cav=np.asarray([n in cav_ids for n in names], bool),
        depart_s=np.asarray([depart.get(n, np.nan) for n in names], np.float64),
        edges=np.asarray(edges), meta_json=np.asarray(json.dumps(meta)))
    data = buf.getvalue()
    (out / "TAPE.npz").write_bytes(data)
    res.update(kind="prefix", job=job, sumo_cmd=cmd, n_requests=len(reqs), n_cav=len(cav_ids),
               snapshots=snaps, sim_s=t_end, wall_s=round(time.time() - wall0, 2),
               tape_sha256=hashlib.sha256(data).hexdigest(), tape_rows=len(cols["step"]),
               routes_sha256=hashlib.sha256((out / "scenario" / "requests.rou.xml").read_bytes()).hexdigest())
    return res


def main(path: str) -> None:
    job = json.loads(Path(path).read_text())
    t = time.time()
    try:
        res = run_prefix(job)
        res["status"] = "ok"
    except Exception as exc:  # failure receipt, never silently retried
        import traceback
        res = dict(status="failed", job=job, error=repr(exc), tb=traceback.format_exc())
    res["elapsed_s"] = round(time.time() - t, 2)
    Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(job["out_dir"]) / "RESULT.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1])
