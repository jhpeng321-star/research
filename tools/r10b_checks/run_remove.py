"""R10B check M2: causal removal test (R9_YIELD 5.3) for every arm.  Same seed with / without one waiting HDV ramp car
(removed from state, history and recorded future); the target lane-1 follower's speed difference is compared for near
pairs (3-25 m behind the ramp car) and far placebos (>= 60 m behind).  Output: one small npz per arm."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import END_M, add_paths  # noqa: E402

SPECS: list = []


def pairs(rec, n_near=3, n_far=2):
    veh = {v["id"]: v for v in rec["initial_state"]["vehicles"] if int(v.get("lane", -1)) >= 0}
    l1 = [v for v in veh.values() if int(v["lane"]) == 1]
    out = []
    for r in veh.values():
        if int(r["lane"]) != 2 or r.get("is_cav") or not (0.0 <= r["s"] <= END_M):
            continue
        behind = [v for v in l1 if v["s"] <= r["s"]]
        if not behind:
            continue
        f = max(behind, key=lambda v: v["s"])
        out.append((r["id"], f["id"], r["s"] - f["s"], bool(f.get("is_cav")), r["s"]))
    near = sorted([p for p in out if 3 <= p[2] <= 25 and not p[3]], key=lambda p: -p[4])[:n_near]
    far = sorted([p for p in out if p[2] >= 60 and not p[3]], key=lambda p: -p[4])[:n_far]
    return [(p, "near") for p in near] + [(p, "far") for p in far]


def _init(specs, repo):
    global SPECS
    SPECS = specs
    add_paths(repo)
    import rollout_worker as rw
    rw.init_arms(specs, "off")


def work(payload):
    import rollout_worker as rw
    from diag_yield_remove import strip
    idx, rec, rid, fid, kind, dA, sample, steps = payload
    out = []
    for name, _ in SPECS:
        traj = {}
        for tag, r in (("with", rec), ("without", strip(rec, rid))):
            res = rw.roll_once(name, r, sample, steps)
            if fid not in res["ids"]:
                traj = None
                break
            i = res["ids"].index(fid)
            traj[tag] = (res["v"][:, i].astype(np.float32), res["pos"][:, i].astype(np.float32), res["mask"][:, i])
        if traj is None:
            continue
        ok = traj["with"][2] & traj["without"][2]
        out.append((name, idx, rid, sample, kind, float(dA), traj["with"][0] - traj["without"][0],
                    traj["with"][1] - traj["without"][1], ok, float(traj["with"][0][0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = [tuple(v.split("=", 1)) for v in a.var]
    with gzip.open(a.states, "rt") as f:
        recs = json.load(f)
    if a.limit:
        recs = recs[: a.limit]
    tasks = [(i, rec, rid, fid, kind, dA, s, a.steps)
             for i, rec in enumerate(recs) for (rid, fid, dA, _, _), kind in pairs(rec) for s in range(a.k)]
    print("tasks", len(tasks), flush=True)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = []
    from multiprocessing import Pool
    with Pool(a.workers, initializer=_init, initargs=(specs, a.repo)) as pool:
        for j, rr in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
            res.extend(rr)
            if (j + 1) % 32 == 0:
                print(f"{j+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    for name, _ in specs:
        R = [r for r in res if r[0] == name]
        np.savez_compressed(os.path.join(a.out, f"remove_{name}.npz"),
                            kind=np.array([r[4] for r in R]), dA=np.array([r[5] for r in R]),
                            state=np.array([r[1] for r in R]), sample=np.array([r[3] for r in R]),
                            dv=np.stack([r[6] for r in R]) if R else np.zeros((0, 1)),
                            dpos=np.stack([r[7] for r in R]) if R else np.zeros((0, 1)),
                            ok=np.stack([r[8] for r in R]) if R else np.zeros((0, 1), bool),
                            v0=np.array([r[9] for r in R]))
    json.dump({"tasks": len(tasks), "seconds": time.time() - t0}, open(os.path.join(a.out, "REMOVE_DONE.json"), "w"))
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
