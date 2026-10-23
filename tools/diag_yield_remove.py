"""S-YIELD Q4b: causal removal test. For waiting HDV ramp cars R with a lane-1 HDV target follower F behind them,
roll the WM (same lane seed) with and without R in the t0 state (and its recorded future/history) and compare F's
speed. Placebo: remove a ramp car whose nearest lane-1 follower is >= 60 m behind. Inference only (gpuR).
  PYTHONPATH=src:.:tools python diag_yield_remove.py --states S.json.gz --var r9=<ckpt> --var r8=<ckpt> --workers 24 --out X.pkl"""
import argparse, copy, gzip, json, os, pickle, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
sys.path.insert(0, "tools")
import diag_yield_wm as W

END = 75.44


def pairs(rec, max_per=3):
    veh = {v["id"]: v for v in rec["initial_state"]["vehicles"] if int(v.get("lane", -1)) >= 0}
    l1 = [v for v in veh.values() if int(v["lane"]) == 1]
    out = []
    for r in veh.values():
        if int(r["lane"]) != 2 or r.get("is_cav") or not (0.0 <= r["s"] <= END):
            continue
        behind = [v for v in l1 if v["s"] <= r["s"]]
        if not behind:
            continue
        f = max(behind, key=lambda v: v["s"])
        out.append((r["id"], f["id"], r["s"] - f["s"], bool(f.get("is_cav")), r["s"]))
    near = sorted([p for p in out if 3 <= p[2] <= 25 and not p[3]], key=lambda p: -p[4])[:max_per]
    far = sorted([p for p in out if p[2] >= 60 and not p[3]], key=lambda p: -p[4])[:1]
    return [(p, "near") for p in near] + [(p, "far") for p in far]


def strip(rec, rid):
    rec = copy.deepcopy(rec)
    rec["initial_state"]["vehicles"] = [v for v in rec["initial_state"]["vehicles"] if v["id"] != rid]
    for arm, frames in rec["recorded"].items():
        for fr in frames:
            fr["vehicles"] = [v for v in fr["vehicles"] if v["id"] != rid]
    h = rec["history"]
    if rid in h["ids"]:
        i = h["ids"].index(rid)
        for row in h["kin"]:
            row[i] = [0.0] * len(row[i])
        for row in h["mask"]:
            row[i] = False
        for row in h.get("target_mask", []):
            row[i] = False
        h["ids"][i] = f"__history_context_slot{i}"
    return rec


def work(payload):
    key, rec, rid, fid, kind, dA, sample, steps = payload
    out = []
    for tag, r in (("with", rec), ("without", strip(rec, rid))):
        res = W.work((key, r, "native", sample, steps))
        for q in res:
            q.update({"tag": tag, "rid": rid, "fid": fid, "kind": kind, "dA": dA})
            if not q.get("error"):
                for k_ in ("rec_pos", "rec_lane", "rec_v", "rec_mask", "acc"):
                    q.pop(k_, None)
            out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True); ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = [tuple(v.split("=", 1)) for v in a.var]
    with gzip.open(a.states, "rt") as f:
        recs = json.load(f)
    tasks = []
    for i, rec in enumerate(recs):
        for (rid, fid, dA, _, _), kind in pairs(rec):
            for s in range(a.k):
                tasks.append((i, rec, rid, fid, kind, dA, s, a.steps))
    print("tasks", len(tasks), flush=True)
    from multiprocessing import Pool
    W.init(specs, "off"); W.G.clear()
    t0 = time.time(); res = []
    with Pool(a.workers, initializer=W._init, initargs=(specs, "off")) as pool:
        for j, rr in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
            res.extend(rr)
            if (j + 1) % 16 == 0:
                print(f"{j+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    pickle.dump({"specs": specs, "results": res}, open(a.out, "wb"))
    print("done", time.time() - t0, "errors", sum(1 for r in res if r.get("error")))
    for r in [r for r in res if r.get("error")][:2]:
        print(r["error"])


if __name__ == "__main__":
    main()
