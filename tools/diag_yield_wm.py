"""S-YIELD (2026-10-23): inference-only WM rollouts on TRAIN-episode state records (round 8 / 9).
Teacher-forced native CAVs, entries off, SUMO exits end slots (the standard score harness semantics).
Saves, per (state, model, sample): WM and recorded trajectories of the t0 roster (small arrays).
Run on gpuR from the fit repo:  PYTHONPATH=src:.:tools python diag_yield_wm.py --states S.json.gz
  --var r9=<ckpt> --var r8=<ckpt> --workers 24 --k 2 --out OUT.pkl
No weights/config/model code changes."""
import argparse, dataclasses, gzip, json, os, pickle, sys, time
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
sys.path.insert(0, "tools")
import wm_eval_common as wc

DT = 0.5
G = {}


def init(specs, entries):
    import torch
    torch.set_num_threads(1)
    G["p"] = {}
    for name, path in specs:
        p, cap = wc.load_predictor(wc.guard(path), "AS_IS", hdv_exits="sumo", entries=entries)
        G["p"][name] = (p, cap)


def work(payload):
    idx, rec, arm, sample, steps = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    out = []
    for name, (p, cap) in G["p"].items():
        try:
            t0, ids, kin, is_cav = t0_vehicles(rec)
            teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, steps, arm, p.backend.tools)
            if getattr(p, "flags", None) is not None:
                p.flags = dataclasses.replace(p.flags, lane_seed=int(sample))
            p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm}, steps * DT, int(sample))
            r = cap["last"]; n = len(ids)
            tr = teacher[0].numpy(); rp = present_seq[0].numpy().astype(bool)
            f32 = lambda x: np.asarray(x, np.float32)
            pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
            lan = np.concatenate([tr[:1, :, 1], r["lanes"][0, :, :n].numpy()], 0)
            msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
            spd = np.concatenate([tr[:1, :, 2], r["speeds"][0, :, :n].numpy()], 0)
            acc = r["accel"][0, :, :n].numpy() if "accel" in r else None
            out.append({"idx": idx, "var": name, "sample": sample, "error": None, "t0": t0, "ids": ids,
                        "is_cav": np.asarray(is_cav, bool), "episode": rec.get("episode"), "key": rec.get("key"),
                        "pos": f32(pos), "lane": f32(lan), "mask": msk, "v": f32(spd),
                        "acc": None if acc is None else f32(acc),
                        "rec_pos": f32(tr[:, :, 0]), "rec_lane": f32(tr[:, :, 1]), "rec_v": f32(tr[:, :, 2]), "rec_mask": rp,
                        "dem": rec.get("demand_vph"), "pen": rec.get("penetration")})
        except Exception as exc:  # noqa: BLE001
            import traceback
            out.append({"idx": idx, "var": name, "sample": sample,
                        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"})
    return out


def _init(sp, en):
    init(sp, en)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--entries", default="off")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = [tuple(v.split("=", 1)) for v in a.var]
    with gzip.open(a.states, "rt") as f:
        recs = json.load(f)
    if a.limit:
        recs = recs[: a.limit]
    tasks = [(i, r, "native", s, a.steps) for i, r in enumerate(recs) for s in range(a.k)]
    t0 = time.time(); res = []
    if a.workers <= 1:
        init(specs, a.entries)
        for t in tasks:
            res.extend(work(t))
    else:
        from multiprocessing import Pool
        init(specs, a.entries); G.clear()
        with Pool(a.workers, initializer=_init, initargs=(specs, a.entries)) as pool:
            for j, rr in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
                res.extend(rr)
                if (j + 1) % 8 == 0:
                    print(f"{j+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"specs": specs, "results": res, "k": a.k}, fh)
    errs = [r for r in res if r.get("error")]
    print(f"done {time.time()-t0:.0f}s results {len(res)} errors {len(errs)}", flush=True)
    for r in errs[:3]:
        print(r["error"])


if __name__ == "__main__":
    main()
