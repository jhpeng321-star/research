"""S-R14 offline inference (CPU only, no training): S-R12C with the 13E flags, +/- R14 D1.

Arms use the NATIVE switches of the code under test (never emulated):
  13E    emergency_braking + secure_gap_veto + learned yield (pass rule)   = the 13E launch physics
  13EL   13E + physics.lane_change_secure_gap (R14 D1)
  13Efb  emergency_braking + learned yield (13E fallback, no gs)
  13EfbL 13Efb + lane_change_secure_gap

Modes
  panel  the R13E TRAIN-role 48-state panel (sanity_roll.py: TRAIN role re-verified, record re-extended
         to H s of recorded SUMO future, entries v2 + queue state, CRN seed 130013 + state index).
  pairs  the LONGDIAG sel48 pairs (COMBO300 harness: ref = native arm, act = random switching arm,
         lane seed k, torch seed 100003*idx + 7919*k + 17; CRN across arms and ref/act).
Outputs are COMBO300-compatible compact 1 s npz files (p<idx>_<tag>_<arm>_k<k>.npz, p<idx>_<tag>_sumo.npz).

  python roll_r14.py --mode panel --repo REPO --out OUT --horizon 120 --n 48 --arms 13E,13EL --workers 3
  python roll_r14.py --mode pairs --repo REPO --out OUT --pairs sel48_pairs.json --order order.json --npairs 5
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanity_roll as SR  # noqa: E402  (sets CPU/thread environment on import)
import numpy as np  # noqa: E402

E_GS = {"emergency_braking": True, "secure_gap_veto": True}
E_FB = {"emergency_braking": True, "secure_gap_veto": False}
SR.ARMS.update({
    "13E": dict(phys={**E_GS, "lane_change_secure_gap": False}, yld=True),
    "13EL": dict(phys={**E_GS, "lane_change_secure_gap": True}, yld=True),
    "13Efb": dict(phys={**E_FB, "lane_change_secure_gap": False}, yld=True),
    "13EfbL": dict(phys={**E_FB, "lane_change_secure_gap": True}, yld=True),
})
SR.ARMS["base"]["phys"] = {"emergency_braking": False, "secure_gap_veto": False,
                           "lane_change_secure_gap": False}
G = SR.G


def init_worker(ckpt: str, repo: str) -> None:
    SR.init_worker(ckpt, repo)
    m = G["m"]
    G["phys0"] = dataclasses.replace(m.physics, lane_change_secure_gap=False)
    G["lc"] = {"events": 0}


def _roll(rec, arm, ctrl_arm, seed_torch, lane_seed, horizon):
    torch, p = G["torch"], G["p"]
    SR.set_arm(arm)
    random.seed(seed_torch)
    np.random.seed(seed_torch % (2 ** 32))
    torch.manual_seed(seed_torch)
    p.flags = dataclasses.replace(G["flags0"], lane_seed=int(lane_seed))
    limit = int(os.environ.get("R14_TIMEOUT_S", "0") or 0)
    if limit:
        def _alarm(*_a):
            raise TimeoutError("rollout timeout")
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(limit)
    try:
        return p.predict_states(rec, {**(rec.get("cav_controllers") or {}), "arm": ctrl_arm}, horizon,
                                int(lane_seed))
    finally:
        if limit:
            signal.alarm(0)
        p.flags = G["flags0"]
        SR.set_arm("base")


def _save(fn: Path, frames, t0_s, horizon, **meta):
    c = SR.compact(frames, float(t0_s), horizon)
    keys = sorted({k for f in frames for v in f.get("vehicles", []) for k in v})
    tmp = fn.with_name(f".{fn.name}.{os.getpid()}.npz")
    np.savez_compressed(tmp, covered_rows=c.pop("covered_rows"), keys=np.array(keys), **meta, **c)
    os.replace(tmp, fn)


def process(task):
    mode, idx, tag, arm, out_dir, horizon, k = task
    rec = G["records"][idx]
    out, errs, t1 = Path(out_dir), [], time.time()
    fn = out / f"p{idx:02d}_{tag}_{arm}_k{k}.npz"
    tmo = out / f"p{idx:02d}_{tag}_{arm}_k{k}.timeout"
    if fn.exists() or tmo.exists():
        return idx, tag, arm, "exists", errs, 0.0
    why = SR.arm_supported(arm)
    if why:
        return idx, tag, arm, f"SKIPPED ({why})", errs, 0.0
    try:
        from wms.eval.rollout_inputs import t0_vehicles
        t0_s, ids, _kin, is_cav = t0_vehicles(rec)
        if mode == "panel":
            ctrl, st, ls = rec["arm"], SR.SEED0 + idx, SR.SEED0 + idx
        else:
            ctrl = rec["ref_arm"] if tag == "ref" else rec["action_arm"]
            st, ls = 100003 * int(idx) + 7919 * int(k) + 17, int(k)
        for s in (st, ls):
            if s in SR.EXCLUDED_SEEDS:
                raise ValueError("evaluation seed")
        try:
            fr = _roll(rec, arm, ctrl, st, ls, horizon)
        except TimeoutError:
            tmo.write_text(f"timeout after {time.time() - t1:.0f} s")
            return idx, tag, arm, "timeout", errs, time.time() - t1
        _save(fn, fr, t0_s, horizon, roster=np.array([str(i) for i in ids]),
              roster_cav=np.asarray(is_cav, bool), t0=float(t0_s), key=str(rec.get("pair_key") or rec["key"]),
              arm=arm, ctrl_arm=str(ctrl), seed=st, lane_seed=ls, sec=time.time() - t1)
    except Exception:
        errs.append(traceback.format_exc())
    return idx, tag, arm, "ERR" if errs else "ok", errs, time.time() - t1


def load_panel(a):
    if SR.sha256(a.states) != SR.STATES_SHA:
        raise SystemExit("panel is not the registered R13E train-role panel")
    with gzip.open(SR.guard(a.states), "rt", encoding="utf-8") as f:
        doc = json.load(f)
    states = doc["states"][a.skip: a.skip + a.n]
    SR.verify_train(states, json.loads(Path(SR.guard(a.receipt)).read_text()), Path(a.data_root))
    records, info = {}, []
    for j, r in enumerate(states):
        rec, meta = SR.extend_record(r, Path(a.data_root), a.horizon)
        records[a.skip + j] = rec
        info.append(meta)
    return records, info, ["ref"]


def load_pairs(a):
    rows = json.loads(Path(SR.guard(a.pairs)).read_text(encoding="utf-8"))["pairs"]
    order = json.loads(Path(SR.guard(a.order)).read_text())[a.skip: a.skip + a.n]
    bykey = {r["pair_key"]: (i, r) for i, r in enumerate(rows)}
    records = {}
    for key in order:
        i, r = bykey[key]
        if int(r.get("seed") or 0) in SR.EXCLUDED_SEEDS:
            raise SystemExit("evaluation seed")
        records[i] = r
    return records, [{"idx": i, "pair_key": r["pair_key"]} for i, r in records.items()], ["ref", "act"]


def write_sumo(mode, records, out, horizon):
    from wms.eval.normalize import recorded_for
    from wms.eval.rollout_inputs import t0_vehicles
    for idx, rec in records.items():
        t0_s, ids, _kin, is_cav = t0_vehicles(rec)
        tags = (("ref", rec["arm"]),) if mode == "panel" else (("ref", rec["ref_arm"]), ("act", rec["action_arm"]))
        for tag, arm in tags:
            fn = out / f"p{idx:02d}_{tag}_sumo.npz"
            if fn.exists():
                continue
            frames = rec["recorded"][arm] if mode == "panel" else recorded_for(rec, arm)
            _save(fn, frames, t0_s, horizon, roster=np.array([str(i) for i in ids]),
                  roster_cav=np.asarray(is_cav, bool), t0=float(t0_s), arm="sumo", ctrl_arm=str(arm),
                  key=str(rec.get("pair_key") or rec["key"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("panel", "pairs"), required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(SR.CKPT))
    ap.add_argument("--states", default=str(SR.STATES))
    ap.add_argument("--receipt", default=str(SR.RECEIPT))
    ap.add_argument("--data-root", default=str(SR.L5 / "data"))
    ap.add_argument("--pairs", default="/root/laneimp_work/pairs/sel48_pairs.json")
    ap.add_argument("--order", default=str(SR.L5 / "diag_r14/from_diag_long/order.json"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--arms", default="13E,13EL")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if not 1 <= a.workers <= 3:
        raise SystemExit("workers must be 1..3 (two concurrent jobs <= 6 workers)")
    if os.name == "posix":
        os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    arms = a.arms.split(",")
    if any(x not in SR.ARMS for x in arms):
        raise SystemExit(f"unknown arm in {arms}")
    repo = Path(SR.guard(a.repo)).resolve()
    for p in (repo, repo / "src", repo / "tools"):
        sys.path.insert(0, str(p))
    out = Path(SR.guard(a.out))
    out.mkdir(parents=True, exist_ok=True)
    if SR.sha256(a.ckpt) != SR.CKPT_SHA:
        raise SystemExit("checkpoint is not the registered S-R12C FINAL")
    records, info, tags = (load_panel if a.mode == "panel" else load_pairs)(a)
    G["records"] = records
    write_sumo(a.mode, records, out, a.horizon)
    meta = {"argv": sys.argv, "pid": os.getpid(), "repo": str(repo), "ckpt_sha256": SR.CKPT_SHA,
            "mode": a.mode, "horizon_s": a.horizon, "arms": arms, "records": info, "k": a.k,
            "started": time.strftime("%F %T")}
    (out / f"run_meta_{os.getpid()}.json").write_text(json.dumps(meta, indent=1))
    todo = [(a.mode, i, tag, arm, str(out), a.horizon, a.k) for i in records for tag in tags for arm in arms]
    print(f"todo {len(todo)} workers {a.workers}", flush=True)
    t0 = time.time()
    from multiprocessing import get_context
    pool = get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(a.ckpt, str(repo)),
                                    maxtasksperchild=4)
    for idx, tag, arm, status, errs, sec in pool.imap_unordered(process, todo, chunksize=1):
        print(f"[{time.time() - t0:6.0f}s] p{idx:02d} {tag} {arm} {status} {sec:.0f}s", flush=True)
        if errs:
            print(errs[0], flush=True)
    (out / f"DONE_{os.getpid()}").write_text(time.strftime("%F %T"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
