"""S-DIAG9 (2026-10-23): inference-only diagnostic of the round-9 stuck ramp cars.  DIAGNOSTIC ONLY.

Reuses the cpuN 72-state k-sample harness (``diag_r9stuck_check_base`` = tools/v8wm3_check_cpun_k.py,
unchanged) and adds, for every HDV that starts on the ramp lane (lane 2):
  * the WM per-step merge-pivot trace (position, speed, lane, c, eligibility, follower/leader gap and
    speed, cdf, acceleration) and the SUMO-recorded counterpart (position, speed, lane and the SUMO
    follower/leader gap and speed in lane 1 computed with the SAME geometry definition);
  * inference-time switches per variant (``--var name=ckpt|opt=val,...``):
      les=0/1   physics.ramp_lane_end_stop       rr=0/1   physics.ramp_risk_retain
      law=r8    round-8 merge law: merge_pivot removed, merge_core secure-gap law, merge_end_wait=1,
                ramp_lane_end_stop=0, ramp_risk_retain=0 (r9 weights; = round 8 exactly if the weights are)
      tf=1      feed the recorded CAV teacher through cav_controller (pivot step ignores cav_teacher)
      yield=1   re-enable the round-8 merge_core follower yield inside the pivot model (monkey patch)
      sf,Tf,bf,sl,Tl,bl,ptau,w   override the effective pivot parameter (softplus-inverse written to raw)
      bfx=<m/s2> bf and bl both set; vfcap=<m/s> not implemented (use parameters)
No weights, configs or model code are modified; nothing is written outside --out.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import pickle
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, "tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_r9stuck_check_base as base  # noqa: E402
base.SET = os.environ.get("R9DIAG_STATES", base.SET)

DT = 0.5
END = 75.44
G = base.G
TRACE_KEYS = ("pos", "v", "lane", "mask", "acc", "c", "elig", "phys", "veto", "gap_f", "v_f",
              "has_f", "gap_l", "v_l", "has_l", "cdf", "emass")


def _raw(v):
    """softplus-inverse of an effective positive value."""
    v = float(v)
    return v + math.log(-math.expm1(-v))


def init(var_specs, entries="off"):
    import torch
    base.init(var_specs, entries)
    for name, path, o in var_specs:
        p, cap, _ = G["p"][name]
        m = p.model
        ch = {}
        if "les" in o:
            ch["ramp_lane_end_stop"] = o["les"] not in ("0", "false", "off")
        if "rr" in o:
            ch["ramp_risk_retain"] = o["rr"] not in ("0", "false", "off")
        if o.get("law") == "r8":
            ch.update(merge_end_wait=True, ramp_lane_end_stop=False, ramp_risk_retain=False)
            m.merge_pivot = None
            m.merge_decision = "secure_gap"
        if "mew" in o:
            ch["merge_end_wait"] = o["mew"] not in ("0", "false", "off")
        if ch:
            new = dataclasses.replace(m.physics, **ch)
            for owner, attr in base.wc._physics_holders(m):
                setattr(owner, attr, new)
        if m.merge_pivot is not None:
            with torch.no_grad():
                for key in ("sf", "Tf", "bf", "sl", "Tl", "bl", "ptau", "w"):
                    if key in o:
                        getattr(m.merge_pivot, {"sf": "s_f", "Tf": "T_f", "bf": "b_f", "sl": "s_l",
                                                "Tl": "T_l", "bl": "b_l", "ptau": "tau",
                                                "w": "w"}[key] + "_raw").fill_(_raw(o[key]))
                if "bfx" in o:
                    m.merge_pivot.b_f_raw.fill_(_raw(o["bfx"]))
                    m.merge_pivot.b_l_raw.fill_(_raw(o["bfx"]))
        if o.get("tf") in ("1", "true", "on") and m.merge_pivot is not None:
            # Replay the recorded CAV teacher through the supported ``cav_controller`` hook (the
            # pivot step ignores ``cav_teacher``); i-th controller call = physics tick i.
            orig_rollout = m.rollout

            def rollout(state, *a, _orig=orig_rollout, **kw):
                teacher = kw.get("cav_teacher")
                if teacher is not None and kw.get("cav_controller") is None:
                    calls = {"i": 0}

                    def controller(_state, _t=teacher, _c=calls):
                        row = _t[:, min(_c["i"] + 1, _t.shape[1] - 1)]
                        _c["i"] += 1
                        return row
                    kw["cav_controller"] = controller
                return _orig(state, *a, **kw)

            m.rollout = rollout
        if o.get("yield") in ("1", "true", "on") and m.merge_pivot is not None:
            orig_forward = m.forward
            mp = m.merge_pivot

            def forward(*a, _orig=orig_forward, _m=m, _mp=mp, **kw):
                _m.merge_pivot = None          # makes forward() compute the round-8 merge_core pass
                try:
                    return _orig(*a, **kw)
                finally:
                    _m.merge_pivot = _mp

            m.forward = forward
        G["p"][name] = (p, cap, o)


def recorded_geometry(tr, rp, k, lengths):
    """SUMO-side ramp-car geometry rows 0..T: speed, gap/speed of nearest lane-1 follower/leader.

    Same definition as ``merge_pivot.geometry`` (front positions, follower gap subtracts the EGO
    length, leader gap subtracts the LEADER length, lane-1 vehicles present at that row only)."""
    T1, n = tr.shape[0], tr.shape[1]
    out = {key: np.full(T1, np.nan) for key in ("gap_f", "v_f", "gap_l", "v_l")}
    s = tr[:, :, 0]
    lane = tr[:, :, 1]
    v = tr[:, :, 2]
    for t in range(T1):
        if not rp[t, k]:
            continue
        cand = rp[t] & (lane[t] == 1) & (np.arange(n) != k)
        if not cand.any():
            continue
        ds = s[t, k] - s[t, cand]
        idx = np.flatnonzero(cand)
        fm = ds >= 0
        if fm.any():
            j = np.argmin(np.where(fm, ds, np.inf))
            out["gap_f"][t] = ds[j] - lengths[k]
            out["v_f"][t] = v[t, idx[j]]
        lm = ds <= 0
        if lm.any():
            j = np.argmin(np.where(lm, -ds, np.inf))
            out["gap_l"][t] = -ds[j] - lengths[idx[j]]
            out["v_l"][t] = v[t, idx[j]]
    return out


def work(payload):
    idx, rec, arm, sample = payload
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    out = []
    for name, (p, cap, o) in G["p"].items():
        try:
            n_steps = 240
            t0, ids, kin, is_cav = t0_vehicles(rec)
            teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, arm,
                                                       p.backend.tools)
            if teacher is None:
                out.append({"idx": idx, "var": name, "sample": sample, "error": "no arm"})
                continue
            if getattr(p, "flags", None) is not None:
                p.flags = dataclasses.replace(p.flags, lane_seed=int(sample))
            p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm},
                             n_steps * DT, int(sample))
            r = cap["last"]
            n = len(ids)
            tr = teacher[0].numpy()
            rp = present_seq[0].numpy().astype(bool)
            pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
            lan = np.concatenate([tr[:1, :, 1], r["lanes"][0, :, :n].numpy()], 0)
            msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
            spd = np.concatenate([tr[:1, :, 2], r["speeds"][0, :, :n].numpy()], 0)
            hdv = ~np.asarray(is_cav, bool)
            ramp = [k for k in range(n) if hdv[k] and int(tr[0, k, 1]) == 2]
            res = {"idx": idx, "var": name, "sample": sample, "error": None,
                   "t_wm": np.array([base.first_off2(lan, msk, k) for k in ramp]),
                   "t_rec": np.array([base.first_off2(tr[:, :, 1], rp, k) for k in ramp]),
                   "d_wm": np.array([base.off2_dist(pos, lan, msk, k) for k in ramp]),
                   "d_rec": np.array([base.off2_dist(tr[:, :, 0], tr[:, :, 1], rp, k)
                                      for k in ramp]),
                   "ramp_ids": [ids[k] for k in ramp], "ramp_k": ramp, "is_cav": hdv.copy()}
            for ts in (30, 60, 120):
                i = int(ts / DT)
                both = msk[i] & rp[i] & hdv
                d = np.abs(pos[i] - tr[i, :, 0])
                res[f"ds{ts}"] = float(np.median(d[both])) if both.any() else np.nan
                rb = both.copy()
                rb[[k for k in range(n) if k not in ramp]] = False
                res[f"ramp_ds{ts}"] = float(np.median(d[rb])) if rb.any() else np.nan
                res[f"ramp_ds_all{ts}"] = d[rb]
            lane0 = tr[0, :, 1].astype(int)
            for ts in (30, 60, 120):
                i = int(ts / DT)
                both = msk[i] & rp[i] & hdv
                res[f"n_both{ts}"] = int(both.sum())
                res[f"spd_wm{ts}"] = float(spd[i][both].mean()) if both.any() else np.nan
                res[f"spd_rec{ts}"] = float(tr[i, :, 2][both].mean()) if both.any() else np.nan
                res[f"slow_wm{ts}"] = int((spd[i][both] < 2.0).sum())
                res[f"slow_rec{ts}"] = int((tr[i, :, 2][both] < 2.0).sum())
                d = np.abs(pos[i] - tr[i, :, 0])
                for ln in (0, 1, 2):
                    sel = both & (lane0 == ln)
                    res[f"ds{ts}_l{ln}"] = float(np.median(d[sel])) if sel.any() else np.nan
                    res[f"n{ts}_l{ln}"] = int(sel.sum())
            for ts in (30, 60, 120):
                i = int(ts / DT)
                both = msk[i] & rp[i] & hdv
                for ln in (0, 1, 2):
                    sw = both & (lan[i] == ln)
                    sr = both & (tr[i, :, 1] == ln)
                    res[f"vlane{ln}_wm{ts}"] = float(spd[i][sw].mean()) if sw.any() else np.nan
                    res[f"vlane{ln}_rec{ts}"] = float(tr[i, :, 2][sr].mean()) if sr.any() else np.nan
                    res[f"nlane{ln}_wm{ts}"] = int(sw.sum())
                    res[f"nlane{ln}_rec{ts}"] = int(sr.sum())
                res[f"merged_wm{ts}"] = int((both & (lane0 == 2) & (lan[i] == 1)).sum())
                res[f"merged_rec{ts}"] = int((both & (lane0 == 2) & (tr[i, :, 1] == 1)).sum())
                res[f"over_wm{ts}"] = int((both & (lane0 == 1) & (lan[i] == 0)).sum())
                res[f"over_rec{ts}"] = int((both & (lane0 == 1) & (tr[i, :, 1] == 0)).sum())
            # per-car traces
            lengths = kin[:, 6]
            traces = {}
            for j, k in enumerate(ramp):
                tw = res["t_wm"][j]
                stuck = bool(np.isinf(tw))
                if not (stuck or G.get("trace_all")):
                    continue
                pad = lambda a: a[0, :, k].numpy()   # noqa: E731
                tt = {"stuck": stuck, "t_rec": float(res["t_rec"][j]),
                      "rec_pos": tr[:, k, 0], "rec_v": tr[:, k, 2], "rec_lane": tr[:, k, 1],
                      "rec_mask": rp[:, k], "pos": pos[:, k], "v": spd[:, k], "lane": lan[:, k],
                      "mask": msk[:, k]}
                for key, src in (("c", "merge_pivot_c"), ("elig", "merge_pivot_eligible"),
                                 ("phys", "merge_pivot_physical_eligible"),
                                 ("veto", "merge_pivot_reservation_veto"),
                                 ("gap_f", "merge_pivot_geometry_gap_f"),
                                 ("v_f", "merge_pivot_geometry_v_f"),
                                 ("has_f", "merge_pivot_geometry_has_f"),
                                 ("gap_l", "merge_pivot_geometry_gap_l"),
                                 ("v_l", "merge_pivot_geometry_v_l"),
                                 ("has_l", "merge_pivot_geometry_has_l"),
                                 ("cdf", "merge_pivot_cdf"), ("emass", "merge_pivot_event_mass"),
                                 ("acc", "accel")):
                    if src in r:
                        tt[key] = np.asarray(pad(r[src]), dtype=np.float64)
                for key, val in recorded_geometry(tr, rp, k, lengths).items():
                    tt["rec_" + key] = val
                if "merge_pivot_s_f" in r:
                    tt["params"] = {q: float(r["merge_pivot_" + q][-1]) for q in
                                    ("s_f", "T_f", "b_f", "s_l", "T_l", "b_l", "tau", "w")
                                    if "merge_pivot_" + q in r}
                traces[ids[k]] = tt
            res["traces"] = traces
            out.append(res)
        except Exception as exc:  # noqa: BLE001
            import traceback
            out.append({"idx": idx, "var": name, "sample": sample,
                        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--states", default="", help="comma list of state indices (overrides every/limit)")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--trace-all", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--entries", choices=("off", "queue"), default="off")
    a = ap.parse_args()
    specs = [base.parse_var(s) for s in a.var]
    from wms.eval.normalize import iter_windows
    allrecs = list(enumerate(iter_windows(base.wc.guard(base.SET))))
    if a.states:
        want = {int(x) for x in a.states.split(",")}
        recs = [(i, r) for i, r in allrecs if i in want]
    else:
        recs = [(i, r) for i, r in allrecs if i % a.every == 0]
        if a.limit:
            recs = recs[: a.limit]
    tasks = [(i, r, "wall", s) for i, r in recs for s in range(a.k)]
    G["trace_all"] = a.trace_all
    t0 = time.time()
    res = []
    if a.workers <= 1:
        init(specs, a.entries)
        for t in tasks:
            res.extend(work(t))
    else:
        from multiprocessing import Pool
        init(specs, a.entries)      # validate every variant once in the parent (a failing Pool
        G.clear()                   # initializer would otherwise respawn workers forever)

        def _init(sp, en, ta):
            G["trace_all"] = ta
            init(sp, en)
        with Pool(a.workers, initializer=_init, initargs=(specs, a.entries, a.trace_all)) as pool:
            for j, rr in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
                res.extend(rr)
                if (j + 1) % 8 == 0:
                    print(f"{j+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"specs": specs, "results": res, "entries": a.entries, "k": a.k}, fh)
    print(f"done {time.time()-t0:.0f}s errors {sum(1 for r in res if r.get('error'))}", flush=True)
    for r in [r for r in res if r.get("error")][:3]:
        print(r)
    print(base.summarize(a.out))


if __name__ == "__main__":
    main()
