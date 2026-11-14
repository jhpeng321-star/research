"""Supplement: survivor composition and N-right-lane error structure at 120 s."""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

with open(HERE.parent / "out" / "rollouts.pkl", "rb") as fh:
    payloads = pickle.load(fh)
states = payloads["C1"]

MERGE_LO, MERGE_HI = 0.0, 75.44


def zone_of(lane, s):
    if lane == 2:
        return "on_ramp"
    if lane in (0, 1):
        return "merge_zone" if MERGE_LO <= s <= MERGE_HI else ("upstream_mainline" if s < MERGE_LO else "downstream_mainline")
    return "junction"


rows = []
n_hdvs, surv120 = [], []
for si, st in enumerate(states):
    rids, rec = st["rids"], st["rec"]
    ridx = {x: i for i, x in enumerate(rids)}
    t0m = [v for v in rids if rec["present"][0, ridx[v]] and rec["lane"][0, ridx[v]] >= 0]
    hdv0 = [v for v in t0m if not bool(rec["is_cav"][0, ridx[v]])]
    n_hdvs.append(len(hdv0))
    surv120.extend(v for v in hdv0 if rec["present"][240, ridx[v]])
    # entrant set + affected (same def as aggregate.py)
    t0set = set(t0m)
    ent = set()
    for j, vid in enumerate(rids):
        if vid in t0set:
            continue
        tk = np.where(rec["present"][:, j] & (rec["lane"][:, j] >= 0))[0]
        if len(tk[tk >= 1]):
            ent.add(vid)
    aff = set()
    for k in range(241):
        rows_k = [j for j in range(len(rids)) if rec["present"][k, j] and rec["lane"][k, j] >= 0]
        for cl in (0, 1, 2):
            mem = sorted((j for j in rows_k if rec["lane"][k, j] == cl), key=lambda j: rec["s"][k, j])
            for a, b in zip(mem, mem[1:]):
                if rids[b] in ent and rids[a] in t0set:
                    aff.add(rids[a])
                if rids[a] in ent and rids[b] in t0set:
                    aff.add(rids[b])
    for model in payloads:
        pidx = {x: i for i, x in enumerate(payloads[model][si]["pids"])}
        pred = payloads[model][si]["pred"]
        for v in hdv0:
            if not (rec["present"][240, ridx[v]] and pred["present"][240, pidx[v]]):
                continue
            wrong = int(pred["lane"][240, pidx[v]]) != int(rec["lane"][240, ridx[v]]) and rec["lane"][240, ridx[v]] >= 0
            rows.append({
                "model": model, "state": si, "id": v, "affected": v in aff, "wrong": wrong,
                "ds": abs(float(pred["s"][240, pidx[v]]) - float(rec["s"][240, ridx[v]])),
                "signed": float(pred["s"][240, pidx[v]]) - float(rec["s"][240, ridx[v]]),
                "v_rec": float(rec["v"][240, ridx[v]]), "v_pred": float(pred["v"][240, pidx[v]]),
                "s_t0": float(rec["s"][0, ridx[v]]), "lane_t0": int(rec["lane"][0, ridx[v]]),
                "zone": zone_of(int(rec["lane"][0, ridx[v]]), float(rec["s"][0, ridx[v]])),
                "v_t0": float(rec["v"][0, ridx[v]]),
            })

print(f"n_hdvs_t0 mean {np.mean(n_hdvs):.1f}; survivors-at-120 mean per state {len(surv120)/67:.1f}; matched rows {len(rows)/2:.0f} per both models => per state {len(rows)/2/66:.1f}")
nr = [r for r in rows if not r["affected"] and not r["wrong"]]
for model in ("C1", "R4"):
    sel = [r for r in nr if r["model"] == model]
    print(f"\n{model} N-right-lane @120s: n={len(sel)}")
    for lo, hi, name in ((-1, 2, "rec v<2 (stopped)"), (2, 10, "2-10"), (10, 99, ">10")):
        b = [r for r in sel if lo <= r["v_rec"] < hi]
        if not b:
            continue
        sg = np.array([r["signed"] for r in b])
        print(f"  {name:18s} n={len(b):4d} med|ds|={np.median([r['ds'] for r in b]):7.1f} "
              f"mean|ds|={np.mean([r['ds'] for r in b]):7.1f} signed p10={np.percentile(sg,10):8.1f} "
              f"p50={np.percentile(sg,50):8.1f} p90={np.percentile(sg,90):8.1f} "
              f"dv_mean={np.mean([r['v_pred']-r['v_rec'] for r in b]):6.2f}")
    # by t0 zone
    for z in ("upstream_mainline", "merge_zone", "on_ramp"):
        b = [r for r in sel if r["zone"] == z]
        if b:
            print(f"  zone {z:18s} n={len(b):4d} med|ds|={np.median([r['ds'] for r in b]):7.1f}")
    # t0 speed of survivors
    print(f"  survivor t0 speed: med={np.median([r['v_t0'] for r in sel]):.1f} m/s, "
          f"t0 s: med={np.median([r['s_t0'] for r in sel]):.0f} m")
    print(f"  recorded speed at 120: med={np.median([r['v_rec'] for r in sel]):.1f} m/s; "
          f"pred speed at 120: med={np.median([r['v_pred'] for r in sel]):.1f} m/s")
