"""EXPLORATORY: ramp-target merge time (s after t0) in the nopulse reference arm, SUMO vs WM variants."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, "tools")
import qual_analyze as Q
rows = []
for jd in sorted(glob.glob("data/pulse/jobs/s*_t300")):
    job = os.path.basename(jd); rec = json.load(open(f"{jd}/record.json"))
    sref = Q.Tape(f"{jd}/traj__nopulse.npz", True)
    for i, tg in enumerate(rec["targets"]):
        ts = sref.slot_of(tg["target_vid"])
        if ts is None or int(sref.lane[0, ts]) != 2:
            continue
        r = {"job": job, "t": i, "sumo": Q.merge_time(sref, Q.ident(sref, ts))}
        for v in ("off", "wmlaw"):
            ms = []
            for k in range(6):
                p = f"out/{v}/pulse/wm/k{k}/{job}/traj__nopulse.npz"
                if os.path.exists(p):
                    w = Q.Tape(p, False); ws = Q.match_wm(sref, w, ts)
                    if ws is not None:
                        ms.append(Q.merge_time(w, ws))
            r[v] = [None if m is None or not np.isfinite(m) else round(float(m), 1) for m in ms]
        rows.append(r)
for r in rows:
    print(r)
json.dump(rows, open("LOOP14C_MERGEWAIT.json", "w"), indent=1, default=float)
