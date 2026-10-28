"""Probe v3: exact e3long road delay with slot-reuse handling, two sources.

A) tape episode.npz: cohort = present at k-1; v_eff = v[k] if present[k] AND
   vid[k]==vid[k-1] (continues), else v[k-1] (exit or reused slot: old vehicle).
B) traj_base.npz (the tracker's own record): cohort = vids present at t-0.5;
   v_eff = v(t) if vid in row(t) else v(t-0.5)."""
import json
import sys
from pathlib import Path

import numpy as np

tape, cum = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
H = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
with np.load(tape / "episode.npz", allow_pickle=False) as d:
    ep = {k: d[k] for k in d.files}
meta = json.loads(str(ep["meta_json"]))
t0, DT = float(meta["warmup_s"]), 0.5
times = np.asarray(ep["time_s"], np.float64)
present = np.asarray(ep["present"], bool)
vid = np.asarray(ep["vid_index"], np.int32)
v = np.where(present, ep["v_mps"], 0.0).astype(np.float64)
tick = int(np.searchsorted(times, t0 - 1e-6))
n = int(round(H / DT))
p0, p1 = present[tick:tick + n], present[tick + 1:tick + 1 + n]
v0, v1 = v[tick:tick + n], v[tick + 1:tick + 1 + n]
cont = p1 & (vid[tick + 1:tick + 1 + n] == vid[tick:tick + n])
v_eff = np.where(cont, v1, v0)
road_a = (DT * np.maximum(0.0, 1.0 - v_eff / 31.29) * p0).sum()
target = float(cum["road_delay_vs"])
print(f"A tape   road={road_a:.6f} target={target:.6f} diff={road_a - target:+.6f}")

traj = tape.parent.parent / "traj_base.npz"
if traj.is_file():
    with np.load(traj, allow_pickle=False) as d:
        t, tv, tvid = d["t"], d["v"].astype(np.float64), d["vid"]
    rows = {}
    for tt, vv, ii in zip(t.tolist(), tv.tolist(), tvid.tolist()):
        rows.setdefault(tt, {})[ii] = vv
    ts = sorted(rows)
    road_b, missing = 0.0, 0
    for t1 in ts:
        if not (t0 < t1 <= t0 + H):
            continue
        prev = rows.get(round(t1 - DT, 3), {})
        cur = rows[t1]
        for veh_id, v_prev in prev.items():
            v_now = cur.get(veh_id, v_prev)
            road_b += DT * max(0.0, 1.0 - v_now / 31.29)
            if veh_id not in cur:
                missing += 1
    print(f"B traj   road={road_b:.6f} diff={road_b - target:+.6f} exits_charged={missing}")
