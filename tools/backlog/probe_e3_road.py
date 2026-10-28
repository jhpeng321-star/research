"""Probe v2: reproduce e3long road_delay_vs exactly from the 0.5 s tape.

e3long (branching.step_pair + BranchTracker): per 0.5 s interval (k-1 -> k],
road += DT_S * (1 - v_end/31.29)+ where the slot cohort is slots PRESENT at k-1
(m0); v_end = v[k] if the vehicle continues, else its v[k-1] (exiting vehicle,
speed carried); entering vehicles (not present at k-1) contribute nothing."""
import json
import sys
from pathlib import Path

import numpy as np

tape, cum = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
with np.load(tape / "episode.npz", allow_pickle=False) as d:
    ep = {k: d[k] for k in d.files}
meta = json.loads(str(ep["meta_json"]))
t0, H, DT = float(meta["warmup_s"]), float(sys.argv[3]) if len(sys.argv) > 3 else 300.0, 0.5
times = np.asarray(ep["time_s"], np.float64)
present = np.asarray(ep["present"], bool)
v = np.where(present, ep["v_mps"], 0.0).astype(np.float64)

tick = int(np.searchsorted(times, t0 - 1e-6))
n = int(round(H / DT))
prev_p = present[tick:tick + n]                     # m0 per interval
cur_v = v[tick + 1:tick + 1 + n]                    # kin1 v
prev_v = v[tick:tick + n]                           # kin0 v (carried for exits)
cohort_v = np.where(present[tick + 1:tick + 1 + n], cur_v, prev_v)
road = (DT * np.maximum(0.0, 1.0 - cohort_v / 31.29) * prev_p).sum()
target = float(cum["road_delay_vs"])
print(f"window [{t0},{t0+H}) road={road:.6f} target={target:.6f} diff={road - target:+.6f}")
