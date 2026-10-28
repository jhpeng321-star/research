"""How much of an issued direct target is realised in SUMO (MAPPO arm, recorded episode only)."""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
out = {}
for arm in ("mappo", "ours"):
    z = dict(np.load(root / f"e3final/main/e3final/shard0/jobs/s977000_{arm}/deploy/episode/episode.npz"))
    times, ctl_t, tgt, iss, req, ctrl, vid = (z[k] for k in ("time_s", "control_time_s", "issued_target_speed_mps",
        "issued_acceleration_mps2", "requested_acceleration_mps2", "controller", "vid_index"))
    rows = []
    for c, t in enumerate(ctl_t):
        if t < 300:
            continue
        i = int(np.flatnonzero(np.abs(times - t) < 1e-6)[0])
        if i + 2 >= len(times):
            break
        for k in np.flatnonzero((ctrl[c] == 3) & z["present"][i]):
            if not (vid[i + 2, k] == vid[i, k] and z["present"][i + 2, k]):
                continue
            v0, v1, v2 = (float(z["v_mps"][j, k]) for j in (i, i + 1, i + 2))
            rows.append((float(iss[c, k]), float(tgt[c, k]), float(req[c, k]), v0, v1, v2))
    a = np.array(rows)
    iss_a, tgt_v, req_a, v0, v1, v2 = a.T
    dv_cmd = tgt_v - v0
    big = np.abs(dv_cmd) > 0.2
    out[arm] = dict(n=len(a), n_big=int(big.sum()),
        realised_fraction_1s_median=float(np.median((v2 - v0)[big] / dv_cmd[big])) if big.any() else None,
        realised_fraction_first_half_median=float(np.median((v1 - v0)[big] / dv_cmd[big])) if big.any() else None,
        frac_target_reached_1s=float(np.mean(np.abs(v2 - tgt_v)[big] < 0.05)) if big.any() else None,
        issued_vs_requested_mean_abs=float(np.nanmean(np.abs(iss_a - req_a))))
print(json.dumps(out, indent=1))
