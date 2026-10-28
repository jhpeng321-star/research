"""Split realised fraction by command sign (recorded episodes only)."""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
out = {}
for arm in ("mappo", "ours"):
    z = dict(np.load(root / f"e3final/main/e3final/shard0/jobs/s977000_{arm}/deploy/episode/episode.npz"))
    times, ctl_t, tgt, iss, ctrl, vid = (z[k] for k in ("time_s", "control_time_s", "issued_target_speed_mps",
        "issued_acceleration_mps2", "controller", "vid_index"))
    rows = []
    for c, t in enumerate(ctl_t):
        if t < 300:
            continue
        i = int(np.flatnonzero(np.abs(times - t) < 1e-6)[0])
        if i + 2 >= len(times):
            break
        for k in np.flatnonzero((ctrl[c] == 3) & z["present"][i]):
            if vid[i + 2, k] == vid[i, k] and z["present"][i + 2, k]:
                rows.append((float(iss[c, k]), float(z["v_mps"][i, k]), float(z["v_mps"][i + 2, k])))
    a = np.array(rows)
    cmd, v0, v2 = a.T
    real = (v2 - v0) / 1.0
    d = {}
    for name, sel in (("cmd<=-0.5", cmd <= -0.5), ("-0.5<cmd<0", (cmd > -0.5) & (cmd < 0)),
                      ("0<=cmd<0.5", (cmd >= 0) & (cmd < 0.5)), ("cmd>=0.5", cmd >= 0.5)):
        if sel.any():
            d[name] = dict(n=int(sel.sum()), mean_cmd=float(cmd[sel].mean()), mean_realised=float(real[sel].mean()),
                           frac_realised_within_0p1_of_cmd=float(np.mean(np.abs(real[sel] - cmd[sel]) < 0.1)))
    out[arm] = dict(n=len(a), cmd_quantiles=[float(x) for x in np.quantile(cmd, [.05, .25, .5, .75, .95])], by_sign=d)
print(json.dumps(out, indent=1))
