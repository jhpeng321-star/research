"""Second-stage stats for residual_probe: authority of the residual in realised SUMO motion."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import residual_probe as RP  # noqa: E402
import torch  # noqa: E402

root = Path(sys.argv[1])
ep = root / "e3final/main/e3final/shard0/jobs/s977000_ours/deploy/episode/episode.npz"
z = dict(np.load(ep, allow_pickle=False))
state = RP.states_from_episode(z)
on, _ = RP.executor(root / "e3final/inputs/policy_latest.pt")
times, ctl_t, req, vid = z["time_s"], z["control_time_s"], z["requested_acceleration_mps2"], z["vid_index"]
rows = []
for c, t in enumerate(ctl_t):
    if t < 300 - 1e-9:
        continue
    i = int(np.flatnonzero(np.abs(times - t) < 1e-6)[0])
    if i + 2 >= len(times):
        break
    st = state(i)
    r = on(st).residual[0]
    act = st["is_cav"][0].numpy() & (st["lane"][0].numpy() >= 0)
    for k in np.flatnonzero(act):
        if not (vid[i + 1, k] == vid[i, k] and z["present"][i + 1, k]):
            continue
        v0, v1 = float(z["v_mps"][i, k]), float(z["v_mps"][i + 1, k])
        rows.append((float(r[k]), float(req[c, k]), (v1 - v0) / 0.5, v0))
a = np.array(rows)
r, rq, a1, v0 = a.T
nat = rq - r
d = a1 - nat
out = dict(n=len(a),
           frac_realised_eq_native_est_1e3=float(np.mean(np.abs(d) < 1e-3)),
           frac_realised_eq_native_est_1e2=float(np.mean(np.abs(d) < 1e-2)),
           frac_realised_eq_requested_1e2=float(np.mean(np.abs(a1 - rq) < 1e-2)),
           frac_realised_eq_half_requested_1e2=float(np.mean(np.abs(a1 - rq / 2) < 1e-2)),
           realised_minus_native_est_quantiles=[float(x) for x in np.quantile(d, [.05, .25, .5, .75, .95])],
           ols_a1_on_native_and_residual=[float(x) for x in np.linalg.lstsq(
               np.c_[nat, r, np.ones_like(r)], a1, rcond=None)[0]],
           residual_quantiles=[float(x) for x in np.quantile(r, [0, .05, .5, .95, 1])])
print(json.dumps(out, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
