#!/usr/bin/env python3
"""Post-hoc lane-command accounting from recorded POLICY branch episodes (no simulation).

    python tools/loop14_checks/lane_parity_npz.py <episode.npz> [...]

For every issued ``lane_command_target`` the SUMO lane of the same vehicle is followed for 6 s
(0.5 s physics rows); counts requested (WM-accepted) / issued / reached / not reached, the reach latency
and the speed of the not-reached vehicles.  Use on pure-policy worlds (every request is WM-mask accepted).
"""
import collections
import json
import sys

import numpy as np

tot, lat, speeds = collections.Counter(), collections.Counter(), []
for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    ct, T, lane, vi = z["control_time_s"], z["time_s"], z["lane"], z["vid_index"]
    cmd, req, V, pres = z["lane_command_target"], z["lane_request_target"], z["v_mps"], z["present"]
    tot["requested"] += int((req >= 0).sum())
    tot["issued"] += int((cmd >= 0).sum())
    for i, k in zip(*np.nonzero(cmd >= 0)):
        r0 = np.searchsorted(T, ct[i] - 1e-6)
        tgt, vid, out = int(cmd[i, k]), vi[r0, k], "not_reached"
        for dt in np.arange(0.5, 6.01, 0.5):
            q = np.searchsorted(T, ct[i] + dt - 1e-6)
            if q >= len(T) or not pres[q, k] or vi[q, k] != vid:
                out = "left_or_slot_changed"
                break
            if lane[q, k] == tgt:
                out = "reached"
                lat[float(dt)] += 1
                break
        tot[out] += 1
        if out == "not_reached":
            speeds.append(float(V[r0, k]))
sp = np.array(speeds)
print(json.dumps({"counts": dict(tot), "reach_latency_s": sorted(lat.items()),
                  "not_reached_v_lt_1": int((sp < 1).sum()), "not_reached_n": len(sp)}, indent=1))
