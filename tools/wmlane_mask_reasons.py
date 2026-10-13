"""Why does the physics feasibility mask veto recorded lane changes? (wall_same, default params).

Usage (workspace root, PYTHONPATH=src): python tools/wmlane_mask_reasons.py [out.json]
"""
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from wmlane_event_diag import _kin, _record_tensors, category  # noqa: E402
from wms.eval.normalize import iter_windows  # noqa: E402
from wms.physics.config import PhysicsConfig  # noqa: E402
from wms.physics.idm import idm_accel  # noqa: E402
from wms.physics.mobil import mobil_merge_logits  # noqa: E402
from wms.physics.neighbours import lane_neighbour  # noqa: E402

p = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary")
reasons = defaultdict(Counter)
gaps = defaultdict(list)
for record in iter_windows("E:/research-dense/data_cache/wallc1_states_h60/wall_same.json"):
    ids, s, v, acc, lane, valid, cav = _record_tensors(record)
    kin = _kin(s, v, acc, lane, valid)
    pres = torch.as_tensor(valid)
    logits = mobil_merge_logits(kin, pres, p)
    hdv = valid & ~cav
    t_n, n = lane.shape
    for t in range(t_n - 1):
        for j in range(n):
            a, b = lane[t, j], lane[t + 1, j]
            if not (hdv[t, j] and hdv[t + 1, j] and a != b and abs(a - b) == 1):
                continue
            cat = category(int(a), int(b))
            reasons[cat]["recorded"] += 1
            if float(logits[t, j, b]) > -8.0 + 1e-6:
                continue
            reasons[cat]["masked"] += 1
            k1, p1 = kin[t:t + 1], pres[t:t + 1]
            tl = torch.full((1, n), int(b))
            gl, lv = lane_neighbour(k1, p1, tl, p, ahead=True)
            gf, fv = lane_neighbour(k1, p1, tl, p, ahead=False)
            gap_l, gap_f = float(gl[0, j]), float(gf[0, j])
            # follower braking if the car cuts in (default IDM)
            a_f = float(idm_accel(fv[0, j:j + 1], torch.tensor([float(v[t, j])]),
                                  gf[0, j:j + 1], p.idm_params(), accel_min=-9.0,
                                  accel_max=3.0))
            tags = []
            if gap_l < p.lane_safe_gap_m:
                tags.append("lead_gap<6m")
            if gap_f < p.lane_safe_gap_m:
                tags.append("follow_gap<6m")
            if a_f < -p.b_safe_end_mps2:
                tags.append("follower_brake>6")
            elif a_f < -p.b_safe_base_mps2:
                tags.append("follower_brake>4")
            if not tags:
                tags.append("zone_or_other")
            for tag in tags:
                reasons[cat][tag] += 1
            gaps[cat].append((gap_l, gap_f, float(v[t, j])))
out = {c: dict(r) for c, r in reasons.items()}
for c, g in gaps.items():
    arr = np.asarray(g)
    out[c]["median_lead_gap_m"] = float(np.median(arr[:, 0]))
    out[c]["median_follow_gap_m"] = float(np.median(arr[:, 1]))
    out[c]["median_speed_mps"] = float(np.median(arr[:, 2]))
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(json.dumps(out, indent=1))
