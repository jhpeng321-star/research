"""Stopped-tick breakdown by lane / 30 s bin for one world (cal_phys roll)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output" / "wd01" / "wm_n1"))
sys.path.insert(0, str(ROOT / "_out"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
CHUNK = Path(r"E:/research-dense/runs/campaign_stage1_chunk4")
W = sys.argv[1] if len(sys.argv) > 1 else "w8676m2700p20t360"

from closedloop.fast_hdv import fast_bundle, install_fast_gap_projection  # noqa: E402

install_fast_gap_projection()
from closedloop.a11_predictor import (NONE_CODE, PLANNER_ACTION_CODES,  # noqa: E402
                                      _rollout_mask, batched_rollout,
                                      load_a11_bundle)
from closedloop.live_state import DepartureHistory, live_state_from_row  # noqa: E402

man = json.load(open(CHUNK / f"{W}__base" / "saved_state" / "STATE_MANIFEST.json"))
t0 = float(W.split("t")[-1])
st = live_state_from_row(man["physics_row_at_t0"], DepartureHistory(CHUNK / f"{W}__base", t0))
rmask = _rollout_mask(st)
cav = np.asarray(st.is_cav, bool) & rmask
K = len(st.slot_vid)
codes = np.full((1, K), NONE_CODE, np.int64)
codes[0, np.flatnonzero(cav)] = PLANNER_ACTION_CODES["native"]
b = fast_bundle(load_a11_bundle())
ro = batched_rollout(b, st, codes, n_steps=240, t0_s=t0, physical_exits=True,
                     open_entries=True, sumo_fidelity=True, merge_cal=True,
                     residual=False)
msk, pos, spd, lns = ro.msk[:, 0], ro.pos[:, 0], ro.spd[:, 0], ro.lns[:, 0]
print(W, "lane x 30s-bin stopped ticks (v<0.5, s_down in [0,75.44]):")
tot = 0
for ln in (0, 1, 2):
    row = []
    for bb in range(4):
        c = 0
        for t in range(bb * 60, min((bb + 1) * 60, msk.shape[0])):
            m = msk[t] & (lns[t] == ln) & (pos[t] >= 0.0) & (pos[t] <= 75.44) & (spd[t] < 0.5)
            c += int(m.sum())
        row.append(c)
        tot += c
    print(f"  lane {ln}: {row}")
print("  total:", tot)
# recorded reference by lane
from features.geometry import is_junction_lane, s_down_of  # noqa: E402

rows = []
with open(CHUNK / f"{W}__pi" / "run" / "physics_steps.jsonl", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
        if len(rows) >= 240:
            break
print("recorded lane x 30s-bin stopped ticks:")
rtot = 0
for ln in (0, 1, 2):
    row = []
    for bb in range(4):
        c = 0
        for r in rows[bb * 60:(bb + 1) * 60]:
            for vid, v in (r.get("vehicles") or {}).items():
                lid = v.get("lane_id")
                if not lid or is_junction_lane(lid):
                    continue
                try:
                    lp, sd = s_down_of(lid, float(v["lane_position_m"]))
                except Exception:
                    continue
                if lp == ln and 0.0 <= sd <= 75.44 and float(v.get("speed_mps", 0.0)) < 0.5:
                    c += 1
        row.append(c)
        rtot += c
    print(f"  lane {ln}: {row}")
print("  total:", rtot)
