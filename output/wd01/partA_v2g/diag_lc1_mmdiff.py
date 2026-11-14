"""Diff the eval counter vs the independent counter for one mismatching
window (w8623m2600p02t540, w=810, H=30): which transition does the eval
count that the independent raw-row counter does not?"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as EA2  # noqa: E402
from partA_v2g.eval_a2 import _run_paths  # noqa: E402

WD = "w8623m2600p02t540"
W = 810.0
H = 30
DT = 0.5

EA.MAPPER = EA.ModelMapper()
plan = json.loads(EA2.PLAN.read_text(encoding="utf-8"))
entry = next(w for w in plan["worlds"] if w["world"] == WD)
entry["chunk"] = 4
entry["native_run"] = _run_paths(WD, EA.Path(EA.wdpaths.RESEARCH_DENSE_ROOT))["physics"]
EA.OPENED_RUN_FILES.clear()
EA.TRAIN_WORLD_NAMES.add(WD)
log = EA.NativeLog(entry)
EA.assert_train_only()

j0 = int(round((W - log.t0) / DT))
n_h = int(round(H / DT))
truth = [log.rows.get(j0 + jj, {}) for jj in range(n_h + 1)]
tracked = sorted(EA.tracked_state(log.truth_at(W), EA.load_params())[0])
print("tracked:", len(tracked))
for j in range(1, n_h + 1):
    prev, cur = truth[j - 1], truth[j]
    for vid in tracked:
        p, c = prev.get(vid), cur.get(vid)
        if p is None or c is None or p[0] < 0 or c[0] < 0 or p[0] == c[0]:
            continue
        print(f"eval-event step {j} t={W + j * DT} vid={vid} "
              f"{p[0]}->{c[0]} s_prev={p[1]:.2f} s_cur={c[1]:.2f} "
              f"v_cur={c[2]:.2f}")

# raw lane_ids of those vehicles around the window
raw = {}
path = entry["native_run"]
with open(path, encoding="utf-8") as fh:
    for line in fh:
        i = line.find('"time_s": ')
        if i < 0:
            continue
        t = float(line[i + 10:line.find(",", i + 10)])
        tick = int(round((t - log.t0) / DT))
        if not (j0 <= tick <= j0 + n_h + 2):
            continue
        row = json.loads(line)
        for vid, v in row["vehicles"].items():
            if vid in tracked:
                raw.setdefault(tick, {})[vid] = (
                    v["lane_id"], float(v["lane_position_m"]))
vids_ev = set()
j = 0
for j in range(1, n_h + 1):
    prev, cur = truth[j - 1], truth[j]
    for vid in tracked:
        p, c = prev.get(vid), cur.get(vid)
        if p is None or c is None or p[0] < 0 or c[0] < 0 or p[0] == c[0]:
            continue
        vids_ev.add(vid)
print("\nraw lane_id sequences of eval-event vehicles in the window:")
for vid in sorted(vids_ev):
    seq = [(t0_ := k, raw[k][vid][0], round(raw[k][vid][1], 1))
           for k in sorted(raw) if vid in raw[k]]
    print(" ", vid)
    for k, lid, pos in seq:
        print(f"    tick {k} (t={log.t0 + k * DT}) {lid} @ {pos}")
