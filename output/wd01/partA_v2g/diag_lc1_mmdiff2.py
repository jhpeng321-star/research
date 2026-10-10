"""Diff eval vs independent counter for (w8632m2000p02t240, 300, 30):
which vehicle event does the eval count that the dest-tracked
independent counter still does not?"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as EA2  # noqa: E402
from partA_v2g.eval_a2 import _run_paths  # noqa: E402

WD, W, H = "w8632m2000p02t240", 300.0, 30
DT = 0.5

EA.MAPPER = EA.ModelMapper()
plan = json.loads(EA2.PLAN.read_text(encoding="utf-8"))
entry = next(w for w in plan["worlds"] if w["world"] == WD)
entry["chunk"] = 4
entry["native_run"] = _run_paths(
    WD, Path(EA.wdpaths.RESEARCH_DENSE_ROOT))["physics"]
EA.OPENED_RUN_FILES.clear()
EA.TRAIN_WORLD_NAMES.add(WD)
log = EA.NativeLog(entry)
EA.assert_train_only()

params = EA.load_params()
zone = params["rsu_zone_default"]["per_lane_s"]
j0 = int(round((W - log.t0) / DT))
n_h = int(round(H / DT))
truth_w = log.truth_at(W)


def zc(lane, s):
    return lane >= 0 and zone[lane][0] <= s < zone[lane][1]


tracked = sorted(v for v, (l, s, _v, _a) in truth_w.items() if zc(l, s))
print("eval tracked:", len(tracked))
print("eval tracked vids:", tracked)

truth = [log.rows.get(j0 + jj, {}) for jj in range(n_h + 1)]
eval_events = []
for j in range(1, n_h + 1):
    prev, cur = truth[j - 1], truth[j]
    for vid in tracked:
        p, c = prev.get(vid), cur.get(vid)
        if p is None or c is None or p[0] < 0 or c[0] < 0 or p[0] == c[0]:
            continue
        eval_events.append((round(W + j * DT, 3), vid, p[0], c[0],
                            round(p[1], 2)))
print("eval events:")
for e in eval_events:
    print(" ", e)

mine = []
for line in open(HERE / "results" / "lc1" / "truth_events.jsonl",
                 encoding="utf-8"):
    e = json.loads(line)
    if e["world"] == WD and W < e["t"] <= W + H + 1e-9:
        mine.append((round(e["t"], 3), e["vid"], e["f"], e["t_"],
                     round(e["s"], 2)))
print("independent events (all vehicles):")
for e in mine:
    print(" ", e)

# raw rows of any eval-event vehicle not covered by an independent event
covered = {(e[1], e[0]) for e in mine}
extra = [e for e in eval_events if (e[1], e[0]) not in covered]
print("eval events with no independent counterpart:")
for vid, t in [(e[1], e[0]) for e in extra]:
    print(f"  --- raw rows of {vid} around t={t}:")
    path = entry["native_run"]
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"time_s": ')
            if i < 0:
                continue
            t2 = float(line[i + 10:line.find(",", i + 10)])
            if not (t - 4.0 <= t2 <= t + 4.0):
                continue
            row = json.loads(line)
            v = row["vehicles"].get(vid)
            if v is not None:
                print(f"    t={t2} {v['lane_id']} @ "
                      f"{v['lane_position_m']:.2f} "
                      f"lc={v.get('lane_changed')}")
