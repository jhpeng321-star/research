"""Compare tracked sets + window counts for (w8623m2600p02t540, 810, 30)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

WD, W, H = "w8623m2600p02t540", 810.0, 30
DT = 0.5

# --- my tracked computation, redone from the audit's own logic
PHYS_OF = {
    "1191554636_1": (0, 0.0), "1487606179_2": (0, 2645.67),
    "1487606178_1": (0, 2721.11), "1191554636_0": (1, 0.0),
    "1487606179_1": (1, 2644.22), "1487606178_0": (1, 2719.66),
    "11553048#1_0": (2, 0.0), "1487606179_0": (2, 378.66),
}
ZONE = {0: (1845.67, 2871.11), 1: (1844.22, 2869.66), 2: (0.0, 454.10)}
t0 = 540.0
j0 = int(round((W - t0) / DT))
tracked = {}
path = Path(r"E:\research-dense\runs\campaign_stage1_chunk4") / \
    f"{WD}__native_base" / "run" / "physics_steps.jsonl"
with open(path, encoding="utf-8") as fh:
    for line in fh:
        i = line.find('"time_s": ')
        if i < 0:
            continue
        t = float(line[i + 10:line.find(",", i + 10)])
        if abs(t - W) > 1e-6:
            continue
        row = json.loads(line)
        for vid, v in row["vehicles"].items():
            hit = PHYS_OF.get(v["lane_id"])
            if hit is None:
                continue
            phys, s = hit[0], hit[1] + float(v["lane_position_m"])
            if ZONE[phys][0] <= s < ZONE[phys][1]:
                tracked[vid] = (phys, round(s, 2))
print("my tracked:", len(tracked))
for name in ("ramp.94", "ramp.95", "ramp.96", "ramp.97", "ramp.99"):
    print(" ", name, tracked.get(name))

evs = []
for line in open(HERE / "results" / "lc1" / "truth_events.jsonl",
                 encoding="utf-8"):
    e = json.loads(line)
    if e["world"] == WD and e["vid"] in tracked and \
            W < e["t"] <= W + H + 1e-9:
        evs.append(e)
print("my events in window:")
for e in sorted(evs, key=lambda e: e["t"]):
    print(f"  t={e['t']} {e['vid']} {e['f']}->{e['t_']} s={e['s']:.2f} "
          f"junc={e['junction']}")
print("my window row:")
for line in open(HERE / "results" / "lc1" / "truth_windows.jsonl",
                 encoding="utf-8"):
    w = json.loads(line)
    if w["world"] == WD and abs(w["w"] - W) < 1e-6 and w["H"] == H:
        print(" ", w)
