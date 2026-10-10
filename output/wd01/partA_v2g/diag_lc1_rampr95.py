"""Dump ramp.95 raw rows + the independent counter's events for
w8623m2600p02t540 (mismatch investigation)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WD = "w8623m2600p02t540"
vid = "ramp.95"
path = Path(r"E:\research-dense\runs\campaign_stage1_chunk4") / \
    f"{WD}__native_base" / "run" / "physics_steps.jsonl"

print(f"--- raw rows of {vid} in [{830}, {846}]:")
with open(path, encoding="utf-8") as fh:
    for line in fh:
        i = line.find('"time_s": ')
        if i < 0:
            continue
        t = float(line[i + 10:line.find(",", i + 10)])
        if not (828.0 <= t <= 846.0):
            continue
        row = json.loads(line)
        v = row["vehicles"].get(vid)
        if v is not None:
            print(f"  t={t} {v['lane_id']} @ {v['lane_position_m']:.2f} "
                  f"v={v['speed_mps']:.2f} lc={v.get('lane_changed')}")

print(f"--- independent-counter events of {vid}:")
for line in open(HERE / "results" / "lc1" / "truth_events.jsonl",
                 encoding="utf-8"):
    e = json.loads(line)
    if e["world"] == WD and e["vid"] == vid and e["t"] >= 820:
        print(" ", e)
