"""Where is ramp.95 at t=810.0 in the raw rows of w8623m2600p02t540?"""
import json
from pathlib import Path

path = Path(r"E:\research-dense\runs\campaign_stage1_chunk4") / \
    "w8623m2600p02t540__native_base" / "run" / "physics_steps.jsonl"
targets = {810.0, 809.5, 810.5}
found = {}
with open(path, encoding="utf-8") as fh:
    for line in fh:
        i = line.find('"time_s": ')
        if i < 0:
            continue
        t = float(line[i + 10:line.find(",", i + 10)])
        if not any(abs(t - x) < 1e-6 for x in targets):
            continue
        row = json.loads(line)
        v = row["vehicles"].get("ramp.95")
        found[t] = v
for t in sorted(found):
    print(t, found[t])
