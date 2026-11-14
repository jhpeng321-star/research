"""Pick v5-written TRAIN runs for the byte-identity spot check (read-only)."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

base = Path("D:/rd-staging/wmn1_feat/logs_v5")
done_keys = set()
for k in range(6):
    for line in (base / f"sub{k}.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+\[end\] (\S+) status=written", line)
        if m:
            done_keys.add(m.group(1))
df = pd.read_parquet("inventory/inventory.parquet")
rows = []
for _, r in df.iterrows():
    run = str(r["run"])
    arm = run.split("__", 1)[1] if "__" in run else str(r["arm"])
    key = f"{r['batch']}/{r['world']}__{arm}"
    if key in done_keys and r["split"] == "TRAIN":
        rows.append((key, str(r["batch"]), str(r["world"]), run,
                     str(r["root_rel"]), float(r["t0_s"])))
rows.sort()
picks = [rows[0], rows[len(rows) // 2], rows[-1]]
for p in picks:
    print(p)
(base / "byteid_picks.json").write_text(
    json.dumps([dict(key=p[0], batch=p[1], world=p[2], run=p[3],
                     root_rel=p[4], t0_s=p[5]) for p in picks], indent=1),
    encoding="utf-8")
print(f"TRAIN v5-written candidates: {len(rows)}")
