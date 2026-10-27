"""S-BACKLOG: dump world->box case lists from worlds_c0X.json (read-only, gpuA)."""
import json
import sys
from pathlib import Path

LOOP = Path("/root/autodl-tmp/c4/restart20261026/arm/loop")

for cyc in sys.argv[1:]:
    cases = json.loads((LOOP / f"worlds_{cyc}.json").read_text())["cases"]
    print(cyc, " ".join(f"w{c['seed']}@{c['box']}" for c in cases))
