"""Dump LOOP14_STATE.json top level + data/returns references (read-only)."""
import json
from pathlib import Path

s = json.loads(Path("/root/autodl-tmp/c4/restart20261026/arm/loop/LOOP14_STATE.json").read_text())
for k, v in s.items():
    if isinstance(v, (dict, list)):
        print(k, f"<{type(v).__name__} len={len(v)}>")
    else:
        print(k, ":", str(v)[:180])
d = s.get("data") or {}
for cyc, info in (d.items() if isinstance(d, dict) else []):
    print("data", cyc, json.dumps(info)[:300])
print(json.dumps(s.get("cycles"), indent=1)[:2400])
