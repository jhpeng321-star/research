"""Dump looplong_boundary doc from gpuA loop state (read-only)."""
import json
from pathlib import Path

s = json.loads(Path("/root/autodl-tmp/c4/restart20261026/arm/loop/LOOP14_STATE.json").read_text())
b = s.get("looplong_boundary") or {}
print(json.dumps({k: v for k, v in b.items() if k != "flags"}, indent=1))
print("flags:", json.dumps(b.get("flags")))
print("cohort:", json.dumps(s.get("looplong_cohort")))
