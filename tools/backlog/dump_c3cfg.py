"""Dump c3 policy-phase argv facts from gpuA arm config (read-only)."""
import json
from pathlib import Path

c = json.loads(Path("/root/autodl-tmp/c4/restart20261026/arm/CONFIG_C3_LONGINT.json").read_text())
p = c["policy"]
print("native_init:", p.get("native_init"))
print("config:", p.get("config"))
print("extra_args:", p.get("extra_args"))
print("wm config:", c.get("wm", {}).get("config"))
