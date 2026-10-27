"""Inspect one e3long job dir (cpuP/cpuQ, read-only)."""
import json
import sys
from pathlib import Path

import numpy as np

job = Path(sys.argv[1])
for name in ("cumulative_120.json", "cumulative_300.json", "result.json"):
    p = job / name
    if p.is_file():
        print(f"== {name}")
        print(json.dumps(json.loads(p.read_text()))[:700])
traj = job / "traj_base.npz"
if traj.is_file():
    with np.load(traj, allow_pickle=False) as d:
        print("== traj_base.npz keys:", sorted(d.files))
        for k in sorted(d.files)[:30]:
            a = d[k]
            print(" ", k, a.shape, a.dtype)
