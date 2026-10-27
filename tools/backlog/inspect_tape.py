"""Inspect a raw c4_sumo_episode tape's npz keys/meta (read-only; never hetdata_truth)."""
import json
import sys
from pathlib import Path

import numpy as np

tape = Path(sys.argv[1])
with np.load(tape / "episode.npz", allow_pickle=False) as d:
    print("keys:", sorted(d.files))
    meta = json.loads(str(d["meta_json"]))
    for k in sorted(meta):
        v = meta[k]
        print(" meta.", k, "=", str(v)[:120])
    t = d["time_s"]
    print("time_s:", float(t[0]), "->", float(t[-1]), "n=", len(t))
    for name in ("entry_first_seen_s", "entry_first_entrance", "vehicle_ids"):
        if name in d.files:
            a = d[name]
            print(name, a.shape, a.dtype, a[:8] if a.ndim == 1 else "")
