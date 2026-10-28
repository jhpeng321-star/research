"""Check traj_base.npz time resolution vs the 0.25 s road-delay convention."""
import numpy as np
import sys
from pathlib import Path

with np.load(sys.argv[1], allow_pickle=False) as d:
    t, vid, v, s = d["t"], d["vid"], d["v"], d["s"]
print("rows", len(t), "t range", float(t.min()), float(t.max()))
u, c = np.unique(np.round(np.diff(np.unique(t)), 6), return_counts=True)
print("unique dt:", dict(zip(u.tolist(), c.tolist())))
print("sample rows:", list(zip(t[:6].tolist(), vid[:6].tolist(), v[:6].tolist())))
