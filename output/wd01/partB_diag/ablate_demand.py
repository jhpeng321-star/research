"""Ridge ablation: does the future-arrival schedule (demand vector) drive tau-skill?"""
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
sys.path.insert(0, "E:/rd-wt/wd01/output/wd01")
from partB_v0.data import load_manifest, V0Dataset
man = load_manifest("E:/research-dense/output/wd01_partB_v0data", verify_files=False, log=lambda *a: None)
ds = V0Dataset(man, log=lambda *a: None)
r = ds.rows; F = ds.feats
lb = F["lane_bins"]; cnt = lb[..., 0]; spd = lb[..., 1]
lane_n = cnt.sum(2); lane_v = (cnt*spd).sum(2)/np.maximum(lane_n, 1e-6)
L = cnt.shape[2]
thirds = [cnt[:, :, i*L//3:(i+1)*L//3].sum((1, 2)) for i in range(3)]
ctx = F["context"]; dem = F["demand"]
# context: [t/1000,(t-t0)/600,ttb/600,loading/2700,0,0,qmain/50,qramp/50]
base = np.column_stack([lane_n, lane_v, *thirds, ctx])
sets = {"state+ctx+SCHEDULE": np.column_stack([base, dem]),
        "state+ctx (no schedule)": base,
        "state+ctx+rate-only": np.column_stack([base, dem[:, :20].reshape(-1, 10, 2).mean(1)])}
ti = r["t_idx"].to_numpy(); y = r["total_delay_ref_s"].to_numpy()
extra = np.column_stack([r["policy_id"].to_numpy(), r["pre_fix"].astype(float).to_numpy()])
print("tau | " + " | ".join(sets))
for tau in sorted(r["tau"].unique()):
    tr = ((r["split"] == "TRAIN") & np.isclose(r["tau"], tau)).to_numpy()
    ca = ((r["split"] == "CALIBRATION") & np.isclose(r["tau"], tau)).to_numpy()
    out = []
    for name, G in sets.items():
        X = np.column_stack([G[ti], extra])
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        out.append(f"{np.abs(m.predict(X[ca]) - y[ca]).mean():.0f}")
    print(f"{tau:.0f} | " + " | ".join(out))
