"""Review diagnostic (world model, 2026-09-29): v0.2 tau=30 vs climatology.

Read-only. It fits simple baselines (per-tau GBM and ridge on summary
features) on TRAIN and scores them on CAL. It also compares them with the
v0.2 run-1 ensemble median and with the tau/policy/demand climatology.
Metric: MAE of the point forecast (median) on CAL rows.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

WD01 = Path("E:/rd-wt/wd01/output/wd01")
sys.path.insert(0, str(WD01))
from partB_v0.data import load_manifest, V0Dataset  # noqa: E402
from partB_v0.evaluate import load_members, forward_mixture  # noqa: E402
import partB_v0.evaluate as EV  # noqa: E402

DATA = "E:/research-dense/output/wd01_partB_v0data"
RUN = "E:/research-dense/output/partB_results/v0.2_run1/extracted"

man = load_manifest(DATA, verify_files=False, log=lambda *a: None)
ds = V0Dataset(man, log=print)
r = ds.rows.copy()
F = ds.feats

# summary features per decision time
lb = F["lane_bins"]                      # [G, 3, L, 3] count, mean v, std v
cnt = lb[..., 0]
spd = lb[..., 1]
lane_n = cnt.sum(axis=2)                                   # [G, 3]
lane_v = (cnt * spd).sum(axis=2) / np.maximum(lane_n, 1e-6)
# near-merge region: bins covering ~ s in [2200, 2700] m (d-coordinate
# unknown here -> use coarse thirds of the bin axis instead)
L = cnt.shape[2]
thirds = [cnt[:, :, i * L // 3:(i + 1) * L // 3].sum(axis=(1, 2))
          for i in range(3)]
G = np.column_stack([lane_n, lane_v, *thirds, F["context"], F["demand"]])
X_all = G[r["t_idx"].to_numpy()]
X_all = np.column_stack([X_all, r["policy_id"].to_numpy(),
                         r["pre_fix"].astype(float).to_numpy()])
y_all = r["total_delay_ref_s"].to_numpy()

# v0.2 ensemble median point forecast on CAL
dev = torch.device("cpu")
nets = load_members(RUN, dev, log=lambda *a: None)
cal_idx = r.index[r["split"] == "CALIBRATION"].to_numpy()
mix = forward_mixture(nets, ds, cal_idx)
probs = mix if isinstance(mix, np.ndarray) else mix[0]
met = EV.met()
med = met.quantile(probs, 0.5)
r.loc[cal_idx, "v02_med"] = med

out = []
for tau in sorted(r["tau"].unique()):
    tr = (r["split"] == "TRAIN") & np.isclose(r["tau"], tau)
    ca = (r["split"] == "CALIBRATION") & np.isclose(r["tau"], tau)
    Xtr, ytr = X_all[tr.to_numpy()], y_all[tr.to_numpy()]
    Xca, yca = X_all[ca.to_numpy()], y_all[ca.to_numpy()]
    gbm = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
                                        learning_rate=0.05,
                                        random_state=0).fit(Xtr, ytr)
    rid = Ridge(alpha=1.0).fit(Xtr, ytr)
    # climatology: TRAIN median per (policy, demand)
    trd = r[tr]
    clim = trd.groupby(["policy", "demand_veh_h"])["total_delay_ref_s"].median()
    cad = r[ca]
    cpred = np.array([clim.get((p, d), trd["total_delay_ref_s"].median())
                      for p, d in zip(cad["policy"], cad["demand_veh_h"])])
    for chunk in ["chunk1", "chunk2", "all"]:
        m = np.ones(len(cad), bool) if chunk == "all" else \
            (cad["chunk_id"] == chunk).to_numpy()
        yy = yca[m]
        out.append(dict(
            tau=tau, chunk=chunk, n=int(m.sum()),
            mean_y=round(float(yy.mean()), 1),
            mae_clim=round(float(np.abs(cpred[m] - yy).mean()), 1),
            mae_gbm=round(float(np.abs(gbm.predict(Xca[m]) - yy).mean()), 1),
            mae_ridge=round(float(np.abs(rid.predict(Xca[m]) - yy).mean()), 1),
            mae_v02=round(float(np.abs(cad["v02_med"].to_numpy()[m] - yy)
                                .mean()), 1),
            bias_v02=round(float((cad["v02_med"].to_numpy()[m] - yy).mean()),
                           1)))
df = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
Path(__file__).with_suffix(".json").write_text(df.to_json(orient="records",
                                                          indent=1))
