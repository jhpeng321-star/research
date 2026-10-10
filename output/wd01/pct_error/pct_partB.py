"""Part B v0.3.2 strict CAL error as % of the mean true target, per tau (report-only,
no selection). MAE numbers from eval_report sections 6/7 (metrics.json item6_baselines,
aggregated n-weighted over chunk x policy); mean true target from the v0.3.2 root CAL rows."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from partB_v0.data import load_manifest, V0Dataset
ROOT = "E:/research-dense/output/wd01_partB_v0data_v032"
RES = Path("E:/research-dense/output/partB_results/v0.3.2_strict/metrics.json")
col = sys.argv[1] if len(sys.argv) > 1 else "total_delay_ref_s"
ds = V0Dataset(load_manifest(ROOT), load_features=False, log=lambda *a: None)
r = ds.rows; cal = r[r["split"] == "CALIBRATION"]
b = json.load(open(RES))["eval"]["item6_baselines"]["rows"]
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for x in b:
    a = agg[x["tau"]]; a["n"] += x["n"]
    for k in ("mae_climatology", "mae_ridge", "mae_model", "crps_climatology", "crps_ridge", "crps_model"):
        a[k] += x[k] * x["n"]
print(f"target col {col}; tau n meanY | MAE% model ridge clim | CRPS% model ridge clim")
for tau in sorted(agg):
    a = agg[tau]; n = a["n"]; y = cal.loc[np.isclose(cal["tau"], tau), col]
    my = float(y.mean())
    f = lambda k: 100 * a[k] / n / my
    print(f"{tau:.0f} {int(n)}/{len(y)} {my:.0f} | {f('mae_model'):.1f} {f('mae_ridge'):.1f} {f('mae_climatology'):.1f} | {f('crps_model'):.1f} {f('crps_ridge'):.1f} {f('crps_climatology'):.1f}")
