"""S-BASEDIAG Q3: inference-only counterfactual forms on the model's teacher-forced replay rows.

Arms (residual handling stated per arm): model as is; IDM core with the ramp partner as a virtual leader
(min(core, IDM to partner), encoded parameters, residual kept); calibrated core (a 1.2, b 2.0, encoded
T/s0/v0, no residual); calibrated core + virtual leader; calibrated core + hold (min(core, 0) on partner rows).
Diagnostic only: none of these is proposed as a hand-set rule.
"""
import sys

import numpy as np

sys.path.insert(0, "tools")
from diag_base_decomp_an import load  # noqa: E402
from diag_base_params_cf import idm  # noqa: E402

for prefix in sys.argv[1:]:
    D = load(prefix)
    P = D["params"]
    v, vl, gap, rec = D["v"], D["lead_v"], D["gap"], D["rec"]
    has = D["has"].astype(bool)
    tm = D["tmerge"]
    zone = has & (D["dend"] >= 0) & (D["dend"] <= 75.44)
    vr = np.where(np.isfinite(D["v_r"]), D["v_r"], v)
    gr = np.where(has & (D["gap_r"] > 0), D["gap_r"], 1e4)
    virt_enc = np.where(has & (D["gap_r"] > 0), D["a_virt"], np.inf)
    cal = idm(v, vl, gap, P[:, 0], P[:, 1], P[:, 2], np.full_like(v, 1.2), np.full_like(v, 2.0))
    virt_cal = idm(v, vr, gr, P[:, 0], P[:, 1], P[:, 2], np.full_like(v, 1.2), np.full_like(v, 2.0))
    arms = {
        "model as is": D["final"],
        "model + virtual leader": np.clip(np.minimum(D["phys"], virt_enc) + D["corr"], -9, 3),
        "calibrated core, no residual": cal,
        "calibrated + virtual leader": np.where(has, np.minimum(cal, virt_cal), cal),
        "calibrated + hold": np.where(has, np.minimum(cal, 0.0), cal),
    }
    groups = {
        "partners last 3 s": zone & (tm > 0) & (tm <= 3),
        "partners [-6,-3) s": zone & (tm > 3) & (tm <= 6),
        "partners, merge > 10 s away / never": zone & ~(tm <= 10),
        "far partner gap_r > 30": zone & (D["gap_r"] > 30),
        "partner overlapping gap_r < 0": zone & (D["gap_r"] < 0),
        "unpartnered zone": ~has & (D["s"] >= 0) & (D["s"] < 75.44),
        "all rows": np.ones_like(has),
    }
    print(f"\n# {prefix}: mean (arm - recorded) [MAE]")
    print("group | n | rec | " + " | ".join(arms))
    for g, m in groups.items():
        print(f"{g} | {int(m.sum())} | {rec[m].mean():+.2f} | " + " | ".join(
            f"{a[m].mean() - rec[m].mean():+.2f} [{np.abs(a[m] - rec[m]).mean():.2f}]" for a in arms.values()))
