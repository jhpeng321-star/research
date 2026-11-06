"""S-BASEDIAG Q2: which simple description fits SUMO target followers (recorded TRAIN rows, textbook IDM)?
Descriptive comparison only (no model rule): own-leader IDM, + ramp car as a full virtual leader,
"hold" (do not accelerate: min(own, 0)), speed matching to the ramp car (min(own, (v_r - v) / 1 s)).
"""
import sys

import numpy as np

sys.path.insert(0, "tools")
from diag_base_vleader import idm  # noqa: E402

D = dict(np.load(sys.argv[1] if len(sys.argv) > 1 else "tmp_basediag/vleader.npz"))
a, b, T, s0 = 1.2, 2.0, 1.2, 2.0
v, rec = D["v"], D["rec"]
own = idm(v, D["v_l"], D["gap_l"], a, b, T, s0)
vr = np.where(np.isfinite(D["v_r"]), D["v_r"], v)
virt = idm(v, vr, np.nan_to_num(D["gap_r"], nan=1e3), a, b, T, s0)
forms = {"own IDM": own, "full virtual leader": np.minimum(own, virt), "hold min(own,0)": np.minimum(own, 0.0),
         "match v_r": np.minimum(own, (vr - v) / 1.0)}
hp = D["has_p"] > 0
zone = hp & (D["dend"] >= 0) & (D["dend"] <= 75.44)
tm = D["tmerge"]
groups = {"last 3 s": zone & (tm > 0) & (tm <= 3), "[-6,-3) s": zone & (tm > 3) & (tm <= 6),
          "[-10,-6) s": zone & (tm > 6) & (tm <= 10), "zone partner, merge > 10 s / never": zone & ~(tm <= 10),
          "far partner gap_r > 30": zone & (D["gap_r"] > 30),
          "unpartnered zone rows": ~hp & (D["s"] >= 0) & (D["s"] < 75.44),
          "unpartnered far upstream s<-400": ~hp & (D["s"] < -400)}
print("group | n | rec | " + " | ".join(f"{k} bias [MAE]" for k in forms))
for g, m in groups.items():
    print(f"{g} | {int(m.sum())} | {rec[m].mean():+.2f} | " + " | ".join(
        f"{f[m].mean() - rec[m].mean():+.2f} [{np.abs(f[m] - rec[m]).mean():.2f}]" for f in forms.values()))
