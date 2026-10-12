"""Print the C4-LANECAL old-vs-new precision/recall table (mainline requests; ramp reported separately)."""
import sys
sys.path.insert(0, "tools")
from lanecal_fit import *

rows = load("runs/active/C4-LANECAL/data"); a = arr(rows)
lane = np.array([r["lane"] for r in rows]); m0 = lane < 2
tr = ~np.isin(a["seed"], list(HELD))
OLD = (6, 1, 1, 6, 1, 1); NEW = (1, 1.2, 4.5, 3, 1.25, 6)
print(f"requests {len(rows)} (mainline {int(m0.sum())}, ramp {int((~m0).sum())}); accepted {a['acc'].mean():.3f}")
print("| split | subset | n req | SUMO accept | old prec | old rec | old n pred | new prec | new rec | new n pred |")
print("|---|---|---|---|---|---|---|---|---|---|")
for split, ms in (("train (seeds 4107-4110)", tr), ("held-out (4111, 4112)", ~tr)):
    for name, m in (("all mainline", m0), ("left (1->0)", m0 & a["left"]), ("right (0->1)", m0 & ~a["left"]),
                    ("ramp (2->1)", ~m0)):
        mm = ms & m
        A = {k: v[mm] for k, v in a.items()}
        po, pn = predq(A, OLD, False), predq(A, NEW, False)
        (p1, r1), (p2, r2) = prec_rec(po, A["acc"]), prec_rec(pn, A["acc"])
        print(f"| {split} | {name} | {int(mm.sum())} | {A['acc'].mean():.3f} | {p1:.3f} | {r1:.3f} | {int(po.sum())} | {p2:.3f} | {r2:.3f} | {int(pn.sum())} |")
