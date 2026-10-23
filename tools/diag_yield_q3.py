"""S-YIELD Q3: is yield propensity consistent per driver and predictable from the driver's own past?"""
import sys, json
sys.path.insert(0, "tools")
import numpy as np, pandas as pd
from diag_yield_common import *

d = load_samples()
cid = cells(d); hdv = d["cav"] == 0; l1 = d["lane"] == 1
ctrl1 = l1 & hdv & (d["nRamp"] == 0)
u1, inv1 = np.unique(cid, return_inverse=True)
n1 = np.bincount(inv1[ctrl1], minlength=len(u1)); mu1 = np.bincount(inv1[ctrl1], d["a"][ctrl1], minlength=len(u1)) / np.maximum(n1, 1)
mu1 = np.where(n1 >= 20, mu1, np.nan)
d["ex"] = d["a"] - mu1[inv1]
enc = hdv & l1 & (d["tf"] == 1) & (d["dA"] >= 0) & (d["dA"] < 20) & np.isfinite(d["ex"])
df = pd.DataFrame({k: d[k][enc] for k in ("ep", "vid", "vidR", "t", "ex", "a", "v", "dA", "dendR", "waitR", "vR", "dem", "pen")})
df["ep"] = df["ep"].astype(int); df["vid"] = df["vid"].astype(int); df["vidR"] = df["vidR"].astype(int)
# encounter = (ep, F, R)
g = df.groupby(["ep", "vid", "vidR"])
E = g.agg(n=("ex", "size"), ex=("ex", "mean"), t0=("t", "min"), dend=("dendR", "mean"), wait=("waitR", "mean"), vR=("vR", "mean"),
          dA=("dA", "mean"), vF=("v", "mean"), dem=("dem", "mean")).reset_index()
E = E[E.n >= 6].copy()   # >= 3 s of exposure
print("encounters (>=6 rows):", len(E), "drivers:", E.groupby(["ep", "vid"]).ngroups)
cnt = E.groupby(["ep", "vid"]).size()
print("encounters per driver: ", cnt.value_counts().sort_index().head(8).to_dict())
# situational adjustment: linear regression of ex on situation
S = np.column_stack([np.ones(len(E)), E.dend, E.dend ** 2, np.log1p(E.wait), E.vR, E.dA, E.vF, E.dem / 1000])
beta, *_ = np.linalg.lstsq(S, E.ex.values, rcond=None)
E["res"] = E.ex.values - S @ beta
print("situation R2 on encounter yield score: %.3f" % (1 - E.res.var() / E.ex.var()))
# consistency across encounters of the same driver
multi = E.sort_values("t0").groupby(["ep", "vid"]).filter(lambda x: len(x) >= 2)
first = multi.groupby(["ep", "vid"]).nth(0).reset_index(); second = multi.groupby(["ep", "vid"]).nth(1).reset_index()
m = first.merge(second, on=["ep", "vid"], suffixes=("1", "2"))
print("drivers with >=2 encounters:", len(m))
if len(m) > 10:
    print("corr(ex1, ex2) = %.3f; corr(res1, res2) = %.3f" % (np.corrcoef(m.ex1, m.ex2)[0, 1], np.corrcoef(m.res1, m.res2)[0, 1]))
# within-encounter split-half (first half vs second half of rows): measures measurement noise vs stable level
dd = df.sort_values("t"); dd["rk"] = dd.groupby(["ep", "vid", "vidR"]).cumcount(); dd["nn"] = dd.groupby(["ep", "vid", "vidR"])["ex"].transform("size")
dd = dd[dd.nn >= 8]; h1 = dd[dd.rk % 2 == 0].groupby(["ep", "vid", "vidR"]).ex.mean(); h2 = dd[dd.rk % 2 == 1].groupby(["ep", "vid", "vidR"]).ex.mean()
print("within-encounter odd/even-row corr: %.3f (n=%d)" % (np.corrcoef(h1.values, h2.values)[0, 1], len(h1)))

# driver features from the PAST (pre-zone rows s in [-1500,-60), strictly before first encounter row)
R = np.load(OUT + "/driver_rows.npy").astype(np.float64)
cols = ["ep", "vid", "t", "s", "v", "a", "gap", "vl"]
P = pd.DataFrame(R, columns=cols); P["ep"] = P.ep.astype(int); P["vid"] = P.vid.astype(int)
first_enc = E.groupby(["ep", "vid"]).t0.min().rename("tenc").reset_index()
P = P.merge(first_enc, on=["ep", "vid"], how="inner"); P = P[P.t < P.tenc]
P["thw"] = np.where((P.v > 2) & np.isfinite(P.gap), P.gap / P.v, np.nan)
P["fol"] = np.isfinite(P.gap) & (P.gap < 60) & (P.v > 1)
P["dv"] = P.vl - P.v
P["apos"] = np.where(P.a > 0.05, P.a, np.nan); P["aneg"] = np.where(P.a < -0.05, P.a, np.nan)
P["fa"] = np.where(P.fol, P.a, np.nan)
P["fa_pos"] = np.where(P.fol & (P.a > 0.05), P.a, np.nan); P["fa_neg"] = np.where(P.fol & (P.a < -0.05), P.a, np.nan)
P["thw_f"] = np.where(P.fol, P.thw, np.nan)
P["fcl"] = np.where(P.fol & (P.dv < -0.5), P.a, np.nan)    # accel while closing on the leader (braking response)
gp = P.groupby(["ep", "vid"])
F = pd.DataFrame({"n_rows": gp.size(), "thw_med": gp.thw_f.median(), "thw_p20": gp.thw_f.quantile(0.2), "apos_p90": gp.fa_pos.quantile(0.9),
                  "aneg_p10": gp.fa_neg.quantile(0.1), "v_p90": gp.v.quantile(0.9), "fol_share": gp.fol.mean(),
                  "closing_a_med": gp.fcl.median()}).reset_index()
F = F[F.n_rows >= 60]
J = E.merge(F, on=["ep", "vid"], how="inner")
print("encounters with >=30 s of past:", len(J), "of", len(E))
feat = ["thw_med", "thw_p20", "apos_p90", "aneg_p10", "v_p90", "fol_share", "closing_a_med"]
J = J.dropna(subset=feat)
print("complete:", len(J))
print("feature-vs-yield correlations (raw score / situation-adjusted residual):")
for f in feat:
    print("  %-14s r_raw=%+.3f  r_res=%+.3f" % (f, np.corrcoef(J[f], J.ex)[0, 1], np.corrcoef(J[f], J.res)[0, 1]))
# group-CV (by episode) ridge: situation only vs situation + past behaviour
rng = np.random.default_rng(0)
eps = J.ep.unique(); folds = {e: i % 5 for i, e in enumerate(rng.permutation(eps))}
J["fold"] = J.ep.map(folds)
def design(df, with_beh):
    cols = [np.ones(len(df)), df.dend, df.dend ** 2, np.log1p(df.wait), df.vR, df.dA, df.vF, df.dem / 1000]
    if with_beh:
        for f in feat:
            z = (df[f] - J[f].mean()) / (J[f].std() + 1e-9); cols.append(z.clip(-3, 3))
    return np.column_stack(cols)
def cv(with_beh, lam=3.0):
    pred = np.zeros(len(J))
    for k in range(5):
        tr = J.fold != k; te = ~tr
        X = design(J[tr], with_beh); y = J.ex[tr].values
        A = X.T @ X + lam * np.eye(X.shape[1]); A[0, 0] -= lam
        b = np.linalg.solve(A, X.T @ y)
        pred[te.values] = design(J[te], with_beh) @ b
    return 1 - ((J.ex - pred) ** 2).sum() / ((J.ex - J.ex.mean()) ** 2).sum()
r_s = cv(False); r_b = cv(True)
print("held-out-episode R2: situation only %.3f ; + past behaviour %.3f" % (r_s, r_b))
# bootstrap CI of the delta
dl = []
for b in range(100):
    pass
# tertile table of yield by past headway (adjusted residual)
J["thw_t"] = pd.qcut(J.thw_med, 3, labels=["tight", "mid", "loose"])
print(J.groupby("thw_t", observed=True)[["ex", "res"]].agg(["mean", "count"]).round(3).to_string())
J["ap_t"] = pd.qcut(J.apos_p90, 3, labels=["gentle", "mid", "sharp"])
print(J.groupby("ap_t", observed=True)[["ex", "res"]].agg(["mean", "count"]).round(3).to_string())
J["an_t"] = pd.qcut(J.aneg_p10, 3, labels=["hard-brake", "mid", "soft"])
print(J.groupby("an_t", observed=True)[["ex", "res"]].agg(["mean", "count"]).round(3).to_string())
E.to_csv(OUT + "/encounters.csv", index=False)
