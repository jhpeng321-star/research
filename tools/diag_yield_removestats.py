"""S-YIELD Q4b: WM causal removal test statistics (with R minus without R, same lane seed)."""
import pickle, sys, collections
sys.path.insert(0, "tools")
import numpy as np
PKL = sys.argv[1]
z = pickle.load(open(PKL, "rb")); res = [r for r in z["results"] if not r.get("error")]
by = collections.defaultdict(dict)
for r in res:
    by[(r["var"], r["idx"], r["rid"], r["sample"])][r["tag"]] = r
rows = []
for key, pair in by.items():
    if "with" not in pair or "without" not in pair:
        continue
    a, b = pair["with"], pair["without"]
    fid = a["fid"]
    if fid not in a["ids"] or fid not in b["ids"]:
        continue
    ia, ib = a["ids"].index(fid), b["ids"].index(fid)
    va, vb = a["v"][:, ia].astype(float), b["v"][:, ib].astype(float)
    pa, pb = a["pos"][:, ia].astype(float), b["pos"][:, ib].astype(float)
    ma, mb = a["mask"][:, ia], b["mask"][:, ib]
    ok = ma & mb
    rows.append(dict(var=key[0], idx=key[1], kind=a["kind"], dA=a["dA"], v0=va[0], dv=va - vb, dpos=pa - pb, ok=ok))
print("pairs", len(rows), collections.Counter((r["var"], r["kind"]) for r in rows))
def boot(vals, n=500):
    vals = np.asarray(vals); vals = vals[np.isfinite(vals)]
    if len(vals) < 5: return (np.nan, np.nan)
    rng = np.random.default_rng(0); m = [rng.choice(vals, len(vals)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])
print("| model | kind | n pairs | Δv(F) at 3 s | 6 s | 10 s | 20 s | Δpos(F) at 10 s | share with Δv(10 s) < -0.5 | mean Δa over 0-6 s |")
for var in ("r9", "r8"):
    for kind in ("near", "far"):
        R = [r for r in rows if r["var"] == var and r["kind"] == kind]
        if not R: continue
        def at(t, k="dv"):
            i = int(t * 2); x = np.array([r[k][i] if r["ok"][i] else np.nan for r in R]); return x
        out = []
        for t in (3, 6, 10, 20):
            x = at(t); ci = boot(x); out.append("%+.2f [%+.2f,%+.2f]" % (np.nanmean(x), *ci))
        x10 = at(10); dp = at(10, "dpos"); da = at(6) / 6.0
        print("| %s | %s | %d | %s | %+.2f [%+.2f,%+.2f] | %.2f | %+.3f |" % (var, kind, len(R), " | ".join(out), np.nanmean(dp), *boot(dp), np.nanmean(x10 < -0.5), np.nanmean(da)))
# dependence of removal effect on dA for near pairs (r9 + r8)
for var in ("r9", "r8"):
    print(var, "near pairs: mean Δv(10 s) by dA bin and F speed bin")
    R = [r for r in rows if r["var"] == var and r["kind"] == "near"]
    for lo, hi in [(3, 8), (8, 14), (14, 25)]:
        S = [r for r in R if lo <= r["dA"] < hi]
        x = np.array([r["dv"][20] if r["ok"][20] else np.nan for r in S]); print("   dA %d-%d n=%d mean Δv(10s) %+.2f" % (lo, hi, len(S), np.nanmean(x)))
