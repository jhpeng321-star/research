"""S-MAPPODAMP analysis (zero simulation). Paired-by-world descriptive statistics.

Inputs: evidence/*_vehicle_metrics.json (this task's remote reduction of recorded views),
WMBREAKDOWN observables (registered 10 s region speed bins, BIC events, delay clocks),
COOPBEHAV per-episode detector counts (existing MAPPO/CTRL0 outputs + this task's
native/CIRCLES run of the unchanged detector code), COOPAUDIT C_G summary.
Sign convention: gain = native - arm team delay (veh-s, includes source waiting);
reduction = native - arm indicator. Positive = arm better / smoother.
"""
import csv, gzip, hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EV = ROOT / 'evidence'
WMB = Path('E:/agent-work/c4-wmbreakdown/docs/paper_results/wmbreakdown')
CB = Path('E:/agent-work/c4-coopbehav/tools/coopbehav/evidence')
B = 10000; RNG_SEED = 20261102
ARMS = ['ctrl0', 'mappo', 'pi', 'circles']
GROUPS = ['cav', 'hdv_follow_cav', 'hdv', 'all', 'upstream_all']
INPUTS = {}


def load(p):
    p = Path(p); INPUTS[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(gzip.open(p, 'rt').read() if p.suffix == '.gz' else p.read_text())


def split_of(seed):
    if 969400 <= seed <= 969431: return 'dev32'
    if 969432 <= seed <= 969439: return 'eval8'
    if 969550 <= seed <= 969599: return 'rep50'
    raise ValueError(seed)


def roughness(x):
    x = np.array([np.nan if v is None else v for v in x], float)
    d = np.diff(x); d = d[np.isfinite(d)]
    return float(np.sqrt((d * d).mean())) if d.size else np.nan


# ---- per-episode table ---------------------------------------------------------
E = {}
for corpus in ('wmdata', 'wmdata2'):
    vm = load(EV / f'{corpus}_vehicle_metrics.json')
    for r in vm['rows']:
        m, wv = r['moments'], r['within_var']; rec = {}
        for g in GROUPS:
            n, s1, s2 = m[g + '|a']; rec[g + '_arms'] = np.sqrt(s2 / n) if n else np.nan
            n, s1, s2 = m[g + '|j']; rec[g + '_jrms'] = np.sqrt(s2 / n) if n else np.nan
            n, s1, s2 = m[g + '|v']; rec[g + '_vmean'] = s1 / n if n else np.nan
            n, ss = wv[g]; rec[g + '_vvar'] = ss / n if n else np.nan
        rec.update(demand=r['demand'], pen=r['penetration'])
        E[(r['seed'], r['arm'])] = rec
    ob = load(WMB / f'{corpus}_observables.json')
    for e in ob['episodes']:
        k = (e['seed'], e['arm'])
        if k not in E: continue
        rec = E[k]
        for h in ('300', '600'): rec['J' + h] = e['clocks'][h]['team_delay_vs']
        sp = e['bins']['upstream']['speed']            # 6 prehistory + 60 post-t0 bins
        rec['up_rough300'] = roughness(sp[6:36]); rec['up_rough600'] = roughness(sp[6:66])
        ev = e['bins']['upstream']['events']['600']
        rec['bic600_depth'] = -ev['drop_mps'] if ev.get('event') else 0.0

# COOPBEHAV per-episode counts (events per CAV-hour)
BEH = ['yield_to_merge', 'gap_creation', 'cross_lane_hold', 'wave_damping']
cb_sources = [CB / 'WMDATA_final/episodes.json', CB / 'WMDATA2_final/episodes.json',
              EV / 'coopbehav_native_circles/WMDATA/episodes.json',
              EV / 'coopbehav_native_circles/WMDATA2/episodes.json']
for p in cb_sources:
    for e in load(p):
        k = (e['seed'], e['arm'])
        if k in E:
            E[k]['cav_s'] = e['exposure_cav_s']
            for b in BEH: E[k]['cb_' + b] = e['counts'].get(b, 0)

seeds = sorted({s for s, _ in E})
rng = np.random.default_rng(RNG_SEED)
BOOT = {}


def boot_idx(n):
    if n not in BOOT: BOOT[n] = rng.integers(0, n, size=(B, n))
    return BOOT[n]


def ci(vals_fn, n):
    idx = boot_idx(n); out = np.array([vals_fn(i) for i in idx]); out = out[np.isfinite(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out.size else (np.nan, np.nan)


def paired(split, arm, key):
    ss = [s for s in seeds if split_of(s) == split and (s, arm) in E and key in E[(s, arm)] and key in E[(s, 'native')]]
    nat = np.array([E[(s, 'native')][key] for s in ss]); a = np.array([E[(s, arm)][key] for s in ss])
    return ss, nat, a


rows = []
KEYS = [f'{g}_{m}' for g in GROUPS for m in ('arms', 'jrms', 'vvar', 'vmean')] + ['up_rough300', 'up_rough600', 'bic600_depth', 'J300', 'J600']
for split in ('dev32', 'eval8', 'rep50'):
    for arm in ARMS:
        for key in KEYS:
            ss, nat, a = paired(split, arm, key)
            if not ss: continue
            red = nat - a; n = len(ss)
            pct = lambda i: (a[i].mean() / nat[i].mean() - 1) * 100
            rows.append(dict(split=split, arm=arm, metric=key, n_worlds=n, native_mean=nat.mean(), arm_mean=a.mean(),
                             reduction_mean=red.mean(), red_lo=ci(lambda i: red[i].mean(), n)[0], red_hi=ci(lambda i: red[i].mean(), n)[1],
                             pct_change=pct(np.arange(n)), pct_lo=ci(pct, n)[0], pct_hi=ci(pct, n)[1],
                             share_worlds_smoother=float((red > 0).mean())))
with open(EV / 'paired_indicators.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- cooperation-behaviour rates -------------------------------------------------
crow = []
for split in ('dev32', 'eval8', 'rep50'):
    for b in BEH:
        ss = [s for s in seeds if split_of(s) == split and all('cb_' + b in E.get((s, x), {}) for x in ('native', 'mappo', 'ctrl0', 'circles'))]
        n = len(ss)
        if not n: continue
        def rate(arm, i):
            c = np.array([E[(ss[j], arm)]['cb_' + b] for j in i]); x = np.array([E[(ss[j], arm)]['cav_s'] for j in i])
            return c.sum() / x.sum() * 3600
        allidx = np.arange(n)
        for arm in ('native', 'ctrl0', 'mappo', 'circles'):
            r0 = rate(arm, allidx); lo, hi = ci(lambda i: rate(arm, i), n)
            rr = rate(arm, allidx) / rate('native', allidx) if rate('native', allidx) > 0 else np.nan
            rlo, rhi = ci(lambda i: rate(arm, i) / rate('native', i) if rate('native', i) > 0 else np.nan, n)
            crow.append(dict(split=split, behaviour=b, arm=arm, n_worlds=n,
                             count=int(sum(E[(s, arm)]['cb_' + b] for s in ss)), per_cav_hour=r0, lo=lo, hi=hi,
                             ratio_vs_native=rr, ratio_lo=rlo, ratio_hi=rhi))
with open(EV / 'coopbehav_rates.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(crow[0])); w.writeheader(); w.writerows(crow)

# ---- attribution: within-world association of gain with smoothing ----------------
PRED = ['cav_arms', 'hdv_follow_cav_arms', 'all_arms', 'all_vvar', 'up_rough300', 'cav_vmean']


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X1, y, rcond=None)[0]
    r = y - X1 @ b; tss = ((y - y.mean()) ** 2).sum()
    return b, (1 - (r * r).sum() / tss) if tss > 0 else np.nan


arow = []
for split in ('dev32', 'eval8', 'rep50'):
    for arm in ('mappo', 'circles'):
        for h in ('300', '600'):
            ss = [s for s in seeds if split_of(s) == split and (s, arm) in E]
            n = len(ss)
            y = np.array([E[(s, 'native')]['J' + h] - E[(s, arm)]['J' + h] for s in ss])
            for p in PRED + ['JOINT_smoothing']:
                cols = ['cav_arms', 'hdv_follow_cav_arms', 'all_arms', 'up_rough300'] if p == 'JOINT_smoothing' else [p]
                X = np.array([[E[(s, 'native')][c] - E[(s, arm)][c] for c in cols] for s in ss])
                if p == 'JOINT_smoothing' and n <= len(cols) + 2: continue
                b, r2 = ols(X, y)
                def stat(i, which):
                    bb, rr = ols(X[i], y[i])
                    return rr if which == 'r2' else (bb[0] / y[i].mean() if which == 'icpt_share' else bb[1])
                rec = dict(split=split, arm=arm, horizon_s=h, predictor=p, n_worlds=n, mean_gain=y.mean(),
                           gain_lo=ci(lambda i: y[i].mean(), n)[0], gain_hi=ci(lambda i: y[i].mean(), n)[1],
                           r2=r2, r2_lo=ci(lambda i: stat(i, 'r2'), n)[0], r2_hi=ci(lambda i: stat(i, 'r2'), n)[1],
                           intercept=b[0], intercept_share=b[0] / y.mean(),
                           icpt_share_lo=ci(lambda i: stat(i, 'icpt_share'), n)[0], icpt_share_hi=ci(lambda i: stat(i, 'icpt_share'), n)[1])
                if len(cols) == 1:
                    rec.update(slope=b[1], slope_lo=ci(lambda i: stat(i, 'slope'), n)[0], slope_hi=ci(lambda i: stat(i, 'slope'), n)[1])
                    xs = X[:, 0]; rk = lambda v: np.argsort(np.argsort(v))
                    rec['spearman'] = float(np.corrcoef(rk(xs), rk(y))[0, 1])
                arow.append(rec)
with open(EV / 'attribution.csv', 'w', newline='') as f:
    keys = sorted({k for r in arow for k in r}, key=lambda k: list(arow[0]).index(k) if k in arow[0] else 99)
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(arow)

coop = (Path('E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md'))
INPUTS[str(coop)] = hashlib.sha256(coop.read_bytes()).hexdigest()
for p in (HERE / 'reduce_remote.py', HERE / 'collect.py', Path(__file__)):
    INPUTS[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
outs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (EV / 'paired_indicators.csv', EV / 'coopbehav_rates.csv', EV / 'attribution.csv')}
(EV / 'manifest.json').write_text(json.dumps(dict(analysis='S-MAPPODAMP', bootstrap=dict(B=B, seed=RNG_SEED, unit='whole world, within split'),
                                                   inputs=INPUTS, outputs=outs), indent=1))
print('ok', len(rows), len(crow), len(arow))
