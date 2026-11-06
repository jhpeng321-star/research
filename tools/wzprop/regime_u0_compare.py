"""WMEV comparison: WM-predicted vs SUMO-measured benefit of the rule strategies on the MAPPO u0 base, per cell.

SUMO (ground truth, read-only): S-REGIMEU0 (regimeu0/REGIMEU0_RESULTS.jsonl f473caad + per-job SUMMARY.json.gz
per-second series; 6 cells x {wall_h_solo2, wall_h} (+ wall_h_solo0 at D3200), 24 paired seeds) and S-LOWSHARE-U0
(lowshare_u0/LOWSHARE_U0_RESULTS.jsonl 3b90b6b9 + run/jobs; D3200_p30/p20 x {wall_h_solo2, wall_h, wall_g}, 36 seeds).
WM: ``regime_u0 score`` JSONL (start states at t0 = 300 s of OTHER TRAIN-type worlds of the same cells).

Sign convention (re-derived here, both sides identical): benefit = J(u0) - J(strategy) over [300, 300 + H) s, veh-s;
POSITIVE = the strategy reduces delay.  (S-REGIMEU0 tables report diff = strategy - u0, negative = better.)
Primary series: road delay (the SUMO gain is entirely road delay; source waiting unchanged).  Team (road + source
waiting) is reported as secondary.

Per cell x strategy: WM mean benefit (mean over start states of the mean over K draws), world-bootstrap CI,
P_WM(benefit > 0) (share of state x draw futures with benefit > 0), relative benefit; SUMO mean benefit, seed-bootstrap
CI, benefit fraction.  Across pairs: sign agreement (all / SUMO-confident pairs), Spearman (pooled pairs, and across
cells per strategy), calibration of P(benefit > 0) (MAE vs the SUMO benefit fraction, Brier vs the SUMO per-seed
outcomes, reliability bins), regime pattern (help / ~0 / hurt per cell), and the group-vs-best-solo dominance.
Bootstrap: 2,000 resamples (RNG 20261106) of WM start states and SUMO seeds within each cell (arms paired).
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import geometry as G

SCHEMA = 'wmev.regime_u0_compare.v1'
T0_S, WINDOW_S, N_WIN = 300, 100, 9
B_BOOT, RNG = 2000, 20261106
BASE = 'u0'
SOLOS = ('wall_h_solo2', 'wall_h_solo0', 'wall_g_solo2')
GROUPS = ('wall_h', 'wall_g')
RUN_ROOT = 'E:/research-dense/runs/active/C4-V7FT-20261017'
DEFAULT_SUMO = (f'{RUN_ROOT}/regimeu0', f'{RUN_ROOT}/lowshare_u0')
RESULT_FILES = {'regimeu0': 'REGIMEU0_RESULTS.jsonl', 'lowshare_u0': 'LOWSHARE_U0_RESULTS.jsonl'}
HORIZON_RULE = dict(pooled_ratio=(0.5, 2.0), cell_ratio=(1 / 3, 3.0), max_error_share=0.0, max_no_slot_share=0.0)
ZERO_BAND_REL = 0.05             # '~0' when the CI covers 0 (or |relative benefit| < 5 % for point classifications)


# ================================================================================ SUMO side
def _series_windows(summary, key, t0=T0_S, w=WINDOW_S, n=N_WIN):
    """Per-window increments of a cumulative per-second series (regimeu0 / imitreplay window_j semantics)."""
    keys = summary['series_keys']
    i = keys.index(key) + 1
    inc, prev = {}, None
    for row in summary['series']:
        cur = float(row[i])
        inc[int(row[0])] = cur - (prev if prev is not None else 0.0)
        prev = cur
    out = []
    for j in range(n):
        ts = [t for t in inc if t0 + j * w <= t < t0 + (j + 1) * w]
        out.append(sum(inc[t] for t in ts) if len(ts) == w else None)
    return out


def _job_dir(base, row):
    jobs = Path(base) / row['root'] / 'jobs'
    for name in (f"{row['cell']}_s{row['seed']}_{row['arm']}", f"s{row['seed']}_{row['arm']}"):
        if (jobs / name).exists():
            return jobs / name
    if (Path(base) / 'run' / 'jobs').exists() and row['root'] == 'run':
        return Path(base) / 'run' / 'jobs' / f"{row['cell']}_s{row['seed']}_{row['arm']}"
    raise FileNotFoundError(f"no job dir for {row['root']} {row['cell']} s{row['seed']} {row['arm']}")


def sumo_windows(bases=DEFAULT_SUMO):
    """{cell: {arm: {seed: dict(road_w, team_w, src_w)}}} from the per-job series; checked against the RESULTS rows."""
    out = defaultdict(lambda: defaultdict(dict))
    report = dict(rows=0, checked=0, max_abs_dev=0.0, sources={})
    for base in bases:
        base = G.check_path(base)
        res = base / RESULT_FILES[base.name]
        report['sources'][base.name] = str(res)
        for line in res.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('status') != 'ok':
                continue
            G.check_seed(r['seed'], where='(SUMO result)')
            jd = _job_dir(base, r)
            with gzip.open(jd / 'SUMMARY.json.gz', 'rt') as fh:
                s = json.load(fh)
            wd = s['world']
            if (wd.get('cell_key', r['cell']) != r['cell'] or int(wd['seed']) != int(r['seed'])
                    or s.get('arm', r['arm']) != r['arm']):
                raise ValueError(f'{jd}: world/arm mismatch')
            team = _series_windows(s, 'team_delay_vs')
            road = _series_windows(s, 'road_delay_vs')
            src = _series_windows(s, 'source_wait_vs')
            if any(x is None for x in team + road + src):
                raise ValueError(f'{jd}: incomplete series')
            dev = abs(sum(team) - float(r['J_300_1200_vs']))
            report['max_abs_dev'] = max(report['max_abs_dev'], dev)
            if dev > 1e-6 * max(1.0, abs(float(r['J_300_1200_vs']))):
                raise ValueError(f'{jd}: window sum {sum(team)} != registered J {r["J_300_1200_vs"]}')
            report['checked'] += 1
            out[r['cell']][r['arm']][int(r['seed'])] = dict(road_w=road, team_w=team, src_w=src)
            report['rows'] += 1
    return {c: {a: dict(v) for a, v in d.items()} for c, d in out.items()}, report


def write_sumo_windows(out_path, bases=DEFAULT_SUMO):
    data, report = sumo_windows(bases)
    with G.check_path(out_path).open('x') as fh:
        fh.write(json.dumps(dict(schema=SCHEMA + '.sumo_windows', t0=T0_S, window_s=WINDOW_S, report=report,
                                 cells=data)))
    return report


def read_sumo_windows(path):
    d = json.loads(G.check_path(path).read_text())
    return {c: {a: {int(s): v for s, v in arms.items()} for a, arms in cell.items()} for c, cell in d['cells'].items()}


# ================================================================================ WM side
def wm_rows(score_paths):
    """{cell: {arm: {sid: dict(road_w=[K][W], src_w, n_w, ...)}}} + bookkeeping (errors, no-slot, CRN)."""
    out = defaultdict(lambda: defaultdict(dict))
    book = dict(states=0, errors=[], no_slot_rows=0, crn_violations=0, crn_checked=0, activations={}, meta=[])
    for p in score_paths:
        p = G.check_path(p)
        mp = Path(str(p) + '.meta.json')
        if mp.exists():
            book['meta'].append(json.loads(mp.read_text()))
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if 'error' in r:
                book['errors'].append(dict(sid=r['sid'], error=r['error']))
                continue
            book['states'] += 1
            for arm, o in r['arms'].items():
                if r['sid'] in out[r['cell']][arm]:          # first file wins (primary run before add-on runs)
                    book['duplicates'] = book.get('duplicates', 0) + 1
                    continue
                out[r['cell']][arm][r['sid']] = dict(o, seed=r['seed'])
                book['no_slot_rows'] += sum(1 for x in o['no_slot'] if x)
                if arm not in (BASE, 'native'):
                    for fa, fd in zip(o['first_act_s'], o['first_div_s']):
                        if fd is None:
                            continue
                        book['crn_checked'] += 1
                        if fa is None or fd < fa:
                            book['crn_violations'] += 1
                    book['activations'].setdefault(f"{r['cell']}|{arm}", []).append(
                        float(np.mean(o['activations'])))
    return {c: {a: dict(v) for a, v in d.items()} for c, d in out.items()}, book


# ================================================================================ estimands
def _wm_unit_gains(wm, cell, arm, n_w, series='road'):
    """[(state benefit draws over the first n_w windows)] per state sid, paired with u0 draws."""
    base, cand = wm[cell][BASE], wm[cell][arm]
    out = {}
    for sid in sorted(set(base) & set(cand)):
        def tot(o, k):
            if series == 'road':
                return sum(o['road_w'][k][:n_w])
            return sum(o['road_w'][k][:n_w]) + sum(o['src_w'][k][:n_w])
        K = min(len(base[sid]['road_w']), len(cand[sid]['road_w']))
        out[sid] = (np.array([tot(base[sid], k) - tot(cand[sid], k) for k in range(K)]),
                    np.array([tot(base[sid], k) for k in range(K)]))
    return out


def _sumo_unit_gains(sumo, cell, arm, n_w, series='road'):
    base, cand = sumo[cell][BASE], sumo[cell][arm]
    key = 'road_w' if series == 'road' else 'team_w'
    return {s: (sum(base[s][key][:n_w]) - sum(cand[s][key][:n_w]), sum(base[s][key][:n_w]))
            for s in sorted(set(base) & set(cand))}


def pairs_of(wm, sumo):
    pairs = []
    for cell in sorted(set(wm) & set(sumo)):
        if BASE not in wm[cell] or BASE not in sumo[cell]:
            continue
        for arm in sorted(set(wm[cell]) & set(sumo[cell])):
            if arm != BASE:
                pairs.append((cell, arm))
    return pairs


def _summ_wm(units, idx=None):
    keys = sorted(units)
    if idx is not None:
        keys = [keys[i] for i in idx]
    g = np.array([units[k][0].mean() for k in keys])
    j = np.array([units[k][1].mean() for k in keys])
    p_draw = float(np.mean([np.mean(units[k][0] > 0) for k in keys]))
    return dict(mean=float(g.mean()), rel=float(g.mean() / j.mean()) if j.mean() else None, p_pos=p_draw,
                p_state=float(np.mean(g > 0)), n=len(keys))


def _summ_sumo(units, idx=None):
    keys = sorted(units)
    if idx is not None:
        keys = [keys[i] for i in idx]
    g = np.array([units[k][0] for k in keys])
    j = np.array([units[k][1] for k in keys])
    return dict(mean=float(g.mean()), rel=float(g.mean() / j.mean()) if j.mean() else None,
                frac=float(np.mean(g > 0)), outcomes=(g > 0).astype(float), n=len(keys))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None

    def rank(x):
        order = np.argsort(x, kind='stable')
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        for v in np.unique(x):                      # average ties
            m = x == v
            r[m] = r[m].mean()
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def classify(lo, hi):
    return 'help' if lo > 0 else 'hurt' if hi < 0 else '~0'


def metrics(wm_s, su_s, pairs, su_conf=None):
    """Agreement metrics from per-pair summaries (point or one bootstrap replicate)."""
    m = {}
    w = [wm_s[p]['mean'] for p in pairs]
    s = [su_s[p]['mean'] for p in pairs]
    m['n_pairs'] = len(pairs)
    m['sign_agreement'] = float(np.mean([np.sign(a) == np.sign(b) for a, b in zip(w, s)])) if pairs else None
    if su_conf is not None:
        conf = [p for p in pairs if su_conf[p]]
        m['n_confident'] = len(conf)
        m['sign_agreement_confident'] = (float(np.mean([np.sign(wm_s[p]['mean']) == np.sign(su_s[p]['mean'])
                                                        for p in conf])) if conf else None)
    m['spearman_pooled'] = spearman(w, s)
    m['spearman_pooled_rel'] = spearman([wm_s[p]['rel'] for p in pairs], [su_s[p]['rel'] for p in pairs])
    for arm in sorted({a for _c, a in pairs}):
        ps = [p for p in pairs if p[1] == arm]
        if len(ps) >= 3:
            m[f'spearman_cells_{arm}'] = spearman([wm_s[p]['mean'] for p in ps], [su_s[p]['mean'] for p in ps])
            m[f'spearman_cells_rel_{arm}'] = spearman([wm_s[p]['rel'] for p in ps], [su_s[p]['rel'] for p in ps])
    pw = np.array([wm_s[p]['p_pos'] for p in pairs])
    fr = np.array([su_s[p]['frac'] for p in pairs])
    m['calib_mae'] = float(np.mean(np.abs(pw - fr))) if pairs else None
    m['calib_spearman'] = spearman(pw, fr)
    briers, brier_base = [], []
    base_rate = float(np.mean(np.concatenate([su_s[p]['outcomes'] for p in pairs]))) if pairs else None
    for p in pairs:
        o = su_s[p]['outcomes']
        briers.extend(list((wm_s[p]['p_pos'] - o) ** 2))
        brier_base.extend(list((base_rate - o) ** 2))
    m['brier'] = float(np.mean(briers)) if briers else None
    m['brier_base_rate'] = float(np.mean(brier_base)) if brier_base else None
    m['brier_skill'] = (1.0 - m['brier'] / m['brier_base_rate']) if briers and m['brier_base_rate'] else None
    return m


def dominance(wm_s, su_s, cells):
    """Group minus best solo (benefit units: positive = group better) per cell, WM and SUMO."""
    out = {}
    for cell in cells:
        for grp in GROUPS:
            if (cell, grp) not in wm_s:
                continue
            solos = [a for a in SOLOS if (cell, a) in wm_s and (cell, a) in su_s and a.startswith('wall_h')]
            if not solos:
                continue
            out[(cell, grp)] = dict(
                wm=wm_s[(cell, grp)]['mean'] - max(wm_s[(cell, a)]['mean'] for a in solos),
                sumo=su_s[(cell, grp)]['mean'] - max(su_s[(cell, a)]['mean'] for a in solos), solos=solos)
    return out


def analyze(score_paths, sumo_windows_path, *, H=900, series='road', label='WM', boot=B_BOOT):
    wm, book = wm_rows(score_paths)
    sumo = read_sumo_windows(sumo_windows_path)
    n_w = int(round(H / WINDOW_S))
    pairs = pairs_of(wm, sumo)
    cells = sorted({c for c, _ in pairs})
    wm_u = {p: _wm_unit_gains(wm, p[0], p[1], n_w, series) for p in pairs}
    su_u = {p: _sumo_unit_gains(sumo, p[0], p[1], n_w, series) for p in pairs}
    wm_s = {p: _summ_wm(wm_u[p]) for p in pairs}
    su_s = {p: _summ_sumo(su_u[p]) for p in pairs}
    rng = np.random.default_rng(RNG)
    # bootstrap: one index draw per cell (WM states, SUMO seeds), shared by every arm of the cell (paired arms)
    reps = defaultdict(list)
    wm_ci, su_ci, dom_reps = defaultdict(list), defaultdict(list), defaultdict(list)
    n_states = {c: len(next(iter(wm_u[p] for p in pairs if p[0] == c))) for c in cells}
    n_seeds = {c: len(next(iter(su_u[p] for p in pairs if p[0] == c))) for c in cells}
    for _ in range(int(boot)):
        iw = {c: rng.integers(0, n_states[c], n_states[c]) for c in cells}
        isu = {c: rng.integers(0, n_seeds[c], n_seeds[c]) for c in cells}
        ws = {p: _summ_wm(wm_u[p], iw[p[0]]) for p in pairs}
        ss = {p: _summ_sumo(su_u[p], isu[p[0]]) for p in pairs}
        for p in pairs:
            wm_ci[p].append(ws[p]['mean'])
            su_ci[p].append(ss[p]['mean'])
        for k, v in metrics(ws, ss, pairs).items():
            if v is not None:
                reps[k].append(v)
        for k, v in dominance(ws, ss, cells).items():
            dom_reps[k].append((v['wm'], v['sumo']))

    def q(x):
        return [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
    wm_q = {p: q(wm_ci[p]) for p in pairs}
    su_q = {p: q(su_ci[p]) for p in pairs}
    su_conf = {p: (su_q[p][0] > 0 or su_q[p][1] < 0) for p in pairs}
    point = metrics(wm_s, su_s, pairs, su_conf)
    ci = {k: q(v) for k, v in reps.items()}
    table = []
    for p in pairs:
        c, a = p
        table.append(dict(cell=c, strategy=a, wm_benefit=round(wm_s[p]['mean'], 1),
                          wm_ci95=[round(x, 1) for x in wm_q[p]], wm_rel=_r(wm_s[p]['rel']),
                          wm_p_benefit=round(wm_s[p]['p_pos'], 3), wm_p_state=round(wm_s[p]['p_state'], 3),
                          wm_states=wm_s[p]['n'], wm_class=classify(*wm_q[p]),
                          sumo_benefit=round(su_s[p]['mean'], 1), sumo_ci95=[round(x, 1) for x in su_q[p]],
                          sumo_rel=_r(su_s[p]['rel']), sumo_benefit_frac=round(su_s[p]['frac'], 3),
                          sumo_seeds=su_s[p]['n'], sumo_class=classify(*su_q[p]),
                          sign_agree=bool(np.sign(wm_s[p]['mean']) == np.sign(su_s[p]['mean'])),
                          class_agree=classify(*wm_q[p]) == classify(*su_q[p])))
    dom = dominance(wm_s, su_s, cells)
    dom_tab = []
    for k, v in dom.items():
        arr = np.array(dom_reps[k])
        dom_tab.append(dict(cell=k[0], group=k[1], solos=v['solos'], wm_group_minus_best_solo=round(v['wm'], 1),
                            wm_ci95=[round(x, 1) for x in q(arr[:, 0])],
                            sumo_group_minus_best_solo=round(v['sumo'], 1),
                            sumo_ci95=[round(x, 1) for x in q(arr[:, 1])],
                            wm_says_solo_dominates=bool(v['wm'] < 0), sumo_solo_dominates=bool(v['sumo'] < 0),
                            p_wm_solo_dominates=float(np.mean(arr[:, 0] < 0))))
    pattern = regime_pattern(table)
    reliability = reliability_bins(wm_s, su_s, pairs)
    return dict(schema=SCHEMA, label=label, H=H, series=series, windows=n_w, scores=[str(p) for p in score_paths],
                sumo_windows=str(sumo_windows_path), cells=cells, pairs=[list(p) for p in pairs], point=point,
                ci95=ci, table=table, dominance=dom_tab, pattern=pattern, reliability=reliability,
                wm_book=dict((k, v) for k, v in book.items() if k != 'meta'),
                wm_meta=book['meta'][:1],
                sign_convention='benefit = J(u0) - J(strategy) over [300, 300+H) s, veh-s; positive = strategy '
                                'better (S-REGIMEU0 diff = strategy - u0 has the opposite sign)')


def _r(x):
    return None if x is None else round(float(x), 4)


def regime_pattern(table):
    """Claim-4a target pattern: help in heavy traffic (D3200), ~0 at D2800_p40, hurt at D2400 (solo2 and wall_h)."""
    want = {'D3200_p60': 'help', 'D3200_p40': 'help', 'D3200_p30': 'help', 'D3200_p20': 'help',
            'D2800_p40': '~0', 'D2400_p60': 'hurt', 'D2400_p40': 'hurt'}
    out = []
    for r in table:
        if r['strategy'] not in ('wall_h_solo2', 'wall_h') or r['cell'] not in want:
            continue
        out.append(dict(cell=r['cell'], strategy=r['strategy'], target=want[r['cell']], sumo=r['sumo_class'],
                        wm=r['wm_class'], wm_point_sign=('help' if r['wm_benefit'] > 0 else 'hurt'),
                        match_target=r['wm_class'] == want[r['cell']], match_sumo=r['wm_class'] == r['sumo_class']))
    return out


def reliability_bins(wm_s, su_s, pairs, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ps = [p for p in pairs if lo <= wm_s[p]['p_pos'] < hi]
        if not ps:
            continue
        out.append(dict(bin=[lo, min(hi, 1.0)], n_pairs=len(ps), wm_p_mean=float(np.mean([wm_s[p]['p_pos']
                                                                                         for p in ps])),
                        sumo_frac_mean=float(np.mean([su_s[p]['frac'] for p in ps]))))
    return out


# ================================================================================ horizon stability
def horizon_check(score_paths, sumo_windows_path, native_ref_path=None, rule=HORIZON_RULE):
    """Per-window WM level vs SUMO: u0 arm (cell means, different worlds) and native arm (paired per world, if scored).

    Stable horizon H* = the largest 100-s multiple such that every window before it passes: no WM error rows and no
    entry-slot shortage; pooled median over cells of WM/SUMO u0 window road delay in rule['pooled_ratio'] and every
    cell's ratio in rule['cell_ratio']."""
    wm, book = wm_rows(score_paths)
    sumo = read_sumo_windows(sumo_windows_path)
    cells = sorted(c for c in wm if c in sumo and BASE in wm[c] and BASE in sumo[c])
    per_cell = {}
    for c in cells:
        w = np.array([np.mean(o['road_w'], 0) for o in wm[c][BASE].values()])        # [states, W]
        s = np.array([v['road_w'] for v in sumo[c][BASE].values()])
        nw = np.array([np.mean(o['n_w'], 0) for o in wm[c][BASE].values()])
        per_cell[c] = dict(wm_u0=w.mean(0).tolist(), sumo_u0=s.mean(0).tolist(),
                           ratio=(w.mean(0) / np.maximum(s.mean(0), 1e-9)).tolist(), wm_n=nw.mean(0).tolist())
    W = min((len(v['ratio']) for v in per_cell.values()), default=0)
    pooled = [float(np.median([per_cell[c]['ratio'][j] for c in cells])) for j in range(W)]
    native = None
    if native_ref_path is not None:
        ref = json.loads(G.check_path(native_ref_path).read_text())['ref']
        rows = [(o, ref[sid]) for c in wm if 'native' in wm[c] for sid, o in wm[c]['native'].items() if sid in ref]
        if rows:
            wmn = np.array([np.mean(o['road_w'], 0) for o, _ in rows])
            ref_a = np.array([r for _, r in rows], float)
            native = dict(n_states=len(rows), wm=wmn.mean(0).tolist(), sumo=ref_a.mean(0).tolist(),
                          ratio_median=np.median(wmn / np.maximum(ref_a, 1e-9), 0).tolist(),
                          spearman_by_window=[spearman(wmn[:, j], ref_a[:, j]) for j in range(wmn.shape[1])],
                          cumulative_spearman=[spearman(wmn[:, :j + 1].sum(1), ref_a[:, :j + 1].sum(1))
                                               for j in range(wmn.shape[1])])
    ok_global = not book['errors'] and book['no_slot_rows'] == 0
    h_star = 0
    for j in range(W):
        good = (ok_global and rule['pooled_ratio'][0] <= pooled[j] <= rule['pooled_ratio'][1]
                and all(rule['cell_ratio'][0] <= per_cell[c]['ratio'][j] <= rule['cell_ratio'][1] for c in cells))
        if not good:
            break
        h_star = (j + 1) * WINDOW_S
    return dict(rule=rule, cells=cells, per_cell=per_cell, pooled_ratio=pooled, native=native,
                errors=len(book['errors']), no_slot_rows=book['no_slot_rows'], H_star=h_star,
                crn=dict(checked=book['crn_checked'], violations=book['crn_violations']))


# ================================================================================ report
def markdown(res, hz=None):
    p, ci = res['point'], res['ci95']

    def f(k, d=3):
        x = p.get(k)
        if x is None:
            return 'n/a'
        s = f'{x:.{d}f}'
        if k in ci:
            s += f' [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]'
        return s
    L = [f"# WMEV claim 4a - {res['label']}: WM vs SUMO benefit of rule strategies on the u0 base (H = {res['H']} s, "
         f"{res['series']} delay)", '',
         f"Sign: {res['sign_convention']}.", '',
         f"- sign agreement {f('sign_agreement')} (n = {p['n_pairs']}); on SUMO-confident pairs "
         f"{f('sign_agreement_confident')} (n = {p.get('n_confident')})",
         f"- Spearman pooled pairs {f('spearman_pooled')}; relative {f('spearman_pooled_rel')}"]
    for k in sorted(p):
        if k.startswith('spearman_cells_') and not k.startswith('spearman_cells_rel'):
            L.append(f"- Spearman across cells, {k[len('spearman_cells_'):]}: {f(k)}")
    L += [f"- P(benefit > 0) calibration: MAE {f('calib_mae')}, Brier {f('brier')} vs base-rate "
          f"{f('brier_base_rate')} (skill {f('brier_skill')}), Spearman {f('calib_spearman')}", '',
          '| cell | strategy | WM benefit [95% CI] | WM rel | P_WM(>0) | WM class | SUMO benefit [95% CI] | SUMO rel |'
          ' SUMO frac>0 | SUMO class | sign | class |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in res['table']:
        L.append(f"| {r['cell']} | {r['strategy']} | {r['wm_benefit']:+.0f} [{r['wm_ci95'][0]:+.0f}, "
                 f"{r['wm_ci95'][1]:+.0f}] | {_pct(r['wm_rel'])} | {r['wm_p_benefit']:.2f} | {r['wm_class']} | "
                 f"{r['sumo_benefit']:+.0f} [{r['sumo_ci95'][0]:+.0f}, {r['sumo_ci95'][1]:+.0f}] | "
                 f"{_pct(r['sumo_rel'])} | {r['sumo_benefit_frac']:.2f} | {r['sumo_class']} | "
                 f"{'Y' if r['sign_agree'] else 'n'} | {'Y' if r['class_agree'] else 'n'} |")
    L += ['', '**Regime pattern** (target: help at D3200, ~0 at D2800_p40, hurt at D2400):', '',
          '| cell | strategy | target | SUMO | WM | WM = target |', '|---|---|---|---|---|---|']
    for r in res['pattern']:
        L.append(f"| {r['cell']} | {r['strategy']} | {r['target']} | {r['sumo']} | {r['wm']} | "
                 f"{'Y' if r['match_target'] else 'n'} |")
    L += ['', '**Group minus best solo** (benefit units; negative = solo dominates the group):', '',
          '| cell | group | WM [95% CI] | P_WM(solo dominates) | SUMO [95% CI] |', '|---|---|---|---|---|']
    for r in res['dominance']:
        L.append(f"| {r['cell']} | {r['group']} | {r['wm_group_minus_best_solo']:+.0f} [{r['wm_ci95'][0]:+.0f}, "
                 f"{r['wm_ci95'][1]:+.0f}] | {r['p_wm_solo_dominates']:.2f} | {r['sumo_group_minus_best_solo']:+.0f} "
                 f"[{r['sumo_ci95'][0]:+.0f}, {r['sumo_ci95'][1]:+.0f}] |")
    if hz is not None:
        L += ['', f"**Horizon check**: stable H* = {hz['H_star']} s (rule {hz['rule']}); WM errors {hz['errors']}, "
                  f"slot-shortage rows {hz['no_slot_rows']}, CRN violations {hz['crn']['violations']}/"
                  f"{hz['crn']['checked']}", '',
              'Pooled median WM/SUMO u0 road delay per 100-s window: ' +
              ', '.join(f'{x:.2f}' for x in hz['pooled_ratio'])]
        if hz.get('native'):
            L.append('Native arm vs the same world\'s SUMO native tape, cumulative Spearman by horizon: ' +
                     ', '.join('n/a' if x is None else f'{x:.2f}' for x in hz['native']['cumulative_spearman']))
    return '\n'.join(L) + '\n'


def _pct(x):
    return 'n/a' if x is None else f'{100 * x:+.1f}%'


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.regime_u0_compare')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sumo', help='per-window SUMO gains from the S-REGIMEU0 / S-LOWSHARE-U0 job series')
    s.add_argument('--bases', default=','.join(DEFAULT_SUMO))
    s.add_argument('--out', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--scores', action='append', required=True)
    a.add_argument('--sumo-windows', required=True)
    a.add_argument('--native-ref')
    a.add_argument('--H', type=float, help='default: the stable horizon H* (900 if every window passes)')
    a.add_argument('--series', default='road', choices=('road', 'team'))
    a.add_argument('--label', default='WM')
    a.add_argument('--out', required=True)
    x = ap.parse_args(argv)
    if x.cmd == 'sumo':
        print(json.dumps(write_sumo_windows(x.out, tuple(x.bases.split(',')))))
        return
    hz = horizon_check(x.scores, x.sumo_windows, x.native_ref)
    H = x.H if x.H else (hz['H_star'] or WINDOW_S)
    res = analyze(x.scores, x.sumo_windows, H=H, series=x.series, label=x.label)
    res['horizon'] = hz
    if not x.H:
        res['H_rule'] = 'H = stable horizon H* of horizon_check (registered rule HORIZON_RULE)'
        res['all_H'] = {str(h): analyze(x.scores, x.sumo_windows, H=h, series=x.series, label=x.label,
                                        boot=200)['point'] for h in range(100, 1000, 100)}
    out = G.check_path(x.out)
    with Path(out).open('x') as fh:
        fh.write(json.dumps(res, indent=1, default=float))
    md = markdown(res, hz)
    with Path(out).with_suffix('.md').open('x') as fh:
        fh.write(md)
    print(md)


if __name__ == '__main__':
    main()
