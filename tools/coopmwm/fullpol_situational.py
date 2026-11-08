"""CLEAN5C4 situational WM guidance (owner REGISTRATION 2026-11-08 19:31; CPU, no physics).

A (WM-sourced, N2-representable plan, onset-situation bin) pair is admitted iff its pooled per-activation effect
vs the paired u0 window has a 95% CI upper bound < 0 AND the plan is fairness-ALLOWED in every WMVERIFY
verification that analysed it. Situation bins (observable at onset, S-MULTICRIT-ACT definitions verbatim):
cong = drop-area mean speed < 15 m/s ('congested') vs not ('free'); lane_diff = WM-lane-0 (SUMO lane 2) mean
speed ahead (500 m) - WM-lane-1 mean speed ahead > 5 m/s ('L2faster') vs not ('L2notfaster').

Effect of activation k (multicrit/act/act_analyze.py, verbatim): change in the paired (arm - u0) 10-s binned
delay gap over [t_k, min(next activation, t_k + 300, 1200)); negative = better than u0. CI: cluster bootstrap
over (source, root, cell, seed), 4,000 resamples, a fresh RNG 20261107 per estimate (order independent).
Penetration is a label: activations are pooled across cells unless the p20-only pool is large enough
(P20_MIN_ACTIVATIONS and P20_MIN_CLUSTERS, fixed before looking at effects).

Admitted plans are offered ONLY in their admitted bins: offer_condition['situation'] (fullpol_bank.offer_ok
evaluates the same features from the acting CAV's own row + own sensing; missing data -> the same 'free' /
'L2notfaster' defaults as the offline bins). B gets the targets, R none; both get the same templates.
"""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


RULE = 'fullpol-clean5c4-situational-v4'      # v4: privileged bin = B's EV-BC target only (REV-SPD B1)
MODES = ('privileged', 'own_sensing')
#: MAIN 2026-11-08 (REV-SPD B1 fix): the privileged bin is a TRAIN-TIME teacher signal used ONLY where B's EV-BC
#: target is chosen (fullpol_learn.ev_bc_records, offline, from the episode's recorded observable tape; no RNG).
#: Slate candidacy ignores it (offer_ok: identical in B and R, train and deploy), so the slate, samples,
#: observations and action RNG draws do not depend on privileged state (asymmetric guidance, disclosed).
TEACHER_NOTE = ('privileged teacher bin: used only to choose B\'s EV-BC target label (offline, recorded tape at the '
                'decision second); the slate / observation / action RNG never see it (asymmetric guidance, disclosed)')


class NoAdmission(ValueError):
    """No (plan, bin) pair admitted: no situational WM guidance (report negative; no launch)."""
    def __init__(self, pairs):
        super().__init__('no admitted (plan, bin) pair: no situational WM guidance (report negative)')
        self.pairs = pairs
B_RESAMPLES = 4000
SEED = 20261107
S_DROP = 75.44                    # multicrit/act/reduce_act.py (wz3to2 drop position, WM s)
CONG_MPS, LANE_DIFF_MPS = 15.0, 5.0
P20_MIN_ACTIVATIONS, P20_MIN_CLUSTERS = 30, 8
MIN_CLUSTERS = 3                  # a cluster-bootstrap CI needs >= 3 clusters (fewer: no valid CI -> not admitted)
ALLOWED_SCOPES = ('all', 'p20')
WM_GROUPS = ('wm', 'wm_fairer', 'wm_reserve')
PARITY_PASS = 'identical'


# ---------------------------------------------------------------- per-activation effects (act_analyze verbatim)

def _lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load(paths):
    eps = {}
    for path in paths:
        tag = Path(path).name.split('act_')[-1].split('.')[0]
        for r in _lines(path):
            if r.get('ok'):
                eps[(tag, r['root'], r['cell'], r['seed'], r['arm'])] = r
    return eps


def _bins_sum(r, b0, b1):
    return float(sum(sum(row) for row in r['bins'][b0:b1]))


def activations(eps, onset_bins):
    """Per-activation effects; ``onset_bins`` {(root, cell, seed, arm, t): bin} from the host reducer
    (ops/sit_reduce.py: onset_bin on the acting CAV's own sensing rows at onset). An activation without a
    reducer record is an error (fail closed)."""
    rows = []
    for key, r in eps.items():
        tag, root, cell, seed, arm = key
        if arm == 'u0':
            continue
        u = eps.get((tag, root, cell, seed, 'u0'))
        if u is None:
            continue
        acts = sorted(r.get('acts') or [], key=lambda a: a['t'])
        for k, a in enumerate(acts):
            t = a['t']
            nxt = acts[k+1]['t'] if k+1 < len(acts) else 1200
            tend = min(nxt, t+300, 1200)
            b0, b1 = (t-300)//10, math.ceil((tend-300)/10)
            b = onset_bins.get((root, cell, seed, arm, int(t)))
            if b is None:
                raise ValueError(f'no own-sensing onset bin for {root}/{cell}/{seed}/{arm}/t{t}')
            rows.append(dict(cluster=(tag, root, cell, seed), cell=cell, pen=int(cell.split('_p')[1]), arm=arm, t=t,
                             members=len(a.get('members') or ()), eff=_bins_sum(r, b0, b1)-_bins_sum(u, b0, b1),
                             bin=b))
    return rows


def cboot(rows):
    import numpy as np
    cl = defaultdict(list)
    for r in rows:
        cl[r['cluster']].append(r['eff'])
    keys = sorted(cl)
    if not keys:
        return None
    sums = np.array([sum(cl[k]) for k in keys])
    ns = np.array([len(cl[k]) for k in keys])
    idx = np.random.default_rng(SEED).integers(0, len(keys), size=(B_RESAMPLES, len(keys)))
    bm = sums[idx].sum(1)/ns[idx].sum(1)
    return dict(n=int(ns.sum()), clusters=len(keys), mean=float(sums.sum()/ns.sum()),
                lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)))


# ---------------------------------------------------------------- situation bins (offline and runtime, same rule)

SENSOR_RANGE_M = 100.0            # merge_features.SENSOR_RANGE_M (track2_backend own_sensing / LCEXEC sensed rows)
UNOBSERVED = dict(cong='unobserved', lane_diff='unobserved')

def situation_bin(sit):
    ds = sit.get('drop_speed')
    lanes = sit.get('v_lane_ahead500') or {}
    l2, l1 = lanes.get('0', lanes.get(0)), lanes.get('1', lanes.get(1))
    return dict(cong='congested' if (ds is not None and ds < CONG_MPS) else 'free',
                lane_diff='L2faster' if (l2 is not None and l1 is not None and l2-l1 > LANE_DIFF_MPS) else 'L2notfaster')


def _mean(x):
    return sum(x)/len(x) if x else None


def runtime_situation(own, rows):
    """The same onset features (reduce_act.situation) from the acting CAV's own row + own sensing rows."""
    s = float(own['s_m'])
    view = [own]+[r for r in rows if r.get('vid') != own.get('vid')]
    up = [(int(r['lane']), float(r['s_m']), float(r['v_mps'])) for r in view if float(r['s_m']) < S_DROP]
    return dict(drop_speed=_mean([v for _, x, v in up if -600 <= x]),
                v_lane_ahead500={str(k): _mean([v for ln, x, v in up if ln == k and s <= x <= s+500]) for k in (0, 1, 2)})


def onset_bin(own, rows, mode='own_sensing'):
    """THE bin rule, offline (ops/sit_reduce.py on the observable tape at onset) and at runtime
    (fullpol_bank.offer_ok): the acting CAV's own row + every vehicle within SENSOR_RANGE_M of it (the
    track2_backend own_sensing filter); drop-area speed or lane-0 / lane-1 ahead speed not observed in that
    view -> UNOBSERVED (its own bin; never admitted, never offered)."""
    if mode not in MODES:
        raise ValueError('onset_bin mode must be one of '+repr(MODES))
    if own is None or own.get('v_mps') is None:
        return dict(UNOBSERVED)
    if mode == 'privileged':
        # the full-tape S-MULTICRIT-ACT features (reduce_act.situation, verbatim): every vehicle, no range
        view = [r for r in rows if r.get('vid') != own.get('vid')]
        if any(r.get('v_mps') is None for r in view):
            return dict(UNOBSERVED)
        return situation_bin(runtime_situation(own, view))
    s = float(own['s_m'])
    view = [r for r in rows if r.get('vid') != own.get('vid') and abs(float(r['s_m'])-s) <= SENSOR_RANGE_M]
    if any(r.get('v_mps') is None for r in view):
        return dict(UNOBSERVED)
    sit = runtime_situation(own, view)
    lanes = sit['v_lane_ahead500']
    if sit['drop_speed'] is None or lanes['0'] is None or lanes['1'] is None:
        return dict(UNOBSERVED)
    return situation_bin(sit)


def situation_ok(cond, own, rows):
    """Slate offer gate. mode 'privileged': NOT a slate condition (always True: candidacy uses only the
    non-privileged conditions, identical in B and R); the bin acts on B's EV-BC target only (teacher_admit).
    mode 'own_sensing': the acting CAV's own sensing bin (a slate condition)."""
    want = (cond or {}).get('situation')
    if want is None or want.get('mode', 'own_sensing') == 'privileged':
        return True
    got = onset_bin(own, rows)
    if got == UNOBSERVED:
        return False
    return any(all(got[k] == b[k] for k in ('cong', 'lane_diff')) for b in want['bins'])


def load_onset_bins(paths, mode=None):
    """ops/sit_reduce.py output lines -> {(root, cell, seed, arm, t): bin}; every line's mode must equal ``mode``."""
    out = {}
    for path in paths:
        for r in _lines(path):
            if mode is not None and r.get('mode', 'own_sensing') != mode:
                raise ValueError(f'onset file {path} has mode {r.get("mode", "own_sensing")!r}, need {mode!r}')
            out[(r['root'], r['cell'], int(r['seed']), r['arm'], int(r['t']))] = r['bin']
    return out


# ---------------------------------------------------------------- admission

def plan_status(analyses, structured=(), allowed_scope='all'):
    """{family: dict(wm, representable, allowed, cells)} over WMVERIFY analysis dirs (PLAN/VERIFY/ANALYSIS[_ALLOWED]).
    allowed_scope 'all': fairness-ALLOWED in every verification cell that analysed the plan (any penetration);
    'p20': in every p20 verification cell (at least one). ``structured``: WM-C structured/rule arm names (not WM)."""
    if allowed_scope not in ALLOWED_SCOPES:
        raise ValueError('allowed_scope must be one of '+repr(ALLOWED_SCOPES))
    out = {}
    for d in analyses:
        d = Path(d)
        plan = json.loads((d/'PLAN.json').read_text(encoding='utf-8'))
        verify = json.loads((d/'VERIFY.json').read_text(encoding='utf-8'))
        analysis = json.loads((d/'ANALYSIS.json').read_text(encoding='utf-8'))
        allowed = json.loads((d/'ANALYSIS_ALLOWED.json').read_text(encoding='utf-8'))
        for fam, v in plan['families'].items():
            st = out.setdefault(fam, dict(wm=True, representable=True, allowed=True, cells=[], verdicts=[]))
            st['wm'] &= all(g in WM_GROUPS for g in (v.get('cells') or {}).values()) and fam not in structured
            verdict = ((verify.get('families') or {}).get(fam, {}).get('n2_parity') or {}).get('verdict')
            st['verdicts'].append(verdict)
            st['representable'] &= verdict == PARITY_PASS
            for cell, fams in analysis['cells'].items():
                if fam in fams and fams[fam].get('n'):
                    ok = fam in (allowed['cells'].get(cell) or {})
                    st['cells'].append(dict(source=d.parent.name, cell=cell, allowed=ok))
    for st in out.values():
        scope = [c for c in st['cells'] if allowed_scope == 'all' or c['cell'].endswith('_p20')]
        st['allowed'] = bool(scope) and all(c['allowed'] for c in scope)
        st['allowed_scope'] = allowed_scope
    return out


BINS = [dict(cong=c, lane_diff=d) for c in ('congested', 'free') for d in ('L2faster', 'L2notfaster')]


def admit(rows, status):
    """Per (family, bin): pool (p20 only when large enough, else all penetrations), effect, verdict."""
    out = []
    for fam in sorted({r['arm'] for r in rows}):
        st = status.get(fam)
        for b in BINS:
            mine = [r for r in rows if r['arm'] == fam and r['bin'] == b]
            p20 = [r for r in mine if r['pen'] == 20]
            p20_eff = cboot(p20)
            use_p20 = p20_eff is not None and p20_eff['n'] >= P20_MIN_ACTIVATIONS and p20_eff['clusters'] >= P20_MIN_CLUSTERS
            eff = p20_eff if use_p20 else cboot(mine)
            if eff is None:
                continue
            reasons = []
            if st is None or not st['wm']:
                reasons.append('not_wm_sourced')
            if st is None or not st['representable']:
                reasons.append('not_n2_representable')
            if st is None or not st['allowed']:
                reasons.append('not_fairness_allowed')
            if any(r['members'] < 2 for r in mine):
                reasons.append('not_multi_cav')
            if eff['clusters'] < MIN_CLUSTERS:
                reasons.append(f'clusters<{MIN_CLUSTERS}')
            elif not eff['hi'] < 0:
                reasons.append('ci_upper>=0')
            out.append(dict(family=fam, bin=b, pool='p20' if use_p20 else 'all_penetrations', effect=eff,
                            p20=p20_eff, admitted=not reasons, reasons=reasons))
    return out


def templates_from_banks(bank_paths):
    """{family: [(cell, template, offer_condition, template_sha256)]} from WMVERIFY bank rows."""
    out = defaultdict(dict)
    for path in bank_paths:
        for r in _lines(path):
            if r.get('template') is not None:
                out[r['family']].setdefault(r['template_sha256'], (r['cell'], r['template'], r.get('offer_condition')))
    return out


class TapeTeacher:
    """Privileged bins of one recorded episode (observable tape rows at the decision second; read lazily once)."""

    def __init__(self, path):
        self.path, self.rows = Path(path), None

    def bin(self, vid, time_s):
        if self.rows is None:
            self.rows = {}
            if self.path.exists():
                for d in _lines(self.path):
                    self.rows[int(round(d['time_s']))] = d['vehicles']
        rows = self.rows.get(int(round(time_s)))
        if rows is None:
            return None
        own = next((r for r in rows if r.get('vid') == vid), None)
        return onset_bin(own, rows, 'privileged')


def teacher_verdict(row, teacher, vid, time_s):
    """'n/a' (no privileged block), 'admit', 'reject' (bin observed, not admitted) or 'unknown' (tape / vehicle /
    feature missing)."""
    sit = row.get('situational')
    if not sit or sit.get('mode') != 'privileged':
        return 'n/a'
    got = None if teacher is None else teacher.bin(vid, time_s)
    if got is None or got == UNOBSERVED:
        return 'unknown'
    return 'admit' if any(all(got[k] == b[k] for k in ('cong', 'lane_diff')) for b in sit['bins']) else 'reject'


def teacher_admit(row, teacher, vid, time_s):
    """May the guidance ``row``'s family be B's EV-BC target for ``vid`` at ``time_s``? Rows without a privileged
    situational block: always. Else the privileged bin at that second must be one of the row's admitted bins
    (tape or vehicle missing -> no)."""
    sit = row.get('situational')
    if not sit or sit.get('mode') != 'privileged':
        return True
    got = None if teacher is None else teacher.bin(vid, time_s)
    return got is not None and got != UNOBSERVED and any(
        all(got[k] == b[k] for k in ('cong', 'lane_diff')) for b in sit['bins'])


def build(act_paths, analyses, bank_paths, *, cells, provenance, onset_paths, structured=(), allowed_scope='all',
          mode='own_sensing'):
    """clean5c guidance doc (fullpol_bank schema) with situational offer conditions."""
    from .fullpol_learn import guidance_doc
    from .fullpol_bank import RULE as BANK_RULE
    rows = activations(load(act_paths), load_onset_bins(onset_paths, mode))
    status = plan_status(analyses, structured, allowed_scope)
    pairs = admit(rows, status)
    banks = templates_from_banks(bank_paths)
    admitted = defaultdict(list)
    for p in pairs:
        if p['admitted']:
            admitted[p['family']].append(p)
    g_rows, templates, rejected = [], [], {}
    for fam, ps in sorted(admitted.items()):
        tpls = banks.get(fam) or {}
        if len(tpls) != 1:
            rejected[fam] = f'{len(tpls)} distinct bank templates (need exactly 1)'
            continue
        (sha, (cell0, tpl, cond)), = tpls.items()
        if not cond:
            rejected[fam] = 'bank offer_condition missing'
            continue
        demands = {c['cell'].split('_')[0] for c in status[fam]['cells']}
        best = min(ps, key=lambda p: p['effect']['hi'])
        situation = dict(rule=RULE, mode=mode, bins=[p['bin'] for p in ps])
        for cell in cells:
            if cell.split('_')[0] not in demands:
                continue
            g_rows.append(dict(arm='B', regime=cell, family=fam, baseline='mappo', n=best['effect']['n'],
                               mean_veh_s=-best['effect']['mean'], lb95_veh_s=-best['effect']['hi'],
                               ub95_veh_s=-best['effect']['lo'], fair=True, source='wm_bank', n2_parity=PARITY_PASS,
                               template_sha256=sha, situational=dict(bins=situation['bins'], mode=mode, rule=RULE,
                               pool=best['pool'],
                               unit='veh-s per activation window (pooled per-activation effect, best admitted bin)')))
            templates.append(dict(cell=cell, family=fam, template=json.loads(json.dumps(tpl)),
                                  offer_condition=dict(cond, situation=situation)))
    for t in templates:
        t['template']['source'] = dict(t['template'].get('source') or {}, variant=t['family'], family=t['family'],
                                       cell=t['cell'])
    sources = []
    for path in list(act_paths)+list(onset_paths)+list(bank_paths)+[Path(d)/n for d in analyses for n in
                                                 ('PLAN.json', 'VERIFY.json', 'ANALYSIS.json', 'ANALYSIS_ALLOWED.json')]:
        p = Path(path)
        sources.append(dict(name=f'{p.parent.name}/{p.name}', sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            baseline='mappo'))
    if not g_rows:
        raise NoAdmission(pairs)
    doc = guidance_doc(g_rows, sources=sources, provenance=provenance)
    doc['source']['rule'] = (RULE+': per-activation effect vs the paired u0 window (S-MULTICRIT-ACT, verbatim), '
                             'pooled per (plan, onset bin), cluster bootstrap over (source, root, cell, seed), 4,000 '
                             'resamples RNG 20261107; lb95/mean/ub95 = gain of the best admitted bin (veh-s/window)')
    doc['clean5c'] = dict(rule=BANK_RULE, provenance=provenance, threshold_s=10.0,
                          verified_cells=sorted({r['regime'] for r in g_rows}), skipped={}, rejected=rejected,
                          templates=templates)
    doc['situational'] = dict(rule=RULE, n_activations=len(rows), pairs=pairs,
                              bin_counts={json.dumps(b, sort_keys=True): sum(r['bin'] == b for r in rows)
                                          for b in BINS+[UNOBSERVED]},
                              mode=mode, teacher_note=TEACHER_NOTE if mode == 'privileged' else None,
                              bin_rule=(f'onset_bin({mode!r}): acting CAV (holder) own row + '
                                        + ('every vehicle of the plant table' if mode == 'privileged' else
                                           f'vehicles within {SENSOR_RANGE_M:.0f} m')
                                        + ' at onset (same function as offer_ok)'),
                              plans={f: {k: v for k, v in s.items()} for f, s in sorted(status.items())},
                              pool_rule=f'p20 only if >= {P20_MIN_ACTIVATIONS} activations and >= {P20_MIN_CLUSTERS} '
                                        'clusters in the (plan, bin), else all penetrations (label only)')
    return doc


def check(doc, cells):
    """Prepare-time (zero physics): a situational doc; rows/templates in the training cells; every template
    carries its admitted bins; every row family admitted. Returns the cells without a target ('none')."""
    sit = doc.get('situational') or {}
    cells = set(cells)
    admitted = {p['family'] for p in sit.get('pairs', ()) if p['admitted']}
    regimes = {r['regime'] for r in doc['rows']}
    c = doc.get('clean5c') or {}
    if (sit.get('rule') != RULE or not regimes <= cells or {r['family'] for r in doc['rows']} - admitted
            or any(t['cell'] not in cells or (t['offer_condition'].get('situation') or {}).get('rule') != RULE
                   or not t['offer_condition']['situation'].get('bins') for t in c.get('templates', ()))
            or any(r['arm'] != 'B' for r in doc['rows'])):
        raise ValueError('situational guidance must be the CLEAN5C4 rule doc for the training cells')
    return tuple(sorted(cells-regimes))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='CLEAN5C4 situational WM guidance (CPU, no physics)')
    ap.add_argument('--act', nargs='+', required=True, help='per-activation records (act_*.jsonl[.gz])')
    ap.add_argument('--analysis', nargs='+', required=True, help='WMVERIFY analysis dirs (PLAN/VERIFY/ANALYSIS[_ALLOWED])')
    ap.add_argument('--bank', nargs='+', required=True, help='WMVERIFY bank jsonl (templates + offer conditions)')
    ap.add_argument('--onset', nargs='+', required=True, help='ops/sit_reduce.py outputs (own-sensing onset bins)')
    ap.add_argument('--structured', nargs='*', default=(), help="WM-C ARMS files: their 'structured' arms are not WM")
    ap.add_argument('--allowed-scope', choices=ALLOWED_SCOPES, default='all')
    ap.add_argument('--mode', choices=MODES, default='own_sensing', help='onset_bin mode (ONSET files must match)')
    ap.add_argument('--cells', nargs='+', default=None, help='training cells (default the three p20 cells)')
    ap.add_argument('--provenance', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    from .clean5b_restart import P20_CELLS as ALL_CELLS, _check_p20_cells
    P20_CELLS = ALL_CELLS if a.cells is None else _check_p20_cells(a.cells)
    out = Path(a.out)
    if out.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    structured = set()
    for path in a.structured:
        for groups in json.loads(Path(path).read_text(encoding='utf-8')).values():
            if isinstance(groups, dict):
                structured.update(n['name'] if isinstance(n, dict) else n for n in groups.get('structured') or ())
    try:
        doc = build(a.act, a.analysis, a.bank, cells=P20_CELLS, provenance=a.provenance, onset_paths=a.onset,
                    structured=sorted(structured), allowed_scope=a.allowed_scope, mode=a.mode)
    except NoAdmission as e:
        report = Path(str(out)+'.NO_ADMISSION.json')
        report.write_bytes((json.dumps(dict(rule=RULE, pairs=e.pairs), indent=1, sort_keys=True)+chr(10)).encode('utf-8'))
        raise SystemExit('NO ADMISSION (report: '+str(report)+')')
    doc['situational']['structured_excluded'] = sorted(structured)
    none = check(doc, P20_CELLS)
    data = (json.dumps(doc, indent=1, sort_keys=True)+chr(10)).encode('utf-8')     # LF bytes on every OS
    out.write_bytes(data)
    file_sha = hashlib.sha256(data).hexdigest()
    Path(str(out)+'.sha256').write_bytes((f'{file_sha}  {out.name}'+chr(10)).encode('utf-8'))
    print(json.dumps(dict(out=str(out), file_sha256=file_sha, rows_sha256=doc['source']['sha256'], none=list(none),
                          admitted=[(p['family'], p['bin'], p['pool'], p['effect']) for p in doc['situational']['pairs']
                                    if p['admitted']])))


if __name__ == '__main__':
    main()
