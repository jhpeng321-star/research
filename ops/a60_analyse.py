"""AMENDMENT 60 (S-WMVALUE60) laptop analysis, exactly as registered (zero physics; numpy + stdlib).

  score  60B S-WMSCORE: Spearman rho between the A42-search WM score (higher = model predicts more saving) and the
         stored SUMO plan - U0 mean delay on 992000-992023 over all listed plans; one-sided permutation p (100,000
         permutations, seed 20261104) = share of permutations with rho <= observed (helpful direction: rho < 0).
         SCORE INFORMATIVE iff rho < 0 and p < 0.05; else NOT SHOWN. Identity: re-scored A42 plans must equal their
         stored search scores exactly, else STOP. Secondary: the same over the 24 plans not selected by WM score
         (A45 + A56); top-half vs bottom-half admission counts (ties at the median broken by family name).
  a      60A S-WMPROP96: per world mean J(six A42 WM plans) - mean J(twelve A56 random plans), paired percentile
         bootstrap 10,000 seed 20261104 (planner_eval._paired_ci verbatim); SUPPORTED iff CI upper < 0, RANDOM AHEAD
         iff CI lower > 0, else NOT SUPPORTED. Secondary: per-plan plan - U0 with CI, admission recomputed, safety.
  c      60C S-RANDBANK60: BIG_BANK - RAND_BANK total delay (the reduce contrast, fe.paired_stats on the panel
         bootstrap 10,000 seed 20261104; cross-checked with _paired_ci): MODEL BANK AHEAD iff CI upper < 0, RANDOM
         BANK AHEAD iff CI lower > 0, else NOT RESOLVED. Secondary: RAND_BANK vs U0 / DUO_HOLD, CVaR90, groups, safety.
Sign everywhere: treatment - reference, negative = less delay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPS, SEED, LEVEL = 10_000, 20261104, .95
N_PERM = 100_000


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_seal(doc, field):
    body = {k: v for k, v in doc.items() if k != field}
    canon = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canon.encode()).hexdigest() == doc[field]


def paired_ci(deltas, n_boot=REPS, boot_seed=SEED, alpha=1-LEVEL):
    """tools/coopmwm/planner_eval._paired_ci verbatim."""
    d = np.asarray(deltas, float)
    idx = np.random.default_rng(boot_seed).integers(0, len(d), (n_boot, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    p = min(1., 2*min(np.sum(boot <= 0)+1, np.sum(boot >= 0)+1)/(n_boot+1))
    return dict(n_worlds=len(d), mean_delta_vs=float(d.mean()), ci95=[float(lo), float(hi)],
                p_two_sided=float(p), worlds_improved=int(np.sum(d < 0)))


def avg_ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and x[order[j+1]] == x[order[i]]:
            j += 1
        r[order[i:j+1]] = (i+j)/2.0+1
        i = j+1
    return r


def spearman(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_test(x, y, n=N_PERM, seed=SEED):
    """One-sided (helpful = negative rho): share of permutations of y with rho <= observed."""
    rx, ry = avg_ranks(x), avg_ranks(y)
    obs = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    rxc = rx-rx.mean()
    ryc = ry-ry.mean()
    den = np.sqrt((rxc**2).sum()*(ryc**2).sum())
    cnt, done, B = 0, 0, 10_000
    while done < n:
        b = min(B, n-done)
        P = np.stack([rng.permutation(ryc) for _ in range(b)])
        r = (P @ rxc)/den
        cnt += int(np.sum(r <= obs+1e-12))
        done += b
    return obs, cnt/n


# ------------------------------------------------------------------------------------------------ 60B

def analyse_score(search, raw, adm42, adm45, randctrl):
    final = {r['strategy']['name']: r for r in search['final']}
    rows = []
    for v in adm42['verdicts']:
        rows.append(dict(family=v['family'], source='A42', score=final[v['family']]['score'], sumo=v['mean'],
                         PASS=bool(v['PASS'])))
    for v in adm45['verdicts']:
        rows.append(dict(family=v['family'], source='A45', score=final[v['family']]['score'], sumo=v['mean'],
                         PASS=bool(v['PASS'])))
    a56 = {p['family']: p for p in randctrl['plans']}
    new = {r['family']: r for r in raw['rows']}
    ident = []
    for r in raw['rows']:
        if r['role'] != 'identity':
            continue
        st = final[r['family']]
        fields = ('score', 'gain_vs_base', 'gain_vs_base_lcb', 'disagreement', 'feasible', 'worst_group_inc')
        ident.append(dict(family=r['family'], stored=st['score'], rescored=r['score'],
                          exact=bool(st['score'] == r['score']),
                          fields={f: [st.get(f), r.get(f)] for f in fields}))
    identity_ok = len(ident) >= 2 and all(x['exact'] for x in ident)
    out = dict(schema='coopmwm.a60b.wmscore.results.v1', amendment='60B', label='S-WMSCORE',
               identity=dict(rule='re-scored A42 plans reproduce their stored search score exactly', rows=ident,
                             verdict='PASS' if identity_ok else 'FAILED'),
               scoring=dict(raw_input_sha256=raw['input_sha256'], settings=raw['settings'], rollouts=raw['rollouts'],
                            wall_s=raw['wall_s'], new_sumo_sim_s=0))
    if not identity_ok:
        out['verdict'] = 'STOPPED: the A42 scoring could not be reproduced exactly (no substitute scoring)'
        return out
    for f in [r['family'] for r in raw['rows'] if r['role'] == 'a56_random']:
        rows.append(dict(family=f, source='A56', score=new[f]['score'], sumo=a56[f]['mean_minus_u0'],
                         PASS=bool(a56[f]['PASS'])))
    if len(rows) != 30 or len({r['family'] for r in rows}) != 30:
        raise SystemExit(f'expected 30 distinct plans, got {len(rows)}')
    s = np.array([r['score'] for r in rows], float)
    y = np.array([r['sumo'] for r in rows], float)
    rho, p = perm_test(s, y)
    sec = [r for r in rows if r['source'] in ('A45', 'A56')]
    rho2, p2 = perm_test([r['score'] for r in sec], [r['sumo'] for r in sec])
    order = sorted(rows, key=lambda r: (-r['score'], r['family']))
    top, bot = order[:15], order[15:]
    out.update(
        convention=('score: A42-search WM score (5% world-cluster LCB of predicted gain vs base minus 0.5 x draw sd / '
                    'sqrt(n)); higher = model predicts more saving. sumo: stored plan - U0 mean total delay (veh-s) on '
                    '992000-992023; negative = less delay. Helpful direction: rho < 0.'),
        primary=dict(n_plans=len(rows), spearman_rho=rho, p_one_sided=p, permutations=N_PERM, seed=SEED,
                     rule='SCORE INFORMATIVE iff rho < 0 (helpful direction) and one-sided permutation p < 0.05',
                     verdict='SCORE INFORMATIVE' if (rho < 0 and p < 0.05) else 'NOT SHOWN'),
        secondary=dict(
            not_selected_by_score=dict(n_plans=len(sec), spearman_rho=rho2, p_one_sided=p2, sources=['A45', 'A56']),
            halves=dict(rule='30 plans by WM score (ties by family name): top 15 vs bottom 15, admitted counts',
                        top_admitted=sum(r['PASS'] for r in top), bottom_admitted=sum(r['PASS'] for r in bot),
                        top=[r['family'] for r in top], bottom=[r['family'] for r in bot])),
        plans=rows, a59_included=False,
        a59_note='A59 HAND3/HAND5 had no stored admission at analysis time; not included (registered "if their '
                 'admission exists at analysis time")')
    return out


# ------------------------------------------------------------------------------------------------ 60A

def analyse_a(red):
    worlds = red['worlds']
    wm, rnd = red['wm_top6'], red['random12']
    if len(wm) != 6 or len(rnd) != 12 or len(worlds) != 96:
        raise SystemExit('reduce file does not have 6 WM plans, 12 random plans, 96 worlds')
    J = {a: np.array([red['per_world'][a][w]['J_veh_s'] for w in worlds], float) for a in ['U0']+wm+rnd}
    mean_wm, mean_rnd = np.mean([J[a] for a in wm], 0), np.mean([J[a] for a in rnd], 0)
    prim = paired_ci(mean_wm-mean_rnd)
    lo, hi = prim['ci95']
    verdict = 'SUPPORTED' if hi < 0 else 'RANDOM AHEAD' if lo > 0 else 'NOT SUPPORTED'
    V = red['verdicts']
    rows = []
    for f in wm+rnd:
        v = V[f]
        rows.append(dict(family=f, source=v['source'], plan_minus_u0=paired_ci(J[f]-J['U0']),
                         admitted_on_60A=bool(v['PASS']), ci95_unc=v['ci95_unc'], allowed=v['allowed'],
                         group_all_pass=(v['groups'] or {}).get('all_pass'),
                         cvar90_diff_s=(v.get('cvar90_vs_u0') or {}).get('diff_mean_s'),
                         activations_per_ep=v['activations_per_ep'], episodes_with_activation=v['episodes_with_activation'],
                         safety_totals=v['safety_totals']))
    safety = {grp: {k: int(sum(V[f]['safety_totals'][k] for f in fams)) for k in V[rnd[0]]['safety_totals']}
              for grp, fams in (('WM6', wm), ('RANDOM12', rnd))}
    safety['U0'] = {k: int(sum(red['per_world']['U0'][w]['safety'][k] for w in worlds))
                    for k in red['per_world']['U0'][worlds[0]]['safety']}
    return dict(schema='coopmwm.a60a.results.v1', amendment='60A', label='S-WMPROP96', cell=red['cell'],
                reduce_analysis_sha256=red['analysis_sha256'],
                convention='diff = treatment - reference (veh-s total delay; negative = less delay)',
                primary=dict(contrast='per world mean J(six A42 WM plans) - mean J(twelve A56 random plans)',
                             bootstrap=dict(reps=REPS, seed=SEED, level=LEVEL, method='paired percentile'),
                             mean_wm6_J=float(mean_wm.mean()), mean_random12_J=float(mean_rnd.mean()),
                             mean_u0_J=float(J['U0'].mean()), **prim,
                             rule='SUPPORTED iff CI upper < 0; RANDOM AHEAD iff CI lower > 0; else NOT SUPPORTED',
                             verdict=verdict),
                admission=dict(wm_k=sum(V[f]['PASS'] for f in wm), wm_n=6, random_k=sum(V[f]['PASS'] for f in rnd),
                               random_n=12),
                plans=rows, safety_totals=safety,
                per_world=dict(worlds=worlds, u0=J['U0'].tolist(), mean_wm6=mean_wm.tolist(),
                               mean_random12=mean_rnd.tolist()),
                charged_sim_s=red['charged_sim_s'])


# ------------------------------------------------------------------------------------------------ 60C

def analyse_c(red):
    worlds = red['worlds']
    J = {a: np.array([red['per_world'][a][w]['J_veh_s'] for w in worlds], float) for a in red['per_world']}
    C = {(c['treatment'], c['reference']): c for c in red['contrasts']}
    prim = C[('BIG_BANK', 'RAND_BANK')]
    t = prim['total_minus_ref']
    lo, hi = t['ci95']
    verdict = 'MODEL BANK AHEAD' if hi < 0 else 'RANDOM BANK AHEAD' if lo > 0 else 'NOT RESOLVED'

    def brief(c):
        return dict(treatment=c['treatment'], reference=c['reference'], total_minus_ref=c['total_minus_ref'],
                    cvar90=c.get('tail_cvar90'), groups=c.get('groups'),
                    safety={k: c.get(f'{k}_minus_ref') for k in ('collisions', 'teleports', 'hard_brakes', 'ttc_events')})
    safety = {a: {k: int(sum(red['per_world'][a][w]['safety'][k] for w in worlds))
                  for k in red['per_world'][a][worlds[0]]['safety']} for a in red['per_world']}
    return dict(schema='coopmwm.a60c.results.v1', amendment='60C', label='S-RANDBANK60', cell=red['cell'],
                reduce_analysis_sha256=red['analysis_sha256'], panel_sha256=red['panel_sha256'],
                convention='diff = treatment - reference (veh-s total delay; negative = less delay)',
                primary=dict(contrast='BIG_BANK - RAND_BANK total delay (96 worlds 998600-998695)', **t,
                             crosscheck_paired_ci=paired_ci(J['BIG_BANK']-J['RAND_BANK']),
                             rule='MODEL BANK AHEAD iff CI upper < 0; RANDOM BANK AHEAD iff CI lower > 0; else NOT '
                                  'RESOLVED (a tie is the registered expectation)', verdict=verdict),
                secondary=[brief(c) for c in red['contrasts'][1:]], primary_detail=brief(prim),
                mean_J={a: float(J[a].mean()) for a in J}, safety_totals=safety, rand_bank=red['rand_bank'],
                charged_sim_s=red['charged_sim_s'])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('part', choices=('score', 'a', 'c'))
    p.add_argument('--search')
    p.add_argument('--raw')
    p.add_argument('--adm42')
    p.add_argument('--adm45')
    p.add_argument('--randctrl')
    p.add_argument('--reduce')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out}: exists (never overwritten)')
    rd = lambda x: json.loads(Path(x).read_text(encoding='utf-8'))  # noqa: E731
    if a.part == 'score':
        res = analyse_score(rd(a.search), rd(a.raw), rd(a.adm42), rd(a.adm45), rd(a.randctrl))
        res['inputs'] = {k: sha_file(getattr(a, k)) for k in ('search', 'raw', 'adm42', 'adm45', 'randctrl')}
    else:
        red = rd(a.reduce)
        if not check_seal(red, 'analysis_sha256'):
            raise SystemExit('reduce file seal does not verify')
        res = analyse_a(red) if a.part == 'a' else analyse_c(red)
        res['inputs'] = dict(reduce_file_sha256=sha_file(a.reduce))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(out=str(out), sha256=sha_file(out), primary=res.get('primary', {}).get('verdict'),
                          identity=res.get('identity', {}).get('verdict'), verdict=res.get('verdict'))))


if __name__ == '__main__':
    main()
