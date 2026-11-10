"""AMENDMENT 58 analysis (zero physics, laptop): FAMACONF.json / FAMACONF.md / per-world table / WINCOUNT.json.

Inputs are offloaded copies (sealed receipts re-verified; cached vehicles_report.json checked against the sealed log sha
by a32_ablation.report_of):
  --fama  <copy of /root/a58/root>: WAVE_A58C_FAMA__*.json + evaluation/a58c_fama_*/vehicles_report.json
  --a50   <copy>: WAVE_A50C_{U0,DUO_HOLD}__*.json + evaluation/a50c_{u0,duo_hold}_*/vehicles_report.json
  --a52   <copy>: WAVE_A52C_MAPPO60__*.json + evaluation/a52c_mappo60_*/vehicles_report.json
  --a42c  <copy>: WAVE_A42C_{U0,DUO_HOLD,BIG_BANK}__*.json (panel-1 win count only; J per world)
  --bank  A42_BIG_BANK_D2800_p60.json (pinned sha; plan roles for the realised group sizes)

Registered (REGISTRATION.md, Amendment 58): PRIMARY FAMA - DUO_HOLD total delay per world, paired percentile bootstrap
10,000, seed 20261104; CONFIRMED iff CI95 upper < 0 AND FAMA meets the group rule vs U0 (every registered group-mean
increase <= 10 s); else NOT CONFIRMED. Secondary (no verdict): FAMA - U0, FAMA - MAPPO60, worst-10 % (CVaR90) and
hard-brake vehicle-seconds per contrast, win counts (FAMA < DUO_HOLD and < U0) for panel 2 and, from stored records,
the A42 BIG_BANK on panel 1; activations by group size; collisions; teleports. diff = treatment - reference;
negative = less delay. The contrast code is the A42 driver's own _contrast / primary_verdict (ops/a42_bigp60.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a58_famaconf as M  # noqa: E402

MAPPO60_SHA256 = '7f0c88695161bae6a243801df94e92fbd27c87551d655be61d338fd369364443'
PRIMARY_RULE = ('A58 CONFIRMED iff FAMA - DUO_HOLD total delay CI95 upper < 0 (paired percentile bootstrap 10,000, '
                'seed 20261104) AND FAMA passes the group rule vs U0 (registered groups, each mean per-vehicle '
                'increase <= 10 s, U0 labels); otherwise NOT CONFIRMED')
CONTRASTS = (('FAMA', 'DUO_HOLD', 'primary'), ('FAMA', 'U0', 'secondary'), ('FAMA', 'MAPPO60', 'secondary'),
             ('DUO_HOLD', 'U0', 'context'), ('MAPPO60', 'U0', 'context'))


def load_arm(root, wave, prefix, worlds):
    recs = M.stored_records(root, wave, len(worlds))
    if set(recs) != set(worlds):
        raise N.Blocked(f'{wave}: worlds differ from the panel')
    for r in recs.values():
        if not r['job_id'].startswith(prefix):
            raise N.Blocked(f'{wave}: unexpected job id {r["job_id"]}')
    return recs


def strategies(bank_path):
    if N.file_sha(bank_path) != M.BANK_SHA256:
        raise N.Blocked('bank file is not the admitted A42 bank')
    doc = json.loads(Path(bank_path).read_text(encoding='utf-8'))
    out = {t['family']: t['offer_condition'] for t in doc['clean5c']['templates']}
    if tuple(out) != M.A42_ADMITTED:
        raise N.Blocked(f'bank families {list(out)}')
    return out


def wins(t, u0, duo, worlds):
    w = [x for x in worlds if t[x]['J_veh_s'] < duo[x]['J_veh_s'] and t[x]['J_veh_s'] < u0[x]['J_veh_s']]
    return dict(n_worlds=len(worlds), wins_both=len(w),
                lower_than_duo_hold=sum(t[x]['J_veh_s'] < duo[x]['J_veh_s'] for x in worlds),
                lower_than_u0=sum(t[x]['J_veh_s'] < u0[x]['J_veh_s'] for x in worlds), worlds=sorted(w))


def analyse(a):
    import numpy as np
    from a32_ablation import report_of
    from tools.coopmwm import a42_bigbank as B
    panel = M.D.setup(M.CONF_SETTING)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    roots = dict(FAMA=Path(a.fama), U0=Path(a.a50), DUO_HOLD=Path(a.a50), MAPPO60=Path(a.a52))
    have = dict(FAMA=load_arm(roots['FAMA'], 'A58C_FAMA', 'a58c_fama_', worlds),
                U0=load_arm(roots['U0'], 'A50C_U0', 'a50c_u0_', worlds),
                DUO_HOLD=load_arm(roots['DUO_HOLD'], 'A50C_DUO_HOLD', 'a50c_duo_hold_', worlds),
                MAPPO60=load_arm(roots['MAPPO60'], 'A52C_MAPPO60', 'a52c_mappo60_', worlds))
    for r in have['FAMA'].values():
        ck = r['checkpoint']
        if ck.get('a37_arm') != 'CELL_BANK' or ck['a37_bank']['file_sha256'] != M.BANK_SHA256 \
                or r.get('a37_sha256_loaded') != ck.get('a37_sha256') or r['panel_sha256'] != panel['panel_sha256']:
            raise N.Blocked(f'{r["job_id"]}: not the A42 bank / a37 executor / panel')
    for r in have['MAPPO60'].values():
        if r['checkpoint'].get('executor_sha256') != MAPPO60_SHA256:
            raise N.Blocked(f'{r["job_id"]}: not the MAPPO60 executor')
    for arm in have:
        for w in worlds:
            if have[arm][w]['world'] != have['U0'][w]['world']:
                raise N.Blocked(f'{arm} {w}: world spec differs from U0')
    reports = {arm: {w: report_of(roots[arm], have[arm][w]) for w in worlds} for arm in have}
    if any(not x['ok'] for arm in reports for x in reports[arm].values()):
        raise N.Blocked('invalid vehicle-log report')
    contrasts = [M.D._contrast(have, reports, worlds, t, r, bootstrap, k) for t, r, k in CONTRASTS]
    prim = M.D.primary_verdict(contrasts[0], contrasts[1])
    prim.update(rule=PRIMARY_RULE, verdict='CONFIRMED' if prim['supported'] else 'NOT CONFIRMED')
    specs = strategies(a.bank)
    rows = []
    for w in worlds:
        rows += B.realised_sizes(have['FAMA'][w].get('a37') or {}, specs)
    fam = {}
    for w in worlds:
        for f, n in ((have['FAMA'][w].get('a37') or {}).get('activations_by_family') or {}).items():
            fam[f] = fam.get(f, 0)+int(n)
    arms = {arm: dict(mean_J_veh_s=float(np.mean([have[arm][w]['J_veh_s'] for w in worlds])),
                      safety_totals={k: int(sum(have[arm][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for arm in have}
    arms['FAMA']['activations'] = dict(
        total=sum(int((have['FAMA'][w].get('a37') or {}).get('activations') or 0) for w in worlds),
        worlds_with_any=sum(1 for w in worlds if (have['FAMA'][w].get('a37') or {}).get('activations')),
        by_family=fam, realised_group_size=B.group_size_summary(rows))
    # panel 1 (stored A42 records; zero simulation) and panel 2 win counts
    p1 = {arm: M.stored_records(a.a42c, f'A42C_{arm}', 96) for arm in ('U0', 'DUO_HOLD', 'BIG_BANK')}
    w1 = sorted(p1['U0'])
    rows1 = []
    for w in w1:
        rows1 += B.realised_sizes(p1['BIG_BANK'][w].get('a37') or {}, specs)
    wincount = dict(schema='coopmwm.a58.wincount.v1', amendment=M.AMENDMENT,
                    rule='worlds in which the searched bank has lower total delay than BOTH DUO_HOLD and U0 '
                         '(strict <, per world, same world seed)',
                    panel1=dict(source='stored A42 confirmation records (WAVE_A42C_*; zero simulation)',
                                arm='BIG_BANK (A42)', **wins(p1['BIG_BANK'], p1['U0'], p1['DUO_HOLD'], w1),
                                group_size_check=B.group_size_summary(rows1)),
                    panel2=dict(source='A58 FAMA vs stored A50 U0 / DUO_HOLD', arm='FAMA',
                                **wins(have['FAMA'], have['U0'], have['DUO_HOLD'], worlds)))
    per_world = [dict(world_id=w, seed=have['U0'][w]['world']['seed'],
                      **{f'J_{arm}': have[arm][w]['J_veh_s'] for arm in have},
                      **{f'hard_brakes_{arm}': have[arm][w]['safety']['hard_brakes'] for arm in have},
                      fama_activations=(have['FAMA'][w].get('a37') or {}).get('activations'))
                 for w in worlds]
    value = N.seal(dict(schema='coopmwm.a58.analysis.v1', amendment=M.AMENDMENT, authority=M.AUTHORITY, cell=M.CELL,
                        setting=M.CONF_SETTING, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                        bootstrap=bootstrap, bank=dict(file=M.BANK_FILE, sha256=M.BANK_SHA256,
                                                       families=list(M.A42_ADMITTED)),
                        primary=prim, contrasts=contrasts, arms=arms,
                        groups_vs_u0={c['treatment']: c['groups'] for c in contrasts if c['reference'] == 'U0'},
                        wincount=wincount, label=M.D.RESULT_LABEL,
                        note='panels are not pooled; A42 panel-1 row reported alongside (A42_RESULTS_D2800_p60.json)'),
                   'analysis_sha256')
    return value, wincount, per_world


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for k in ('fama', 'a50', 'a52', 'a42c', 'bank', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args(argv)
    value, wincount, per_world = analyse(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'FAMACONF.json').write_text(json.dumps(value, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    (out/'WINCOUNT.json').write_text(json.dumps(wincount, indent=1, sort_keys=True)+'\n', encoding='utf-8',
                                     newline='\n')
    cols = list(per_world[0])
    lines = [','.join(cols)] + [','.join(str(r[c]) for c in cols) for r in per_world]
    (out/'FAMACONF_PER_WORLD.csv').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(analysis_sha256=value['analysis_sha256'], primary=value['primary']['verdict'],
                          ci=value['primary']['a_total_ci95'], wins=(wincount['panel1']['wins_both'],
                                                                     wincount['panel2']['wins_both']))))


if __name__ == '__main__':
    main()
