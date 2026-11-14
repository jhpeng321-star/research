"""Transcribe recorded E1/E2 results into E1_SUMMARY.csv and E2_SUMMARY.csv.

Zero simulation, zero new statistics: every value is copied from an existing JSON field (path given per row)
or, where noted, from a REGISTRATION.md line. Run with:
  C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I build_summaries.py
"""
import csv
import hashlib
import json
from pathlib import Path

CV = Path('E:/research-dense/experiments/claim_validation_20261113_0231')
DEL = CV / 'deliverables'
REG = 'runs/active/C4-V7FT-20261017/REGISTRATION.md'

E1A = CV / 'e1_conf' / 'E1_ANALYSIS.json'
FB = DEL / 'RAW_RESULTS' / 'e1_bank_box4_extracted' / 'bank' / 'FROZEN_BANKS.json'
E2A = CV / 'box4_out' / 'e2' / 'E2_ANALYSIS.json'
E2S = CV / 'box4_out' / 'e2' / 'E2_SELECTION.json'

SEEDS = [11, 23, 42, 67, 89, 103]
HOLM = CV / 'e2_secondary' / 'box4' / 'E2_HOLM.json'


def holm_rows():
    h = json.loads(HOLM.read_text())
    out = []
    for k in ('E2 F - H* delay', 'E1 admitted count'):
        out.append(row(experiment='E1+E2', row_type='Holm over the two registered primaries', seed='all',
                       arm_or_contrast=k, value=h['adjusted'][k], p_two_sided=h['raw'][k],
                       unit='value = Holm-adjusted p; p_two_sided = raw p',
                       status='REGISTERED MULTIPLICITY STEP - rejected: ' + str(h['rejected'][k]),
                       source_file=rel(HOLM) + ' (via e2_secondary/E2_SECONDARY_REPORT.md)',
                       source_sha256=sha(HOLM), source_json_path=f'adjusted/raw/rejected["{k}"]',
                       registration_line=f'{REG}:11671',
                       note='computed 13:13 by the e2_secondary agent; recorded at REG:11671 (header 14:50; '
                            'clock correction at REG:11673 says it was written before 13:17 real time)'))
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p)).replace('\\', '/').replace('E:/research-dense/', '')


FIELDS = ['experiment', 'row_type', 'seed', 'arm_or_contrast', 'budget', 'value', 'ci95_low', 'ci95_high',
          'p_two_sided', 'pct', 'pct_ci95_low', 'pct_ci95_high', 'mean_treatment', 'mean_reference',
          'n', 'worlds_lower', 'worlds_higher', 'unit', 'status', 'source_file', 'source_sha256',
          'source_json_path', 'registration_line', 'note']


def row(**kw):
    r = {k: '' for k in FIELDS}
    r.update({k: ('' if v is None else v) for k, v in kw.items()})
    return r


def contrast_cols(c):
    out = dict(value=c.get('diff'), n=c.get('n'), p_two_sided=c.get('p_two_sided'), pct=c.get('pct'),
               mean_treatment=c.get('mean_treatment'), mean_reference=c.get('mean_reference'),
               worlds_lower=c.get('worlds_lower'), worlds_higher=c.get('worlds_higher'))
    if c.get('ci95'):
        out['ci95_low'], out['ci95_high'] = c['ci95']
    if c.get('pct_ci95'):
        out['pct_ci95_low'], out['pct_ci95_high'] = c['pct_ci95']
    return out


def build_e1():
    a = json.loads(E1A.read_text())
    fb = json.loads(FB.read_text())
    sa, sf = sha(E1A), sha(FB)
    fbsrc = rel(FB) + ' (extracted unchanged from e1_bank_box4.tgz sha256 ' + sha(CV / 'e1_bank_box4.tgz') + ')'
    rows = []
    pac = fb['primary_admitted_count']
    for i, s in enumerate(SEEDS):
        for arm in ('WM', 'RANDOM'):
            rows.append(row(experiment='E1', row_type='admitted_count (PRIMARY input)', seed=s, arm_or_contrast=arm,
                            budget=6, value=pac[arm][i], unit='plans admitted of 6', status='REGISTERED PRIMARY',
                            source_file=fbsrc, source_sha256=sf,
                            source_json_path=f'primary_admitted_count.{arm}[{i}]', registration_line=f'{REG}:11605'))
        rows.append(row(experiment='E1', row_type='admitted_count_difference (PRIMARY)', seed=s,
                        arm_or_contrast='WM - Random', budget=6, value=pac['wm_minus_random'][i],
                        unit='plans', status='REGISTERED PRIMARY', source_file=fbsrc, source_sha256=sf,
                        source_json_path=f'primary_admitted_count.wm_minus_random[{i}]',
                        registration_line=f'{REG}:11605', note='positive = more admitted under WM order'))
    perm = pac['permutation']
    rows.append(row(experiment='E1', row_type='PRIMARY test: exact sign-flip permutation', seed='all',
                    arm_or_contrast='WM - Random admitted count', budget=6, value=perm['mean'],
                    p_two_sided=perm['p_two_sided'], n=perm['n'], unit='mean plans per seed',
                    status='REGISTERED PRIMARY - NOT SUPPORTED', source_file=fbsrc, source_sha256=sf,
                    source_json_path='primary_admitted_count.permutation', registration_line=f'{REG}:11605',
                    note=(f"64 assignments; p_less {perm['p_less']}; min attainable two-sided p "
                          f"{perm['min_p_two_sided']}; totals WM {sum(pac['WM'])} vs Random {sum(pac['RANDOM'])}")))
    for b in ('2', '4'):
        ad = a['by_budget'][b]['admitted']
        for i, s in enumerate(SEEDS):
            for arm in ('WM', 'RANDOM'):
                rows.append(row(experiment='E1', row_type='admitted_count (descriptive budget)', seed=s,
                                arm_or_contrast=arm, budget=int(b), value=ad[arm][i], unit=f'plans admitted of {b}',
                                status='DESCRIPTIVE (registered as descriptive)', source_file=rel(E1A),
                                source_sha256=sa, source_json_path=f'by_budget.{b}.admitted.{arm}[{i}]',
                                note=a['by_budget'][b].get('not_run', '')))
    b6 = a['by_budget']['6']
    for i, s in enumerate(SEEDS):
        rows.append(row(experiment='E1', row_type='bank delay contrast per seed (secondary)', seed=s,
                        arm_or_contrast='WM bank - Random bank', budget=6, value=b6['wm_minus_random_per_seed'][i],
                        n=96, unit='veh-s total delay per world', status='REGISTERED SECONDARY',
                        source_file=rel(E1A), source_sha256=sa,
                        source_json_path=f'by_budget.6.wm_minus_random_per_seed[{i}]',
                        registration_line=f'{REG}:11620',
                        note='per-seed CI not recorded in E1_ANALYSIS.json; not computed here'))
    w = b6['wm_minus_random']
    pr = b6['permutation']
    rows.append(row(experiment='E1', row_type='bank delay contrast overall (secondary)', seed='all',
                    arm_or_contrast='WM bank - Random bank', budget=6, value=w['diff'], ci95_low=w['ci95'][0],
                    ci95_high=w['ci95'][1], p_two_sided=pr['p_two_sided'], n='6 seeds x 96 worlds',
                    unit='veh-s total delay per world', status='REGISTERED SECONDARY - NOT SUPPORTED',
                    source_file=rel(E1A), source_sha256=sa,
                    source_json_path='by_budget.6.wm_minus_random (CI: seed x world crossed bootstrap, reps 10000, '
                                     'seed 20261104) + by_budget.6.permutation (p: exact sign-flip over 6 seeds)',
                    registration_line=f'{REG}:11620',
                    note=(f"permutation p_less {pr['p_less']}; positive point estimate = Random bank less delay; "
                          'not separated. FLAG: E1_ANALYSIS.json marks by_budget.6 "primary": true while its "role" '
                          'field says secondary (delay); the registered primary is the admitted count')))
    for arm in ('WM', 'RANDOM'):
        for i, s in enumerate(SEEDS):
            k = f'{arm}_seed{s}'
            c = b6['bank_minus_u0'][k]
            r = row(experiment='E1', row_type='bank - U0 (secondary)', seed=s, arm_or_contrast=f'{arm} bank - U0',
                    budget=6, unit='veh-s total delay per world; pct = % of U0 mean',
                    status='REGISTERED SECONDARY', source_file=rel(E1A), source_sha256=sa,
                    source_json_path=f'by_budget.6.bank_minus_u0.{k}', registration_line=f'{REG}:11621',
                    note=c.get('note', 'paired percentile bootstrap 10,000, seed 20261104'))
            r.update(contrast_cols(c))
            rows.append(r)
    rows.extend(holm_rows())
    return rows


def build_e2():
    a = json.loads(E2A.read_text())
    s = json.loads(E2S.read_text())
    sa, ss = sha(E2A), sha(E2S)
    rows = []
    for t in s['table']:
        rows.append(row(experiment='E2', row_type='tuning (Development worlds, selection input)',
                        arm_or_contrast=t['arm'], value=t['mean_total_delay'], n=24,
                        unit='veh-s mean total delay per world', status='REGISTERED SELECTION STEP',
                        source_file=rel(E2S), source_sha256=ss, source_json_path=f"table[{t['arm']}]",
                        registration_line=f'{REG}:11559',
                        note=f"mean minus U0 = {t['mean_minus_u0']}; selected H* = {s['selected_arm']} by rule: "
                             + s['rule']))
    p = a['primary']
    r = row(experiment='E2', row_type='PRIMARY', arm_or_contrast='F - H20', unit='veh-s total delay per world',
            status='REGISTERED PRIMARY - NOT SUPPORTED', source_file=rel(E2A), source_sha256=sa,
            source_json_path='primary', registration_line=f'{REG}:11560',
            note='paired percentile bootstrap 10,000, seed 20261104; positive = F more delay; not separated')
    r.update(contrast_cols(p))
    rows.append(r)
    wc = a['F_minus_H_worst_case']
    rows.append(row(experiment='E2', row_type='primary worst case over confirmed H arms', arm_or_contrast='F - H*',
                    value=wc['ci95_upper'], p_two_sided=wc['p_two_sided_max'], unit='veh-s (CI95 upper)',
                    status='REGISTERED (analysis field)', source_file=rel(E2A), source_sha256=sa,
                    source_json_path='F_minus_H_worst_case',
                    note=f"F_better_than_every_H = {wc['F_better_than_every_H']}; only H20 was confirmed"))
    r = row(experiment='E2', row_type='secondary', arm_or_contrast='F - S', unit='veh-s total delay per world',
            status='REGISTERED SECONDARY - recorded', source_file=rel(E2A), source_sha256=sa,
            source_json_path='F_minus_S', registration_line=f'{REG}:11560', note='not separated')
    r.update(contrast_cols(a['F_minus_S']))
    rows.append(r)
    for arm in ('F', 'H20', 'S'):
        r = row(experiment='E2', row_type='secondary', arm_or_contrast=f'{arm} - U0',
                unit='veh-s total delay per world; pct = % of U0 mean', status='REGISTERED SECONDARY - recorded',
                source_file=rel(E2A), source_sha256=sa, source_json_path=f'vs_u0.{arm}',
                registration_line=f'{REG}:11560')
        r.update(contrast_cols(a['vs_u0'][arm]))
        rows.append(r)
    for arm in ('F', 'H20', 'S'):
        ce = a['control_effort'][arm]
        for k, v in ce['per_episode_mean'].items():
            rows.append(row(experiment='E2', row_type='secondary: executed control effort',
                            arm_or_contrast=f'{arm}: {k}', value=v, n=ce['n_worlds'], unit='per world (mean)',
                            status='REGISTERED SECONDARY - recorded', source_file=rel(E2A), source_sha256=sa,
                            source_json_path=f'control_effort.{arm}.per_episode_mean.{k}',
                            registration_line=f'{REG}:11560', note=f"total {ce['total'][k]}"))
    rep = CV / 'e2_secondary' / 'E2_SECONDARY_REPORT.md'
    rows.append(row(experiment='E2', row_type='secondary: same-state timing replay (<= 24 Development worlds)',
                    arm_or_contrast='F1 - S1 same state (also F1 - U0, S1 - U0)',
                    status='PREPARED, NOT RUN (0 sim-s charged; cap 115,200)', source_file=rel(rep),
                    source_sha256=sha(rep), registration_line=f'{REG}:11560',
                    note='04:51 record: "Same-state timing replay not yet run"; E2_SECONDARY_REPORT.md: zero-SUMO '
                         'dry run 13:15 only'))
    rows.append(row(experiment='E2', row_type='reproduction check of sealed E2_ANALYSIS.json',
                    arm_or_contrast='all E2 contrasts above', value='byte-identical',
                    status='DONE (e2_secondary agent)', source_file=rel(rep), source_sha256=sha(rep),
                    note='E2_ANALYSIS_REPRO.json sha256 ' + sha(CV / 'e2_secondary' / 'box4' / 'E2_ANALYSIS_REPRO.json')))
    rows.extend(holm_rows())
    return rows


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    write(DEL / 'E1_SUMMARY.csv', build_e1())
    write(DEL / 'E2_SUMMARY.csv', build_e2())
    print('ok')
