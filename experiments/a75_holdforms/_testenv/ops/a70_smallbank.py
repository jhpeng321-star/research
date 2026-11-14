"""AMENDMENT 70 (S-SMALLBANK60C; MAIN 2026-11-13; POST-HOC): do the best new A68 plans work when the bank is small?

Motivated by the failed A68 confirmation (MAIN bank of 8 stacked plans, +187 veh-s, L2 +13.1 s), which stays reported.
Plans, executor, caps and the MAPPO60C checkpoint (A65 policy_00084, 2abed317) are unchanged; nothing is retuned.

Stages (own root, one ledger phase ``smallbank_a70``, cap 506,880 sim-s = identity 4,800 + arms 460,800 + 41,280
reserved for recorded infrastructure relaunches only):
  prepare   zero SUMO: plan choice from the A68 ADMISSION receipt only (never the A68 confirmation panel): admitted
            plans in the A68 bank order (admission CI95 upper of plan - MAPPO60C, most negative first; ties by
            strategy key), minus any plan whose 24 per-world admission differences equal those of an admitted plan
            selected earlier by the search (duplicate). SMALL1 = the first plan; SMALL2 = the first two.
  identity  integrity repeat for the changed code tree (table-only edits): MAPPO60C alone and old BIG_BANK@60C on the
            first 2 worlds of panel 998600-998695 must equal the stored A66 / A67 records exactly (as A68)
  confirm   fresh 96 worlds 996300-996395: MAPPO60C, DUO_HOLD_60C, SMALL1_60C, SMALL2_60C
  analyse   primary SMALL1_60C - MAPPO60C: SUPPORTED iff total delay CI95 upper < 0 AND the five-group 10 s rule
            relative to MAPPO60C; every other contrast descriptive. Paired percentile bootstrap 10,000, seed 20261104,
            diff = treatment - reference (negative = less delay). Never pooled with the A67 / A68 panels.
No automatic replay: a failed dispatch raises and is reported. No executor / runner behaviour code is changed.

Commands (python ops/a70_smallbank.py <cmd> --final-root R --a68-root R68 --a66-root R66 --a67-root R67 --a42-root R42):
  prepare ; dry --stage i|c ; identity ; identity-compare ; confirm ; analyse ; status --stage i|c
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

AMENDMENT = '70'
CELL = 'D2800_p60'
PHASE = 'smallbank_a70'
IDENT_N, CONFIRM_N = 2, 96
IDENT_CAP, ARMS_SIM_S, CAP_SIM_S = 4_800, 460_800, 506_880
SETTINGS = dict(i='a42c_D2800_p60', c='a70c_D2800_p60')
RANGES = dict(i='A42_RANGES', c='A70_RANGES')
SEEDS = dict(i=tuple(range(998600, 998696)), c=tuple(range(996300, 996396)))
PANEL_SHA256 = dict(c='1b4b59ee4ab717b931647fbae7cfa489726cd2cbacad30e268c57e32cb2c78e1')     # replaced by the pinned value after the panel build (before any A70 episode)
BANK_DIR = 'a70_banks'
SMALL1_BANK = f'A70_SMALL1_BANK_{CELL}.json'
SMALL2_BANK = f'A70_SMALL2_BANK_{CELL}.json'
OLD_BANK = f'A42_BIG_BANK_{CELL}.json'
OLD_BANK_SHA256 = '7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff'
A68_ADMISSION_RECEIPT = 'd527efbe32c931e29726f4ce1c14fc9743e1ac9f216bf67515ff23f7fc3a5b48'
SELECTION_FILE = f'A70_SELECTION_{CELL}.json'
IDENTITY_FILE = 'IDENTITY_A70.json'
RESULTS_FILE = f'A70_RESULTS_{CELL}.json'
REF = 'MAPPO60C'
DUO_BANK_CELL = 'D2800_p20'
BANK_OF = dict(SMALL1_60C=SMALL1_BANK, SMALL2_60C=SMALL2_BANK)
ARMS = ('MAPPO60C', 'DUO_HOLD_60C', 'SMALL1_60C', 'SMALL2_60C')
PRIMARY = ('SMALL1_60C', 'MAPPO60C')
SECONDARY = (('SMALL2_60C', 'MAPPO60C'), ('SMALL1_60C', 'DUO_HOLD_60C'), ('SMALL2_60C', 'DUO_HOLD_60C'),
             ('SMALL2_60C', 'SMALL1_60C'), ('DUO_HOLD_60C', 'MAPPO60C'))
AUTHORITY = ('MAIN AMENDMENT 70 (S-SMALLBANK60C, 2026-11-13; POST-HOC, motivated by the failed A68 confirmation): the '
             'best one / two A68-admitted plans as a small bank on MAPPO60C (2abed317), fresh 96-world panel '
             '996300-996395; phase smallbank_a70 cap 506,880')
DISCLOSURE = ('POST-HOC test: designed after the A68 confirmation failed (MAIN bank +187 veh-s, origin L2 +13.1 s); '
              'that result stays reported. Plan choice uses the A68 admission receipt only; the A68 confirmation '
              'panel plays no part. The admission lower bounds that rank the plans are selection-biased upward (best '
              'of 12 on 24 worlds). Never pooled with the A67 / A68 panels')
SELECT_RULE = ('A68-admitted (MAIN) plans in the A68 bank order = admission CI95 upper of (plan - MAPPO60C) most '
               'negative first, ties by strategy key (a37_cellbank.admitted_plans); a plan whose 24 per-world admission '
               'differences, mean and CI equal those of an admitted plan with an earlier search selection rank is '
               'dropped as a duplicate; SMALL1 = the first remaining plan, SMALL2 = the first two; plans, lb95 values '
               'and every executor field unchanged')
PRIMARY_RULE = ('SUPPORTED iff SMALL1_60C - MAPPO60C total delay CI95 upper < 0 (paired percentile bootstrap 10,000, '
                'seed 20261104, n = 96) AND the five-group 10 s rule relative to MAPPO60C; diff = treatment - '
                'reference, negative = less delay')


def _mods():
    import nm_finaleval as N
    import a50_fullp60 as A50
    import a67_plans_on_60c as A67
    import a68_research60c as A68
    return N, A50, A67, A68


def core(which):
    """The A42 driver code with the A70 names; ``which`` = i (identity panel 998600-) or c (the A70 panel)."""
    N, A50, _, _ = _mods()
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A70',
        HOST_BASE=os.environ.get('A70_HOST_BASE', '/root/a70'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A70_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A70', JOB_PREFIX='a70',
        VERIFY_TAG='a70v', CONFIRM_TAG=f'a70{which}', IDENT_TAG='a70i',
        VERIFY_WAVE_PREFIX='A70V_', CONFIRM_WAVE_PREFIX='A70C_',
        RESULTS_FILE=RESULTS_FILE, LOG_FILE='A70_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a70',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 70; POST-HOC small-bank test of the A68-admitted plans on MAPPO60C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a70_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    return m


def setup(which):
    N = _mods()[0]
    D = core(which)
    panel = D.setup(SETTINGS[which])
    seeds = tuple(r['world']['seed'] for r in panel['worlds'])
    if seeds != SEEDS[which] or panel['cell']['cell_key'] != CELL:
        raise N.Blocked(f'{SETTINGS[which]}: not the registered panel')
    if which in PANEL_SHA256 and panel['panel_sha256'] != PANEL_SHA256[which]:
        raise N.Blocked(f'{SETTINGS[which]}: panel sha differs from the registered one')
    return D, panel


# ------------------------------------------------------------------------------------------------ plan choice

def choose(a68_root):
    """Zero SUMO. The registered plan choice (SELECT_RULE) from the A68 admission receipt and the pinned A68
    candidates: -> (ordered kept plans [dict(family, strategy, lb95_veh_s)], audit rows, dropped)."""
    N, _, _, A68 = _mods()
    from tools.coopmwm import a37_cellbank as C
    cands = A68.candidates()
    adm = A68.admission(a68_root)
    if adm is None or adm['receipt_sha256'] != A68_ADMISSION_RECEIPT or adm['amendment'] != '68':
        raise N.Blocked('not the registered A68 admission receipt')
    v = {x['family']: x for x in adm['verdicts']}
    plans = C.admitted_plans(cands, v)
    if [p['family'] for p in plans] != adm['main'] or any(not v[p['family']]['MAIN'] for p in plans):
        raise N.Blocked('admitted plans differ from the A68 MAIN list')
    kept, dropped, rows = [], [], []
    for p in plans:
        x = v[p['family']]
        twin = [q for q in plans if q is not p and v[q['family']]['per_seed'] == x['per_seed']
                and v[q['family']]['mean'] == x['mean'] and v[q['family']]['ci95'] == x['ci95']
                and v[q['family']]['selected_rank'] < x['selected_rank']]
        row = dict(family=p['family'], key=x['key'], selected_rank=x['selected_rank'], mean=x['mean'], ci95=x['ci95'],
                   lb95_veh_s=p['lb95_veh_s'], n_roles=x['n_roles'], activations_per_ep=x['activations_per_ep'],
                   duplicate_of=twin[0]['family'] if twin else None)
        rows.append(row)
        (dropped if twin else kept).append(p)
    if len(kept) < 2:
        raise N.Blocked('fewer than two distinct admitted plans')
    return kept, rows, [p['family'] for p in dropped]


def prepare(D, root, a68_root):
    """Writes the two bank files and the sealed selection record once (zero SUMO); idempotent re-read afterwards."""
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    root = Path(root)
    kept, rows, dropped = choose(a68_root)
    banks = {}
    for name, plans in ((SMALL1_BANK, kept[:1]), (SMALL2_BANK, kept[:2])):
        p = D.bank_path(root, name)
        text = D.C37.bank_text(C.bank_doc(plans, CELL))
        if p.is_file():
            if p.read_text(encoding='utf-8') != text:
                raise N.Blocked(f'{p}: exists with different content')
        else:
            D.C37.write_once(p, text)
        banks[name] = dict(path=f'{D.HOST_ROOT}/{BANK_DIR}/{name}', file_sha256=N.file_sha(p),
                           families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans])
    value = N.seal(dict(schema='coopmwm.a70.selection.v1', amendment=AMENDMENT, cell=CELL, rule=SELECT_RULE,
                        source=dict(a68_admission_receipt_sha256=A68_ADMISSION_RECEIPT,
                                    a68_admission_seeds='992500-992523 (24 worlds)'),
                        ranking=rows, dropped_duplicates=dropped, TOP1=kept[0]['family'],
                        TOP2=[kept[0]['family'], kept[1]['family']], banks=banks), 'receipt_sha256')
    p = root/SELECTION_FILE
    if p.is_file():
        old = N.verify(N.read(p), 'receipt_sha256', p.name)
        if old['receipt_sha256'] != value['receipt_sha256']:
            raise N.Blocked(f'{p.name}: exists with a different selection')
    else:
        N.publish(p, value)
    return value


def selection(root):
    N = _mods()[0]
    p = Path(root)/SELECTION_FILE
    if not p.is_file():
        raise N.Blocked(f'{SELECTION_FILE}: missing (run prepare)')
    return N.verify(N.read(p), 'receipt_sha256', p.name)


# ------------------------------------------------------------------------------------------------ identity

def ident_waves(D, panel, root, a66_root):
    A68 = _mods()[3]
    exe = A68.executor(a66_root)
    return [('A70_IDENT_MAPPO60C', 'episode', A68.mappo_specs(D, panel, root, exe, 'a70i_mappo60c_full')[:IDENT_N]),
            ('A70_IDENT_BIG_BANK_60C', 'episode',
             A68.bank_specs(D, panel, root, OLD_BANK, exe, 'a70i_big_bank', OLD_BANK_SHA256)[:IDENT_N])]


def identity_compare(D, panel, root, a66_root, a67_root, a42_root):
    N, _, A67, _ = _mods()
    waves = ident_waves(D, panel, root, a66_root)
    ref60, _ = A67.stored_ref(panel, a66_root)
    D67 = A67.core()
    refbig, _ = D67.C37._load(a67_root, A67.wave_of('BIG_BANK_60C'),
                              A67.confirm_specs('BIG_BANK_60C', panel, a67_root, a66_root, a42_root))
    rows = []
    for (wave, _, specs), ref in zip(waves, (ref60, refbig)):
        got, _, cov = N.published(root, wave, specs, D.C37.check_a37)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        for r in got:
            a = ref[r['world_id']]
            ra, aa = r.get('a37') or {}, a.get('a37') or {}
            rows.append(dict(wave=wave, world_id=r['world_id'], j_a70=r['J_veh_s'], j_stored=a['J_veh_s'],
                             abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_equal=r['safety'] == a['safety'],
                             same_world=r['world'] == a['world'],
                             executor_is_mappo60c=r.get('executor_sha256_loaded') == A67.MAPPO60C_SHA256
                             and a.get('executor_sha256_loaded') == A67.MAPPO60C_SHA256,
                             activations=[int(ra.get('activations') or 0), int(aa.get('activations') or 0)],
                             vehicle_log_equal=(r.get('vehicle_log') or {}).get('sha256')
                             == (a.get('vehicle_log') or {}).get('sha256')))
    ok = (len(rows) == 2*IDENT_N and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world']
                                         and x['executor_is_mappo60c']
                                         and x['activations'][0] == x['activations'][1] for x in rows))
    return N.seal(dict(schema='coopmwm.a70.identity.v1', amendment=AMENDMENT,
                       rule='MAPPO60C alone and old BIG_BANK@60C on the first 2 worlds of panel 998600-998695 run '
                            'from the A70 code tree (table-only edits) equal the stored A66 / A67 records exactly: '
                            'J_veh_s difference 0, identical safety counters, same activations',
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: the arms may run; FAILED: stop and report'), 'receipt_sha256')


def identity_passed(root):
    N = _mods()[0]
    p = Path(root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ arms

def confirm_specs(D, arm, panel, root, a66_root, sel):
    N, _, A67, A68 = _mods()
    exe = A68.executor(a66_root)
    tag = f'a70c_{arm.lower()}'
    if arm == 'MAPPO60C':
        specs = A68.mappo_specs(D, panel, root, exe, tag)
    elif arm == 'DUO_HOLD_60C':
        specs = A67.with_exec(D.confirm_specs('DUO_HOLD', panel, root), exe, tag)
    else:
        specs = A68.bank_specs(D, panel, root, BANK_OF[arm], exe, tag, sel['banks'][BANK_OF[arm]]['file_sha256'])
    if len(specs) != CONFIRM_N:
        raise N.Blocked(f'{arm}: {len(specs)} specs')
    return specs


def confirm_waves(D, panel, root, a66_root):
    sel = selection(root)
    return [(f'A70C_{a}', 'episode', confirm_specs(D, a, panel, root, a66_root, sel)) for a in ARMS]


def plan_specs(D, root, arm):
    """{family: strategy spec} exactly as the executor rebuilds them from the arm's bank file."""
    N = _mods()[0]
    from tools.coopmwm import a32_arms, oracle_plan as O
    if arm == 'DUO_HOLD_60C':
        duo = a32_arms.duo_bank(N.read(Path(root)/D.A.DUO_BANK_REL), DUO_BANK_CELL)
        return {p['family']: p['spec'] for p in O.bank_plans(duo, DUO_BANK_CELL)[0]}
    return {p['family']: p['spec'] for p in O.bank_plans(N.read(D.bank_path(root, BANK_OF[arm])), CELL)[0]}


def analyse(D, panel, root, a66_root):
    import numpy as np
    N, _, A67, _ = _mods()
    from a32_ablation import report_of
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    sel = selection(root)
    have, sources = {}, {}
    for arm in ARMS:
        have[arm], sources[arm] = D.C37._load(root, f'A70C_{arm}', confirm_specs(D, arm, panel, root, a66_root, sel))
    reports = {a: {w: report_of(root, have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    row = D._contrast(have, reports, worlds, *PRIMARY, bootstrap, 'primary')
    ci = row['total_minus_ref']['ci95']
    groups = row['groups'] or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    primary = dict(rule=PRIMARY_RULE, contrast='SMALL1_60C - MAPPO60C', diff=row['total_minus_ref']['diff'],
                   ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass,
                   group_rule_pass=b_pass, supported=bool(a_pass and b_pass),
                   verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED')
    rows = [row]+[D._contrast(have, reports, worlds, a, r, bootstrap, 'secondary_descriptive') for a, r in SECONDARY]
    arms = {}
    for a in have:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        if a != 'MAPPO60C':
            arms[a]['activations'] = A67.activation_table(have[a], worlds, plan_specs(D, root, a))
            arms[a]['activations_per_world'] = sum(int((have[a][w].get('a37') or {}).get('activations') or 0)
                                                   for w in worlds)/len(worlds)
    return N.seal(dict(schema='coopmwm.a70.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, selection=dict(receipt_sha256=sel['receipt_sha256'], TOP1=sel['TOP1'],
                                                           TOP2=sel['TOP2'], banks=sel['banks'],
                                                           dropped_duplicates=sel['dropped_duplicates']),
                       arms_run=list(ARMS), root=str(root), primary=primary, contrasts=rows, arms=arms,
                       multiplicity='one registered primary; every other contrast is descriptive',
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

STAGE_CAP = dict(identity=IDENT_CAP, confirm=CAP_SIM_S-IDENT_CAP)
STAGE_PREFIX = dict(identity='a70i_', confirm='a70c_')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('prepare', 'dry', 'identity', 'identity-compare', 'confirm', 'analyse',
                                       'status'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--a68-root')
    p.add_argument('--a66-root')
    p.add_argument('--a67-root')
    p.add_argument('--a42-root')
    p.add_argument('--stage', choices=('i', 'c'), default='c')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    N = _mods()[0]
    root = Path(a.final_root)
    for other in (a.a68_root, a.a66_root, a.a67_root, a.a42_root):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A70 final root is a new root')
    which = dict(identity='i', confirm='c', prepare='c', analyse='c').get(a.command, a.stage)
    if a.command == 'identity-compare':
        which = 'i'
    D, panel = setup(which)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTINGS[which]),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(stage):
        if stage == 'identity':
            return ident_waves(D, panel, root, a.a66_root)
        return confirm_waves(D, panel, root, a.a66_root)

    def brief(ws):
        return {n: dict(n=len(s), first=s[0]['job_id'], last=s[-1]['job_id'], runner_arm=s[0]['arm'],
                        seeds=[s[0]['world']['seed'], s[-1]['world']['seed']], cell=s[0]['world']['cell_key'],
                        checkpoint={k: v for k, v in s[0]['checkpoint'].items() if k != 'label'})
                for n, _, s in ws}
    if a.command == 'prepare':
        v = prepare(D, root, a.a68_root)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], TOP1=v['TOP1'], TOP2=v['TOP2'],
                              dropped=v['dropped_duplicates'], banks=v['banks'],
                              ranking=[(r['family'], round(r['ci95'][1], 1), r['duplicate_of']) for r in v['ranking']])))
    elif a.command == 'dry':
        stage = dict(i='identity', c='confirm')[which]
        ws = waves_of(stage)
        print(json.dumps(dict(stage=stage, panel_sha256=panel['panel_sha256'], waves=brief(ws),
                              sim_s=sum(len(s)*N.HORIZON_S for _, _, s in ws), stage_cap=STAGE_CAP[stage],
                              charged=D.charged(root), cap=CAP_SIM_S), indent=1))
    elif a.command in STAGE_CAP:
        import fcntl
        if a.command != 'identity' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: nothing is run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(a.command)
            pending = sum(len(s)-N.published(root, w, s, D.C37.check_a37)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            used = D.charged(root, STAGE_PREFIX[a.command])
            if used+need > STAGE_CAP[a.command] or D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{a.command}: {used} + {need} > stage cap {STAGE_CAP[a.command]} (or phase cap)')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay: a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(D, panel, root, a.a66_root, a.a67_root, a.a42_root)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['wave'], x['world_id'], x['abs_diff'],
                                                            x['safety_equal'], x['activations'],
                                                            x['vehicle_log_equal']) for x in v['worlds']])))
    elif a.command == 'analyse':
        v = analyse(D, panel, root, a.a66_root)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        stage = dict(i='identity', c='confirm')[which]
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root), stage=stage)
        try:
            for n, _, s in waves_of(stage):
                out[n] = f'{N.published(root, n, s, D.C37.check_a37)[2]}/{len(s)}'
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
