"""AMENDMENT 61 (S-COOPL2SPLIT60; registered before any outcome of this test): lane-2 split of the A57 PART_L2 arm.

PART_L2A / PART_L2B = A57 PART_L2 (frozen A42 BIG_BANK, CELL_BANK executor, full-group activation eligibility, a57
lane filter lane 2) plus the a61 within-lane pick (tools/coopmwm/a61_rankpick.py): A = only the lane-2 member with the
lower planned rank, B = only the other. Panel 1 (setting a42c_D2800_p60, 96 worlds 998600-998695). Reused, read-only:
A42 U0 + JOINT (A42 root), A57 PART_L2 (A57 root), after an identity check of this build.

Driver = ops/a42_bigp60.py module copy with A61 names (as A57): own root /root/a61/root, phase coopl2split_a61 cap
245,000, job ids a61*, stop file STOP_A61. No automatic replay.

Commands (python ops/a61_coopl2split.py <cmd> --final-root <root>):
  prepare / dry / budget / status
  identity          1 episode: PART_L2 (no pick) on world index 0 (wz3to2_998600) with this build
  identity-compare  -> IDENTITY_A61.json vs the stored A57 PART_L2 record (J diff 0, safety, activations, a57 receipt)
  smoke             world index 0 of PART_L2A / PART_L2B (first chunk of the main waves) ; smoke-check (outcome-blind)
  run               the 2 x 96 waves
  analyse           -> COOPL2SPLIT.json + per-world CSV in --out
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a50_fullp60 as A50  # noqa: E402
import a57_coopsyn as A57  # noqa: E402   (A57 helpers + the A57 view for the reused PART_L2 records)

AMENDMENT = '61'
CELL, SETTING = A57.CELL, A57.SETTING
LANE = 2
PICKS = ('A', 'B')
PICK_ARMS = ('PART_L2A', 'PART_L2B')
N_WORLDS = 96
CAP_SIM_S = 245_000
A57_ROOT = '/root/a57/root'
BIG_BANK = A57.BIG_BANK
BOOTSTRAP = A57.BOOTSTRAP
AUTHORITY = ('MAIN AMENDMENT 61 (2026-11-11): S-COOPL2SPLIT60, PART_L2A / PART_L2B = A57 PART_L2 with one lane-2 member '
             'commanded (planned within-lane rank), panel 1, reuse A57 U0 / PART_L2 / A42 JOINT; cap 245,000')
PRIMARY_RULE = ('I2 = (PART_L2 - U0) - (PART_L2A - U0) - (PART_L2B - U0) per world (total delay); paired percentile '
                'bootstrap 10,000, seed 20261104; WITHIN-LANE SYNERGY if CI95 upper < 0, SUB-ADDITIVE if CI95 lower > '
                '0, else NOT RESOLVED')
OVERRIDES = dict(
    A42_PHASE='coopl2split_a61', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A61',
    HOST_BASE=os.environ.get('A61_HOST_BASE', '/root/a61'),
    BANK_DIR='a61_banks', IDENT_WAVE='A61_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A61', JOB_PREFIX='a61',
    VERIFY_TAG='a61v', CONFIRM_TAG='a61c', IDENT_TAG='a61i', VERIFY_WAVE_PREFIX='A61V_', CONFIRM_WAVE_PREFIX='A61C_',
    CANDS_BANK_FILE=f'A61_UNUSED_CANDS_{CELL}.json', BIG_BANK_FILE=BIG_BANK,
    ADMISSION_FILE=f'ADMISSION_A61_UNUSED_{CELL}.json', RESULTS_FILE='COOPL2SPLIT.json', LOG_FILE='A61_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a61', RUNNER_RANGES_ATTR='A42_RANGES',
    RESULT_LABEL='amendment 61; lane-2 split of the A57 PART_L2 arm; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a61_coopl2split_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()
IDENT_WAVE = 'A61_IDENT_PART_L2'
IDENTITY_FILE = 'IDENTITY_A61.json'
SMOKE_FILE = 'SMOKE_A61.json'


def wave(arm):
    return f'{D.CONFIRM_WAVE_PREFIX}{arm}'


def prepare(final_root):
    copy = D.bank_path(final_root, BIG_BANK)
    adm = A57.a42_admission()
    if not copy.is_file() or N.file_sha(copy) != adm['big_bank']['file_sha256']:
        raise N.Blocked(f'{copy}: missing or not the admitted A42 BIG_BANK')
    return dict(path=str(copy), file_sha256=N.file_sha(copy))


def _entry(final_root, check_files=True):
    if check_files:
        prepare(final_root)
    return D._bank_entry(final_root, BIG_BANK, check_files=check_files) or dict(path='', file_sha256='')


def lane2_specs(panel, final_root, *, tag, pick=None, check_files=True):
    from tools.coopmwm import a57_partlane as P
    from tools.coopmwm import a61_rankpick as K
    out = []
    for s in D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, check_files), tag=tag):
        ck = deepcopy(s['checkpoint'])
        ck.update(a57_member_lane=LANE, a57_sha256=P.a57_digest())
        if pick is not None:
            ck.update(a61_member_pick=pick, a61_sha256=K.a61_digest())
        out.append(dict(s, checkpoint=ck))
    return out


def pick_waves(panel, final_root, *, check_files=True, first_only=False):
    out = []
    for pick, arm in zip(PICKS, PICK_ARMS):
        sp = lane2_specs(panel, final_root, tag=f'{D.CONFIRM_TAG}_l2{pick.lower()}', pick=pick,
                         check_files=check_files)
        out.append((wave(arm), 'episode', sp[:1] if first_only else sp))
    return out


def ident_waves(panel, final_root, *, check_files=True):
    s = lane2_specs(panel, final_root, tag=f'{D.IDENT_TAG}_part_l2', check_files=check_files)[0]
    return [(IDENT_WAVE, 'episode', [s])]


def a57_part_l2(panel):
    specs = A57.part_specs(LANE, panel, A57_ROOT, check_files=True)
    return A57.D.C37._load(A57_ROOT, A57.wave(f'PART_L{LANE}'), specs)


def identity_compare(final_root, panel):
    final_root = Path(final_root)
    have, _ = a57_part_l2(panel)
    (name, _, specs), = ident_waves(panel, final_root)
    got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
    if cov != 1:
        raise N.Blocked(f'{name}: {cov}/1 published')
    r = got[0]
    a = have[r['world_id']]
    strip = lambda p: {k: v for k, v in p.items() if k != 'schema'}  # noqa: E731
    row = dict(world_id=r['world_id'], j_a61=r['J_veh_s'], j_a57=a['J_veh_s'], abs_diff=abs(r['J_veh_s']-a['J_veh_s']),
               safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
               activations=[r['a37'].get('activations'), a['a37'].get('activations')],
               acts_equal=A57._acts(r) == A57._acts(a),
               a57_receipt_equal=strip(r['a37']['a57_part']) == strip(a['a37']['a57_part']),
               no_pick='a61_pick' not in r['a37'])
    ok = (row['abs_diff'] == 0.0 and row['safety_equal'] and row['same_world'] and row['acts_equal']
          and row['a57_receipt_equal'] and row['no_pick'])
    return N.seal(dict(schema='coopmwm.a61.identity.v1', amendment=AMENDMENT,
                       rule='A61 build, no pick, reproduces the stored A57 PART_L2 record exactly (J, safety, '
                            'activations, a57 receipt)', verdict='PASS' if ok else 'FAILED', world=row,
                       consequence='PASS: stored U0 / PART_L2 / JOINT reused; FAILED: stop'), 'receipt_sha256')


def _passed(final_root, name):
    p = Path(final_root)/name
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


def smoke_check(final_root, panel):
    from tools.coopmwm import a61_rankpick as K
    rows, ok = [], True
    for (name, _, specs), pick in zip(pick_waves(panel, final_root, first_only=True), PICKS):
        got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
        if cov != 1:
            raise N.Blocked(f'{name}: smoke episode not published')
        k = (got[0].get('a37') or {}).get('a61_pick') or {}
        good = (k.get('pick') == pick and k.get('member_lane') == LANE and k.get('a61_sha256') == K.a61_digest()
                and all(set(a['picked']) <= set(a['on_lane']) for a in k.get('acts', ())))
        ok = ok and good
        rows.append(dict(wave=name, receipt_ok=good, activations=k.get('activations'),
                         two_lane_members=k.get('activations_two_lane_members'),
                         fewer_than_two=k.get('activations_fewer_than_two'),
                         no_picked=k.get('activations_no_picked_member'), picked=k.get('members_picked_total'),
                         kept_accel=k.get('kept_accel'), dropped_accel=k.get('dropped_accel')))
    return N.seal(dict(schema='coopmwm.a61.smoke.v1', verdict='PASS' if ok else 'FAILED', rows=rows,
                       rule='record valid + a61 receipt of its own pick + registered digest (no outcome read)'),
                  'receipt_sha256')


def _reports(roots, have, worlds):
    from a32_ablation import report_of
    out = {}
    for a in have:
        out[a] = {}
        for w in worlds:
            rec = have[a][w]
            d = Path(roots[a])/'evaluation'/rec.get('work_dir', rec['job_id'])
            if str(roots[a]) in (A57.A42_ROOT, A57_ROOT) and not (d/'vehicles_report.json').is_file():
                raise N.Blocked(f'{d}: no cached report (reused roots are read-only here)')
            out[a][w] = report_of(roots[a], rec)
            if not out[a][w]['ok']:
                raise N.Blocked(f'{rec["job_id"]}: invalid vehicle-log report')
    return out


def i2(l2, a, b, u0):
    import numpy as np
    u = np.asarray(u0, float)
    return (np.asarray(l2, float)-u) - (np.asarray(a, float)-u) - (np.asarray(b, float)-u)


def verdict_of(ci):
    return 'WITHIN-LANE SYNERGY' if ci[1] < 0 else 'SUB-ADDITIVE' if ci[0] > 0 else 'NOT RESOLVED'


def analyse(final_root, panel):
    import numpy as np
    from tools.coopmwm import a61_rankpick as K
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != N_WORLDS:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    if not _passed(final_root, IDENTITY_FILE):
        raise N.Blocked('identity not PASS')
    have, sources = A57.a42_records(panel)
    roots = {a: A57.A42_ROOT for a in have}
    have['PART_L2'], sources['PART_L2'] = a57_part_l2(panel)
    roots['PART_L2'] = A57_ROOT
    for (name, _, specs), pick, arm in zip(pick_waves(panel, final_root), PICKS, PICK_ARMS):
        have[arm], sources[arm] = D.C37._load(final_root, name, specs)
        roots[arm] = str(final_root)
        for w in worlds:
            k = (have[arm][w].get('a37') or {}).get('a61_pick') or {}
            if k.get('pick') != pick or k.get('a61_sha256') != K.a61_digest():
                raise N.Blocked(f'{arm} {w}: a61 receipt missing / wrong pick / digest')
    reports = _reports(roots, have, worlds)
    arms = ('U0', 'JOINT', 'PART_L2')+PICK_ARMS
    J = {a: np.array([have[a][w]['J_veh_s'] for w in worlds], float) for a in arms}
    HB = {a: np.array([have[a][w]['safety']['hard_brakes'] for w in worlds], float) for a in arms}
    four = ('U0', 'PART_L2')+PICK_ARMS
    CV, n_paired = A57._five_way_cvar90(reports, worlds, four)
    CV = {a: np.array(v, float) for a, v in CV.items()}
    I2v = i2(J['PART_L2'], J['PART_L2A'], J['PART_L2B'], J['U0'])
    prim = A57._mean_ci(I2v, bootstrap)
    prim.update(rule=PRIMARY_RULE, verdict=verdict_of(prim['ci95']),
                convention='I2 < 0: both lane-2 members together save more than the two single-member arms add up to')
    means = {a: float(J[a].mean()) for a in PICK_ARMS}
    better = min(PICK_ARMS, key=lambda a: means[a])
    sec = dict(
        pick_minus_u0={a: N._bank_compare(have, reports, worlds, a, 'U0', bootstrap) for a in PICK_ARMS},
        part_l2_minus_u0=N._bank_compare(have, reports, worlds, 'PART_L2', 'U0', bootstrap),
        part_l2_minus_better_single=dict(better_single=better, note='better single arm = lower panel-mean total '
                                         'delay, named after the fact', **N.fe.paired_stats(J['PART_L2'], J[better],
                                                                                              bootstrap)),
        i2_cvar90_s=dict(A57._mean_ci(i2(CV['PART_L2'], CV['PART_L2A'], CV['PART_L2B'], CV['U0']), bootstrap),
                         note='per world CVaR90 over vehicles present in U0, PART_L2, PART_L2A, PART_L2B',
                         paired_vehicles_per_world=dict(min=min(n_paired), max=max(n_paired))),
        i2_hard_brake_veh_s=A57._mean_ci(i2(HB['PART_L2'], HB['PART_L2A'], HB['PART_L2B'], HB['U0']), bootstrap),
        sum_of_singles_minus_u0=A57._mean_ci((J['PART_L2A']-J['U0'])+(J['PART_L2B']-J['U0']), bootstrap))
    arm_rows = {}
    for a in arms:
        row = dict(mean_J_veh_s=float(J[a].mean()),
                   safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        recs = [(have[a][w].get('a37') or {}) for w in worlds]
        if a != 'U0':
            row.update(activations=sum(int(r.get('activations') or 0) for r in recs),
                       members_planned_total=sum(len(x['members']) for r in recs for x in r.get('acts') or []))
        if a in PICK_ARMS:
            ks = [r['a61_pick'] for r in recs]
            for f in ('activations_two_lane_members', 'activations_fewer_than_two', 'activations_no_picked_member',
                      'members_picked_total', 'vehicles_with_accel_command', 'kept_accel', 'dropped_accel'):
                row[f] = sum(int(k[f]) for k in ks)
        if a == 'PART_L2':
            row.update(members_on_lane2_total=sum(int(r['a57_part']['members_on_lane_total']) for r in recs))
        arm_rows[a] = row
    per_world = [dict(world_id=w, seed=seed[w], **{f'J_{a}': float(J[a][i]) for a in arms}, I2=float(I2v[i]),
                      **{f'cvar90_{a}': float(CV[a][i]) for a in four},
                      **{f'hard_brakes_{a}': int(HB[a][i]) for a in arms},
                      **{f'acts_{a}': int((have[a][w].get('a37') or {}).get('activations') or 0) for a in arms[1:]},
                      **{f'two_lane2_{a}': int(have[a][w]['a37']['a61_pick']['activations_two_lane_members'])
                         for a in PICK_ARMS},
                      **{f'collisions_{a}': int(have[a][w]['safety']['collisions']) for a in arms},
                      **{f'teleports_{a}': int(have[a][w]['safety']['teleports']) for a in arms})
                 for i, w in enumerate(worlds)]
    return N.seal(dict(schema='coopmwm.a61.coopl2split.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       setting=SETTING, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       convention='diff = treatment - reference; negative = less delay', primary=prim, secondary=sec,
                       arms=arm_rows, per_world=per_world, a61_sha256=K.a61_digest(),
                       reused=dict(U0=f'{A57.A42_ROOT} WAVE_A42C_U0', JOINT=f'{A57.A42_ROOT} WAVE_A42C_BIG_BANK',
                                   PART_L2=f'{A57_ROOT} WAVE_A57C_PART_L2'),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       charged_sim_s=D.charged(final_root), root=str(final_root)), 'analysis_sha256')


def budget_plan():
    ident = N.HORIZON_S
    run = len(PICK_ARMS)*N_WORLDS*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, identity=ident, run=run, smoke_included_in_run=2*N.HORIZON_S,
                planned=ident+run, margin=CAP_SIM_S-ident-run)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('prepare', 'dry', 'identity', 'identity-compare', 'smoke', 'smoke-check', 'run',
                                       'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk', type=int, default=8)
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    panel = D.setup(SETTING)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root)))
    elif a.command == 'dry':
        iw = ident_waves(panel, root)
        pw = pick_waves(panel, root)
        ref = A57.part_specs(LANE, panel, A57_ROOT, check_files=True)

        def norm(s, drop=()):
            s = dict(s, checkpoint={k: v for k, v in s['checkpoint'].items() if k not in drop})
            s.pop('job_id', None)
            t = json.dumps(s, sort_keys=True, default=str)
            t = t.replace(f'{D.HOST_ROOT}/{D.BANK_DIR}', '<bank>').replace(f'{A57_ROOT}/{A57.D.BANK_DIR}', '<bank>')
            return t.replace(D.HOST_ROOT, '<root>').replace(A57_ROOT, '<root>')
        print(json.dumps(dict(
            budget=budget_plan(), charged=D.charged(root), identity_world=iw[0][2][0]['world_id'],
            identity_spec_equals_a57_part_l2=norm(iw[0][2][0]) == norm(ref[0]),
            pick_specs_equal_part_l2_except_pick=all(norm(s, ('a61_member_pick', 'a61_sha256')) == norm(r)
                                                     for _, _, sp in pw for s, r in zip(sp, ref)),
            picks=[(n, len(sp), sp[0]['job_id'], sp[0]['checkpoint']['a61_member_pick'],
                    sp[0]['checkpoint']['a61_sha256'][:12], [sp[0]['world']['seed'], sp[-1]['world']['seed']])
                   for n, _, sp in pw],
            a37_sha256=iw[0][2][0]['checkpoint']['a37_sha256'], a57_sha256=iw[0][2][0]['checkpoint']['a57_sha256'],
            worlds=len(panel['worlds'])), indent=1, default=str))
    elif a.command in ('identity', 'smoke', 'run'):
        import fcntl
        if a.command != 'identity' and not _passed(root, IDENTITY_FILE):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS')
        if a.command == 'run' and not _passed(root, SMOKE_FILE):
            raise N.Blocked(f'{SMOKE_FILE} is not PASS')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = (ident_waves(panel, root) if a.command == 'identity' else
                     pick_waves(panel, root, first_only=a.command == 'smoke'))
            done = sum(N.published(root, n, s, D.C37.check_a37)[2] for n, _, s in waves)
            need = (sum(len(s) for _, _, s in waves)-done)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], **{k: v['world'][k] for k in ('world_id', 'abs_diff', 'safety_equal',
                                                                                 'acts_equal', 'a57_receipt_equal',
                                                                                 'activations')})))
    elif a.command == 'smoke-check':
        v = smoke_check(root, panel)
        N.publish(root/SMOKE_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], rows=v['rows'])))
    elif a.command == 'analyse':
        v = analyse(root, panel)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'COOPL2SPLIT.json', v, replace=True)
        rows = v['per_world']
        with open(out/'COOPL2SPLIT_PER_WORLD.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], verdict=v['primary']['verdict'],
                              I2=v['primary']['mean'], ci95=v['primary']['ci95'])))
    else:
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=_passed(root, IDENTITY_FILE))
        for name, _, specs in ident_waves(panel, root)+pick_waves(panel, root):
            out[name] = f'{N.published(root, name, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
