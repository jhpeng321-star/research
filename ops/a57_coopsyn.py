"""AMENDMENT 57 (S-COOPSYN60; registered before any outcome): joint-versus-separate cooperation test at D2800_p60.

Arms PART_L0 / PART_L1 / PART_L2 = the A37 CELL_BANK executor on the frozen admitted A42 BIG_BANK (byte copy of
/root/a42v/root/a42_banks/A42_BIG_BANK_D2800_p60.json, sha = the A42 admission receipt's big_bank), same trigger, same
activation eligibility on the FULL planned group; only the planned members on SUMO lane k receive plan commands
(tools/coopmwm/a57_partlane.py, switched on per spec by checkpoint a57_member_lane / a57_sha256; absent = the A42
path unchanged). Every other planned member stays under frozen MAPPO U0. 96 A42 confirmation worlds (setting
a42c_D2800_p60). U0 and JOINT (= A42 BIG_BANK) are the stored A42 per-world records (A42 root, read-only).

This driver IS ops/a42_bigp60.py (a private module copy, as A50 / A52 / A56) with the A57 names: own final root
/root/a57/root, ledger phase coopsyn_a57 cap 390,000 sim-s, job ids a57*, waves A57_IDENT_BIG_BANK / A57C_PART_L<k>,
stop file STOP_A57. No automatic replay (a failed dispatch raises and is reported).

Commands (python ops/a57_coopsyn.py <cmd> --final-root <root>):
  prepare           zero physics: check the a57_banks BIG_BANK byte copy (vs the A42 root file and admission receipt)
  dry               spec summary + budget (zero physics)
  identity          1 episode: BIG_BANK, filter OFF, on the identity world (the first panel world whose stored A42
                    BIG_BANK record has >= 1 activation; rule fixed before any A57 outcome)
  identity-compare  -> IDENTITY_A57.json: J_veh_s diff 0, identical safety, identical activation receipts
  smoke             3 episodes: world index 0 of each PART wave (refused unless identity PASS); published as the
                    first chunk of the main waves (not run twice)
  run               the 3 x 96 PART waves (refused unless identity PASS and the smoke chunk is published)
  analyse           registered primary + secondaries -> COOPSYN.json (+ per-world CSV) in --out
  status / budget
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
import a50_fullp60 as A50  # noqa: E402   (only for its _clone helper)

AMENDMENT = '57'
CELL = 'D2800_p60'
SETTING = 'a42c_D2800_p60'
LANES = (0, 1, 2)
PART_ARMS = tuple(f'PART_L{k}' for k in LANES)
N_WORLDS = 96
CAP_SIM_S = 390_000
A42_ROOT = '/root/a42v/root'
BIG_BANK = f'A42_BIG_BANK_{CELL}.json'
A42_ADMISSION = 'ADMISSION_D2800_p60.json'
A42_ADMISSION_RECEIPT = '5fa2a7015fb4ae9a95e1d17456e5caacb6114405cc197831d2c3d674f1b332d5'
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
AUTHORITY = ('MAIN AMENDMENT 57 (owner 2026-11-11: "go ahead and run the cooperation test"): S-COOPSYN60, PART_L0/1/2 '
             '= frozen A42 BIG_BANK with a member-lane filter at command time, 96 A42 confirmation worlds, paired with '
             'the stored A42 U0 and BIG_BANK runs; phase coopsyn_a57 cap 390,000')
PRIMARY_RULE = ('I = (JOINT - U0) - [(PART_L0 - U0) + (PART_L1 - U0) + (PART_L2 - U0)] per world (total delay, veh-s); '
                'paired percentile bootstrap 10,000, seed 20261104; SUPPORTED (super-additive) if CI95 upper < 0, '
                'SUB-ADDITIVE if CI95 lower > 0, otherwise NOT RESOLVED')
OVERRIDES = dict(
    A42_PHASE='coopsyn_a57', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A57',
    HOST_BASE=os.environ.get('A57_HOST_BASE', '/root/a57'),
    BANK_DIR='a57_banks', IDENT_WAVE='A57_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A57', JOB_PREFIX='a57',
    VERIFY_TAG='a57v', CONFIRM_TAG='a57c', IDENT_TAG='a57i', VERIFY_WAVE_PREFIX='A57V_', CONFIRM_WAVE_PREFIX='A57C_',
    CANDS_BANK_FILE=f'A57_UNUSED_CANDS_{CELL}.json', BIG_BANK_FILE=BIG_BANK,
    ADMISSION_FILE=f'ADMISSION_A57_UNUSED_{CELL}.json', RESULTS_FILE='COOPSYN.json', LOG_FILE='A57_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a57', RUNNER_RANGES_ATTR='A42_RANGES',
    RESULT_LABEL='amendment 57; joint-versus-separate cooperation test at 60% CAVs; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a57_coopsyn_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()                       # A42 driver code, A57 names (own root / phase / cap / ids / stop file)
D42 = A50._clone('a57_a42_view', HERE/'a42_bigp60.py', {})   # the A42 driver itself (A42 names; read-only use)
IDENT_WAVE = 'A57_IDENT_BIG_BANK'
IDENTITY_FILE = 'IDENTITY_A57.json'


def wave(arm):
    return f'{D.CONFIRM_WAVE_PREFIX}{arm}'


# ------------------------------------------------------------------------------------------------ bank / specs

def a42_admission():
    p = Path(A42_ROOT)/A42_ADMISSION
    v = N.verify(N.read(p), 'receipt_sha256', p.name)
    if v['receipt_sha256'] != A42_ADMISSION_RECEIPT:
        raise N.Blocked('A42 admission receipt is not the registered one')
    return v


def prepare(final_root):
    copy = D.bank_path(final_root, BIG_BANK)
    orig = Path(A42_ROOT)/D42.BANK_DIR/BIG_BANK
    adm = a42_admission()
    if not copy.is_file() or N.file_sha(copy) != N.file_sha(orig) or N.file_sha(copy) != adm['big_bank']['file_sha256']:
        raise N.Blocked(f'{copy}: missing or not a byte copy of the admitted A42 BIG_BANK')
    return dict(path=str(copy), file_sha256=N.file_sha(copy), admitted=adm['admitted'])


def _entry(final_root, check_files=True):
    if check_files:
        prepare(final_root)
    return D._bank_entry(final_root, BIG_BANK, check_files=check_files) or dict(path='', file_sha256='')


def part_specs(lane, panel, final_root, *, check_files=True):
    from tools.coopmwm import a57_partlane as P
    out = []
    for s in D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, check_files),
                          tag=f'{D.CONFIRM_TAG}_part_l{lane}'):
        ck = deepcopy(s['checkpoint'])
        ck.update(a57_member_lane=int(lane), a57_sha256=P.a57_digest())
        out.append(dict(s, checkpoint=ck))
    return out


def part_waves(panel, final_root, *, check_files=True, first_only=False):
    out = []
    for k, arm in zip(LANES, PART_ARMS):
        specs = part_specs(k, panel, final_root, check_files=check_files)
        out.append((wave(arm), 'episode', specs[:1] if first_only else specs))
    return out


def a42_records(panel):
    """{'U0': {world: rec}, 'JOINT': {world: rec}}, sources: the stored A42 confirmation waves (A42 root, read-only)."""
    have, src = {}, {}
    for arm, name in (('U0', 'U0'), ('JOINT', 'BIG_BANK')):
        specs = D42.confirm_specs(name, panel, A42_ROOT, check_files=True)
        have[arm], src[arm] = D42.C37._load(A42_ROOT, D42.confirm_wave(name), specs)
    return have, src


def identity_index(panel):
    """The first panel world (panel order) whose stored A42 BIG_BANK record has >= 1 activation."""
    have, _ = a42_records(panel)
    for i, r in enumerate(panel['worlds']):
        if int((have['JOINT'][r['world_id']].get('a37') or {}).get('activations') or 0) > 0:
            return i
    raise N.Blocked('no stored A42 BIG_BANK world with an activation')


def ident_waves(panel, final_root, *, check_files=True):
    i = identity_index(panel)
    s = D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, _entry(final_root, check_files),
                     tag=f'{D.IDENT_TAG}_big_bank')[i]
    return [(IDENT_WAVE, 'episode', [dict(s, job_id=f'{D.IDENT_TAG}_big_bank_0000')])]


def _acts(rec):
    a = rec.get('a37') or {}
    return [(x['t'], x['family'], list(x['members']), x.get('release')) for x in a.get('acts') or []]


def identity_compare(final_root, panel):
    final_root = Path(final_root)
    have, _ = a42_records(panel)
    (name, _, specs), = ident_waves(panel, final_root)
    got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
    if cov != 1:
        raise N.Blocked(f'{name}: {cov}/1 published')
    r = got[0]
    a = have['JOINT'][r['world_id']]
    row = dict(world_id=r['world_id'], seed=r['world']['seed'], j_a57=r['J_veh_s'], j_a42=a['J_veh_s'],
               abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a57=r['safety'], safety_a42=a['safety'],
               safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
               activations=[r['a37'].get('activations'), a['a37'].get('activations')],
               acts_equal=_acts(r) == _acts(a), a37_sha256_equal=r['a37_sha256_loaded'] == a['a37_sha256_loaded'],
               no_filter=('a57_part' not in r['a37']))
    ok = (row['abs_diff'] == 0.0 and row['safety_equal'] and row['same_world'] and row['acts_equal']
          and row['a37_sha256_equal'] and row['no_filter'])
    return N.seal(dict(schema='coopmwm.a57.identity.v1', amendment=AMENDMENT,
                       rule=('A57 code path, filter off, reproduces the stored A42 BIG_BANK record of the identity world '
                             'exactly: J_veh_s difference 0, identical safety counters, identical activation receipts, '
                             'same a37 digest'),
                       verdict='PASS' if ok else 'FAILED', world=row,
                       consequence='PASS: stored A42 U0 / BIG_BANK reused; FAILED: stop and report'), 'receipt_sha256')


def identity_passed(final_root):
    p = Path(final_root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


def smoke_check(final_root, panel):
    """Zero physics, outcome-blind: each smoke record is valid (check_a37), carries the a57_part receipt of its own lane
    and the registered digest, and every commanded member was planned on that lane."""
    from tools.coopmwm import a57_partlane as P
    rows, ok = [], True
    for (name, _, specs), k in zip(part_waves(panel, final_root, first_only=True), LANES):
        got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
        if cov != 1:
            raise N.Blocked(f'{name}: smoke episode not published')
        part = (got[0].get('a37') or {}).get('a57_part') or {}
        good = (part.get('member_lane') == k and part.get('a57_sha256') == P.a57_digest()
                and all(set(a['on_lane']) <= set(a['members']) for a in part.get('acts', ())))
        ok = ok and good
        rows.append(dict(wave=name, world_id=got[0]['world_id'], member_lane=part.get('member_lane'), receipt_ok=good,
                         activations=part.get('activations'), members_on_lane=part.get('members_on_lane_total'),
                         members_planned=part.get('members_planned_total'),
                         no_member_on_lane=part.get('activations_no_member_on_lane'),
                         kept_accel=part.get('kept_accel'), dropped_accel=part.get('dropped_accel')))
    return N.seal(dict(schema='coopmwm.a57.smoke.v1', verdict='PASS' if ok else 'FAILED', rows=rows,
                       rule='record valid + a57_part receipt of its own lane + registered digest (no outcome read)'),
                  'receipt_sha256')


# ------------------------------------------------------------------------------------------------ analysis

def interaction(joint, parts, u0):
    """Per-world I = (J - U0) - sum_k (P_k - U0) (pure; unit-tested)."""
    import numpy as np
    j, u = np.asarray(joint, float), np.asarray(u0, float)
    return (j-u) - sum(np.asarray(p, float)-u for p in parts)


def verdict_of(ci):
    return 'SUPPORTED (super-additive)' if ci[1] < 0 else 'SUB-ADDITIVE' if ci[0] > 0 else 'NOT RESOLVED'


def _mean_ci(x, bootstrap):
    import numpy as np
    x = np.asarray(x, float)
    s = N.fe.paired_stats(x, np.zeros_like(x), bootstrap, pct=False)
    return dict(mean=s['diff'], ci95=s['ci95'], p_two_sided=s['p_two_sided'], worlds_negative=s['worlds_lower'],
                worlds_positive=s['worlds_higher'])


def _reports_readonly(root, have, worlds):
    """Vehicle-log reports (as A56); for the A42 root every report cache must already exist (nothing written there)."""
    from a32_ablation import report_of
    out = {}
    for a in have:
        out[a] = {}
        for w in worlds:
            rec = have[a][w]
            d = Path(root[a])/'evaluation'/rec.get('work_dir', rec['job_id'])
            if str(root[a]) == A42_ROOT and not (d/'vehicles_report.json').is_file():
                raise N.Blocked(f'{d}: no cached report (A42 root is read-only here)')
            out[a][w] = report_of(root[a], rec)
            if not out[a][w]['ok']:
                raise N.Blocked(f'{rec["job_id"]}: invalid vehicle-log report')
    return out


def _five_way_cvar90(reports, worlds, arms):
    """Per world, per arm: CVaR90 of per-vehicle delay over the vehicles present in ALL arms (paired by id)."""
    out = {a: [] for a in arms}
    n = []
    for w in worlds:
        ids = sorted(set.intersection(*[set(reports[a][w]['per_vehicle']) for a in arms]))
        if not ids:
            raise N.Blocked(f'{w}: no vehicle present in every arm')
        for a in arms:
            out[a].append(N.cvar90([reports[a][w]['per_vehicle'][v][0] for v in ids]))
        n.append(len(ids))
    return out, n


def _arm_counts(arm, recs, worlds):
    rows = [(recs[w].get('a37') or {}) for w in worlds]
    acts = sum(int(r.get('activations') or 0) for r in rows)
    planned = sum(len(x['members']) for r in rows for x in (r.get('acts') or []))
    out = dict(activations=acts, worlds_with_activation=sum(1 for r in rows if r.get('activations')),
               members_planned_total=planned)
    if arm.startswith('PART_'):
        parts = [r['a57_part'] for r in rows]
        out.update(members_commanded_total=sum(p['members_on_lane_total'] for p in parts),
                   activations_no_member_on_lane=sum(p['activations_no_member_on_lane'] for p in parts),
                   vehicles_with_accel_command=sum(p['vehicles_with_accel_command'] for p in parts),
                   kept_accel_steps=sum(p['kept_accel'] for p in parts),
                   dropped_accel_steps=sum(p['dropped_accel'] for p in parts))
    elif arm == 'JOINT':
        out.update(members_commanded_total=planned, activations_no_member_on_lane=None)
    return out


def analyse(final_root, panel):
    import numpy as np
    from tools.coopmwm import a57_partlane as P
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != N_WORLDS:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    if not identity_passed(final_root):
        raise N.Blocked('identity not PASS')
    have, sources = a42_records(panel)
    roots = {a: A42_ROOT for a in have}
    for (name, _, specs), arm, k in zip(part_waves(panel, final_root), PART_ARMS, LANES):
        have[arm], sources[arm] = D.C37._load(final_root, name, specs)
        roots[arm] = str(final_root)
        for w in worlds:
            p = (have[arm][w].get('a37') or {}).get('a57_part') or {}
            if p.get('member_lane') != k or p.get('a57_sha256') != P.a57_digest():
                raise N.Blocked(f'{arm} {w}: a57_part receipt missing / wrong lane / wrong digest')
    for w in worlds:
        if 'a57_part' in (have['JOINT'][w].get('a37') or {}):
            raise N.Blocked('JOINT record carries a filter receipt')
    reports = _reports_readonly(roots, have, worlds)
    arms = ('U0', 'JOINT')+PART_ARMS
    J = {a: np.array([have[a][w]['J_veh_s'] for w in worlds], float) for a in arms}
    HB = {a: np.array([have[a][w]['safety']['hard_brakes'] for w in worlds], float) for a in arms}
    CV, n_paired = _five_way_cvar90(reports, worlds, arms)
    CV = {a: np.array(v, float) for a, v in CV.items()}
    I_total = interaction(J['JOINT'], [J[a] for a in PART_ARMS], J['U0'])
    prim = _mean_ci(I_total, bootstrap)
    prim.update(rule=PRIMARY_RULE, verdict=verdict_of(prim['ci95']),
                convention='I < 0: the joint plan saves more delay than its per-lane parts add up to')
    means = {a: float(J[a].mean()) for a in PART_ARMS}
    best = min(PART_ARMS, key=lambda a: means[a])
    sec = dict(
        joint_minus_best_part=dict(best_part=best, note='best part = lowest panel-mean total delay, named after the '
                                   'fact (disclosed)', **N.fe.paired_stats(J['JOINT'], J[best], bootstrap)),
        part_minus_u0={a: N._bank_compare(have, reports, worlds, a, 'U0', bootstrap) for a in PART_ARMS},
        joint_minus_u0=N._bank_compare(have, reports, worlds, 'JOINT', 'U0', bootstrap),
        interaction_cvar90_s=dict(_mean_ci(interaction(CV['JOINT'], [CV[a] for a in PART_ARMS], CV['U0']), bootstrap),
                                  note='per world CVaR90 (worst 10 %) of per-vehicle delay over vehicles present in '
                                       'all five arms', paired_vehicles_per_world=dict(min=min(n_paired),
                                                                                       max=max(n_paired))),
        interaction_hard_brake_veh_s=_mean_ci(interaction(HB['JOINT'], [HB[a] for a in PART_ARMS], HB['U0']),
                                              bootstrap),
        sum_of_parts_minus_u0=_mean_ci(sum(J[a]-J['U0'] for a in PART_ARMS), bootstrap),
        joint_minus_u0_total=_mean_ci(J['JOINT']-J['U0'], bootstrap))
    arm_rows = {a: dict(mean_J_veh_s=float(J[a].mean()),
                        safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                        **({} if a == 'U0' else _arm_counts(a, have[a], worlds))) for a in arms}
    per_world = [dict(world_id=w, seed=seed[w], **{f'J_{a}': float(have[a][w]['J_veh_s']) for a in arms},
                      I_total=float(I_total[i]), **{f'cvar90_{a}': float(CV[a][i]) for a in arms},
                      **{f'hard_brakes_{a}': int(HB[a][i]) for a in arms},
                      **{f'acts_{a}': int((have[a][w].get('a37') or {}).get('activations') or 0) for a in arms[1:]},
                      **{f'commanded_{a}': int(have[a][w]['a37']['a57_part']['members_on_lane_total'])
                         for a in PART_ARMS},
                      **{f'collisions_{a}': int(have[a][w]['safety']['collisions']) for a in arms},
                      **{f'teleports_{a}': int(have[a][w]['safety']['teleports']) for a in arms})
                 for i, w in enumerate(worlds)]
    return N.seal(dict(schema='coopmwm.a57.coopsyn.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       setting=SETTING, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       convention='diff = treatment - reference; negative = less delay', primary=prim, secondary=sec,
                       arms=arm_rows, per_world=per_world, a57_sha256=P.a57_digest(),
                       a42_admission_receipt=A42_ADMISSION_RECEIPT,
                       reused=dict(U0=f'{A42_ROOT} WAVE_A42C_U0', JOINT=f'{A42_ROOT} WAVE_A42C_BIG_BANK'),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       charged_sim_s=D.charged(final_root), root=str(final_root)), 'analysis_sha256')


def write_table(v, path):
    rows = v['per_world']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan():
    ident = N.HORIZON_S
    run = len(PART_ARMS)*N_WORLDS*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, identity=ident, smoke_included_in_run=len(PART_ARMS)*N.HORIZON_S,
                run=run, planned=ident+run, margin=CAP_SIM_S-ident-run)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('prepare', 'dry', 'identity', 'identity-compare', 'smoke', 'smoke-check', 'run',
                                       'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--chunk', type=int, default=16)
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
        pw = part_waves(panel, root)
        ref = D42.confirm_specs('BIG_BANK', panel, A42_ROOT, check_files=True)
        i = identity_index(panel)
        mine = iw[0][2][0]

        def norm(s):
            s = dict(s)
            s.pop('job_id', None)
            t = json.dumps(s, sort_keys=True, default=str)
            t = t.replace(f'{D.HOST_ROOT}/{D.BANK_DIR}', '<bank>').replace(f'{A42_ROOT}/{D42.BANK_DIR}', '<bank>')
            return t.replace(str(root), '<root>').replace(D.HOST_ROOT, '<root>').replace(A42_ROOT, '<root>')
        print(json.dumps(dict(
            budget=budget_plan(), charged=D.charged(root), identity_index=i,
            identity_world=mine['world_id'], identity_spec_equals_a42_except_job_id_and_bank_path=norm(mine) == norm(ref[i]),
            identity_bank=[mine['checkpoint']['a37_bank'], ref[i]['checkpoint']['a37_bank']],
            parts=[(n, len(sp), sp[0]['job_id'], sp[0]['checkpoint']['a57_member_lane'],
                    sp[0]['checkpoint']['a57_sha256'][:12], [sp[0]['world']['seed'], sp[-1]['world']['seed']])
                   for n, _, sp in pw],
            part_specs_equal_joint_except_filter=all(
                norm(dict(s, checkpoint={k: v for k, v in s['checkpoint'].items()
                                         if k not in ('a57_member_lane', 'a57_sha256')})) == norm(r)
                for _, _, sp in pw for s, r in zip(sp, ref)),
            a37_sha256=mine['checkpoint']['a37_sha256'], worlds=len(panel['worlds'])), indent=1, default=str))
    elif a.command in ('identity', 'smoke', 'run'):
        import fcntl
        if a.command != 'identity' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: the PART arms are not run')
        if a.command == 'run':
            sp = root/'SMOKE_A57.json'
            if not sp.is_file() or N.verify(N.read(sp), 'receipt_sha256', sp.name)['verdict'] != 'PASS':
                raise N.Blocked('SMOKE_A57.json is not PASS')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = (ident_waves(panel, root) if a.command == 'identity' else
                     part_waves(panel, root, first_only=a.command == 'smoke'))
            done = sum(N.published(root, n, s, D.C37.check_a37)[2] for n, _, s in waves)
            need = (sum(len(s) for _, _, s in waves)-done)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay (A57 registration): a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], world=v['world']['world_id'], abs_diff=v['world']['abs_diff'],
                              safety_equal=v['world']['safety_equal'], acts_equal=v['world']['acts_equal'],
                              activations=v['world']['activations'])))
    elif a.command == 'smoke-check':
        v = smoke_check(root, panel)
        N.publish(root/'SMOKE_A57.json', v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], rows=v['rows'])))
    elif a.command == 'analyse':
        v = analyse(root, panel)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'COOPSYN.json', v, replace=True)
        write_table(v, out/'COOPSYN_PER_WORLD.csv')
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], verdict=v['primary']['verdict'],
                              I=v['primary']['mean'], ci95=v['primary']['ci95'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root))
        for name, _, specs in ident_waves(panel, root)+part_waves(panel, root):
            out[name] = f'{N.published(root, name, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
