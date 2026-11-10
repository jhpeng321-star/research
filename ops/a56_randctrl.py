"""AMENDMENT 56 (S-RANDCTRL60; registered before any SUMO): random-proposals control at D2800_p60.

12 plans drawn with the A42/A47 search's own random plan generator (ops/a56_draw.py, seed 20261111, no WM score;
pinned plan list ops/a56/RANDPLANS_D2800_p60.json) verified EXACTLY as the amendment-42 admission: the same 24
verification worlds (setting a42v_D2800_p60 = 992000-992023), runner arm A37_PLAN (one plan, start whenever feasible)
on frozen MAPPO U0 (policy_00169 d9042cc5), paired with the EXISTING A42 U0 episodes of those worlds (common random
numbers; read in place, read-only, from the A42 final root /root/a42v/root).

This driver IS ops/a42_bigp60.py (a private module copy, as A50 / A52) with the A56 names: own final root
/root/a56/root, ledger phase randctrl_a56 cap 390,000 sim-s, job ids a56*, waves A56V_* / A56_IDENT_*, stop file
STOP_A56. No automatic replay (a failed dispatch raises and is reported).

Identity (before any random plan): U0 and the admitted A42 plan rand_3a452b1348b01083 (with a byte copy of the A42
candidates bank) on the first 4 verification worlds through this code path must reproduce the stored A42 records
exactly (J_veh_s diff 0, identical safety counters): 8 episodes = 9,600 sim-s. FAILED -> stop, nothing else runs.

Commands (python ops/a56_randctrl.py <cmd> --final-root <root>):
  prepare            zero physics: a56_banks/A56_RAND_CANDS_D2800_p60.json (12 plans) + check of the copied A42 bank
  dry                spec summary + budget
  identity           8 episodes (U0 + rand_3a452b13 on 992000-992003)
  identity-compare   -> IDENTITY_A56.json (PASS / FAILED) vs the A42 records
  verify             12 plans x 24 worlds (refused unless identity PASS)
  reduce             zero physics: per-world J / safety / activations of U0 + the six A42 WM plans (A42 root,
                     read-only) + the 12 random plans; UNC admission per plan (same code as A42) ->
                     A56_REDUCE_D2800_p60.json (sealed)
  status / budget
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
import nm_finaleval as N  # noqa: E402
import a50_fullp60 as A50  # noqa: E402   (only for its _clone helper)

AMENDMENT = '56'
CELL = 'D2800_p60'
SETTING = 'a42v_D2800_p60'
PLANS_FILE = HERE/'a56'/f'RANDPLANS_{CELL}.json'
PLANS_SHA256 = '687663b078c768643d56758b42dc8113d297be1d606ecaf7994217964ec3dbaa'
N_PLANS = 12
N_WORLDS = 24
CAP_SIM_S = 390_000
IDENT_N_WORLDS = 4
IDENT_FAMILY = 'rand_3a452b1348b01083'          # admitted A42 plan used for the identity check
A42_ROOT = '/root/a42v/root'
A42_BANK = f'A42_CANDS_{CELL}.json'
A42_ADMISSION = 'ADMISSION_D2800_p60.json'
A42_ADMISSION_RECEIPT = '5fa2a7015fb4ae9a95e1d17456e5caacb6114405cc197831d2c3d674f1b332d5'
AUTHORITY = ('MAIN AMENDMENT 56 (owner 2026-11-11: "go ahead and run it"): S-RANDCTRL60, 12 random plans from the '
             'search generator (seed 20261111, no WM score) verified as the A42 admission on the same 24 worlds, '
             'paired with the stored A42 U0 runs; phase randctrl_a56 cap 390,000')
OVERRIDES = dict(
    A42_PHASE='randctrl_a56', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A56',
    HOST_BASE=os.environ.get('A56_HOST_BASE', '/root/a56'),
    VERIFY_SETTING=SETTING, CONFIRM_SETTING='a42c_D2800_p60',
    BANK_DIR='a56_banks', IDENT_WAVE='A56_IDENT_U0', AMENDMENT=AMENDMENT, LABEL='A56', JOB_PREFIX='a56',
    VERIFY_TAG='a56v', CONFIRM_TAG='a56c', IDENT_TAG='a56i', VERIFY_WAVE_PREFIX='A56V_', CONFIRM_WAVE_PREFIX='A56C_',
    CANDS_BANK_FILE=f'A56_RAND_CANDS_{CELL}.json', BIG_BANK_FILE=f'A56_UNUSED_{CELL}.json',
    ADMISSION_FILE=f'ADMISSION_A56_{CELL}.json', RESULTS_FILE=f'A56_REDUCE_{CELL}.json', LOG_FILE='A56_LOG.jsonl',
    SCHEMA_PREFIX='coopmwm.a56', RUNNER_RANGES_ATTR='A42_RANGES',
    RESULT_LABEL='amendment 56; random-proposals control at 60% CAVs; never pooled; one shot', AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a56_randctrl_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()                       # A42 driver code, A56 names (own root / phase / cap / ids / waves / stop file)
D42 = A50._clone('a56_a42_view', HERE/'a42_bigp60.py', {})   # the A42 driver itself (A42 names; read-only use)
IDENT_PLAN_WAVE = 'A56_IDENT_PLAN'
IDENTITY_FILE = 'IDENTITY_A56.json'


# ------------------------------------------------------------------------------------------------ plans / banks

def plans():
    """The pinned 12 random plans (file sha checked; strategy sha per plan re-checked)."""
    from tools.coopmwm import a37_cellbank as C
    if N.file_sha(PLANS_FILE) != PLANS_SHA256:
        raise N.Blocked(f'{PLANS_FILE.name}: not the registered plan list {PLANS_SHA256[:12]}')
    doc = json.loads(PLANS_FILE.read_text(encoding='utf-8'))
    out = doc['plans']
    if len(out) != N_PLANS or doc['cell'] != CELL or doc.get('wm_score_used') is not False:
        raise N.Blocked('plan list is not the registered 12 random plans')
    for c in out:
        if C.spec_sha(c['strategy']) != c['strategy_sha256'] or C.executor_refusal(c['strategy']) is not None:
            raise N.Blocked(f'{c["family"]}: strategy sha / executability')
    return out


def a42_admission():
    p = Path(A42_ROOT)/A42_ADMISSION
    v = N.verify(N.read(p), 'receipt_sha256', p.name)
    if v['receipt_sha256'] != A42_ADMISSION_RECEIPT:
        raise N.Blocked('A42 admission receipt is not the registered one')
    return v


def wm_top6():
    """The six WM top-ranked A42 plans (selected_rank order) as cands for D42.verify_waves."""
    rows = sorted(a42_admission()['verdicts'], key=lambda v: v['selected_rank'])
    if len(rows) != 6:
        raise N.Blocked('A42 admission does not list six plans')
    return [dict(family=v['family'], key=v['key']) for v in rows]


def prepare(final_root):
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    ps = plans()
    doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in ps], CELL)
    p = D.C37.write_once(D.bank_path(final_root, D.CANDS_BANK_FILE), D.C37.bank_text(doc))
    copy = D.bank_path(final_root, A42_BANK)
    orig = Path(A42_ROOT)/D42.BANK_DIR/A42_BANK
    if not copy.is_file() or N.file_sha(copy) != N.file_sha(orig):
        raise N.Blocked(f'{copy}: missing or not a byte copy of {orig}')
    return dict(path=str(p), file_sha256=N.file_sha(p), families=[c['family'] for c in ps],
                a42_bank_copy=dict(path=str(copy), file_sha256=N.file_sha(copy)))


# ------------------------------------------------------------------------------------------------ specs

def verify_waves(panel, final_root, *, check_files=True):
    """12 waves A56V_<family> (plan-list order), each the A37_PLAN specs on the 24 worlds; no U0 wave."""
    entry = D._bank_entry(final_root, D.CANDS_BANK_FILE, check_files=check_files) or dict(path='', file_sha256='')
    return [(f'{D.VERIFY_WAVE_PREFIX}{c["family"]}', 'episode',
             D._a37_specs(D.RUNNER_ARM['PLAN'], panel, final_root, entry, family=c['family'],
                          tag=f'{D.VERIFY_TAG}_{c["key"][:8]}'))
            for c in plans()]


def ident_waves(panel, final_root, *, check_files=True):
    u0_wave = D.verify_waves(panel, final_root, check_files=False, cands=[])[0]
    u0 = D._retag(u0_wave[2][:IDENT_N_WORLDS], f'{D.IDENT_TAG}_u0')
    entry = D._bank_entry(final_root, A42_BANK, check_files=check_files) or dict(path='', file_sha256='')
    plan = D._a37_specs(D.RUNNER_ARM['PLAN'], panel, final_root, entry, family=IDENT_FAMILY,
                        tag=f'{D.IDENT_TAG}_{IDENT_FAMILY.split("_")[1][:8]}')[:IDENT_N_WORLDS]
    return [(D.IDENT_WAVE, 'episode', u0), (IDENT_PLAN_WAVE, 'episode', plan)]


def a42_waves(panel):
    """[(wave, specs)] of the A42 verification U0 + six WM plans on the A42 root (A42 names, host paths)."""
    return [(name, specs) for name, _, specs in D42.verify_waves(panel, A42_ROOT, check_files=True, cands=wm_top6())]


def load_a42(panel):
    have, sources = {}, {}
    for name, specs in a42_waves(panel):
        arm = name[len(D42.VERIFY_WAVE_PREFIX):]
        have[arm], sources[arm] = D42.C37._load(A42_ROOT, name, specs)
    return have, sources


def _reports_readonly(root, have, worlds):
    """Vehicle-log reports; for the A42 root every report cache must already exist (nothing is written there)."""
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


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(final_root, panel):
    final_root = Path(final_root)
    a42, _ = load_a42(panel)
    rows = []
    for (name, _, specs), arm in zip(ident_waves(panel, final_root), ('U0', IDENT_FAMILY)):
        got, _, cov = N.published(final_root, name, specs, D.C37.check_a37)
        if cov != len(specs):
            raise N.Blocked(f'{name}: {cov}/{len(specs)} published')
        for r in got:
            a = a42[arm][r['world_id']]
            rows.append(dict(arm=arm, world_id=r['world_id'], j_a56=r['J_veh_s'], j_a42=a['J_veh_s'],
                             abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a56=r['safety'], safety_a42=a['safety'],
                             safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
                             activations=[r.get('a37', {}).get('activations'), a.get('a37', {}).get('activations')]))
    ok = len(rows) == 2*IDENT_N_WORLDS and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world']
                                                for x in rows)
    return N.seal(dict(schema='coopmwm.a56.identity.v1', amendment=AMENDMENT,
                       rule=('A56 path reproduces the stored A42 verification records exactly for U0 and '
                             f'{IDENT_FAMILY} on the first {IDENT_N_WORLDS} worlds: J_veh_s difference 0 and '
                             'identical safety counters'),
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: stored A42 U0 reused (no U0 rerun); FAILED: stop and report'),
                  'receipt_sha256')


def identity_passed(final_root):
    p = Path(final_root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ reduce

def reduce(final_root, panel):
    """Per-world records + the A42 UNC admission per plan (random 12; the six WM plans recomputed as a check)."""
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources = load_a42(panel)
    roots = {a: A42_ROOT for a in have}
    ps = plans()
    for (name, _, specs), c in zip(verify_waves(panel, final_root), ps):
        have[c['family']], sources[c['family']] = D.C37._load(final_root, name, specs)
        roots[c['family']] = str(final_root)
    reports = _reports_readonly(roots, have, worlds)
    adm42 = {v['family']: v for v in a42_admission()['verdicts']}
    verdicts = {}
    for fam in [x['family'] for x in wm_top6()]+[c['family'] for c in ps]:
        cmp_ = N._bank_compare(have, reports, worlds, fam, 'U0', bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have['U0'][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a37'] for w in worlds]
        v.update(family=fam, source='A42' if fam in adm42 else 'A56', groups=cmp_['groups'],
                 cvar90_vs_u0=cmp_['tail_cvar90'], total_minus_u0_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 episodes_with_activation=sum(int(a.get('activations') or 0) > 0 for a in acts),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        if fam in adm42:
            ref = adm42[fam]
            v['matches_a42_admission'] = bool(v['PASS'] == ref['PASS'] and v['per_seed'] == ref['per_seed']
                                              and v['ci95'] == ref['ci95'])
        verdicts[fam] = v
    per_world = {a: {w: dict(seed=seed[w], J_veh_s=have[a][w]['J_veh_s'], safety=have[a][w]['safety'],
                             activations=(have[a][w].get('a37') or {}).get('activations'))
                     for w in worlds} for a in have}
    return N.seal(dict(schema='coopmwm.a56.reduce.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       setting=SETTING, panel_sha256=panel['panel_sha256'], bootstrap=bootstrap, worlds=worlds,
                       plans_sha256=PLANS_SHA256, wm_top6=[x['family'] for x in wm_top6()],
                       random12=[c['family'] for c in ps], u0_reused_from=f'{A42_ROOT} WAVE_A42V_U0',
                       a42_admission_receipt=A42_ADMISSION_RECEIPT, unc_rule=C.UNC_RULE, verdicts=verdicts,
                       per_world=per_world, sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       charged_sim_s=D.charged(final_root), root=str(final_root)), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan():
    ident = 2*IDENT_N_WORLDS*N.HORIZON_S
    verify = N_PLANS*N_WORLDS*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, identity=ident, verification=verify, planned=ident+verify,
                u0_contingency=N_WORLDS*N.HORIZON_S, margin=CAP_SIM_S-ident-verify-N_WORLDS*N.HORIZON_S)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('prepare', 'dry', 'identity', 'identity-compare', 'verify', 'reduce',
                                       'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--chunk', type=int, default=33)
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
        vw = verify_waves(panel, root)
        a42 = a42_waves(panel)
        print(json.dumps(dict(
            budget=budget_plan(), charged=D.charged(root),
            ident=[(n, [(s['job_id'], s['world']['seed'], s['checkpoint'].get('a37_family')) for s in sp])
                   for n, _, sp in iw],
            verify=[(n, len(sp), sp[0]['job_id'], sp[0]['checkpoint']['a37_family'], sp[0]['checkpoint']['a37_bank'],
                     [sp[0]['world']['seed'], sp[-1]['world']['seed']]) for n, _, sp in vw],
            a42=[(n, len(sp), sp[0]['job_id']) for n, sp in a42],
            worlds=[r['world_id'] for r in panel['worlds']]), indent=1, default=str))
    elif a.command in ('identity', 'verify'):
        import fcntl
        if a.command == 'verify' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: the random plans are not run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = ident_waves(panel, root) if a.command == 'identity' else verify_waves(panel, root)
            need = sum(len(s) for _, _, s in waves)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay (A56 registration): a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['arm'], x['world_id'], x['abs_diff'],
                                                             x['safety_equal']) for x in v['worlds']])))
    elif a.command == 'reduce':
        v = reduce(root, panel)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/D.RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], charged=v['charged_sim_s'],
                              passes={f: x['PASS'] for f, x in v['verdicts'].items()})))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root))
        for name, _, specs in ident_waves(panel, root)+verify_waves(panel, root):
            out[name] = f'{N.published(root, name, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
