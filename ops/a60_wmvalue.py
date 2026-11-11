"""AMENDMENT 60 (S-WMVALUE60; registered before any outcome): host driver for 60C (S-RANDBANK60) and 60A (S-WMPROP96).

This driver IS ops/a42_bigp60.py (private module copies, as A50 / A52 / A56) with only amendment names replaced, plus
an execution-only worker count chosen per dispatch round from the cores other jobs leave free (Box 1 is shared; the
other jobs have priority). Episodes, executor arms (A37_PLAN / CELL_BANK on frozen U0 policy_00169), plant, bindings,
checks and ledgers are the A42 ones. No automatic replay: a failed dispatch raises and is reported.

60C (setting a42c_D2800_p60, panel 1 = 998600-998695; root <base>/rootC, phase randbank_a60c, cap 125,000):
  RAND_BANK = the five A56-admitted random plans, bank built by the A42 BIG_BANK builder
  (a37_cellbank.admitted_plans + bank_doc on the sealed A56 reduce verdicts), runner arm CELL_BANK (= BIG_BANK's).
  identity-c   U0 + BIG_BANK (byte copy of the A42 BIG_BANK file) on 998600-998603 (8 episodes = 9,600 sim-s)
  compare-c    -> IDENTITY_A60C.json: exact reproduction of the stored A42 confirmation records (|dJ| = 0, identical
               safety, same world) -> stored A42 U0 / DUO_HOLD / BIG_BANK reused; FAILED -> stop
  run-c        RAND_BANK on the 96 worlds (--limit 1 = the one-world smoke)
  reduce-c     zero physics: per-world J / safety / reports of U0, DUO_HOLD, BIG_BANK (A42 root, read-only) and
               RAND_BANK; contrasts BIG_BANK - RAND_BANK (primary), RAND_BANK - U0, RAND_BANK - DUO_HOLD (sealed)

60A (setting a60a_D2800_p60 = 995200-995295; root <base>/rootA, phase wmprop_a60a, cap 2,250,000):
  identity-a   (setting a42v_D2800_p60) U0 + rand_3a452b1348b01083 on 992000-992003 vs the stored A42 verification
  compare-a    records -> IDENTITY_A60A.json (as A56); FAILED -> stop
  run-a        U0 + the six A42 WM-ranked plans + the twelve A56 random plans, each a single-plan bank (A37_PLAN), on
               the 96 worlds (--limit 1 = the one-world smoke: world 995200 of every wave)
  reduce-a     zero physics: per-world J / safety / activations + per-plan comparison vs U0 (sealed)

    python ops/a60_wmvalue.py <command> --final-root <root> [--workers-max N] [--limit 1]
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

AMENDMENT = '60'
CELL = 'D2800_p60'
HOST_BASE = os.environ.get('A60_HOST_BASE', '/root/a60')
A42_ROOT = '/root/a42v/root'
A56_ROOT = '/root/a56/root'
A42_ADMISSION_RECEIPT = '5fa2a7015fb4ae9a95e1d17456e5caacb6114405cc197831d2c3d674f1b332d5'
A56_REDUCE_FILE_SHA256 = '2aea2de482050bebfdddb0046bb2c3529025aca37b7f551eeb7dde067b4a4dcc'
RANDPLANS_FILE = HERE/'a56'/f'RANDPLANS_{CELL}.json'
RANDPLANS_SHA256 = '687663b078c768643d56758b42dc8113d297be1d606ecaf7994217964ec3dbaa'
RAND_BANK_FAMILIES = ('rand_2c80042490ec698f', 'rand_645eff3ebc6306c4', 'rand_9958c362eb4af331',
                      'rand_d25a87bf304bd297', 'rand_1194a86754a5c92d')          # registered (A56-admitted five)
IDENT_N_WORLDS = 4
IDENT_FAMILY_A = 'rand_3a452b1348b01083'
A42_CANDS = f'A42_CANDS_{CELL}.json'
A56_CANDS = f'A56_RAND_CANDS_{CELL}.json'
A42_BIG = f'A42_BIG_BANK_{CELL}.json'
RAND_BANK_FILE = f'A60_RAND_BANK_{CELL}.json'
LOAD_TARGET = int(os.environ.get('A60_LOAD_TARGET', '44'))    # keep total busy cores <= this
MIN_WORKERS = 2

SETTING_A, SETTING_IDENT_A, SETTING_C = 'a60a_D2800_p60', 'a42v_D2800_p60', 'a42c_D2800_p60'
CAP_A, CAP_C = 2_250_000, 125_000
AUTH = ('MAIN AMENDMENT 60 S-WMVALUE60 (registered 2026-11-11 before any outcome; owner approved running all three '
        'tests 11-11)')


def _driver(tag, phase, cap, verify_setting, confirm_setting, ranges_attr, root_name):
    m = A50._clone(f'a60{tag}_core', HERE/'a42_bigp60.py', dict(
        A42_PHASE=phase, A42_CAP_SIM_S=cap, STOP_A42=f'STOP_A60{tag.upper()}', HOST_BASE=HOST_BASE,
        VERIFY_SETTING=verify_setting, CONFIRM_SETTING=confirm_setting, BANK_DIR='a60_banks',
        IDENT_WAVE=f'A60{tag.upper()}_IDENT_U0', AMENDMENT=AMENDMENT, LABEL=f'A60{tag.upper()}',
        JOB_PREFIX=f'a60{tag}', VERIFY_TAG=f'a60{tag}v', CONFIRM_TAG=f'a60{tag}c', IDENT_TAG=f'a60{tag}i',
        VERIFY_WAVE_PREFIX=f'A60{tag.upper()}V_', CONFIRM_WAVE_PREFIX=f'A60{tag.upper()}C_',
        CANDS_BANK_FILE=f'A60_UNUSED_CANDS_{CELL}.json', BIG_BANK_FILE=f'A60_UNUSED_BIG_{CELL}.json',
        ADMISSION_FILE=f'ADMISSION_A60{tag.upper()}_UNUSED.json', RESULTS_FILE=f'A60{tag.upper()}_REDUCE_{CELL}.json',
        LOG_FILE=f'A60{tag.upper()}_LOG.jsonl', SCHEMA_PREFIX=f'coopmwm.a60{tag}', RUNNER_RANGES_ATTR=ranges_attr,
        RESULT_LABEL=f'amendment 60 part 60{tag.upper()}; never pooled; one shot', AUTHORITY=AUTH))
    m.HOST_ROOT = f'{HOST_BASE}/{root_name}'
    m.SETTINGS = (verify_setting, confirm_setting)
    m.A42Runtime.phase = phase
    return m


DA = _driver('a', 'wmprop_a60a', CAP_A, SETTING_A, SETTING_IDENT_A, 'A60_RANGES', 'rootA')
DC = _driver('c', 'randbank_a60c', CAP_C, SETTING_C, SETTING_C, 'A42_RANGES', 'rootC')
D42 = A50._clone('a60_a42_view', HERE/'a42_bigp60.py', {})      # the A42 driver itself (A42 names; read-only use)


# ------------------------------------------------------------------------------------------------ inputs

def a42_admission():
    p = Path(A42_ROOT)/'ADMISSION_D2800_p60.json'
    v = N.verify(N.read(p), 'receipt_sha256', p.name)
    if v['receipt_sha256'] != A42_ADMISSION_RECEIPT:
        raise N.Blocked('A42 admission receipt is not the registered one')
    return v


def wm_top6():
    rows = sorted(a42_admission()['verdicts'], key=lambda v: v['selected_rank'])
    if len(rows) != 6:
        raise N.Blocked('A42 admission does not list six plans')
    return [dict(family=v['family'], key=v['key']) for v in rows]


def rand12():
    from tools.coopmwm import a37_cellbank as C
    if N.file_sha(RANDPLANS_FILE) != RANDPLANS_SHA256:
        raise N.Blocked(f'{RANDPLANS_FILE.name}: not the registered plan list')
    doc = json.loads(RANDPLANS_FILE.read_text(encoding='utf-8'))
    out = doc['plans']
    if len(out) != 12 or doc['cell'] != CELL:
        raise N.Blocked('plan list is not the registered 12 random plans')
    for c in out:
        if C.spec_sha(c['strategy']) != c['strategy_sha256'] or C.executor_refusal(c['strategy']) is not None:
            raise N.Blocked(f'{c["family"]}: strategy sha / executability')
    return out


def a56_reduce():
    p = Path(A56_ROOT)/f'A56_REDUCE_{CELL}.json'
    if N.file_sha(p) != A56_REDUCE_FILE_SHA256:
        raise N.Blocked(f'{p}: not the registered A56 reduce file')
    return N.verify(N.read(p), 'analysis_sha256', p.name)


def bank_entry(D, final_root, name):
    p = Path(final_root)/D.BANK_DIR/name
    if not p.is_file():
        raise N.Blocked(f'{p}: missing (run prepare)')
    return dict(path=f'{D.HOST_ROOT}/{D.BANK_DIR}/{name}', file_sha256=N.file_sha(p))


def copy_bank(D, final_root, src, name):
    """Byte copy of a stored bank file into this root (refuses an existing different file)."""
    dst = Path(final_root)/D.BANK_DIR/name
    data = Path(src).read_bytes()
    if dst.exists():
        if dst.read_bytes() != data:
            raise N.Blocked(f'{dst}: exists and differs from {src}')
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return dict(path=str(dst), file_sha256=N.file_sha(dst), source=str(src))


def prepare_c(final_root):
    from tools.coopmwm import a37_cellbank as C
    red = a56_reduce()
    cands = rand12()
    verdicts = {f: red['verdicts'][f] for f in red['random12']}
    plans = C.admitted_plans(cands, verdicts)
    if sorted(p['family'] for p in plans) != sorted(RAND_BANK_FAMILIES):
        raise N.Blocked(f'admitted A56 plans {[p["family"] for p in plans]} are not the registered five')
    p = DC.C37.write_once(Path(final_root)/DC.BANK_DIR/RAND_BANK_FILE, DC.C37.bank_text(C.bank_doc(plans, CELL)))
    adm = D42.admission(A42_ROOT)
    big = copy_bank(DC, final_root, Path(A42_ROOT)/D42.BANK_DIR/A42_BIG, A42_BIG)
    if big['file_sha256'] != adm['big_bank']['file_sha256']:
        raise N.Blocked('A42 BIG_BANK copy differs from the A42 admission receipt')
    return dict(rand_bank=dict(path=str(p), file_sha256=N.file_sha(p), order=[(x['family'], x['lb95_veh_s'])
                                                                               for x in plans]), big_bank_copy=big)


def prepare_a(final_root):
    a = copy_bank(DA, final_root, Path(A42_ROOT)/D42.BANK_DIR/A42_CANDS, A42_CANDS)
    b = copy_bank(DA, final_root, Path(A56_ROOT)/'a56_banks'/A56_CANDS, A56_CANDS)
    return dict(a42_cands=a, a56_cands=b, wm_top6=[x['family'] for x in wm_top6()],
                random12=[c['family'] for c in rand12()])


# ------------------------------------------------------------------------------------------------ specs

def waves_c(panel, final_root, *, identity=False):
    if identity:
        u0 = DC._retag(DC.confirm_specs('U0', panel, final_root)[:IDENT_N_WORLDS], 'a60ci_u0')
        big = DC._a37_specs(DC.RUNNER_ARM['BIG_BANK'], panel, final_root, bank_entry(DC, final_root, A42_BIG),
                            tag='a60ci_big_bank')[:IDENT_N_WORLDS]
        return [('A60C_IDENT_U0', 'episode', u0), ('A60C_IDENT_BIG_BANK', 'episode', big)]
    return [('A60CC_RAND_BANK', 'episode',
             DC._a37_specs(DC.RUNNER_ARM['BIG_BANK'], panel, final_root, bank_entry(DC, final_root, RAND_BANK_FILE),
                           tag='a60cc_rand_bank'))]


def plan_list_a():
    return [(x['family'], x['key'], A42_CANDS, 'WM_A42') for x in wm_top6()] + \
           [(c['family'], c['key'], A56_CANDS, 'RANDOM_A56') for c in rand12()]


def waves_a(panel, final_root, *, identity=False):
    if identity:
        u0_wave = DA.verify_waves(panel, final_root, check_files=False, cands=[])[0]
        u0 = DA._retag(u0_wave[2][:IDENT_N_WORLDS], 'a60ai_u0')
        plan = DA._a37_specs(DA.RUNNER_ARM['PLAN'], panel, final_root, bank_entry(DA, final_root, A42_CANDS),
                             family=IDENT_FAMILY_A, tag='a60ai_3a452b13')[:IDENT_N_WORLDS]
        return [('A60A_IDENT_U0', 'episode', u0), ('A60A_IDENT_PLAN', 'episode', plan)]
    u0_wave = DA.verify_waves(panel, final_root, check_files=False, cands=[])[0]
    out = [('A60AV_U0', 'episode', DA._retag(u0_wave[2], 'a60av_u0'))]
    for fam, key, bank, _src in plan_list_a():
        out.append((f'A60AV_{fam}', 'episode',
                    DA._a37_specs(DA.RUNNER_ARM['PLAN'], panel, final_root, bank_entry(DA, final_root, bank),
                                  family=fam, tag=f'a60av_{key[:8]}')))
    return out


# ------------------------------------------------------------------------------------------------ execution

def busy_cores(sample_s=3.0):
    """Cores busy over a short /proc/stat sample (between dispatch rounds: this driver has no workers running)."""
    def snap():
        with open('/proc/stat') as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle = v[3]+v[4]
        return sum(v), idle
    t0, i0 = snap()
    time.sleep(sample_s)
    t1, i1 = snap()
    n = os.cpu_count() or 1
    return n*(1-(i1-i0)/max(1, t1-t0))


def execute_adaptive(D, final_root, waves, *, workers_max, log, panel):
    """D.execute (a42_bigp60.execute) with the worker count re-chosen before every dispatch round:
    min(workers_max, LOAD_TARGET - cores busy with other jobs), at least MIN_WORKERS. Execution-only."""
    final_root = Path(final_root)
    runtime = D.A42Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if runtime.physics:
        if not os.environ.get('COOPMWM_MAP_SPEC'):
            raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
        status = runtime.runner_status()
        if not status['ok'] or not status.get('learned'):
            raise N.Blocked(f'runner cannot run the arms: {status}')
        from tools.coopmwm import nm_eval_runner as R
        if getattr(R, 'A37_ARMS', None) != D.C37.A37_ARMS or getattr(R, D.RUNNER_RANGES_ATTR, None) is None:
            raise N.Blocked(f'runner lacks the amendment-37 executor arms / {D.RUNNER_RANGES_ATTR}')
        N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = D.A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, D.C37.check_a37)[2], len(specs))
                for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/D.STOP_A42).exists():
            log(dict(stopped=True, coverage=cov))
            return dict(status='STOPPED', coverage=cov, charged_sim_s=D.charged(final_root))
        busy = busy_cores()
        workers = max(MIN_WORKERS, min(int(workers_max), int(LOAD_TARGET-busy)))
        budget, limits, selection = workers+1, {}, []
        for name, kind, specs in pending:
            take = min(cov[name][1]-cov[name][0], budget)
            selection.append((name, kind, specs))
            limits[name] = take
            budget -= take
            if budget == 0:
                break
        log(dict(round_start=rounds+1, busy_other_cores=round(busy, 1), workers=workers, episodes=sum(limits.values())))
        N.run_waves(selection, root=plant, final_root=final_root, cfg=cfg, runtime=runtime, workers=workers,
                    base='mappo_frozen', chunked=True, limit=limits)
        rounds += 1
        log(dict(round=rounds, published=sum(v[0] for v in coverage().values()), charged_sim_s=D.charged(final_root)))
    return dict(status='PASS', coverage=coverage(), charged_sim_s=D.charged(final_root))


# ------------------------------------------------------------------------------------------------ identity

def _ident_rows(final_root, waves, refs, arms):
    rows = []
    for (name, _, specs), arm in zip(waves, arms):
        got, _, cov = N.published(final_root, name, specs, DA.C37.check_a37)
        if cov != len(specs):
            raise N.Blocked(f'{name}: {cov}/{len(specs)} published')
        for r in got:
            a = refs[arm][r['world_id']]
            rows.append(dict(arm=arm, world_id=r['world_id'], j_a60=r['J_veh_s'], j_a42=a['J_veh_s'],
                             abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a60=r['safety'], safety_a42=a['safety'],
                             safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
                             activations=[(r.get('a37') or {}).get('activations'),
                                          (a.get('a37') or {}).get('activations')]))
    return rows


def _ident_seal(part, rule, rows, n):
    ok = len(rows) == n and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world'] for x in rows)
    return N.seal(dict(schema=f'coopmwm.a60{part}.identity.v1', amendment=AMENDMENT, rule=rule,
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: stored A42 records reused; FAILED: stop and report'), 'receipt_sha256')


def load_a42_confirm(panel, arms=('U0', 'DUO_HOLD', 'BIG_BANK')):
    have, src = {}, {}
    for arm in arms:
        specs = D42.confirm_specs(arm, panel, A42_ROOT, check_files=True)
        have[arm], src[arm] = D42.C37._load(A42_ROOT, D42.confirm_wave(arm), specs)
    return have, src


def load_a42_verify(panel):
    have = {}
    for name, _, specs in D42.verify_waves(panel, A42_ROOT, check_files=True, cands=wm_top6()):
        have[name[len(D42.VERIFY_WAVE_PREFIX):]] = D42.C37._load(A42_ROOT, name, specs)[0]
    return have


def compare_c(final_root, panel):
    refs, _ = load_a42_confirm(panel, ('U0', 'BIG_BANK'))
    rows = _ident_rows(final_root, waves_c(panel, final_root, identity=True), refs, ('U0', 'BIG_BANK'))
    return _ident_seal('c', 'A60 rootC reproduces the stored A42 confirmation records exactly for U0 and BIG_BANK on '
                            '998600-998603 (|J diff| = 0, identical safety, same world)', rows, 2*IDENT_N_WORLDS)


def compare_a(final_root, panel):
    refs = load_a42_verify(panel)
    rows = _ident_rows(final_root, waves_a(panel, final_root, identity=True), refs, ('U0', IDENT_FAMILY_A))
    return _ident_seal('a', f'A60 rootA reproduces the stored A42 verification records exactly for U0 and '
                            f'{IDENT_FAMILY_A} on 992000-992003 (|J diff| = 0, identical safety, same world)',
                       rows, 2*IDENT_N_WORLDS)


def identity_passed(final_root, name):
    p = Path(final_root)/name
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ reduce

def _reports_readonly(roots, have, worlds):
    from a32_ablation import report_of
    out = {}
    for a in have:
        out[a] = {}
        for w in worlds:
            rec = have[a][w]
            d = Path(roots[a])/'evaluation'/rec.get('work_dir', rec['job_id'])
            if str(roots[a]) == A42_ROOT and not (d/'vehicles_report.json').is_file():
                raise N.Blocked(f'{d}: no cached report (A42 root is read-only here)')
            out[a][w] = report_of(roots[a], rec)
            if not out[a][w]['ok']:
                raise N.Blocked(f'{rec["job_id"]}: invalid vehicle-log report')
    return out


def _per_world(have, worlds, seed):
    return {a: {w: dict(seed=seed[w], J_veh_s=have[a][w]['J_veh_s'], safety=have[a][w]['safety'],
                        activations=(have[a][w].get('a37') or {}).get('activations'),
                        activations_by_family=(have[a][w].get('a37') or {}).get('activations_by_family'))
                for w in worlds} for a in have}


def reduce_c(final_root, panel):
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources = load_a42_confirm(panel)
    roots = {a: A42_ROOT for a in have}
    (name, _, specs), = waves_c(panel, final_root)
    have['RAND_BANK'], sources['RAND_BANK'] = DC.C37._load(final_root, name, specs)
    roots['RAND_BANK'] = str(final_root)
    reports = _reports_readonly(roots, have, worlds)
    contrasts = [D42._contrast(have, reports, worlds, 'BIG_BANK', 'RAND_BANK', bootstrap, 'primary'),
                 D42._contrast(have, reports, worlds, 'RAND_BANK', 'U0', bootstrap, 'secondary'),
                 D42._contrast(have, reports, worlds, 'RAND_BANK', 'DUO_HOLD', bootstrap, 'secondary'),
                 D42._contrast(have, reports, worlds, 'BIG_BANK', 'U0', bootstrap, 'context'),
                 D42._contrast(have, reports, worlds, 'BIG_BANK', 'DUO_HOLD', bootstrap, 'context')]
    return N.seal(dict(schema='coopmwm.a60c.reduce.v1', amendment=AMENDMENT, part='60C', authority=AUTH, cell=CELL,
                       setting=SETTING_C, panel_sha256=panel['panel_sha256'], bootstrap=bootstrap, worlds=worlds,
                       rand_bank=bank_entry(DC, final_root, RAND_BANK_FILE), contrasts=contrasts,
                       per_world=_per_world(have, worlds, seed),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       a42_reused_from=A42_ROOT, charged_sim_s=DC.charged(final_root), root=str(final_root)),
                  'analysis_sha256')


def reduce_a(final_root, panel):
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources = {}, {}
    arms = ['U0']+[p[0] for p in plan_list_a()]
    for (name, _, specs), arm in zip(waves_a(panel, final_root), arms):
        have[arm], sources[arm] = DA.C37._load(final_root, name, specs)
    reports = _reports_readonly({a: str(final_root) for a in have}, have, worlds)
    verdicts = {}
    for fam, _key, _bank, src in plan_list_a():
        cmp_ = N._bank_compare(have, reports, worlds, fam, 'U0', bootstrap)
        diffs = [float(have[fam][w]['J_veh_s'])-float(have['U0'][w]['J_veh_s']) for w in sorted(worlds, key=seed.get)]
        lo, hi = C.boot_ci(diffs)                         # the UNC bootstrap (a37_cellbank.boot_ci) on these worlds
        allowed = bool((cmp_['groups'] or {}).get('all_pass') is True)
        acts = [have[fam][w].get('a37') or {} for w in worlds]
        verdicts[fam] = dict(family=fam, source=src, n=len(diffs), mean=sum(diffs)/len(diffs), ci95_unc=[lo, hi],
                             allowed=allowed, PASS=bool(hi < 0 and allowed),
                             rule='UNC rule recomputed on the 96 60A worlds: (plan - U0) CI95 upper < 0 (a37_cellbank.'
                                  'boot_ci) AND ALLOWED (five groups present, each mean increase <= 10 s)',
                             total_minus_u0=cmp_['total_minus_ref'], groups=cmp_['groups'],
                             cvar90_vs_u0=cmp_['tail_cvar90'],
                             activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                             episodes_with_activation=sum(int(a.get('activations') or 0) > 0 for a in acts),
                             safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
    return N.seal(dict(schema='coopmwm.a60a.reduce.v1', amendment=AMENDMENT, part='60A', authority=AUTH, cell=CELL,
                       setting=SETTING_A, panel_sha256=panel['panel_sha256'], bootstrap=bootstrap, worlds=worlds,
                       wm_top6=[p[0] for p in plan_list_a() if p[3] == 'WM_A42'],
                       random12=[p[0] for p in plan_list_a() if p[3] == 'RANDOM_A56'],
                       randplans_sha256=RANDPLANS_SHA256, a42_admission_receipt=A42_ADMISSION_RECEIPT,
                       verdicts=verdicts, per_world=_per_world(have, worlds, seed),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       charged_sim_s=DA.charged(final_root), root=str(final_root)), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

COMMANDS = dict(
    **{'prepare-c': (DC, SETTING_C), 'identity-c': (DC, SETTING_C), 'compare-c': (DC, SETTING_C),
       'run-c': (DC, SETTING_C), 'reduce-c': (DC, SETTING_C), 'dry-c': (DC, SETTING_C),
       'prepare-a': (DA, SETTING_A), 'identity-a': (DA, SETTING_IDENT_A), 'compare-a': (DA, SETTING_IDENT_A),
       'run-a': (DA, SETTING_A), 'reduce-a': (DA, SETTING_A), 'dry-a': (DA, SETTING_A), 'dry-ident-a': (DA, SETTING_IDENT_A)})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=tuple(COMMANDS))
    p.add_argument('--final-root', required=True)
    p.add_argument('--workers-max', type=int, default=24)
    p.add_argument('--limit', type=int, help='run-a / run-c: only the first N worlds of every wave (smoke)')
    a = p.parse_args(argv)
    D, setting = COMMANDS[a.command]
    root = Path(a.final_root)
    if str(root) != D.HOST_ROOT and not os.environ.get('A60_LAPTOP'):
        raise N.Blocked(f'{a.command}: final root must be {D.HOST_ROOT}')
    panel = D.setup(setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=setting, command=a.command),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare-c':
        print(json.dumps(prepare_c(root), default=str))
    elif a.command == 'prepare-a':
        print(json.dumps(prepare_a(root), default=str))
    elif a.command.startswith('dry'):
        ws = (waves_c(panel, root, identity=False) + waves_c(panel, root, identity=True) if a.command == 'dry-c'
              else waves_a(panel, root, identity=True) if a.command == 'dry-ident-a' else waves_a(panel, root))
        print(json.dumps(dict(
            charged=D.charged(root), cap=D.A42_CAP_SIM_S, phase=D.A42_PHASE,
            waves=[(n, len(sp), sp[0]['job_id'], sp[0]['arm'], sp[0]['checkpoint'].get('a37_family'),
                    (sp[0]['checkpoint'].get('a37_bank') or {}).get('path'), [sp[0]['world']['seed'], sp[-1]['world']['seed']],
                    sum(s['sim_s'] for s in sp)) for n, _, sp in ws],
            worlds=[panel['worlds'][0]['world_id'], panel['worlds'][-1]['world_id']], n_worlds=len(panel['worlds']),
            panel_sha256=panel['panel_sha256']), indent=1, default=str))
    elif a.command in ('identity-c', 'identity-a', 'run-c', 'run-a'):
        import fcntl
        if a.command == 'run-c' and not identity_passed(root, 'IDENTITY_A60C.json'):
            raise N.Blocked('IDENTITY_A60C.json is not PASS')
        if a.command == 'run-a' and not identity_passed(root, 'IDENTITY_A60A.json'):
            raise N.Blocked('IDENTITY_A60A.json is not PASS')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/f'LAUNCHER_{a.command}.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()),
                      replace=True)
            if a.command == 'identity-c':
                waves = waves_c(panel, root, identity=True)
            elif a.command == 'identity-a':
                waves = waves_a(panel, root, identity=True)
            elif a.command == 'run-c':
                waves = waves_c(panel, root)
            else:
                waves = waves_a(panel, root)
            if a.limit:
                waves = [(n, k, s[:a.limit]) for n, k, s in waves]
            need = sum(len(s) for _, _, s in waves)*N.HORIZON_S
            if D.charged(root)+need > D.A42_CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {D.A42_CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers_max=a.workers_max,
                     limit=a.limit, pid=os.getpid(), charged_before=D.charged(root)))
            res = execute_adaptive(D, root, waves, workers_max=a.workers_max, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command in ('compare-c', 'compare-a'):
        v = compare_c(root, panel) if a.command == 'compare-c' else compare_a(root, panel)
        name = 'IDENTITY_A60C.json' if a.command == 'compare-c' else 'IDENTITY_A60A.json'
        N.publish(root/name, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['arm'], x['world_id'], x['abs_diff'], x['safety_equal'])
                                                             for x in v['worlds']])))
    elif a.command in ('reduce-c', 'reduce-a'):
        v = reduce_c(root, panel) if a.command == 'reduce-c' else reduce_a(root, panel)
        N.publish(root/D.RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], charged=v['charged_sim_s'])))


if __name__ == '__main__':
    main()
