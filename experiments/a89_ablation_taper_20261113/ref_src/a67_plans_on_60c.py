"""AMENDMENT 67 (S-PLANS-ON-60C; MAIN 2026-11-12, owner-approved ~19:00): the stored plan arms on top of MAPPO60C.

Question: does the frozen searched bank (BIG_BANK = the four admitted A42 plans, bank 7e7dc76b) layered on the
retrained MAPPO60C policy (A65 policy_00084, 2abed317) reduce total delay relative to MAPPO60C alone?
Changed variable versus A42 / A66: only the base policy under the plan executor (MAPPO60C instead of u0 d9042cc5).
Same executor code (A37 CELL_BANK for BIG_BANK, A32 DUO_HOLD for the hand-written hold), same bank files and
hashes, same cell D2800_p60, same panel 1 (the A42 confirmation panel, worlds 998600-998695), same noise seeds.

Arms run here (96 worlds each):
  BIG_BANK_60C = runner arm CELL_BANK, bank A42_BIG_BANK_D2800_p60.json (byte copy, sha pinned), executor MAPPO60C
  DUO_HOLD_60C = runner arm DUO_HOLD (duo12_v20 of UNION_UNC 39f384a2, bank_cell_key D2800_p20), executor MAPPO60C
Only the six executor fields of the registered hold specs are replaced (as A52.with_executor does for U0 specs).
Stored arms (never re-run): MAPPO60C (A66 root, wave A66P1C_MAPPO60C), U0 / DUO_HOLD / BIG_BANK (A42 root copy).

Disclosure: the plans were searched in the WM and admitted in SUMO against u0; on MAPPO60C they run unverified
(no re-search, no re-admission, no plan selection, no tuning); panel 1 is a reused panel.

Identity (before any production episode): the plan-executor path with an EMPTY plan set on MAPPO60C, 4 worlds =
the first four panel worlds. CELL_BANK refuses an empty bank by its registered definition (a37_cellbank.make_core),
so the empty-plan episode is the runner arm ORACLE = oracle_plan.OracleCore (the class DUO_HOLD instantiates and
CELL_BANK's CellBankCore subclasses; same hook wrapper around controller.apply) on the frozen bank fd385a2b, which
has no row for D2800_p60. No executor code or bank file is changed or added. The new records must equal the stored
A66 MAPPO60C records exactly (J_veh_s difference 0, identical safety counters, same world, zero plans, zero
activations); otherwise FAILED and nothing else runs.

Budget: ONE ledger phase ``plans_on_60c_a67`` cap 240,000 sim-s on a NEW final root (2 x 96 x 1,200 = 230,400;
identity 4,800; margin 4,800). No automatic replay: a failed dispatch raises and is reported.

Analysis (zero physics): the A42 driver's ``_contrast`` (paired percentile bootstrap 10,000, seed 20261104, two-sided;
diff = treatment - reference, negative = less delay; total, five groups with the 10 s limit, CVaR90, safety counters).
  primary     BIG_BANK_60C - MAPPO60C (total delay): "adds" iff CI95 upper < 0; "interferes" iff CI95 lower > 0;
              otherwise "does not add"
  secondary   DUO_HOLD_60C - MAPPO60C ; BIG_BANK_60C - DUO_HOLD_60C ; BIG_BANK_60C - BIG_BANK (stored, on u0)
  context     BIG_BANK_60C - U0 ; DUO_HOLD_60C - U0 (group tables relative to U0) ; MAPPO60C - U0 (reproduces A66)
plus plan activations and realised group sizes (full-group formation) per hold arm, new and stored.

Commands (python ops/a67_plans_on_60c.py <cmd> --final-root <root> --a66-root <root> --a42-root <root copy>):
  dry / status ; identity ; identity-compare ; confirm (refused unless identity PASS) ; analyse --out <dir>
"""
from __future__ import annotations

import argparse
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
import a52_mappo60 as A52  # noqa: E402   (_a50_records: sealed chunk receipts of a root)
import a66_mappo60c as A66  # noqa: E402  (MAPPO60C binding + stored A66 / A42 arms; never runs here)

AMENDMENT = '67'
CELL = 'D2800_p60'
PHASE = 'plans_on_60c_a67'
CAP_SIM_S = 240_000
CONFIRM_N = 96
IDENT_N = 4
SETTING = 'a42c_D2800_p60'
PANEL_SEEDS = tuple(range(998600, 998696))
MAPPO60C_SHA256 = A66.MAPPO60C_SHA256
BIG_BANK_SHA256 = '7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff'
ADMISSION_RECEIPT_SHA256 = '5fa2a7015fb4ae9a95e1d17456e5caacb6114405cc197831d2c3d674f1b332d5'
ADMITTED = ('rand_3a452b1348b01083', 'mut_2922c89d0b31d4c8', 'rand_bb1cb39d47f0a524', 'mut_d7251d07b4f749e0')
EMPTY_BANK_SHA256 = 'fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82'
DUO_BANK_CELL = 'D2800_p20'
#: analysis name -> (runner arm, stored arm on u0)
ARMS = dict(BIG_BANK_60C=('CELL_BANK', 'BIG_BANK'), DUO_HOLD_60C=('DUO_HOLD', 'DUO_HOLD'))
STORED_A42 = ('U0', 'DUO_HOLD', 'BIG_BANK')
REF = 'MAPPO60C'
IDENT_WAVE = 'A67_IDENT_EMPTY'
IDENTITY_FILE = 'IDENTITY_A67.json'
RESULTS_FILE = f'A67_RESULTS_p1_{CELL}.json'
PRIMARY = ('BIG_BANK_60C', 'MAPPO60C')
SECONDARY = (('DUO_HOLD_60C', 'MAPPO60C'), ('BIG_BANK_60C', 'DUO_HOLD_60C'), ('BIG_BANK_60C', 'BIG_BANK'))
CONTEXT = (('BIG_BANK_60C', 'U0'), ('DUO_HOLD_60C', 'U0'), ('MAPPO60C', 'U0'))
AUTHORITY = ('MAIN AMENDMENT 67 (S-PLANS-ON-60C, owner-approved 2026-11-12 ~19:00): BIG_BANK and DUO_HOLD on top of '
             'MAPPO60C (2abed317) on panel 1 (998600-998695), D2800_p60; identity = empty plan set on MAPPO60C vs the '
             'stored A66 MAPPO60C records on 4 worlds; phase plans_on_60c_a67 cap 240,000')
DISCLOSURE = ('the plans were searched and admitted against u0 (d9042cc5); on MAPPO60C they run unverified: no '
              're-search, no re-admission, no plan selection, no tuning; panel 1 is a reused panel (A42, A66), not an '
              'independent panel; the reference arm MAPPO60C is the stored A66 run (same host), not re-run; one shot')
READING_RULE = ('primary BIG_BANK_60C - MAPPO60C total delay CI95: "adds" iff upper < 0; "interferes" iff lower > 0; '
                '"does not add" iff the CI contains 0 (diff = treatment - reference; negative = less delay)')


def core():
    """The A42 driver code with the A67 names (own ledger phase / cap / root)."""
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A67',
        HOST_BASE=os.environ.get('A67_HOST_BASE', '/root/a67'),
        VERIFY_SETTING='a42v_D2800_p60', CONFIRM_SETTING=SETTING,
        BANK_DIR='a67_banks', IDENT_WAVE=IDENT_WAVE, AMENDMENT=AMENDMENT, LABEL='A67', JOB_PREFIX='a67',
        VERIFY_TAG='a67v', CONFIRM_TAG='a67c', IDENT_TAG='a67i',
        VERIFY_WAVE_PREFIX='A67V_', CONFIRM_WAVE_PREFIX='A67C_',
        RESULTS_FILE=RESULTS_FILE, LOG_FILE='A67_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a67',
        RUNNER_RANGES_ATTR='A42_RANGES',
        RESULT_LABEL='amendment 67; plan arms on MAPPO60C, panel 1 (reused); never pooled; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone('a67_core', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


def check_panel(panel):
    seeds = tuple(r['world']['seed'] for r in panel['worlds'])
    if seeds != PANEL_SEEDS or panel['cell']['cell_key'] != CELL:
        raise N.Blocked('panel is not the registered panel 1 (998600-998695, D2800_p60)')
    return panel


def wave_of(arm):
    return f'A67C_{arm}'


def executor(a66_root):
    """The sealed A66 MAPPO60C binding record (read where A66 published it; never rebound)."""
    exe = A66.read_bound(a66_root, 'MAPPO60C')
    if exe['sha256'] != MAPPO60C_SHA256:
        raise N.Blocked('MAPPO60C binding is not 2abed317')
    return exe


def with_exec(specs, exe, tag):
    """Hold-arm (or empty-plan) specs with ONLY the six executor fields replaced: u0 d9042cc5 -> ``exe``."""
    out = []
    for i, s in enumerate(specs):
        ck = deepcopy(s['checkpoint'])
        if ck.get('executor_sha256') != N.FROZEN_MAPPO_SHA or ck.get('kind') not in ('a32', 'a37', 'oracle'):
            raise N.Blocked(f'{s["job_id"]}: not a registered u0 plan-executor spec')
        ck.update(comparator_kind=exe['kind'], label=exe['label'], executor_path=exe['path'],
                  executor_sha256=exe['sha256'], binding_sha256=exe['binding_sha256'],
                  binding_path=exe['binding_path'])
        out.append(dict(s, checkpoint=ck, job_id=f'{tag}_{i:04}'))
    return out


def admission(a42_root):
    p = Path(a42_root)/'ADMISSION_D2800_p60.json'
    adm = N.verify(N.read(p), 'receipt_sha256', p.name)
    if (adm['receipt_sha256'] != ADMISSION_RECEIPT_SHA256 or tuple(adm['admitted']) != ADMITTED
            or adm['big_bank']['file_sha256'] != BIG_BANK_SHA256):
        raise N.Blocked('A42 admission receipt is not the registered one (5fa2a701 / bank 7e7dc76b)')
    return adm


def base_specs(arm, panel, final_root, a42_root):
    """The registered u0 specs of the hold arm built on the A67 root (same builders as A42)."""
    D = core()
    if arm == 'DUO_HOLD_60C':
        return D.confirm_specs('DUO_HOLD', panel, final_root)
    if arm == 'BIG_BANK_60C':
        admission(a42_root)
        entry = D._bank_entry(final_root, D.BIG_BANK_FILE)
        if entry['file_sha256'] != BIG_BANK_SHA256:
            raise N.Blocked('the A67 copy of the BIG_BANK bank is not 7e7dc76b')
        return D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, entry, tag='a67c_big_bank')
    raise N.Blocked(f'no A67 arm {arm!r}')


def confirm_specs(arm, panel, final_root, a66_root, a42_root):
    specs = with_exec(base_specs(arm, panel, final_root, a42_root), executor(a66_root), f'a67c_{arm.lower()}')
    if len(specs) != CONFIRM_N or any(s['arm'] != ARMS[arm][0] for s in specs):
        raise N.Blocked(f'{arm}: {len(specs)} specs / wrong runner arm')
    return specs


def empty_plans(final_root):
    """Zero physics: the frozen bank fd385a2b admits no plan for D2800_p60 (the identity arm's plan set is empty)."""
    from tools.coopmwm import oracle_plan as O
    D = core()
    _, oracle = D.A.bindings(final_root)
    bank = N.arm_bank(oracle, 'ORACLE')
    if bank['file_sha256'] != EMPTY_BANK_SHA256 or N.file_sha(bank['path']) != EMPTY_BANK_SHA256:
        raise N.Blocked('the bound oracle bank is not fd385a2b')
    plans, refused = O.bank_plans(N.read(bank['path']), CELL, None)
    if plans or refused:
        raise N.Blocked(f'fd385a2b is not empty for {CELL}: {[p["family"] for p in plans]} {refused}')
    return dict(bank=bank, cell=CELL, plans=0, refused=refused)


def ident_specs(panel, final_root, a66_root):
    D = core()
    empty_plans(final_root)
    u0, oracle = D.A.bindings(final_root)
    base = N.episode_specs('ORACLE', panel, None, Path(final_root)/N.PRERUN_PLANT, 'mappo_frozen', oracle, u0)
    return with_exec(base[:IDENT_N], executor(a66_root), 'a67i_empty')


def stored_ref(panel, a66_root):
    """The stored A66 MAPPO60C panel-1 records (checked against their registered A66 specs)."""
    D66 = A66.core('p1')
    return D66.C37._load(a66_root, A66.confirm_wave('p1', 'MAPPO60C'),
                         A66.confirm_specs('p1', 'MAPPO60C', panel, a66_root))


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(panel, final_root, a66_root):
    D = core()
    specs = ident_specs(panel, final_root, a66_root)
    got, _, cov = N.published(final_root, IDENT_WAVE, specs, D.C37.check_a37)
    if cov != len(specs):
        raise N.Blocked(f'{IDENT_WAVE}: {cov}/{len(specs)} published')
    ref, _ = stored_ref(panel, a66_root)
    rows = []
    for r in got:
        a = ref[r['world_id']]
        o = r.get('oracle') or {}
        rows.append(dict(world_id=r['world_id'], j_a67=r['J_veh_s'], j_stored=a['J_veh_s'],
                         abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a67=r['safety'], safety_stored=a['safety'],
                         safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
                         executor_loaded=r.get('executor_sha256_loaded'),
                         executor_is_mappo60c=r.get('executor_sha256_loaded') == MAPPO60C_SHA256
                         and a.get('executor_sha256_loaded') == MAPPO60C_SHA256,
                         plans=len(o.get('plans') or []), activations=int(o.get('activations') or 0),
                         issued=int(o.get('issued') or 0),
                         vehicle_log_equal=(r.get('vehicle_log') or {}).get('sha256')
                         == (a.get('vehicle_log') or {}).get('sha256')))
    ok = (len(rows) == IDENT_N and len({x['world_id'] for x in rows}) == IDENT_N
          and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world'] and x['executor_is_mappo60c']
                  and x['plans'] == 0 and x['activations'] == 0 and x['issued'] == 0 for x in rows))
    return N.seal(dict(schema='coopmwm.a67.identity.v1', amendment=AMENDMENT,
                       rule='the plan-executor path with an empty plan set (runner arm ORACLE, oracle_plan.OracleCore, '
                            'frozen bank fd385a2b: no row for D2800_p60) on MAPPO60C reproduces the stored A66 '
                            'MAPPO60C records exactly on the first 4 panel-1 worlds: J_veh_s difference 0, identical '
                            'safety counters, zero plans / activations',
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: the two arms may run; FAILED: stop and report, nothing else runs'),
                  'receipt_sha256')


def identity_passed(final_root):
    p = Path(final_root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ analysis

def plan_specs(final_root):
    """{analysis arm: {family: strategy spec}} exactly as the executor rebuilds them from the bank files."""
    from tools.coopmwm import a32_arms, oracle_plan as O
    D = core()
    big = N.read(D.bank_path(final_root, D.BIG_BANK_FILE))
    duo = a32_arms.duo_bank(N.read(Path(final_root)/D.A.DUO_BANK_REL), DUO_BANK_CELL)
    return dict(BIG_BANK={p['family']: p['spec'] for p in O.bank_plans(big, CELL)[0]},
                DUO_HOLD={p['family']: p['spec'] for p in O.bank_plans(duo, DUO_BANK_CELL)[0]})


def activation_table(recs, worlds, specs):
    """Plan activations and realised group sizes of one hold arm (receipts a37 / a32 of the sealed records)."""
    from tools.coopmwm import a42_bigbank as B
    rows, fam, any_ = [], {}, 0
    for w in worlds:
        rec = recs[w].get('a37') or recs[w].get('a32') or {}
        any_ += 1 if rec.get('activations') else 0
        for f, n in (rec.get('activations_by_family') or {}).items():
            fam[f] = fam.get(f, 0)+int(n)
        rows += B.realised_sizes(rec, specs)
    full = {}
    for r in rows:
        x = full.setdefault(r['family'], dict(activations=0, full_group=0, planned=r['planned']))
        x['activations'] += 1
        x['full_group'] += 1 if r['held'] == r['planned'] else 0
    return dict(worlds_with_any=any_, total=sum(fam.values()), by_family=fam,
                realised_group_size=B.group_size_summary(rows),
                full_group=dict(total=sum(1 for r in rows if r['held'] == r['planned']), by_family=full,
                                rule='an activation forms its full group iff every planned member held past its '
                                     'role onset (a42_bigbank.realised_sizes: held == planned)'))


def reading_of(ci):
    return 'adds' if ci[1] < 0 else 'interferes' if ci[0] > 0 else 'does not add'


def analyse(final_root, panel, a66_root, a42_root):
    import numpy as np
    from a32_ablation import report_of
    final_root = Path(final_root)
    D = core()
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    if len(worlds) != CONFIRM_N:
        raise N.Blocked(f'{len(worlds)} worlds, not {CONFIRM_N}')
    have, roots, sources = A66.load_a42_arms(panel, a42_root, STORED_A42)
    have[REF], sources[REF] = stored_ref(panel, a66_root)
    roots[REF] = Path(a66_root)
    for arm in ARMS:
        have[arm], sources[arm] = D.C37._load(final_root, wave_of(arm),
                                              confirm_specs(arm, panel, final_root, a66_root, a42_root))
        roots[arm] = final_root
    reports = {a: {w: report_of(roots[a], have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = [D._contrast(have, reports, worlds, *PRIMARY, bootstrap, 'primary')]
    rows += [D._contrast(have, reports, worlds, a, r, bootstrap, 'secondary_descriptive') for a, r in SECONDARY]
    rows += [D._contrast(have, reports, worlds, a, r, bootstrap, 'context_descriptive') for a, r in CONTEXT]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    specs = plan_specs(final_root)
    for arm, (_, stored) in ARMS.items():
        arms[arm]['activations'] = activation_table(have[arm], worlds, specs[stored])
        arms[stored]['activations'] = activation_table(have[stored], worlds, specs[stored])
    ci = rows[0]['total_minus_ref']['ci95']
    return N.seal(dict(
        schema='coopmwm.a67.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL, panel='p1',
        panel_worlds='998600-998695 (A42 confirmation panel; reused)', disclosure=DISCLOSURE,
        panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
        executor=dict(sha256=MAPPO60C_SHA256, binding_sha256=executor(a66_root)['binding_sha256']),
        banks=dict(BIG_BANK=BIG_BANK_SHA256, admission_receipt=ADMISSION_RECEIPT_SHA256,
                   DUO_HOLD='39f384a29fb8e952a94f918234c52bb3ff2059f2155b99f17b2f5c26a38ec239'),
        primary=dict(contrast='BIG_BANK_60C - MAPPO60C', rule=READING_RULE, ci95=list(ci),
                     diff=rows[0]['total_minus_ref']['diff'], p_two_sided=rows[0]['total_minus_ref']['p_two_sided'],
                     reading=reading_of(ci)),
        contrasts=rows, arms=arms,
        multiplicity='one registered primary; every other contrast is descriptive (no multiplicity correction)',
        identity=N.read(final_root/IDENTITY_FILE).get('receipt_sha256'),
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
        ref_roots=dict(a66=str(a66_root), a42=str(a42_root)), label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan():
    conf = len(ARMS)*CONFIRM_N*N.HORIZON_S
    ident = IDENT_N*N.HORIZON_S
    return dict(phase=PHASE, cap_sim_s=CAP_SIM_S, confirmation=conf, identity=ident, planned=conf+ident,
                margin=CAP_SIM_S-conf-ident)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('dry', 'identity', 'identity-compare', 'confirm', 'analyse', 'status'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--a66-root', required=True)
    p.add_argument('--a42-root', required=True)
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    root = Path(a.final_root)
    if root.resolve() in (Path(a.a66_root).resolve(), Path(a.a42_root).resolve()):
        raise N.Blocked('the A67 final root is a new root (never the A66 / A42 root)')
    D = core()
    panel = check_panel(D.setup(SETTING))

    def specs_of(arm):
        return confirm_specs(arm, panel, root, a.a66_root, a.a42_root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'dry':
        ident = ident_specs(panel, root, a.a66_root)
        conf = {arm: specs_of(arm) for arm in ARMS}
        ref, src = stored_ref(panel, a.a66_root)
        stored, _, _ = A66.load_a42_arms(panel, a.a42_root, STORED_A42)
        ck = {arm: {k: v for k, v in c[0]['checkpoint'].items() if k not in ('label',)} for arm, c in conf.items()}
        print(json.dumps(dict(
            budget=budget_plan(), panel_sha256=panel['panel_sha256'], empty=empty_plans(root),
            ident=[(s['job_id'], s['world_id'], s['arm'], s['checkpoint']['kind'],
                    s['checkpoint']['executor_sha256'][:12], s['world']['seed']) for s in ident],
            ident_checkpoint={k: v for k, v in ident[0]['checkpoint'].items() if k != 'label'},
            confirm={arm: dict(n=len(c), first=c[0]['job_id'], last=c[-1]['job_id'], runner_arm=c[0]['arm'],
                               seeds=[c[0]['world']['seed'], c[-1]['world']['seed']], cell=c[0]['world']['cell_key'],
                               checkpoint=ck[arm]) for arm, c in conf.items()},
            stored={k: len(v) for k, v in dict(stored, MAPPO60C=ref).items()},
            plans={k: {f: len(s['roles']) for f, s in v.items()} for k, v in plan_specs(root).items()},
            charged=D.charged(root)), indent=1))
    elif a.command in ('identity', 'confirm'):
        import fcntl
        if a.command == 'confirm' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: nothing is run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = [(IDENT_WAVE, 'episode', ident_specs(panel, root, a.a66_root))]
            else:
                waves = [(wave_of(arm), 'episode', specs_of(arm)) for arm in ARMS]
            pending = sum(len(s)-N.published(root, w, s, D.C37.check_a37)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay: a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(panel, root, a.a66_root)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['world_id'], x['abs_diff'], x['safety_equal'],
                                                            x['same_world'], x['plans'], x['activations'],
                                                            x['vehicle_log_equal']) for x in v['worlds']])))
    elif a.command == 'analyse':
        v = analyse(root, panel, a.a66_root, a.a42_root)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root))
        for arm in ARMS:
            specs = specs_of(arm)
            out[wave_of(arm)] = f'{N.published(root, wave_of(arm), specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
