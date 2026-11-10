"""AMENDMENT 45 (registered): DIVERSE candidates from the SAME A42 WM strategy search `final` at D2800_p60, verified
in SUMO with the SAME A37 UNC admission rule on the SAME 24 verification seeds (992000-992023; the A42 U0 episodes are
shared, never rerun), DIVERSE_BANK = every admitted plan of the A42 top 6 + the diverse candidates, one extra arm on
the SAME 96 confirmation worlds (998600-998695). Exploratory, never pooled.

Driver on top of ops/a42_bigp60.py (same final root, same mirrored bindings, same A37 executor UNCHANGED: runner arm
A37_PLAN = one diverse candidate, CELL_BANK = DIVERSE_BANK). The A42 driver and its outputs are not changed; A45 reads
the A42 admission receipt and the A42 U0 / DUO_HOLD / BIG_BANK episodes.

Selection (zero physics; tools.coopmwm.a42_bigbank.select_diverse, written into the pinned A42 candidates document as
``diverse``): eligible rows (the A42 skips) NOT in the top 6, grouped by structural class = descriptor fields
(n_roles, n_lanes, staggered, zone, phases); best row by WM score per class; classes by that score; at most 12; ties
by the order in `final`.

Budget: ledger phase ``divp60_a45`` capped at 480,000 sim-s (separate from bigp60_a42 = 720,000; own ledger
<final_root>/CHARGES.jsonl, total over both phases <= 1,200,000) = verification <= 12 x 24 x 1,200 = 345,600 +
confirmation 1 x 96 x 1,200 = 115,200 + retry reserve 19,200. One charged retry per failed episode (A42 rule).
Order: the unchanged A42 runtime counts EVERY row of the shared ledger against its 720,000 cap, so A45 spends only
after every A42 stage has finished (run_a42.sh calls run_a45.sh after A42_ALL_DONE).

Stages (python ops/a45_divp60.py <cmd> --final-root <root> --setting <setting>), run after the A42 chain:
  check-search  zero physics: --search <host search json>; the pinned candidates document must equal a fresh
                selection of that file (file sha, top 6 and diverse list)
  prepare       zero physics: a42_banks/A45_DIV_CANDS_D2800_p60.json (unconditional rows of the diverse candidates)
  dry           zero physics: specs / counts of the setting
  verify        (a42v_D2800_p60) the diverse candidates (A37_PLAN) x 24 worlds; U0 = the A42 wave A42V_U0
  admit         (a42v_D2800_p60) zero physics: UNC rule per diverse candidate -> ADMISSION_A45_D2800_p60.json and, if
                any diverse plan passes, a42_banks/A45_DIVERSE_BANK_D2800_p60.json (A42 admitted + A45 admitted)
  confirm       (a42c_D2800_p60) DIVERSE_BANK on the 96 worlds (skipped when no diverse plan is admitted)
  reports       zero physics: vehicles_report.json for every A45 episode of the setting
  analyse       (a42c_D2800_p60) A45 reading -> A45_RESULTS_D2800_p60.json (--out)
  status        coverage + ledger totals per phase
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
import a42_bigp60 as D  # noqa: E402

A45_PHASE = 'divp60_a45'
A45_CAP_SIM_S = 480_000
TOTAL_ALLOCATION_SIM_S = D.A42_CAP_SIM_S+A45_CAP_SIM_S         # both phases share the A42 final root's own ledger
MAX_DIVERSE = 12
VERIFY_SIM_S, CONFIRM_SIM_S = MAX_DIVERSE*24*1_200, 1*96*1_200
#: amendment-specific names (amendment 50 reuses this driver with its own values; see ops/a50_divp60.py)
AMENDMENT, LABEL, JOB_PREFIX = '45', 'A45', 'a45'
VERIFY_TAG, CONFIRM_TAG = 'a45v', 'a45c'                              # job id prefixes
VERIFY_WAVE_PREFIX = 'A45V_'
SCHEMA_PREFIX = 'coopmwm.a45'
LOG_FILE = 'A45_LOG.jsonl'
RESULT_LABEL = 'amendment 45; exploratory row at 60% CAVs; never pooled; one shot'
RETRY_RESERVE_SIM_S = A45_CAP_SIM_S-VERIFY_SIM_S-CONFIRM_SIM_S
STOP_A45 = 'STOP_A45'
CELL = D.CELL
ARM = 'DIVERSE_BANK'
ADMISSION_FILE = f'ADMISSION_A45_{CELL}.json'
CANDS_BANK = f'A45_DIV_CANDS_{CELL}.json'
DIVERSE_BANK_FILE = f'A45_DIVERSE_BANK_{CELL}.json'
RESULTS_FILE = f'A45_RESULTS_{CELL}.json'
AUTHORITY = ('MAIN AMENDMENT 45 (registered): diverse structural-class candidates from the same A42 search, same 24 '
             'verification seeds and UNC rule, DIVERSE_BANK = admitted top 6 + diverse, one extra arm on the same 96 '
             'confirmation worlds; phase divp60_a45 cap 480,000')
PRIMARY_RULE = ('diverse plans beat the hand-written hold at D2800_p60 iff DIVERSE_BANK - DUO_HOLD total delay CI95 '
                'upper < 0 (paired percentile bootstrap 10,000, seed 20261104) AND DIVERSE_BANK passes the group rule '
                'vs U0 (five registered groups present, each mean per-vehicle increase <= 10 s, U0 labels)')
NOT_BUILT = 'no diverse plan admitted: DIVERSE_BANK is not built (reported as the A45 result; no confirmation arm)'


def budget_plan():
    if RETRY_RESERVE_SIM_S < 0 or VERIFY_SIM_S+CONFIRM_SIM_S+RETRY_RESERVE_SIM_S != A45_CAP_SIM_S:
        raise N.Blocked(f'{LABEL} budget arithmetic')
    return dict(phase=A45_PHASE, cap_sim_s=A45_CAP_SIM_S, verification_max=VERIFY_SIM_S, confirmation=CONFIRM_SIM_S,
                retry_reserve=RETRY_RESERVE_SIM_S, separate_from=dict(phase=D.A42_PHASE, cap_sim_s=D.A42_CAP_SIM_S),
                own_ledger_total_allocation=TOTAL_ALLOCATION_SIM_S)


def charged_phase(final_root, phase=None):
    phase = A45_PHASE if phase is None else phase       # looked up at call time (A50 reuses this module)
    return sum(r['sim_s'] for r in N.fe.ledger_rows(final_root) if r.get('phase') == phase)


# ------------------------------------------------------------------------------------------------ candidates / banks

def diverse_candidates(path=None, pinned=None):
    """The diverse list of the pinned A42 candidates document (same pin, same STOP refusal as A42)."""
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    D.candidates(path, pinned)                                    # pin / sha / cell / K / STOP checks (A42)
    doc = N.read(Path(path or D.CANDS_FILE))
    div = doc.get('diverse')
    if not isinstance(div, dict) or div.get('amendment') != '45' or div.get('cap') != B.DIVERSE_CAP:
        raise N.Blocked('candidates document has no registered A45 diverse list')
    cands = div['candidates']
    if len(cands) > MAX_DIVERSE:
        raise N.Blocked(f'{LABEL}: more than 12 diverse candidates')
    top = {c['key'] for c in doc['candidates']}
    for c in cands:
        if c['key'] in top:
            raise N.Blocked(f'{c["family"]}: a top-6 row in the diverse list')
        if C.spec_sha(c['strategy']) != c['strategy_sha256']:
            raise N.Blocked(f'{c["family"]}: strategy sha differs')
    return cands


def check_search(search_path, path=None, pinned=None):
    """The host search file reproduces the pinned candidates document (top 6 and diverse) exactly."""
    from tools.coopmwm import a42_bigbank as B
    search_path = Path(search_path)
    D.candidates(path, pinned)
    doc = N.read(Path(path or D.CANDS_FILE))
    sha = N.file_sha(search_path)
    if sha != doc['search']['file_sha256']:
        raise N.Blocked(f'{search_path}: sha {sha[:12]} is not the pinned search {doc["search"]["file_sha256"][:12]}')
    fresh = B.select(json.loads(search_path.read_text(encoding='utf-8')), sha)
    for k in ('candidates', 'diverse'):
        if json.dumps(fresh[k], sort_keys=True) != json.dumps(doc[k], sort_keys=True):
            raise N.Blocked(f'{k}: a fresh selection of {search_path.name} differs from the pinned document')
    return dict(search=str(search_path), file_sha256=sha, n_top=len(doc['candidates']),
                n_diverse=len(doc['diverse']['candidates']))


def prepare(final_root, cands=None):
    from tools.coopmwm import a37_cellbank as C
    cands = diverse_candidates() if cands is None else cands
    if not cands:
        return dict(path=None, families=[], note='no diverse candidates (nothing to verify)')
    doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    p = D.C37.write_once(D.bank_path(final_root, CANDS_BANK), D.C37.bank_text(doc))
    return dict(path=str(p), file_sha256=N.file_sha(p), families=[c['family'] for c in cands])


def u0_wave(panel, final_root):
    """The shared A42 verification U0 wave (A42V_U0), never rerun by A45."""
    return D.verify_waves(panel, final_root, check_files=False, cands=[])[0]


def verify_waves(panel, final_root, *, check_files=True, cands=None):
    """[(wave, 'episode', specs)]: every diverse candidate (diverse order) on the 24-world block; no U0 wave."""
    cands = diverse_candidates() if cands is None else cands
    if not cands:
        return []
    entry = D._bank_entry(final_root, CANDS_BANK, check_files=check_files) or dict(path='', file_sha256='')
    return [(f'{VERIFY_WAVE_PREFIX}{c["family"]}', 'episode',
             D._a37_specs(D.RUNNER_ARM['PLAN'], panel, final_root, entry, family=c['family'],
                          tag=f'{VERIFY_TAG}_{c["key"][:8]}'))
            for c in cands]


def admission(final_root):
    p = Path(final_root)/ADMISSION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def build_diverse_bank(top_cands, top_verdicts, div_cands, div_verdicts):
    """Pure: DIVERSE_BANK plans = every admitted plan of the top 6 + diverse (A37 admitted_plans order), or None when
    no diverse plan is admitted (DIVERSE_BANK is not built)."""
    from tools.coopmwm import a37_cellbank as C
    div_admitted = C.admitted_plans(div_cands, div_verdicts)
    if not div_admitted:
        return None
    return C.admitted_plans(list(top_cands)+list(div_cands), dict(top_verdicts, **div_verdicts))


def admit(final_root, panel, cands=None, top=None):
    """A37 UNC rule per diverse candidate (U0 = the shared A42V_U0 episodes) -> ADMISSION_A45 (sealed) + DIVERSE_BANK."""
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    cands = diverse_candidates() if cands is None else cands
    top = D.candidates() if top is None else top
    if admission(final_root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    adm42 = D.admission(final_root)
    if adm42 is None:
        raise N.Blocked(f'{D.LABEL} ADMISSION missing ({LABEL} admits after the {D.LABEL} admission)')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources = {}, {}
    name, _, specs = u0_wave(panel, final_root)
    have['U0'], sources['U0'] = D.C37._load(final_root, name, specs)
    for (name, _, specs), c in zip(verify_waves(panel, final_root, cands=cands), cands):
        have[c['family']], sources[c['family']] = D.C37._load(final_root, name, specs)
    reports = D.C37._reports(final_root, have, worlds)
    verdicts, rows = {}, []
    specs_by_family = D._spec_map(list(top)+list(cands))
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, 'U0', bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have['U0'][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a37'] for w in worlds]
        v.update(family=fam, wm_rank=cand['wm_rank'], diverse_rank=cand['diverse_rank'], key=cand['key'],
                 class_key=cand['class_key'], structural_class=cand['structural_class'],
                 n_roles=len(cand['strategy']['roles']), cvar90_vs_u0=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_u0_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 realised_group_size=D._group_sizes(have[fam], worlds, specs_by_family),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        verdicts[fam] = v
        rows.append(v)
    top_verdicts = {r['family']: r for r in adm42['verdicts']}
    plans = build_diverse_bank(top, top_verdicts, cands, verdicts)
    bank = None
    if plans is not None:
        p = D.C37.write_once(D.bank_path(final_root, DIVERSE_BANK_FILE), D.C37.bank_text(C.bank_doc(plans, CELL)))
        bank = dict(path=f'{D.HOST_ROOT}/{D.BANK_DIR}/{p.name}', file_sha256=N.file_sha(p))
    div_fams = {c['family'] for c in cands}
    value = N.seal(dict(schema=f'{SCHEMA_PREFIX}.admission.v1', amendment=AMENDMENT, cell=CELL, rule=C.UNC_RULE,
                        multiplicity=f'{len(rows)} diverse picks each tested at 95% (disclosed, as A37/A42)',
                        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), verdicts=rows,
                        u0_shared_from=dict(wave=u0_wave(panel, final_root)[0], receipt=sources['U0'].get('receipt_sha256')),
                        a42_admission=dict(receipt_sha256=adm42['receipt_sha256'], admitted=adm42['admitted']),
                        admitted_diverse=[f for f in (p['family'] for p in plans or []) if f in div_fams],
                        diverse_bank_families=[p['family'] for p in plans or []],
                        priority=[(p['family'], p['lb95_veh_s']) for p in plans or []],
                        admitted_classes={v['family']: v['class_key'] for v in rows if v['PASS']},
                        diverse_bank=bank, diverse_bank_built=plans is not None,
                        consequence=NOT_BUILT if plans is None else 'confirmation runs DIVERSE_BANK on the 96 worlds',
                        sources={k: v.get('receipt_sha256') for k, v in sources.items()}),
                   'receipt_sha256')
    N.publish(final_root/ADMISSION_FILE, value)
    return value


def confirm_specs(panel, final_root, *, check_files=True):
    adm = admission(final_root)
    if adm is None:
        raise N.Blocked(f'no {ADMISSION_FILE} (run {LABEL} verify + admit first)')
    if not adm['diverse_bank_built']:
        return []
    entry = D._bank_entry(final_root, DIVERSE_BANK_FILE, check_files=check_files)
    if entry is not None and entry['file_sha256'] != adm['diverse_bank']['file_sha256']:
        raise N.Blocked('DIVERSE_BANK bank differs from the A45 admission receipt')
    entry = entry or dict(adm['diverse_bank'])
    return D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, final_root, entry, tag=f'{CONFIRM_TAG}_diverse_bank')


CONFIRM_WAVE = f'A45C_{ARM}'


def stage_waves(command, setting, panel, root, *, check_files=True):
    if command == 'verify':
        if setting != D.VERIFY_SETTING:
            raise N.Blocked(f'verify runs on {D.VERIFY_SETTING}')
        return verify_waves(panel, root, check_files=check_files)
    if command == 'confirm':
        if setting != D.CONFIRM_SETTING:
            raise N.Blocked(f'confirm runs on {D.CONFIRM_SETTING}')
        s = confirm_specs(panel, root, check_files=check_files)
        return [(CONFIRM_WAVE, 'episode', s)] if s else []
    raise N.Blocked(command)


# ------------------------------------------------------------------------------------------------ runtime

class A45Runtime(N.PrerunRuntime):
    """The pre-run plant of the A42 final root, own ledger phase divp60_a45 (cap 480,000; never bigp60_a42)."""
    phase = A45_PHASE

    def check_capacity(self, amount):
        if charged_phase(self.final_root)+amount > A45_CAP_SIM_S:
            raise N.Blocked(f'{LABEL}: {A45_PHASE} cap {A45_CAP_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        if any(not str(j['id']).startswith(JOB_PREFIX) for j in jobs):
            raise N.Blocked(f'{LABEL}: job ids are {JOB_PREFIX}*')
        self.check_capacity(sum(j['sim_s'] for j in jobs))
        N.own_reserve(self.final_root, jobs, A45_PHASE, cap=A45_CAP_SIM_S, calls=self.calls,
                      allocation=TOTAL_ALLOCATION_SIM_S)


def stage_fits(final_root, waves):
    """Before a stage: the remaining A45 cap must hold every episode of the stage (no partial panel)."""
    need = sum(len(s) for _, _, s in waves)*N.HORIZON_S
    left = A45_CAP_SIM_S-charged_phase(final_root)
    if need > left:
        raise N.Blocked(f'{LABEL}: stage needs {need} sim-s, {left} left in {A45_PHASE} (MAIN decides)')
    return dict(need=need, left=left)


def run_stage(final_root, waves, *, workers, chunk, log, panel, runtime=None):
    final_root = Path(final_root)
    if (final_root/STOP_A45).exists():
        log(dict(stopped=True, file=STOP_A45))
        return dict(status='STOPPED')
    stage_fits(final_root, waves)
    runtime = runtime or A45Runtime(N.Runtime(N.RUNNER), final_root)
    return D.execute_with_retry(final_root, waves, workers=workers, chunk=chunk, log=log, panel=panel, runtime=runtime)


# ------------------------------------------------------------------------------------------------ analysis

def analyse(final_root, panel, cands=None, top=None):
    """A45 reading: DIVERSE_BANK - DUO_HOLD total (CI upper < 0) AND the group rule vs U0; DIVERSE_BANK - BIG_BANK
    (BIG_BANK := U0 when A42 admitted nothing) and the class of each admitted diverse plan. One panel, never pooled."""
    import numpy as np
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    adm = admission(final_root)
    if adm is None:
        raise N.Blocked(f'{ADMISSION_FILE} missing')
    base = dict(schema=f'{SCHEMA_PREFIX}.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                admission=dict(receipt_sha256=adm['receipt_sha256'], admitted_diverse=adm['admitted_diverse'],
                               admitted_classes=adm['admitted_classes'], diverse_bank=adm['diverse_bank'],
                               diverse_bank_families=adm['diverse_bank_families']))
    if not adm['diverse_bank_built']:
        return N.seal(dict(base, primary=dict(rule=PRIMARY_RULE, supported=False, verdict='NOT SUPPORTED',
                                              note=NOT_BUILT)), 'analysis_sha256')
    cands = diverse_candidates() if cands is None else cands
    top = D.candidates() if top is None else top
    adm42 = D.admission(final_root)
    big_is_u0 = bool(adm42 is None or adm42['big_bank_is_u0'])
    have, sources = {}, {}
    for arm in ('U0', 'DUO_HOLD') + (() if big_is_u0 else ('BIG_BANK',)):
        have[arm], sources[arm] = D.C37._load(final_root, D.confirm_wave(arm),
                                              D.confirm_specs(arm, panel, final_root, check_files=False))
    have[ARM], sources[ARM] = D.C37._load(final_root, CONFIRM_WAVE, confirm_specs(panel, final_root, check_files=False))
    reports = D.C37._reports(final_root, have, worlds)
    big_ref = 'U0' if big_is_u0 else 'BIG_BANK'
    contrasts = [D._contrast(have, reports, worlds, ARM, 'DUO_HOLD', bootstrap, 'primary_part_a'),
                 D._contrast(have, reports, worlds, ARM, 'U0', bootstrap, 'primary_part_b_groups'),
                 dict(D._contrast(have, reports, worlds, ARM, big_ref, bootstrap, 'secondary_vs_big_bank'),
                      big_bank_is_u0=big_is_u0,
                      note=('A42 admitted nothing: BIG_BANK := U0, so DIVERSE_BANK - BIG_BANK = DIVERSE_BANK - U0'
                            if big_is_u0 else 'DIVERSE_BANK - BIG_BANK on the same 96 worlds'))]
    prim = dict(D.primary_verdict(contrasts[0], contrasts[1]), rule=PRIMARY_RULE)
    div_by_fam = {c['family']: c for c in cands}
    arm = dict(mean_J_veh_s=float(np.mean([have[ARM][w]['J_veh_s'] for w in worlds])),
               safety_totals={k: int(sum(have[ARM][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
               realised_group_size=D._group_sizes(have[ARM], worlds, D._spec_map(list(top)+list(cands))))
    fam = {}
    for w in worlds:
        for f, n in ((have[ARM][w].get('a37') or {}).get('activations_by_family') or {}).items():
            fam[f] = fam.get(f, 0)+int(n)
    arm['activations_by_family'] = fam
    per_plan = [dict(family=f, class_key=div_by_fam[f]['class_key'], structural_class=div_by_fam[f]['structural_class'],
                     wm_rank=div_by_fam[f]['wm_rank'], diverse_rank=div_by_fam[f]['diverse_rank'],
                     activations_in_diverse_bank=fam.get(f, 0))
                for f in adm['admitted_diverse']]
    return N.seal(dict(base, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       primary=prim, contrasts=contrasts, arm=arm, admitted_diverse_plans=per_plan,
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
                       label=RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('check-search', 'prepare', 'dry', 'verify', 'admit', 'confirm', 'reports',
                                       'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--setting', default=D.VERIFY_SETTING, choices=D.SETTINGS)
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=46)
    p.add_argument('--search')
    p.add_argument('--candidates')
    p.add_argument('--out')
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if a.command == 'check-search':
        if not a.search:
            p.error('--search required')
        print(json.dumps(check_search(a.search, a.candidates)))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    panel = D.setup(a.setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=a.setting, amendment=AMENDMENT),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root), indent=1))
    elif a.command == 'dry':
        c = 'verify' if a.setting == D.VERIFY_SETTING else 'confirm'
        out = {'setting': a.setting, 'budget': budget_plan(), f'charged_{JOB_PREFIX}': charged_phase(root)}
        try:
            ws = stage_waves(c, a.setting, panel, root)
        except N.Blocked as e:
            ws, out['note'] = [], str(e)
        out[c] = {n: dict(n=len(s), first=s[0]['job_id'], seeds=[s[0]['world']['seed'], s[-1]['world']['seed']],
                          arm=s[0]['arm']) for n, _, s in ws}
        out[c+'_sim_s'] = sum(len(s)*N.HORIZON_S for _, _, s in ws)
        print(json.dumps(out, indent=1))
    elif a.command in ('verify', 'confirm'):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = stage_waves(a.command, a.setting, panel, root)
            if not waves:
                log(dict(done=a.command, status='SKIPPED_NOTHING_TO_RUN'))
                return
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), fits=stage_fits(root, waves), **{f'charged_{JOB_PREFIX}_before': charged_phase(root)}))
            res = run_stage(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'admit':
        if a.setting != D.VERIFY_SETTING:
            p.error(f'admit runs on {D.VERIFY_SETTING}')
        v = admit(root, panel)
        print(json.dumps(dict(admitted_diverse=v['admitted_diverse'], diverse_bank_built=v['diverse_bank_built'],
                              admitted_classes=v['admitted_classes'], receipt_sha256=v['receipt_sha256'])))
    elif a.command == 'analyse':
        if a.setting != D.CONFIRM_SETTING:
            p.error(f'analyse runs on {D.CONFIRM_SETTING}')
        v = analyse(root, panel)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], verdict=v['primary']['verdict'])))
    elif a.command == 'status':
        out = {D.A42_PHASE: charged_phase(root, D.A42_PHASE), A45_PHASE: charged_phase(root),
               f'{JOB_PREFIX}_cap': A45_CAP_SIM_S}
        for c in ('verify', 'confirm'):
            try:
                for n, _, s in stage_waves(c, a.setting, panel, root, check_files=False):
                    out[n] = f'{N.published(root, n, s, D.C37.check_a37)[2]}/{len(s)}'
            except N.Blocked:
                pass
        print(json.dumps(out, indent=1))
    else:   # reports
        n = 0
        for c in ('verify', 'confirm'):
            try:
                waves = stage_waves(c, a.setting, panel, root, check_files=False)
            except N.Blocked:
                continue
            for name, _, specs in waves:
                recs, _, _ = N.published(root, name, specs, D.C37.check_a37)
                for r in recs:
                    rep = N.episode_log_report(root, r)
                    if rep is None or not rep['ok']:
                        raise N.Blocked(f'{r["job_id"]}: report failed')
                    n += 1
        print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
