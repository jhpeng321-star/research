"""AMENDMENT 99 follow-on (A99 PANEL, POST-HOC, exploratory): the matched 20 % panel 984400-984591 (D2800_p20, 192
worlds, panel 6c161a56, file 0bb84f3c = the A96(c) staged panel, byte copy), EVERY plan arm on MAPPO20C 606e59e0.

Library: ops/a96_p20.py (A96 parts c/d driver b29c45d4, imported UNCHANGED, part c names/block/setting code); this
module re-points it process-locally:
  BANK_S4_20C    runner arm A75_BANK on the A99 S4-p20 bank 682c01ae (/root/a99adm_s4, admitted on MAPPO20C), specs as
                 a76_d2p20.confirm_waves builds A76C_BANK, comparator = the A77 MAPPO20C binding of the A96 reference
                 root (rebind: the six executor fields; = the A99 admission construction)          SECONDARY row
  BANK_HAND_20C  same builder on the A99 hand bank 59acb747 (/root/a99adm_hand, 3/10 admitted on MAPPO20C)
  DUO_HOLD_20C   a32_ablation.a32_specs('DUO_HOLD') (duo12_v20) with the six executor fields replaced (A67 with_exec rule)
  MAPPO20C       U0 specs through A52.with_executor with the same binding (= the A96V_MAPPO20C reference construction)
  BANK_S5_20C    NOT RUN: S5-p20 admitted 0/12 on MAPPO20C (no bank file) -> reported as "empty"
Mechanics (gate, no extra worlds): the first 2 panel worlds (984400/984401) of every arm run first (8 episodes, 9,600
sim-s, part of the 921,600 panel budget); A99P_MECH.json PASS iff every record loaded executor 606e59e0, the bank arms
pass the A96 per-plan rule checks (a80_s2b.plan_checks, zone +-1 m, a75 d1af7e0f, no CAV in two sessions, no refusal,
0 collisions / teleports) and DUO_HOLD_20C carries its a32 receipt. Then confirm runs the remaining 760 episodes.
Analysis: every plan arm - MAPPO20C; BANK_S4_20C - BANK_HAND_20C; BANK_S4_20C - DUO_HOLD_20C; BANK_HAND_20C -
DUO_HOLD_20C; paired percentile bootstrap 10,000 seed 20261104, diff = treatment - reference (negative = less delay),
rule per contrast = CI95 upper < 0 AND five-group <= +10 s (nm_finaleval._bank_compare groups.all_pass); CVaR90,
hard brakes, TTC, collisions, teleports, activations. No registered primary (the registered S5 pick is empty).
Fixes carried: nm_finaleval cache fix fa98b552 (tree = the A99 admission tree); every histogram dict that is sealed
uses STRING keys (the A98 int-key seal bug: overlap_stats / engaged_hist re-wrapped, own analyse_core).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

S4_BANK_SHA256 = '682c01ae1deb28c774701ff0a5102cd77c2addf2ca302e4e42a90409b7302ab2'
HAND_BANK_SHA256 = '59acb7471c51da728ffe7e1f0600082a8e6486a0455108a21df41d7b851bd946'
if os.environ.get('A96_PART') != 'c':
    raise SystemExit('A96_PART must be c (the A99 panel reuses the part (c) block / setting code)')
os.environ['A96_HAND_SHA256'] = HAND_BANK_SHA256          # process-local: a96_p20 import-time check only

import a96_p20 as M  # noqa: E402  (unchanged)

M76, D2 = M.M76, M.D2
AMENDMENT = '99'
CELL = M.CELL
HORIZON = M.HORIZON
REFROOT = Path(os.environ.get('A99P_REF_ROOT', '/root/a96ref/root'))
BANK_SRC = {'BANK_S4_20C': (Path('/root/a99adm_s4/root/a96_banks/A96_BANK_D2800_p20.json'), S4_BANK_SHA256),
            'BANK_HAND_20C': (Path('/root/a99adm_hand/root/a96_banks/A96_BANK_D2800_p20.json'), HAND_BANK_SHA256)}
OMITTED = {'BANK_S5_20C': 'empty: S5-p20 admitted 0/12 on MAPPO20C (/root/a99adm_s5 wrote no bank file); not run'}
MAPPO20C_SHA256 = '606e59e0f579a4c7c3be88b44da2237ce9ddbdc065780dbd4b385128775ad659'
MAPPO20C_BINDING_SHA256 = '466bfcbfc3d4e0fad1731271008ece23bc469abd68a07400c7fc4825550afac8'
FROZEN_U0_SHA256 = M.FROZEN_U0_SHA256
PANEL_SHA256 = '6c161a56e1c913e35441570774f1df6ee62999a0ca9e11b27a309ab3b0dfc5ff'
PANEL_FILE_SHA256 = '0bb84f3c129ca7347236337cb4bcc316b447a0186f6478fa2ca9451646145c56'
N_WORLDS = 192
EXEC_FIELDS = (('comparator_kind', 'kind'), ('label', 'label'), ('executor_path', 'path'),
               ('executor_sha256', 'sha256'), ('binding_sha256', 'binding_sha256'), ('binding_path', 'binding_path'))

S4, HAND, DUO, REF = 'BANK_S4_20C', 'BANK_HAND_20C', 'DUO_HOLD_20C', 'MAPPO20C'
BANK_ARMS = (S4, HAND)
ARMS = (S4, HAND, DUO, REF)
PLAN_ARMS = (S4, HAND, DUO)
VS_REF = tuple((a, REF) for a in PLAN_ARMS)
BETWEEN = ((S4, HAND), (S4, DUO), (HAND, DUO))
MECH_N = 2
MECH_CAP = len(ARMS)*MECH_N*HORIZON                     # 9,600 (inside the panel budget: same episodes)
TOTAL_CAP = len(ARMS)*N_WORLDS*HORIZON                  # 921,600
WAVE = 'A99PC_'

AUTHORITY = ('MAIN AMENDMENT 99 follow-on (A99 PANEL, 2026-11-14; POST-HOC, exploratory): the matched 20 % panel '
             '984400-984591 (D2800_p20) with every plan arm on MAPPO20C 606e59e0 (A77 iteration 84): BANK_S4_20C '
             '(secondary, not the registered pick), BANK_HAND_20C, DUO_HOLD_20C vs MAPPO20C alone; BANK_S5_20C empty')
DISCLOSURE = ('POST-HOC and exploratory (Amendment 99 follow-on). The registered S4/S5 pick (A97 rule, A98) is S5, whose '
              'MAPPO20C screen admitted 0/12, so the registered matched arm is EMPTY and not run; BANK_S4_20C is a '
              'disclosed SECONDARY row (S4 is not the registered pick). Banks from uncorrected 24-world screens on '
              'reused worlds 988900-988923 (best admitted per form); hand-written plans are the search seeds; '
              'MAPPO20C is a single training run; all contrasts uncorrected; never pooled. File/wave/job names '
              'a99p*/A99P* are this package; the setting/block code and runner row A96C_RANGES are the A96(c) ones')

# ------------------------------------------------------------------------------------------------ re-point a96_p20
M.AMENDMENT = AMENDMENT
M.PHASE = 'a99panel_p20'
M.SETTING = 'a99p_D2800_p20'
M.STAGE = 'P20A99'
M.AUTHORITY, M.DISCLOSURE = AUTHORITY, DISCLOSURE
M.PREFIX = dict(i='a99pi_unused_', c='a99pc_', v='a99pv_unused_', m='a99pm_unused_')
M.PANEL_FILE, M.PANEL_PIN = 'A99P_PANEL_C.json', 'A99P_PANEL_C.sha256'
M.BANK_DIR = 'a99p_banks'
M.BANK_FILE_OF = {S4: f'A99P_BANK_S4_ON20C_{CELL}.json', HAND: f'A99P_BANK_HAND_ON20C_{CELL}.json'}
M.BANK_ENV = {S4: ('A99P_S4_BANK_FILE', 'A99P_S4_BANK_SHA256'), HAND: ('A99P_HAND_BANK_FILE', 'A99P_HAND_BANK_SHA256')}
M.BANK_PIN = 'A99P_BANKS.json'
M.MECH_FILE = 'A99P_MECH.json'
M.RESULTS_FILE = f'A99P_RESULTS_{CELL}.json'
M.LOG_FILE = 'A99P_LOG.jsonl'
M.STOP_FILE = 'STOP_A99P'
M.DROP_MANIFEST = 'A99P_DROPPED_RAW.jsonl'
M.DROP_RULE = D2.DROP_RULE.replace('a75d2v_* / a75d2c_*', M.PREFIX['c']+'*')
M.BANK_ARMS, M.ARMS, M.RULE_ARMS, M.HAND_RUN = BANK_ARMS, ARMS, BANK_ARMS, True
M.TAG = {S4: 'bank_s4_20c', HAND: 'bank_hand_20c', DUO: 'duo_hold_20c', REF: 'mappo20c'}
M.MECH_CAP = MECH_CAP

_OV, _ENG = M.overlap_stats, M.engaged_hist


def overlap_stats(r, horizon=HORIZON):
    """(A91 definition, = a96_p20.overlap_stats) with STRING keys in seconds_by_n_plans_active (A98 seal fix)."""
    d = _OV(r, horizon)
    d['seconds_by_n_plans_active'] = {str(k): v for k, v in sorted(d['seconds_by_n_plans_active'].items())}
    return d


def engaged_hist(r, horizon=HORIZON):
    """{k: seconds with exactly k distinct CAVs in a plan session (any plan), k >= 1}; keys are strings (A98 fix)."""
    return {str(k): v for k, v in sorted(_ENG(r, horizon).items())}


M.overlap_stats, M.engaged_hist = overlap_stats, engaged_hist


def _N():
    return M._N()


def core(n):
    """The a42 driver code with the A99P names; ledger cap = the 4-arm panel budget (mech episodes are panel episodes)."""
    N = _N()
    import a50_fullp60 as A50
    if n != N_WORLDS:
        raise N.Blocked(f'panel size {n} != {N_WORLDS}')
    over = dict(
        A42_PHASE=M.PHASE, A42_CAP_SIM_S=TOTAL_CAP, STOP_A42=M.STOP_FILE, HOST_BASE=M.host_base(), CELL=CELL,
        VERIFY_SETTING=M.SETTING, CONFIRM_SETTING=M.SETTING, BANK_DIR=M.BANK_DIR, IDENT_WAVE='A99P_IDENT_UNUSED',
        AMENDMENT=AMENDMENT, LABEL='A99P', JOB_PREFIX='a99p', VERIFY_TAG='a99pv_unused', CONFIRM_TAG='a99pc',
        IDENT_TAG='a99pi_unused', VERIFY_WAVE_PREFIX='A99PV_', CONFIRM_WAVE_PREFIX=WAVE, RESULTS_FILE=M.RESULTS_FILE,
        LOG_FILE=M.LOG_FILE, SCHEMA_PREFIX='coopmwm.a99p', RUNNER_RANGES_ATTR=M.RUNNER_ATTR,
        RESULT_LABEL='amendment 99 follow-on; POST-HOC exploratory 20 % panel, plan arms on MAPPO20C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a99_panel_core_{n}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (M.SETTING,)
    m.A42Runtime.phase = M.PHASE
    N.CHECKS['episode'] = D2.check_d2
    return m


M.core = core


# ------------------------------------------------------------------------------------------------ executor / arms

def mappo_bound():
    """The A77 MAPPO20C binding sealed in the A96 reference root (binding 466bfcbf, policy 606e59e0), read-only."""
    N = _N()
    b = M._A77().read_bound(REFROOT, 'MAPPO20C')
    if (b['sha256'] != MAPPO20C_SHA256 or b['binding_sha256'] != MAPPO20C_BINDING_SHA256
            or N.file_sha(b['path']) != MAPPO20C_SHA256):
        raise N.Blocked('reference-root MAPPO20C binding is not 466bfcbf / 606e59e0')
    return b


def mappo_comp():
    """= ops/a99_on20c.mappo_comp: the binding as an a76 comparator dict (rebind: the six executor fields replaced)."""
    b = mappo_bound()
    return dict(rebind=True, name='MAPPO20C', kind=b['kind'], label=b['label'], path=b['path'], sha256=b['sha256'],
                binding_sha256=b['binding_sha256'], binding_path=b['binding_path'], export=b.get('export'))


def with_exec6(specs, exe):
    """A67 with_exec rule: u0-executor hold specs (kind a32) with ONLY the six executor fields replaced by ``exe``."""
    out = []
    for s in specs:
        ck = deepcopy(s['checkpoint'])
        if ck.get('executor_sha256') != FROZEN_U0_SHA256 or ck.get('kind') != 'a32':
            raise _N().Blocked(f'{s["job_id"]}: not a u0 a32 spec')
        ck.update({k: exe[f] for k, f in EXEC_FIELDS})
        out.append(dict(s, checkpoint=ck))
    return out


def arm_specs(D, arm, panel, root, comp_u0, tag):
    exe = mappo_comp()
    if arm in BANK_ARMS:
        return M.bank_specs(D, panel, root, M.BANK_FILE_OF[arm], tag, exe)
    if arm == DUO:
        return with_exec6(M.duo_specs(panel, root, tag), exe)
    if arm == REF:
        return M._A52().with_executor(M76.u0_specs(panel, root, comp_u0, tag), mappo_bound(), tag)
    raise _N().Blocked(f'no A99P arm {arm!r}')


def u0_arm_specs(D, arm, panel, root, comp_u0, tag):
    """The same arm on the frozen U0 executor (the A96(c) construction) - dry spec diff only."""
    if arm in BANK_ARMS:
        return M.bank_specs(D, panel, root, M.BANK_FILE_OF[arm], tag, comp_u0)
    if arm == DUO:
        return M.duo_specs(panel, root, tag)
    return M76.u0_specs(panel, root, comp_u0, tag)


def confirm_waves(D, panel, root):
    N = _N()
    comp = M.comparator(root)                    # frozen U0 d9042cc5: base specs only
    exe = mappo_comp()
    out = [(f'{WAVE}{arm}', 'episode', arm_specs(D, arm, panel, root, comp, f'{M.PREFIX["c"]}{M.TAG[arm]}'))
           for arm in ARMS]
    n = len(panel['worlds'])
    if any(len(s) != n for _, _, s in out) or len({s['job_id'] for _, _, ss in out for s in ss}) != len(ARMS)*n:
        raise N.Blocked('confirmation waves: wrong size / duplicate job ids')
    for w, _, ss in out:
        for s in ss:
            if any(s['checkpoint'].get(k) != exe[f] for k, f in EXEC_FIELDS):
                raise N.Blocked(f'{w} {s["job_id"]}: executor fields are not MAPPO20C')
    return out


M.arm_specs, M.confirm_waves = arm_specs, confirm_waves


def mech_waves(D, panel, root):
    return [(w, k, s[:MECH_N]) for w, k, s in confirm_waves(D, panel, root)]


# ------------------------------------------------------------------------------------------------ zero SUMO stages

def stage_panel(base):
    """Write-once: the A96(c) staged 192-world panel (byte copy in $BASE/panels, file 0bb84f3c) into the root."""
    N = _N()
    src = Path(base)/'panels'/f'A96C_PANEL_C_{N_WORLDS}.json'
    if N.file_sha(src) != PANEL_FILE_SHA256:
        raise N.Blocked(f'{src}: not the A96(c) panel file {PANEL_FILE_SHA256[:8]}')
    return M.panel_in_root(Path(base)/'panels', Path(base)/'root', N_WORLDS, PANEL_SHA256)


def prepare(D, root):
    """Write-once byte copies of the two frozen A99 banks + sealed A99P_BANKS.json (zero SUMO)."""
    N = _N()
    import a37_celltailor as C37
    from tools.coopmwm import a75_holdforms as H
    root = Path(root)
    out = {}
    for arm in BANK_ARMS:
        src, want = BANK_SRC[arm]
        if N.file_sha(src) != want:
            raise N.Blocked(f'{src}: sha256 {N.file_sha(src)} != {want}')
        q = D.bank_path(root, M.BANK_FILE_OF[arm])
        if q.exists():
            if N.file_sha(q) != want:
                raise N.Blocked(f'{q}: exists with other bytes')
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            C37.write_once(q, src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != want:
            raise N.Blocked(f'{q}: not a byte copy')
        doc = json.loads(q.read_text(encoding='utf-8'))
        got, refused = H.a75_bank_plans(doc, CELL)
        if not got or (refused and any(refused.values())) or sorted(p['family'] for p in got) != \
                sorted(r['family'] for r in doc['rows'] if r.get('regime') == CELL):
            raise N.Blocked(f'{arm}: the executor runs {len(got)} plans, refused {refused}')
        core_ = H.make_core('A75_BANK', CELL, doc)
        out[arm] = dict(file_sha256=want, source=str(src), what=arm, families=[p['family'] for p in got],
                        executor_priority=[(p['family'], p['lb95'], type(c).__name__)
                                           for p, c in zip(core_.plans, core_.cores)])
    pin = root/M.BANK_PIN
    v = N.seal(dict(schema='coopmwm.a99p.banks.v1', amendment=AMENDMENT, part='c', hand_run=True, banks=out,
                    omitted=OMITTED), 'receipt_sha256')
    if pin.is_file():
        old = N.verify(N.read(pin), 'receipt_sha256', pin.name)
        if old['banks'] != json.loads(json.dumps(out)):
            raise N.Blocked(f'{pin.name}: pinned with other banks')
        return old
    N.publish(pin, v)
    return v


def runner_loaded(R, s):
    """What the runner will load for this spec, by its own pure checks (zero physics): the bound u0 residual and
    nm_eval_runner.bound_executor (binding seal, spec = binding, file sha) -> loaded executor sha256."""
    ck = s['checkpoint']
    p = Path(ck['path'])
    if p.name != 'A_checkpoint_000.pt' or _N().file_sha(p) != ck['file_sha256']:
        raise _N().Blocked(f'{s["job_id"]}: bound u0 residual changed')
    return R.bound_executor(ck)[1]


def dry(root):
    N = _N()
    D, panel = M.setup(root)
    R = M._runner()
    banks = M.bank_pinned(root)
    ws = confirm_waves(D, panel, root)
    comp = M.comparator(root)
    arms = {}
    for (w, _, ss), arm in zip(ws, ARMS):
        loaded = Counter(runner_loaded(R, s) for s in ss)
        base = u0_arm_specs(D, arm, panel, root, comp, f'{M.PREFIX["c"]}{M.TAG[arm]}')
        diff = sorted({k for x, y in zip(ss, base) for k in M.spec_diff(x, y)})
        arms[arm] = dict(wave=w, n=len(ss), runner_arm=ss[0]['arm'], kind=ss[0]['checkpoint']['kind'],
                         jobs=[ss[0]['job_id'], ss[-1]['job_id']], seeds=[ss[0]['world']['seed'], ss[-1]['world']['seed']],
                         executor_fields={k: ss[0]['checkpoint'][k] for k, _ in EXEC_FIELDS},
                         runner_loaded_executor=dict(loaded), all_loaded_mappo20c=set(loaded) == {MAPPO20C_SHA256},
                         spec_diff_vs_u0_construction=diff, sim_s=len(ss)*HORIZON,
                         mech_episodes=MECH_N, mech_seeds=[s['world']['seed'] for s in ss[:MECH_N]])
    worlds = [x['world']['seed'] for x in panel['worlds']]
    return dict(panel_sha256=panel['panel_sha256'], panel_file_sha256=N.file_sha(Path(root)/M.PANEL_FILE),
                n_worlds=len(worlds), seeds=[worlds[0], worlds[-1]], setting=M.SETTING, arms=arms, omitted=OMITTED,
                contrasts=contrasts(), banks={a: (banks['banks'][a]['file_sha256'], banks['banks'][a]['families'])
                                              for a in BANK_ARMS},
                bank_pin=banks['receipt_sha256'], sim_s_total=sum(a['sim_s'] for a in arms.values()),
                caps=dict(mech=MECH_CAP, total=TOTAL_CAP), charged=D.charged(root),
                a75_sha256=__import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest(),
                nm_finaleval_sha256=N.file_sha(HERE/'nm_finaleval.py'),
                runner_sha256=N.file_sha(Path(R.__file__)), runner_attr=[M.RUNNER_ATTR, list(getattr(R, M.RUNNER_ATTR))])


# ------------------------------------------------------------------------------------------------ mechanics gate

def mech_compare(root):
    N = _N()
    root = Path(root)
    D, panel = M.setup(root)
    fails, checks = [], {}
    for (wave, _, specs), arm in zip(mech_waves(D, panel, root), ARMS):
        recs, _, cov = N.published(root, wave, specs, D2.check_d2)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        have = {r['world_id']: r for r in recs}
        if arm in BANK_ARMS:
            fams, plans, zones = M.bank_plans_of(D, root, arm)
            checks[arm] = M.rule_rows(root, have, arm, fams, plans, zones, sorted(have), fails, exe=MAPPO20C_SHA256)
            checks[arm]['zones'] = {f: list(z) for f, z in zones.items()}
            continue
        rows = []
        for w in sorted(have):
            r, errs = have[w], []
            if r.get('executor_sha256_loaded') != MAPPO20C_SHA256:
                errs.append(f'executor {str(r.get("executor_sha256_loaded"))[:8]} is not MAPPO20C')
            if arm == DUO and not isinstance(r.get('a32'), dict):
                errs.append('no a32 receipt')
            rows.append(dict(world_id=w, seed=r['world'].get('seed'), executor=r.get('executor_sha256_loaded'),
                             errors=errs))
            if errs:
                fails.append(f'{arm} {w}: {errs}')
        checks[arm] = dict(rows=rows)
    if sorted(checks) != sorted(ARMS):
        fails.append(f'checks for {sorted(checks)}')
    rule = ('PASS iff on the first 2 panel worlds of every arm (984400 / 984401): every record loaded executor MAPPO20C '
            '606e59e0; each bank arm passes the A96 per-plan rule checks (a80_s2b.plan_checks per plan, activation '
            'inside that plan zone +-1 m, a75 d1af7e0f, no CAV in two sessions at once, no refusal, 0 collisions / '
            'teleports; a plan that never activates is reported, not a failure); DUO_HOLD_20C carries its a32 receipt. '
            'J / delay values are not read by this gate')
    return N.seal(dict(schema='coopmwm.a99p.mech.v1', amendment=AMENDMENT, rule=rule,
                       verdict='PASS' if not fails else 'FAIL', failures=fails, checks=checks,
                       consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


def mech_passed(root):
    N = _N()
    p = Path(root)/M.MECH_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ analysis

def contrasts():
    return [(a, r, 'vs_mappo20c') for a, r in VS_REF] + [(a, r, 'between_plan_arms') for a, r in BETWEEN]


def rule_verdict(row):
    """CI95 upper < 0 AND five-group rule vs the row's reference (the A96(b)/A99 rule); uncorrected, descriptive."""
    ci = row['total_minus_ref']['ci95']
    groups = row.get('groups') or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    ok = a_pass and b_pass
    return dict(contrast=f'{row["treatment"]} - {row["reference"]}', kind=row['kind'],
                diff=row['total_minus_ref']['diff'], ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'),
                delay_pass=a_pass, group_rule_pass=b_pass, verdict='SUPPORTED' if ok else 'NOT SUPPORTED',
                direction='TREATMENT_BETTER' if ok else ('REFERENCE_BETTER' if ci[0] > 0 else 'NOT_SEPARATED'),
                worst_10pct_cvar90=row.get('tail_cvar90'), hard_brakes_minus_ref=row.get('hard_brakes_minus_ref'),
                ttc_events_minus_ref=row.get('ttc_events_minus_ref'))


def analyse_core(D, have, reports, worlds, bootstrap, *, fams, meta):
    """= a96_p20.analyse_core with the A99P arms / contrasts and string-keyed histograms."""
    import numpy as np
    N = _N()
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = [D._contrast(have, reports, worlds, a, r, bootstrap, k) for a, r, k in contrasts()]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                    executors_loaded=dict(Counter(str(have[a][w].get('executor_sha256_loaded')) for w in worlds)))
            for a in have}
    for a in BANK_ARMS:
        by_fam, offers, tot, eng, nact, per_world = Counter(), Counter(), Counter(), Counter(), Counter(), {}
        for w in worlds:
            r = have[a][w].get('a75') or {}
            for f, n in (r.get('activations_by_family') or {}).items():
                by_fam[str(f)] += int(n)
            for k, n in (r.get('offers') or {}).items():
                offers[str(k)] += int(n)
            ov = overlap_stats(r)
            per_world[w] = ov['concurrent_s']
            for k in ('concurrent_s', 'acts_started_while_other_active', 'cav_in_two_sessions'):
                tot[k] += int(ov[k])
            for f, n in ov['active_s_by_family'].items():
                tot[f'active_s|{f}'] += int(n)
            for k, n in ov['seconds_by_n_plans_active'].items():
                nact[int(k)] += int(n)
            for k, n in engaged_hist(r).items():
                eng[int(k)] += int(n)
        arms[a].update(families=list(fams[a]), activation_counts=D2.form_activations(have[a], worlds),
                       activations_by_family=dict(by_fam), offers_by_family_outcome=dict(offers),
                       offers_by_plan=M.offers_by_plan(offers),
                       overlap=dict(tot, worlds_with_overlap=sum(1 for v in per_world.values() if v > 0),
                                    seconds_by_n_plans_active={str(k): v for k, v in sorted(nact.items())},
                                    definition=overlap_stats.__doc__.strip()),
                       cavs_engaged_at_once=dict(seconds_by_k={str(k): v for k, v in sorted(eng.items())},
                                                 max_k=max(eng) if eng else 0,
                                                 definition=engaged_hist.__doc__.strip()))
    arms[DUO]['activations_per_world'] = sum(
        int((have[DUO][w].get('a32') or have[DUO][w].get('oracle') or {}).get('activations') or 0)
        for w in worlds)/len(worlds)
    return N.seal(dict(schema='coopmwm.a99p.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, n_worlds=len(worlds), bootstrap=bootstrap, arms_run=list(ARMS),
                       arms_omitted=OMITTED, pooled_with=None, registered_primary=None,
                       primary_note=('no registered primary in this package: the registered matched arm (S5 bank on '
                                     'MAPPO20C) is empty; BANK_S4_20C is a disclosed secondary row'),
                       rule_verdicts=[rule_verdict(r) for r in rows], contrasts=rows, arms=arms,
                       families={k: list(v) for k, v in fams.items()},
                       multiplicity='all contrasts uncorrected (6 tests)', **meta), 'analysis_sha256')


def summary(v):
    return dict(analysis_sha256=v['analysis_sha256'],
                rules=[{k: x[k] for k in ('contrast', 'diff', 'ci95', 'delay_pass', 'group_rule_pass', 'verdict',
                                          'direction')} for x in v['rule_verdicts']],
                banks={a: dict(concurrent_s=v['arms'][a]['overlap']['concurrent_s'],
                               activations_by_family=v['arms'][a].get('activations_by_family')) for a in BANK_ARMS})


def analyse(D, panel, root):
    N = _N()
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != M.BOOTSTRAP or len(worlds) != N_WORLDS:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    have, sources, reports = {}, {}, {}
    for (wave, _, specs), arm in zip(confirm_waves(D, panel, root), ARMS):
        recs, src, cov = N.published(root, wave, specs, D2.check_d2)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        have[arm], sources[arm] = {r['world_id']: r for r in recs}, src
        reports[arm] = D2._own_reports(root, have[arm], worlds)
    banks = M.bank_pinned(root)
    fams = {a: tuple(banks['banks'][a]['families']) for a in BANK_ARMS}
    meta = dict(panel_sha256=panel['panel_sha256'],
                seeds=[panel['worlds'][0]['world']['seed'], panel['worlds'][-1]['world']['seed']], banks=banks,
                executors=dict(MAPPO20C=MAPPO20C_SHA256, binding=MAPPO20C_BINDING_SHA256, u0_base_specs=FROZEN_U0_SHA256),
                root=str(root), mech=N.read(root/M.MECH_FILE).get('receipt_sha256'),
                sources={k: v.get('receipt_sha256') for k, v in sources.items()}, label=D.RESULT_LABEL)
    return analyse_core(D, have, reports, worlds, bootstrap, fams=fams, meta=meta)


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('stage-panel', 'prepare', 'dry', 'mech', 'mech-compare', 'confirm', 'analyse',
                                       'drop-raw', 'status'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    N = _N()
    M._A77()
    M._A52()
    root = Path(a.final_root)
    if root.resolve() == REFROOT.resolve() or any(root.resolve() == s.parent.parent.resolve() for s, _ in BANK_SRC.values()):
        raise N.Blocked('the A99P root is a new root')
    if root.parent.as_posix() != M.host_base():
        raise N.Blocked(f'root {root} is not under A96_HOST_BASE {M.host_base()}')
    if a.command == 'stage-panel':
        print(json.dumps(stage_panel(root.parent)))
        return
    D, panel = M.setup(root)
    if panel['panel_sha256'] != PANEL_SHA256:
        raise N.Blocked('panel is not 6c161a56')
    if a.command == 'prepare':
        print(json.dumps(prepare(D, root), indent=1))
        return
    if a.command == 'dry':
        print(json.dumps(dry(root), indent=1, default=str))
        return

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=M.SETTING), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/M.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    if a.command in ('mech', 'confirm'):
        import fcntl
        if __import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest() != M.A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        if a.command == 'mech' and (root/M.MECH_FILE).is_file():
            raise N.Blocked(f'{M.MECH_FILE} exists: mechanics ran')
        if a.command == 'confirm' and not mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        M.bank_pinned(root)
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            full = confirm_waves(D, panel, root)
            ws = mech_waves(D, panel, root) if a.command == 'mech' else full
            pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in ws)
            need, used = pending*HORIZON, D.charged(root)
            cap = MECH_CAP if a.command == 'mech' else TOTAL_CAP
            if used+need > cap:
                raise SystemExit(f'{a.command}: {used} + {need} > cap {cap}')
            drop = a.drop_raw and a.command == 'confirm'
            log(dict(start=a.command, n_waves=len(ws), episodes=sum(len(w[2]) for w in ws), pending=pending,
                     workers=a.workers, chunk=a.chunk, pid=os.getpid(), charged_before=used, drop_raw=drop))
            X = M.dropper() if drop else None

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    X.drop_raw(root, full, procs=a.drop_procs, log=log)
            N.CHECKS['episode'] = D2.check_d2
            res = D2.execute(D, root, ws, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'mech-compare':
        if (root/M.MECH_FILE).is_file():
            raise N.Blocked(f'{M.MECH_FILE} exists (one shot)')
        v = mech_compare(root)
        N.publish(root/M.MECH_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'])))
        if v['verdict'] != 'PASS':
            sys.exit(5)
    elif a.command == 'drop-raw':
        print(json.dumps(M.dropper().drop_raw(root, confirm_waves(D, panel, root), procs=a.drop_procs, log=log)))
    elif a.command == 'analyse':
        if (root/M.RESULTS_FILE).is_file():
            raise N.Blocked(f'{M.RESULTS_FILE} exists (one shot)')
        v = analyse(D, panel, root)
        N.publish(root/M.RESULTS_FILE, v)
        print(json.dumps(summary(v), default=str))
    else:
        ws = confirm_waves(D, panel, root)
        done = [N.published(root, w, s, D2.check_d2)[2] for w, _, s in ws]
        print(json.dumps(dict(charged=D.charged(root), caps=dict(mech=MECH_CAP, total=TOTAL_CAP), mech=mech_passed(root),
                              episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}'), indent=1))


if __name__ == '__main__':
    main()
