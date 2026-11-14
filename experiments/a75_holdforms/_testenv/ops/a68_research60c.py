"""AMENDMENT 68 (S-RESEARCH60C; MAIN 2026-11-12): new world-model plan search, then admission and confirmation in
SUMO on top of MAPPO60C (A65 policy_00084, 2abed317).

Stages (one final root, one ledger phase ``research60c_a68``, cap 1,070,400 sim-s = 4,800 + 374,400 + 691,200):
  select     zero SUMO: top K = 12 plans of the search `final` list (a42_bigbank.select_candidates, A42 skip rule)
  identity   integrity repeat for the changed code tree (table-only edits): MAPPO60C alone and old BIG_BANK@60C on
             the first 2 worlds of panel 998600-998695 must equal the stored A66 / A67 records exactly
  verify     24 fresh worlds 992500-992523: MAPPO60C alone + each candidate as a single unconditional plan on the
             A37 executor with only the six executor fields bound to MAPPO60C (A67.with_exec, unchanged)
  admit      per candidate (plan - MAPPO60C): delay test CI95 upper < 0 (a37_cellbank.unc_verdict bootstrap) and the
             five-group 10 s rule relative to MAPPO60C. MAIN = both; BACKUP = delay test only (secondary, never
             substituted). MAIN_BANK_REDUCED2 by the registered rule (reduce2). Bank priority: verified lb95 desc.
  confirm    fresh 96 worlds 996200-996295: MAPPO60C, DUO_HOLD_60C, OLD_BIG_BANK_60C (bank 7e7dc76b), MAIN_BANK_60C,
             MAIN_BANK_R2_60C, BACKUP_BANK_60C (only if it differs from MAIN). No plan in MAIN and none in BACKUP:
             nothing runs. MAIN empty, BACKUP not: the MAIN arms are not run, the primary is reported as null.
  analyse    primary MAIN_BANK_60C - MAPPO60C: SUPPORTED iff total delay CI95 upper < 0 AND the group rule relative to
             MAPPO60C; every other contrast descriptive. Paired percentile bootstrap 10,000, seed 20261104,
             diff = treatment - reference (negative = less delay).
No automatic replay: a failed dispatch raises and is reported. No executor / runner behaviour code is changed.

Commands (python ops/a68_research60c.py <cmd> --final-root R --a66-root R66 --a67-root R67 --a42-root R42):
  select --search F --out F ; prepare ; dry ; identity ; identity-compare ; verify ; admit ; confirm ; analyse ; status
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

AMENDMENT = '68'
CELL = 'D2800_p60'
PHASE = 'research60c_a68'
IDENT_N, VERIFY_N, CONFIRM_N, K = 2, 24, 96, 12
IDENT_CAP, VERIFY_CAP, CONFIRM_CAP = 4_800, 374_400, 691_200
CAP_SIM_S = IDENT_CAP+VERIFY_CAP+CONFIRM_CAP
SETTINGS = dict(i='a42c_D2800_p60', v='a68v_D2800_p60', c='a68c_D2800_p60')
RANGES = dict(i='A42_RANGES', v='A68_RANGES', c='A68_RANGES')
SEEDS = dict(i=tuple(range(998600, 998696)), v=tuple(range(992500, 992524)), c=tuple(range(996200, 996296)))
PANEL_SHA256 = dict(v='989248f9d449b234724d74cd569e324e1be7bc1038529947429f336652412ea4',
                    c='37197aa25e53d98efed3d5b8886c0748c7f0f35a6dc258632b11af5a9270baa4')
A68_DIR = HERE/'a68'
CANDS_FILE = A68_DIR/f'CANDIDATES_{CELL}.json'
CANDS_PIN = A68_DIR/f'CANDIDATES_{CELL}.sha256'        # written after select, recorded in REGISTRATION.md
BANK_DIR = 'a68_banks'
CANDS_BANK = f'A68_CANDS_{CELL}.json'
MAIN_BANK = f'A68_MAIN_BANK_{CELL}.json'
BACKUP_BANK = f'A68_BACKUP_BANK_{CELL}.json'
R2_BANK = f'A68_MAIN_BANK_R2_{CELL}.json'
OLD_BANK = f'A42_BIG_BANK_{CELL}.json'
OLD_BANK_SHA256 = '7e7dc76b8a0cd5c158205a0e3f6ef96d73a36a483c46d03ac422428dc8c573ff'
ADMISSION_FILE = f'A68_ADMISSION_{CELL}.json'
IDENTITY_FILE = 'IDENTITY_A68.json'
RESULTS_FILE = f'A68_RESULTS_{CELL}.json'
REF = 'MAPPO60C'
DUO_BANK_CELL = 'D2800_p20'
#: confirmation arm -> bank file of the CELL_BANK runner arm (None: not a bank arm)
BANK_OF = dict(OLD_BIG_BANK_60C=OLD_BANK, MAIN_BANK_60C=MAIN_BANK, MAIN_BANK_R2_60C=R2_BANK,
               BACKUP_BANK_60C=BACKUP_BANK)
CONFIRM_ORDER = ('MAPPO60C', 'DUO_HOLD_60C', 'OLD_BIG_BANK_60C', 'MAIN_BANK_60C', 'MAIN_BANK_R2_60C',
                 'BACKUP_BANK_60C')
PRIMARY = ('MAIN_BANK_60C', 'MAPPO60C')
SECONDARY = (('MAIN_BANK_60C', 'DUO_HOLD_60C'), ('MAIN_BANK_60C', 'MAIN_BANK_R2_60C'),
             ('MAIN_BANK_60C', 'OLD_BIG_BANK_60C'), ('BACKUP_BANK_60C', 'MAPPO60C'),
             ('BACKUP_BANK_60C', 'DUO_HOLD_60C'), ('BACKUP_BANK_60C', 'MAIN_BANK_60C'),
             ('BACKUP_BANK_60C', 'OLD_BIG_BANK_60C'), ('MAIN_BANK_R2_60C', 'MAPPO60C'),
             ('DUO_HOLD_60C', 'MAPPO60C'), ('OLD_BIG_BANK_60C', 'MAPPO60C'))
AUTHORITY = ('MAIN AMENDMENT 68 (S-RESEARCH60C, 2026-11-12): hold-only WM search (cfg 9c905aca), top 12 candidates, '
             'admission on 24 fresh worlds vs MAPPO60C (2abed317), confirmation on a fresh 96-world panel; phase '
             'research60c_a68 cap 1,070,400')
DISCLOSURE = ('the world model has no background policy (non-member CAVs follow its native driver law), so the search '
              'does not see MAPPO60C; start states are the native-tape states 74aef8f8; harm_cal was fitted on u0 SUMO '
              'outcomes; 12 picks each tested at 95 % in admission (no multiplicity correction); MAPPO60C enters only '
              'through SUMO admission and confirmation')
ADMIT_RULE = ('MAIN: (plan - MAPPO60C) total delay CI95 upper < 0 on the 24 verification worlds (a37_cellbank.boot_ci) '
              'AND all five groups present with mean increase <= 10 s relative to MAPPO60C; BACKUP: the delay test '
              'only (secondary; never substituted for MAIN); bank priority lb95(gain) desc then strategy key')
R2_RULE = ('each MAIN plan cut to two roles: roles sorted by (v_w asc, hold_s desc, onset_s asc, original index asc), '
           'the first two kept in their original order; every other field unchanged; family suffix _r2; bank order '
           'and lb95 inherited; a refused reduced plan makes the arm not runnable')
PRIMARY_RULE = ('SUPPORTED iff MAIN_BANK_60C - MAPPO60C total delay CI95 upper < 0 (paired percentile bootstrap '
                '10,000, seed 20261104, n = 96) AND the five-group 10 s rule relative to MAPPO60C; diff = treatment - '
                'reference, negative = less delay')


def _mods():
    import nm_finaleval as N
    import a50_fullp60 as A50
    import a52_mappo60 as A52
    import a67_plans_on_60c as A67
    return N, A50, A52, A67


def core(which):
    """The A42 driver code with the A68 names; ``which`` = i (identity panel 998600-), v (verification), c."""
    N, A50, _, _ = _mods()
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A68',
        HOST_BASE=os.environ.get('A68_HOST_BASE', '/root/a68'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A68_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A68', JOB_PREFIX='a68',
        VERIFY_TAG='a68v', CONFIRM_TAG=f'a68{which}', IDENT_TAG='a68i',
        VERIFY_WAVE_PREFIX='A68V_', CONFIRM_WAVE_PREFIX='A68C_',
        RESULTS_FILE=RESULTS_FILE, LOG_FILE='A68_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a68',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 68; plans searched in the WM, admitted and confirmed on MAPPO60C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a68_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    return m


def setup(which):
    N, _, _, _ = _mods()
    D = core(which)
    panel = D.setup(SETTINGS[which])
    seeds = tuple(r['world']['seed'] for r in panel['worlds'])
    if seeds != SEEDS[which] or panel['cell']['cell_key'] != CELL:
        raise N.Blocked(f'{SETTINGS[which]}: not the registered panel')
    if which in PANEL_SHA256 and panel['panel_sha256'] != PANEL_SHA256[which]:
        raise N.Blocked(f'{SETTINGS[which]}: panel sha differs from the registered one')
    return D, panel


# ------------------------------------------------------------------------------------------------ candidates

def select(search_path, out):
    """Zero SUMO, WM ranking only: the registered top-12 selection, sealed; never overwrites."""
    from tools.coopmwm import a42_bigbank as B
    search_path, out = Path(search_path), Path(out)
    raw = search_path.read_bytes()
    import hashlib
    search = json.loads(raw.decode('utf-8'))
    sel = B.select_candidates(search, CELL, k=K)
    for c in sel['candidates']:
        c['n_roles'] = len(c['strategy']['roles'])
    doc = B.seal(dict(
        schema='coopmwm.a68.candidates.v1', amendment=AMENDMENT, cell=CELL, k=K,
        search=dict(file_sha256=hashlib.sha256(raw).hexdigest(), schema=search.get('schema'), cell=search.get('cell'),
                    config=search.get('config'), evaluated=search.get('evaluated'), n_states=search.get('n_states'),
                    n_final=len(search['final'])),
        rule='search `final` list by WM score desc (ties: strategy key), skip comparator / banned rows, plans the '
             'registered executor refuses and rows with feasibility 0 (A42 skip rule), top K = 12',
        STOP=not sel['candidates'], skip_counts=dict(sorted(Counter(e['skip'] for e in sel['skipped']).items())),
        n_skipped=len(sel['skipped']), n_eligible=len(sel['ranking'])-len(sel['skipped']),
        n_selected=len(sel['candidates']), candidates=sel['candidates'], skipped=sel['skipped'],
        ranking=sel['ranking']))
    if out.exists():
        raise SystemExit(f'{out}: exists (pre-outcome file; never overwrite)')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return doc


def candidates(path=None):
    """The pinned pre-outcome candidates; refuses an unpinned / edited file."""
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    path = Path(path or CANDS_FILE)
    if not path.is_file() or not CANDS_PIN.is_file():
        raise N.Blocked(f'{path} / {CANDS_PIN.name}: missing (run select, record the doc sha in REGISTRATION.md)')
    doc = N.read(path)
    pin = CANDS_PIN.read_text(encoding='utf-8').split()[0]
    if B.doc_sha(doc) != doc.get('doc_sha256') or doc['doc_sha256'] != pin:
        raise N.Blocked(f'{path.name}: not the registered selection (pin {pin})')
    if doc['cell'] != CELL or doc['k'] != K or doc.get('amendment') != AMENDMENT or doc['STOP']:
        raise N.Blocked(f'{path.name}: not the registered cell / K, or no candidate')
    fams = [c['family'] for c in doc['candidates']]
    if len(set(fams)) != len(fams) or not 1 <= len(fams) <= K:
        raise N.Blocked(f'{path.name}: {len(fams)} candidates / duplicate family names')
    for c in doc['candidates']:
        if C.spec_sha(c['strategy']) != c['strategy_sha256'] or C.executor_refusal(c['strategy']) is not None:
            raise N.Blocked(f'{c["family"]}: strategy sha differs / not an executable hold plan')
    return doc['candidates']


def reduce2(strategy):
    """The registered MAIN_BANK_REDUCED2 rule (R2_RULE): pure, deterministic."""
    roles = strategy['roles']
    order = sorted(range(len(roles)), key=lambda i: (float(roles[i]['v_w']), -int(roles[i]['hold_s']),
                                                     int(roles[i]['onset_s']), i))
    keep = sorted(order[:2])
    out = deepcopy(strategy)
    out['roles'] = [deepcopy(roles[i]) for i in keep]
    out['name'] = f'{strategy["name"]}_r2'
    return out, keep


def executor(a66_root):
    return _mods()[3].executor(a66_root)


def bank_entry(D, root, name, pin=None):
    N = _mods()[0]
    e = D._bank_entry(root, name)
    if pin is not None and e['file_sha256'] != pin:
        raise N.Blocked(f'{name}: sha differs from the registered {pin[:8]}')
    return e


def bank_specs(D, panel, root, name, exe, tag, pin=None):
    """CELL_BANK runner arm on bank ``name`` with only the six executor fields bound to MAPPO60C (A67.with_exec)."""
    A67 = _mods()[3]
    base = D._a37_specs(D.RUNNER_ARM['BIG_BANK'], panel, root, bank_entry(D, root, name, pin), tag=tag)
    return A67.with_exec(base, exe, tag)


def mappo_specs(D, panel, root, exe, tag):
    A52 = _mods()[2]
    return A52.with_executor(D.confirm_specs('U0', panel, root), exe, tag)


# ------------------------------------------------------------------------------------------------ identity

def ident_waves(D, panel, root, a66_root):
    exe = executor(a66_root)
    return [('A68_IDENT_MAPPO60C', 'episode', mappo_specs(D, panel, root, exe, 'a68i_mappo60c_full')[:IDENT_N]),
            ('A68_IDENT_BIG_BANK_60C', 'episode',
             bank_specs(D, panel, root, OLD_BANK, exe, 'a68i_big_bank', OLD_BANK_SHA256)[:IDENT_N])]


def identity_compare(D, panel, root, a66_root, a67_root, a42_root):
    N, _, _, A67 = _mods()
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
            rows.append(dict(wave=wave, world_id=r['world_id'], j_a68=r['J_veh_s'], j_stored=a['J_veh_s'],
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
    return N.seal(dict(schema='coopmwm.a68.identity.v1', amendment=AMENDMENT,
                       rule='MAPPO60C alone and old BIG_BANK@60C on the first 2 worlds of panel 998600-998695 run '
                            'from the A68 code tree (table-only edits) equal the stored A66 / A67 records exactly: '
                            'J_veh_s difference 0, identical safety counters, same activations',
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: admission may run; FAILED: stop and report'), 'receipt_sha256')


def identity_passed(root):
    N = _mods()[0]
    p = Path(root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ verification

def prepare(D, root, cands):
    from tools.coopmwm import a37_cellbank as C
    doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    p = D.C37.write_once(D.bank_path(root, CANDS_BANK), D.C37.bank_text(doc))
    return dict(path=str(p), file_sha256=_mods()[0].file_sha(p), families=[c['family'] for c in cands])


def verify_waves(D, panel, root, a66_root, cands):
    """[(wave, 'episode', specs)]: MAPPO60C alone, then every candidate (selection order) on the 24 worlds."""
    A67 = _mods()[3]
    exe = executor(a66_root)
    entry = bank_entry(D, root, CANDS_BANK)
    waves = [('A68V_MAPPO60C', 'episode', mappo_specs(D, panel, root, exe, 'a68v_mappo60c'))]
    for c in cands:
        tag = f'a68v_{c["key"][:8]}'
        base = D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=c['family'], tag=tag)
        waves.append((f'A68V_{c["family"]}', 'episode', A67.with_exec(base, exe, tag)))
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise _mods()[0].Blocked('verification waves: wrong size / duplicate wave')
    return waves


def admission(root):
    N = _mods()[0]
    p = Path(root)/ADMISSION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def admit(D, panel, root, a66_root, cands):
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    root = Path(root)
    if admission(root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    waves = verify_waves(D, panel, root, a66_root, cands)
    have, sources = {}, {}
    for (name, _, specs), arm in zip(waves, [REF]+[c['family'] for c in cands]):
        have[arm], sources[arm] = D.C37._load(root, name, specs)
    reports = D.C37._reports(root, have, worlds)
    smap = D._spec_map(cands)
    main_v, back_v, rows = {}, {}, []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, REF, bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have[REF][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a37'] for w in worlds]
        delay_pass = bool(v['ci95'][1] < 0)
        v.update(family=fam, wm_rank=cand['wm_rank'], selected_rank=cand['selected_rank'], key=cand['key'],
                 wm_score=cand['score'], n_roles=len(cand['strategy']['roles']), reference=REF,
                 convention='diff = plan - MAPPO60C (negative = less delay); gain = -diff',
                 delay_test_pass=delay_pass, group_rule_pass=bool(v['allowed']), MAIN=bool(v['PASS']),
                 BACKUP=delay_pass, cvar90_vs_ref=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_ref_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 realised_group_size=D._group_sizes(have[fam], worlds, smap),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        main_v[fam] = v
        back_v[fam] = dict(v, PASS=delay_pass)
        rows.append(v)
    main = C.admitted_plans(cands, main_v)
    backup = C.admitted_plans(cands, back_v)

    def write(name, plans):
        p = D.C37.write_once(D.bank_path(root, name), D.C37.bank_text(C.bank_doc(plans, CELL)))
        return dict(path=f'{D.HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p),
                    families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans])
    banks = dict(MAIN=write(MAIN_BANK, main) if main else None, BACKUP=None, R2=None)
    backup_differs = [p['family'] for p in backup] != [p['family'] for p in main]
    if backup and backup_differs:
        banks['BACKUP'] = write(BACKUP_BANK, backup)
    r2 = dict(rule=R2_RULE, runnable=False, reason='no MAIN plan' if not main else None, plans=[])
    if main:
        red = []
        for p in main:
            s2, keep = reduce2(p['strategy'])
            red.append(dict(family=s2['name'], strategy=s2, lb95_veh_s=p['lb95_veh_s']))
            r2['plans'].append(dict(parent=p['family'], family=s2['name'], kept_role_indices=keep,
                                    parent_n_roles=len(p['strategy']['roles']),
                                    strategy_key=B.O.strategy_key(s2), refusal=C.executor_refusal(s2)))
        shape = B.shape_check([x['strategy'] for x in red], CELL) if all(x['refusal'] is None
                                                                         for x in r2['plans']) else []
        r2['shape_check'] = shape
        if shape and all(x['admitted'] for x in shape):
            banks['R2'] = write(R2_BANK, red)
            r2['runnable'] = True
        else:
            r2['reason'] = 'a reduced plan is refused by the registered executor (not runnable; no improvisation)'
    value = N.seal(dict(
        schema='coopmwm.a68.admission.v1', amendment=AMENDMENT, cell=CELL, rule=ADMIT_RULE, reference=REF,
        multiplicity=f'{len(rows)} picks each tested at 95% (disclosed, as A37/A42)',
        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), verdicts=rows,
        main=[p['family'] for p in main], backup=[p['family'] for p in backup], backup_differs=backup_differs,
        banks=banks, reduced2=r2,
        consequence=('no plan passes the delay test: no confirmation run; the null is reported' if not backup else
                     'no MAIN plan: MAIN arms are not run (primary null); BACKUP runs as a secondary arm' if not main
                     else 'confirmation runs'),
        ref_mean_J_veh_s=sum(have[REF][w]['J_veh_s'] for w in worlds)/len(worlds),
        ref_safety_totals={k: int(sum(have[REF][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    N.publish(root/ADMISSION_FILE, value)
    return value


# ------------------------------------------------------------------------------------------------ confirmation

def confirm_arms(adm):
    """The arms the registered design runs given the admission receipt."""
    if adm is None:
        raise _mods()[0].Blocked('no admission receipt (run verify + admit first)')
    if not adm['main'] and not adm['backup']:
        return ()
    out = ['MAPPO60C', 'DUO_HOLD_60C', 'OLD_BIG_BANK_60C']
    if adm['banks']['MAIN']:
        out.append('MAIN_BANK_60C')
    if adm['banks']['R2']:
        out.append('MAIN_BANK_R2_60C')
    if adm['banks']['BACKUP']:
        out.append('BACKUP_BANK_60C')
    return tuple(out)


def confirm_specs(D, arm, panel, root, a66_root, adm):
    N, _, _, A67 = _mods()
    exe = executor(a66_root)
    tag = f'a68c_{arm.lower()}'
    if arm == 'MAPPO60C':
        specs = mappo_specs(D, panel, root, exe, tag)
    elif arm == 'DUO_HOLD_60C':
        specs = A67.with_exec(D.confirm_specs('DUO_HOLD', panel, root), exe, tag)
    else:
        pin = OLD_BANK_SHA256 if arm == 'OLD_BIG_BANK_60C' else adm['banks'][
            dict(MAIN_BANK_60C='MAIN', MAIN_BANK_R2_60C='R2', BACKUP_BANK_60C='BACKUP')[arm]]['file_sha256']
        specs = bank_specs(D, panel, root, BANK_OF[arm], exe, tag, pin)
    if len(specs) != CONFIRM_N:
        raise N.Blocked(f'{arm}: {len(specs)} specs')
    return specs


def confirm_waves(D, panel, root, a66_root):
    adm = admission(root)
    return [(f'A68C_{a}', 'episode', confirm_specs(D, a, panel, root, a66_root, adm)) for a in confirm_arms(adm)]


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
    N, _, _, A67 = _mods()
    from a32_ablation import report_of
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    adm = admission(root)
    arms_run = confirm_arms(adm)
    base = dict(schema='coopmwm.a68.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                admission=dict(receipt_sha256=adm['receipt_sha256'], main=adm['main'], backup=adm['backup'],
                               banks=adm['banks'], reduced2_runnable=adm['reduced2']['runnable']),
                arms_run=list(arms_run), root=str(root))
    if not arms_run:
        return N.seal(dict(base, primary=dict(rule=PRIMARY_RULE, supported=False, verdict='NULL',
                                              note='no plan passed admission: no confirmation run')),
                      'analysis_sha256')
    have, sources = {}, {}
    for arm in arms_run:
        have[arm], sources[arm] = D.C37._load(root, f'A68C_{arm}',
                                              confirm_specs(D, arm, panel, root, a66_root, adm))
    reports = {a: {w: report_of(root, have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = []
    if PRIMARY[0] in have:
        row = D._contrast(have, reports, worlds, *PRIMARY, bootstrap, 'primary')
        rows.append(row)
        ci = row['total_minus_ref']['ci95']
        groups = row['groups'] or {}
        a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
        primary = dict(rule=PRIMARY_RULE, contrast='MAIN_BANK_60C - MAPPO60C', diff=row['total_minus_ref']['diff'],
                       ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass,
                       group_rule_pass=b_pass, supported=bool(a_pass and b_pass),
                       verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED')
    else:
        primary = dict(rule=PRIMARY_RULE, supported=False, verdict='NULL',
                       note='no plan passed MAIN admission: MAIN arms not run')
    rows += [D._contrast(have, reports, worlds, a, r, bootstrap, 'secondary_descriptive')
             for a, r in SECONDARY if a in have and r in have]
    arms = {}
    for a in have:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        if a != 'MAPPO60C':
            arms[a]['activations'] = A67.activation_table(have[a], worlds, plan_specs(D, root, a))
    return N.seal(dict(base, primary=primary, contrasts=rows, arms=arms,
                       multiplicity='one registered primary; every other contrast is descriptive',
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

STAGE = dict(identity='i', verify='v', confirm='c')
STAGE_CAP = dict(identity=IDENT_CAP, verify=VERIFY_CAP, confirm=CONFIRM_CAP)
STAGE_PREFIX = dict(identity='a68i_', verify='a68v_', confirm='a68c_')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('select', 'prepare', 'dry', 'identity', 'identity-compare', 'verify', 'admit',
                                       'confirm', 'analyse', 'status'))
    p.add_argument('--final-root')
    p.add_argument('--a66-root')
    p.add_argument('--a67-root')
    p.add_argument('--a42-root')
    p.add_argument('--stage', choices=('i', 'v', 'c'), default='v')
    p.add_argument('--search')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    if a.command == 'select':
        doc = select(a.search, a.out or CANDS_FILE)
        print(json.dumps(dict(doc_sha256=doc['doc_sha256'], STOP=doc['STOP'], skip_counts=doc['skip_counts'],
                              n_eligible=doc['n_eligible'], n_final=doc['search']['n_final'],
                              candidates=[(c['family'], c['score'], c['n_roles'], c['wm']['feasible'])
                                          for c in doc['candidates']])))
        return
    N = _mods()[0]
    root = Path(a.final_root)
    for other in (a.a66_root, a.a67_root, a.a42_root):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A68 final root is a new root')
    which = dict(identity='i', verify='v', confirm='c', prepare='v', admit='v', analyse='c').get(a.command, a.stage)
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
        if stage == 'verify':
            return verify_waves(D, panel, root, a.a66_root, candidates())
        return confirm_waves(D, panel, root, a.a66_root)

    def brief(ws):
        return {n: dict(n=len(s), first=s[0]['job_id'], last=s[-1]['job_id'], runner_arm=s[0]['arm'],
                        seeds=[s[0]['world']['seed'], s[-1]['world']['seed']], cell=s[0]['world']['cell_key'],
                        checkpoint={k: v for k, v in s[0]['checkpoint'].items() if k != 'label'})
                for n, _, s in ws}
    if a.command == 'prepare':
        print(json.dumps(prepare(D, root, candidates()), indent=1))
    elif a.command == 'dry':
        stage = dict(i='identity', v='verify', c='confirm')[which]
        ws = waves_of(stage)
        print(json.dumps(dict(stage=stage, panel_sha256=panel['panel_sha256'], waves=brief(ws),
                              sim_s=sum(len(s)*N.HORIZON_S for _, _, s in ws), stage_cap=STAGE_CAP[stage],
                              charged=D.charged(root), cap=CAP_SIM_S), indent=1))
    elif a.command in STAGE:
        import fcntl
        if a.command != 'identity' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: nothing is run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(a.command)
            if not waves:
                log(dict(done=a.command, status='SKIPPED_NO_PLAN_ADMITTED'))
                return
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
    elif a.command == 'admit':
        v = admit(D, panel, root, a.a66_root, candidates())
        print(json.dumps(dict(main=v['main'], backup=v['backup'], backup_differs=v['backup_differs'],
                              reduced2_runnable=v['reduced2']['runnable'], receipt_sha256=v['receipt_sha256'],
                              rows=[(x['family'], round(x['mean'], 1), [round(c, 1) for c in x['ci95']],
                                     x['group_rule_pass'], x['MAIN']) for x in v['verdicts']])))
    elif a.command == 'analyse':
        v = analyse(D, panel, root, a.a66_root)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        stage = dict(i='identity', v='verify', c='confirm')[which]
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root), stage=stage)
        try:
            for n, _, s in waves_of(stage):
                out[n] = f'{N.published(root, n, s, D.C37.check_a37)[2]}/{len(s)}'
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
