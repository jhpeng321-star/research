"""AMENDMENT 69 (S-LCSIDE60C; MAIN 2026-11-12): lane-change plans as a SIDE track of A68 (exploratory; never merged
into the A68 MAIN bank, never changes the A68 primary).

Uses the A44 lane-change executor variant (tools/coopmwm/a44_lcvariant.py, runner arms A44_PLAN / A44_BANK; the A37
executor path is unchanged) with only the six executor fields bound to MAPPO60C (A65 policy_00084, 2abed317).

Stages (own root, one ledger phase ``lcside_a69``, cap 297,600 sim-s = 9,600 + 172,800 + 115,200):
  select    zero SUMO: top K = 6 plans of the search-L `final` list that contain >= 1 lane-change role (WM score desc,
            ties by key; skip comparator / banned / variant-executor refusal / feasibility 0)
  identity  worlds 992400, 992401 (A44 mechanics block): the hold plan a44_fix_hold on MAPPO60C by the A67 executor
            path (A37_PLAN) and by the variant (A44_PLAN) must give exactly equal records (J, safety, executor
            receipts, no lane-change member; vehicle logs by DECOMPRESSED content: the stored sha covers the gzip
            container, whose header carries the write time)
  verify    the 24 A68 verification worlds 992500-992523: each candidate as one unconditional plan (A44_PLAN)
  admit     (plan - MAPPO60C) vs the A68 MAPPO60C records of those worlds (A68 root, read-only): delay test CI95
            upper < 0 AND five-group 10 s rule; LC_BANK = plans passing both, priority verified lb95 desc
  confirm   one arm LC_BANK_60C (A44_BANK) on the A68 confirmation panel 996200-996295
  analyse   exploratory rows LC_BANK_60C - {MAPPO60C, DUO_HOLD_60C, MAIN_BANK_60C} (A68 records, read-only)
No automatic replay. Nothing is written outside the A69 root.

Commands (python ops/a69_lcside.py <cmd> --final-root R --a68-root R68 --a66-root R66):
  select --search F ; prepare-ident ; prepare ; dry --stage i|v|c ; identity ; identity-compare ; verify ; admit ;
  confirm ; analyse ; status --stage i|v|c
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

AMENDMENT = '69'
CELL = 'D2800_p60'
PHASE = 'lcside_a69'
IDENT_N, VERIFY_N, CONFIRM_N, K = 2, 24, 96, 6
IDENT_CAP, VERIFY_CAP, CONFIRM_CAP = 9_600, 172_800, 115_200
CAP_SIM_S = IDENT_CAP+VERIFY_CAP+CONFIRM_CAP
MECH_SEEDS = '992400-992407'
MECH_PANEL_SHA256 = '49a271551277ab2d2378a733ce245a47768bcb02c47eedf5dfd6dc700a4fbb87'
A44_SHA8, A37_SHA8 = '53b64cbe', '0a902437'
HOST_BASE = os.environ.get('A69_HOST_BASE', '/root/a69')
HOST_ROOT = f'{HOST_BASE}/root'
BANK_DIR = 'a69_banks'
CANDS_BANK = f'A69_CANDS_{CELL}.json'
LC_BANK = f'A69_LC_BANK_{CELL}.json'
A69_DIR = HERE/'a69'
CANDS_FILE = A69_DIR/f'CANDIDATES_LC_{CELL}.json'
CANDS_PIN = A69_DIR/f'CANDIDATES_LC_{CELL}.sha256'
ADMISSION_FILE = f'A69_ADMISSION_{CELL}.json'
IDENTITY_FILE = 'IDENTITY_A69.json'
RESULTS_FILE = f'A69_RESULTS_{CELL}.json'
LOG_FILE = 'A69_LOG.jsonl'
REF = 'MAPPO60C'
ARM = 'LC_BANK_60C'
CONFIRM_WAVE = f'A69C_{ARM}'
REF_ARMS = ('MAPPO60C', 'DUO_HOLD_60C', 'MAIN_BANK_60C')
AUTHORITY = ('MAIN AMENDMENT 69 (S-LCSIDE60C, 2026-11-12): lane-change plans side track; variant executor 53b64cbe on '
             'MAPPO60C (2abed317); phase lcside_a69 cap 297,600; exploratory, never merged into the A68 MAIN bank')
DISCLOSURE = ('exploratory side track; the search does not see MAPPO60C (world-model non-members follow its native '
              'driver law); 6 picks each tested at 95 %; the empty-plan half of the identity could not be built on '
              'the variant (it refuses a bank without a plan) and was not run; A69 records come from a second code '
              'tree that differs from the A68 tree only by the additive A44 arm code')
ADMIT_RULE = ('(plan - MAPPO60C) total delay CI95 upper < 0 on the 24 verification worlds (a37_cellbank.unc_verdict) '
              'AND all five groups present with mean increase <= 10 s relative to MAPPO60C; reference = the A68 '
              'MAPPO60C records of the same worlds; LC_BANK priority lb95(gain) desc then strategy key')


def _mods():
    import nm_finaleval as N
    import a50_fullp60 as A50
    import a67_plans_on_60c as A67
    import a68_research60c as A68
    return N, A50, A67, A68


def mech():
    """The A44 mechanics driver code (same file) with the A69 names: own phase / cap / root / bank dir."""
    N, A50, _, _ = _mods()
    m = A50._clone('a69_mech_core', HERE/'a44_lcmech.py', dict(
        A44_PHASE=PHASE, A44_CAP_SIM_S=CAP_SIM_S, STOP_A44='STOP_A69', HOST_BASE=HOST_BASE, HOST_ROOT=HOST_ROOT,
        BANK_DIR=BANK_DIR))
    m.A44Runtime.phase = PHASE
    N.CHECKS['episode'] = m.check_a44
    return m


def runtime(root):
    N = _mods()[0]
    M = mech()

    class A69Runtime(M.A44Runtime):
        phase = PHASE

        def reserve(self, cfg, jobs):
            if not all(str(j['id']).startswith('a69') for j in jobs):
                raise N.Blocked('A69: only a69* jobs in this root')
            self._caps(sum(j['sim_s'] for j in jobs))
            N.own_reserve(self.final_root, jobs, PHASE, cap=CAP_SIM_S, calls=self.calls, allocation=CAP_SIM_S)
    return A69Runtime(N.Runtime(N.RUNNER), Path(root))


def digests():
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a44_lcvariant as V
    a37, a44 = C.a37_digest(), V.a44_digest()
    if not a37.startswith(A37_SHA8) or not a44.startswith(A44_SHA8):
        raise N.Blocked(f'executor digests differ from the registered ones: a37 {a37[:8]} a44 {a44[:8]}')
    return dict(a37_sha256=a37, a44_sha256=a44)


def setup(which):
    """-> (panel, D68 or None). i: the A44 mechanics panel; v / c: the A68 verification / confirmation panel."""
    N, _, _, A68 = _mods()
    M = mech()
    digests()
    if which == 'i':
        panel = M.setup(MECH_SEEDS)
        if panel['panel_sha256'] != MECH_PANEL_SHA256 or M.PANEL_SHA256 != MECH_PANEL_SHA256:
            raise N.Blocked('the mechanics panel is not the registered / pinned one')
        return panel, None
    D, panel = A68.setup(which)
    N.CHECKS['episode'] = M.check_a44
    return panel, D


def executor(a66_root):
    return _mods()[2].executor(a66_root)


def with_exec44(specs, exe, tag):
    """A44-arm specs with ONLY the six executor fields replaced: u0 d9042cc5 -> MAPPO60C (as A67.with_exec)."""
    N = _mods()[0]
    out = []
    for i, s in enumerate(specs):
        ck = deepcopy(s['checkpoint'])
        if ck.get('executor_sha256') != N.FROZEN_MAPPO_SHA or ck.get('kind') != 'a44':
            raise N.Blocked(f'{s["job_id"]}: not a registered u0 A44 spec')
        ck.update(comparator_kind=exe['kind'], label=exe['label'], executor_path=exe['path'],
                  executor_sha256=exe['sha256'], binding_sha256=exe['binding_sha256'],
                  binding_path=exe['binding_path'])
        out.append(dict(s, checkpoint=ck, job_id=f'{tag}_{i:04}'))
    return out


def a44_specs(arm, panel, root, entry, exe, tag, family=None):
    return with_exec44(mech().a44_specs(arm, panel, root, entry, family=family, tag=tag), exe, tag)


def published(root, wave, specs):
    N = _mods()[0]
    recs, src, cov = N.published(root, wave, specs, mech().check_a44)
    if cov != len(specs):
        raise N.Blocked(f'{wave}: {cov}/{len(specs)} published in {root}')
    return {r['world_id']: r for r in recs}, src


def log_content_sha(root, record):
    """sha256 of the DECOMPRESSED vehicle log of one record (the record's own sha covers the gzip container)."""
    N = _mods()[0]
    p = Path(root)/'evaluation'/record.get('work_dir', record['job_id'])/N.VEHICLE_LOG
    with gzip.open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------------------------------------ candidates

def select(search_path, out):
    """Zero SUMO, WM ranking only: the registered top-6 lane-change selection, sealed; never overwrites."""
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    from tools.coopmwm import a44_lcvariant as V
    search_path, out = Path(search_path), Path(out)
    raw = search_path.read_bytes()
    search = json.loads(raw.decode('utf-8'))
    if search.get('cell') != CELL:
        raise SystemExit(f'search cell {search.get("cell")!r} is not {CELL}')
    rows = sorted(search['final'], key=lambda x: (-float(x['score']), x['key']))
    ranking, skipped, cands = [], [], []
    for i, x in enumerate(rows):
        s = x['strategy']
        if B.O.strategy_key(s) != x['key']:
            raise SystemExit(f'{s["name"]}: strategy_key differs from the search key')
        why = ('comparator' if x['comparator'] else 'banned' if x['banned'] else
               'no_lc_role' if not V.is_lc_plan(s) else V.executor_refusal(s))
        if why is None and not float(x['feasible']) > 0:
            why = B.NEVER_ACTIVATED
        entry = dict(wm_rank=i+1, name=s['name'], key=x['key'], score=x['score'], skip=why)
        ranking.append(entry)
        if why is not None:
            skipped.append(entry)
            continue
        if len(cands) < K:
            cands.append(dict(entry, family=s['name'], strategy=s, strategy_sha256=C.spec_sha(s),
                              n_roles=len(s['roles']),
                              lc_roles=[dict(index=j, lane=r['lane'], lc_target=r['lc_target'])
                                        for j, r in enumerate(s['roles']) if r.get('lc_target') is not None],
                              wm=dict({f: x.get(f) for f in ('feasible', 'gain_vs_base', 'gain_vs_base_lcb',
                                                               'gain_vs_inc', 'worst_group_inc', 'cvar90_inc',
                                                               'bounded', 'fair_cal')}, n_states=x.get('n_states')),
                              selected_rank=len(cands)+1))
    doc = B.seal(dict(
        schema='coopmwm.a69.candidates.v1', amendment=AMENDMENT, cell=CELL, k=K,
        search=dict(file_sha256=hashlib.sha256(raw).hexdigest(), schema=search.get('schema'), cell=search.get('cell'),
                    config=search.get('config'), evaluated=search.get('evaluated'), n_states=search.get('n_states'),
                    n_final=len(search['final'])),
        rule='search-L `final` list by WM score desc (ties: strategy key); skip comparator / banned rows, plans '
             'without a lane-change role, plans the variant executor refuses and rows with feasibility 0; top K = 6',
        STOP=not cands, skip_counts=dict(sorted(Counter(e['skip'] for e in skipped).items())),
        n_skipped=len(skipped), n_eligible=len(ranking)-len(skipped), n_selected=len(cands), candidates=cands,
        skipped=skipped, ranking=ranking))
    if out.exists():
        raise SystemExit(f'{out}: exists (pre-outcome file; never overwrite)')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return doc


def candidates(path=None):
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    from tools.coopmwm import a44_lcvariant as V
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
        s = c['strategy']
        if C.spec_sha(s) != c['strategy_sha256'] or V.executor_refusal(s) is not None or not V.is_lc_plan(s):
            raise N.Blocked(f'{c["family"]}: strategy sha differs / not an executable lane-change plan')
    return doc['candidates']


# ------------------------------------------------------------------------------------------------ identity

def ident_waves(panel, root, a66_root):
    _, _, A67, _ = _mods()
    M = mech()
    exe = executor(a66_root)
    fam = M.fixtures()['hold']['name']
    hold = M.bank_entry(root, M.HOLD_BANK)
    a37 = A67.with_exec(M.C37._a37_specs('A37_PLAN', panel, root, hold, family=fam, tag='a69i_a37'), exe, 'a69i_a37')
    a44 = a44_specs('A44_PLAN', panel, root, hold, exe, 'a69i_a44', family=fam)
    return [('A69I_A37_HOLD', 'episode', a37[:IDENT_N]), ('A69I_A44_HOLD', 'episode', a44[:IDENT_N])]


def identity_compare(panel, root, a66_root):
    N, _, A67, _ = _mods()
    M = mech()
    (n37, _, s37), (n44, _, s44) = ident_waves(panel, root, a66_root)
    r37, r44 = published(root, n37, s37)[0], published(root, n44, s44)[0]
    rows = []
    for wid in [s['world_id'] for s in s37]:
        a, b = r37[wid], r44[wid]
        ea, eb = a['a37'], b['a44']
        diff = [k for k in ('J_veh_s', 'safety') if a.get(k) != b.get(k)]
        ca, cb = log_content_sha(root, a), log_content_sha(root, b)
        if ca != cb:
            diff.append('vehicle_log content')
        diff += [f'executor.{k}' for k in M.EXECUTOR_FIELDS if ea.get(k) != eb.get(k)]
        if eb.get('lc_members'):
            diff.append('executor.lc_members (hold plan produced lane-change members)')
        if not (a.get('executor_sha256_loaded') == b.get('executor_sha256_loaded') == A67.MAPPO60C_SHA256):
            diff.append('executor is not MAPPO60C in both records')
        rows.append(dict(world_id=wid, equal=not diff, differences=diff, activations=ea.get('activations'),
                         issued=ea.get('issued'), vehicle_log_content_sha256=[ca, cb],
                         container_sha_equal=(a.get('vehicle_log') or {}).get('sha256')
                         == (b.get('vehicle_log') or {}).get('sha256')))
    ok = len(rows) == IDENT_N and all(r['equal'] for r in rows)
    return N.seal(dict(schema='coopmwm.a69.identity.v1', amendment=AMENDMENT,
                       rule='hold plan a44_fix_hold on MAPPO60C: A67 executor path (A37_PLAN) vs the variant '
                            '(A44_PLAN) on worlds 992400 / 992401: J, safety, executor receipts identical, no '
                            'lane-change member, decompressed vehicle logs byte-identical',
                       not_run='the empty-plan half (the variant refuses a bank without a plan); disclosed',
                       verdict='PASS' if ok else 'FAILED',
                       non_vacuous=any((r['activations'] or 0) > 0 for r in rows), worlds=rows,
                       consequence='PASS (and A44 mechanics pass): A69 SUMO may run; FAILED: proposed, not simulated'),
                  'receipt_sha256')


def identity_passed(root):
    N = _mods()[0]
    p = Path(root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ verification

def prepare(root, cands):
    N = _mods()[0]
    M = mech()
    from tools.coopmwm import a44_lcvariant as V
    doc = V.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    p = M.C37.write_once(Path(root)/BANK_DIR/CANDS_BANK, M.C37.bank_text(doc))
    return dict(path=str(p), file_sha256=N.file_sha(p), families=[c['family'] for c in cands])


def verify_waves(panel, root, a66_root, cands):
    N = _mods()[0]
    M = mech()
    exe = executor(a66_root)
    entry = M.bank_entry(root, CANDS_BANK)
    waves = []
    for c in cands:
        tag = f'a69v_{c["key"][:8]}'
        waves.append((f'A69V_{c["family"]}', 'episode',
                      a44_specs('A44_PLAN', panel, root, entry, exe, tag, family=c['family'])))
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def a68_ref(D, panel, a68_root, a66_root, which):
    """The A68 MAPPO60C records of this panel (read-only; checked against their registered A68 specs)."""
    A68 = _mods()[3]
    tag, wave = dict(v=('a68v_mappo60c', 'A68V_MAPPO60C'), c=('a68c_mappo60c', 'A68C_MAPPO60C'))[which]
    return D.C37._load(a68_root, wave, A68.mappo_specs(D, panel, a68_root, executor(a66_root), tag))


def admission(root):
    N = _mods()[0]
    p = Path(root)/ADMISSION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def _reports(roots, have, worlds):
    N = _mods()[0]
    from a32_ablation import report_of
    reports = {a: {w: report_of(roots[a], have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    return reports


def lc_counts(recs, worlds):
    tot = dict(members=0, requested=0, completed=0, gave_up=0, never_requested=0)
    for w in worlds:
        s = (recs[w].get('a44') or {}).get('lc_summary') or {}
        for k in tot:
            tot[k] += int(s.get(k) or 0)
    return tot


def admit(D, panel, root, a68_root, a66_root, cands):
    N = _mods()[0]
    M = mech()
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    from tools.coopmwm import a44_lcvariant as V
    root = Path(root)
    if admission(root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources, roots = {}, {}, {}
    have[REF], sources[REF] = a68_ref(D, panel, a68_root, a66_root, 'v')
    roots[REF] = a68_root
    for (name, _, specs), c in zip(verify_waves(panel, root, a66_root, cands), cands):
        have[c['family']], sources[c['family']] = published(root, name, specs)
        roots[c['family']] = root
    reports = _reports(roots, have, worlds)
    smap = {c['family']: c['strategy'] for c in cands}
    verdicts, rows = {}, []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, REF, bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have[REF][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a44'] for w in worlds]
        sizes = []
        for w in worlds:
            sizes += B.realised_sizes(have[fam][w]['a44'], smap)
        v.update(family=fam, wm_rank=cand['wm_rank'], selected_rank=cand['selected_rank'], key=cand['key'],
                 wm_score=cand['score'], n_roles=len(cand['strategy']['roles']), lc_roles=cand['lc_roles'],
                 reference=REF, convention='diff = plan - MAPPO60C (negative = less delay); gain = -diff',
                 delay_test_pass=bool(v['ci95'][1] < 0), group_rule_pass=bool(v['allowed']), ADMITTED=bool(v['PASS']),
                 cvar90_vs_ref=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_ref_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 realised_group_size=B.group_size_summary(sizes), lane_changes=lc_counts(have[fam], worlds),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        verdicts[fam] = v
        rows.append(v)
    plans = C.admitted_plans(cands, verdicts)
    bank = None
    if plans:
        p = M.C37.write_once(root/BANK_DIR/LC_BANK, M.C37.bank_text(V.bank_doc(plans, CELL)))
        bank = dict(path=f'{HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p),
                    families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans])
    value = N.seal(dict(
        schema='coopmwm.a69.admission.v1', amendment=AMENDMENT, cell=CELL, rule=ADMIT_RULE, reference=REF,
        disclosure=DISCLOSURE, multiplicity=f'{len(rows)} picks each tested at 95% (disclosed)',
        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), verdicts=rows,
        admitted=[p['family'] for p in plans], bank=bank,
        consequence='LC_BANK arm runs after the A68 arms' if plans else 'no lane-change plan admitted: no LC arm',
        ref_mean_J_veh_s=sum(have[REF][w]['J_veh_s'] for w in worlds)/len(worlds),
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    N.publish(root/ADMISSION_FILE, value)
    return value


# ------------------------------------------------------------------------------------------------ confirmation

def confirm_waves(panel, root, a66_root):
    N = _mods()[0]
    M = mech()
    adm = admission(root)
    if adm is None:
        raise N.Blocked('no A69 admission receipt')
    if not adm['bank']:
        return []
    entry = M.bank_entry(root, LC_BANK)
    if entry['file_sha256'] != adm['bank']['file_sha256']:
        raise N.Blocked('LC_BANK file differs from the admission receipt')
    specs = a44_specs('A44_BANK', panel, root, entry, executor(a66_root), 'a69c_lc_bank')
    if len(specs) != CONFIRM_N:
        raise N.Blocked(f'{ARM}: {len(specs)} specs')
    return [(CONFIRM_WAVE, 'episode', specs)]


def analyse(D, panel, root, a68_root, a66_root):
    import numpy as np
    N, _, A67, A68 = _mods()
    from tools.coopmwm import a42_bigbank as B
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    adm = admission(root)
    ((wave, _, specs),) = confirm_waves(panel, root, a66_root)
    have, sources, roots = {}, {}, {}
    have[ARM], sources[ARM] = published(root, wave, specs)
    roots[ARM] = root
    adm68 = A68.admission(a68_root)
    for arm in REF_ARMS:
        if arm in A68.confirm_arms(adm68):
            have[arm], sources[arm] = D.C37._load(a68_root, f'A68C_{arm}',
                                                  A68.confirm_specs(D, arm, panel, a68_root, a66_root, adm68))
            roots[arm] = a68_root
    if REF not in have:
        raise N.Blocked('no A68 MAPPO60C confirmation records')
    reports = _reports(roots, have, worlds)
    rows = [D._contrast(have, reports, worlds, ARM, r, bootstrap, 'exploratory') for r in REF_ARMS if r in have]
    cands = {c['family']: c['strategy'] for c in candidates()}
    sizes, fam, any_ = [], {}, 0
    for w in worlds:
        rec = have[ARM][w]['a44']
        any_ += 1 if rec.get('activations') else 0
        for f, n in (rec.get('activations_by_family') or {}).items():
            fam[f] = fam.get(f, 0)+int(n)
        sizes += B.realised_sizes(rec, cands)
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    arms[ARM].update(activations=dict(worlds_with_any=any_, total=sum(fam.values()), by_family=fam,
                                      realised_group_size=B.group_size_summary(sizes),
                                      full_group=sum(1 for r in sizes if r['held'] == r['planned'])),
                     lane_changes=lc_counts(have[ARM], worlds))
    return N.seal(dict(schema='coopmwm.a69.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, kind='exploratory (no decision; never merged into the A68 MAIN bank)',
                       admission=dict(receipt_sha256=adm['receipt_sha256'], admitted=adm['admitted'], bank=adm['bank']),
                       contrasts=rows, arms=arms,
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

STAGE = dict(identity='i', verify='v', confirm='c')
STAGE_CAP = dict(identity=IDENT_CAP, verify=VERIFY_CAP, confirm=CONFIRM_CAP)
STAGE_PREFIX = dict(identity='a69i_', verify='a69v_', confirm='a69c_')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('select', 'prepare-ident', 'prepare', 'dry', 'identity', 'identity-compare',
                                       'verify', 'admit', 'confirm', 'analyse', 'status'))
    p.add_argument('--final-root')
    p.add_argument('--a68-root')
    p.add_argument('--a66-root')
    p.add_argument('--stage', choices=('i', 'v', 'c'), default='i')
    p.add_argument('--search')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    if a.command == 'select':
        doc = select(a.search, a.out or CANDS_FILE)
        print(json.dumps(dict(doc_sha256=doc['doc_sha256'], STOP=doc['STOP'], skip_counts=doc['skip_counts'],
                              n_eligible=doc['n_eligible'], n_final=doc['search']['n_final'],
                              candidates=[(c['family'], c['score'], c['n_roles'], c['lc_roles'], c['wm']['feasible'])
                                          for c in doc['candidates']])))
        return
    N = _mods()[0]
    root = Path(a.final_root)
    for other in (a.a68_root, a.a66_root):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A69 final root is its own root')
    which = dict(identity='i', verify='v', confirm='c', prepare='v', admit='v', analyse='c').get(a.command, a.stage)
    if a.command in ('identity-compare', 'prepare-ident'):
        which = 'i'
    panel, D = setup(which)
    M = mech()

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), stage=which), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(stage):
        if stage == 'identity':
            return ident_waves(panel, root, a.a66_root)
        if stage == 'verify':
            return verify_waves(panel, root, a.a66_root, candidates())
        return confirm_waves(panel, root, a.a66_root)

    def brief(ws):
        return {n: dict(n=len(s), first=s[0]['job_id'], last=s[-1]['job_id'], runner_arm=s[0]['arm'],
                        seeds=[s[0]['world']['seed'], s[-1]['world']['seed']], cell=s[0]['world']['cell_key'],
                        checkpoint={k: v for k, v in s[0]['checkpoint'].items() if k != 'label'})
                for n, _, s in ws}
    if a.command == 'prepare-ident':
        print(json.dumps(M.prepare(root), indent=1))
    elif a.command == 'prepare':
        print(json.dumps(prepare(root, candidates()), indent=1))
    elif a.command == 'dry':
        stage = dict(i='identity', v='verify', c='confirm')[which]
        ws = waves_of(stage)
        print(json.dumps(dict(stage=stage, panel_sha256=panel['panel_sha256'], digests=digests(), waves=brief(ws),
                              sim_s=sum(len(s)*N.HORIZON_S for _, _, s in ws), stage_cap=STAGE_CAP[stage],
                              charged=M.charged(root), cap=CAP_SIM_S), indent=1))
    elif a.command in STAGE:
        import fcntl
        if a.command != 'identity' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: nothing is run')
        if a.command == 'verify':
            a68_ref(D, panel, a.a68_root, a.a66_root, 'v')      # the A68 reference wave must be complete
        if a.command == 'confirm':
            a68_ref(D, panel, a.a68_root, a.a66_root, 'c')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(a.command)
            if not waves:
                log(dict(done=a.command, status='SKIPPED_NO_PLAN_ADMITTED'))
                return
            pending = sum(len(s)-N.published(root, w, s, M.check_a44)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            used = M.charged(root, STAGE_PREFIX[a.command])
            if used+need > STAGE_CAP[a.command] or M.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{a.command}: {used} + {need} > stage cap {STAGE_CAP[a.command]} (or phase cap)')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=M.charged(root)))
            # no automatic replay: a failed dispatch raises and is reported
            res = M.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel,
                            runtime=runtime(root))
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(panel, root, a.a66_root)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], non_vacuous=v['non_vacuous'],
                              worlds=[(r['world_id'], r['equal'], r['differences'], r['activations'],
                                       r['container_sha_equal']) for r in v['worlds']])))
    elif a.command == 'admit':
        v = admit(D, panel, root, a.a68_root, a.a66_root, candidates())
        print(json.dumps(dict(admitted=v['admitted'], receipt_sha256=v['receipt_sha256'],
                              rows=[(x['family'], round(x['mean'], 1), [round(c, 1) for c in x['ci95']],
                                     x['group_rule_pass'], x['ADMITTED'], x['lane_changes'])
                                    for x in v['verdicts']])))
    elif a.command == 'analyse':
        v = analyse(D, panel, root, a.a68_root, a.a66_root)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'],
                              rows=[(r['treatment'], r['reference'], r['total_minus_ref']['diff'],
                                     r['total_minus_ref']['ci95']) for r in v['contrasts']],
                              lane_changes=v['arms'][ARM]['lane_changes'])))
    else:   # status
        stage = dict(i='identity', v='verify', c='confirm')[which]
        out = dict(charged=M.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root), stage=stage)
        try:
            for n, _, s in waves_of(stage):
                out[n] = f'{N.published(root, n, s, M.check_a44)[2]}/{len(s)}'
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
