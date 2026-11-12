"""AMENDMENT 71 (S-WIDEADMIT60C; MAIN 2026-11-13; POST-HOC widening of the A68 admission).

Motivated by the world model's unreliable ranking across group sizes; the A68 results stay as reported. Verify in SUMO
EVERY fully scored hold plan the world model predicts beats the base, not only the top 12. Plans, executor, caps and
the MAPPO60C checkpoint (A65 policy_00084, 2abed317) are unchanged; nothing is retuned.

Stages (own root; ledger phase ``wideadmit_a71``; stage caps: admission 1,620,000 sim-s, confirmation 126,720):
  select    zero SUMO: every row of the A68 hold search ``final`` list with n_roles >= 3 and gain_vs_base > 0, minus the
            12 plans already run in the A68 admission (their verdicts are reused as they are), exact-content dedupe
            (same strategy key = same plan content). No other selection; a plan the registered executor refuses cannot
            be run and is recorded as skipped, never replaced.
  verify    each new plan alone (A37_PLAN arm on MAPPO60C) on the 24 A68 admission worlds 992500-992523; the MAPPO60C
            reference is the stored A68 wave A68V_MAPPO60C (read-only, not re-run)
  admit     the A68 rule: (plan - MAPPO60C) total delay CI95 upper < 0 AND the five-group 10 s rule. Then, over the new
            verdicts plus the 12 reused A68 verdicts: a plan whose 24 per-world results equal those of a plan ranked
            earlier by the world model is marked duplicate and cannot take a bank slot; WIDE_BANK = the single best
            admitted non-duplicate plan of each group size 3 / 4 / 5 / 6 by most negative admission CI95 upper (ties:
            strategy key) -> at most 4 plans. Per-size counts (admitted / tested) are recorded.
  confirm   one arm WIDE_BANK_60C on the A70 96-world panel 996300-996395, paired with the stored A70 MAPPO60C and
            DUO_HOLD_60C runs (read-only). Not run if no plan is admitted.
  analyse   primary WIDE_BANK_60C - MAPPO60C: SUPPORTED iff CI95 upper < 0 AND the five-group 10 s rule; the rest is
            descriptive. Paired percentile bootstrap 10,000, seed 20261104, diff = treatment - reference.
No automatic replay: a failed dispatch raises and is reported. No executor / runner behaviour code is changed.

Commands (python ops/a71_wideadmit.py <cmd> --final-root R --a68-root R68 --a70-root R70 --a66-root R66):
  select --search F --out ops/a71/CANDIDATES_D2800_p60.json ; prepare ; dry --stage v|c ; verify ; admit ; confirm ;
  analyse ; status --stage v|c
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

AMENDMENT = '71'
CELL = 'D2800_p60'
PHASE = 'wideadmit_a71'
VERIFY_N, CONFIRM_N = 24, 96
VERIFY_CAP, CONFIRM_CAP = 1_620_000, 126_720
CAP_SIM_S = VERIFY_CAP+CONFIRM_CAP
MIN_ROLES = 3
SIZES = (3, 4, 5, 6)
SETTINGS = dict(v='a68v_D2800_p60', c='a70c_D2800_p60')
RANGES = dict(v='A68_RANGES', c='A70_RANGES')
SEEDS = dict(v=tuple(range(992500, 992524)), c=tuple(range(996300, 996396)))
A68_ADMISSION_RECEIPT = 'd527efbe32c931e29726f4ce1c14fc9743e1ac9f216bf67515ff23f7fc3a5b48'
A71_DIR = HERE/'a71'
CANDS_FILE = A71_DIR/f'CANDIDATES_{CELL}.json'
CANDS_PIN = A71_DIR/f'CANDIDATES_{CELL}.sha256'
BANK_DIR = 'a71_banks'
CANDS_BANK = f'A71_CANDS_{CELL}.json'
WIDE_BANK = f'A71_WIDE_BANK_{CELL}.json'
ADMISSION_FILE = f'A71_ADMISSION_{CELL}.json'
RESULTS_FILE = f'A71_RESULTS_{CELL}.json'
REF = 'MAPPO60C'
ARM = 'WIDE_BANK_60C'
CONFIRM_WAVE = f'A71C_{ARM}'
REF_ARMS = ('MAPPO60C', 'DUO_HOLD_60C')
AUTHORITY = ('MAIN AMENDMENT 71 (S-WIDEADMIT60C, 2026-11-13; POST-HOC widening of the A68 admission): every fully '
             'scored A68 hold-search plan with n_roles >= 3 and WM gain_vs_base > 0 verified in SUMO on the 24 A68 '
             'admission worlds; WIDE_BANK = best admitted plan per group size; one arm on the A70 panel; phase '
             'wideadmit_a71, caps 1,620,000 + 126,720')
DISCLOSURE = ('POST-HOC widening of the A68 admission, motivated by the world model\'s unreliable ranking across group '
              'sizes; the A68 results stay as reported. Multiplicity: about 63 plans are each tested at 95% on the '
              'same 24 worlds, so some passes are expected by chance; admission is a screen, not a result. The '
              'admission bounds that rank plans are selection-biased. The A70 panel was already viewed for SMALL1 / '
              'SMALL2 but played no part in choosing WIDE_BANK (the bank rule uses admission data only and was fixed '
              'before any A71 episode). Never pooled with the A67 / A68 panels')
SELECT_RULE = ('every `final` row of the A68 hold search (WM order = score desc, ties strategy key) with n_roles >= 3 '
               'and gain_vs_base > 0; minus the 12 plans of the A68 admission (verdicts reused as they are, not '
               're-run); exact-content dedupe (same strategy key); a plan the registered executor refuses is skipped '
               'and recorded; no other selection')
ADMIT_RULE = ('(plan - MAPPO60C) total delay CI95 upper < 0 on the 24 A68 admission worlds (a37_cellbank.unc_verdict) '
              'AND the five-group 10 s rule relative to MAPPO60C; reference = the stored A68 wave A68V_MAPPO60C')
BANK_RULE = ('over the new verdicts and the 12 reused A68 verdicts: a plan whose 24 per-world results (per_seed, plan '
             '- MAPPO60C, 0.1 veh-s) equal those of a plan ranked earlier by the world model is marked duplicate and '
             'cannot take a bank slot; WIDE_BANK = for each group size 3 / 4 / 5 / 6 the single admitted non-duplicate '
             'plan with the most negative admission CI95 upper bound (ties: strategy key); at most 4 plans; each '
             'plan keeps its admission lower bound as bank priority; plans and executor fields unchanged')
PRIMARY_RULE = ('SUPPORTED iff WIDE_BANK_60C - MAPPO60C total delay CI95 upper < 0 (paired percentile bootstrap 10,000, '
                'seed 20261104, n = 96) AND the five-group 10 s rule relative to MAPPO60C; diff = treatment - '
                'reference, negative = less delay')


def _mods():
    import nm_finaleval as N
    import a50_fullp60 as A50
    import a67_plans_on_60c as A67
    import a68_research60c as A68
    import a70_smallbank as A70
    return N, A50, A67, A68, A70


def core(which):
    N, A50 = _mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A71',
        HOST_BASE=os.environ.get('A71_HOST_BASE', '/root/a71'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A71_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A71', JOB_PREFIX='a71',
        VERIFY_TAG='a71v', CONFIRM_TAG='a71c', IDENT_TAG='a71i',
        VERIFY_WAVE_PREFIX='A71V_', CONFIRM_WAVE_PREFIX='A71C_',
        RESULTS_FILE=RESULTS_FILE, LOG_FILE='A71_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a71',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 71; POST-HOC widened admission of the A68 hold-search plans on MAPPO60C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a71_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    return m


def setup(which):
    """-> (D, panel, Dref): Dref = the A68 (v) / A70 (c) driver core whose stored records are the references."""
    N, _, _, A68, A70 = _mods()
    Dref, pref = (A68.setup('v') if which == 'v' else A70.setup('c'))
    D = core(which)
    panel = D.setup(SETTINGS[which])
    seeds = tuple(r['world']['seed'] for r in panel['worlds'])
    if seeds != SEEDS[which] or panel['cell']['cell_key'] != CELL or panel['panel_sha256'] != pref['panel_sha256']:
        raise N.Blocked(f'{SETTINGS[which]}: not the registered panel')
    return D, panel, Dref


# ------------------------------------------------------------------------------------------------ candidates

def select(search_path, out):
    """Zero SUMO: the registered candidate list (SELECT_RULE), sealed; never overwrites."""
    N, _, _, A68, _ = _mods()
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    search_path, out = Path(search_path), Path(out)
    raw = search_path.read_bytes()
    search = json.loads(raw.decode('utf-8'))
    if search.get('cell') != CELL:
        raise SystemExit('search cell')
    a68 = {c['key']: c['family'] for c in A68.candidates()}
    rows = sorted(search['final'], key=lambda x: (-float(x['score']), x['key']))
    eligible, cands, skipped, reused, seen = [], [], [], [], {}
    for i, x in enumerate(rows):
        s = x['strategy']
        if B.O.strategy_key(s) != x['key']:
            raise SystemExit(f'{s["name"]}: strategy_key differs from the search key')
        n = len(s['roles'])
        if n < MIN_ROLES or not float(x['gain_vs_base']) > 0:
            continue
        e = dict(wm_rank=i+1, name=s['name'], key=x['key'], score=x['score'], n_roles=n,
                 gain_vs_base=x['gain_vs_base'], descriptor=x.get('descriptor'), comparator=bool(x['comparator']),
                 banned=bool(x['banned']), feasible=x['feasible'])
        eligible.append(e)
        if x['key'] in a68:
            reused.append(dict(e, status='A68 admission verdict reused (not re-run)'))
            continue
        if x['key'] in seen:
            skipped.append(dict(e, skip='duplicate_content', duplicate_of=seen[x['key']]))
            continue
        why = C.executor_refusal(s)
        if why is not None:
            skipped.append(dict(e, skip=why))
            continue
        seen[x['key']] = s['name']
        cands.append(dict(e, family=s['name'], strategy=s, strategy_sha256=C.spec_sha(s),
                          wm=dict({f: x[f] for f in ('feasible', 'gain_vs_base', 'gain_vs_base_lcb', 'gain_vs_inc',
                                                    'worst_group_inc', 'cvar90_inc', 'bounded', 'fair_cal')},
                                  n_states=x['n_states']),
                          selected_rank=len(cands)+1))
    if len(reused) != len(a68):
        raise SystemExit(f'{len(reused)} of the {len(a68)} A68 plans are in the eligible set (expected all)')
    doc = B.seal(dict(
        schema='coopmwm.a71.candidates.v1', amendment=AMENDMENT, cell=CELL, rule=SELECT_RULE,
        search=dict(file_sha256=hashlib.sha256(raw).hexdigest(), schema=search.get('schema'), cell=search.get('cell'),
                    config=search.get('config'), evaluated=search.get('evaluated'), n_states=search.get('n_states'),
                    n_final=len(search['final'])),
        n_eligible=len(eligible), eligible_by_size=dict(sorted(Counter(e['n_roles'] for e in eligible).items())),
        n_reused=len(reused), reused=reused, n_skipped=len(skipped), skipped=skipped,
        n_candidates=len(cands), candidates_by_size=dict(sorted(Counter(c['n_roles'] for c in cands).items())),
        n_distinct_descriptors=len({json.dumps(c['descriptor']) for c in cands}),
        flags=dict(comparator=sum(e['comparator'] for e in eligible), banned=sum(e['banned'] for e in eligible),
                   feasible_0=sum(not float(e['feasible']) > 0 for e in eligible)),
        STOP=not cands, candidates=cands))
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
    if doc['cell'] != CELL or doc.get('amendment') != AMENDMENT or doc['STOP']:
        raise N.Blocked(f'{path.name}: not the registered cell, or no candidate')
    fams = [c['family'] for c in doc['candidates']]
    keys = [c['key'] for c in doc['candidates']]
    if len(set(fams)) != len(fams) or len(set(keys)) != len(keys) or len(fams) != doc['n_candidates']:
        raise N.Blocked(f'{path.name}: {len(fams)} candidates / duplicate family names')
    for c in doc['candidates']:
        if (C.spec_sha(c['strategy']) != c['strategy_sha256'] or C.executor_refusal(c['strategy']) is not None
                or len(c['strategy']['roles']) != c['n_roles'] or c['n_roles'] < MIN_ROLES):
            raise N.Blocked(f'{c["family"]}: strategy sha differs / not an executable hold plan')
    return doc['candidates']


# ------------------------------------------------------------------------------------------------ verification

def prepare(D, root, cands):
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    p = D.bank_path(root, CANDS_BANK)
    text = D.C37.bank_text(doc)
    if p.is_file():
        if p.read_text(encoding='utf-8') != text:
            raise N.Blocked(f'{p}: exists with different content')
    else:
        D.C37.write_once(p, text)
    return dict(path=str(p), file_sha256=N.file_sha(p), n=len(cands))


def verify_waves(D, panel, root, a66_root, cands):
    """[(wave, 'episode', specs)]: every new candidate (WM order) alone on the 24 worlds; the reference is not re-run."""
    N, _, A67, A68, _ = _mods()
    exe = A68.executor(a66_root)
    entry = A68.bank_entry(D, root, CANDS_BANK)
    waves = []
    for c in cands:
        tag = f'a71v_{c["key"]}'
        base = D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=c['family'], tag=tag)
        waves.append((f'A71V_{c["family"]}', 'episode', A67.with_exec(base, exe, tag)))
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def a68_ref(D68, panel, a68_root, a66_root):
    """The stored A68 MAPPO60C records of the 24 admission worlds (read-only; checked against their A68 specs)."""
    A68 = _mods()[3]
    return D68.C37._load(a68_root, 'A68V_MAPPO60C',
                         A68.mappo_specs(D68, panel, a68_root, A68.executor(a66_root), 'a68v_mappo60c'))


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


def bank_choice(rows):
    """Pure (BANK_RULE). rows: verdict rows with family, key, wm_rank, n_roles, PASS, ci95, per_seed.
    -> (chosen families in size order, {family: duplicate_of}, per-size table)."""
    order = sorted(rows, key=lambda r: (r['wm_rank'], r['key']))
    first, dup = {}, {}
    for r in order:
        sig = json.dumps(sorted((str(k), float(v)) for k, v in r['per_seed'].items()))
        if sig in first:
            dup[r['family']] = first[sig]
        else:
            first[sig] = r['family']
    chosen, table = [], {}
    for n in SIZES:
        of = [r for r in rows if r['n_roles'] == n]
        ok = sorted((r for r in of if r['PASS'] and r['family'] not in dup), key=lambda r: (r['ci95'][1], r['key']))
        table[str(n)] = dict(tested=len(of), delay_test_pass=sum(bool(r['ci95'][1] < 0) for r in of),
                             admitted=sum(bool(r['PASS']) for r in of),
                             admitted_non_duplicate=len(ok), duplicates=sum(r['family'] in dup for r in of),
                             best=ok[0]['family'] if ok else None, best_ci95_upper=ok[0]['ci95'][1] if ok else None)
        if ok:
            chosen.append(ok[0]['family'])
    other = [r for r in rows if r['n_roles'] not in SIZES]
    if other:
        raise ValueError('a plan outside the registered group sizes')
    return chosen, dup, table


def admit(D, D68, panel, root, a68_root, a66_root, cands):
    N, _, _, A68, _ = _mods()
    from tools.coopmwm import a37_cellbank as C
    root = Path(root)
    if admission(root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    adm68 = A68.admission(a68_root)
    if adm68 is None or adm68['receipt_sha256'] != A68_ADMISSION_RECEIPT or adm68['panel_sha256'] != panel['panel_sha256']:
        raise N.Blocked('not the registered A68 admission receipt / panel')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources, roots = {}, {}, {}
    have[REF], sources[REF] = a68_ref(D68, panel, a68_root, a66_root)
    roots[REF] = a68_root
    for (name, _, specs), c in zip(verify_waves(D, panel, root, a66_root, cands), cands):
        have[c['family']], sources[c['family']] = D.C37._load(root, name, specs)
        roots[c['family']] = root
    reports = _reports(roots, have, worlds)
    smap = D._spec_map(cands)
    rows = []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, REF, bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have[REF][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a37'] for w in worlds]
        v.update(family=fam, source='A71 (new run)', wm_rank=cand['wm_rank'], selected_rank=cand['selected_rank'],
                 key=cand['key'], wm_score=cand['score'], wm_gain_vs_base=cand['gain_vs_base'],
                 n_roles=len(cand['strategy']['roles']), reference=REF,
                 convention='diff = plan - MAPPO60C (negative = less delay); gain = -diff',
                 delay_test_pass=bool(v['ci95'][1] < 0), group_rule_pass=bool(v['allowed']),
                 cvar90_vs_ref=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_ref_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 realised_group_size=D._group_sizes(have[fam], worlds, smap),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rows.append(v)
    c68 = {c['family']: c for c in A68.candidates()}
    old = []
    for x in adm68['verdicts']:
        if x['n_roles'] < MIN_ROLES or x['family'] not in c68:
            raise N.Blocked(f'{x["family"]}: A68 verdict outside the registered set')
        old.append(dict(x, source='A68 admission (reused as it is)', PASS=bool(x['MAIN'])))
    allrows = old+rows
    if len({r['family'] for r in allrows}) != len(allrows):
        raise N.Blocked('a plan appears twice')
    chosen, dup, table = bank_choice(allrows)
    for r in allrows:
        r['duplicate_of'] = dup.get(r['family'])
        r['WIDE_BANK'] = r['family'] in chosen
    allc = list(c68.values())+list(cands)
    plans = C.admitted_plans(allc, {r['family']: dict(r, PASS=r['family'] in chosen) for r in allrows})
    bank = None
    if plans:
        p = D.C37.write_once(D.bank_path(root, WIDE_BANK), D.C37.bank_text(C.bank_doc(plans, CELL)))
        bank = dict(path=f'{D.HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p),
                    families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans],
                    sizes={x['family']: len(x['strategy']['roles']) for x in plans})
    value = N.seal(dict(
        schema='coopmwm.a71.admission.v1', amendment=AMENDMENT, cell=CELL, rule=ADMIT_RULE, bank_rule=BANK_RULE,
        reference=REF, disclosure=DISCLOSURE,
        multiplicity=f'{len(allrows)} plans each tested at 95% on the same 24 worlds ({len(rows)} new, {len(old)} '
                     'reused A68 verdicts); some passes are expected by chance; admission is a screen, not a result',
        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()),
        a68_admission_receipt_sha256=A68_ADMISSION_RECEIPT, n_new=len(rows), n_reused=len(old),
        per_size=table, admitted=[r['family'] for r in allrows if r['PASS']],
        n_admitted=sum(bool(r['PASS']) for r in allrows), n_delay_pass=sum(bool(r['ci95'][1] < 0) for r in allrows),
        duplicates=dup, wide_bank=chosen, bank=bank, verdicts=allrows,
        consequence='WIDE_BANK_60C arm runs on the A70 panel' if plans else 'no plan admitted: no confirmation',
        ref_mean_J_veh_s=sum(have[REF][w]['J_veh_s'] for w in worlds)/len(worlds),
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    N.publish(root/ADMISSION_FILE, value)
    return value


# ------------------------------------------------------------------------------------------------ confirmation

def confirm_waves(D, panel, root, a66_root):
    N, _, _, A68, _ = _mods()
    adm = admission(root)
    if adm is None or not adm['bank']:
        raise N.Blocked('no admission receipt / no plan admitted: no confirmation')
    specs = A68.bank_specs(D, panel, root, WIDE_BANK, A68.executor(a66_root), 'a71c_wide_bank_60c',
                           adm['bank']['file_sha256'])
    if len(specs) != CONFIRM_N:
        raise N.Blocked(f'{ARM}: {len(specs)} specs')
    return [(CONFIRM_WAVE, 'episode', specs)]


def analyse(D, D70, panel, root, a70_root, a66_root):
    import numpy as np
    N, _, A67, _, A70 = _mods()
    from tools.coopmwm import oracle_plan as O
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    adm = admission(root)
    sel70 = A70.selection(a70_root)
    have, sources, roots = {}, {}, {}
    for arm in REF_ARMS:
        have[arm], sources[arm] = D70.C37._load(a70_root, f'A70C_{arm}',
                                                A70.confirm_specs(D70, arm, panel, a70_root, a66_root, sel70))
        roots[arm] = a70_root
    (wave, _, specs), = confirm_waves(D, panel, root, a66_root)
    have[ARM], sources[ARM] = D.C37._load(root, wave, specs)
    roots[ARM] = root
    reports = _reports(roots, have, worlds)
    row = D._contrast(have, reports, worlds, ARM, REF, bootstrap, 'primary')
    ci = row['total_minus_ref']['ci95']
    groups = row['groups'] or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    primary = dict(rule=PRIMARY_RULE, contrast=f'{ARM} - MAPPO60C', diff=row['total_minus_ref']['diff'],
                   ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass,
                   group_rule_pass=b_pass, supported=bool(a_pass and b_pass),
                   verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED')
    rows = [row, D._contrast(have, reports, worlds, ARM, 'DUO_HOLD_60C', bootstrap, 'secondary_descriptive'),
            D._contrast(have, reports, worlds, 'DUO_HOLD_60C', REF, bootstrap, 'secondary_descriptive')]
    pspecs = {p['family']: p['spec'] for p in O.bank_plans(N.read(D.bank_path(root, WIDE_BANK)), CELL)[0]}
    arms = {}
    for a in have:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        if a != REF:
            arms[a]['activations_per_world'] = sum(int((have[a][w].get('a37') or have[a][w].get('a32') or {})
                                                       .get('activations') or 0) for w in worlds)/len(worlds)
    arms[ARM]['activations'] = A67.activation_table(have[ARM], worlds, pspecs)
    return N.seal(dict(schema='coopmwm.a71.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, admission=dict(receipt_sha256=adm['receipt_sha256'], bank=adm['bank'],
                                                           per_size=adm['per_size'], wide_bank=adm['wide_bank']),
                       arms_run=[ARM], arms_reused=dict(root=str(a70_root), arms=list(REF_ARMS)), root=str(root),
                       primary=primary, contrasts=rows, arms=arms,
                       multiplicity='one registered primary; every other contrast is descriptive',
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

STAGE = dict(verify='v', confirm='c')
STAGE_CAP = dict(verify=VERIFY_CAP, confirm=CONFIRM_CAP)
STAGE_PREFIX = dict(verify='a71v_', confirm='a71c_')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('select', 'prepare', 'dry', 'verify', 'admit', 'confirm', 'analyse', 'status'))
    p.add_argument('--final-root')
    p.add_argument('--a68-root')
    p.add_argument('--a70-root')
    p.add_argument('--a66-root')
    p.add_argument('--search')
    p.add_argument('--stage', choices=('v', 'c'), default='v')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    N = _mods()[0]
    if a.command == 'select':
        doc = select(a.search, a.out or CANDS_FILE)
        print(json.dumps({k: doc[k] for k in ('doc_sha256', 'n_eligible', 'eligible_by_size', 'n_reused', 'n_skipped',
                                              'n_candidates', 'candidates_by_size', 'n_distinct_descriptors',
                                              'flags')}))
        return
    root = Path(a.final_root)
    for other in (a.a68_root, a.a70_root, a.a66_root):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A71 final root is a new root')
    which = dict(verify='v', admit='v', prepare='v', confirm='c', analyse='c').get(a.command, a.stage)
    D, panel, Dref = setup(which)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTINGS[which]),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(stage):
        if stage == 'verify':
            return verify_waves(D, panel, root, a.a66_root, candidates())
        return confirm_waves(D, panel, root, a.a66_root)

    def brief(ws):
        return dict(n_waves=len(ws), first={n: dict(n=len(s), first=s[0]['job_id'], last=s[-1]['job_id'],
                                                   runner_arm=s[0]['arm'],
                                                   seeds=[s[0]['world']['seed'], s[-1]['world']['seed']],
                                                   cell=s[0]['world']['cell_key'],
                                                   checkpoint={k: v for k, v in s[0]['checkpoint'].items()
                                                               if k != 'label'})
                                               for n, _, s in ws[:1]},
                    waves=[n for n, _, _ in ws])
    if a.command == 'prepare':
        print(json.dumps(prepare(D, root, candidates())))
    elif a.command == 'dry':
        stage = dict(v='verify', c='confirm')[which]
        ws = waves_of(stage)
        print(json.dumps(dict(stage=stage, panel_sha256=panel['panel_sha256'], **brief(ws),
                              sim_s=sum(len(s)*N.HORIZON_S for _, _, s in ws), stage_cap=STAGE_CAP[stage],
                              charged=D.charged(root), cap=CAP_SIM_S), indent=1))
    elif a.command in STAGE_CAP:
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(a.command)
            pending = sum(len(s)-N.published(root, w, s, D.C37.check_a37)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            used = D.charged(root, STAGE_PREFIX[a.command])
            if used+need > STAGE_CAP[a.command] or D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{a.command}: {used} + {need} > stage cap {STAGE_CAP[a.command]} (or phase cap)')
            log(dict(start=a.command, n_waves=len(waves), episodes=sum(len(w[2]) for w in waves), workers=a.workers,
                     chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay: a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'admit':
        v = admit(D, Dref, panel, root, a.a68_root, a.a66_root, candidates())
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n_new=v['n_new'], n_reused=v['n_reused'],
                              n_admitted=v['n_admitted'], n_delay_pass=v['n_delay_pass'], per_size=v['per_size'],
                              wide_bank=v['wide_bank'], bank=v['bank'], duplicates=v['duplicates'],
                              consequence=v['consequence'])))
    elif a.command == 'analyse':
        v = analyse(D, Dref, panel, root, a.a70_root, a.a66_root)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        stage = dict(v='verify', c='confirm')[which]
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, stage=stage, admission=admission(root) is not None)
        try:
            ws = waves_of(stage)
            done = [N.published(root, n, s, D.C37.check_a37)[2] for n, _, s in ws]
            out.update(waves=len(ws), waves_complete=sum(d == len(s) for d, (_, _, s) in zip(done, ws)),
                       episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
