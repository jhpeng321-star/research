"""AMENDMENT 42 (MAIN 2026-11-10 ~22:20; registered before any A42 search, SUMO or outcome): WM search for 3-6 CAV group
plans at D2800_p60, verified in SUMO against the hand-written hold.

Zero-physics helpers only; the executor is the amendment-37 executor UNCHANGED (tools.coopmwm.a37_cellbank:
CellBankCore = oracle_plan.OracleCore with the BANK_ALWAYS start rule; runner arms A37_PLAN / CELL_BANK; a37 digest
re-verified by the runner). BIG_BANK is the runner arm CELL_BANK on the A42 bank doc (cell D2800_p60).

- Candidates (registered text, verbatim): "`final` list by WM score desc, skip comparator/banned rows and plans the
  registered executor refuses, top K = 6 (as R7B). If fewer than 1 plan has >= 3 roles and feasibility > 0, STOP:
  reported as 'search proposes no usable larger group'." MAIN ruling (REGISTRATION.md 2026-11-10 22:40, pre-outcome):
  "(as R7B)" also skips rows with feasibility 0 (never activated on the final states). The ranking and row form are
  a37_cellbank.select_candidates' (re-implemented here with the extra skip; the a37 module is untouched); the STOP rule
  is evaluated on the K selected candidates (the plans that go to SUMO).
- Admission: a37_cellbank.unc_verdict / admitted_plans (the A37 UNC rule, same functions).
- Realised group size (registered secondary): per activation, the members that actually held = members whose release
  came after their role onset (or that were never released inside the episode).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

from . import a37_cellbank as C
from . import oracle_plan as O

SCHEMA = 'coopmwm.a42_bigbank.v1'
CELL = 'D2800_p60'
K = 6
MIN_ROLES = 3
STOP_TEXT = 'search proposes no usable larger group'
STOP_RULE = 'fewer than 1 candidate plan with >= 3 roles and WM feasibility > 0 -> STOP'


def usable_larger(c):
    return len(c['strategy']['roles']) >= MIN_ROLES and float(c['wm']['feasible']) > 0


NEVER_ACTIVATED = 'never_activated'
RULE_TEXT = ('search `final` list by WM score desc (ties: strategy key), skip comparator / banned rows, plans the '
             'registered executor refuses (a37_cellbank.executor_refusal) and rows with feasibility 0 (never activated '
             'on the final states; MAIN ruling recorded in REGISTRATION.md 2026-11-10 22:40, pre-outcome: this is what '
             '"(as R7B)" means), top K = 6')


def select_candidates(search, cell=CELL, k=K):
    """a37_cellbank.select_candidates (same ranking, same row form, a37 module untouched) plus the registered R7B skip
    of rows with feasibility 0. Skip reason precedence: comparator, banned, executor refusal class, never_activated."""
    if search.get('cell') != cell:
        raise ValueError(f'search cell {search.get("cell")!r} is not {cell}')
    rows = sorted(search['final'], key=lambda x: (-float(x['score']), x['key']))
    ranking, skipped, cands = [], [], []
    for i, x in enumerate(rows):
        s = x['strategy']
        if O.strategy_key(s) != x['key']:
            raise ValueError(f'{s["name"]}: strategy_key differs from the search key')
        why = ('comparator' if x['comparator'] else 'banned' if x['banned'] else C.executor_refusal(s))
        if why is None and not float(x['feasible']) > 0:
            why = NEVER_ACTIVATED
        entry = dict(wm_rank=i+1, name=s['name'], key=x['key'], score=x['score'], skip=why)
        ranking.append(entry)
        if why is not None:
            skipped.append(entry)
            continue
        if len(cands) < k:
            cands.append(dict(entry, family=s['name'], strategy=s, strategy_sha256=C.spec_sha(s),
                              wm=dict({f: x[f] for f in ('feasible', 'gain_vs_base', 'gain_vs_base_lcb', 'gain_vs_inc',
                                                        'worst_group_inc', 'cvar90_inc', 'bounded', 'fair_cal')},
                                      n_states=x['n_states']),
                              selected_rank=len(cands)+1))
    return dict(cell=cell, k=k, candidates=cands, skipped=skipped, ranking=ranking)


def select(search, file_sha256):
    """Registered A42 selection (zero SUMO, WM ranking only) -> sealed-able dict with the STOP verdict."""
    sel = select_candidates(search, CELL, k=K)
    for c in sel['candidates']:
        c['n_roles'] = len(c['strategy']['roles'])
        c['usable_larger_group'] = usable_larger(c)
    n_usable = sum(c['usable_larger_group'] for c in sel['candidates'])
    # disclosure only (does not change the rule): how many rows of the whole `final` list would satisfy the predicate
    whole = sum(1 for x in search['final'] if not x['comparator'] and not x['banned']
                and C.executor_refusal(x['strategy']) is None
                and len(x['strategy']['roles']) >= MIN_ROLES and float(x['feasible']) > 0)
    stop = n_usable < 1
    by_reason = Counter(e['skip'] for e in sel['skipped'])
    diverse = select_diverse(search, sel)
    return dict(schema=SCHEMA+'.candidates', amendment='42', cell=CELL, k=K, diverse=diverse,
                search=dict(file_sha256=file_sha256, schema=search.get('schema'), cell=search.get('cell'),
                            config=search.get('config'), evaluated=search.get('evaluated'),
                            n_states=search.get('n_states'), n_final=len(search['final'])),
                rule=RULE_TEXT, stop_rule=STOP_RULE, n_usable_larger_in_candidates=n_usable,
                n_usable_larger_in_whole_final_list=whole, STOP=stop, verdict=STOP_TEXT if stop else 'CONTINUE',
                skip_counts=dict(sorted(by_reason.items())), n_skipped=len(sel['skipped']),
                n_eligible=len(sel['ranking'])-len(sel['skipped']), n_selected=len(sel['candidates']),
                candidates=sel['candidates'], skipped=sel['skipped'], ranking=sel['ranking'])


# ------------------------------------------------------------------------------------------------ AMENDMENT 45
#: the search descriptor is the tuple tools.wzprop.strategy_search.descriptor(s) (a JSON list in `final`):
#: (n_roles, n_lanes, n_lc, speed_class, staggered, zone, trigger, gate, start_gate, phases)
DESCRIPTOR_FIELDS = ('n_roles', 'n_lanes', 'n_lc', 'speed_class', 'staggered', 'zone', 'trigger', 'gate', 'start_gate',
                     'phases')
CLASS_FIELDS = ('n_roles', 'n_lanes', 'staggered', 'zone', 'phases')
DIVERSE_CAP = 12
DIVERSE_RULE = ('AMENDMENT 45: among eligible `final` rows (same skips as the A42 rule) NOT in the A42 top 6, group by '
                'structural class (n_roles, n_lanes, staggered, zone, phases) read from the row descriptor; per class '
                'the best row by WM score; classes ordered by that score; at most 12; ties by the order in `final`')


def structural_class(row):
    """The A45 class of a search `final` row from its descriptor; the descriptor is cross-checked against the
    strategy (fail closed on a mismatch)."""
    d = row['descriptor']
    if len(d) != len(DESCRIPTOR_FIELDS):
        raise ValueError(f'{row["key"]}: descriptor has {len(d)} fields, expected {len(DESCRIPTOR_FIELDS)}')
    named = dict(zip(DESCRIPTOR_FIELDS, d))
    s = row['strategy']
    on = [int(r['onset_s']) for r in s['roles']]
    expect = dict(n_roles=len(s['roles']),
                  n_lanes=len({r['lane'] if r.get('lc_target') is None else r['lc_target'] for r in s['roles']}),
                  staggered=int(max(on) > min(on)), zone=s['zone'], phases=s.get('phases') is not None)
    got = dict(n_roles=int(named['n_roles']), n_lanes=int(named['n_lanes']), staggered=int(named['staggered']),
               zone=named['zone'], phases=bool(named['phases']))
    if got != expect:
        raise ValueError(f'{row["key"]}: descriptor {got} differs from the strategy {expect}')
    return got


def class_key(cls):
    return '|'.join(f'{f}={cls[f]}' for f in CLASS_FIELDS)


def select_diverse(search, sel, cap=DIVERSE_CAP):
    """A45 diverse candidates from the SAME search `final` and the A42 selection ``sel`` (select_candidates output).
    Eligible = not skipped by the A42 rule; the A42 top-K rows are never selected again. Deterministic."""
    skip_of = {e['key']: e['skip'] for e in sel['ranking']}
    rank_of = {e['key']: e['wm_rank'] for e in sel['ranking']}
    top = {c['key'] for c in sel['candidates']}
    order = sorted(range(len(search['final'])), key=lambda i: (-float(search['final'][i]['score']), i))
    best, members, counts = {}, {}, dict(not_eligible=0, in_top_k=0, not_class_best=0, over_cap=0)
    for i in order:
        x = search['final'][i]
        if skip_of[x['key']] is not None:
            counts['not_eligible'] += 1
            continue
        ck = class_key(structural_class(x))
        if x['key'] in top:
            counts['in_top_k'] += 1
            members.setdefault(ck, dict(eligible_not_top=0, in_top_k=0))['in_top_k'] += 1
            continue
        m = members.setdefault(ck, dict(eligible_not_top=0, in_top_k=0))
        m['eligible_not_top'] += 1
        if ck in best:
            counts['not_class_best'] += 1
            continue
        best[ck] = (i, x)
    classes = list(best.items())            # insertion order = (-score, final index) of the class best
    counts['over_cap'] = max(0, len(classes)-cap)
    out = []
    for ck, (i, x) in classes[:cap]:
        s = x['strategy']
        out.append(dict(wm_rank=rank_of[x['key']], final_index=i, name=s['name'], key=x['key'], score=x['score'],
                        skip=None, family=s['name'], strategy=s, strategy_sha256=C.spec_sha(s),
                        wm=dict({f: x[f] for f in ('feasible', 'gain_vs_base', 'gain_vs_base_lcb', 'gain_vs_inc',
                                                  'worst_group_inc', 'cvar90_inc', 'bounded', 'fair_cal')},
                                n_states=x['n_states']),
                        class_key=ck, structural_class=structural_class(x), class_counts=members[ck],
                        diverse_rank=len(out)+1, n_roles=len(s['roles'])))
    fams = [c['family'] for c in out]+[c['family'] for c in sel['candidates']]
    if len(set(fams)) != len(fams):
        raise ValueError('A45: family names of the diverse and top-K candidates collide')
    return dict(amendment='45', rule=DIVERSE_RULE, cap=cap, descriptor_fields=list(DESCRIPTOR_FIELDS),
                class_fields=list(CLASS_FIELDS), n_classes=len(classes),
                n_classes_with_top_k_rows=sum(1 for m in members.values() if m['in_top_k']),
                skip_counts=counts, n_selected=len(out), candidates=out)


def doc_sha(doc):
    body = {k: v for k, v in doc.items() if k != 'doc_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def seal(doc):
    return dict(doc, doc_sha256=doc_sha(doc))


def shape_check(strategies, cell_key=CELL):
    """Can the registered executor (A37 CellBankCore / oracle_plan.bank_plans) admit these plans? Builds the A37 bank
    doc per plan and reports admitted vs refused (refusal class); never changes the executor."""
    out = []
    for s in strategies:
        why = C.executor_refusal(s)
        row = dict(name=s['name'], key=O.strategy_key(s), trigger=s['trigger'], zone=s['zone'], n_roles=len(s['roles']),
                   lanes_ranks=[[r['lane'], r['rank']] for r in s['roles']], pre_refusal=why)
        if why is None:
            bank = C.bank_doc([dict(family=s['name'], strategy=s, lb95_veh_s=0.0)], cell_key)
            core = C.CellBankCore(bank, cell_key)
            row.update(admitted=[p['family'] for p in core.plans] == [s['name']], refused=core.refused)
        else:
            row.update(admitted=False, refused={why: [s['name']]})
        out.append(row)
    return out


def realised_sizes(receipt, specs):
    """receipt: the a37 receipts of one episode (record['a37']); specs: {family: strategy}. -> list of dict(family,
    planned, held) per activation; a member held iff its release elapsed time > its role onset or it was never
    released (episode end)."""
    out = []
    for a in receipt.get('acts') or []:
        s = specs[a['family']]
        roles = O.active_roles(s, 0)
        rel = a.get('release') or {}
        held = 0
        for vid, role in zip(a['members'], roles):
            r = rel.get(vid)
            if r is None or float(r[0]) > int(role['onset_s']):
                held += 1
        out.append(dict(family=a['family'], planned=len(a['members']), held=held))
    return out


def group_size_summary(rows):
    n = len(rows)
    planned = Counter(r['planned'] for r in rows)
    held = Counter(r['held'] for r in rows)
    return dict(activations=n, planned_size_counts={str(k): v for k, v in sorted(planned.items())},
                held_size_counts={str(k): v for k, v in sorted(held.items())},
                share_held_ge3=(sum(1 for r in rows if r['held'] >= 3)/n) if n else None,
                n_held_ge3=sum(1 for r in rows if r['held'] >= 3))
