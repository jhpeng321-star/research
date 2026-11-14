"""E1-FIX (Amendment 74), stages 3-5: verification addendum, admission, budget-6 banks, confirmation waves and the
WM-FIX vs Random analysis (zero SUMO; this file launches nothing). POST HOC REPAIR: it never replaces the registered
E1 null, is outside the A72 Holm family, and never writes a registered or WM-ACT file (everything goes to
out/e1/fix/).

Reads (never writes): out/e1/pool/POOL_seed<N>.json, out/e1/order/ORDER_WMFIX_seed<N>.json (e1_order_fix.py, written
only after gate G0 passed), out/e1/verify/VERIFY_PLAN.json, the registered VERIFY_RESULTS_seed<N>.json,
FROZEN_BANKS.json, ADMISSION_seed<N>.json, CONFIRM_RESULTS.json (the Random arm is the registered one, reused as it
is), and the WM-ACT addendum when it exists: act/VERIFY_PLAN_ACT.json, act/VERIFY_RESULTS_ACT_seed<N>.json,
act/ADMISSION_ACT_seed<N>.json, act/FROZEN_BANKS_ACT.json, act/CONFIRM_RESULTS_ACT.json.

  plan     per seed the WM-FIX first six; a candidate whose 24 plan x world episodes already ran is REUSED
           ('registered' = the registered E1 verification, 'act' = the WM-ACT addendum, only if its results file
           holds that candidate); the others get one PLAN wave each on the same 24 worlds of block e1v_seed<N>, plus
           that block's registered U0 wave (same name and worlds: a root that already published it re-runs nothing).
           Cap: new plans <= 36 x 28,800 = 1,036,800 sim-s.  -> fix/banks/E1_FIX_CANDS_seed<N>.json,
           fix/VERIFY_PLAN_FIX.json
  admit    the registered rule (e1_bank.admit_seed = a37 unc_verdict); every U0 used (WM-ACT addendum, FIX addendum)
           must equal the registered U0 world by world, exactly.  -> fix/ADMISSION_FIX_seed<N>.json
  build    budget-6 WM-FIX bank per seed (e1_bank.bank_for, executor order), admitted-count contrast WM-FIX - Random
           (exact paired sign-flip), confirmation waves = registered CONF_U0 + one wave per WM-FIX bank whose content
           neither the registered confirmation nor the collected WM-ACT confirmation already ran.
           Cap: <= 6 x 96 x 1200 = 691,200 sim-s (+ <= 115,200 U0 on another root).
           -> fix/banks/E1_BANK_seed<N>_WMFIX_b6.json, fix/FROZEN_BANKS_FIX.json
  analyse  bank total delay on the 96 confirmation worlds: WM-FIX bank vs the registered Random bank (seed x world
           bootstrap 10,000 / seed 20261104 and exact sign-flip over the six seeds, as registered E1), each bank vs
           U0.  -> fix/E1_FIX_ANALYSIS.json
  has-waves  exit 0 if the block (--seed N, or --confirm) has something to run in this plan, 10 if not.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import e1_bank as B  # noqa: E402
import e1_order as E  # noqa: E402
import e1_verify as V  # noqa: E402
import stats as S  # noqa: E402

ARM = 'WMFIX'
LABEL = 'POST HOC REPAIR (Amendment 74 E1-FIX); never replaces the registered E1 null'
NEVER = ('never replaces the registered E1 result (WM vs Random admitted count, FROZEN_BANKS.primary_admitted_count); '
         'outside the A72 Holm family')
BUDGET = B.PRIMARY_PREFIX
EP_S = L.N_VERIFY_WORLDS*L.HORIZON_S                    # 28,800 sim-s per verified plan
CAP_VERIFY_S = 36*EP_S                                  # 1,036,800
CAP_CONFIRM_S = 691_200
CAP_CONFIRM_U0_S = 115_200


def fdir(out):
    return Path(out)/'e1'/'fix'


def act_dir(a):
    return Path(a.act_dir or Path(a.out)/'e1'/'act')


def cand_bank_name(seed):
    return f'E1_FIX_CANDS_seed{int(seed)}.json'


def bank_name(seed):
    return f'E1_BANK_seed{int(seed)}_{ARM}_b{BUDGET}.json'


def exact_u0(seed, what, u0, ref):
    """The pairing is the registered one only if this U0 equals the registered U0 world by world, exactly."""
    if sorted(u0) != sorted(ref):
        raise L.Refused(f'seed {seed}: {what} U0 worlds differ from the registered U0 worlds')
    worst = max(abs(float(u0[w])-float(ref[w])) for w in ref)
    if worst != 0.0:
        raise L.Refused(f'seed {seed}: {what} U0 differs from the registered U0 (max |dJ| {worst} veh-s): '
                        'the pairing is not the registered one')


# ------------------------------------------------------------------------------------------------------- plan
def fix_seed_plan(top, pool, reg_sp, act_done=None):
    """top = WM-FIX first-six hashes; pool = sealed POOL doc; reg_sp = that seed of the registered VERIFY_PLAN;
    act_done = {desc_hash: family} of the WM-ACT addendum candidates whose verification results exist.
    -> (seed plan, candidate bank doc or None). Pure. Wave names follow the registered convention E1V<seed>_<key>."""
    seed, block = int(reg_sp['search_seed']), reg_sp['block']
    act_done = dict(act_done or {})
    if pool['doc_sha256'] != reg_sp['pool_doc_sha256']:
        raise L.Refused(f'seed {seed}: the registered verification planned another pool')
    by_hash = {e['desc_hash']: e for e in pool['pool']}
    done = {c['desc_hash']: c for c in reg_sp['candidates']}
    if len(top) != L.TOP_K or len(set(top)) != L.TOP_K:
        raise L.Refused(f'seed {seed}: WM-FIX top is not {L.TOP_K} distinct plans')
    cands = []
    for rank, h in enumerate(top, 1):
        e = by_hash[h]
        src = 'registered' if h in done else 'act' if h in act_done else 'fix'
        if src == 'act' and act_done[h] != e['strategy']['name']:
            raise L.Refused(f'seed {seed}: WM-ACT verified {h[:12]} under another family name')
        cands.append(dict(desc_hash=h, key=e['key'], family=e['strategy']['name'], size=e['size'],
                          category=e['category'], selected_by={ARM: rank}, verified=src,
                          registered_selected_by=done[h]['selected_by'] if h in done else None))
    new = [c for c in cands if c['verified'] == 'fix']
    if {c['family'] for c in new} & ({c['family'] for c in reg_sp['candidates']} | set(act_done.values())):
        raise L.Refused(f'seed {seed}: a new candidate reuses an already verified family name')
    waves, bank, bank_ref = [], None, None
    if new:
        (u0,) = [w for w in reg_sp['waves'] if w['kind'] == 'U0']
        bank = L.bank_doc([dict(family=c['family'], strategy=by_hash[c['desc_hash']]['strategy'], lb95_veh_s=0.0)
                           for c in new])
        bank_ref = dict(name=cand_bank_name(seed), sha256=L.sha_bytes(L.text_of(bank).encode()))
        waves = [u0]+[dict(wave=f'E1V{seed}_{c["key"]}', kind='PLAN', block=block, n_worlds=L.N_VERIFY_WORLDS,
                           bank=bank_ref['name'], family=c['family'], desc_hash=c['desc_hash'],
                           cache_keys=[V.cache_key(c['desc_hash'], block, i) for i in range(L.N_VERIFY_WORLDS)])
                      for c in new]
    n = {s: sum(1 for c in cands if c['verified'] == s) for s in ('registered', 'act', 'fix')}
    return dict(search_seed=seed, block=block, world_seeds=reg_sp.get('world_seeds'),
                pool_doc_sha256=pool['doc_sha256'], top=list(top), candidates=cands, n_new=n['fix'],
                n_reused=n['registered']+n['act'], n_reused_registered=n['registered'], n_reused_act=n['act'],
                bank=bank_ref, waves=waves, episodes_new_plans=n['fix']*L.N_VERIFY_WORLDS,
                sim_s_new_plans=n['fix']*EP_S, sim_s_u0_if_not_published_in_root=EP_S if new else 0), bank


def act_verified(adir, seed, reg_sp, top=()):
    """{desc_hash: family} of the WM-ACT addendum candidates of this seed whose collected results exist, plus the
    identity of what was read. The WM-ACT plan must be built on the same registered plan. A WM-FIX candidate the
    WM-ACT addendum planned but has not collected is refused (its wave would be planned twice): collect WM-ACT
    first, or use --no-act on a root that never ran the WM-ACT addendum."""
    pp, rp = Path(adir)/'VERIFY_PLAN_ACT.json', Path(adir)/f'VERIFY_RESULTS_ACT_seed{int(seed)}.json'
    if not pp.is_file():
        return {}, None
    plan = L.verify_seal(L.read_json(pp), what='VERIFY_PLAN_ACT')
    sps = [sp for sp in plan['seeds'] if sp['search_seed'] == int(seed)]
    if not sps:
        return {}, None
    if sps[0]['pool_doc_sha256'] != reg_sp['pool_doc_sha256'] or sps[0]['block'] != reg_sp['block']:
        raise L.Refused(f'seed {seed}: the WM-ACT plan was built on another pool or block')
    add = {c['desc_hash']: c['family'] for c in sps[0]['candidates'] if c['verified'] == 'addendum'}
    res = L.read_json(rp) if rp.is_file() else dict(plans={})
    got = {h: f for h, f in add.items() if f in res['plans']}
    pending = [h for h in top if h in add and h not in got]
    if pending:
        raise L.Refused(f'seed {seed}: {len(pending)} WM-FIX candidate(s) are planned by the WM-ACT addendum but '
                        'its results are not collected')
    return got, dict(verify_plan_act_sha256=plan['doc_sha256'], act_results_sha256=L.sha_file(rp)) if got else None


def cmd_plan(a):
    reg = L.verify_seal(L.read_json(Path(a.out)/'e1'/'verify'/'VERIFY_PLAN.json'), what='VERIFY_PLAN')
    if reg.get('symbolic'):
        raise L.Refused('the registered VERIFY_PLAN is symbolic (no world seeds bound)')
    seeds, rep = [], []
    for reg_sp in reg['seeds']:
        seed = reg_sp['search_seed']
        if seed not in a.seeds:
            continue
        pool = L.verify_seal(L.read_json(E.pool_file(a.out, seed)), what=f'POOL seed {seed}')
        od = L.verify_seal(L.read_json(E.order_path(a.out, ARM, seed)), what=f'ORDER {ARM} seed {seed}')
        if (od['arm'] != ARM or od['search_seed'] != seed or od['pool_doc_sha256'] != pool['doc_sha256']
                or (od.get('g0') or {}).get('verdict') != 'PASS'):
            raise L.Refused(f'ORDER {ARM} seed {seed}: not the G0-passed WM-FIX order of this pool')
        done, ident = ({}, None) if a.no_act else act_verified(act_dir(a), seed, reg_sp, od['top'])
        sp, bank = fix_seed_plan(od['top'], pool, reg_sp, done)
        sp.update(order_doc_sha256=od['doc_sha256'], act_reuse=ident)
        by_hash = {e['desc_hash']: e for e in pool['pool']}
        new = [by_hash[c['desc_hash']]['strategy'] for c in sp['candidates'] if c['verified'] == 'fix']
        cc = L.crosscheck(a.code_root, new) if a.code_root and new else None
        state = None
        if bank is not None:
            state = 'dry-run' if a.dry_run else L.write_once(fdir(a.out)/'banks'/sp['bank']['name'], L.text_of(bank))
        seeds.append(sp)
        rep.append(dict(seed=seed, new=sp['n_new'], reused_registered=sp['n_reused_registered'],
                        reused_act=sp['n_reused_act'], sim_s_new_plans=sp['sim_s_new_plans'], bank=state,
                        crosscheck=None if cc is None else cc['status']))
    if len(seeds) != len(a.seeds):
        raise L.Refused('a requested seed is not in the registered VERIFY_PLAN')
    new_s = sum(sp['sim_s_new_plans'] for sp in seeds)
    u0_s = sum(sp['sim_s_u0_if_not_published_in_root'] for sp in seeds)
    if new_s > CAP_VERIFY_S:
        raise L.Refused(f'verification {new_s} sim-s exceeds the registered cap {CAP_VERIFY_S}')
    doc = L.seal(dict(
        schema='claim_validation.e1.fix.verify_plan.v1', run_id=L.RUN_ID, cell=L.CELL, arm=ARM, label=LABEL,
        never_replaces=NEVER, versions=V.VERSIONS, registered_verify_plan_sha256=reg['doc_sha256'],
        act_reuse='off (--no-act)' if a.no_act else 'WM-ACT addendum candidates with collected results',
        reference='U0 = frozen speed-only base MAPPO d9042cc5 (not MAPPO60C); the registered U0 wave of each block',
        seeds=seeds, new_candidates=sum(sp['n_new'] for sp in seeds),
        reused_candidates=sum(sp['n_reused'] for sp in seeds),
        reused_from_act=sum(sp['n_reused_act'] for sp in seeds), sim_s_new_plans=new_s, cap_sim_s=CAP_VERIFY_S,
        sim_s_u0_if_rerun_everywhere=u0_s,
        budget_note='exact: new candidates x 24 worlds x 1200 s (cap 36 x 28,800). The block\'s U0 wave '
                    '(E1V<seed>_U0, 28,800 s) is re-run ONLY IF it is not already published in the run root (the '
                    'driver reuses a published wave and charges nothing); a U0 re-run is outside the registered '
                    'verification cap and must be recorded before launch'))
    state = 'dry-run' if a.dry_run else L.write_once(fdir(a.out)/'VERIFY_PLAN_FIX.json', L.text_of(doc))
    return dict(stage='e1_fix plan', label=LABEL, seeds=rep, new_candidates=doc['new_candidates'],
                reused_candidates=doc['reused_candidates'], reused_from_act=doc['reused_from_act'],
                sim_s_new_plans=new_s, cap_sim_s=CAP_VERIFY_S, sim_s_u0_if_rerun_everywhere=u0_s, plan=state,
                doc_sha256=doc['doc_sha256'])


def read_plan(out):
    return L.verify_seal(L.read_json(fdir(out)/'VERIFY_PLAN_FIX.json'), what='VERIFY_PLAN_FIX')


# ------------------------------------------------------------------------------------------------------ admit
def merged_results(sp, reg_results, act_results=None, fix_results=None):
    """VERIFY_RESULTS-shaped dict for the WM-FIX first six: registered rows, WM-ACT addendum rows and FIX addendum
    rows, each for the candidates verified there. Every addendum U0 (same block, same worlds) must equal the
    registered U0 exactly. Pure."""
    seed = sp['search_seed']
    src = dict(registered=reg_results, act=act_results, fix=fix_results)
    plans = {}
    for c in sp['candidates']:
        r = src[c['verified']]
        if r is None:
            raise L.Refused(f'seed {seed}: no {c["verified"]} verification results')
        if c['verified'] != 'registered':
            exact_u0(seed, c['verified'], r['u0'], reg_results['u0'])
        if c['family'] not in r['plans']:
            raise L.Refused(f'seed {seed}: no {c["verified"]} verification result for {c["family"]}')
        plans[c['family']] = r['plans'][c['family']]
    return dict(u0=reg_results['u0'], plans=plans)


def cmd_admit(a):
    plan, rep = read_plan(a.out), []
    for sp in plan['seeds']:
        seed = sp['search_seed']
        if seed not in a.seeds:
            continue
        rp = Path(a.results_dir)/f'VERIFY_RESULTS_seed{seed}.json'
        ap = act_dir(a)/f'VERIFY_RESULTS_ACT_seed{seed}.json'
        fp = Path(a.fix_results_dir or fdir(a.out))/f'VERIFY_RESULTS_FIX_seed{seed}.json'
        act = L.read_json(ap) if sp['n_reused_act'] else None
        if act is not None and L.sha_file(ap) != sp['act_reuse']['act_results_sha256']:
            raise L.Refused(f'seed {seed}: the WM-ACT results changed since the plan')
        fix = L.read_json(fp) if sp['n_new'] else None
        v = B.admit_seed(sp, merged_results(sp, L.read_json(rp), act, fix))
        for c in sp['candidates']:
            v[c['desc_hash']]['verified'] = c['verified']
        doc = L.seal(dict(schema='claim_validation.e1.fix.admission.v1', run_id=L.RUN_ID, arm=ARM, label=LABEL,
                          never_replaces=NEVER, search_seed=seed, rule=next(iter(v.values()))['rule'],
                          results_sha256=L.sha_file(rp), act_results_sha256=L.sha_file(ap) if act else None,
                          fix_results_sha256=L.sha_file(fp) if fix else None,
                          verify_plan_fix_sha256=plan['doc_sha256'], verdicts=v,
                          admitted=[h for h, x in v.items() if x['PASS']]))
        state = 'dry-run' if a.dry_run else L.write_once(fdir(a.out)/f'ADMISSION_FIX_seed{seed}.json', L.text_of(doc))
        rep.append(dict(seed=seed, tested=len(v), new=sp['n_new'], reused_act=sp['n_reused_act'],
                        admitted=len(doc['admitted']), file=state))
    return dict(stage='e1_fix admit', label=LABEL, seeds=rep)


# ------------------------------------------------------------------------------------------------------ build
def check_prior_admission(seed, verdicts, prior, what):
    """A candidate another arm's verification shares must get that arm's verdict (same inputs, same rule)."""
    for h, v in verdicts.items():
        r = (prior or {}).get('verdicts', {}).get(h)
        if r is not None and (bool(r['PASS']) != bool(v['PASS']) or r['lb95_gain_veh_s'] != v['lb95_gain_veh_s']):
            raise L.Refused(f'seed {seed}: {v["family"]} verdict differs from the {what} admission')


def admitted_contrast(rows, reg_frozen):
    """Outcome (a): admitted plans at budget 6 per seed, WM-FIX (rows) vs the registered Random arm. Pure."""
    reg = {(b['search_seed'], b['arm']): b['n_admitted'] for b in reg_frozen['banks'] if b['budget'] == BUDGET}
    seeds = sorted(r['search_seed'] for r in rows)
    fx = {r['search_seed']: r['n_admitted'] for r in rows}
    wf, rnd, wm = ([fx[s] for s in seeds], [reg[(s, 'RANDOM')] for s in seeds], [reg[(s, 'WM')] for s in seeds])
    d = [x-y for x, y in zip(wf, rnd)]
    return dict(label=LABEL, never_replaces=NEVER, budget=BUDGET, seeds=seeds, WMFIX=wf, RANDOM=rnd, WM_registered=wm,
                wmfix_minus_random=d, permutation=S.exact_paired_permutation(d),
                convention='count difference = WM-FIX - Random; positive = more admitted plans under the WM-FIX order',
                wmfix_minus_wm_registered=[x-y for x, y in zip(wf, wm)],
                note='exact paired sign-flip over the six seed pairs, two-sided; smallest attainable p = 2/64 = '
                     '0.03125; post hoc, not part of the registered Holm family')


def fix_confirm_waves(rows, reg_frozen, also_confirmed=()):
    """Registered CONF_U0 (verbatim) + one wave per distinct WM-FIX bank that neither the registered confirmation
    nor ``also_confirmed`` (bank sha256s with collected WM-ACT confirmation results) already ran."""
    (u0,) = [w for w in reg_frozen['confirm_waves'] if w['kind'] == 'U0']
    seen = {w['bank_sha256'] for w in reg_frozen['confirm_waves'] if w['kind'] == 'BANK'} | set(also_confirmed)
    waves = [u0]
    for r in rows:
        if r['is_u0'] or r['sha256'] in seen:
            continue
        seen.add(r['sha256'])
        waves.append(dict(wave=f'E1C_{r["sha256"][:12]}', kind='BANK', block=u0['block'], n_worlds=u0['n_worlds'],
                          bank=r['file'], bank_sha256=r['sha256'], arm=f'{ARM}_seed{r["search_seed"]}'))
    return waves


def act_confirmed(adir, reg_frozen):
    """{bank sha256: {world: J}} of the collected WM-ACT confirmation (its CONF_U0 must equal the registered one is
    checked in analyse), plus the file identity; ({}, None) when there is none."""
    p = Path(adir)/'CONFIRM_RESULTS_ACT.json'
    if not p.is_file():
        return {}, None
    doc = L.read_json(p)
    return dict(doc['banks']), dict(file=p.name, sha256=L.sha_file(p))


def build_rows(out, seeds, reg_frozen, reg_dir, adir, act_banks, dry_run=False):
    reg_sha = {b['sha256']: b for b in reg_frozen['banks'] if b['budget'] == BUDGET and not b['is_u0']}
    rows, files = [], {}
    for seed in seeds:
        pool = L.verify_seal(L.read_json(E.pool_file(out, seed)))
        by_hash = {e['desc_hash']: e for e in pool['pool']}
        adm = L.verify_seal(L.read_json(fdir(out)/f'ADMISSION_FIX_seed{int(seed)}.json'))
        for what, p in (('registered', Path(reg_dir)/f'ADMISSION_seed{int(seed)}.json'),
                        ('WM-ACT', Path(adir)/f'ADMISSION_ACT_seed{int(seed)}.json')):
            check_prior_admission(seed, adm['verdicts'], L.verify_seal(L.read_json(p)) if p.is_file() else None, what)
        order = L.verify_seal(L.read_json(E.order_path(out, ARM, seed)))['top']
        if sorted(adm['verdicts']) != sorted(order[:BUDGET]):
            raise L.Refused(f'seed {seed}: admission does not cover the WM-FIX first six')
        plans = B.bank_for(order, adm['verdicts'], by_hash, BUDGET)
        row = dict(search_seed=int(seed), arm=ARM, budget=BUDGET, tested=order[:BUDGET], n_admitted=len(plans),
                   families=[p['family'] for p in plans], priority=[p['lb95_veh_s'] for p in plans],
                   is_u0=not plans, file=None, sha256=None, primary=False, confirmed_by='U0 (empty bank)')
        if plans:
            text = L.text_of(L.bank_doc(plans))
            sha = L.sha_bytes(text.encode())
            same = reg_sha.get(sha)
            row.update(file=bank_name(seed), sha256=sha,
                       confirmed_by=(f'registered e1conf (identical to {same["file"]})' if same
                                     else 'WM-ACT confirmation (identical bank content)' if sha in act_banks
                                     else 'FIX confirmation wave'))
            files[row['file']] = text
        rows.append(row)
    if not dry_run:
        for name, text in files.items():
            L.write_once(fdir(out)/'banks'/name, text)
    return rows


def cmd_build(a):
    reg_dir = Path(a.registered_bank_dir or Path(a.out)/'e1'/'bank')
    reg = L.verify_seal(L.read_json(reg_dir/'FROZEN_BANKS.json'), what='registered FROZEN_BANKS')
    act_banks, act_ident = ({}, None) if a.no_act else act_confirmed(act_dir(a), reg)
    rows = build_rows(a.out, a.seeds, reg, reg_dir, act_dir(a), act_banks, a.dry_run)
    waves = fix_confirm_waves(rows, reg, act_banks)
    n_new = len(waves)-1
    new_s = n_new*waves[0]['n_worlds']*L.HORIZON_S
    if new_s > CAP_CONFIRM_S:
        raise L.Refused(f'confirmation {new_s} sim-s exceeds the registered cap {CAP_CONFIRM_S}')
    doc = L.seal(dict(
        schema='claim_validation.e1.fix.frozen_banks.v1', run_id=L.RUN_ID, cell=L.CELL, arm=ARM, label=LABEL,
        never_replaces=NEVER, bank_order=B.BANK_ORDER, max_plans=None, empty_bank='bank := U0 (no file, no run)',
        registered_frozen_banks_sha256=reg['doc_sha256'], act_confirmation_reused=act_ident, banks=rows,
        admitted_count=admitted_contrast(rows, reg), confirm_waves=waves, confirm_new_bank_waves=n_new,
        confirm_sim_s_new_banks=new_s, cap_sim_s=CAP_CONFIRM_S, cap_u0_sim_s=CAP_CONFIRM_U0_S,
        confirm_sim_s_u0_if_not_published_in_root=waves[0]['n_worlds']*L.HORIZON_S if n_new else 0))
    state = 'dry-run' if a.dry_run else L.write_once(fdir(a.out)/'FROZEN_BANKS_FIX.json', L.text_of(doc))
    return dict(stage='e1_fix build', label=LABEL, file=state, admitted_count=doc['admitted_count'],
                u0_banks=sum(1 for r in rows if r['is_u0']), confirm_new_bank_waves=n_new,
                confirm_sim_s_new_banks=new_s, doc_sha256=doc['doc_sha256'])


# ---------------------------------------------------------------------------------------------------- analyse
def analyse(fix_frozen, reg_frozen, reg_confirm, extra_confirms=()):
    """Outcome (b): bank total delay at budget 6 on the shared confirmation worlds, WM-FIX bank vs the registered
    Random bank. Pure. reg_confirm / each of extra_confirms (WM-ACT, FIX) = dict(u0={world: J},
    banks={bank sha256: {world: J}}); an empty bank takes U0; every extra CONF_U0 must equal the registered one."""
    worlds = sorted(reg_confirm['u0'])
    banks = dict(reg_confirm['banks'])
    for what, conf in extra_confirms:
        exact_u0('confirmation', what, conf['u0'], reg_confirm['u0'])
        for sha, v in conf['banks'].items():
            if banks.setdefault(sha, v) != v:
                raise L.Refused(f'bank {sha[:12]}: {what} result differs from an earlier result of the same bank')
    u0 = [float(reg_confirm['u0'][w]) for w in worlds]

    def J(row):
        if row['is_u0']:
            return u0
        if row['sha256'] not in banks:
            raise L.Refused(f'bank {row["file"]} was not confirmed')
        src = banks[row['sha256']]
        if sorted(src) != worlds:
            raise L.Refused('confirmation worlds differ between runs')
        return [float(src[w]) for w in worlds]
    reg = {(b['search_seed'], b['arm']): b for b in reg_frozen['banks'] if b['budget'] == BUDGET}
    fx = {r['search_seed']: r for r in fix_frozen['banks']}
    seeds = sorted(fx)
    mean = lambda v: sum(v)/len(v)  # noqa: E731
    mat, per_seed, vs_wm, vs_u0, level = [], [], [], {}, {}
    for s in seeds:
        rows = dict(WMFIX=fx[s], RANDOM=reg[(s, 'RANDOM')], WM_registered=reg[(s, 'WM')])
        j = {k: J(r) for k, r in rows.items()}
        mat.append([x-y for x, y in zip(j['WMFIX'], j['RANDOM'])])
        per_seed.append(mean(mat[-1]))
        vs_wm.append(mean(j['WMFIX'])-mean(j['WM_registered']))
        for k, r in rows.items():
            level[f'{k}_seed{s}'] = mean(j[k])
            vs_u0[f'{k}_seed{s}'] = (dict(diff=0.0, ci95=[0.0, 0.0], pct=0.0, note='bank is U0') if r['is_u0']
                                     else S.paired_bootstrap(j[k], u0))
    return dict(label=LABEL, never_replaces=NEVER, budget=BUDGET, n_worlds=len(worlds), seeds=seeds,
                convention='diff = treatment - reference; negative = less delay (veh-s per world, total delay)',
                admitted_count=fix_frozen['admitted_count'], u0_mean=mean(u0), bank_mean=level,
                wmfix_minus_random_per_seed=per_seed, wmfix_minus_random=S.two_way_bootstrap(mat, crossed=True),
                permutation=S.exact_paired_permutation(per_seed), bank_minus_u0=vs_u0,
                wmfix_minus_wm_registered_per_seed=vs_wm,
                reused=dict(u0='registered CONF_U0', random='registered Random budget-6 banks (e1conf results)'))


def cmd_analyse(a):
    reg_dir = Path(a.registered_bank_dir or Path(a.out)/'e1'/'bank')
    reg = L.verify_seal(L.read_json(reg_dir/'FROZEN_BANKS.json'), what='registered FROZEN_BANKS')
    fz = L.verify_seal(L.read_json(fdir(a.out)/'FROZEN_BANKS_FIX.json'), what='FROZEN_BANKS_FIX')
    if fz['registered_frozen_banks_sha256'] != reg['doc_sha256']:
        raise L.Refused('FROZEN_BANKS_FIX was built against another registered FROZEN_BANKS')
    rp, fp = reg_dir/'CONFIRM_RESULTS.json', Path(a.fix_results or fdir(a.out)/'CONFIRM_RESULTS_FIX.json')
    need = fz['confirm_new_bank_waves'] > 0
    if need and not fp.is_file():
        raise L.Refused(f'{fp}: the FIX confirmation has not been collected')
    extra, ident = [], fz['act_confirmation_reused']
    if ident is not None:
        ap = act_dir(a)/ident['file']
        if L.sha_file(ap) != ident['sha256']:
            raise L.Refused('the WM-ACT confirmation results changed since the banks were built')
        extra.append(('WM-ACT', L.read_json(ap)))
    if need:
        extra.append(('FIX', L.read_json(fp)))
    doc = L.seal(dict(schema='claim_validation.e1.fix.analysis.v1', run_id=L.RUN_ID, arm=ARM,
                      frozen_banks_fix_sha256=fz['doc_sha256'], registered_frozen_banks_sha256=reg['doc_sha256'],
                      registered_results_sha256=L.sha_file(rp), act_results=ident,
                      fix_results_sha256=L.sha_file(fp) if need else None, **analyse(fz, reg, L.read_json(rp), extra)))
    state = 'dry-run' if a.dry_run else L.write_once(fdir(a.out)/'E1_FIX_ANALYSIS.json', L.text_of(doc))
    return dict(stage='e1_fix analyse', label=LABEL, file=state, admitted_count=doc['admitted_count'],
                wmfix_minus_random=doc['wmfix_minus_random'], permutation=doc['permutation'],
                wmfix_minus_random_per_seed=doc['wmfix_minus_random_per_seed'])


def has_waves(a):
    if a.confirm:
        return L.verify_seal(L.read_json(fdir(a.out)/'FROZEN_BANKS_FIX.json'))['confirm_new_bank_waves'] > 0
    (sp,) = [sp for sp in read_plan(a.out)['seeds'] if sp['search_seed'] == a.seed]
    return bool(sp['waves'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('plan', 'admit', 'build', 'analyse', 'has-waves'))
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--code-root', help='plan: cross-check the new candidates against the registered executor')
    p.add_argument('--results-dir', help='admit: directory of the REGISTERED VERIFY_RESULTS_seed<N>.json')
    p.add_argument('--act-dir', help='directory of the WM-ACT addendum files (default out/e1/act)')
    p.add_argument('--no-act', action='store_true', help='plan / build: reuse nothing from the WM-ACT addendum')
    p.add_argument('--fix-results-dir', help='admit: directory of VERIFY_RESULTS_FIX_seed<N>.json (default out/e1/fix)')
    p.add_argument('--registered-bank-dir', help='build / analyse: directory of the registered FROZEN_BANKS.json, '
                                                 'ADMISSION_seed<N>.json, CONFIRM_RESULTS.json (default out/e1/bank)')
    p.add_argument('--fix-results', help='analyse: CONFIRM_RESULTS_FIX.json (default out/e1/fix/)')
    p.add_argument('--seed', type=int, help='has-waves: search seed')
    p.add_argument('--confirm', action='store_true', help='has-waves: the confirmation block')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.command == 'has-waves':
        return 0 if has_waves(a) else 10
    if a.command == 'admit' and not a.results_dir:
        p.error('admit needs --results-dir')
    fn = dict(plan=cmd_plan, admit=cmd_admit, build=cmd_build, analyse=cmd_analyse)[a.command]
    print(L.text_of(dict(fn(a), dry_run=a.dry_run)), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
