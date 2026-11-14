"""E1 sensitivity arm WM-ACT, stages 3-5: verification addendum, admission, budget-6 banks, confirmation waves and the
WM-ACT vs Random analysis (zero SUMO; this file launches nothing). SENSITIVITY, POST-REGISTRATION, PRE-OUTCOME: it
never replaces the registered primary and never writes a registered file (everything goes to out/e1/act/).

Reads (never writes): out/e1/pool/POOL_seed<N>.json, out/e1/order/ORDER_WMACT_seed<N>.json (e1_order_act.py),
out/e1/verify/VERIFY_PLAN.json, the registered VERIFY_RESULTS_seed<N>.json, FROZEN_BANKS.json, ADMISSION_seed<N>.json
and CONFIRM_RESULTS.json (the Random arm is the registered one, reused as it is).

  plan     per seed the WM-ACT top six; candidates the registered verification already ran are REUSED, the others
           get one PLAN wave each on the same 24 worlds of block e1v_seed<N>, plus that block's registered U0 wave
           (same name and worlds: a root that already published it re-runs nothing).
           -> act/banks/E1_ACT_CANDS_seed<N>.json, act/VERIFY_PLAN_ACT.json
  admit    the registered rule (e1_bank.admit_seed = a37 unc_verdict) on registered results (shared candidates) +
           addendum results (new ones); the addendum U0 must equal the registered U0 world by world.
           -> act/ADMISSION_ACT_seed<N>.json
  build    budget-6 WM-ACT bank per seed (e1_bank.bank_for, executor order), admitted-count contrast WM-ACT - Random
           (exact paired sign-flip), confirmation waves = registered CONF_U0 + one wave per WM-ACT bank whose content
           the registered confirmation did not already run.  -> act/banks/E1_BANK_seed<N>_WMACT_b6.json,
           act/FROZEN_BANKS_ACT.json
  analyse  bank delay on the 96 confirmation worlds: WM-ACT bank vs the registered Random bank (seed x world
           bootstrap, exact sign-flip over the six seeds), each bank vs U0.  -> act/E1_ACT_ANALYSIS.json
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

ARM = 'WMACT'
LABEL = 'sensitivity, post-registration, pre-outcome'
NEVER = 'never replaces the registered primary (WM vs Random admitted count, FROZEN_BANKS.primary_admitted_count)'
BUDGET = B.PRIMARY_PREFIX


def adir(out):
    return Path(out)/'e1'/'act'


def cand_bank_name(seed):
    return f'E1_ACT_CANDS_seed{int(seed)}.json'


def bank_name(seed):
    return f'E1_BANK_seed{int(seed)}_{ARM}_b{BUDGET}.json'


# ------------------------------------------------------------------------------------------------------- plan
def act_seed_plan(top, pool, reg_sp):
    """top = WM-ACT top-six hashes; pool = sealed POOL doc; reg_sp = that seed of the registered VERIFY_PLAN.
    -> (seed plan, candidate bank doc or None). Pure. Wave names follow the registered convention E1V<seed>_<key>."""
    seed, block = int(reg_sp['search_seed']), reg_sp['block']
    if pool['doc_sha256'] != reg_sp['pool_doc_sha256']:
        raise L.Refused(f'seed {seed}: the registered verification planned another pool')
    by_hash = {e['desc_hash']: e for e in pool['pool']}
    done = {c['desc_hash']: c for c in reg_sp['candidates']}
    if len(top) != L.TOP_K or len(set(top)) != L.TOP_K:
        raise L.Refused(f'seed {seed}: WM-ACT top is not {L.TOP_K} distinct plans')
    cands = []
    for rank, h in enumerate(top, 1):
        e = by_hash[h]
        cands.append(dict(desc_hash=h, key=e['key'], family=e['strategy']['name'], size=e['size'],
                          category=e['category'], selected_by={ARM: rank},
                          verified='registered' if h in done else 'addendum',
                          registered_selected_by=done[h]['selected_by'] if h in done else None))
    new = [c for c in cands if c['verified'] == 'addendum']
    if {c['family'] for c in new} & {c['family'] for c in reg_sp['candidates']}:
        raise L.Refused(f'seed {seed}: a new candidate reuses a registered family name')
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
    n_new = len(new)
    return dict(search_seed=seed, block=block, world_seeds=reg_sp.get('world_seeds'),
                pool_doc_sha256=pool['doc_sha256'], top=list(top), candidates=cands, n_new=n_new,
                n_reused=len(cands)-n_new, bank=bank_ref, waves=waves,
                episodes_new_plans=n_new*L.N_VERIFY_WORLDS, sim_s_new_plans=n_new*L.N_VERIFY_WORLDS*L.HORIZON_S,
                sim_s_u0_if_not_published_in_root=L.N_VERIFY_WORLDS*L.HORIZON_S if n_new else 0), bank


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
        if od['arm'] != ARM or od['search_seed'] != seed or od['pool_doc_sha256'] != pool['doc_sha256']:
            raise L.Refused(f'ORDER {ARM} seed {seed}: not the WM-ACT order of this pool')
        sp, bank = act_seed_plan(od['top'], pool, reg_sp)
        sp['order_doc_sha256'] = od['doc_sha256']
        by_hash = {e['desc_hash']: e for e in pool['pool']}
        new = [by_hash[c['desc_hash']]['strategy'] for c in sp['candidates'] if c['verified'] == 'addendum']
        cc = L.crosscheck(a.code_root, new) if a.code_root and new else None
        state = None
        if bank is not None:
            state = 'dry-run' if a.dry_run else L.write_once(adir(a.out)/'banks'/sp['bank']['name'], L.text_of(bank))
        seeds.append(sp)
        rep.append(dict(seed=seed, new=sp['n_new'], reused=sp['n_reused'], sim_s_new_plans=sp['sim_s_new_plans'],
                        bank=state, crosscheck=None if cc is None else cc['status']))
    if len(seeds) != len(a.seeds):
        raise L.Refused('a requested seed is not in the registered VERIFY_PLAN')
    new_s = sum(sp['sim_s_new_plans'] for sp in seeds)
    u0_s = sum(sp['sim_s_u0_if_not_published_in_root'] for sp in seeds)
    doc = L.seal(dict(
        schema='claim_validation.e1.act.verify_plan.v1', run_id=L.RUN_ID, cell=L.CELL, arm=ARM, label=LABEL,
        never_replaces=NEVER, versions=V.VERSIONS, registered_verify_plan_sha256=reg['doc_sha256'],
        reference='U0 = frozen speed-only base MAPPO d9042cc5 (not MAPPO60C); the registered U0 wave of each block',
        seeds=seeds, new_candidates=sum(sp['n_new'] for sp in seeds),
        reused_candidates=sum(sp['n_reused'] for sp in seeds), sim_s_new_plans=new_s, sim_s_u0_if_rerun_everywhere=u0_s,
        budget_note='exact: new candidates x 24 worlds x 1200 s; + 24 x 1200 = 28,800 s per seed with a new '
                    'candidate ONLY IF that block\'s U0 wave (E1V<seed>_U0) is not already published in the run root '
                    '(the driver reuses a published wave and charges nothing for it)'))
    state = 'dry-run' if a.dry_run else L.write_once(adir(a.out)/'VERIFY_PLAN_ACT.json', L.text_of(doc))
    return dict(stage='e1_act plan', label=LABEL, seeds=rep, new_candidates=doc['new_candidates'],
                reused_candidates=doc['reused_candidates'], sim_s_new_plans=new_s,
                sim_s_u0_if_rerun_everywhere=u0_s, plan=state, doc_sha256=doc['doc_sha256'])


def read_plan(out):
    return L.verify_seal(L.read_json(adir(out)/'VERIFY_PLAN_ACT.json'), what='VERIFY_PLAN_ACT')


# ------------------------------------------------------------------------------------------------------ admit
def merged_results(sp, reg_results, act_results):
    """VERIFY_RESULTS-shaped dict for the WM-ACT top six: registered rows for reused candidates, addendum rows for new
    ones. The addendum's U0 (same block, same worlds) must equal the registered U0 exactly. Pure."""
    seed = sp['search_seed']
    new = [c for c in sp['candidates'] if c['verified'] == 'addendum']
    if new:
        if act_results is None:
            raise L.Refused(f'seed {seed}: no addendum verification results')
        u0, ref = act_results['u0'], reg_results['u0']
        if sorted(u0) != sorted(ref):
            raise L.Refused(f'seed {seed}: addendum U0 worlds differ from the registered U0 worlds')
        worst = max(abs(float(u0[w])-float(ref[w])) for w in ref)
        if worst != 0.0:
            raise L.Refused(f'seed {seed}: addendum U0 differs from the registered U0 (max |dJ| {worst} veh-s): '
                            'the pairing is not the registered one')
    plans = {}
    for c in sp['candidates']:
        src = reg_results if c['verified'] == 'registered' else act_results
        if c['family'] not in src['plans']:
            raise L.Refused(f'seed {seed}: no {c["verified"]} verification result for {c["family"]}')
        plans[c['family']] = src['plans'][c['family']]
    return dict(u0=reg_results['u0'], plans=plans)


def cmd_admit(a):
    plan, rep = read_plan(a.out), []
    for sp in plan['seeds']:
        seed = sp['search_seed']
        if seed not in a.seeds:
            continue
        rp = Path(a.results_dir)/f'VERIFY_RESULTS_seed{seed}.json'
        ap = Path(a.act_results_dir or adir(a.out))/f'VERIFY_RESULTS_ACT_seed{seed}.json'
        act = L.read_json(ap) if sp['n_new'] else None
        v = B.admit_seed(sp, merged_results(sp, L.read_json(rp), act))
        for c in sp['candidates']:
            v[c['desc_hash']]['verified'] = c['verified']
        doc = L.seal(dict(schema='claim_validation.e1.act.admission.v1', run_id=L.RUN_ID, arm=ARM, label=LABEL,
                          never_replaces=NEVER, search_seed=seed, rule=next(iter(v.values()))['rule'],
                          results_sha256=L.sha_file(rp), act_results_sha256=L.sha_file(ap) if act else None,
                          verify_plan_act_sha256=plan['doc_sha256'], verdicts=v,
                          admitted=[h for h, x in v.items() if x['PASS']]))
        state = 'dry-run' if a.dry_run else L.write_once(adir(a.out)/f'ADMISSION_ACT_seed{seed}.json', L.text_of(doc))
        rep.append(dict(seed=seed, tested=len(v), new=sp['n_new'], admitted=len(doc['admitted']), file=state))
    return dict(stage='e1_act admit', label=LABEL, seeds=rep)


# ------------------------------------------------------------------------------------------------------ build
def check_registered_admission(seed, verdicts, reg_adm):
    """A candidate both arms' verification shares must get the registered verdict (same inputs, same rule)."""
    for h, v in verdicts.items():
        r = (reg_adm or {}).get('verdicts', {}).get(h)
        if r is not None and (bool(r['PASS']) != bool(v['PASS']) or r['lb95_gain_veh_s'] != v['lb95_gain_veh_s']):
            raise L.Refused(f'seed {seed}: {v["family"]} verdict differs from the registered admission')


def admitted_contrast(rows, reg_frozen):
    """Admitted plans at budget 6 per seed: WM-ACT (rows) vs the registered Random and WM arms. Pure."""
    reg = {(b['search_seed'], b['arm']): b['n_admitted'] for b in reg_frozen['banks'] if b['budget'] == BUDGET}
    seeds = sorted(r['search_seed'] for r in rows)
    act = {r['search_seed']: r['n_admitted'] for r in rows}
    wa, rnd, wm = ([act[s] for s in seeds], [reg[(s, 'RANDOM')] for s in seeds], [reg[(s, 'WM')] for s in seeds])
    d = [x-y for x, y in zip(wa, rnd)]
    return dict(label=LABEL, never_replaces=NEVER, budget=BUDGET, seeds=seeds, WMACT=wa, RANDOM=rnd, WM_registered=wm,
                wmact_minus_random=d, permutation=S.exact_paired_permutation(d),
                convention='count difference = WM-ACT - Random; positive = more admitted plans under the WM-ACT order',
                wmact_minus_wm_registered=[x-y for x, y in zip(wa, wm)],
                note='exact paired sign-flip over the six seed pairs, two-sided; smallest attainable p = 2/64 = '
                     '0.03125; not part of the registered Holm family')


def act_confirm_waves(rows, reg_frozen):
    """Registered CONF_U0 (verbatim) + one wave per distinct WM-ACT bank the registered confirmation did not run."""
    (u0,) = [w for w in reg_frozen['confirm_waves'] if w['kind'] == 'U0']
    seen = {w['bank_sha256'] for w in reg_frozen['confirm_waves'] if w['kind'] == 'BANK'}
    waves = [u0]
    for r in rows:
        if r['is_u0'] or r['sha256'] in seen:
            continue
        seen.add(r['sha256'])
        waves.append(dict(wave=f'E1C_{r["sha256"][:12]}', kind='BANK', block=u0['block'], n_worlds=u0['n_worlds'],
                          bank=r['file'], bank_sha256=r['sha256'], arm=f'{ARM}_seed{r["search_seed"]}'))
    return waves


def build_rows(out, seeds, reg_frozen, reg_dir, dry_run=False):
    reg_sha = {b['sha256']: b for b in reg_frozen['banks'] if b['budget'] == BUDGET and not b['is_u0']}
    rows, files = [], {}
    for seed in seeds:
        pool = L.verify_seal(L.read_json(E.pool_file(out, seed)))
        by_hash = {e['desc_hash']: e for e in pool['pool']}
        adm = L.verify_seal(L.read_json(adir(out)/f'ADMISSION_ACT_seed{int(seed)}.json'))
        rp = Path(reg_dir)/f'ADMISSION_seed{int(seed)}.json'
        check_registered_admission(seed, adm['verdicts'], L.verify_seal(L.read_json(rp)) if rp.is_file() else None)
        top = list(adm['verdicts'])
        order = L.verify_seal(L.read_json(E.order_path(out, ARM, seed)))['top']
        if sorted(top) != sorted(order[:BUDGET]):
            raise L.Refused(f'seed {seed}: admission does not cover the WM-ACT top six')
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
                                     else 'addendum confirmation wave'))
            files[row['file']] = text
        rows.append(row)
    if not dry_run:
        for name, text in files.items():
            L.write_once(adir(out)/'banks'/name, text)
    return rows


def cmd_build(a):
    reg_dir = Path(a.registered_bank_dir or Path(a.out)/'e1'/'bank')
    reg = L.verify_seal(L.read_json(reg_dir/'FROZEN_BANKS.json'), what='registered FROZEN_BANKS')
    rows = build_rows(a.out, a.seeds, reg, reg_dir, a.dry_run)
    waves = act_confirm_waves(rows, reg)
    n_new = len(waves)-1
    doc = L.seal(dict(
        schema='claim_validation.e1.act.frozen_banks.v1', run_id=L.RUN_ID, cell=L.CELL, arm=ARM, label=LABEL,
        never_replaces=NEVER, bank_order=B.BANK_ORDER, max_plans=None, empty_bank='bank := U0 (no file, no run)',
        registered_frozen_banks_sha256=reg['doc_sha256'], banks=rows, admitted_count=admitted_contrast(rows, reg),
        confirm_waves=waves, confirm_new_bank_waves=n_new,
        confirm_sim_s_new_banks=n_new*waves[0]['n_worlds']*L.HORIZON_S,
        confirm_sim_s_u0_if_not_published_in_root=waves[0]['n_worlds']*L.HORIZON_S if n_new else 0))
    state = 'dry-run' if a.dry_run else L.write_once(adir(a.out)/'FROZEN_BANKS_ACT.json', L.text_of(doc))
    return dict(stage='e1_act build', label=LABEL, file=state, admitted_count=doc['admitted_count'],
                u0_banks=sum(1 for r in rows if r['is_u0']), confirm_new_bank_waves=n_new,
                confirm_sim_s_new_banks=doc['confirm_sim_s_new_banks'], doc_sha256=doc['doc_sha256'])


# ---------------------------------------------------------------------------------------------------- analyse
def analyse(act_frozen, reg_frozen, reg_confirm, act_confirm=None):
    """Bank delay at budget 6 on the shared confirmation worlds: WM-ACT bank vs the registered Random bank. Pure.
    reg_confirm / act_confirm = dict(u0={world: J}, banks={bank sha256: {world: J}}); an empty bank takes U0."""
    worlds = sorted(reg_confirm['u0'])
    banks = dict(reg_confirm['banks'])
    if act_confirm is not None:
        if sorted(act_confirm['u0']) != worlds:
            raise L.Refused('addendum confirmation worlds differ from the registered confirmation worlds')
        worst = max(abs(float(act_confirm['u0'][w])-float(reg_confirm['u0'][w])) for w in worlds)
        if worst != 0.0:
            raise L.Refused(f'addendum CONF_U0 differs from the registered CONF_U0 (max |dJ| {worst} veh-s)')
        for sha, v in act_confirm['banks'].items():
            if banks.setdefault(sha, v) != v:
                raise L.Refused(f'bank {sha[:12]}: addendum result differs from the registered result')
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
    act = {r['search_seed']: r for r in act_frozen['banks']}
    seeds = sorted(act)
    mean = lambda v: sum(v)/len(v)  # noqa: E731
    mat, per_seed, vs_wm, vs_u0, level = [], [], [], {}, {}
    for s in seeds:
        rows = dict(WMACT=act[s], RANDOM=reg[(s, 'RANDOM')], WM_registered=reg[(s, 'WM')])
        j = {k: J(r) for k, r in rows.items()}
        mat.append([x-y for x, y in zip(j['WMACT'], j['RANDOM'])])
        per_seed.append(mean(mat[-1]))
        vs_wm.append(mean(j['WMACT'])-mean(j['WM_registered']))
        for k, r in rows.items():
            level[f'{k}_seed{s}'] = mean(j[k])
            vs_u0[f'{k}_seed{s}'] = (dict(diff=0.0, ci95=[0.0, 0.0], pct=0.0, note='bank is U0') if r['is_u0']
                                     else S.paired_bootstrap(j[k], u0))
    return dict(label=LABEL, never_replaces=NEVER, budget=BUDGET, n_worlds=len(worlds), seeds=seeds,
                convention='diff = treatment - reference; negative = less delay (veh-s per world, total delay)',
                admitted_count=act_frozen['admitted_count'], u0_mean=mean(u0), bank_mean=level,
                wmact_minus_random_per_seed=per_seed, wmact_minus_random=S.two_way_bootstrap(mat, crossed=True),
                permutation=S.exact_paired_permutation(per_seed), bank_minus_u0=vs_u0,
                wmact_minus_wm_registered_per_seed=vs_wm,
                reused=dict(u0='registered CONF_U0', random='registered Random budget-6 banks (e1conf results)'))


def cmd_analyse(a):
    reg_dir = Path(a.registered_bank_dir or Path(a.out)/'e1'/'bank')
    reg = L.verify_seal(L.read_json(reg_dir/'FROZEN_BANKS.json'), what='registered FROZEN_BANKS')
    act = L.verify_seal(L.read_json(adir(a.out)/'FROZEN_BANKS_ACT.json'), what='FROZEN_BANKS_ACT')
    if act['registered_frozen_banks_sha256'] != reg['doc_sha256']:
        raise L.Refused('FROZEN_BANKS_ACT was built against another registered FROZEN_BANKS')
    rp, ap = reg_dir/'CONFIRM_RESULTS.json', Path(a.act_results or adir(a.out)/'CONFIRM_RESULTS_ACT.json')
    need = act['confirm_new_bank_waves'] > 0
    if need and not ap.is_file():
        raise L.Refused(f'{ap}: the addendum confirmation has not been collected')
    doc = L.seal(dict(schema='claim_validation.e1.act.analysis.v1', run_id=L.RUN_ID, arm=ARM,
                      frozen_banks_act_sha256=act['doc_sha256'], registered_frozen_banks_sha256=reg['doc_sha256'],
                      registered_results_sha256=L.sha_file(rp), act_results_sha256=L.sha_file(ap) if need else None,
                      **analyse(act, reg, L.read_json(rp), L.read_json(ap) if need else None)))
    state = 'dry-run' if a.dry_run else L.write_once(adir(a.out)/'E1_ACT_ANALYSIS.json', L.text_of(doc))
    return dict(stage='e1_act analyse', label=LABEL, file=state, admitted_count=doc['admitted_count'],
                wmact_minus_random=doc['wmact_minus_random'], permutation=doc['permutation'],
                wmact_minus_random_per_seed=doc['wmact_minus_random_per_seed'])


def has_waves(a):
    if a.confirm:
        return L.verify_seal(L.read_json(adir(a.out)/'FROZEN_BANKS_ACT.json'))['confirm_new_bank_waves'] > 0
    (sp,) = [sp for sp in read_plan(a.out)['seeds'] if sp['search_seed'] == a.seed]
    return bool(sp['waves'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('plan', 'admit', 'build', 'analyse', 'has-waves'))
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--code-root', help='plan: cross-check the new candidates against the registered executor')
    p.add_argument('--results-dir', help='admit: directory of the REGISTERED VERIFY_RESULTS_seed<N>.json')
    p.add_argument('--act-results-dir', help='admit: directory of VERIFY_RESULTS_ACT_seed<N>.json (default out/e1/act)')
    p.add_argument('--registered-bank-dir', help='build / analyse: directory of the registered FROZEN_BANKS.json, '
                                                 'ADMISSION_seed<N>.json, CONFIRM_RESULTS.json (default out/e1/bank)')
    p.add_argument('--act-results', help='analyse: CONFIRM_RESULTS_ACT.json (default out/e1/act/)')
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
