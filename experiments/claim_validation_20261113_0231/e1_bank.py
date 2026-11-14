"""E1 stage 4: admission, bank construction and the confirmation analysis (zero SUMO).

  admit    per seed, every verified candidate: PASS iff the 95% CI upper bound of (plan - U0) total delay < 0 AND
           each of the five group means (L0 / L1 / L2 / HDV / CAV, groups labelled by the U0 run) <= +10 s
           (a37_cellbank.unc_verdict, the rule that admitted the original four-plan bank).  -> ADMISSION_seed<N>.json
  build    per seed x arm x budget prefix (2 / 4 / 6): bank = the admitted plans among the arm's first b candidates.
           -> E1_BANK_seed<N>_<ARM>_b<b>.json, FROZEN_BANKS.json, CONFIRM_WAVES.json. No admitted plan -> the bank
           IS U0 (no file, no run; its confirmation result is the U0 result, difference exactly 0).
           FROZEN_BANKS.primary_admitted_count = THE REGISTERED E1 PRIMARY: admitted plans at budget 6, WM minus
           Random per search seed, exact paired sign-flip permutation over the six seed pairs (two-sided).
           Confirmation waves cover ONLY the twelve budget-6 banks, on the confirmation block shared with E2.
  analyse  confirmation contrasts (secondary): WM bank vs Random bank (seed x world bootstrap, exact paired
           permutation over the six seeds) and every bank vs U0 (paired percentile bootstrap). Budgets whose banks
           were not all confirmed are reported as not run.

BANK ORDER AND ACTIVATION CAP (explicit; nothing new is invented):
  * The executor (oracle_plan.OracleCore / a37_cellbank.CellBankCore) orders unconditional plans by verified
    lb95 gain descending, then strategy key, and tries them in that order every second. That is BANK_ORDER below.
  * The executor has NO bank-level activation cap. Each plan may start whenever its group is feasible, its own
    cooldown has elapsed since its last start, its single slot is free and no member is in another session. So the
    cap is PER PLAN (about (1200 - 300) / cooldown starts per episode) and a bank of N plans can hold roughly N
    times as often as one plan (A68: an 8-plan bank held about 44 times per world against 5 to 10 for a single
    plan, and failed confirmation).
  * --max-plans N (default: no limit = the registered behaviour) keeps only the first N admitted plans in
    BANK_ORDER. It is the only cap this code offers; using it is a registered design decision, not a default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import e1_order as E  # noqa: E402
import stats as S  # noqa: E402

BANK_ORDER = 'verified lb95 gain (= -CI95 upper of plan - U0) descending, then strategy key (executor order)'
PRIMARY_PREFIX = 6


def admit_seed(plan, results):
    """plan = one seed of VERIFY_PLAN; results = VERIFY_RESULTS_seed<N>.json (box_driver collect):
    dict(u0={world: J}, plans={family: dict(J={world: J}, group_increase={group: s}, ...)})."""
    rows = {}
    for c in plan['candidates']:
        r = results['plans'].get(c['family'])
        if r is None:
            raise L.Refused(f'seed {plan["search_seed"]}: no verification result for {c["family"]}')
        v = S.unc_admit(r['J'], results['u0'], r.get('group_increase'), L.N_VERIFY_WORLDS)
        if r.get('groups_all_pass') is not None and bool(r['groups_all_pass']) != v['allowed']:
            raise L.Refused(f'{c["family"]}: group rule differs from the registered co_primary_groups verdict')
        rows[c['desc_hash']] = dict(v, family=c['family'], key=c['key'], size=c['size'], selected_by=c['selected_by'],
                                    activations_per_ep=r.get('activations_per_ep'))
    return rows


def bank_for(order, admission, pool_by_hash, budget, max_plans=None):
    """The bank of one arm at one budget: admitted plans among the first ``budget`` candidates of the arm's order,
    in BANK_ORDER, optionally truncated to ``max_plans``. -> [dict(family, strategy, lb95_veh_s)] (may be empty)."""
    plans = []
    for h in order[:budget]:
        v = admission[h]
        if v['PASS']:
            plans.append(dict(family=v['family'], strategy=pool_by_hash[h]['strategy'],
                              lb95_veh_s=float(v['lb95_gain_veh_s'])))
    plans.sort(key=lambda p: (-p['lb95_veh_s'], L.strategy_key(p['strategy'])))
    return plans if max_plans is None else plans[:int(max_plans)]


def build(out, seeds, max_plans=None, dry_run=False):
    banks, files = [], {}
    for seed in seeds:
        pool = L.verify_seal(L.read_json(E.pool_file(out, seed)))
        by_hash = {e['desc_hash']: e for e in pool['pool']}
        adm = L.verify_seal(L.read_json(Path(out)/'e1'/'bank'/f'ADMISSION_seed{int(seed)}.json'))
        for arm in E.ARMS:
            order = L.verify_seal(L.read_json(E.order_path(out, arm, seed)))['top']
            for b in L.PREFIXES:
                plans = bank_for(order, adm['verdicts'], by_hash, b, max_plans)
                row = dict(search_seed=int(seed), arm=arm, budget=b, tested=order[:b], n_admitted=len(plans),
                           families=[p['family'] for p in plans], priority=[p['lb95_veh_s'] for p in plans],
                           is_u0=not plans, file=None, sha256=None, primary=b == PRIMARY_PREFIX)
                if plans:
                    text = L.text_of(L.bank_doc(plans))
                    row.update(file=f'E1_BANK_seed{int(seed)}_{arm}_b{b}.json', sha256=L.sha_bytes(text.encode()))
                    files[row['file']] = text
                banks.append(row)
    if not dry_run:
        for name, text in files.items():
            L.write_once(Path(out)/'e1'/'banks'/name, text)
    return banks


CONFIRM_BLOCK = 'conf'      # shared with E2 (e2_select.confirm_waves): same worlds, one U0 wave named CONF_U0


def admitted_primary(banks):
    """The registered E1 primary: admitted-plan count at budget 6, WM - Random per search seed (positive = the WM
    order yields more admitted plans), exact paired permutation over the seed pairs, two-sided."""
    rows = {(b['search_seed'], b['arm']): b['n_admitted'] for b in banks if b['budget'] == PRIMARY_PREFIX}
    seeds = sorted({k[0] for k in rows})
    wm, rnd = [rows[(s, 'WM')] for s in seeds], [rows[(s, 'RANDOM')] for s in seeds]
    diffs = [a-b for a, b in zip(wm, rnd)]
    return dict(budget=PRIMARY_PREFIX, seeds=seeds, WM=wm, RANDOM=rnd, wm_minus_random=diffs,
                convention='count difference = WM - Random; positive = more admitted plans under the WM order',
                permutation=S.exact_paired_permutation(diffs),
                holm_note='two primaries (this one and E2 F - H*), Holm, two-sided, alpha 0.05; the smallest '
                          'attainable p of this exact test with 6 pairs is 2/64 = 0.03125')


def confirm_waves(banks, n_worlds, block=CONFIRM_BLOCK):
    """U0 + one CELL_BANK wave per DISTINCT budget-6 bank content (identical banks share one run) on the shared
    confirmation block (crossed design: every bank on the same worlds). Budget-2 / -4 banks are not confirmed."""
    waves = [dict(wave='CONF_U0', kind='U0', block=block, n_worlds=n_worlds)]
    seen = {}
    for b in banks:
        if not b['primary'] or b['is_u0'] or b['sha256'] in seen:
            continue
        seen[b['sha256']] = b['file']
        waves.append(dict(wave=f'E1C_{b["sha256"][:12]}', kind='BANK', block=block, n_worlds=n_worlds, bank=b['file'],
                          bank_sha256=b['sha256']))
    return waves


def analyse(frozen, confirm):
    """confirm = dict(u0={world: J}, banks={bank sha256: {world: J}}). A U0 bank takes the U0 values."""
    worlds = sorted(confirm['u0'])

    def J(row):
        src = confirm['u0'] if row['is_u0'] else confirm['banks'][row['sha256']]
        if sorted(src) != worlds:
            raise L.Refused('confirmation worlds differ between runs')
        return [float(src[w]) for w in worlds]
    u0 = [float(confirm['u0'][w]) for w in worlds]
    out = dict(convention='diff = treatment - reference; negative = less delay', n_worlds=len(worlds), by_budget={})
    for b in L.PREFIXES:
        rows = {(r['search_seed'], r['arm']): r for r in frozen['banks'] if r['budget'] == b}
        seeds = sorted({k[0] for k in rows})
        adm = dict(WM=[rows[(s, 'WM')]['n_admitted'] for s in seeds],
                   RANDOM=[rows[(s, 'RANDOM')]['n_admitted'] for s in seeds])
        if any(not r['is_u0'] and r['sha256'] not in confirm['banks'] for r in rows.values()):
            out['by_budget'][str(b)] = dict(primary=False, seeds=seeds, admitted=adm,
                                            not_run='not every bank of this budget was confirmed (by design)')
            continue
        mat, per_seed, vs_u0 = [], [], {}
        for s in seeds:
            wm, rnd = J(rows[(s, 'WM')]), J(rows[(s, 'RANDOM')])
            mat.append([x-y for x, y in zip(wm, rnd)])
            per_seed.append(sum(mat[-1])/len(worlds))
            for arm, v in (('WM', wm), ('RANDOM', rnd)):
                vs_u0[f'{arm}_seed{s}'] = (dict(diff=0.0, ci95=[0.0, 0.0], pct=0.0, note='bank is U0')
                                           if rows[(s, arm)]['is_u0'] else S.paired_bootstrap(v, u0))
        out['by_budget'][str(b)] = dict(
            primary=b == PRIMARY_PREFIX, role='secondary (delay); the registered E1 primary is FROZEN_BANKS.primary_admitted_count',
            seeds=seeds, wm_minus_random_per_seed=per_seed,
            wm_minus_random=S.two_way_bootstrap(mat, crossed=True),
            permutation=S.exact_paired_permutation(per_seed), bank_minus_u0=vs_u0, admitted=adm)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('admit', 'build', 'analyse'))
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--results-dir', help='admit: VERIFY_RESULTS_seed<N>.json; analyse: CONFIRM_RESULTS.json')
    p.add_argument('--max-plans', type=int, help='build: optional bank size cap (default none = registered executor)')
    p.add_argument('--confirm-worlds', type=int, help='build: size of the confirmation block (registered by MAIN)')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    bdir = Path(a.out)/'e1'/'bank'
    if a.command == 'admit':
        plan = L.verify_seal(L.read_json(Path(a.out)/'e1'/'verify'/'VERIFY_PLAN.json'))
        rep = []
        for sp in plan['seeds']:
            if sp['search_seed'] not in a.seeds:
                continue
            rp = Path(a.results_dir)/f'VERIFY_RESULTS_seed{sp["search_seed"]}.json'
            v = admit_seed(sp, L.read_json(rp))
            doc = L.seal(dict(schema='claim_validation.e1.admission.v1', run_id=L.RUN_ID, search_seed=sp['search_seed'],
                              rule=next(iter(v.values()))['rule'], results_sha256=L.sha_file(rp),
                              verify_plan_sha256=plan['doc_sha256'], verdicts=v,
                              admitted=[h for h, x in v.items() if x['PASS']]))
            state = 'dry-run' if a.dry_run else L.write_once(bdir/f'ADMISSION_seed{sp["search_seed"]}.json',
                                                             L.text_of(doc))
            rep.append(dict(seed=sp['search_seed'], tested=len(v), admitted=len(doc['admitted']), file=state))
        print(L.text_of(dict(stage='e1_bank admit', dry_run=a.dry_run, seeds=rep)), end='')
    elif a.command == 'build':
        banks = build(a.out, a.seeds, a.max_plans, a.dry_run)
        doc = dict(schema='claim_validation.e1.frozen_banks.v1', run_id=L.RUN_ID, cell=L.CELL, bank_order=BANK_ORDER,
                   activation_cap='none at bank level; per plan: own cooldown + one slot + member exclusivity',
                   max_plans=a.max_plans, empty_bank='bank := U0 (no file, no run)', banks=banks,
                   n_primary=sum(1 for b in banks if b['primary']),
                   primary_admitted_count=admitted_primary(banks))
        if a.confirm_worlds:
            doc['confirm_waves'] = confirm_waves(banks, a.confirm_worlds)
        doc = L.seal(doc)
        state = 'dry-run' if a.dry_run else L.write_once(bdir/'FROZEN_BANKS.json', L.text_of(doc))
        print(L.text_of(dict(stage='e1_bank build', dry_run=a.dry_run, file=state, banks=len(banks),
                             u0_banks=sum(1 for b in banks if b['is_u0']),
                             distinct_bank_files=len({b['sha256'] for b in banks if not b['is_u0']}),
                             primary_admitted_count=doc['primary_admitted_count'],
                             doc_sha256=doc['doc_sha256'])), end='')
    else:
        frozen = L.verify_seal(L.read_json(bdir/'FROZEN_BANKS.json'))
        rp = Path(a.results_dir)/'CONFIRM_RESULTS.json'
        doc = L.seal(dict(schema='claim_validation.e1.analysis.v1', run_id=L.RUN_ID,
                          frozen_banks_sha256=frozen['doc_sha256'], results_sha256=L.sha_file(rp),
                          **analyse(frozen, L.read_json(rp))))
        state = 'dry-run' if a.dry_run else L.write_once(bdir/'E1_ANALYSIS.json', L.text_of(doc))
        print(L.text_of(dict(stage='e1_bank analyse', dry_run=a.dry_run, file=state,
                             primary=doc['by_budget'][str(PRIMARY_PREFIX)])), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
