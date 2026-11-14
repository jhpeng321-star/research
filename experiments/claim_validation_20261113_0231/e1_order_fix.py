"""E1-FIX (Amendment 74): order each seed's fixed pool from the repaired WM scoring (zero SUMO). POST HOC REPAIR -
never replaces the registered E1 null (ORDER_WM / ORDER_RANDOM / ORDER_WMACT are neither read for ranking nor
written here).

Rule (wm_fix_score.py scores, same frozen WM, pools, config and start states as the registered WM arm):
  screen  finalists = first 12 by FIX screen score among screen-activated plans (ties in pool order, the registered
          sort); never-screen-activated plans last, by the same sort, only to fill up to 12 (Fix 1);
  final   finalists the WM activated on a final state, by FIX final score descending (ties by screen rank); then
          the finalists it never activated (by screen rank) (Fix 1);
  rest    only to complete the permutation: non-finalists, screen-activated first, each group by screen score
          descending (ties in pool order).
  first six = the plans to verify.

  g0      gate G0: the final stage was scored twice per seed in separate invocations with different chunk sizes
          (wm_fix_score.py main / repeat). PASS only if the first six are the same six in the same order on ALL six
          registered seeds; prints max |score difference| over the finalists for the record. Exit 0 = PASS, 4 = FAIL
          ("WM cannot rank these plans reproducibly": STOP, no SUMO).
  ingest  runs g0 on all six seeds first and refuses unless it passes; writes
          out/e1/order/ORDER_WMFIX_seed<N>.json (write-once) from the main run.
  show    WM-FIX next to the other arms' first six.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import e1_order as E  # noqa: E402
import wm_fix_score as F  # noqa: E402  (pure helpers only; the WM imports there are inside main())

ARM = 'WMFIX'
LABEL = F.LABEL
OTHER_ARMS = ('WMACT',)+E.ARMS
G0_FAIL = 'WM cannot rank these plans reproducibly'


def fix_rank(pool_hashes, screen_rows, finalists, final_rows, field='score'):
    """The WM-FIX permutation. ``screen_rows`` = one row per pool plan in pool order (desc_hash, screen_activated,
    screen_summary); ``finalists`` = finalist hashes in screen order; ``final_rows`` = one row per finalist
    (desc_hash, activated, summary). Pure."""
    by = {r['desc_hash']: r for r in screen_rows}
    if [r['desc_hash'] for r in screen_rows] != list(pool_hashes):
        raise L.Refused('FIX scores do not cover exactly the pool in pool order')
    want = F.fix_finalists([dict(desc_hash=r['desc_hash'], activated=r['screen_activated'],
                                 summary=r['screen_summary']) for r in screen_rows], len(finalists), field)
    f = {r['desc_hash']: r for r in final_rows}
    if list(finalists) != want or sorted(f) != sorted(finalists) or len(finalists) < L.TOP_K:
        raise L.Refused('finalists / final rows are not the Fix-1 finalists of these screen scores')
    pos = {h: i for i, h in enumerate(finalists)}
    act = sorted((h for h in finalists if f[h]['activated']), key=lambda h: (-float(f[h]['summary'][field]), pos[h]))
    idx = {h: i for i, h in enumerate(pool_hashes)}

    def rest(flag):
        return sorted((h for h in pool_hashes if h not in pos and bool(by[h]['screen_activated']) == flag),
                      key=lambda h: (-float(by[h]['screen_summary'][field]), idx[h]))
    return act+[h for h in finalists if not f[h]['activated']]+rest(True)+rest(False)


def g0_seed(pool_hashes, main_doc, repeat_doc):
    """One seed of gate G0 from the main and the repeat score documents. Pure."""
    field = main_doc['score_field']
    fin = list(main_doc['finalist_hashes'])
    rows = main_doc['rows']
    m = [dict(desc_hash=r['desc_hash'], activated=r['activated'], summary=r['summary']) for r in rows if r['finalist']]
    r2 = repeat_doc['final_rows']
    if (repeat_doc['finalist_hashes'] != fin or repeat_doc['score_field'] != field
            or int(repeat_doc['max_rows']) == int(main_doc['max_rows'])):
        raise L.Refused('the repeat did not final-score the same finalists with another chunk size')
    o1 = fix_rank(pool_hashes, rows, fin, m, field)
    o2 = fix_rank(pool_hashes, rows, fin, r2, field)
    a, b = {r['desc_hash']: r for r in m}, {r['desc_hash']: r for r in r2}
    diff = max(abs(float(a[h]['summary'][field])-float(b[h]['summary'][field])) for h in fin)
    return dict(search_seed=int(main_doc['search_seed']), same_first_six=o1[:L.TOP_K] == o2[:L.TOP_K],
                same_full_order=o1 == o2, max_abs_score_diff=diff,
                activation_flags_equal=all(bool(a[h]['activated']) == bool(b[h]['activated']) for h in fin),
                max_rows=[int(main_doc['max_rows']), int(repeat_doc['max_rows'])],
                first_six_main=[h[:12] for h in o1[:L.TOP_K]], first_six_repeat=[h[:12] for h in o2[:L.TOP_K]])


def g0(rows, seeds=L.SEARCH_SEEDS):
    """Gate G0 over the per-seed rows of g0_seed: PASS only if every registered seed is there and agrees. Pure."""
    got = {r['search_seed']: r for r in rows}
    ok = sorted(got) == sorted(int(s) for s in seeds) and all(r['same_first_six'] for r in rows)
    return dict(gate='G0', verdict='PASS' if ok else 'FAIL', stop=None if ok else G0_FAIL,
                max_abs_score_diff=max((r['max_abs_score_diff'] for r in rows), default=None), seeds=rows)


def read_scores(out, seed, fix_dir, repeat_dir, scores_dir):
    """(pool sha, pool hashes, main doc, repeat doc, registered doc, paths) with the identity checks."""
    pool_sha, hashes = E.read_pool_hashes(out, seed)
    name = f'WM_FIX_SCORES_seed{int(seed)}.json'
    mp, rp, sp = Path(fix_dir)/name, Path(repeat_dir)/name, Path(scores_dir)/f'WM_SCORES_seed{int(seed)}.json'
    m, r, sc = L.read_json(mp), L.read_json(rp), L.read_json(sp)
    if m['mode'] != 'main' or r['mode'] != 'repeat' or r['main_scores_sha256'] != L.sha_file(mp):
        raise L.Refused(f'seed {seed}: not a main run and its repeat')
    for d in (m, r):
        if (d['pool_doc_sha256'] != pool_sha or d['search_seed'] != int(seed) or d['cfg_sha256'] != sc['cfg_sha256']
                or d['wm_checkpoint_sha256'] != L.WM_CKPT_SHA256 or d['states_sha256'] != sc['states_sha256']
                or d['n_final_states'] != sc['n_final_states'] or d['H'] != sc['H'] or d['K_screen'] != sc['K_screen']
                or d['finalists'] != sc['finalists'] or d['K_final_registered'] != sc['K']
                or d['K'] != F.K_FINAL_FACTOR*sc['K'] or d['crn_module_sha256'] != m['crn_module_sha256']
                or d['scorer_sha256'] != m['scorer_sha256'] or d['score_field'] != m['score_field']):
            raise L.Refused(f'seed {seed}: {d["mode"]} scores are not the registered protocol with the three fixes')
    if sc['pool_doc_sha256'] != pool_sha or m['screen_worlds'] != sc['screen_worlds']:
        raise L.Refused(f'seed {seed}: screen states differ from the registered scoring')
    return pool_sha, hashes, m, r, sc, (mp, rp, sp)


def cmd_g0(a):
    rows = [g0_seed(h, m, r) for _p, h, m, r, _sc, _paths in
            (read_scores(a.out, s, a.fix_dir, a.repeat_dir, a.scores_dir) for s in a.seeds)]
    return dict(stage='e1_order_fix g0', label=LABEL, **g0(rows))


def fix_doc(out, seed, fix_dir, repeat_dir, scores_dir, gate):
    pool_sha, hashes, m, r, sc, (mp, rp, sp) = read_scores(out, seed, fix_dir, repeat_dir, scores_dir)
    field = m['score_field']
    fin = list(m['finalist_hashes'])
    by = {x['desc_hash']: x for x in m['rows']}
    order = fix_rank(hashes, m['rows'], fin, [dict(desc_hash=h, activated=by[h]['activated'],
                                                   summary=by[h]['summary']) for h in fin], field)
    reg_fin = set(sc['finalist_hashes'])
    return E.order_doc(ARM, seed, pool_sha, order, dict(
        label=LABEL, never_replaces='the registered E1 result (ORDER_WM vs ORDER_RANDOM); outside the A72 Holm family',
        fix_scores_file_sha256=L.sha_file(mp), repeat_scores_file_sha256=L.sha_file(rp),
        registered_scores_file_sha256=L.sha_file(sp), scorer_sha256=m['scorer_sha256'],
        crn_module_sha256=m['crn_module_sha256'], cfg_sha256=m['cfg_sha256'],
        wm_checkpoint_sha256=m['wm_checkpoint_sha256'], states_sha256=m['states_sha256'],
        n_final_states=m['n_final_states'], H=m['H'], n_screen_states=m['n_screen_states'], K_screen=m['K_screen'],
        K_final_registered=m['K_final_registered'], K=m['K'], finalists=m['finalists'], score_field=field,
        max_rows=[m['max_rows'], r['max_rows']], finalist_hashes=fin,
        finalists_shared_with_registered=len(reg_fin & set(fin)), wm_rollouts=m['wm_rollouts']+r['wm_rollouts'],
        crn_log=dict(main=m['crn_log'], repeat=r['crn_log']),
        g0=dict(verdict=gate['verdict'], max_abs_score_diff=gate['max_abs_score_diff'],
                seed=next(x for x in gate['seeds'] if x['search_seed'] == int(seed))),
        scores={h: by[h]['summary'][field] for h in order if by[h]['finalist']},
        final_activated={h: bool(by[h]['activated']) for h in order if by[h]['finalist']},
        screen_scores={h: by[h]['screen_summary'][field] for h in order},
        screen_activated_in_pool=sum(1 for x in m['rows'] if x['screen_activated']),
        unscored=sum(1 for h in order[:L.TOP_K] if not (by[h]['finalist'] and by[h]['activated'])),
        inputs_read=[f'POOL_seed{int(seed)}.json', f'{Path(fix_dir).name}/{mp.name}', f'{Path(repeat_dir).name}/{rp.name}',
                     sp.name+' (protocol identity only)'],
        rule='WM-FIX (Amendment 74): Fix 1 never-activated plans after all activated plans at the screen and the '
             'final stage, same stable sort otherwise; Fix 2 common random numbers (wm_fix_crn); Fix 3 final draws '
             '= 2 x registered. No archive filter, no SUMO outcome. `unscored` = first-six plans the WM never '
             'activated on a final state.'))


def cmd_ingest(a):
    gate = cmd_g0(argparse.Namespace(**dict(vars(a), seeds=list(L.SEARCH_SEEDS))))
    if gate['verdict'] != 'PASS':
        raise L.Refused(f'G0 FAIL: {G0_FAIL}; no order is written, no SUMO')
    rep = []
    for seed in a.seeds:
        doc = fix_doc(a.out, seed, a.fix_dir, a.repeat_dir, a.scores_dir, gate)
        state = 'dry-run' if a.dry_run else L.write_once(E.order_path(a.out, ARM, seed), L.text_of(doc))
        rep.append(dict(seed=seed, top=[h[:12] for h in doc['top']], unscored=doc['unscored'],
                        finalists_shared_with_registered=doc['finalists_shared_with_registered'],
                        doc_sha256=doc['doc_sha256'], file=state))
    return dict(stage='e1_order_fix ingest', label=LABEL, g0=gate['verdict'],
                max_abs_score_diff=gate['max_abs_score_diff'], orders=rep)


def cmd_show(a):
    rep = []
    for seed in a.seeds:
        row = dict(seed=seed)
        for arm in (ARM,)+OTHER_ARMS:
            p = E.order_path(a.out, arm, seed)
            row[arm] = [h[:12] for h in L.verify_seal(L.read_json(p))['top']] if p.is_file() else None
        for arm in OTHER_ARMS:
            if row[ARM] and row[arm]:
                row[f'shared_with_{arm}'] = len(set(row[ARM]) & set(row[arm]))
        rep.append(row)
    return dict(stage='e1_order_fix show', label=LABEL, seeds=rep)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('g0', 'ingest', 'show'))
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--fix-dir', help='directory of the main WM_FIX_SCORES_seed<N>.json')
    p.add_argument('--repeat-dir', help='directory of the repeat WM_FIX_SCORES_seed<N>.json')
    p.add_argument('--scores-dir', default=str(L.HERE/'wm_scores'),
                   help='directory of the registered WM_SCORES_seed<N>.json (protocol identity only)')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.command != 'show' and not (a.fix_dir and a.repeat_dir):
        p.error(f'{a.command} needs --fix-dir and --repeat-dir')
    if a.command == 'g0':
        if sorted(a.seeds) != sorted(L.SEARCH_SEEDS):
            p.error('g0 is defined on all six registered seeds')
        rep = cmd_g0(a)
        print(L.text_of(rep), end='')
        print('G0', rep['verdict'], 'max |score diff|', rep['max_abs_score_diff'])
        return 0 if rep['verdict'] == 'PASS' else 4
    print(L.text_of(dict(dict(ingest=cmd_ingest, show=cmd_show)[a.command](a), dry_run=a.dry_run)), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
