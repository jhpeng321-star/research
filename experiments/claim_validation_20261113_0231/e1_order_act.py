"""E1 sensitivity arm WM-ACT: order each seed's fixed pool with never-activated plans ranked last (zero SUMO).
SENSITIVITY, POST-REGISTRATION, PRE-OUTCOME - never replaces the registered primary (ORDER_WM / ORDER_RANDOM are
neither read for ranking nor written here).

Rule (same frozen WM, same pools, same screen scores as the registered WM arm):
  screen  finalists = top 12 by screen summary.score among screen-activated plans (ties in pool order, the registered
          sort); never-screen-activated plans last, by the same sort, only to fill up to 12;
  final   finalists the WM activated on a final state, by final summary.score descending (ties by screen rank);
          then the finalists it never activated (by screen rank);
  rest    only to complete the permutation: non-finalists, screen-activated first, each group by screen score
          descending (ties in pool order).
  top six = the first six.

  ingest  reads WM_SCORES_seed<N>.json (registered scoring) + WM_ACT_SCORES_seed<N>.json (wm_act_score.py) and writes
          out/e1/order/ORDER_WMACT_seed<N>.json (write-once).
  show    WM-ACT next to the registered WM and Random top six.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import e1_order as E  # noqa: E402

ARM = 'WMACT'
LABEL = 'sensitivity, post-registration, pre-outcome'
N_FINALISTS = 12


def act_finalists(rows, n=N_FINALISTS):
    """rows in pool order -> WM-ACT finalist hashes in screen order (identical to wm_act_score.act_finalists). Pure."""
    def top(rs):
        return sorted(rs, key=lambda r: r['screen_summary']['score'], reverse=True)
    ranked = top([r for r in rows if r['screen_activated']])+top([r for r in rows if not r['screen_activated']])
    return [r['desc_hash'] for r in ranked[:int(n)]]


def act_rank(pool_hashes, rows, final_rows, n=N_FINALISTS):
    """The WM-ACT permutation. ``rows`` = registered rows (pool order); ``final_rows`` = one row per WM-ACT finalist
    with activated + summary (final stage). Pure."""
    by = {r['desc_hash']: r for r in rows}
    if [r['desc_hash'] for r in rows] != list(pool_hashes):
        raise L.Refused('WM scores do not cover exactly the pool in pool order')
    fin = act_finalists(rows, n)
    f = {r['desc_hash']: r for r in final_rows}
    if [r['desc_hash'] for r in final_rows] != fin or len(fin) < L.TOP_K:
        raise L.Refused('final rows are not exactly the WM-ACT finalists')
    pos = {h: i for i, h in enumerate(fin)}
    act = sorted((h for h in fin if f[h]['activated']), key=lambda h: (-float(f[h]['summary']['score']), pos[h]))
    idx = {h: i for i, h in enumerate(pool_hashes)}

    def rest(flag):
        return sorted((h for h in pool_hashes if h not in pos and bool(by[h]['screen_activated']) == flag),
                      key=lambda h: (-float(by[h]['screen_summary']['score']), idx[h]))
    return act+[h for h in fin if not f[h]['activated']]+rest(True)+rest(False)


def act_doc(out, seed, scores_dir, act_dir):
    pool_sha, hashes = E.read_pool_hashes(out, seed)
    sp, ap = Path(scores_dir)/f'WM_SCORES_seed{int(seed)}.json', Path(act_dir)/f'WM_ACT_SCORES_seed{int(seed)}.json'
    sc, ac = L.read_json(sp), L.read_json(ap)
    if sc['pool_doc_sha256'] != pool_sha or sc['search_seed'] != int(seed) or sc.get('mode') != 'two_stage':
        raise L.Refused(f'{sp.name}: not the registered two-stage scoring of this pool')
    if (ac['pool_doc_sha256'] != pool_sha or ac['search_seed'] != int(seed) or ac['base_scores_sha256'] != L.sha_file(sp)
            or ac['cfg_sha256'] != sc['cfg_sha256'] or ac['n_final_states'] != sc['n_final_states']
            or ac['K'] != sc['K'] or ac['H'] != sc['H']):
        raise L.Refused(f'{ap.name}: not derived from {sp.name} under the registered final protocol')
    if not sc['wm_checkpoint_sha256'] == ac['wm_checkpoint_sha256'] == L.WM_CKPT_SHA256:
        raise L.Refused('not the registered frozen world model')
    by = {r['desc_hash']: r for r in sc['rows']}
    for r in ac['final_rows']:
        b = by[r['desc_hash']]
        if (r['source'] == 'reused') != bool(b['finalist']) and ac['mode'] == 'reuse':
            raise L.Refused(f'{ap.name}: source flag of {r["desc_hash"][:12]} contradicts the registered scoring')
        if r['source'] == 'reused' and (r['summary'] != b['summary'] or bool(r['activated']) != bool(b['activated'])):
            raise L.Refused(f'{ap.name}: reused row {r["desc_hash"][:12]} differs from the registered final row')
    order = act_rank(hashes, sc['rows'], ac['final_rows'], sc['finalists'])
    f = {r['desc_hash']: r for r in ac['final_rows']}
    return E.order_doc(ARM, seed, pool_sha, order, dict(
        label=LABEL, never_replaces='the registered primary (ORDER_WM vs ORDER_RANDOM)',
        scores_file_sha256=L.sha_file(sp), act_scores_file_sha256=L.sha_file(ap), act_mode=ac['mode'],
        cfg_sha256=sc['cfg_sha256'], wm_checkpoint_sha256=sc['wm_checkpoint_sha256'],
        n_final_states=sc['n_final_states'], K=sc['K'], H=sc['H'], n_screen_states=sc['n_screen_states'],
        K_screen=sc['K_screen'], finalists=sc['finalists'], finalist_hashes=ac['act_finalist_hashes'],
        finalists_new=len(ac['new']), finalists_reused=len(ac['reused']), probe=ac.get('probe'),
        wm_rollouts_added=ac['wm_rollouts'],
        scores={h: f[h]['summary']['score'] for h in order if h in f},
        final_activated={h: bool(f[h]['activated']) for h in order if h in f},
        screen_scores={h: by[h]['screen_summary']['score'] for h in order},
        screen_activated_in_pool=sum(1 for r in sc['rows'] if r['screen_activated']),
        unscored=sum(1 for h in order[:L.TOP_K] if not (h in f and f[h]['activated'])),
        inputs_read=[f'POOL_seed{int(seed)}.json', sp.name, ap.name],
        rule='WM-ACT: finalists = top 12 by screen score among screen-activated plans (never-activated last, only '
             'to fill); order = final-activated finalists by final score descending (ties by screen rank), then '
             'never-activated finalists, then non-finalists (screen-activated first) by screen score. No archive '
             'filter, no SUMO outcome. `unscored` = top-six plans the WM never activated on a final state.'))


def cmd_ingest(a):
    rep = []
    for seed in a.seeds:
        doc = act_doc(a.out, seed, a.scores_dir, a.act_scores_dir)
        state = 'dry-run' if a.dry_run else L.write_once(E.order_path(a.out, ARM, seed), L.text_of(doc))
        rep.append(dict(seed=seed, top=[h[:12] for h in doc['top']], unscored=doc['unscored'],
                        finalists_new=doc['finalists_new'], probe_identical=(doc['probe'] or {}).get('identical'),
                        doc_sha256=doc['doc_sha256'], file=state))
    return dict(stage='e1_order_act ingest', label=LABEL, orders=rep)


def cmd_show(a):
    rep = []
    for seed in a.seeds:
        row = dict(seed=seed)
        for arm in (ARM,)+E.ARMS:
            p = E.order_path(a.out, arm, seed)
            row[arm] = [h[:12] for h in L.verify_seal(L.read_json(p))['top']] if p.is_file() else None
        for arm in E.ARMS:
            if row[ARM] and row[arm]:
                row[f'shared_with_{arm}'] = len(set(row[ARM]) & set(row[arm]))
        rep.append(row)
    return dict(stage='e1_order_act show', label=LABEL, seeds=rep)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('ingest', 'show'))
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--scores-dir', help='ingest: directory of the registered WM_SCORES_seed<N>.json')
    p.add_argument('--act-scores-dir', help='ingest: directory of WM_ACT_SCORES_seed<N>.json')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.command == 'ingest' and not (a.scores_dir and a.act_scores_dir):
        p.error('ingest needs --scores-dir and --act-scores-dir')
    print(L.text_of(dict(dict(ingest=cmd_ingest, show=cmd_show)[a.command](a), dry_run=a.dry_run)), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
