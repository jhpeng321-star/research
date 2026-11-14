"""E1 stage 2: order each seed's fixed pool in two ways and keep the top six of each (zero SUMO).

  random     Random arm. The permutation comes from e1_random_arm.random_order(pool_hashes, seed): a function that
             receives only the pool hashes and the seed, in a module with no file access at all. This command opens
             exactly one file per seed, POOL_seed<N>.json (which holds no score and no outcome).
  wm-jobs    WM arm, part 1: write the scoring job of every seed for the GPU box (pool descriptors + exact command).
             Nothing is scored here.
  wm-ingest  WM arm, part 2: read the WM_SCORES_seed<N>.json files produced on the GPU box, rank the pool by the
             frozen WM score (descending; ties and unscored plans by descriptor hash) and write the order.
  show       Both arms side by side per seed (top six, budget prefixes 2/4/6, overlap).

Order files are written once (resume = identical file kept, different file refused). Every command has --dry-run.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402
import e1_random_arm  # noqa: E402

ARMS = ('WM', 'RANDOM')
#: GPU-box layout of the registered WM tree (A68: /root/autodl-tmp/a68); a parameter of wm-jobs
BOX_A = '/root/autodl-tmp/a68'


def order_path(out, arm, seed):
    return Path(out)/'e1'/'order'/f'ORDER_{arm}_seed{int(seed)}.json'


def pool_file(out, seed):
    return Path(out)/'e1'/'pool'/f'POOL_seed{int(seed)}.json'


def read_pool_hashes(out, seed):
    """(pool doc sha, [descriptor hash in generation order]) - the ONLY read of the Random arm."""
    doc = L.verify_seal(L.read_json(pool_file(out, seed)), what=f'POOL seed {seed}')
    return doc['doc_sha256'], [e['desc_hash'] for e in doc['pool']]


def order_doc(arm, seed, pool_sha, order, extra):
    return L.seal(dict(schema='claim_validation.e1.order.v1', run_id=L.RUN_ID, arm=arm, search_seed=int(seed),
                       pool_doc_sha256=pool_sha, n=len(order), order=order, top=order[:L.TOP_K],
                       prefixes={str(b): order[:b] for b in L.PREFIXES}, **extra))


def random_doc(out, seed):
    pool_sha, hashes = read_pool_hashes(out, seed)
    order = e1_random_arm.random_order(hashes, seed)
    return order_doc('RANDOM', seed, pool_sha, order, dict(
        stream=e1_random_arm.STREAM, inputs_read=[f'POOL_seed{int(seed)}.json (descriptor hashes only)'],
        rule='sorted hashes shuffled by random.Random(sha256(stream|seed)); independent of the pool stream'))


def wm_rank(pool_hashes, scores):
    """Rank by frozen WM score, descending; ties by hash; a plan the WM never activated (no score) goes last, by
    hash. ``scores`` = {desc_hash: float | None}. Pure."""
    if set(scores) != set(pool_hashes):
        raise L.Refused('WM scores do not cover exactly the pool')
    scored = sorted((h for h in pool_hashes if scores[h] is not None), key=lambda h: (-float(scores[h]), h))
    return scored+sorted(h for h in pool_hashes if scores[h] is None)


def wm_two_stage_rank(pool_hashes, rows, finalist_hashes):
    """The registered protocol. Finalists (screen order = finalist_hashes) ranked by FINAL summary.score descending,
    ties by screen rank; then, only to complete the permutation (never inside the top six when there are >= 6
    finalists), the non-finalists by screen summary.score descending, ties by pool order. Pure."""
    by = {r['desc_hash']: r for r in rows}
    if set(by) != set(pool_hashes) or len(rows) != len(pool_hashes):
        raise L.Refused('WM scores do not cover exactly the pool')
    if {h for h, r in by.items() if r['finalist']} != set(finalist_hashes) or len(finalist_hashes) < L.TOP_K:
        raise L.Refused('finalist list inconsistent with the rows or shorter than the top six')
    pos = {h: i for i, h in enumerate(finalist_hashes)}
    fin = sorted(finalist_hashes, key=lambda h: (-float(by[h]['summary']['score']), pos[h]))
    idx = {h: i for i, h in enumerate(pool_hashes)}
    rest = sorted((h for h in pool_hashes if h not in pos),
                  key=lambda h: (-float(by[h]['screen_summary']['score']), idx[h]))
    return fin+rest


def wm_doc(out, seed, scores_dir):
    pool_sha, hashes = read_pool_hashes(out, seed)
    sp = Path(scores_dir)/f'WM_SCORES_seed{int(seed)}.json'
    sc = L.read_json(sp)
    if sc['pool_doc_sha256'] != pool_sha or sc['search_seed'] != int(seed):
        raise L.Refused(f'{sp.name}: scored another pool')
    if sc['wm_checkpoint_sha256'] != L.WM_CKPT_SHA256:
        raise L.Refused(f'{sp.name}: not the registered frozen world model')
    mode = sc.get('mode', 'full')
    common = dict(scores_file_sha256=L.sha_file(sp), cfg_sha256=sc['cfg_sha256'], mode=mode,
                  wm_checkpoint_sha256=sc['wm_checkpoint_sha256'], wm_rollouts=sc['wm_rollouts'],
                  counters=sc.get('counters'), n_final_states=sc['n_final_states'], K=sc['K'], H=sc['H'],
                  inputs_read=[f'POOL_seed{int(seed)}.json', sp.name])
    if mode == 'two_stage':
        order = wm_two_stage_rank(hashes, sc['rows'], sc['finalist_hashes'])
        by = {r['desc_hash']: r for r in sc['rows']}
        return order_doc('WM', seed, pool_sha, order, dict(
            common, n_screen_states=sc['n_screen_states'], K_screen=sc['K_screen'], finalists=sc['finalists'],
            finalist_hashes=sc['finalist_hashes'],
            scores={h: by[h]['summary']['score'] for h in order if by[h]['finalist']},
            screen_scores={h: by[h]['screen_summary']['score'] for h in order},
            unscored=sum(1 for h in order[:L.TOP_K] if not by[h]['activated']),
            rule='two-stage (the search protocol): screen all on n_screen_states x K_screen, final-score the top '
                 '`finalists` by screen summary.score, rank finalists by final summary.score descending (ties by '
                 'screen rank); top six = first six. No archive filter, no SUMO outcome. `unscored` = top-six '
                 'plans the WM never activated on a final state.'))
    scores = {r['desc_hash']: (r['summary'] or {}).get('score') if r.get('activated') else None for r in sc['rows']}
    order = wm_rank(hashes, scores)
    return order_doc('WM', seed, pool_sha, order, dict(
        common, scores={h: scores[h] for h in order}, unscored=sum(1 for h in order if scores[h] is None),
        rule='FULL mode (not the registered protocol): summary.score descending, ties by descriptor hash; '
             'never-activated plans last. No archive filter, no SUMO outcome.'))


def wm_job(out, seed, box_a, cfg_name, workers, chunk):
    """The scoring job of one seed: the pool descriptors as the scorer reads them + the exact launch command."""
    pool = L.verify_seal(L.read_json(pool_file(out, seed)), what=f'POOL seed {seed}')
    job = dict(schema='claim_validation.e1.wm_pool.v1', run_id=L.RUN_ID, search_seed=int(seed), cell=pool['cell'],
               pool_doc_sha256=pool['doc_sha256'], wm_checkpoint_sha256=L.WM_CKPT_SHA256,
               pool=[dict(desc_hash=e['desc_hash'], key=e['key'], size=e['size'], strategy=e['strategy'])
                     for e in pool['pool']])
    d = f'{box_a}/claim_validation_{L.RUN_ID}'
    cmd = (f'cd {box_a}/code_split/code_v12r7b && WZPROP_SPLIT_ACTIONS=32 '
           f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. nice -n 5 /root/miniconda3/bin/python -B '
           f'{d}/wm_pool_score.py {box_a}/cfg/{cfg_name} {d}/wm_jobs/WM_POOL_seed{int(seed)}.json {d}/wm_scores '
           f'{box_a}/ckpt/ckpt/wm_s.pt {workers} {chunk} two_stage >> {d}/wm_scores/run_seed{int(seed)}.log 2>&1')
    return job, cmd


def cmd_random(a):
    rep = []
    for seed in a.seeds:
        doc = random_doc(a.out, seed)
        state = 'dry-run' if a.dry_run else L.write_once(order_path(a.out, 'RANDOM', seed), L.text_of(doc))
        rep.append(dict(seed=seed, top=[h[:12] for h in doc['top']], doc_sha256=doc['doc_sha256'], file=state))
    return dict(stage='e1_order random', orders=rep)


def cmd_wm_jobs(a):
    rep, cmds = [], []
    for seed in a.seeds:
        job, cmd = wm_job(a.out, seed, a.box_a, a.cfg_name, a.workers, a.chunk)
        p = Path(a.out)/'e1'/'wm_jobs'/f'WM_POOL_seed{int(seed)}.json'
        state = 'dry-run' if a.dry_run else L.write_once(p, L.text_of(job))
        rep.append(dict(seed=seed, n=len(job['pool']), file=state))
        cmds.append(cmd)
    script = ('#!/bin/bash\n# E1 WM arm: score the six fixed pools with the frozen world model (zero SUMO). One launch;\n'
              '# finished chunks / seeds are kept on a re-launch. Run on the GPU box only.\nset -e\n'
              f'mkdir -p {a.box_a}/claim_validation_{L.RUN_ID}/wm_scores\n'+'\n'.join(cmds)+'\n')
    state = 'dry-run' if a.dry_run else L.write_once(Path(a.out)/'e1'/'wm_jobs'/'run_wm_pool.sh', script)
    return dict(stage='e1_order wm-jobs', jobs=rep, script=state, commands=cmds,
                protocol='two_stage: screen 120 x n_screen_states x K_screen, final 12 x final states x K_final; see PROTOCOL.md')


def cmd_wm_ingest(a):
    rep = []
    for seed in a.seeds:
        doc = wm_doc(a.out, seed, a.scores_dir)
        state = 'dry-run' if a.dry_run else L.write_once(order_path(a.out, 'WM', seed), L.text_of(doc))
        rep.append(dict(seed=seed, top=[h[:12] for h in doc['top']], unscored=doc['unscored'], file=state))
    return dict(stage='e1_order wm-ingest', orders=rep)


def cmd_show(a):
    rep = []
    for seed in a.seeds:
        row = dict(seed=seed)
        for arm in ARMS:
            p = order_path(a.out, arm, seed)
            row[arm] = [h[:12] for h in L.verify_seal(L.read_json(p))['top']] if p.is_file() else None
        if row['WM'] and row['RANDOM']:
            row['shared_top6'] = sorted(set(row['WM']) & set(row['RANDOM']))
        rep.append(row)
    return dict(stage='e1_order show', seeds=rep)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('random', 'wm-jobs', 'wm-ingest', 'show'))
    p.add_argument('--out', default=str(L.HERE/'out'))
    p.add_argument('--seeds', type=int, nargs='*', default=list(L.SEARCH_SEEDS))
    p.add_argument('--scores-dir', help='wm-ingest: directory holding WM_SCORES_seed<N>.json from the GPU box')
    p.add_argument('--box-a', default=BOX_A, help='wm-jobs: WM root on the GPU box')
    p.add_argument('--cfg-name', default='A47_SEARCH_CFG.json',
                   help='wm-jobs: scoring config under <box-a>/cfg (the config of the search that built the bank)')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--chunk', type=int, default=120)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.command == 'wm-ingest' and not a.scores_dir:
        p.error('wm-ingest needs --scores-dir')
    out = dict(random=cmd_random, show=cmd_show)
    out.update({'wm-jobs': cmd_wm_jobs, 'wm-ingest': cmd_wm_ingest})
    print(L.text_of(dict(out[a.command](a), dry_run=a.dry_run)), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
