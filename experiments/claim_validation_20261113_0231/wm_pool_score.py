"""E1 WM arm, GPU-box script (zero SUMO): score every descriptor of one seed's fixed pool with the FROZEN world
model, by the same evaluate()/summarize() calls and config as the registered search finals. Pattern = the A68
descriptive scorer (runs/active/C4-V7FT-20261017/a68_research60c/wm_desc/wm_desc_score.py, sha 2d99c446).

Run from the WM code tree (code_split/code_v12r7b) with
    PYTHONPATH=src:. WZPROP_SPLIT_ACTIONS=32 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
usage: wm_pool_score.py <cfg.json> <WM_POOL_seedN.json> <out_dir> <ckpt wm_s.pt> [workers=10] [chunk=120] [mode]

mode = two_stage (DEFAULT, the registered protocol = the search's own two stages, strategy_search.search_cell):
    screen  all pool candidates on cfg n_screen_states start states with K_screen draws
            (states = random.Random(search_seed).sample(states, n_screen_states): the search's own line with the
            search seed in place of cfg['seed']);
    final   the first cfg finalists candidates by screen summary.score (stable sort in pool order, as the search)
            on the final states (cfg n_final_states rule) with K_final draws;
    rank    finalists by final summary.score. Non-finalists are never final-scored.
    Difference from search_cell, stated: the search also added archive leaders (up to 4 x finalists) and the two
    comparators to its finals and bred children between screens; here the pool is fixed, so the finals are exactly
    the top `finalists` by screen score.
mode = preflight (ZERO WM rollouts, seconds, no GPU): everything up to the first evaluate() on the REAL inputs - imports,
    config switches, checkpoint + start-states sha pins, every pool descriptor through SS.validate / strategy_key /
    uncond, the screen / final state lists - then prints PREFLIGHT_OK with the planned counts and writes nothing.
mode = full (option, NOT the registered protocol): every candidate under the final protocol, chunked (the chunk
    paragraph below applies to this mode only).

Counters are reported separately per stage: candidate_state_draws (candidates x states x K, nominal),
wm_trajectories (what strategy_search.evaluate returns as "rollouts": distinct actions incl. base / incumbent x K,
summed over states - the unit of the search's "944"), batch_calls (one WM call per start state), wall_s.

Checkpoint/resume: the pool is scored in chunks of ``chunk`` descriptors; each finished chunk is written once
(out_dir/WM_SCORES_seedN.partK.json) and skipped on a re-launch; the final WM_SCORES_seedN.json is assembled from
the parts and refused if it already exists with other content. The default chunk (120 = the whole pool) makes one
evaluate() call per seed, exactly like the search finals. A smaller chunk only changes how candidates are batched;
the WM draws use each state's own common-random-number seed (strategy_search.evaluate), so scores should not move,
but that equality is a mechanics check to run once (score one chunk both ways) before relying on it.

This script NEVER reads a SUMO outcome. It does not rank or select: ordering happens in e1_order.py wm-ingest.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_once(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise SystemExit(f'REFUSE: {path} exists with different content')
        return
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


STATES_SHA256 = '74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3'   # data/regime_STATES.json (A42/A47/A68)


def main():
    from tools.wzprop import strategy_search as SS
    cfgf, poolf, outd, ckpt = sys.argv[1:5]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    chunk = int(sys.argv[6]) if len(sys.argv) > 6 else 120
    mode = sys.argv[7] if len(sys.argv) > 7 else 'two_stage'
    if mode not in ('two_stage', 'full', 'preflight'):
        raise SystemExit('REFUSE: mode must be two_stage, full or preflight')
    pool = json.loads(Path(poolf).read_text(encoding='utf-8'))
    seed = int(pool['search_seed'])
    outd = Path(outd)
    outd.mkdir(parents=True, exist_ok=True)
    final = outd/f'WM_SCORES_seed{seed}.json'
    if final.exists() and mode != 'preflight':
        got = json.loads(final.read_text(encoding='utf-8'))
        if got['pool_doc_sha256'] != pool['pool_doc_sha256'] or got.get('mode') != mode:
            raise SystemExit(f'REFUSE: {final} belongs to another pool or mode')
        print('WM_POOL_DONE (kept)', seed, sha(final), flush=True)
        return
    raw = json.loads(Path(cfgf).read_text(encoding='utf-8'))
    cfg = dict(SS.DEFAULTS, **raw.get('config', raw))
    if sha(ckpt) != pool['wm_checkpoint_sha256']:
        raise SystemExit('REFUSE: checkpoint is not the registered frozen world model')
    # the module switches exactly as search_cell sets them (as wm_desc_score.py)
    SS.UNCOND, SS.F_VS_BASE = bool(cfg.get('unconditional')), bool(cfg.get('f_vs_base'))
    SS.NOV_W = float(cfg.get('novelty_weight', 0.0))
    SS.MIN_ROLES = int(cfg.get('min_roles', 0))
    SS.LANE2_EXTRA_S = float(cfg.get('lane2_extra_margin_s', 0.0))
    SS.REC_W = float(cfg.get('rec_weight', 0.0))
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    sk = dict(fair_tol=float(cfg['fair_tol']), bound_s=float(cfg['bound_s']), bound_stat=cfg['bound_stat'],
              fairer_total=cfg['fairer_total'], harm_cal=cfg['harm_cal'])
    cands = []
    for e in pool['pool']:
        s = SS.validate(dict(e['strategy']))
        if SS.strategy_key(s) != e['key'] or SS.strategy_key(SS.uncond(s)) != e['key']:
            raise SystemExit(f"REFUSE: key mismatch for {e['desc_hash'][:12]}")
        cands.append((e, s))
    kw = dict(checkpoint=ckpt, map_spec='maps/wz3to2/MAP_SPEC.json', device='cuda', max_rows=64)
    if sha('data/regime_STATES.json') != STATES_SHA256:
        raise SystemExit('REFUSE: data/regime_STATES.json is not the registered start-states file 74aef8f8')
    states = [m for m in json.loads(Path('data/regime_STATES.json').read_text()) if m.get('valid')]
    states = [m for m in states if m['cell'] == pool['cell']]
    fstates = states
    if int(cfg['n_final_states']) and len(states) > int(cfg['n_final_states']):
        fstates = random.Random(cfg['seed']+1).sample(states, int(cfg['n_final_states']))
    screen = random.Random(seed).sample(states, min(len(states), int(cfg['n_screen_states'])))
    if len(cands) < int(cfg['finalists']) or len(screen) != int(cfg['n_screen_states']) or not fstates:
        raise SystemExit('REFUSE: pool smaller than the finalists or too few start states')
    if mode == 'preflight':
        print('PREFLIGHT_OK', json.dumps(dict(
            search_seed=seed, candidates=len(cands), sizes=sorted({len(s['roles']) for _, s in cands}),
            n_states=len(states), screen_worlds=[int(m['seed']) for m in screen], n_final_states=len(fstates),
            K_screen=cfg['K_screen'], K_final=cfg['K_final'], finalists=int(cfg['finalists']), H=cfg['H'],
            lam=cfg['lam'], cfg_sha256=sha(cfgf), ckpt_sha256=sha(ckpt), states_sha256=STATES_SHA256,
            switches=dict(UNCOND=SS.UNCOND, F_VS_BASE=SS.F_VS_BASE, MIN_ROLES=SS.MIN_ROLES, ONSET_MAX=SS.ONSET_MAX),
            max_trajectories=dict(screen=len(screen)*(len(cands)+2)*int(cfg['K_screen']),
                                  final=len(fstates)*(int(cfg['finalists'])+2)*int(cfg['K_final'])))), flush=True)
        return
    from tools.wzprop import wm_backend as W
    box = dict(wm=None)

    def backend():
        if box['wm'] is None:
            box['wm'] = W.ParallelWM('wma:WMA_READY.json', workers, api={}, local_kw=dict(kw, device='cpu'),
                                     mem_fraction=None, **kw)
        return box['wm']

    def stage(tag, batch, sts, K):
        """One evaluate() call, written once and kept on a re-launch -> (rows, counters)."""
        part = outd/f'WM_SCORES_seed{seed}.{mode}.{tag}.json'
        ident = dict(pool_doc_sha256=pool['pool_doc_sha256'], keys=[e['key'] for e, _ in batch],
                     worlds=[int(m['seed']) for m in sts], K=int(K))
        if part.exists():
            got = json.loads(part.read_text(encoding='utf-8'))
            if got['ident'] != ident:
                raise SystemExit(f'REFUSE: {part} was written for other inputs')
            return got['rows'], got['counters']
        t0 = time.time()
        per, n = SS.evaluate(backend(), sts, [s for _, s in batch], K, cfg['H'], groups=bool(cfg['fairness']))
        rows = []
        for e, s in batch:
            summ = SS.summarize(per[e['key']], float(cfg['lam']), **sk)
            rows.append(dict(desc_hash=e['desc_hash'], key=e['key'], n_roles=len(s['roles']),
                             activated=any(r['feasible'] for r in per[e['key']]), summary=summ))
        counters = dict(candidates=len(batch), states=len(sts), K=int(K),
                        candidate_state_draws=len(batch)*len(sts)*int(K), wm_trajectories=int(n),
                        batch_calls=len(sts), wall_s=round(time.time()-t0, 1))
        write_once(part, json.dumps(dict(ident=ident, rows=rows, counters=counters), indent=1, default=str))
        print(tag, counters, flush=True)
        return rows, counters

    doc = dict(schema='claim_validation.e1.wm_scores.v2', mode=mode, search_seed=seed, cell=pool['cell'],
               pool_doc_sha256=pool['pool_doc_sha256'], cfg_sha256=sha(cfgf), wm_checkpoint_sha256=sha(ckpt),
               n_states=len(states), states_sha256=STATES_SHA256, n_final_states=len(fstates), K=cfg['K_final'], H=cfg['H'], lam=cfg['lam'],
               score_field='summary.score (LCB of gain vs base minus lam * disagreement / sqrt(n))',
               new_sumo_sim_s=0)
    if mode == 'two_stage':
        srows, sc = stage('screen', cands, screen, cfg['K_screen'])
        by = {e['key']: (e, s) for e, s in cands}
        ranked = sorted(srows, key=lambda r: r['summary']['score'], reverse=True)[:int(cfg['finalists'])]
        frows, fc = stage('final', [by[r['key']] for r in ranked], fstates, cfg['K_final'])
        fin = {r['desc_hash']: r for r in frows}
        rows = [dict(desc_hash=r['desc_hash'], key=r['key'], n_roles=r['n_roles'], screen_summary=r['summary'],
                     screen_activated=r['activated'], finalist=r['desc_hash'] in fin,
                     activated=fin[r['desc_hash']]['activated'] if r['desc_hash'] in fin else None,
                     summary=fin[r['desc_hash']]['summary'] if r['desc_hash'] in fin else None) for r in srows]
        doc.update(n_screen_states=len(screen), screen_worlds=[int(m['seed']) for m in screen],
                   K_screen=cfg['K_screen'], finalists=int(cfg['finalists']),
                   finalist_hashes=[r['desc_hash'] for r in ranked], counters=dict(screen=sc, final=fc),
                   wm_rollouts=sc['wm_trajectories']+fc['wm_trajectories'], rows=rows)
    else:
        rows, cs = [], []
        for ci in range(0, len(cands), chunk):
            got, c = stage(f'part{ci//chunk:03d}of{chunk}', cands[ci:ci+chunk], fstates, cfg['K_final'])
            rows += got
            cs.append(c)
        doc.update(chunk=chunk, counters=dict(final_parts=cs), wm_rollouts=sum(c['wm_trajectories'] for c in cs),
                   rows=rows)
    doc['backend'] = getattr(box['wm'], 'meta', None) if box['wm'] is not None else None
    write_once(final, json.dumps(doc, indent=1, sort_keys=True, default=str))
    print('WM_POOL_DONE', seed, sha(final), flush=True)


if __name__ == '__main__':   # the WM pool spawns children that re-import this file
    main()
