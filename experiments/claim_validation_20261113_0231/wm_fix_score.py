"""E1-FIX (Amendment 74), GPU-box script (zero SUMO). POST HOC REPAIR: never replaces the registered E1 null.

Rescoring of the SAME per-seed pool with the SAME frozen WM, config and start states as wm_pool_score.py (sha
94824b8a), with exactly the three registered fixes:

  Fix 1  plans the WM never activates on the states they are scored on rank after all activated plans, at the screen
         stage (finalists = first cfg `finalists` (12) by screen score among screen-activated plans, stable in pool
         order as the registered sort; never-activated plans only to fill) and at the final stage (e1_order_fix.py).
  Fix 2  common random numbers: wm_fix_crn.evaluate replaces wms.planapi:evaluate through the backend `api`
         override (the WM tree is not edited). Arrivals and merge noise depend on (seed, state, draw index) only.
  Fix 3  screen draws unchanged (cfg K_screen); final draws = 2 x cfg K_final.

Score field: the registered one, summary.score (LCB of gain vs base minus lam * disagreement / sqrt(n)); gain = base
minus plan under the same draw. `--score-field gain_vs_base` (plain mean) exists only so the choice is explicit and
recorded; it must be fixed before launch.

Run from the WM code tree (code_split/code_v12r7b), same environment as wm_pool_score.py:
  wm_fix_score.py preflight <cfg> <pool> <out_dir> <ckpt>
  wm_fix_score.py main      <cfg> <pool> <out_dir> <ckpt> [--workers 10] [--max-rows 64]
  wm_fix_score.py repeat    <cfg> <pool> <out_dir_2> <ckpt> --main-dir <out_dir> [--workers 10] [--max-rows 32]
`repeat` (gate G0) final-scores the SAME finalists again in a separate invocation with another chunk size and writes
to another directory. Outputs are write-once: <out_dir>/WM_FIX_SCORES_seedN.json (+ .<mode>.<stage>.json parts).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

STATES_SHA256 = '74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3'   # as wm_pool_score.py
LABEL = 'POST HOC REPAIR (Amendment 74 E1-FIX); never replaces the registered E1 null'
K_FINAL_FACTOR = 2
MAX_ROWS = dict(main=64, repeat=32)
SCORE_FIELDS = ('score', 'gain_vs_base')
HERE = Path(__file__).resolve().parent


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


def fix_finalists(rows, n, field='score'):
    """Fix 1 at the screen stage. rows in pool order with `activated` and `summary` of the screen stage -> finalist
    hashes: activated plans by score descending (stable in pool order), then never-activated ones the same way.
    Pure."""
    def top(rs):
        return sorted(rs, key=lambda r: r['summary'][field], reverse=True)
    ranked = top([r for r in rows if r['activated']])+top([r for r in rows if not r['activated']])
    return [r['desc_hash'] for r in ranked[:int(n)]]


def crn_log(d):
    """Aggregate of the per-call lines written by wm_fix_crn (one file per worker)."""
    calls, pivot, resid, no_entries = 0, 0, {}, 0
    for p in sorted(Path(d).glob('crn_*.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            calls += 1
            pivot += r['pivot_key_calls']
            no_entries += 0 if r['entries'] else 1
            for k, v in r['residual_global_rng_draws'].items():
                resid[k] = resid.get(k, 0)+v
    return dict(evaluate_calls=calls, pivot_key_calls=pivot, calls_without_entry_process=no_entries,
                residual_global_rng_draws=resid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('mode', choices=('preflight', 'main', 'repeat'))
    ap.add_argument('cfg')
    ap.add_argument('pool')
    ap.add_argument('out_dir')
    ap.add_argument('ckpt')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--max-rows', type=int)
    ap.add_argument('--main-dir')
    ap.add_argument('--score-field', choices=SCORE_FIELDS, default='score')
    a = ap.parse_args(argv)
    mode, field = a.mode, a.score_field
    max_rows = int(a.max_rows or MAX_ROWS['repeat' if mode == 'repeat' else 'main'])
    pool = json.loads(Path(a.pool).read_text(encoding='utf-8'))
    seed = int(pool['search_seed'])
    outd = Path(a.out_dir)
    final = outd/f'WM_FIX_SCORES_seed{seed}.json'
    crn_sha = sha(HERE/'wm_fix_crn.py')
    main_doc = None
    if mode == 'repeat':
        if not a.main_dir or Path(a.main_dir).resolve() == outd.resolve():
            raise SystemExit('REFUSE: repeat needs --main-dir, different from its own out_dir')
        mp_ = Path(a.main_dir)/f'WM_FIX_SCORES_seed{seed}.json'
        main_doc = json.loads(mp_.read_text(encoding='utf-8'))
        if (main_doc['mode'] != 'main' or main_doc['pool_doc_sha256'] != pool['pool_doc_sha256']
                or main_doc['cfg_sha256'] != sha(a.cfg) or main_doc['crn_module_sha256'] != crn_sha
                or main_doc['score_field'] != field):
            raise SystemExit(f'REFUSE: {mp_} is not the main run of these inputs')
        if int(main_doc['max_rows']) == max_rows:
            raise SystemExit('REFUSE: the repeat must use another chunk size (max_rows) than the main run')
    if final.exists() and mode != 'preflight':
        got = json.loads(final.read_text(encoding='utf-8'))
        if got['pool_doc_sha256'] != pool['pool_doc_sha256'] or got['mode'] != mode or got['max_rows'] != max_rows:
            raise SystemExit(f'REFUSE: {final} belongs to another pool, mode or chunk size')
        print('WM_FIX_DONE (kept)', mode, seed, sha(final), flush=True)
        return
    from tools.wzprop import strategy_search as SS
    raw = json.loads(Path(a.cfg).read_text(encoding='utf-8'))
    cfg = dict(SS.DEFAULTS, **raw.get('config', raw))
    if sha(a.ckpt) != pool['wm_checkpoint_sha256']:
        raise SystemExit('REFUSE: checkpoint is not the registered frozen world model')
    # module switches exactly as wm_pool_score.py / search_cell
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
    k_screen, k_reg = int(cfg['K_screen']), int(cfg['K_final'])
    k_final, nfin = K_FINAL_FACTOR*k_reg, int(cfg['finalists'])
    counters_on = [bool((m.get('observed') or {}).get('counter') is not None) for m in states]
    split = int(os.environ.get('WZPROP_SPLIT_ACTIONS', '0') or 0)
    if mode == 'preflight':
        na_s = min(len(cands)+2, split) if split else len(cands)+2

        def est(n_states, k, n_actions, rows):
            return dict(states=n_states, draws=k, max_actions_per_task=n_actions, max_rows_per_task=n_actions*k,
                        max_batches_per_task=-(-n_actions*k//rows), max_rows=rows)
        print('PREFLIGHT_OK', json.dumps(dict(
            search_seed=seed, candidates=len(cands), n_states=len(states), states_with_entrance_counter=sum(counters_on),
            screen_worlds=[int(m['seed']) for m in screen], n_final_states=len(fstates), K_screen=k_screen,
            K_final_registered=k_reg, K_final=k_final, finalists=nfin, H=cfg['H'], lam=cfg['lam'], score_field=field,
            max_rows=MAX_ROWS, split_actions=split, cfg_sha256=sha(a.cfg), ckpt_sha256=sha(a.ckpt),
            crn_module_sha256=crn_sha, scorer_sha256=sha(__file__), states_sha256=STATES_SHA256,
            timing_inputs=dict(
                note='only activated plans roll out; a task = one (state, <=split actions) call on one worker; '
                     'each batch runs H/0.5 s steps', steps_per_batch=int(round(float(cfg['H'])/0.5)),
                registered=dict(screen_wall_s=157, final_wall_s=239, workers=10, K_final=k_reg),
                screen=est(len(screen), k_screen, na_s, MAX_ROWS['main']),
                final=est(len(fstates), k_final, nfin+2, MAX_ROWS['main']),
                repeat=est(len(fstates), k_final, nfin+2, MAX_ROWS['repeat'])))), flush=True)
        return
    outd.mkdir(parents=True, exist_ok=True)
    logd = outd/f'crn_log_seed{seed}_{mode}'
    logd.mkdir(exist_ok=True)
    os.environ['WM_FIX_LOG_DIR'] = str(logd)
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE)]+[p for p in [os.environ.get('PYTHONPATH')] if p])
    sys.path.insert(0, str(HERE))
    from tools.wzprop import wm_backend as W
    kw = dict(checkpoint=a.ckpt, map_spec='maps/wz3to2/MAP_SPEC.json', device='cuda', max_rows=max_rows)
    box = dict(wm=None)

    def backend():
        if box['wm'] is None:
            box['wm'] = W.ParallelWM('wma:WMA_READY.json', a.workers, api=dict(evaluate='wm_fix_crn:evaluate'),
                                     local_kw=dict(kw, device='cpu'), mem_fraction=None, **kw)
        return box['wm']

    def stage(tag, batch, sts, K):
        """One evaluate() call, written once and kept on a re-launch -> (rows, counters)."""
        part = outd/f'WM_FIX_SCORES_seed{seed}.{mode}.{tag}.json'
        ident = dict(pool_doc_sha256=pool['pool_doc_sha256'], keys=[e['key'] for e, _ in batch],
                     worlds=[int(m['seed']) for m in sts], K=int(K), max_rows=max_rows, crn_module_sha256=crn_sha)
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

    by = {e['desc_hash']: (e, s) for e, s in cands}
    doc = dict(schema='claim_validation.e1.wm_fix_scores.v1', label=LABEL, mode=mode, search_seed=seed,
               cell=pool['cell'], pool_doc_sha256=pool['pool_doc_sha256'], cfg_sha256=sha(a.cfg),
               wm_checkpoint_sha256=sha(a.ckpt), states_sha256=STATES_SHA256, n_states=len(states),
               states_with_entrance_counter=sum(counters_on), n_final_states=len(fstates), H=cfg['H'], lam=cfg['lam'],
               K_screen=k_screen, K_final_registered=k_reg, K=k_final, finalists=nfin, score_field=field,
               max_rows=max_rows, split_actions=split, crn_module_sha256=crn_sha, scorer_sha256=sha(__file__),
               fixes=dict(fix1='never-activated plans after all activated plans (screen here, final in e1_order_fix)',
                          fix2='wm_fix_crn.evaluate: arrivals, entrant CAV share and merge-noise keys depend on '
                               '(seed, state digest, draw index) only', fix3=f'K_final = {K_FINAL_FACTOR} x {k_reg}'),
               new_sumo_sim_s=0)
    if mode == 'main':
        srows, sc = stage('screen', cands, screen, k_screen)
        fin = fix_finalists(srows, nfin, field)
        frows, fc = stage('final', [by[h] for h in fin], fstates, k_final)
        f = {r['desc_hash']: r for r in frows}
        rows = [dict(desc_hash=r['desc_hash'], key=r['key'], n_roles=r['n_roles'], screen_summary=r['summary'],
                     screen_activated=r['activated'], finalist=r['desc_hash'] in f,
                     activated=f[r['desc_hash']]['activated'] if r['desc_hash'] in f else None,
                     summary=f[r['desc_hash']]['summary'] if r['desc_hash'] in f else None) for r in srows]
        doc.update(n_screen_states=len(screen), screen_worlds=[int(m['seed']) for m in screen],
                   finalist_hashes=fin, counters=dict(screen=sc, final=fc),
                   wm_rollouts=sc['wm_trajectories']+fc['wm_trajectories'], rows=rows)
    else:
        fin = list(main_doc['finalist_hashes'])
        frows, fc = stage('final', [by[h] for h in fin], fstates, k_final)
        doc.update(main_scores_sha256=sha(Path(a.main_dir)/f'WM_FIX_SCORES_seed{seed}.json'),
                   main_max_rows=main_doc['max_rows'], finalist_hashes=fin, counters=dict(final=fc),
                   wm_rollouts=fc['wm_trajectories'],
                   final_rows=[dict(desc_hash=r['desc_hash'], key=r['key'], activated=r['activated'],
                                    summary=r['summary']) for r in frows])
    doc['crn_log'] = crn_log(logd)
    doc['backend'] = getattr(box['wm'], 'meta', None) if box['wm'] is not None else None
    try:
        import torch
        doc['torch'] = dict(version=torch.__version__, cuda=torch.version.cuda,
                            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as exc:   # record only
        doc['torch'] = dict(error=str(exc))
    write_once(final, json.dumps(doc, indent=1, sort_keys=True, default=str))
    print('WM_FIX_DONE', mode, seed, sha(final), 'residual_global_rng_draws',
          doc['crn_log']['residual_global_rng_draws'], flush=True)


if __name__ == '__main__':   # the WM pool spawns children that re-import this file
    main()
