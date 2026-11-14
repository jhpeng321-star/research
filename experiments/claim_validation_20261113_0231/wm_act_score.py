"""E1 sensitivity arm WM-ACT, GPU-box script (zero SUMO). SENSITIVITY, POST-REGISTRATION, PRE-OUTCOME: it never
replaces the registered WM order. Same frozen world model, same pool, same config, same start states, same screen
scores as wm_pool_score.py (sha 94824b8a); only the finalist list changes:

    WM-ACT finalists = the first cfg `finalists` (12) pool candidates by screen summary.score among the plans the WM
    ACTIVATED on a screen state (screen_activated), stable in pool order exactly as the registered sort; plans never
    activated at the screen stage are ranked last and enter only if fewer than 12 were activated (same sort).

This script final-scores the WM-ACT finalists that the registered run did not final-score (ONE evaluate() call on
the registered final states with K_final draws) and reuses the registered final rows of the others unchanged.
A probe is added to that call when possible: one candidate the registered run already final-scored AND activated
(the first such WM-ACT finalist, else the first such registered finalist). The probe is never used for ranking; it
only reports whether a final score depends on how candidates are batched (probe.identical).

Run from the WM code tree (code_split/code_v12r7b), same environment as wm_pool_score.py:
usage: wm_act_score.py <cfg.json> <WM_POOL_seedN.json> <WM_SCORES_seedN.json> <out_dir> <ckpt wm_s.pt> [workers=10] [mode]
mode = reuse (default) | all (final-score all 12 WM-ACT finalists in one call, no reuse, no probe; only if the probe
       of a `reuse` run is not identical; needs another out_dir) | preflight (zero WM rollouts; prints the plan).
Writes out_dir/WM_ACT_SCORES_seedN.json (+ .final.json part), write-once. <out_dir> must not be the registered
wm_scores directory. Nothing is ranked here beyond the finalist list: ordering happens in e1_order_act.py ingest.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

STATES_SHA256 = '74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3'   # as wm_pool_score.py
RULE = ('WM-ACT finalists: top `finalists` by screen summary.score among screen-activated plans (stable in pool '
        'order); never-screen-activated plans last, by the same sort, only to fill')


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


def act_finalists(rows, n):
    """rows in pool order (as WM_SCORES rows) -> the WM-ACT finalist hashes in screen order. Pure."""
    def top(rs):
        return sorted(rs, key=lambda r: r['screen_summary']['score'], reverse=True)
    ranked = top([r for r in rows if r['screen_activated']])+top([r for r in rows if not r['screen_activated']])
    return [r['desc_hash'] for r in ranked[:int(n)]]


def plan(base, n, mode='reuse'):
    """(finalists, to_score, reused, probe hash | None) from the registered scores document. Pure."""
    rows = base['rows']
    by = {r['desc_hash']: r for r in rows}
    fin = act_finalists(rows, n)
    if mode == 'all':
        return fin, list(fin), [], None
    new = [h for h in fin if not by[h]['finalist']]
    reused = [h for h in fin if by[h]['finalist']]
    probe = None
    if new:
        probe = next((h for h in reused if by[h]['activated']), None) or \
            next((h for h in base['finalist_hashes'] if by[h]['activated'] and h not in fin), None)
    return fin, new, reused, probe


def main():
    cfgf, poolf, basef, outd, ckpt = sys.argv[1:6]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    mode = sys.argv[7] if len(sys.argv) > 7 else 'reuse'
    if mode not in ('reuse', 'all', 'preflight'):
        raise SystemExit('REFUSE: mode must be reuse, all or preflight')
    pool = json.loads(Path(poolf).read_text(encoding='utf-8'))
    base = json.loads(Path(basef).read_text(encoding='utf-8'))
    seed = int(pool['search_seed'])
    outd = Path(outd)
    if outd.resolve() == Path(basef).resolve().parent:
        raise SystemExit('REFUSE: the output directory is the registered wm_scores directory')
    if (base['pool_doc_sha256'] != pool['pool_doc_sha256'] or int(base['search_seed']) != seed
            or base.get('mode') != 'two_stage' or base['wm_checkpoint_sha256'] != pool['wm_checkpoint_sha256']
            or [r['desc_hash'] for r in base['rows']] != [e['desc_hash'] for e in pool['pool']]):
        raise SystemExit('REFUSE: the scores file is not the registered two-stage scoring of this pool')
    if sha(cfgf) != base['cfg_sha256']:
        raise SystemExit('REFUSE: not the config of the registered scoring')
    final = outd/f'WM_ACT_SCORES_seed{seed}.json'
    if final.exists() and mode != 'preflight':
        got = json.loads(final.read_text(encoding='utf-8'))
        if got['base_scores_sha256'] != sha(basef) or got['mode'] != mode:
            raise SystemExit(f'REFUSE: {final} belongs to other inputs or another mode')
        print('WM_ACT_DONE (kept)', seed, sha(final), flush=True)
        return
    from tools.wzprop import strategy_search as SS
    raw = json.loads(Path(cfgf).read_text(encoding='utf-8'))
    cfg = dict(SS.DEFAULTS, **raw.get('config', raw))
    if sha(ckpt) != pool['wm_checkpoint_sha256']:
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
    cands = {}
    for e in pool['pool']:
        s = SS.validate(dict(e['strategy']))
        if SS.strategy_key(s) != e['key'] or SS.strategy_key(SS.uncond(s)) != e['key']:
            raise SystemExit(f"REFUSE: key mismatch for {e['desc_hash'][:12]}")
        cands[e['desc_hash']] = (e, s)
    if sha('data/regime_STATES.json') != STATES_SHA256:
        raise SystemExit('REFUSE: data/regime_STATES.json is not the registered start-states file 74aef8f8')
    states = [m for m in json.loads(Path('data/regime_STATES.json').read_text()) if m.get('valid')]
    states = [m for m in states if m['cell'] == pool['cell']]
    fstates = states
    if int(cfg['n_final_states']) and len(states) > int(cfg['n_final_states']):
        fstates = random.Random(cfg['seed']+1).sample(states, int(cfg['n_final_states']))
    if (len(fstates) != int(base['n_final_states']) or int(cfg['K_final']) != int(base['K'])
            or int(cfg['finalists']) != int(base['finalists']) or cfg['H'] != base['H']):
        raise SystemExit('REFUSE: final protocol differs from the registered scoring')
    fin, new, reused, probe = plan(base, int(cfg['finalists']), 'reuse' if mode == 'preflight' else mode)
    batch_h = new+([probe] if probe else [])
    if mode == 'preflight':
        print('PREFLIGHT_OK', json.dumps(dict(
            search_seed=seed, finalists=len(fin), to_score=len(new), reused=len(reused), probe=probe and probe[:12],
            n_final_states=len(fstates), K_final=cfg['K_final'], H=cfg['H'], cfg_sha256=sha(cfgf),
            ckpt_sha256=sha(ckpt), base_scores_sha256=sha(basef),
            max_trajectories=len(fstates)*(len(batch_h)+2)*int(cfg['K_final']) if batch_h else 0)), flush=True)
        return
    outd.mkdir(parents=True, exist_ok=True)
    by = {r['desc_hash']: r for r in base['rows']}
    rows, counters, backend_meta = {}, None, None
    if batch_h:
        part = outd/f'WM_ACT_SCORES_seed{seed}.{mode}.final.json'
        ident = dict(pool_doc_sha256=pool['pool_doc_sha256'], keys=[cands[h][0]['key'] for h in batch_h],
                     worlds=[int(m['seed']) for m in fstates], K=int(cfg['K_final']))
        if part.exists():
            got = json.loads(part.read_text(encoding='utf-8'))
            if got['ident'] != ident:
                raise SystemExit(f'REFUSE: {part} was written for other inputs')
        else:
            from tools.wzprop import wm_backend as W
            kw = dict(checkpoint=ckpt, map_spec='maps/wz3to2/MAP_SPEC.json', device='cuda', max_rows=64)
            wm = W.ParallelWM('wma:WMA_READY.json', workers, api={}, local_kw=dict(kw, device='cpu'),
                              mem_fraction=None, **kw)
            t0 = time.time()
            per, n = SS.evaluate(wm, fstates, [cands[h][1] for h in batch_h], cfg['K_final'], cfg['H'],
                                 groups=bool(cfg['fairness']))
            prow = []
            for h in batch_h:
                e, s = cands[h]
                prow.append(dict(desc_hash=h, key=e['key'], n_roles=len(s['roles']),
                                 activated=any(r['feasible'] for r in per[e['key']]),
                                 summary=SS.summarize(per[e['key']], float(cfg['lam']), **sk)))
            got = dict(ident=ident, rows=prow, backend=getattr(wm, 'meta', None), counters=dict(
                candidates=len(batch_h), states=len(fstates), K=int(cfg['K_final']),
                candidate_state_draws=len(batch_h)*len(fstates)*int(cfg['K_final']), wm_trajectories=int(n),
                batch_calls=len(fstates), wall_s=round(time.time()-t0, 1)))
            write_once(part, json.dumps(got, indent=1, default=str))
            got = json.loads(part.read_text(encoding='utf-8'))
            print('final', got['counters'], flush=True)
        rows, counters, backend_meta = {r['desc_hash']: r for r in got['rows']}, got['counters'], got.get('backend')
    final_rows, pr = [], None
    for h in fin:
        src = 'new' if h in new else 'reused'
        r = rows[h] if src == 'new' else by[h]
        final_rows.append(dict(desc_hash=h, key=by[h]['key'], source=src, activated=bool(r['activated']),
                               summary=r['summary']))
    if probe:
        a, b = by[probe], rows[probe]
        pr = dict(desc_hash=probe, in_finalists=probe in fin, registered_score=a['summary']['score'],
                  rescored_score=b['summary']['score'], registered_activated=bool(a['activated']),
                  rescored_activated=bool(b['activated']),
                  abs_diff=abs(float(a['summary']['score'])-float(b['summary']['score'])),
                  identical=json.dumps(a['summary'], sort_keys=True, default=str)
                  == json.dumps(b['summary'], sort_keys=True, default=str) and bool(a['activated']) == bool(b['activated']))
    doc = dict(schema='claim_validation.e1.wm_act_scores.v1', label='sensitivity, post-registration, pre-outcome',
               mode=mode, rule=RULE, search_seed=seed, cell=pool['cell'], pool_doc_sha256=pool['pool_doc_sha256'],
               base_scores_sha256=sha(basef), cfg_sha256=sha(cfgf), wm_checkpoint_sha256=sha(ckpt),
               states_sha256=STATES_SHA256, n_final_states=len(fstates), K=cfg['K_final'], H=cfg['H'],
               lam=cfg['lam'], finalists=int(cfg['finalists']), act_finalist_hashes=fin, new=new, reused=reused,
               probe=pr, final_rows=final_rows, counters=counters,
               wm_rollouts=counters['wm_trajectories'] if counters else 0, backend=backend_meta, new_sumo_sim_s=0)
    write_once(final, json.dumps(doc, indent=1, sort_keys=True, default=str))
    print('WM_ACT_DONE', seed, sha(final), 'probe_identical', pr and pr['identical'], flush=True)


if __name__ == '__main__':   # the WM pool spawns children that re-import this file
    main()
