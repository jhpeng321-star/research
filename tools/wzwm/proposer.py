"""Staged WM search over lane-drop cooperation moves; uncertainty-aware ranking; top-k proposals.

Per moment (one observable/belief state at t0):
  0. pool    every (family, member set, knob) candidate (families.generate)
  1. screen  n_screen candidates, stratified over member sets (the structured N3 knobs first, then
             a seeded random fill), WM-scored with the base in one call: advantage T = J(base) - J(group)
  2. synergy the n_synergy best by T, re-scored with every member's solo in the same call:
             S = min_i J(solo_i) - J(group), M = min(J(base), min_i J(solo_i)) - J(group) (per draw,
             CRN-paired). M > 0 <=> T > 0 and S > 0, the two conditions SUMO admission tests.
  3. refine  refine_rounds x (n_parents best by score -> up to n_children one-step neighbours each:
             one knob to an adjacent grid value or one member swapped within the geometry rule)
  4. final   the `finalists` best, re-scored with K_final draws (group + solos + base, one call)
  5. select  [optional W3 head re-rank] -> greedy top-k with at most max_per_set per member set;
             abstain (no SUMO spend) only when abstain_below is set and the best score is below it.
Score (default 'lcb'): mean(M) - z * sd(M)/sqrt(K) over WM draws (one-sided; z = 1.645); 'mean'
and 'p_pos' (fraction of draws with M > 0) are alternatives. Proposals are N3 variants (family,
params, members) plus the WM prediction ('wm') and go to verify_plugin unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from . import families as F
from . import geometry as G
from . import wm_backend as W

SCHEMA = 'wmc-proposals-v1'
DEFAULTS = dict(k=5, H=300, K_screen=4, K_final=8, n_screen=48, n_synergy=16, refine_rounds=2, n_parents=6,
                n_children=6, finalists=12, max_per_set=2, z=1.645, score='lcb', abstain_below=None,
                families=list(F.BUILTIN_FAMILIES) + ['harmonize_rows'], max_sets=F.MAX_SETS, grid=None)
EXCLUDED_INPUTS = ('cell', 'demand', 'penetration', 'world seed', 'approach_speed', 'closing_lane_slow',
                   'lane_counts', 'source variants / outcomes')


def config(overrides=None):
    cfg = json.loads(json.dumps(DEFAULTS))
    for key, val in (overrides or {}).items():
        if key not in DEFAULTS:
            raise KeyError(f'unknown proposer config key {key!r}')
        cfg[key] = val
    if cfg['score'] not in ('lcb', 'mean', 'p_pos'):
        raise ValueError(f"score must be lcb|mean|p_pos, got {cfg['score']!r}")
    return cfg


def stat(x, z):
    x = np.asarray(x, float)
    k = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if k > 1 else 0.0
    se = sd / math.sqrt(k)
    return dict(mean=round(mean, 3), sd=round(sd, 3), lcb=round(mean - z * se, 3), p_pos=round(float((x > 0).mean()), 4),
                draws=[round(float(v), 1) for v in x])


def score_of(rec, basis):
    st = rec.get('M') or rec['T']
    if basis == 'lcb':
        return (st['lcb'], st['mean'])
    if basis == 'mean':
        return (st['mean'], st['lcb'])
    return (st['p_pos'], st['mean'])


def solo_key(spec):
    return 'solo|' + spec['id'] + '|' + json.dumps(spec['params'], sort_keys=True)


def evaluate_batch(wm, state, moves, K, H, seed, cavs, z, ledger, solos=True):
    """One WM call: base + groups (+ every member's solo); returns per-move stats (CRN-paired)."""
    acts = [W.action('base', 'base', None, 'base', [], H)]
    gid, sid = {}, {}
    for i, mv in enumerate(moves):
        gid[mv['key']] = f'g{i}'
        acts.append(W.action(f'g{i}', 'group', mv['family'], mv['key'], mv['members'], H, cavs))
        if solos:
            for spec in mv['members']:
                sk = solo_key(spec)
                if sk not in sid:
                    sid[sk] = f's{len(sid)}'
                    acts.append(W.action(sid[sk], 'solo', mv['family'], sk, [spec], H, cavs))
    out = W.normalize(wm.evaluate(state, acts, K, H, seed), acts, K, H)
    ledger['calls'] += 1
    ledger['rollouts'] += len(acts) * int(K)
    base = np.asarray(out['base'])
    recs = []
    for mv in moves:
        g = np.asarray(out[gid[mv['key']]])
        rec = dict(move=mv, K=int(K), T=stat(base - g, z))
        if solos:
            so = np.stack([np.asarray(out[sid[solo_key(s)]]) for s in mv['members']])
            best = so.min(0)
            rec.update(S=stat(best - g, z), M=stat(np.minimum(base, best) - g, z),
                       best_solo=mv['members'][int(np.argmin(so.mean(1)))]['id'],
                       solo_adv={s['id']: round(float((base - so[i]).mean()), 3) for i, s in enumerate(mv['members'])})
        recs.append(rec)
    return recs


def stratified(pool, n, seed):
    """Round-robin over member sets: structured knobs first, then a seeded random order."""
    rng = random.Random(seed)
    by, order = {}, []
    anchors = {}
    for vid in F.STRUCTURED:
        fam, knob = F.knob_of_structured(vid)
        anchors.setdefault(fam, []).append(knob)
    for mv in pool:
        if mv['set_key'] not in by:
            by[mv['set_key']] = []
            order.append(mv['set_key'])
        by[mv['set_key']].append(mv)
    queues = []
    for sk in order:
        items = by[sk]
        fam = items[0]['family']
        first = [m for kn in anchors.get(fam, []) for m in items if m['knob'] == kn]
        rest = [m for m in items if m not in first]
        rng.shuffle(rest)
        queues.append(first + rest)
    out = []
    j = 0
    while len(out) < n and any(j < len(q) for q in queues):
        for q in queues:
            if j < len(q) and len(out) < n:
                out.append(q[j])
        j += 1
    return out


def diverse_top(ranked, k, max_per_set):
    picks, per = [], {}
    for r in ranked:
        sk = r['move']['set_key']
        if per.get(sk, 0) < max_per_set:
            picks.append(r)
            per[sk] = per.get(sk, 0) + 1
        if len(picks) == k:
            return picks
    for r in ranked:                                   # relax the cap rather than propose fewer than k
        if len(picks) == k:
            break
        if r not in picks:
            picks.append(r)
    return picks


def pred(rec, rank, cfg, backend):
    mv = rec['move']
    out = dict(rank=rank, score_basis=cfg['score'], score=list(score_of(rec, cfg['score'])), K=rec['K'], H=cfg['H'],
               key=mv['key'], knob=mv['knob'], T=rec['T'], S=rec.get('S'), M=rec.get('M'),
               best_solo=rec.get('best_solo'), solo_adv=rec.get('solo_adv'), backend=backend)
    if rec.get('head'):
        out['head'] = rec['head']
    return out


def propose_moment(wm, m, cfg, state_file=None, head=None):
    """Top-k WM proposals for one N3/PLANGEN moment row (MOMENTS.json format)."""
    t_start = time.time()
    G.check_seed(m['seed'], where=f"(moment {m.get('sid')})")
    mi = W.moment_input(m, state_file)
    state = wm.observe(mi)
    cavs = mi['cavs']
    fams = list(cfg['families'])
    nf = F.n9_module() if 'harmonize_rows' in fams else None
    if nf is None:
        fams = [f for f in fams if f != 'harmonize_rows']
    pool = F.generate(cavs, cfg['grid'], fams, cfg['max_sets'], nf)
    seed = mi['crn_seed']
    ledger = dict(calls=0, rollouts=0)
    row = dict(schema=SCHEMA, sid=m['sid'], t0=int(m['t0']), map=G.MAP,
               bookkeeping=dict(world_seed=m['seed'], cell=m.get('cell'), demand=m.get('demand'),
                                penetration=m.get('penetration'), window=m.get('window'), note='labels, not inputs'),
               backend=wm.meta, config=cfg, families_searched=fams, n9=F.n9_status(nf),
               observable_inputs=dict(cav_self_states=len(cavs), wm_rows=len(state.get('wm_rows') or [])),
               excluded_inputs=list(EXCLUDED_INPUTS), n_pool=len(pool), n_member_sets=len({p['set_key'] for p in pool}))
    k = int(cfg['k'])
    if not pool or k <= 0:
        row.update(abstain=True, abstain_reason='no candidate geometry' if not pool else 'k=0', variants={},
                   ledger=ledger, n_evaluated=0, wall_s=round(time.time() - t_start, 3))
        return row
    z, H, basis = float(cfg['z']), int(cfg['H']), cfg['score']
    screened = evaluate_batch(wm, state, stratified(pool, int(cfg['n_screen']), seed), cfg['K_screen'], H, seed,
                              cavs, z, ledger, solos=False)
    screened.sort(key=lambda r: score_of(r, basis), reverse=True)
    evaluated = {}
    first = [r['move'] for r in screened[:int(cfg['n_synergy'])]]
    for r in evaluate_batch(wm, state, first, cfg['K_screen'], H, seed, cavs, z, ledger):
        evaluated[r['move']['key']] = r
    rng = random.Random(seed + 1)
    tried = {r['move']['key'] for r in screened} | set(evaluated)
    for _ in range(int(cfg['refine_rounds'])):
        parents = sorted(evaluated.values(), key=lambda r: score_of(r, basis), reverse=True)[:int(cfg['n_parents'])]
        children = []
        for p in parents:
            nb = [c for c in F.neighbors(p['move'], cavs, cfg['grid'], nf) if c['key'] not in tried]
            rng.shuffle(nb)
            for c in nb[:int(cfg['n_children'])]:
                tried.add(c['key'])
                children.append(c)
        if not children:
            break
        for r in evaluate_batch(wm, state, children, cfg['K_screen'], H, seed, cavs, z, ledger):
            evaluated[r['move']['key']] = r
    fin = sorted(evaluated.values(), key=lambda r: score_of(r, basis), reverse=True)[:int(cfg['finalists'])]
    final = evaluate_batch(wm, state, [r['move'] for r in fin], cfg['K_final'], H, seed, cavs, z, ledger)
    final.sort(key=lambda r: score_of(r, basis), reverse=True)
    if head is not None:
        from . import wallfit_bridge
        final = wallfit_bridge.rerank(final, state, head, H)
    picks = diverse_top(final, k, int(cfg['max_per_set']))
    best = score_of(picks[0], basis)[0] if picks else None
    abstain = cfg['abstain_below'] is not None and (best is None or best <= float(cfg['abstain_below']))
    variants = {}
    if not abstain:
        for i, r in enumerate(picks):
            variants[f'wm{i}'] = dict(F.to_variant(r['move']), wm=pred(r, i, cfg, wm.name))
    row.update(abstain=abstain, abstain_reason='best score <= abstain_below' if abstain else None, best_score=best,
               variants=variants, n_evaluated=len(evaluated), n_screened=len(screened), ledger=ledger,
               finalists=[dict(key=r['move']['key'], M=r['M']['mean'], M_lcb=r['M']['lcb'], T=r['T']['mean'],
                               S=r['S']['mean']) for r in final],
               structured_in_pool=sorted(v for v, mv in F.structured_moves(cavs).items()
                                         if mv['key'] in {p['key'] for p in pool}),
               wall_s=round(time.time() - t_start, 3))
    return row


def matched_k(m, default=5):
    """k = number of structured (non-random) variants the source pipeline screened at this moment."""
    n = sum(1 for v in (m.get('variants') or {}).values() if v.get('family') != 'R_random')
    return n or default


def native_state(source_root, m):
    if source_root is None:
        return None
    return (Path(source_root) / 'runs' / m['cell'] / f"s{m['seed']}" / 'native' / 'states' / f"t{m['t0']}.xml.gz")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def propose_file(moments_path, out_path, wm, cfg, *, source_root=None, sids=None, head=None, k_matched=False,
                 limit=None):
    """PROPOSALS.jsonl (one row per valid moment) + PROPOSALS_RECEIPT.json; never overwrites."""
    moments_path, out_path = G.check_path(moments_path), G.check_path(out_path)
    moments = [m for m in json.loads(moments_path.read_text()) if m.get('valid')]
    if sids:
        want = set(sids)
        moments = [m for m in moments if m['sid'] in want]
    if limit:
        moments = moments[:int(limit)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = dict(path=str(moments_path), sha256=sha256(moments_path))
    t0 = time.time()
    n = rollouts = abstained = 0
    with out_path.open('x') as fh:
        for m in moments:
            c = dict(cfg, k=matched_k(m, cfg['k'])) if k_matched else cfg
            row = propose_moment(wm, m, c, native_state(source_root, m), head)
            row['source_moments'] = src
            fh.write(json.dumps(row, sort_keys=True) + '\n')
            fh.flush()
            n += 1
            rollouts += row['ledger']['rollouts']
            abstained += bool(row['abstain'])
    receipt = dict(schema=SCHEMA + '-receipt', proposals=str(out_path), proposals_sha256=sha256(out_path),
                   moments=str(moments_path), moments_sha256=src['sha256'], n_moments=n, abstained=abstained,
                   wm_rollouts=rollouts, backend=wm.meta, config=cfg, k_matched=bool(k_matched),
                   wallfit_head=None if head is None else (str(head) if not isinstance(head, dict) else 'in-memory'), wall_s=round(time.time() - t0, 2), new_sumo_sim_s=0)
    rp = out_path.with_name(out_path.stem + '_RECEIPT.json')
    with rp.open('x') as fh:
        fh.write(json.dumps(receipt, indent=1, sort_keys=True, default=str))
    return receipt


# ------------------------------------------------------------------------------------------ score existing variants
def variant_move(vid, v):
    """A move dict (evaluate_batch input) for an existing N3/PLANGEN/WMC variant."""
    ids = [m['id'] for m in v['members']]
    return dict(key=f"{v['family']}|{','.join(sorted(ids))}|{json.dumps(v['params'], sort_keys=True)}",
                set_key=F.set_key(v['family'], ids), family=v['family'], member_ids=ids, members=v['members'],
                params=v['params'], knob=None, vid=vid)


def score_moments(wm, moments_path, out_path, *, K=8, H=300, z=1.645, source_root=None, sids=None):
    """WM predictions (T, S, M per CRN-paired draw) for every variant already in a MOMENTS.json, zero SUMO.
    Feeds the zero-SUMO WM ranking check (compare.rank_check) and the W3 labels (labels --wm-scores)."""
    moments_path, out_path = G.check_path(moments_path), G.check_path(out_path)
    moments = [m for m in json.loads(moments_path.read_text()) if m.get('valid')]
    if sids:
        moments = [m for m in moments if m['sid'] in set(sids)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = rollouts = 0
    with out_path.open('x') as fh:
        for m in moments:
            mi = W.moment_input(m, native_state(source_root, m))
            state = wm.observe(mi)
            moves = [variant_move(vid, v) for vid, v in sorted(m['variants'].items())]
            ledger = dict(calls=0, rollouts=0)
            recs = evaluate_batch(wm, state, moves, K, H, mi['crn_seed'], mi['cavs'], z, ledger) if moves else []
            for r in recs:
                fh.write(json.dumps(dict(schema='wmc-variant-scores-v1', sid=m['sid'], variant=r['move']['vid'],
                                         family=r['move']['family'], members=r['move']['member_ids'],
                                         wm=dict(T=r['T'], S=r['S'], M=r['M'], K=int(K), H=int(H),
                                                 best_solo=r['best_solo'], solo_adv=r['solo_adv'], backend=wm.name)),
                                    sort_keys=True) + '\n')
            n += len(recs)
            rollouts += ledger['rollouts']
    return dict(scores=str(out_path), n_variants=n, n_moments=len(moments), wm_rollouts=rollouts, backend=wm.meta,
                wall_s=round(time.time() - t0, 2), new_sumo_sim_s=0)
