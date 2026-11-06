"""W3 wall-fit head (c4/wm-wallfit) as an OPTIONAL re-ranker of the WM finalists, plus the wz3to2
label builder that lets the head be trained later on N3 / PLANGEN / LOWCAV / WMC SUMO labels.

Re-rank (proposer --wallfit-head HEAD.pt, default OFF): every finalist keeps its WM statistics; the head
adds head.score/synergy/admit_prob and the finalists are re-sorted by head score (WM score breaks ties).
Head inputs: the WM-frame observable rows of the state (closing lane = WM lane 2, drop at s = 75.44, the
N1 projection the head's features were written for), the move (family 'wall' for abreast walls, other
families -> fam_other; onset = start_s, hold = hold_s, depth = families.strength = dmax x relative speed
cut) and the WM's own prediction (synergy = mean S, saving = mean T, best-solo saving = mean T - mean S,
draws = S draws). World labels are never inputs.

Labels (python -m tools.wzwm labels): one wallfit-labels-v1 pair per (moment, verified move) with one
seed row per replicate (crn1 = undithered r0, crn2/crn3 = the dithered replicates): synergy S, its LB95,
advantage T, best-solo advantage T - S, admission (final funnel verdict; None while undetermined).
Train later with the map-scoped guard:  WALLFIT_MAP=wz3to2 python -m tools.wallfit run --labels ...
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import families as F
from . import geometry as G

FAMILY_MAP = {'abreast_wall': 'wall'}
FINAL_REJECT = {'solos', 'r1', 'r2', 'r0ext'}
UNDETERMINED = {'screen_missing', 'r0_not_picked', 'r1_missing', 'r2_missing', 'r0ext_missing'}


def wallfit_move(family, member_ids, params):
    return dict(family=FAMILY_MAP.get(family, family), members=[str(v) for v in member_ids],
                onset=int(params.get('start_s', 0)), hold=int(params['hold_s']), depth=F.strength(params))


def wm_dict(T, S, H):
    if not T or not S:
        return None
    return dict(mean_synergy=S['mean'], saving_vs_policy=T['mean'], best_solo_saving=round(T['mean'] - S['mean'], 3),
                draws=list(S.get('draws') or []), horizon_s=int(H))


def rerank(recs, state, head, H):
    from tools.wallfit import rank_api
    rows, moves = [], []
    for r in recs:
        w = wm_dict(r['T'], r['S'], H)
        rows.append(dict(wm_mean_synergy=w['mean_synergy'], wm_saving_vs_policy=w['saving_vs_policy'],
                         wm_best_solo_saving=w['best_solo_saving'], wm_synergy_draws=w['draws']))
        mv = r['move']
        moves.append(wallfit_move(mv['family'], mv['member_ids'], mv['params']))
    query = dict(record=dict(rows=list(state.get('wm_rows') or [])))
    preds = rank_api.score_rows(rows, moves, query, int(H), head=head)
    out = []
    for r, p in zip(recs, preds):
        out.append(dict(r, head=dict(score=p['score'], synergy_veh_s=p['synergy_veh_s'], admit_prob=p['admit_prob'],
                                     latent_used=p['latent_used'])))
    lcb = {id(o): (o.get('M') or o['T'])['lcb'] for o in out}
    out.sort(key=lambda o: (-o['head']['score'], -lcb[id(o)]))
    return out


# ------------------------------------------------------------------------------------------ labels
def _admitted(vd, rep_failed_at=None):
    st = vd['stage']
    if st == 'admitted':
        return True
    if st in FINAL_REJECT:
        return False
    if st == 'screen':
        t_ok = (vd.get('T_lb_r0') or 0) > 0 and (vd.get('T_r0') or 0) >= 1
        safe = (vd.get('safety_r0') or {}).get('collisions', 0) == 0 and (vd.get('safety_r0') or {}).get('teleports', 0) == 0
        return None if (t_ok and safe) else False
    return None


def seed_rows(vd, move, source, path):
    from tools.wallfit.dataset import _seed_row
    base = dict(source=source, source_path=str(path), tag=vd.get('variant'), state_id=vd['sid'],
                world_seed=G.check_seed(vd['seed'], where=f'({path})'), t0=vd['t0'], setting='wz3to2',
                arm=vd.get('proposer') or vd.get('arm'), plan_id=vd.get('variant'), move=move, tail_s=300,
                admitted=_admitted(vd), status=vd['stage'])
    reps = vd.get('reps') or {}
    if not reps:
        if vd.get('T_r0') is None:
            return []
        S = vd.get('S_r0')
        return [_seed_row(**base, seed_id='crn1', seed_index=1, J_policy=vd.get('J_base_r0'), J_group=vd.get('J_group_r0'),
                          group_adv=vd['T_r0'], adv_lb95=vd.get('T_lb_r0'), synergy=S, synergy_lb95=vd.get('S_lb_r0'),
                          best_solo_adv=None if S is None else vd['T_r0'] - S,
                          solo_advs=vd.get('solo_gain_r0'))]
    out = []
    for k in sorted(reps, key=int):
        r = reps[k]
        S = r.get('S')
        out.append(_seed_row(**base, seed_id=f'crn{int(k)}', seed_index=int(k), J_policy=r.get('J_base'),
                             J_group=r.get('J_group'), group_adv=r.get('T'), adv_lb95=r.get('T_lb'), synergy=S,
                             synergy_lb95=r.get('S_lb'), best_solo_adv=None if S is None or r.get('T') is None
                             else r['T'] - S, solo_advs=r.get('solo_gain')))
    return out


def build_labels(summaries, moments_files, out_dir, wm_scores=None):
    """LABELS.jsonl (wallfit-labels-v1) + BUILD_RECEIPT.json from wz3to2 funnel summaries. wm_scores
    (proposer.score_moments output) supplies the WM prediction of variants that were not WM proposals."""
    from tools.wallfit.dataset import SCHEMA, aggregate, move_key
    begin = time.time()
    moments = {}
    for p in moments_files:
        for m in json.loads(G.check_path(p).read_text()):
            if m.get('valid'):
                G.check_seed(m['seed'], where=f'({p})')
                moments.setdefault(m['sid'], m)
    pairs, skips = {}, dict(no_moment=0, no_outcome=0)
    scores = {}
    for path in wm_scores or ():
        for line in G.check_path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                scores[(r['sid'], r['variant'])] = r['wm']
    for p in summaries:
        doc = json.loads(G.check_path(p).read_text())
        source = doc.get('phase_id') or Path(p).stem
        for vd in doc['verdicts']:
            m = moments.get(vd['sid'])
            if m is None:
                skips['no_moment'] += 1
                continue
            move = wallfit_move(vd['family'], vd['members'], vd['params'])
            rows = seed_rows(vd, move, source, p)
            if not rows:
                skips['no_outcome'] += 1
                continue
            pid = f"{vd['sid']}#{move_key(move)}"
            wp = vd.get('wm_prediction') or scores.get((vd['sid'], vd['variant'])) or {}
            wm = wm_dict(wp.get('T'), wp.get('S'), wp.get('H') or 300) if wp else None
            cur = pairs.get(pid)
            if cur is None:
                situation = dict(source='context', query_id=None,
                                 rows=[r for r in (G.wm_row(c) for c in m['all_cavs_t0']) if r is not None])
                pairs[pid] = dict(schema=SCHEMA, pair_id=pid, state_id=vd['sid'], world_seed=int(vd['seed']),
                                  t0=vd['t0'], settings=['wz3to2'], move=move, plan_ids=[vd['variant']],
                                  situation=situation, wm=wm, seeds=rows)
            else:
                cur['seeds'] += rows
                cur['plan_ids'] = sorted(set(cur['plan_ids']) | {vd['variant']})
                cur['wm'] = cur['wm'] or wm
    out_rows = []
    for r in pairs.values():
        r['labels'] = aggregate(r['seeds'])
        out_rows.append(r)
    out = G.check_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lp = out / 'LABELS.jsonl'
    with lp.open('x') as fh:
        fh.write(''.join(json.dumps(r, sort_keys=True) + '\n' for r in sorted(out_rows, key=lambda r: r['pair_id'])))
    receipt = dict(schema='wmc-wallfit-labels-receipt-v1', summaries=[str(s) for s in summaries],
                   wm_scores=[str(s) for s in wm_scores or ()],
                   moments=[str(m) for m in moments_files], pairs=len(out_rows),
                   pairs_measured=sum(r['labels']['n_measured'] > 0 for r in out_rows),
                   pairs_admitted=sum(bool(r['labels']['admitted_any']) for r in out_rows),
                   pairs_with_wm=sum(r['wm'] is not None for r in out_rows), skips=skips,
                   world_seeds=sorted({r['world_seed'] for r in out_rows}), wall_s=round(time.time() - begin, 2),
                   train_command='WALLFIT_MAP=wz3to2 python -B -m tools.wallfit run --labels LABELS.jsonl --out <dir>',
                   new_sumo_sim_s=0)
    with (out / 'BUILD_RECEIPT.json').open('x') as fh:
        fh.write(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt
