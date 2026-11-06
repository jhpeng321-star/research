"""BOOST-PRE learnability readout (zero SUMO): can the ACTUAL trigger head learn the examples?

MAIN 2026-11-06 ~17:00. Uses the warm-start pipeline unchanged (coop_warmstart): the clean4
EpisodePolicy trigger (frozen MAPPO u0 trunk + trigger_head over the runtime descriptor and the
N2 lane-drop extension columns), decision tables from the online causal observer, the registered
do-not-act rule, the identical optimiser/effort. Seed-grouped K-fold CV: no world seed is in both
train and test. Reports held-out top-1 accuracy, AUC and precision@k of P(act with this move)
against the base rate for (a) B rows + negatives, (b) R rows + negatives, (c) within admitted
moments, whether the top-scored move is an admitted one.

Observation source (honest limitation): the raw mining rows carry no observation. The only
local observable state per moment is the miners' MOMENTS.json ``all_cavs_t0`` (every CAV's own
edge/lane/x/v at t0 = CAV self-reports). Frames are those CAV rows projected through the N1
map descriptor; HDVs are NOT recorded locally, so own-sensing neighbour slots only contain CAVs,
and t0-1 is the constant-speed back-extrapolation. Rows of sources without MOMENTS (N8
coopkinds, plangen2) are not used. SUMO state files are never opened.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

from . import coop_warmstart as cw

ROWS_ROOT = 'E:/research-dense/runs/active/C4-V7FT-20261017'
MAP_SPEC = 'E:/research-dense/assets/research/newmap_candidates/wz3to2'
SOURCES = dict(
    moments=('newmap2/results/MOMENTS.json', 'plangen/results/MOMENTS.json', 'lowcav/results/MOMENTS.json'),
    B=('NEWMAP_MOVES_B.jsonl', 'PLANGEN_MOVES_B.jsonl', 'LOWCAV_MOVES_B.jsonl'),
    R=('NEWMAP_MOVES_R.jsonl', 'PLANGEN_MOVES_R.jsonl'),
    negatives=('NEWMAP_NEGATIVES.jsonl', 'PLANGEN_NEGATIVES.jsonl', 'LOWCAV_NEGATIVES.jsonl'))
EDGE_OFFSET = {'up': 0.0, ':n1_0': 2696.0, 'wz': 2704.0}   # probe xglob (nm2_core/lc_core)
CAV_LENGTH_M = 5.0


def clean4_like_config(desc, *, training_seed=20261101):
    """The clean3 production trigger architecture + N2 lane-drop features/moves/v2v ON.

    Same widths as launch_production.OPTIMIZATION (hidden 64, 8 candidates, 4 partner
    slots), V2V intent base (v2v_dim 48, horizon 5) and G5 lane families, as the clean4 B/R
    config is built on clean3. W1/W2/N9 add only zero-initialised columns and are left off.
    """
    from . import lane_drop
    cfg = cw.dry_config()
    cfg.update(training_seed=int(training_seed), v2v_intent_horizon=5, v2v_intent_detach_trunk=True,
               v2v_intent_loss_weight=1.0, group_lane_moves=True)
    cfg.update(lane_drop.config_keys(provenance='BOOST-PRE learnability readout (clean4-like)',
                                     descriptor=desc, features=True, moves_on=True, v2v=True))
    return cfg


def moment_frames(paths, desc):
    """{sid: (seed, t0, {t0-1: rows, t0: rows})} from CAV self-reports at t0 (N1 projection)."""
    out = {}
    for path in paths:
        for m in json.loads(cw._guard(path).read_text(encoding='utf-8')):
            rows = []
            for c in m['all_cavs_t0']:
                edge, lane = c['edge'], int(c['lane'])
                if edge not in EDGE_OFFSET or (edge, lane) not in desc.lane_table:
                    continue
                wm, s = desc.lane_and_s(f'{edge}_{lane}', float(c['x'])-EDGE_OFFSET[edge])
                rows.append(dict(vid=c['id'], s_m=float(s), v_mps=float(c['v']), lane=int(wm), is_cav=True,
                                 length_m=CAV_LENGTH_M))
            t0 = int(m['t0'])
            previous = [dict(r, s_m=r['s_m']-r['v_mps']) for r in rows]
            if m['sid'] in out:
                raise ValueError('duplicate moment id '+m['sid'])
            out[m['sid']] = (int(m['seed']), t0, {t0-1: cw.whitelist(previous), t0: cw.whitelist(rows)})
    return out


def r_random_as_joint(row, frames, desc):
    """R's random group as the joint lane-drop family its OBSERVED lanes form (else None)."""
    lanes = {r['vid']: r['lane'] for r in frames}
    members = row['members']
    if any(v not in lanes for v in members):
        return None
    observed = [lanes[v] for v in members]
    closing, adjacent = desc.bottleneck['closing_lane'], desc.bottleneck['adjacent_lane']
    if len(members) == 2 and sorted(observed) == sorted((closing, adjacent)):
        return dict(row, family='closing_pair')
    if len(members) == 3 and sorted(observed) == [0, 1, 2]:
        return dict(row, family='abreast_wall')
    return None


def _key(row):
    return '|'.join((str(row.get('situation_id')), str(row.get('family')), str(row.get('variant')),
                     ','.join(sorted(row['members']))))


def build(policy, desc, frames_by_sid, rows, *, kind, slate_seed, cells=cw.TRAIN_CELLS):
    records, reasons = [], Counter()
    seen = set()
    for row in rows:
        why = cw.admission(row, cells)
        if why is not None:
            reasons['excluded:'+why] += 1
            continue
        if kind == 'negative':
            label, why = cw.do_not_act(row)
            if not label:
                reasons['negative_rule:'+why] += 1
                continue
        moment = frames_by_sid.get(row.get('situation_id'))
        if moment is None:
            reasons['no_moment_snapshot'] += 1
            continue
        seed, t0, frames = moment
        if seed != cw.world_seed(row) or t0 != int(row['t0']):
            raise ValueError('moment/row world or t0 mismatch '+str(row.get('situation_id')))
        if row.get('family') == 'R_random':
            row = r_random_as_joint(row, frames[t0], desc)
            if row is None:
                reasons['R_random_not_a_joint_lane_geometry'] += 1
                continue
        key = _key(row)
        if key in seen:
            reasons['crn_replicate_row'] += 1
            continue
        seen.add(key)
        out, why = cw.frame_records(policy, row, frames, target=1 if kind == 'positive' else 0, kind=kind,
                                    seed=seed, dry=False, slate_seed=slate_seed)
        reasons.update(why)
        if not out:
            reasons['row_without_usable_ego'] += 1
        for r in out:
            r['row_key'] = key
        records += out
    return records, reasons


def _input_key(record):
    b = record['batch']
    return cw.digest(dict(context=json.loads(json.dumps(b.context, default=str)),
                          descriptors=b.descriptors.tolist(), mask=b.mask.tolist()))


def folds(seeds, k):
    ranked = sorted(set(seeds), key=lambda s: hashlib.sha256(f'boost-pre-cv:{s}'.encode()).hexdigest())
    k = min(k, len(ranked))
    return [set(ranked[i::k]) for i in range(k)]


@torch.no_grad()
def scores(policy, records):
    out = []
    for r in records:
        b = r['batch']
        logits = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0]
        prob = logits.softmax(-1)
        for i in range(len(b.actions)):
            out.append(dict(kind=r['kind'], family=r['family'], seed=r['seed'], sid=r['situation_id'],
                            row_key=r['row_key'], score=float(prob[i, 1]), target=int(b.actions[i]),
                            correct=int(logits[i].argmax()) == int(b.actions[i]), p_none=float(prob[i, 0])))
    return out


def auc(values):
    pos = [v['score'] for v in values if v['kind'] == 'positive']
    neg = [v['score'] for v in values if v['kind'] == 'negative']
    if not pos or not neg:
        return None
    wins = sum((p > n)+.5*(p == n) for p in pos for n in neg)
    return wins/(len(pos)*len(neg))


def metrics(values):
    n = len(values)
    if not n:
        return dict(n=0)
    positives = sum(v['kind'] == 'positive' for v in values)
    ranked = sorted(values, key=lambda v: -v['score'])
    out = dict(n=n, positives=positives, base_rate=positives/n, accuracy=sum(v['correct'] for v in values)/n,
               always_none_accuracy=(n-positives)/n, auc=auc(values),
               positive_accuracy=(sum(v['correct'] for v in values if v['kind'] == 'positive')/positives
                                  if positives else None),
               negative_accuracy=(sum(v['correct'] for v in values if v['kind'] == 'negative')/(n-positives)
                                  if n-positives else None))
    for k in (5, 10, 20):
        if n >= k:
            out[f'precision_at_{k}'] = sum(v['kind'] == 'positive' for v in ranked[:k])/k
    if positives:
        out['precision_at_npos'] = sum(v['kind'] == 'positive' for v in ranked[:positives])/positives
    return out


def within_moment(values):
    """(c) per moment with >=1 admitted and >=1 failed move: is the top-scored row admitted?"""
    rows = defaultdict(list)
    for v in values:
        rows[(v['sid'], v['row_key'], v['kind'])].append(v['score'])
    by_sid = defaultdict(list)
    for (sid, _key, kind), s in rows.items():
        by_sid[sid].append((sum(s)/len(s), kind))
    hits, chance, n = 0, 0., 0
    for items in by_sid.values():
        kinds = {k for _, k in items}
        if kinds != {'positive', 'negative'}:
            continue
        n += 1
        top = max(items, key=lambda x: x[0])
        hits += top[1] == 'positive'
        chance += sum(k == 'positive' for _, k in items)/len(items)
    return dict(moments=n, top1_admitted=hits/n if n else None, chance=chance/n if n else None)


def run(*, base, out_path, k=5, training=None, rows_root=ROWS_ROOT, map_spec=MAP_SPEC):
    from .map_descriptor import load
    settings = cw.training_settings(training)
    desc = load(map_spec)
    cfg = clean4_like_config(desc)
    root = Path(rows_root)
    frames = moment_frames([root/p for p in SOURCES['moments']], desc)
    template = cw.build_policy(base, cfg, 'B')
    initial = cw.state_sha256(template)
    built, reasons = {}, {}
    for name, kind in (('B', 'positive'), ('R', 'positive'), ('negatives', 'negative')):
        rows = [r for p in SOURCES[name] for r in cw.read_rows(root/p)]
        if name in ('B', 'R') and any(r.get('arm') != name for r in rows):
            raise ValueError(f'{name} rows must be own-arm rows')
        built[name], reasons[name] = build(template, desc, frames, rows, kind=kind, slate_seed=settings['slate_seed'])
    report = dict(schema='coopmwm-boost-pre-learncheck-v1', rule=cw.RULE, negative_rule=cw.RULE_NEGATIVE,
                  observation=__doc__.split('Observation source')[1].strip(), k_folds=k, training=settings,
                  config=dict(hidden_dim=cfg['layer_config']['hidden_dim'], trigger_candidates=cfg['trigger_candidates'],
                              partner_slots=cfg['trigger_partner_slots'], v2v_dim=cfg['v2v_dim'],
                              lane_drop=dict(features=True, moves=True, v2v=True), group_lane_moves=True),
                  base_sha256=cw.state_sha256(template.layer.base), moments=len(frames),
                  record_counts={n: len(v) for n, v in built.items()},
                  reasons={n: dict(v) for n, v in reasons.items()}, tasks={})
    conflicts = Counter()
    for name in ('B', 'R'):
        records = cw.deduplicate(built[name]+built['negatives'])
        labels = defaultdict(set)
        for r in records:
            labels[_input_key(r)].add(r['kind'])
        conflicts[name] = sum(1 for kinds in labels.values() if len(kinds) > 1)
        seeds = sorted({r['seed'] for r in records})
        task = dict(records=len(records), positives=sum(r['kind'] == 'positive' for r in records),
                    seeds=len(seeds), identical_input_conflicting_label=conflicts[name], folds=[])
        pooled_test, pooled_train = [], []
        for i, test in enumerate(folds(seeds, k)):
            train = [r for r in records if r['seed'] not in test]
            hold = [r for r in records if r['seed'] in test]
            policy = cw.build_policy(base, cfg, name)
            if cw.state_sha256(policy) != initial:
                raise ValueError('fold policy does not start from the common initial layer')
            fold = dict(fold=i, test_seeds=sorted(test), train_records=len(train), test_records=len(hold),
                        train_positives=sum(r['kind'] == 'positive' for r in train),
                        test_positives=sum(r['kind'] == 'positive' for r in hold))
            if not any(r['kind'] == 'positive' for r in train):
                fold['skipped'] = 'no positive TRAIN records in this fold'
                task['folds'].append(fold)
                continue
            trained = cw.train(policy, train, settings)
            tr, te = scores(policy, train), scores(policy, hold)
            pooled_train += tr
            pooled_test += te
            fold.update(train=metrics(tr), test=metrics(te), effort_steps=trained['effort']['optimizer_steps'])
            task['folds'].append(fold)
        task['heldout_pooled'] = metrics(pooled_test)
        task['train_pooled'] = metrics(pooled_train)
        task['within_admitted_moments_heldout'] = within_moment(pooled_test)
        task['heldout_by_family'] = {f: metrics([v for v in pooled_test if v['family'] == f])
                                     for f in sorted({v['family'] for v in pooled_test})}
        report['tasks'][name] = task
    Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True, default=str)+'\n', encoding='utf-8')
    return report


def verdict(task):
    test = task['heldout_pooled']
    if not test.get('n') or test.get('auc') is None:
        return 'NOT TESTABLE (no held-out positives/negatives)'
    beats = test['auc'] > .5 and test.get('precision_at_npos', 0) > test['base_rate']
    return 'BEATS base rate' if beats else 'DOES NOT beat base rate'


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint')
    parser.add_argument('--base-sha256')
    parser.add_argument('--out', required=True)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--training')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(min(4, max(1, args.threads)))
    from .map_descriptor import load
    cfg = clean4_like_config(load(MAP_SPEC))
    if args.base_checkpoint:
        path = cw._guard(args.base_checkpoint)
        if args.base_sha256 and cw._sha_file(path) != args.base_sha256:
            raise SystemExit('base checkpoint sha256 mismatch')
        from .v2v_runtime import load_checkpoint
        base = load_checkpoint(path, config=cfg)[0]
    else:
        from .v2v_intent import IntentPolicy
        torch.manual_seed(int(cfg['training_seed']))
        base = IntentPolicy()
    report = run(base=base, out_path=args.out, k=args.k,
                 training=json.loads(args.training) if args.training else None)
    for name, task in report['tasks'].items():
        t = task['heldout_pooled']
        print(name, verdict(task), json.dumps(dict(heldout=t, train=task['train_pooled'],
                                                  within=task['within_admitted_moments_heldout']), default=str))
    return report


if __name__ == '__main__':
    main()
