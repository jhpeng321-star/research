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
    """The clean3 production trigger architecture with the N4 clean4 B/R option set (17:40).

    Widths as launch_production.OPTIMIZATION (hidden 64, 8 candidates, 4 partner slots),
    V2V intent base (v2v_dim 48, horizon 5), G5 lane families; N4: lane_drop features+moves+
    v2v (range 200 m, member_cap 3), W1 v2v_wall (member_cap 4, range 200 m), W2 pressure
    features + release + negative labels ON, nm_families and coop_awareness OFF, RSU off
    (pending its smoke check; it is not a trigger-logit input).
    """
    from . import lane_drop, v2v_wall
    cfg = cw.dry_config()
    cfg.update(training_seed=int(training_seed), v2v_intent_horizon=5, v2v_intent_detach_trunk=True,
               v2v_intent_loss_weight=1.0, group_lane_moves=True, wall_pressure_features=True,
               wall_release=True, wall_negative_labels=True)
    cfg.update(lane_drop.config_keys(provenance='BOOST-PRE learnability readout (clean4 N4 17:40 option set)',
                                     descriptor=desc, features=True, moves_on=True, v2v_range_m=200.,
                                     member_cap=3))
    cfg.update(v2v_wall.config_keys(provenance='BOOST-PRE learnability readout (clean4 N4 17:40 option set)',
                                    member_cap=4, range_m=200.))
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
            moment = frames_by_sid.get((cw.world_seed(row), int(row['t0'])))
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


def runtime_frames(paths):
    """EXACT runtime frames (Plan A replays, W2 frame format: world seed, t0, frames {t0-1, t0}
    with sensor-whitelisted rows of ALL vehicles incl. HDVs; optional situation_id). Keys: the
    situation id when present and (world seed, t0)."""
    out = {}
    for path in paths:
        rows = cw.read_rows(path)
        index = cw.frame_entries(rows)
        for row in rows:
            seed, t0 = cw.world_seed(row), int(row['t0'])
            value = (seed, t0, index[(seed, t0)])
            out[(seed, t0)] = value
            if row.get('situation_id'):
                out[row['situation_id']] = value
    return out


def prepare(base, *, training=None, rows_root=ROWS_ROOT, map_spec=MAP_SPEC, frames_paths=()):
    """Decision tables built ONCE (weights-independent) on the common initial layer."""
    from .map_descriptor import load
    settings = cw.training_settings(training)
    desc = load(map_spec)
    cfg = clean4_like_config(desc)
    root = Path(rows_root)
    moments = {m['sid']: m for p in SOURCES['moments']
               for m in json.loads(cw._guard(root/p).read_text(encoding='utf-8'))}
    frames = (runtime_frames(frames_paths) if frames_paths else
              moment_frames([root/p for p in SOURCES['moments']], desc))
    template = cw.build_policy(base, cfg, 'B')
    built, reasons, rows_by = {}, {}, {}
    for name, kind in (('B', 'positive'), ('R', 'positive'), ('negatives', 'negative')):
        rows = [r for p in SOURCES[name] for r in cw.read_rows(root/p)]
        if name in ('B', 'R') and any(r.get('arm') != name for r in rows):
            raise ValueError(f'{name} rows must be own-arm rows')
        rows_by[name] = rows
        built[name], reasons[name] = build(template, desc, frames, rows, kind=kind, slate_seed=settings['slate_seed'])
    source = ([dict(path=str(x), sha256=cw._sha_file(x)) for x in frames_paths] if frames_paths
              else 'MOMENTS all_cavs_t0 (CAV-only, t0-1 extrapolated)')
    return dict(settings=settings, desc=desc, cfg=cfg, moments=moments, frames=frames, template=template,
                frames_source=source,
                initial=cw.state_sha256(template), built=built, reasons=reasons, rows=rows_by)


def task_records(prep, name):
    records = cw.deduplicate(prep['built'][name]+prep['built']['negatives'])
    labels = defaultdict(set)
    for r in records:
        labels[_input_key(r)].add(r['kind'])
    return records, sum(1 for kinds in labels.values() if len(kinds) > 1)


def _cap_positives(train, cap):
    """Keep whole positive moments (hash order) until at least `cap` positive records."""
    if cap is None:
        return train
    order = sorted({r['situation_id'] for r in train if r['kind'] == 'positive'},
                   key=lambda sid: hashlib.sha256(f'boost-pre-curve:{sid}'.encode()).hexdigest())
    keep, count = set(), 0
    for sid in order:
        if count >= cap:
            break
        keep.add(sid)
        count += sum(r['kind'] == 'positive' and r['situation_id'] == sid for r in train)
    return [r for r in train if r['kind'] == 'negative' or r['situation_id'] in keep]


def cross_validate(base, prep, name, *, k, positive_cap=None):
    records, conflicts = task_records(prep, name)
    seeds = sorted({r['seed'] for r in records})
    task = dict(records=len(records), positives=sum(r['kind'] == 'positive' for r in records),
                seeds=len(seeds), identical_input_conflicting_label=conflicts, positive_cap=positive_cap, folds=[])
    pooled_test, pooled_train = [], []
    for i, test in enumerate(folds(seeds, k)):
        train = _cap_positives([r for r in records if r['seed'] not in test], positive_cap)
        hold = [r for r in records if r['seed'] in test]
        policy = cw.build_policy(base, prep['cfg'], name)
        if cw.state_sha256(policy) != prep['initial']:
            raise ValueError('fold policy does not start from the common initial layer')
        fold = dict(fold=i, test_seeds=sorted(test), train_records=len(train), test_records=len(hold),
                    train_positives=sum(r['kind'] == 'positive' for r in train),
                    train_positive_moments=len({r['situation_id'] for r in train if r['kind'] == 'positive'}),
                    test_positives=sum(r['kind'] == 'positive' for r in hold))
        if not any(r['kind'] == 'positive' for r in train):
            fold['skipped'] = 'no positive TRAIN records in this fold'
            task['folds'].append(fold)
            continue
        trained = cw.train(policy, train, prep['settings'])
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
    task['heldout_moment_level'] = moment_level(pooled_test)
    return task


def moment_level(values):
    """Moment score = max P(act with move) over its moves; label = any admitted move there."""
    by = defaultdict(list)
    for v in values:
        by[v['sid']].append(v)
    rows = [dict(kind='positive' if any(x['kind'] == 'positive' for x in items) else 'negative',
                 score=max(x['score'] for x in items)) for items in by.values()]
    return dict(moments=len(rows), positive_moments=sum(r['kind'] == 'positive' for r in rows),
                base_rate=(sum(r['kind'] == 'positive' for r in rows)/len(rows)) if rows else None, auc=auc(rows))


def report_header(prep, k):
    cfg = prep['cfg']
    return dict(schema='coopmwm-boost-pre-learncheck-v1', rule=cw.RULE, negative_rule=cw.RULE_NEGATIVE,
                observation=(__doc__.split('Observation source')[1].strip()
                             if isinstance(prep.get('frames_source'), str) else 'EXACT runtime frames (W2 format)'),
                frames_source=prep.get('frames_source'), k_folds=k, training=prep['settings'],
                config={key: v for key, v in cfg.items() if key not in ('run_root',)},
                trained_logit_parameters=[n for n, _ in cw.logit_parameters(prep['template'])],
                base_sha256=cw.state_sha256(prep['template'].layer.base), moments=len(prep['frames']),
                record_counts={n: len(v) for n, v in prep['built'].items()},
                reasons={n: dict(v) for n, v in prep['reasons'].items()}, tasks={})


def run(*, base, out_path, k=5, training=None, rows_root=ROWS_ROOT, map_spec=MAP_SPEC, prep=None):
    prep = prep or prepare(base, training=training, rows_root=rows_root, map_spec=map_spec)
    report = report_header(prep, k)
    for name in ('B', 'R'):
        report['tasks'][name] = cross_validate(base, prep, name, k=k)
    Path(out_path).write_text(json.dumps(report, indent=1, sort_keys=True, default=str)+'\n', encoding='utf-8')
    return report


def curve(*, base, out_path, caps=(10, 20, None), k=5, prep=None, training=None):
    """Held-out AUC vs number of TRAIN positives (whole moments; negatives unchanged)."""
    prep = prep or prepare(base, training=training)
    out = dict(report_header(prep, k), tasks=None, curve=[])
    for cap in caps:
        task = cross_validate(base, prep, 'B', k=k, positive_cap=cap)
        out['curve'].append(dict(positive_cap=cap, heldout=task['heldout_pooled'], train=task['train_pooled'],
                                 moment_level=task['heldout_moment_level'],
                                 train_positives_per_fold=[f['train_positives'] for f in task['folds']],
                                 train_positive_moments_per_fold=[f['train_positive_moments'] for f in task['folds']]))
    Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True, default=str)+'\n', encoding='utf-8')
    return out


def coarse_features(moment, desc):
    """Regime-level context of one moment (cell labels + miners' aggregates + CAV lane counts)."""
    counts = Counter()
    for c in moment['all_cavs_t0']:
        edge, lane = c['edge'], int(c['lane'])
        if edge in EDGE_OFFSET and (edge, lane) in desc.lane_table:
            wm, s = desc.lane_and_s(f'{edge}_{lane}', float(c['x'])-EDGE_OFFSET[edge])
            if -1000. <= s <= 75.44:
                counts[wm] += 1
    return dict(demand=float(moment['demand']), penetration=float(moment['penetration']),
                queue_closing_lane=float(moment.get('closing_lane_slow') or 0.),
                approach_speed=float(moment.get('approach_speed') or 0.),
                n_zone_cavs=float(moment.get('n_zone_cavs') or 0.),
                **{f'cav_lane{lane}_1km': float(counts[lane]) for lane in range(3)})


def coarse_baseline(prep, *, k=5):
    """Seed-grouped logistic regression: does a moment have an admitted B move (moment level),
    and is an evaluated move admitted (row level: + family and move parameters)?"""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    evaluated = defaultdict(list)
    for name, kind in (('B', 'positive'), ('negatives', 'negative')):
        for row in prep['rows'][name]:
            if cw.admission(row) is not None or row.get('family') not in cw.LANE_DROP_JOINT:
                continue
            if kind == 'negative' and not cw.do_not_act(row)[0]:
                continue
            if row.get('situation_id') in prep['moments']:
                evaluated[row['situation_id']].append((row, kind))

    def fit_eval(items, featurize):
        seeds = sorted({it['seed'] for it in items})
        out = []
        for test in folds(seeds, k):
            train = [it for it in items if it['seed'] not in test]
            hold = [it for it in items if it['seed'] in test]
            if len({it['y'] for it in train}) < 2 or not hold:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(np.array([featurize(it) for it in train]), np.array([it['y'] for it in train]))
            prob = model.predict_proba(np.array([featurize(it) for it in hold]))[:, 1]
            out += [dict(kind='positive' if it['y'] else 'negative', score=float(p),
                         correct=bool(p > .5) == bool(it['y'])) for it, p in zip(hold, prob)]
        return metrics(out)

    moments = []
    for sid, items in evaluated.items():
        m = prep['moments'][sid]
        moments.append(dict(seed=int(m['seed']), y=int(any(kind == 'positive' for _, kind in items)),
                            x=coarse_features(m, prep['desc'])))
    names = sorted(moments[0]['x']) if moments else []
    rows = {}
    for sid, items in evaluated.items():
        m = prep['moments'][sid]
        for row, kind in items:
            params = cw.params_of(row)
            rows.setdefault(_key(row), dict(seed=int(m['seed']), y=int(kind == 'positive'),
                x=dict(coarse_features(m, prep['desc']), closing_pair=float(row['family'] == 'closing_pair'),
                       v_w=float(params.get('v_w', 0.)), hold_s=float(params.get('hold_s', 0.)),
                       dmax=float(params.get('dmax', 0.)))))
    row_items = list(rows.values())
    row_names = sorted(row_items[0]['x']) if row_items else []
    return dict(features=names, row_features=row_names, moments=len(moments),
                positive_moments=sum(m['y'] for m in moments), rows=len(row_items),
                moment_level=fit_eval(moments, lambda it: [it['x'][n] for n in names]),
                row_level=fit_eval(row_items, lambda it: [it['x'][n] for n in row_names]),
                note=('demand/penetration are cell LABELS (not runtime inputs); queue/approach speed are the '
                      "miners' aggregates over all vehicles (RSU-like, includes HDVs); lane counts are CAVs only"))


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
    parser.add_argument('--what', default='run', choices=('run', 'curve', 'baseline', 'all'))
    parser.add_argument('--caps', default='10,20,all', help='TRAIN positive caps for --what curve')
    parser.add_argument('--frames', nargs='*', default=(), help='EXACT runtime frames (W2 format) instead of MOMENTS')
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
    prep = prepare(base, training=json.loads(args.training) if args.training else None,
                   frames_paths=tuple(args.frames))
    out = Path(args.out)
    result = {}
    if args.what in ('run', 'all'):
        report = run(base=base, out_path=out, k=args.k, prep=prep)
        for name, task in report['tasks'].items():
            print(name, verdict(task), json.dumps(dict(heldout=task['heldout_pooled'], train=task['train_pooled'],
                within=task['within_admitted_moments_heldout'], moments=task['heldout_moment_level']), default=str),
                flush=True)
        result['run'] = report
    if args.what in ('baseline', 'all'):
        baseline = coarse_baseline(prep, k=args.k)
        out.with_name(out.stem+'_coarse.json').write_text(json.dumps(baseline, indent=1, default=str)+chr(10),
                                                          encoding='utf-8')
        print('coarse', json.dumps(baseline, default=str), flush=True)
        result['baseline'] = baseline
    if args.what in ('curve', 'all'):
        caps = tuple(None if c == 'all' else int(c) for c in args.caps.split(','))
        result['curve'] = curve(base=base, out_path=out.with_name(out.stem+'_curve.json'), k=args.k, prep=prep,
                                caps=caps)
        for point in result['curve']['curve']:
            print('curve', point['positive_cap'], json.dumps(dict(heldout=point['heldout'],
                  moments=point['moment_level'], train_auc=point['train'].get('auc')), default=str), flush=True)
    return result


if __name__ == '__main__':
    main()
