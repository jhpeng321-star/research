"""FIT-only paired frozen-LC continuation diagnostic, no fitting/SUMO/GPU.

One source-default LC scene and K persistent sampled-profile scenes share the
same pooled motion-fitted IDM point and recorded post-motion geometry. The
posterior is inferred ONLY from240..300; score300..360 and (optionally)..600.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from wms.control.fitted_hidden import LC_ORDER
from wms.control.hidden_state import seed_of
from wms.physics.config import PhysicsConfig
from wms.physics.lane_continuation import LaneContinuation, postmotion_snapshot
from wms.physics.lc2013 import lane_change
from .validate import ARMS, FILES, MOTIVATIONS, safe_path, sha, match_events, compress_proposals


def metrics(actual, predicted, motivations, dt):
    return dict(exact=match_events(actual, predicted, 0),
        window_2s=match_events(actual, predicted, 2),
        run_window_2s=match_events(actual, compress_proposals(predicted, dt), 2),
        positive_runs=len(compress_proposals(predicted, dt)),
        by_motivation={name:dict(exact=match_events(actual, motivations[name], 0),
            run_window_2s=match_events(actual, compress_proposals(motivations[name], dt), 2))
            for name in MOTIVATIONS.values()})


@torch.no_grad()
def episode(path, root, continuation, *, samples=8, horizon=60.):
    hashes = {name:sha(safe_path(path/name, root)) for name in FILES}
    meta = json.loads((path/'episode.json').read_text())
    world, arm = int(meta['seed']), meta['arm']
    if not 969400 <= world <= 969431 or arm not in ARMS:
        raise ValueError('FIT-only lane diagnostic')
    a = {name:np.load(path/(name+'.npy'), allow_pickle=False, mmap_mode='r')
         for name in ('time_s', 'veh_kin', 'veh_mask', 'vid', 'is_cav')}
    t, kin, masks, ids, cav = (a[name] for name in ('time_s', 'veh_kin', 'veh_mask', 'vid', 'is_cav'))
    if t[-1] < 300.+horizon or meta['branch']['t0_s'] != 300.:
        raise ValueError('complete diagnostic horizon required')
    p = PhysicsConfig(sumo_fidelity=True, open_road=True)
    pre = {name:np.array(value[t <= 300.], copy=True) for name, value in a.items()}
    records, receipt = continuation.condition(pre, world=world, t0_s=300., physics=p)
    batch, slots = samples+1, kin.shape[1]
    profiles, weights = continuation.support, continuation.weights
    drawn = {}

    def settings(i):
        params = {name:torch.full((batch, slots), value, dtype=torch.float64)
                  for name, value in continuation.population['idm_parameters'].items()}
        params.update({name:torch.ones((batch, slots), dtype=torch.float64) for name in LC_ORDER})
        for j in np.flatnonzero(masks[i].astype(bool) & ~cav[i].astype(bool)):
            vid = int(ids[i, j])
            if vid not in drawn:
                record = records.get((world, vid))
                support = profiles if record is None else profiles[record['support_indices']]
                ww = weights if record is None else torch.as_tensor(record['weights'], dtype=torch.float64)
                drawn[vid] = torch.stack([support[int(torch.multinomial(ww, 1,
                    generator=torch.Generator().manual_seed(seed_of(190, world, s, 'fit-lc', vid))))]
                    for s in range(samples)])
            for c, name in enumerate(LC_ORDER):
                params[name][1:, j] = drawn[vid][:, c]
        return params

    actual, events, motivated = [], [[] for _ in range(batch)], [defaultdict(list) for _ in range(batch)]
    exposure, cells, wishes, memory = Counter(), [defaultdict(Counter) for _ in range(batch)], [Counter() for _ in range(batch)], None
    outputs = {}
    brier = [Counter() for _ in range(2)]
    horizons = sorted(set((60., float(horizon))))
    for i in np.flatnonzero((t[:-1] >= 240.) & (t[1:] <= 300.+horizon)):
        present = (masks[i].astype(bool) & masks[i+1].astype(bool) & (ids[i] == ids[i+1])
            & (kin[i, :, 1:4].sum(-1) == 1) & (kin[i+1, :, 1:4].sum(-1) == 1))
        eligible = present & ~cav[i].astype(bool) & ~cav[i+1].astype(bool)
        moved = postmotion_snapshot(kin[i], kin[i+1], p)
        params = settings(i)
        memory = lane_change(moved.expand(batch, slots, 7), torch.as_tensor(present).expand(batch, slots), p,
            memory=memory, identity=torch.as_tensor(np.array(ids[i], copy=True)).expand(batch, slots),
            decision_mask=torch.as_tensor(eligible).expand(batch, slots), driver_params=params)
        if t[i] < 300.:
            continue
        source, dest = kin[i, :, 1:4].argmax(-1), kin[i+1, :, 1:4].argmax(-1)
        proposed, reason = memory['next_lane'].numpy(), memory['lc2013_reason'].numpy()
        js = np.flatnonzero(eligible)
        labels = np.where(dest[js] == source[js], 0, dest[js]+1)
        predicted_labels = np.where(proposed[:, js] == source[None, js], 0, proposed[:, js]+1)
        for b, probability in enumerate((np.eye(3)[predicted_labels[0]],
                np.eye(3)[predicted_labels[1:]].mean(0))):
            brier[b]['n'] += len(js)
            brier[b]['categorical_brier_sum'] += float(np.square(probability-np.eye(3)[labels]).sum())
            brier[b]['event_brier_sum'] += float(np.square(1-probability[:, 0]-(labels!=0)).sum())
        for j in np.flatnonzero(eligible):
            lane, s, at = int(source[j]), float(moved[j, 0]), float(t[i+1])
            segment = 'ramp' if lane == 2 else 'upstream' if s < 0 else 'merge' if s <= p.merge_zone_s1_m else 'downstream'
            cell = f'{segment}/{lane}'
            exposure[cell] += 1
            def event(destination):
                return [int(ids[i, j]), lane, int(destination), at, float(kin[i+1, j, 0])]
            observed = dest[j] != lane
            if observed:
                actual.append(event(dest[j]))
            for b in range(batch):
                c = cells[b][cell]
                c['recorded'] += int(observed)
                c['exposure_veh_s'] += p.dt_s
                if memory['lc2013_wish'][b, j]:
                    wishes[b][MOTIVATIONS[int(reason[b, j])] + ('/safe' if memory['lc2013_safe'][b, j] else '/blocked')] += 1
                if proposed[b, j] != lane:
                    e = event(proposed[b, j])
                    # Coordinates are scoring only, not behavioral thresholds.
                    from .snapshot import proposal_position
                    e[4] = float(proposal_position(s, lane, int(proposed[b, j]), p))
                    events[b].append(e)
                    motivated[b][MOTIVATIONS[int(reason[b, j])]].append(e)
                    c['predicted_steps'] += 1
                    c['matched_steps'] += int(proposed[b, j] == dest[j])
        elapsed = float(t[i+1]-300.)
        if elapsed in horizons:
            def cell_counts(b, ten_second=False):
                counts = {name:dict(c) for name,c in cells[b].items()}
                for e in compress_proposals(events[b], p.dt_s):
                    lane, pos = e[1], e[4]
                    region = 'ramp' if lane==2 else 'upstream' if pos<0 else 'merge' if pos<=p.merge_zone_s1_m else 'downstream'
                    cell = f'{region}/{lane}'
                    counts.setdefault(cell, {}).setdefault('predicted_runs', 0)
                    counts[cell]['predicted_runs'] += 1
                return counts
            outputs[str(int(elapsed))] = dict(recorded=len(actual),
                source_default=metrics(actual, events[0], motivated[0], p.dt_s),
                fitted_samples=[metrics(actual, events[b], motivated[b], p.dt_s) for b in range(1, batch)],
                lane_segment_source_default=cell_counts(0),
                lane_segment_fitted_samples=[cell_counts(b) for b in range(1, batch)],
                probability_scores=[dict(c) for c in brier],
                recorded_events=actual.copy(),
                source_default_events=events[0].copy(),
                fitted_sample_events=[ev.copy() for ev in events[1:]],
                source_default_wishes=dict(wishes[0]), fitted_sample_wishes=[dict(wishes[b]) for b in range(1, batch)])
    return dict(seed=world, arm=arm, episode=meta['episode'], input_sha256=hashes,
        conditioning=receipt, horizons=outputs)


def summarize(rows, samples):
    result = {}
    for horizon in rows[0]['horizons']:
        models = {}
        for model in ('source_default', 'fitted_samples'):
            reports = [r['horizons'][horizon][model] for r in rows]
            if model == 'fitted_samples':
                reports = [s for values in reports for s in values]
            divisor = samples if model == 'fitted_samples' else 1
            out = {}
            def pooled(scores):
                names = ('matched', 'recorded', 'predicted', 'timing_abs_sum_s', 'timing_signed_sum_s', 'position_abs_sum_m')
                sums = {name:sum(s[name] for s in scores)/divisor for name in names}
                sums.update(precision=sums['matched']/sums['predicted'] if sums['predicted'] else None,
                    recall=sums['matched']/sums['recorded'] if sums['recorded'] else None,
                    timing_mae_s=sums['timing_abs_sum_s']/sums['matched'] if sums['matched'] else None)
                return sums
            for key in ('exact', 'window_2s', 'run_window_2s'):
                out[key] = pooled([r[key] for r in reports])
            out['positive_runs'] = sum(r['positive_runs'] for r in reports)/divisor
            out['by_motivation'] = {name:{key:pooled([r['by_motivation'][name][key] for r in reports])
                for key in ('exact', 'run_window_2s')} for name in MOTIVATIONS.values()}
            cell_reports = [r['horizons'][horizon]['lane_segment_' + model] for r in rows]
            if model == 'fitted_samples':
                cell_reports = [s for values in cell_reports for s in values]
            totals = defaultdict(Counter)
            for report in cell_reports:
                for cell, values in report.items():
                    totals[cell].update(values)
            for cell, values in totals.items():
                values = {name:value/divisor for name,value in values.items()}
                den = values['exposure_veh_s']
                values.update(recorded_rate_per_veh_h=3600*values['recorded']/den if den else None,
                    proposed_step_rate_per_veh_h=3600*values.get('predicted_steps', 0)/den if den else None)
                totals[cell] = values
            out['lane_segment'] = dict(totals)
            probability = Counter()
            for row in rows:
                probability.update(row['horizons'][horizon]['probability_scores'][int(model=='fitted_samples')])
            out['probability_scores'] = dict(probability,
                event_brier=probability['event_brier_sum']/probability['n'],
                categorical_brier=probability['categorical_brier_sum']/probability['n'])
            models[model] = out
        result[horizon] = models
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--views', type=Path, default=Path('/home/ubuntu/wmdata20261031/views'))
    ap.add_argument('--continuation', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--worlds', type=int, nargs='+', default=list(range(969400, 969432)))
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=['native'])
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--horizon', type=int, choices=(60, 300), default=60)
    args = ap.parse_args()
    if not set(args.worlds) <= set(range(969400, 969432)) or len(args.worlds) != len(set(args.worlds)):
        raise ValueError('unique development worlds required')
    if args.samples < 1 or args.out.exists() or not args.out.parent.is_dir():
        raise ValueError('positive sample count and new output required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.getloadavg()[0] >= 20:
        raise ValueError('CPU load admission failed')
    args.out.mkdir()
    continuation = LaneContinuation.from_directory(args.continuation)
    begin, rows = time.time(), []
    for world in args.worlds:
        for arm in args.arms:
            paths = [p for p in (args.views/'episodes').glob(f'w{world}m*__{arm}')
                if re.fullmatch(rf'w{world}m[0-9]+p[0-9]+t300__{arm}', p.name)]
            if len(paths) != 1:
                raise ValueError('unique FIT episode required')
            row = episode(paths[0], args.views, continuation, samples=args.samples, horizon=args.horizon)
            rows.append(row)
            (args.out/(paths[0].name+'.json')).write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
            print(json.dumps(dict(completed=len(rows), episode=paths[0].name, elapsed_s=time.time()-begin)), flush=True)
    report = dict(status='COMPLETE', worlds=args.worlds, arms=args.arms, samples=args.samples,
        settings='same frozen pooled IDM; source-default versus persistent sampled raw-fitted LC',
        diagnosis='teacher-forced conditional proposals, not free-running lane rates',
        motivation_contract='predicted reason; recall is contribution to all-event recall, no true reason labels',
        continuation=continuation.provenance, summary=summarize(rows, args.samples), elapsed_s=time.time()-begin,
        source_sha256={str(p):sha(p) for p in [Path(__file__),
            *sorted((Path(__file__).resolve().parents[2]/'src/wms/physics').glob('lc2013*.py')),
            Path(__file__).resolve().parents[2]/'src/wms/physics/lane_continuation.py']})
    (args.out/'SUMMARY.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
