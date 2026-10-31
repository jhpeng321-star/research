"""FIT-only teacher-forced lane diagnostic; no SUMO, training or hidden settings.

Fixed panel: worlds 969400..969431, all five arms, 60s causal wish warmup,
then first 60s from registered t0. Post-motion decision snapshots use recorded
longitudinal motion and PRE-change lanes. Both models get exactly that snapshot.
Time-window tolerances are scoring tolerances, never behavioural parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch

from wms.physics.config import PhysicsConfig
from wms.physics.lc2013 import lane_change
from wms.physics.mobil import mobil_merge_logits

FORBIDDEN = ('hold' + 'out', 'held' + 'out', 'cal' + '_split', 'chunk' + '5',
             'j4' + 'b', 'inc' + '1', 'sealed', 'hetdata' + '_truth',
             'vehicle' + '_params', 'driver' + '_params', 'assignment')
FILES = ('episode.json', 'time_s.npy', 'veh_kin.npy', 'veh_mask.npy', 'is_cav.npy', 'vid.npy')
ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')
MOTIVATIONS = {1: 'strategic', 2: 'cooperative', 3: 'speed_gain', 4: 'keep_right'}


def safe_path(path: Path, root: Path) -> Path:
    if any(term in str(path).lower() for term in FORBIDDEN):
        raise ValueError(f'prohibited path: {path}')
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('observable view escapes permitted root')
    if any(term in str(resolved).lower() for term in FORBIDDEN):
        raise ValueError('prohibited resolved path')
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def match_events(recorded, predicted, tolerance):
    """Max cardinality then min timing cost, per identity/directed transition.

    Sorted 1-D event matching has an order-preserving optimum. Dynamic
    programming avoids nearest-first greedy undercounting crowded events.
    """
    matches = []
    keys = {tuple(e[:3]) for e in recorded}
    for key in keys:
        real = sorted((e for e in recorded if tuple(e[:3]) == key), key=lambda e: e[3])
        pred = sorted((e for e in predicted if tuple(e[:3]) == key), key=lambda e: e[3])
        n, m = len(real), len(pred)
        dp = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]
        move = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                best, action = dp[i - 1][j], 1
                if dp[i][j - 1] > best:
                    best, action = dp[i][j - 1], 2
                delta = abs(real[i - 1][3] - pred[j - 1][3])
                if delta <= tolerance + 1e-8:
                    prev = dp[i - 1][j - 1]
                    candidate = (prev[0] + 1, prev[1] - delta)
                    if candidate > best:
                        best, action = candidate, 3
                dp[i][j], move[i][j] = best, action
        i, j = n, m
        while i and j:
            action = move[i][j]
            if action == 3:
                r, p = real[i - 1], pred[j - 1]
                matches.append((p[3] - r[3], p[4] - r[4]))
                i, j = i - 1, j - 1
            elif action == 2:
                j -= 1
            else:
                i -= 1
    n = len(matches)
    return dict(matched=n, recorded=len(recorded), predicted=len(predicted),
                recall=n / len(recorded) if recorded else None,
                precision=n / len(predicted) if predicted else None,
                timing_abs_sum_s=sum(abs(t) for t, _ in matches),
                timing_signed_sum_s=sum(t for t, _ in matches),
                position_abs_sum_m=sum(abs(s) for _, s in matches))


def compress_proposals(events, dt):
    """Count contiguous positive-decision runs separately from positive steps."""
    last, runs = {}, []
    for event in events:
        key = tuple(event[:3])
        if key not in last or event[3] - last[key] > dt + 1e-6:
            runs.append(event)
        last[key] = event[3]
    return runs


@torch.no_grad()
def episode(path: Path, root: Path, physics: PhysicsConfig, horizon: float, reuse_mobil=None):
    hashes = {name: sha(safe_path(path / name, root)) for name in FILES}
    if reuse_mobil is not None and reuse_mobil['input_sha256'] != hashes:
        raise ValueError('MOBIL reuse requires exact observable input hashes')
    meta = json.loads((path / 'episode.json').read_text())
    if not 969400 <= int(meta['seed']) <= 969431:
        raise ValueError('not a permitted FIT world')
    if meta['arm'] not in ARMS:
        raise ValueError('unexpected arm')
    a = {name: np.load(path / (name + '.npy'), allow_pickle=False, mmap_mode='r')
         for name in ('time_s', 'veh_kin', 'veh_mask', 'is_cav', 'vid')}
    times, kin, mask, cav, ids = (a[name] for name in
                                ('time_s', 'veh_kin', 'veh_mask', 'is_cav', 'vid'))
    if kin.shape[:2] != mask.shape or ids.shape != mask.shape or cav.shape != mask.shape:
        raise ValueError('array shape mismatch')
    t0 = float(meta['branch']['t0_s'])
    if times[0] > t0 - 60 or times[-1] < t0 + horizon:
        raise ValueError('missing registered prehistory or diagnostic horizon')
    if not np.allclose(np.diff(times), physics.dt_s, rtol=0, atol=1e-8):
        raise ValueError('unexpected time discretization')
    memory = None
    actual, proposals = [], {'lc2013': [], 'mobil': []}
    motivated = {name: [] for name in MOTIVATIONS.values()}
    exposure = Counter()
    wishes, blocked = Counter(), Counter()
    for t in range(len(times) - 1):
        if times[t] < t0 - 60 or times[t] >= t0 + horizon:
            continue
        same = (mask[t].astype(bool) & mask[t + 1].astype(bool) & (ids[t] == ids[t + 1]))
        normal0 = kin[t, :, 1:4].sum(-1) > 0
        normal1 = kin[t + 1, :, 1:4].sum(-1) > 0
        # Only known, normal-road bodies appear in the LC decision geometry.
        present = same & normal0 & normal1
        eligible = present & ~cav[t].astype(bool)
        moved = np.array(kin[t], copy=True)
        moved[:, [0, 4, 5]] = kin[t + 1, :, [0, 4, 5]].T
        k = torch.from_numpy(moved).unsqueeze(0)
        m = torch.from_numpy(present).unsqueeze(0)
        identity = torch.from_numpy(np.array(ids[t], copy=True)).unsqueeze(0)
        d = torch.from_numpy(eligible).unsqueeze(0)
        memory = lane_change(k, m, physics, memory=memory, identity=identity, decision_mask=d)
        if times[t] < t0:
            continue
        choices = dict(lc2013=memory['next_lane'][0].numpy())
        if reuse_mobil is None:
            choices['mobil'] = mobil_merge_logits(k, m, physics)[0].argmax(-1).numpy()
        source = kin[t, :, 1:4].argmax(-1)
        destination = kin[t + 1, :, 1:4].argmax(-1)
        for j in np.flatnonzero(eligible):
            direction = {0: 'return', 1: 'overtake', 2: 'merge'}[int(source[j])]
            exposure[direction] += 1
            if memory['lc2013_wish'][0, j]:
                wishes[direction] += 1
                if not memory['lc2013_safe'][0, j]:
                    blocked[direction] += 1
            def event(dest):
                return [int(ids[t, j]), int(source[j]), int(dest),
                        float(times[t + 1]), float(kin[t + 1, j, 0])]
            if destination[j] != source[j]:
                actual.append(event(destination[j]))
            for name, choice in choices.items():
                if choice[j] != source[j]:
                    proposals[name].append(event(choice[j]))
                    if name == 'lc2013':
                        motivated[MOTIVATIONS[int(memory['lc2013_reason'][0, j])]].append(event(choice[j]))
    scores = {}
    for name, events in proposals.items():
        if name == 'mobil' and reuse_mobil is not None:
            if reuse_mobil['recorded'] != len(actual) or reuse_mobil['exposure'] != dict(exposure):
                raise ValueError('MOBIL reuse cohort mismatch')
            scores[name] = reuse_mobil['scores']['mobil']
            continue
        runs = compress_proposals(events, physics.dt_s)
        scores[name] = dict(proposals=len(events), positive_runs=len(runs),
            exact=match_events(actual, events, 0),
            window_2s=match_events(actual, events, 2),
            run_window_2s=match_events(actual, runs, 2),
            by_transition={key: match_events([e for e in actual if e[1] == lane],
                [e for e in events if e[1] == lane], 0)
                for lane, key in ((0, 'return'), (1, 'overtake'), (2, 'merge'))})
        if name == 'lc2013':
            scores[name]['by_motivation'] = {
                reason: dict(exact=match_events(actual, ev, 0),
                             run_window_2s=match_events(actual, compress_proposals(ev, physics.dt_s), 2))
                for reason, ev in motivated.items()}
    return dict(episode=meta['episode'], seed=int(meta['seed']), arm=meta['arm'],
                input_sha256=hashes, recorded=len(actual), exposure=dict(exposure),
                wishes=dict(wishes), blocked=dict(blocked), scores=scores)


def aggregate(rows):
    result = {}
    for model in ('lc2013', 'mobil'):
        result[model] = {}
        for key in ('exact', 'window_2s', 'run_window_2s'):
            cells = [r['scores'][model][key] for r in rows]
            sums = {name: sum(c[name] for c in cells) for name in
                    ('matched', 'recorded', 'predicted', 'timing_abs_sum_s',
                     'timing_signed_sum_s', 'position_abs_sum_m')}
            n = sums['matched']
            sums.update(recall=sums['matched'] / sums['recorded'] if sums['recorded'] else None,
                        precision=sums['matched'] / sums['predicted'] if sums['predicted'] else None,
                        timing_mae_s=sums['timing_abs_sum_s'] / n if n else None,
                        position_mae_m=sums['position_abs_sum_m'] / n if n else None)
            result[model][key] = sums
        result[model]['positive_runs'] = sum(r['scores'][model]['positive_runs'] for r in rows)
        result[model]['by_transition'] = {}
        for transition in ('merge', 'overtake', 'return'):
            cells = [r['scores'][model]['by_transition'][transition] for r in rows]
            sums = {name: sum(c[name] for c in cells) for name in ('matched', 'recorded', 'predicted')}
            sums['recall'] = sums['matched'] / sums['recorded'] if sums['recorded'] else None
            sums['precision'] = sums['matched'] / sums['predicted'] if sums['predicted'] else None
            result[model]['by_transition'][transition] = sums
        if model == 'lc2013' and all('by_motivation' in r['scores'][model] for r in rows):
            result[model]['by_motivation'] = {}
            for motivation in MOTIVATIONS.values():
                result[model]['by_motivation'][motivation] = {}
                for metric in ('exact', 'run_window_2s'):
                    cells = [r['scores'][model]['by_motivation'][motivation][metric] for r in rows]
                    sums = {key: sum(c[key] for c in cells) for key in ('matched', 'recorded', 'predicted')}
                    sums['precision'] = sums['matched'] / sums['predicted'] if sums['predicted'] else None
                    # Recorded tapes have no motivation labels. This is the
                    # contribution to ALL-event recall, not class-conditional recall.
                    sums['recall_contribution'] = sums['matched'] / sums['recorded'] if sums['recorded'] else None
                    result[model]['by_motivation'][motivation][metric] = sums
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--views', type=Path, default=Path('/home/ubuntu/wmdata20261031/views'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--horizon-s', type=float, default=60)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--reuse-mobil-from', type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.getloadavg()[0] > 20:
        raise RuntimeError('CPU load admission failed')
    if not args.out.parent.is_dir() or args.out.exists():
        raise ValueError('output must be NEW with existing parent')
    args.out.mkdir()
    physics = PhysicsConfig(sumo_fidelity=True, open_road=True)
    baseline = None
    if args.reuse_mobil_from is not None:
        baseline = json.loads((args.reuse_mobil_from / 'SUMMARY.json').read_text())
        if (baseline['status'] != 'COMPLETE' or baseline['episodes'] != 160
                or baseline['physics'] != asdict(physics) or baseline['horizon_s'] != args.horizon_s):
            raise ValueError('MOBIL reuse panel/settings mismatch')
    paths = []
    # Never glob the full world range or open a global manifest/split.
    for world in range(969400, 969432):
        for path in sorted((args.views / 'episodes').glob(f'w{world}m*')):
            if re.fullmatch(rf'w{world}m[0-9]+p[0-9]+t300__(native|ctrl0|mappo|pi|circles)', path.name):
                paths.append(safe_path(path, args.views))
    if len(paths) != 160:
        raise ValueError(f'expected 160 FIT episodes, found {len(paths)}')
    if args.smoke:
        paths = [p for p in paths if p.name.endswith('__native')][:1]
    start = time.time()
    rows = []
    for path in paths:
        reuse = (json.loads((args.reuse_mobil_from / (path.name + '.json')).read_text())
                 if baseline is not None else None)
        row = episode(path, args.views, physics, args.horizon_s, reuse)
        rows.append(row)
        (args.out / (path.name + '.json')).write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(dict(completed=len(rows), total=len(paths), episode=path.name,
                              elapsed_s=time.time()-start)), flush=True)
    report = dict(status='COMPLETE', episodes=len(rows), worlds=len({r['seed'] for r in rows}),
                  horizon_s=args.horizon_s, prehistory_s=60, settings='population only',
                  snapshot='recorded post-motion s/v/a with pre-change lanes; teacher forced',
                  cohort='HDV, same identity, normal road in both adjacent frames',
                  motivation_contract='predicted reason; recall contribution uses all recorded events; no observed motivation labels',
                  mobil_reuse=(str(args.reuse_mobil_from) if baseline is not None else None),
                  physics=asdict(physics), aggregate=aggregate(rows),
                  by_arm={arm: aggregate([r for r in rows if r['arm'] == arm]) for arm in ARMS
                          if any(r['arm'] == arm for r in rows)},
                  elapsed_s=time.time()-start,
                  source_sha256={str(p): sha(p) for p in (Path(__file__),
                    Path(__file__).parents[2] / 'src/wms/physics/lc2013.py')})
    (args.out / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['aggregate']), flush=True)


if __name__ == '__main__':
    main()
