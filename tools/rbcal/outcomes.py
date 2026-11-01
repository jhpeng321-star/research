"""Development-only saved-future calibration diagnostics using S-RB-DECIDE.

The mean-preserving outcome-scale experiment is a diagnostic, NOT a physical
candidate. Only new physical rollouts can qualify the calibrated latent sampler.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tools.rbcal.fit import bounded
from tools.rbeval import core, score
from tools.wmdecide.metrics import crps
from tools.wmdecide.rbeval_adapter import observations, safe_path, sha256
from tools.wmdecide.report import summarize_rows, write_json, write_csv, reliability_svg

DEV = tuple(range(969420, 969432))


def development_truth(path):
    """Read sorted factual FIT episodes, STOP at first reserved episode header.

    Refuse format/order drift rather than falling back to loading the mixed
    panel. No reserved episode outcome is decoded or evaluated.
    """
    records, block = [], None
    with safe_path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('   "seed":'):
                seed = int(line.split(':')[1].strip().rstrip(','))
                if seed >= 969432:
                    if seed != 969432 or len(records) != 160:
                        raise ValueError('factual development-first contract differs')
                    break
            if line.rstrip() == '  {':
                if block is not None:
                    raise ValueError('nested episode format drift')
                block = [line]
            elif block is not None:
                block.append(line)
                if line.rstrip() in ('  },', '  }'):
                    row = json.loads(''.join(block).rstrip().rstrip(','))
                    if row['seed'] not in range(969400, 969432):
                        raise ValueError('non-FIT truth refused')
                    records.append(row)
                    block = None
    if {(r['seed'], r['arm']) for r in records} != {
            (world, arm) for world in range(969400, 969432) for arm in core.ARMS}:
        raise ValueError('incomplete factual FIT rows')
    return records


def snapshot_dev(source, destination, worlds=DEV):
    """Exact filename admission BEFORE opening predictions (no reserved glob)."""
    if tuple(worlds) != DEV:
        raise ValueError('development12 only')
    if not destination.parent.is_dir() or destination.exists():
        raise ValueError('exclusive new snapshot under existing parent required')
    destination.mkdir()
    manifest = []
    for world in worlds:
        for future in range(8):
            name = f'w{world}_k{future}.json'
            paths = [safe_path(source/f'shard{i}'/name) for i in range(3)]
            paths = [p for p in paths if p.is_file()]
            if len(paths) != 1:
                raise ValueError(f'exactly one registered source required: {name}')
            path = paths[0]
            target = destination/path.parent.name
            target.mkdir(exist_ok=True)
            identity = target/'IDENTITY.json'
            if not identity.exists():
                shutil.copyfile(path.parent/'IDENTITY.json', identity)
            shutil.copyfile(path, target/name)
            manifest.append(dict(source=str(path), sha256=sha256(path), world=world, future=future))
    write_json(destination/'ADMISSION.json', dict(worlds=list(worlds), tasks=manifest,
        reserved_prediction_tasks_opened=0, factual_reader='stops at first reserved seed header'))


def scale_predictions(pred, scales):
    result = copy.deepcopy(pred)
    for (world, _), rows in result.items():
        for h in core.HS:
            key = str(h)
            x = np.array([r['clocks'][key]['team_delay_vs'] for r in rows])
            scale = scales[world][key]
            transformed = x.mean()+scale*(x-x.mean())
            for row, value in zip(rows, transformed):
                row['clocks'][key]['team_delay_vs'] = float(value)
    return result


def fit_outcome_scale(pred, real, horizon, excluded=None):
    lookup = {(r['seed'], r['arm']):r for r in real}
    cases = []
    from itertools import combinations
    for world in DEV:
        if world == excluded:
            continue
        for arm, reference in [(a, None) for a in core.ARMS]+list(combinations(core.ARMS, 2)):
            x = np.array([r['clocks'][str(horizon)]['team_delay_vs'] for r in pred[world, arm]])
            y = lookup[world, arm]['clocks'][str(horizon)]['team_delay_vs']
            if reference is not None:
                x -= np.array([r['clocks'][str(horizon)]['team_delay_vs'] for r in pred[world, reference]])
                y -= lookup[world, reference]['clocks'][str(horizon)]['team_delay_vs']
            cases.append((x, y, reference is not None))
    norms = {paired:np.mean([crps(x, y) for x, y, p in cases if p == paired]) for paired in (False, True)}

    def objective(coordinate):
        if coordinate == 1:
            return np.inf
        scale = coordinate/(1-coordinate)
        return float(sum(np.mean([crps(x.mean()+scale*(x-x.mean()), y)/norms[p]
                        for x, y, p in cases if p == paired]) for paired in (False, True)))

    coordinate = bounded(objective)
    return float(coordinate/(1-coordinate))


def emit(pred, real, out, title, audit):
    out.mkdir()
    data = observations(pred, real, core.ARMS, core.HS, core.REGIONS)
    summaries, diagrams = summarize_rows(data)
    # Development cells lack replication. Strip ALL bootstrap CIs, including
    # reliability-bin intervals: don't resurrect historical conditional CIs.
    for row in summaries:
        row.update(low=None, high=None, uncertainty_available=False)
    for row in diagrams:
        row.update(observed_low=None, observed_high=None, uncertainty_available=False)
    for name, value in [('observations.json', data), ('summary.json', summaries), ('audit.json', audit)]:
        write_json(out/name, value)
    write_csv(out/'summary.csv', summaries)
    write_csv(out/'reliability.csv', diagrams)
    for target in ('preference', 'selected_preference'):
        with (out/f'reliability_{target}.svg').open('x', encoding='utf-8') as f:
            f.write(reliability_svg(diagrams, target, title))
    pooled = [r for r in summaries if r['scope'] == 'pooled' and r['metric'] in (
        'crps', 'brier', 'covered', 'probability', 'outcome', 'switched')]
    return pooled


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--physical-only', action='store_true')
    args = p.parse_args()
    if not args.out.parent.is_dir() or args.out.exists():
        raise ValueError('new outcome directory required')
    args.out.mkdir()
    real = development_truth(core.HERE/'real_observables.json')
    snapshot_dev(args.source, args.snapshot)
    pred, audit = score.read_predictions([args.snapshot], real, partial=True)
    if audit['complete_worlds'] != list(DEV):
        raise ValueError('complete dev12 K8 required')
    baseline = emit(pred, real, args.out/'physical', 'Physical dev12', audit)
    result = dict(physical=baseline, worlds=list(DEV), reserved_tasks_opened=0,
                  factual_read='stopped at reserved header', outcome_qualification=False)
    if not args.physical_only:
        full = {str(h):fit_outcome_scale(pred, real, h) for h in core.HS}
        crossfit = {world:{str(h):fit_outcome_scale(pred, real, h, excluded=world) for h in core.HS}
                    for world in DEV}
        scaled = scale_predictions(pred, crossfit)
        transformed = emit(scaled, real, args.out/'diagnostic_oof_scale',
            'DIAGNOSTIC outcome-scale OOF dev12', dict(audit,
             qualification='NOT simulated futures; mean-preserving scale limitation diagnostic'))
        result.update(full_fit_scale=full, whole_world_excluded_scales=crossfit,
                      diagnostic_oof_scale=transformed)
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(args.out/'RESULTS.json', result)
    print(json.dumps(dict(out=str(args.out), scales=result.get('full_fit_scale'), worlds=len(DEV))))


if __name__ == '__main__':
    main()
