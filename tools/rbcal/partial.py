"""Cache-only recovery of owner-stopped dev futures; no fitting or dispatch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from tools.rbcal.outcomes import DEV, development_truth
from tools.rbcal.summarize import compact
from tools.rbeval import core, score
from tools.wmdecide.rbeval_adapter import observations, safe_path, sha256
from tools.wmdecide.report import reliability_svg, summarize_rows, write_csv, write_json


def inventory(source):
    """Admit exact dev filenames only, reject duplicate-device/shard copies."""
    files, counts = [], {}
    for world in DEV:
        counts[world] = 0
        for future in range(8):
            name = f'w{world}_k{future}.json'
            matches = [safe_path(source/f'shard{i}'/name) for i in range(3)]
            matches = [p for p in matches if p.is_file()]
            if len(matches) > 1:
                raise ValueError(f'duplicate development future: {name}')
            if matches:
                path = matches[0]
                counts[world] += 1
                files.append(dict(world=world, future=future,
                                  source=str(path), sha256=sha256(path)))
    complete = [world for world in DEV if counts[world] == 8]
    if not complete:
        raise ValueError('no complete development K8 worlds')
    return dict(counts=counts, complete_worlds=complete, tasks=files,
                retained_tasks=len(files), scored_tasks=8*len(complete),
                excluded_incomplete_tasks=len(files)-8*len(complete))


def matched_observations(data, worlds):
    worlds = set(worlds)
    if not worlds or not worlds <= set(DEV):
        raise ValueError('nonempty development-only world set required')
    if any(r['seed'] not in DEV for rows in data.values() for r in rows):
        raise ValueError('non-development observation refused')
    result = {name: [r for r in rows if r['seed'] in worlds] for name, rows in data.items()}
    if any({r['seed'] for r in rows} != worlds for rows in result.values()):
        raise ValueError('incomplete matched observation worlds')
    return result


def emit(data, out, audit, title):
    out.mkdir()
    summaries, diagrams = summarize_rows(data)
    # These development cells are unreplicated. Suppress all bootstrap CIs.
    for row in summaries:
        row.update(low=None, high=None, uncertainty_available=False)
    for row in diagrams:
        row.update(observed_low=None, observed_high=None, uncertainty_available=False)
    for name, value in [('observations.json', data), ('summary.json', summaries), ('audit.json', audit)]:
        write_json(out/name, value)
    write_csv(out/'summary.csv', summaries)
    write_csv(out/'reliability.csv', diagrams)
    for target in ('preference', 'selected_preference'):
        if any(r['target'] == target and r['count'] for r in diagrams):
            with (out/f'reliability_{target}.svg').open('x', encoding='utf-8') as stream:
                stream.write(reliability_svg(diagrams, target, title))
    rows = [r for r in summaries if r['scope'] == 'pooled']
    result = compact(rows)
    for h, record in result.items():
        for target in ('delay', 'paired_delay'):
            coverage = [r for r in data['coverage'] if r['horizon_s'] == int(h) and r['target'] == target]
            values = [r['covered'] for r in coverage if r['interval'] == 'range']
            record[target]['range_coverage'] = float(np.mean(values))
            distributions = [r for r in data['distributions'] if r['horizon_s'] == int(h) and r['target'] == target]
            record[target]['mean_bias_vs'] = float(np.mean([r['expected_delay']-r['outcome'] for r in distributions]))
            record[target]['mean_predictive_sd_vs'] = float(np.mean([r['predictive_sd'] for r in distributions]))
    return result


def run(source, baseline, snapshot, out):
    for path in (source, baseline, snapshot, out):
        safe_path(path)
    if snapshot.exists() or out.exists() or not snapshot.parent.is_dir() or not out.parent.is_dir():
        raise ValueError('exclusive new snapshot/output with existing parents required')
    admitted = inventory(source)
    worlds = admitted['complete_worlds']
    snapshot.mkdir()
    for item in admitted['tasks']:
        if item['world'] not in worlds:
            continue
        path = Path(item['source'])
        target = snapshot/path.parent.name
        target.mkdir(exist_ok=True)
        if not (target/'IDENTITY.json').exists():
            shutil.copyfile(safe_path(path.parent/'IDENTITY.json'), target/'IDENTITY.json')
        shutil.copyfile(path, target/path.name)
        if sha256(target/path.name) != item['sha256']:
            raise ValueError('snapshot hash drift')
    write_json(snapshot/'ADMISSION.json', admitted)
    real = development_truth(core.HERE/'real_observables.json')
    pred, audit = score.read_predictions([snapshot], real, partial=True)
    if audit['complete_worlds'] != worlds or audit['tasks'] != len(worlds)*8:
        raise ValueError('complete-ensemble admission differs')
    baseline_path = safe_path(baseline/'observations.json')
    old = matched_observations(json.loads(baseline_path.read_bytes()), worlds)
    current = observations(pred, real, core.ARMS, core.HS, core.REGIONS)
    if {(r['seed'], r['action'], r['reference'], r['horizon_s'], r['outcome']) for r in old['distributions']} != {
            (r['seed'], r['action'], r['reference'], r['horizon_s'], r['outcome']) for r in current['distributions']}:
        raise ValueError('baseline/candidate factual targets differ')
    out.mkdir()
    common = dict(scored_worlds=worlds, outcome_qualification=False,
                  panel='owner-stopped partial development; availability-selected',
                  full_world_confidence_intervals_available=False)
    physical = emit(current, out/'physical', dict(audit, **common), 'Physical v1: partial dev6')
    baseline_audit = json.loads(safe_path(baseline/'audit.json').read_bytes())
    old_scores = emit(old, out/'baseline_matched', dict(baseline_audit, **common,
        saved_observations_sha256=sha256(baseline_path)), 'FULL-v4: matched partial dev6')
    result = dict(schema='rb2cal.partial_development.v1', **common,
        created_at=datetime.now(timezone.utc).isoformat(), admission=admitted,
        physical=physical, baseline=old_scores, source=audit['model'],
        comparison_caveat='CPU candidate versus GPU baseline; not a device-controlled ablation',
        reserved_tasks_opened=0, parameter_fits=0, new_inference=0,
        probability_artifact_refitted=False, code_sha256=sha256(Path(__file__)))
    write_json(out/'RESULTS.json', result)
    print(json.dumps({k: result[k] for k in ('scored_worlds', 'admission', 'physical', 'baseline')}, indent=1))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    run(args.source, args.baseline, args.snapshot, args.out)


if __name__ == '__main__':
    main()
