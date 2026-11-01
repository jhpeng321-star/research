"""Export only causal recorded starts, sanitized receipts and pre-t0 speed bins.

No future trajectory/cumulative target is opened. The named producer result JSON
container is parsed only for start hashes and initial-state identity; its outcome
fields are neither copied nor supplied to the forecaster.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')
ARMS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')


def safe(path):
    path = Path(path)
    if any(s in str(path.resolve()).lower() for s in BAD):
        raise ValueError('prohibited observable input path')
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8'))


def save(path, data):
    with safe(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=1, allow_nan=False)


def prefix_bins(history):
    import numpy as np
    kin, mask = history['kin'], history['mask'].astype(bool)
    if kin.ndim != 3 or kin.shape[0] != 241 or kin.shape[-1] != 7 or mask.shape != kin.shape[:2]:
        raise ValueError('registered241-frame causal history required')
    kin, mask = kin[120:240], mask[120:240]  # [240,300), excluding t0 duplication.
    valid = mask & (kin[..., 1:4].sum(-1) == 1)
    lane = kin[..., 1:4].argmax(-1)
    pos, speed = kin[..., 0], kin[..., 4]
    masks = dict(upstream=valid & (lane < 2) & (pos < 0),
                 merge=valid & (lane < 2) & (pos >= 0) & (pos <= 75.44),
                 ramp=valid & (lane == 2))
    result = {}
    for name, region in masks.items():
        values = []
        for j in range(6):
            x = speed[j*20:(j+1)*20][region[j*20:(j+1)*20]]
            if not np.isfinite(x).all():
                raise ValueError('nonfinite causal speed')
            values.append(float(x.astype(float).mean()) if len(x) else None)
        result[name] = dict(speed=values)
    return result


def build(registration_path, metadata_path, source_jobs, out, worlds, *, legacy_jobs=None):
    import numpy as np
    import torch
    registration_path, out, source_jobs = map(safe, (registration_path, out, source_jobs))
    registration = read(registration_path)
    metadata = read(metadata_path)
    if sha(metadata_path) != registration['metadata_sha256']:
        raise ValueError('outcome-blind metadata registration differs')
    roster = {w['seed']: w for w in metadata['worlds']}
    if not worlds or len(worlds) != len(set(worlds)) or not set(worlds) <= set(registration['final_comparison_worlds']):
        raise ValueError('only the registered final source roster may be exported')
    if not out.parent.is_dir() or out.exists():
        raise ValueError('new input root in existing parent required')
    if shutil.disk_usage(out.parent).free < 512*2**20:
        raise ValueError('wait for free causal input export disk')
    out.mkdir()
    save(out/'BUILD_ONCE.json', dict(registration_sha256=sha(registration_path), worlds=list(worlds),
        source_jobs=str(source_jobs), future_targets_exported=False, training=False))
    jobs = out/'data/jobs'
    jobs.mkdir(parents=True)
    expected, provenance, episodes = {}, [], []
    for seed in worlds:
        base = safe(legacy_jobs) if seed < 969550 and legacy_jobs is not None else source_jobs
        original = safe(base/f's{seed}_ctrl0')
        producer = read(original/'result.json')
        hashes = {name: sha(original/name) for name in ('start.pt', 'start_history.npz')}
        if any(producer['output_hashes'][n] != h for n, h in hashes.items()):
            raise ValueError('recorded start/export hash differs')
        start = torch.load(original/'start.pt', map_location='cpu', weights_only=False)
        if (start['seed'] != seed or start['t0'] != 300 or
                start['initial_state_sha256'] != producer['metrics']['initial_state_sha256']):
            raise ValueError('recorded observable start identity differs')
        queue = np.asarray(start.get('observed_queue_history', []), dtype=float)
        if (any(not 0 < e['time'] <= 300 for e in start['past_entries']) or
                queue.ndim != 2 or queue.shape[1] != 3 or not len(queue) or queue[-1, 0] != 300):
            raise ValueError('missing/noncausal source observation')
        for arm in ARMS:
            if read(base/f's{seed}_{arm}/result.json')['metrics']['initial_state_sha256'] != start['initial_state_sha256']:
                raise ValueError('recorded arms do not share their observable warmup')
        with np.load(original/'start_history.npz', allow_pickle=False) as data:
            history = {k: np.array(data[k], copy=True) for k in ('kin', 'mask')}
        bins = prefix_bins(history)
        job = jobs/f's{seed}_ctrl0'
        job.mkdir()
        for name in hashes:
            shutil.copyfile(original/name, job/name)
            if sha(job/name) != hashes[name]:
                raise ValueError('observable copy changed bytes')
        sanitized = dict(metrics=dict(initial_state_sha256=start['initial_state_sha256']),
                         output_hashes=hashes, causal_input_receipt_only=True)
        save(job/'result.json', sanitized)
        for name in (*hashes, 'result.json'):
            expected[f'jobs/s{seed}_ctrl0/{name}'] = sha(job/name)
        provenance.append(dict(seed=seed, original_job=str(original),
            original_result_sha256=sha(original/'result.json'), observable_sha256=hashes,
            sanitized_result_sha256=sha(job/'result.json'), future_fields_exported=False))
        for arm in ARMS:
            episodes.append(dict(seed=seed, arm=arm, demand=roster[seed]['demand'],
                penetration=roster[seed]['penetration'], initial_sha256=start['initial_state_sha256'], bins=bins))
    save(out/'data/design.json', dict(worlds=[{k: roster[s][k] for k in ('seed', 'demand', 'penetration')} for s in worlds]))
    save(out/'expected_hashes.json', expected)
    save(out/'prefix_observables.json', dict(schema='rb2int-causal-prefix/1',
        interval_s=[240, 300], future_outcome_fields=False, episodes=episodes))
    shutil.copyfile(registration_path, out/'registration.json')
    result = dict(schema='rb2int-causal-input-build/1', passed=True,
        built_at=datetime.now().astimezone().isoformat(), worlds=list(worlds), provenance=provenance,
        registration_sha256=sha(out/'registration.json'),
        observation_registry_sha256=sha(out/'expected_hashes.json'),
        prefix_observables_sha256=sha(out/'prefix_observables.json'),
        original_result_container_fields_used=['output_hashes', 'metrics.initial_state_sha256'],
        future_targets_opened=False, final_outcomes_exported=False, model_training=False)
    save(out/'INPUTS_BUILD.json', result)
    return result


def package(root, archive):
    root, archive = safe(root), safe(archive)
    if not read(root/'INPUTS_BUILD.json')['passed'] or not archive.parent.is_dir():
        raise ValueError('complete causal input build and existing archive parent required')
    with tarfile.open(archive, 'x') as stream:
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise ValueError('causal export archive must contain actual approved bytes')
            if path.is_file():
                stream.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--source-jobs', required=True)
    parser.add_argument('--legacy-jobs')
    parser.add_argument('--out', required=True)
    parser.add_argument('--archive')
    parser.add_argument('--scope', choices=('new50', 'all58'), default='new50')
    args = parser.parse_args()
    reg = read(args.registration)
    worlds = [s for s in reg['final_comparison_worlds'] if args.scope == 'all58' or 969550 <= s <= 969599]
    result = build(args.registration, args.metadata, args.source_jobs, args.out, worlds, legacy_jobs=args.legacy_jobs)
    packed = package(args.out, args.archive) if args.archive else None
    print(json.dumps(dict(passed=result['passed'], worlds=len(worlds),
        prefix_observables_sha256=result['prefix_observables_sha256'], package=packed), indent=1))


if __name__ == '__main__':
    main()
