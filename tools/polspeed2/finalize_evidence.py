"""Recompute last-row parity and export compact owned benchmark evidence."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

import torch
from benchmark import BASE, RT, RESOLVED, VARIANTS, compare, update_row


def tensor_leaves(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, prefix + str(key) + '.')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, prefix + str(index) + '.')
    else:
        yield prefix.rstrip('.'), value


def state_parity(reference, actual):
    a, b = (dict(tensor_leaves(torch.load(path, map_location='cpu', weights_only=False)))
            for path in (reference, actual))
    diffs = []
    tensors = 0
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        if torch.is_tensor(x) and torch.is_tensor(y):
            tensors += 1
            equal = (x.shape == y.shape and x.dtype == y.dtype and torch.equal(
                x.contiguous().reshape(-1).view(torch.uint8),
                y.contiguous().reshape(-1).view(torch.uint8)))
            if not equal:
                delta = (x.double() - y.double()).abs()
                denom = torch.maximum(x.double().abs(), y.double().abs()).clamp_min(1e-30)
                diffs.append(dict(key=key, max_absolute=float(delta.max()),
                                  max_relative=float((delta/denom).max()),
                                  relative_to_reference_max=float(delta.max()/x.double().abs().max().clamp_min(1e-30))))
        elif key not in a or key not in b or x != y:
            diffs.append(dict(key=key, reference=x, actual=y))
    return dict(bitwise_identical=not diffs, compared_tensors=tensors,
                differences=diffs, n_diff=len(diffs))


export = BASE/'evidence'
export.mkdir(exist_ok=True)
reference = update_row(RESOLVED/'DRY_RUN/A/updates.jsonl')
(export/'reference181.json').write_text(json.dumps(reference, indent=2))
fga_path = Path('/root/autodl-tmp/c4/polspeed20261028/out/FGA/updates.jsonl')
fga = update_row(fga_path) if fga_path.exists() else None
if fga:
    (export/'FGA181.json').write_text(json.dumps(fga, indent=2))
report = {}
for name in VARIANTS:
    out = BASE/'out'/name
    if not (out/'RESULT.json').exists():
        continue
    result = json.loads((out/'RESULT.json').read_text())
    if (out/'updates.jsonl').exists():
        row = update_row(out/'updates.jsonl')
        result['parity'] = compare(reference, row)
        if fga:
            result['FGA_parity'] = compare(fga, row)
        serial = BASE/'out/six_serial'
        if name.startswith('six_') and (serial/'updates.jsonl').exists():
            result['six_serial_parity'] = compare(update_row(serial/'updates.jsonl'), row)
        if (out/'actor181.pt').exists():
            for other in ('nine_N2', 'six_serial'):
                path = BASE/'out'/other/'actor181.pt'
                if path.exists():
                    result[other+'_state_parity'] = state_parity(path, out/'actor181.pt')
    result['worker_peaks'] = [dict(path=str(p.relative_to(out)), **json.loads(p.read_text()))
                              for p in sorted(out.glob('future_workers/*/[0-7]/memory.json'))]
    (out/'RESULT.json').write_text(json.dumps(result, indent=2))
    report[name] = result
    target = export/name
    target.mkdir(exist_ok=True)
    for filename in ('RESULT.json', 'LAUNCH.json', 'updates.jsonl', 'memory_execution.jsonl',
                     'gpu_memory.jsonl', 'console.log', 'PID'):
        source = out/filename
        if source.exists():
            shutil.copyfile(source, target/filename)
    for source in out.glob('future_workers/*/*'):
        if source.name in ('STOP', 'PIDS.json') or source.suffix == '.log':
            destination = target/source.relative_to(out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for source in out.glob('future_workers/*/[0-7]/*'):
        if source.name in ('memory_execution.jsonl', 'memory.json') or source.suffix == '.error':
            destination = target/source.relative_to(out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    config = BASE/(name+'.toml')
    if config.exists():
        shutil.copyfile(config, target/'CONFIG.toml')
source_files = [RT/'src/wms/control'/name for name in
                ('parallel_futures.py', 'imagination_memory.py', 'dreamer.py')]
source_files += list((RT/'tools/polspeed2').glob('*.py'))
(export/'SOURCE_HASHES.json').write_text(json.dumps({str(p.relative_to(RT)):
    hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}, indent=2))
(export/'REPORT.json').write_text(json.dumps(report, indent=2))
for filename in ('qualification.log', 'qualification2.log', 'qualification3.log'):
    source = BASE/filename
    if source.exists():
        shutil.copyfile(source, export/filename)
with tarfile.open(BASE/'evidence.tar.gz', 'w:gz') as archive:
    archive.add(export, arcname='evidence')
print(json.dumps({name: dict(s_per_update=r.get('s_per_update'),
    parity=r.get('parity', {}).get('bitwise_identical'),
    leaves=r.get('parity', {}).get('compared_leaves'),
    diffs=r.get('parity', {}).get('n_diff')) for name, r in report.items()}, indent=2))
