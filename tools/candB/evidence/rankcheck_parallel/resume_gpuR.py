"""Explicit gpuR-only continuation after PI's .40 allocator OOM; no training."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/c4')
OLD = BASE / 'candB_rankcheck_2dcb6af5_host_20261030'
ROOT = BASE / 'candB_rankcheck_82dbd7bc_resume_20261030'


def write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, indent=1)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
    except FileNotFoundError:
        return None


def prepare():
    assert ROOT.is_dir()
    source = json.loads((ROOT / 'SOURCE.json').read_text())
    assert source['commit'] == '82dbd7bc7e32026998bf3ec5b912f9b23c5d67e6'
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    bad = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)
    for name, expected in manifest.items():
        assert not bad.search(name)
        assert digest(ROOT / name) == expected, name
    template = json.loads((OLD / 'measurement_PI/COMMAND.json').read_text())
    argv = template['argv']
    (ROOT / 'inputs').mkdir(exist_ok=True)
    inputs = {}
    for flag in ('--mappo', '--mappo-speed', '--current-policy', '--current-policy-driver',
                 '--horizon-gate', '--boundary', '--pairing-receipt', '--wm-config', '--input-starts'):
        original = Path(argv[argv.index(flag) + 1])
        assert original.is_file() and not bad.search(str(original))
        target = ROOT / 'inputs' / original.name
        assert not target.exists()
        shutil.copy2(original, target)
        inputs[flag] = dict(path=str(target), sha256=digest(target))
        assert inputs[flag]['sha256'] == digest(original)
    write(ROOT / 'RESUME_PREPARED.json', dict(source=source, files_verified=len(manifest),
          inputs=inputs, time_unix=time.time()))
    print('Prepared immutable gpuR source and input copies', len(manifest))


def launch(device, begin, end, suffix='', native_only=False):
    assert device in ('cpu', 'cuda') and 0 <= begin < end <= 16
    assert suffix in ('', '_a2')
    assert (native_only and device == 'cpu' and end <= 11) or (not native_only and begin >= 11)
    prepared = json.loads((ROOT / 'RESUME_PREPARED.json').read_text())
    old_owned = json.loads((OLD / 'measurement_PI/OWNERSHIP.json').read_text())
    assert ticks(old_owned['pid']) != str(old_owned['start_ticks']), 'PI is still running'
    old_doc = json.loads((OLD / 'measurement_PI/RAW.json').read_text())
    assert len(old_doc['rows']) == 11 and all('PI' in r['arms'] for r in old_doc['rows'])
    name = 'native' if native_only else 'PI_native'
    out = ROOT / f'measurement_{name}_{device}_{begin}_{end}{suffix}'
    assert not out.exists()
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    used, total = map(int, memory.split(','))
    if device == 'cuda':
        assert used + math.ceil(total * .40) + 512 <= 22000, memory
    template = json.loads((OLD / 'measurement_PI/COMMAND.json').read_text())
    args = template['argv']
    args = args[:args.index('--measurement-arms')]
    for flag, item in prepared['inputs'].items():
        # WM TOML extends sibling configs; retain its original absolute path.
        # The immutable copy is still verified, and rankcheck hashes both paths.
        if flag in args and flag != '--wm-config':
            args[args.index(flag) + 1] = item['path']
    args[args.index('--device') + 1] = device
    args[args.index('--out') + 1] = str(out / 'RAW.json')
    args += ['--measurement-arms', *(['native'] if native_only else ['PI', 'native']),
             '--future-prefix', '4',
             '--input-starts', prepared['inputs']['--input-starts']['path'],
             '--start-index', str(begin), '--stop-after', str(end), '--host-controller-sensors']
    env = dict(os.environ, **template['environment'])
    env['PYTHONPATH'] = str(ROOT / 'src') + ':' + str(ROOT)
    env['CUDA_VISIBLE_DEVICES'] = '0' if device == 'cuda' else ''
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    out.mkdir()
    write(out / 'COMMAND.json', dict(argv=args, environment={k: v for k, v in env.items()
          if k.startswith(('C4_', 'WMS_', 'OMP_', 'MKL_', 'OPENBLAS_', 'CUDA_', 'PYTHON', 'PYTORCH'))}))
    with (out / 'run.log').open('x') as stream:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: os.nice(19))
    owned = dict(pid=child.pid, start_ticks=ticks(child.pid), argv=args,
                 time_unix=time.time(), nvidia_memory_before=memory,
                 source=prepared['source'], plan_indices=[begin, end], device=device,
                 no_optimizer_updates=True, no_sumo=True)
    write(out / 'OWNERSHIP.json', owned)
    print(json.dumps(owned))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'launch':
        launch(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
               sys.argv[5] if len(sys.argv) > 5 else '')
    elif sys.argv[1] == 'native-cpu':
        launch('cpu', int(sys.argv[2]), int(sys.argv[3]), native_only=True)
    else:
        raise ValueError(sys.argv[1])
