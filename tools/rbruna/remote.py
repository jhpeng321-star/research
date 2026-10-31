"""Receipt-bound gpuA supervisor; never discovers or signals peer processes."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEADLINE = 1793503800  # 2026-11-01 11:30 +08:00, owner02:08 continuation
FREEZE = 1793496600  # 2026-11-01 09:30 +08:00
BASE = Path('/root/rbeval20261031/runa')
BAD = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
       'sealed', 'hetdata_truth', 'vehicle_params', 'driver_params', 'assignment')


def safe(value):
    p = Path(value)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(x in str(p).lower() for x in BAD), p
    return p


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def incarnation(pid):
    return dict(pid=pid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=Path(f'/proc/{pid}/stat').read_text().split()[21])


def alive(receipt):
    try:
        return incarnation(receipt['pid']) == {k: receipt[k] for k in ('pid', 'boot_id', 'start_ticks')}
    except (FileNotFoundError, ProcessLookupError):
        return False


def gpu(field):
    return subprocess.check_output(['nvidia-smi', f'--query-gpu={field}',
        '--format=csv,noheader,nounits'], text=True).strip()


def preflight(spec):
    assert time.time() < FREEZE, 'candidate admission frozen at09:30'
    assert spec['host'] == 'gpuA'
    if spec['variant'].upper().startswith('WM-PHYS'):
        assert spec.get('recipe_variant') in ('A', 'B', 'C'), 'publish the INT recipe variant'
        assert spec.get('future_builder_function'), 'publish the registered INT future builder'
    assert os.getloadavg()[0] < 16, 'CPU busy (20-core host quota)'
    assert int(gpu('memory.free').splitlines()[0]) >= 19500, 'GPU headroom unavailable'
    paths = [spec[k] for k in ('runtime_root', 'wm', 'wm_config', 'smoke_receipt')]
    paths += spec['identity_sources'] + spec.get('pythonpath', [])
    paths += ['/root/wmcalib/expected_hashes.json', '/root/wmcalib/data/design.json',
              '/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',
              '/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json',
              '/root/autodl-tmp/c4/speedhead20261030/e3final/runtime']
    for path in paths:
        assert safe(path).exists(), f'missing gpuA dependency: {path}'
    sources = spec['sources']
    assert all(p in sources for p in (spec['wm'], spec['wm_config'], *spec['identity_sources']))
    if spec.get('switches'):
        assert spec['switches'] in sources
    for path, expected in sources.items():
        assert sha(path) == expected, f'published artifact changed: {path}'
    assert sha(spec['smoke_receipt']) == spec['smoke_sha256'], 'smoke receipt changed'
    design = json.loads(Path('/root/wmcalib/data/design.json').read_text())
    for world in design['worlds']:
        seed = world['seed']
        assert 969400 <= seed <= 969439
        for name in ('result.json', 'start.pt', 'start_history.npz'):
            assert (Path('/root/wmcalib/data/jobs')/f's{seed}_ctrl0'/name).is_file(), (seed, name)
    return dict(load=os.getloadavg(), gpu_free_mib=gpu('memory.free'), checked_unix=time.time())


def candidate(spec):
    sys.path[:0] = [spec['runtime_root'], str(Path(spec['runtime_root'])/'src')]
    if spec.get('recipe_variant'):
        os.environ['WMPHYS_VARIANT'] = spec['recipe_variant']
    module = importlib.import_module(spec['loader_module'])
    loader = getattr(module, spec['loader_function'])
    from run_wm import main
    options = dict(model_loader=loader, identity_sources=list(dict.fromkeys(
        spec['identity_sources'] + list(spec['sources']) + [spec['smoke_receipt']])))
    if spec.get('future_builder_function'):
        assert 'future_builder' in inspect.signature(main).parameters, 'EVAL must publish its future_builder-capable runner before READY'
        builder_module = importlib.import_module(spec.get('future_builder_module', spec['loader_module']))
        builder = getattr(builder_module, spec['future_builder_function'])
        assert str(Path(inspect.getfile(builder)).resolve()) in spec['sources'], 'bind future builder source'
        options['future_builder'] = builder
    main(**options)


def completed(root):
    total = 0
    for i in range(3):
        d = root/f'shard{i}'
        marker = d/'COMPLETE.json'
        if not marker.is_file():
            return False
        value = json.loads(marker.read_text())
        names = {f'w{s}_k{k}.json' for j, s in enumerate(range(969420, 969440))
                 if j % 3 == i for k in range(8)}
        assert value['tasks'] == len(names) and set(value['hashes']) == names
        assert value['identity_sha256'] == sha(d/'IDENTITY.json')
        for name, expected in value['hashes'].items():
            assert sha(d/name) == expected, ('changed task', name)
        total += len(names)
    assert total == 160
    return True


def stop_own(root, children):
    # Cooperative first; exact child incarnations only if a resource/deadline stop persists.
    for i in range(3):
        marker = root/f'shard{i}'/'STOP'
        if marker.parent.is_dir() and not marker.exists():
            marker.touch(exist_ok=False)
    end = time.monotonic()+20
    while any(p.poll() is None for p, _ in children) and time.monotonic() < end:
        time.sleep(1)
    for p, receipt in children:
        if p.poll() is None and alive(receipt):
            p.terminate()


def supervise(spec, root):
    import fcntl
    lock = (root/'SUPERVISOR.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    children = []
    try:
        resources = preflight(spec)
        assert not (root/'WORKERS.json').exists(), 'one dispatch only'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=os.pathsep.join(spec.get('pythonpath', []) + [spec['runtime_root']]))
        for i in range(3):
            out = root/f'shard{i}'
            assert not out.exists()
            out.mkdir()
            argv = ['nice', '-n', '19', spec.get('python', '/root/miniconda3/bin/python'),
                    '-B', '-u', str(root/'code/candidate.py'), '--out', str(out),
                    '--runtime-root', spec['runtime_root'], '--wm', spec['wm'],
                    '--wm-config', spec['wm_config'], '--model-name', spec['variant'],
                    '--shard', str(i), '--shards', '3', '--memory-gib', '5']
            if spec.get('switches'):
                argv += ['--switches', spec['switches']]
            with (root/f'shard{i}.log').open('x') as log:
                p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=root, env=env, start_new_session=True)
            receipt = dict(**incarnation(p.pid), argv=argv)
            children.append((p, receipt))
            save(root/f'CHILD_{i}.json', receipt)
        save(root/'WORKERS.json', dict(workers=[r for _, r in children], resources=resources))
        peak = 0
        while True:
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                          '--format=csv,noheader,nounits'], text=True)
            owned = {r['pid'] for p, r in children if p.poll() is None and alive(r)}
            used = sum(int(row.split(',')[1]) for row in raw.splitlines()
                       if row.strip() and int(row.split(',')[0]) in owned)
            peak = max(peak, used)
            assert used < 18500, f'own GPU guard: {used}MiB'
            assert time.time() < DEADLINE, '11:30 evaluation deadline'
            if all(p.poll() is not None for p, _ in children):
                assert all(p.returncode == 0 for p, _ in children), 'worker failed; no retry'
                assert completed(root), 'incomplete panel'
                break
            assert all(p.poll() in (None, 0) for p, _ in children), 'worker failed; no retry'
            time.sleep(2)
        while os.getloadavg()[0] >= 16:
            assert time.time() < DEADLINE, 'scoring CPU deadline'
            time.sleep(30)
        reports = root/'reports'
        reports.mkdir()
        out = reports/'complete'
        with (root/'score.log').open('x') as log:
            subprocess.run(['nice', '-n', '19', spec.get('python', '/root/miniconda3/bin/python'),
                '-B', '-u', str(root/'code/score.py'), '--inputs',
                *[str(root/f'shard{i}') for i in range(3)], '--out', str(out)],
                env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        gate = json.loads((out/'gate.json').read_text())
        assert gate['complete_worlds'] == 20
        save(root/'RESULT.json', dict(status='complete', variant=spec['variant'], tasks=160,
            report=str(out), peak_own_gpu_mib=peak, gate=gate, finished_unix=time.time(),
            report_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    except BaseException as error:
        stop_own(root, children)
        save(root/'FAILED.json', dict(error=repr(error), unix=time.time(),
                                     workers=[r for _, r in children]))
        raise


def launch(spec, root):
    import fcntl
    lock = (root/'LAUNCH.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root/'DISPATCH.json').exists(), 'never duplicate dispatch'
    try:
        preflight(spec)
    except Exception as error:
        save(root/'FAILED.json', dict(error=repr(error), unix=time.time(), workers=[]))
        raise
    with (root/'supervisor.log').open('x') as log:
        p = subprocess.Popen([spec.get('python', '/root/miniconda3/bin/python'), '-B', '-u',
            str(root/'code/remote.py'), 'supervise', '--root', str(root)],
            stdout=log, stderr=subprocess.STDOUT, cwd=root, start_new_session=True)
    receipt = dict(**incarnation(p.pid), variant=spec['variant'], root=str(root))
    save(root/'DISPATCH.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'supervise'))
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    run_root = safe(args.root)
    assert run_root.parent == BASE
    published = json.loads((run_root/'spec.json').read_text())
    globals()[args.action](published, run_root)
