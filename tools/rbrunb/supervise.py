"""Run only the published RBEVAL CLI on gpuR, with durable ownership receipts."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def path_ok(value):
    s = str(value).lower().replace('\\', '/')
    forbidden = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
                 'hetdata_truth', 'vehicle_params', 'driver_params', 'assignment', 'sealed', 'vtype')
    if any(x in s for x in forbidden):
        raise ValueError('Prohibited path')
    return Path(value)


def verify_panel(root, shards):
    """Verify every expected file hash and its identity before full-panel scoring."""
    total = 0
    for i in range(shards):
        out = root/f'shard{i}'
        complete = load(out/'COMPLETE.json')
        expected = {f'w{s}_k{k}.json' for j, s in enumerate(range(969420, 969440))
                    if j % shards == i for k in range(8)}
        assert set(complete['hashes']) == expected
        assert complete['tasks'] == len(expected)
        identity = digest(out/'IDENTITY.json')
        assert complete['identity_sha256'] == identity
        assert set(p.name for p in out.glob('w*_k*.json')) == expected
        for name, sha in complete['hashes'].items():
            assert digest(out/name) == sha
            task = load(out/name)
            assert task['identity_sha256'] == identity
            assert name == f"w{task['seed']}_k{task['future']}.json"
        total += len(expected)
    assert total == 160
    return total


def key_numbers(report):
    result = {}
    for name in ('distribution_summary.csv', 'rates.csv'):
        with (report/name).open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        # Keep exact scorer rows, including grouping and uncertainty, rather
        # than invent a pooled or precision-qualified metric here.
        if name.startswith('distribution'):
            rows = [r for r in rows if r.get('metric') == 'team_delay_vs' and
                    r.get('level') == 'arm' and r.get('arm') == 'native']
            fields = ('split', 'arm', 'horizon_s', 'worlds', 'wm_crps', 'wm_fair_crps',
                      'pooled_crps', 'cell_crps', 'error', 'range_coverage')
        else:
            rows = [r for r in rows if r.get('region') == 'upstream' and r.get('arm') == 'native']
            fields = ('split', 'demand', 'penetration', 'horizon_s', 'worlds',
                      'real_rate', 'wm_rate', 'precision_sufficient')
        result[name] = [{k: r[k] for k in fields} for r in rows]
    return result


def resources():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = raw.strip().splitlines()
    assert len(rows) == 1, 'Expected the assigned single gpuR GPU'
    used, util = map(int, rows[0].split(','))
    return dict(used_mib=used, utilization=util, load=os.getloadavg()[0], cpus=os.cpu_count())


def capacity(r):
    if r['load'] >= 24 or r['utilization'] >= 90:
        return 0
    return max(0, min(3, (20000-r['used_mib']-1024)//5120))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--code', required=True)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    import fcntl
    root, code = path_ok(a.root), path_ok(a.code)
    assert root.parent.is_dir()
    root.mkdir(exist_ok=True)
    lock = (root/'SUPERVISOR.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'FINALIZED.json').exists() or (root/'FAILED.json').exists() or (root/'DEADLINE.json').exists():
        return
    # A prior dispatch is never automatically retried after supervisor failure.
    if (root/'DISPATCH.json').exists():
        raise RuntimeError('Existing dispatch preserved; manual reconciliation required')
    m = load(path_ok(a.manifest))
    assert m['host'] == 'gpuR' and m['model_name'] != 'wm2'
    for k in ('runtime_root', 'wm', 'wm_config', 'smoke_receipt'):
        assert path_ok(m[k]).exists(), ('Missing published candidate path', k)
    assert digest(m['smoke_receipt']) == m['smoke_sha256']
    if not m.get('entrypoint'):
        entrypoint = root/'entrypoint.py'
        sources = list(dict.fromkeys([*m.get('identity_sources', []), *m.get('sources', {}),
                                      m['wm_config'], str(root/'MANIFEST.json')]))
        for source in sources:
            path_ok(source)
        text = ('import sys\nfrom importlib import import_module\n'
                f'sys.path[:0] = {[str(code), m["runtime_root"]+"/src", m["runtime_root"]]!r}\n'
                'from run_wm import main\n'
                f'loader = getattr(import_module({m["loader_module"]!r}), {m.get("loader_function", "load_candidate")!r})\n'
                f'main(model_loader=loader, identity_sources={sources!r})\n')
        with entrypoint.open('x', encoding='utf-8') as f:
            f.write(text)
    else:
        entrypoint = path_ok(m['entrypoint'])
        assert entrypoint.is_file()
    bound = {str(code/n): digest(code/n) for n in
             ('supervise.py', 'run_wm.py', 'score.py', 'core.py', 'references.json', 'real_observables.json')}
    bound.update({m[k]: digest(m[k]) for k in ('wm', 'wm_config', 'smoke_receipt')})
    bound[str(entrypoint)] = digest(entrypoint)
    for source, expected in m.get('sources', {}).items():
        source = str(path_ok(source))
        assert digest(source) == expected, 'Published candidate artifact changed'
        bound[source] = expected
    if m.get('switches'):
        bound[str(path_ok(m['switches']))] = digest(m['switches'])
    receipt = dict(owner='S-RB-RUNB', pid=os.getpid(), started_unix=time.time(),
                   boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                   manifest=m, sources=bound, deadline_unix=a.deadline)
    save(root/f'SUPERVISOR_{os.getpid()}_{time.time_ns()}.json', receipt)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=':'.join(m.get('pythonpath', ['/root/autodl-tmp/c4/speedhead20261030/pythondeps'])))
    workers = []
    try:
        while time.time() < a.deadline:
            admission = (root.parent.parent/'RB_GPU_DISPATCH.lock').open('a')
            fcntl.flock(admission, fcntl.LOCK_EX)
            r = resources()
            shards = capacity(r)
            if not shards:
                fcntl.flock(admission, fcntl.LOCK_UN)
                admission.close()
                print(json.dumps(dict(state='WAITING_RESOURCES', **r)), flush=True)
                time.sleep(min(60, max(0, a.deadline-time.time())))
                continue
            assert all(digest(x) == sha for x, sha in bound.items()), 'Published source changed'
            save(root/'DISPATCH.json', dict(state='DISPATCHING', shards=shards, resources=r,
                                           owner='S-RB-RUNB', manifest=m, unix=time.time()))
            for i in range(shards):
                out = root/f'shard{i}'
                assert not out.exists(), 'Never reuse unknown prior outputs'
                out.mkdir()
                argv = ['nice', '-n', '19', sys.executable, '-B', '-u', str(entrypoint),
                        '--out', str(out), '--runtime-root', m['runtime_root'],
                        '--wm', m['wm'], '--wm-config', m['wm_config'],
                        '--model-name', m['model_name'], '--shard', str(i),
                        '--shards', str(shards), '--memory-gib', '5']
                if m.get('switches'):
                    argv += ['--switches', m['switches']]
                with (root/f'shard{i}.log').open('x') as log:
                    child = subprocess.Popen(argv, cwd=entrypoint.parent,
                                             stdout=log, stderr=subprocess.STDOUT, env=env,
                                             start_new_session=True)
                workers.append(child)
                save(root/f'LAUNCHED_{i}.json', dict(pid=child.pid, argv=argv, unix=time.time()))
            # Hold admission through startup so other compliant dispatchers see
            # the allocated workers, rather than the pre-allocation snapshot.
            time.sleep(30)
            fcntl.flock(admission, fcntl.LOCK_UN)
            admission.close()
            break
        if not workers:
            save(root/'DEADLINE.json', dict(state='NO_FREE_RESOURCES', unix=time.time()))
            return
        stopping = False
        while any(w.poll() is None for w in workers):
            failed = any(w.poll() not in (None, 0) for w in workers)
            if (time.time() >= a.deadline or failed) and not stopping:
                # Only own directories; the public harness checks at boundaries.
                for i in range(len(workers)):
                    with (root/f'shard{i}'/'STOP').open('x') as f:
                        f.write('S-RB-RUNB deadline or failed peer shard; preserve evidence\n')
                stopping = True
            time.sleep(15)
        codes = [w.returncode for w in workers]
        if any(c != 0 for c in codes):
            raise RuntimeError(f'Owned workers exited {codes}; no automatic retry')
        tasks = verify_panel(root, len(workers))
        while os.getloadavg()[0] >= 24:
            if time.time() >= a.deadline:
                save(root/'DEADLINE.json', dict(state='FULL_INFERENCE_CPU_SCORING_BLOCKED', unix=time.time()))
                return
            time.sleep(30)
        assert all(digest(x) == sha for x, sha in bound.items()), 'Bound source changed'
        reports = root/'reports'
        reports.mkdir()
        report = reports/'complete'
        argv = ['nice', '-n', '19', sys.executable, '-B', '-u', str(code/'score.py'),
                '--inputs', *[str(root/f'shard{i}') for i in range(len(workers))],
                '--out', str(report)]
        with (root/'score.log').open('x') as log:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        gate = load(report/'gate.json')
        assert gate['complete_worlds'] == 20
        save(root/'FINALIZED.json', dict(owner='S-RB-RUNB', tasks=tasks, worlds=20,
             output=str(root), report=str(report), gate=gate, key_numbers=key_numbers(report),
             sources=bound, report_hashes={f.name: digest(f) for f in report.iterdir() if f.is_file()},
             finished_unix=time.time()))
    except Exception as e:
        if not (root/'FAILED.json').exists():
            save(root/'FAILED.json', dict(error=repr(e), unix=time.time(),
                                         owned_pids=[w.pid for w in workers]))
        raise


if __name__ == '__main__':
    main()
