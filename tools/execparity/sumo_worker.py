"""Owned one-thread sequential worker; local registration ACK precedes every SUMO."""
from pathlib import Path
import contextlib
import datetime
import fcntl
import gzip
import hashlib
import json
import os
import sys
import time
import traceback

R = Path('/home/ubuntu/e3final')
CTRLARM = Path('/home/ubuntu/ctrlarm/amend2258')
OWN = Path('/home/ubuntu/execparity/always0020')
SOURCE = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
sys.path[:0] = [str(OWN), str(R / 'runtime/src'), str(R / 'runtime'), str(R / 'operations')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, doc, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if replace or path.name.endswith('PREFLIGHT.json') else 'x'
    with path.open(mode) as f:
        json.dump(doc, f, indent=1, allow_nan=False)
        f.write('\n')


def emit(kind, **data):
    print('@@EXECPARITY ' + json.dumps(dict(kind=kind, **data), allow_nan=False), flush=True)


def verify_sources():
    manifest = R / 'runtime/SOURCE_MANIFEST.json'
    assert sha(manifest) == SOURCE
    entries = json.loads(manifest.read_text())
    assert len(entries) == 543
    assert all(sha(R / 'runtime' / n) == h for n, h in entries.items())
    return len(entries)


def jobs_for(mode, host, only=None):
    if mode == 'canary':
        assert host == 'cpuP'
        jobs = [dict(id='s977000_always0', seed=977000, arm='always0')]
    else:
        shard = 0 if host == 'cpuP' else 1
        seeds = {977002} if shard == 0 else {977001, 977003}
        jobs = [dict(id=f's{seed}_alwaysx', seed=seed, arm='alwaysx') for seed in sorted(seeds)]
    if only is not None:
        jobs = [j for j in jobs if j['id'] == only]
        assert jobs, f'job {only} not registered for {mode}/{host}'
    return jobs


def main():
    mode, host = sys.argv[1:3]
    only = sys.argv[3] if len(sys.argv) > 3 else None
    assert mode in ('canary', 'production') and host in ('cpuP', 'cpuQ')
    lock = (OWN / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    torch.set_num_threads(1)
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert os.environ['C4_POLICY_STOCHASTIC'] == os.environ['C4_POLICY_ACCEL_NOISE'] == '0'
    verified = verify_sources()
    adapter = sha(OWN / 'tools/execparity/alwayscmd_adapter.py')
    import e3final_runner as original
    original.SEEDS.verify()
    from traffic_shepherd.c4 import e3long as L
    from tools.execparity.alwayscmd_adapter import run_e3
    inputs = json.loads((R / 'inputs/INPUTS.json').read_text())
    bindings = original.validate(inputs, json.loads((R / 'inputs/GO.json').read_text()))
    shard = 0 if host == 'cpuP' else 1
    base = R / f'main/e3final/shard{shard}/jobs'
    jobs = []
    for spec in jobs_for(mode, host, only):
        stored = json.loads((base / f"s{spec['seed']}_ours/result.json").read_text())
        j = dict(stored['job'])
        j.update(id=spec['id'], arm=spec['arm'])
        assert j['seed'] == spec['seed'] and j['warmup'] == 300 and j['horizon'] == 600 and j['charge'] == 900
        assert not (OWN / 'jobs' / j['id']).exists(), 'existing attempt: no implicit retry'
        jobs.append(j)
    preflight = dict(mode=mode, host=host, pid=os.getpid(), nice=19, threads=1,
                     source_manifest_sha256=SOURCE, verified_source_files=verified,
                     input_hashes=bindings, load=os.getloadavg(), jobs=jobs, only=only,
                     adapter_sha256=adapter,
                     worker_sha256=sha(__file__), time=time.time())
    write(OWN / f'{mode}_PREFLIGHT.json', preflight)
    emit('preflight', **preflight)
    for job in jobs:
        while os.getloadavg()[0] > 8:
            emit('waiting_load', load=os.getloadavg(), job=job)
            time.sleep(120)
        assert not (OWN / 'STOP').exists()
        emit('allocation_request', host=host, job=job)
        ack = json.loads(sys.stdin.readline())
        assert ack['authorize'] == job['id'] and ack['charged_sim_s'] == 900
        out = OWN / 'jobs' / job['id']
        out.mkdir(parents=True, exist_ok=False)
        started = dict(job=job, host=host, pid=os.getpid(), start_unix=time.time(),
                       utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       charged_sim_s=900, registration_sha256=ack['registration_sha256'])
        with (OWN / 'budget.jsonl').open('a') as ledger:
            ledger.write(json.dumps(started) + '\n')
            ledger.flush()
            os.fsync(ledger.fileno())
        write(out / 'START.json', started)
        emit('started', **started)
        measurement = L.Measurement()
        measurement.path = out
        log_path = None if job['arm'] == 'always0' else str(out / 'execution_log.jsonl.gz')
        start = time.monotonic()
        try:
            with (out / 'worker.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = run_e3(job, inputs, out, measurement=measurement, log_path=log_path)
            result.update(ok=True, job=job, wall_s=time.monotonic() - start,
                          actual_sim_s=float(measurement.progress), started=started)
            assert result['actual_sim_s'] == result['sim_s'] == 900
            m = result['metrics']
            assert set(m['windows']) == {'120', '300', '600'}
            assert not any(m[k] for k in ('collisions', 'teleports', 'dropped_vehicle_steps'))
            references = {}
            for arm in ('native', 'ours', 'mappo'):
                p = base / f"s{job['seed']}_{arm}/result.json"
                ref = json.loads(p.read_text())
                assert ref['ok'] and ref['sim_s'] == 900
                assert m['initial_state_sha256'] == ref['metrics']['initial_state_sha256']
                references[arm] = dict(path=str(p), sha256=sha(p), result=ref)
            result['alwayscmd_execution_log_rows'] = 0
            if log_path is not None:
                rows = 0
                with gzip.open(log_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if json.loads(line)['t_s'] >= 300. - 1e-9:
                            rows += 1
                assert rows > 0, 'empty post-warmup always-commanded execution ledger'
                result['alwayscmd_execution_log_rows'] = rows
            write(out / 'result.json', result)
            trajectory_hashes = {n: sha(out / n) for n in ('traj_base.npz', 'deploy/episode/episode.npz')}
            identity = None
            if job['arm'] == 'always0':
                ctrl0 = CTRLARM / 'jobs/s977000_ctrl0'
                ref_result = json.loads((ctrl0 / 'result.json').read_text())
                comparisons = {n: dict(current=sha(out / n), stored=sha(ctrl0 / n)) for n in (
                    'cumulative_120.json', 'cumulative_300.json', 'cumulative_600.json',
                    'traj_base.npz', 'deploy/episode/episode.npz')}
                scientific_sha = lambda v: hashlib.sha256(
                    json.dumps(v, sort_keys=True, allow_nan=False).encode()).hexdigest()
                current = scientific_sha(m)
                stored = scientific_sha(ref_result['metrics'])
                identity = dict(metrics_current_sha256=current, metrics_stored_sha256=stored,
                                files=comparisons, ctrl0_reference=str(ctrl0),
                                bit_identical=current == stored and
                                all(c['current'] == c['stored'] for c in comparisons.values()))
                write(out / 'CTRL0_IDENTITY.json', identity)
                assert identity['bit_identical'], 'ALWAYS0 != stored CTRL0: STOP, no production'
            verify_sources()
            emit('complete', host=host, result=result, references=references,
                 trajectory_hashes=trajectory_hashes, identity=identity)
        except BaseException:
            actual = float(getattr(measurement, 'progress', 0.0))
            failure = dict(job=job, charged_sim_s=900, actual_sim_s=actual,
                           traceback=traceback.format_exc(), host=host)
            write(out / 'failure.json', failure)
            emit('failed', **failure)
            raise
    emit('finished', host=host, mode=mode, jobs=len(jobs), runtime_unchanged=True)


if __name__ == '__main__':
    main()
