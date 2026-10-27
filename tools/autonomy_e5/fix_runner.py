"""Receipt-bound cpuQ operational continuation; immutable scientific worker reused."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import sys
import time
import traceback

R = Path('/home/ubuntu/e3long')
OP = R/'operations_e5'
FAILED = tuple(f's{s}_ours' for s in (973001, 973003, 973005, 973007, 973009, 973011))
SUMO_HOME = '/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo'
PATH = SUMO_HOME+'/bin:/home/ubuntu/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin'
CAP = 102600


def ledger_total(rows):
    assert len(rows) == len({r['id'] for r in rows}), 'duplicate attempt charge'
    assert all(r['sim_s'] == 900 for r in rows)
    total = sum(r['sim_s'] for r in rows)
    assert total <= CAP
    return total


def attempt(job):
    return job['id']+'.e5fix1' if job['id'] in FAILED else job['id']


def verify_row(row, job, baseline, on):
    assert row['ok'] and row['job'] == job
    assert 973000 <= job['seed'] <= 973227 and job['arm'] == 'ours'
    m = row['metrics']
    assert m['initial_state_sha256'] == baseline['metrics']['initial_state_sha256'] == on['metrics']['initial_state_sha256']
    assert not m['dropped_vehicle_steps']
    assert set(m['windows']) == {'120', '300', '600'}
    for h in (120, 300, 600):
        w = m['windows'][str(h)]
        assert w['t0_s'] == 300 and w['t_end_s'] == 300+h and not w['dropped_vehicle_steps']
        assert abs(w['team_delay_vs']-w['road_delay_vs']-w['source_wait_vs']) < 1e-3
        assert abs(w['team_delay_vs']-w['mainline_delay_vs']-w['ramp_delay_vs']) < 1e-3
        assert not w['collisions'] and not w['teleports']


def main(mode):
    assert mode in ('preflight', 'canary', 'remainder')
    os.environ.update(SUMO_HOME=SUMO_HOME, PATH=PATH, OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    sys.path.insert(0, str(OP))
    import e5_runner as W
    import fcntl
    import hashlib
    import shutil
    import subprocess
    os.chdir(R/'runtime')
    with (OP/'RUNNER_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        authority = json.loads((OP/'FIX_AUTHORITY.json').read_text())
        assert authority['remaining_sim_s'] == 182700 and authority['cap_per_box'] == CAP
        assert authority['fix_runner_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert authority['amendment_sha256'] == hashlib.sha256((OP/'E5FIX_AMENDMENT.md').read_bytes()).hexdigest()
        inputs = W.L.E.read(R/'inputs/coopoff_cycle1/INPUTS.json')
        go = W.L.E.read(R/'inputs/coopoff_cycle1/GO_E5.json')
        hashes = W.validate(inputs, go)
        W.L.check_baseline(1, hashes, W.SOURCE)
        out = R/'main/e3long_coopoff/shard1'
        ledger = out/'budget.jsonl'
        charges = [json.loads(s) for s in ledger.read_text().splitlines()]
        charged = ledger_total(charges)
        old = W.L.E.read(OP/'FAILURE_RECEIPT.json')
        assert old['charged_sim_s'] == 5400 and old['n_failures'] == 6
        for jid in FAILED:
            failure = W.L.E.read(out/'jobs'/jid/'failure.json')
            assert 'SUMO binary not found on PATH' in failure['traceback']
            assert failure['job']['id'] == jid and failure['charged_sim_s'] == 900
        if mode == 'preflight':
            assert charged == 5400
            manifest = W.L.E.read(R/'runtime/SOURCE_MANIFEST.json')
            files = manifest.get('files', manifest)
            for name, sha in files.items():
                p = Path(name)
                assert not p.is_absolute() and '..' not in p.parts
                assert not any(s in name.lower() for s in ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype'))
                assert hashlib.sha256((R/'runtime'/p).read_bytes()).hexdigest() == sha
            assert len(files) == 543
            gate = W.L.E.read(R/'inputs/coopoff_cycle1/native_gate_result.json')
            assert gate['passed'] and gate['update'] == 180 and gate['checkpoint_sha256'] == W.POLICY
            binary = shutil.which('sumo')
            assert binary and Path(binary).is_file()
            version = subprocess.run([binary, '--version'], capture_output=True, text=True, check=True).stdout
            from traffic_shepherd.config import load_config
            from traffic_shepherd.damping import CirclesController
            from wms.control.policy import load_policy_fn
            CirclesController(load_config())
            load_policy_fn(R/'inputs/coopoff_cycle1/policy_latest.pt', device='cpu')
            doc = dict(status='PASS', no_simulation=True, charged_sim_s=charged,
                       new_charged_sim_s=0, sumo=binary, sumo_version=version,
                       SUMO_HOME=SUMO_HOME, PATH=PATH, onnx_load='PASS', policy_cpu_load='PASS',
                       source_files_verified=543, gate='PASS', go_sha256=W.L.E.digest(R/'inputs/coopoff_cycle1/GO_E5.json'),
                       priority=W.probe(), t=time.time())
            W.atomic(OP/'FIX_PREFLIGHT.json', doc)
            print(json.dumps(doc), flush=True)
            return
        assert W.L.E.read(OP/'FIX_PREFLIGHT.json')['status'] == 'PASS'
        W.install()
        jobs = [j for j in W.L.plan()['jobs'] if j['shard'] == 1 and j['arm'] == 'ours']
        base = {r['job']['seed']: r for r in W.L.E.read(R/'main/e3long_baseline/shard1/RESULTS.json') if r['job']['arm'] == 'native'}
        on = {r['job']['seed']: r for r in W.L.E.read(R/'main/e3long_policy/shard1/RESULTS.json')}
        done = W.L.E.read(out/'RESULTS.json') if (out/'RESULTS.json').exists() else []
        completed = {r['job']['id'] for r in done}
        assert len(completed) == len(done)
        for row in done:
            job = next(j for j in jobs if j['id'] == row['job']['id'])
            verify_row(row, job, base[job['seed']], on[job['seed']])
            assert attempt(job) in {r['id'] for r in charges}
        todo = [j for j in jobs if j['id'] not in completed]
        if mode == 'canary':
            assert not done and charged == 5400
            todo = [next(j for j in todo if j['id'] == FAILED[0])]
        else:
            assert W.L.E.read(OP/'FIX_CANARY_VERIFIED.json')['passed']
            assert FAILED[0] in completed
        def state(status, **kw):
            W.atomic(OP/'FIX_RUNNER_STATUS.json', dict(status=status, pid=os.getpid(), t=time.time(),
                     n_results=len(done), charged_sim_s=charged, mode=mode, **kw))
        state('running')
        with cf.ProcessPoolExecutor(max_workers=1 if mode == 'canary' else 6) as pool:
            while todo:
                assert not (R/'STOP').exists()
                while W.checked_load()[0] > 8:
                    state('paused')
                    assert not (R/'STOP').exists()
                    time.sleep(10)
                available = (CAP-charged)//900
                if not available:
                    state('WAIT_OWNER_BUDGET', missing_worlds=[j['seed'] for j in todo])
                    return
                # Finish all six authorized repairs before other cpuQ worlds.
                repairs = [j for j in todo if j['id'] in FAILED]
                batch = (repairs or todo)[:min(1 if mode == 'canary' else 6, available)]
                for job in batch:
                    aid = attempt(job)
                    assert aid not in {r['id'] for r in charges}
                    target = out/('retry_jobs' if job['id'] in FAILED else 'jobs')/job['id']
                    assert not target.exists(), 'retained attempt; no further retry'
                    charge = dict(id=aid, sim_s=900, job_id=job['id'], task='S-E5FIX')
                    with ledger.open('a') as f:
                        f.write(json.dumps(charge)+'\n'); f.flush(); os.fsync(f.fileno())
                    charges.append(charge)
                charged = ledger_total(charges)
                state('running', in_flight=[j['seed'] for j in batch])
                futures = {pool.submit(W.L.worker, j, inputs, str(out/('retry_jobs' if j['id'] in FAILED else 'jobs')/j['id'])): j for j in batch}
                errors = []
                for future in cf.as_completed(futures):
                    job = futures[future]
                    try:
                        row = future.result()
                        verify_row(row, job, base[job['seed']], on[job['seed']])
                        done.append(row)
                        W.atomic(out/'RESULTS.json', done)
                    except Exception:
                        errors.append(traceback.format_exc())
                if errors:
                    state('failed', errors=errors)
                    raise RuntimeError('retained failed repair/new attempt; no further dispatch')
                todo = [j for j in todo if j not in batch]
                if mode == 'canary':
                    W.atomic(OP/'FIX_CANARY_VERIFIED.json', dict(passed=True, seed=973001,
                             charged_sim_s=900, historical_charged_sim_s=5400,
                             CRN_clocks_conservation='PASS', t=time.time()))
        state('canary_complete' if mode == 'canary' else 'complete')


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except BaseException:
        doc = dict(status='failed', pid=os.getpid(), t=time.time(), traceback=traceback.format_exc())
        (OP/'FIX_RUNNER_STATUS.json').write_text(json.dumps(doc, indent=2))
        raise
