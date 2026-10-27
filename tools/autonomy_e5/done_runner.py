"""One-shot S-E5DONE completion; reuse immutable worker and priority adapter."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

R = Path('/home/ubuntu/e3long')
OP = R/'operations_e5'
SEEDS = (973217, 973219, 973221, 973223, 973225, 973227)


def main():
    import fcntl
    import fix_runner as F
    import e5_runner as W
    os.environ.update(SUMO_HOME=F.SUMO_HOME, PATH=F.PATH, OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    os.chdir(R/'runtime')
    os.nice(19)
    with (OP/'RUNNER_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a = json.loads((OP/'DONE_AUTHORITY.json').read_text())
        assert a['seeds'] == list(SEEDS) and a['additional_sim_s'] == 5400
        assert a['cpuQ_cap_sim_s'] == 108000 and a['E5_cap_sim_s'] == 210600
        assert a['done_runner_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert a['amendment_sha256'] == hashlib.sha256((OP/'E5DONE_AMENDMENT.md').read_bytes()).hexdigest()
        assert json.loads((OP/'DONE_PREFLIGHT.json').read_text())['status'] == 'PASS'
        inputs = W.L.E.read(R/'inputs/coopoff_cycle1/INPUTS.json')
        W.validate(inputs, W.L.E.read(R/'inputs/coopoff_cycle1/GO_E5.json'))
        out = R/'main/e3long_coopoff/shard1'
        ledger = out/'budget.jsonl'
        charges = [json.loads(s) for s in ledger.read_text().splitlines()]
        assert len(charges) == len({x['id'] for x in charges}) == 114
        assert sum(x['sim_s'] for x in charges) == 102600
        done = W.L.E.read(out/'RESULTS.json')
        assert len(done) == 108
        jobs = [j for j in W.L.plan()['jobs'] if j['arm'] == 'ours' and j['shard'] == 1]
        todo = [j for j in jobs if j['id'] not in {x['job']['id'] for x in done}]
        assert sorted(j['seed'] for j in todo) == list(SEEDS)
        base = {x['job']['seed']: x for x in W.L.E.read(R/'main/e3long_baseline/shard1/RESULTS.json') if x['job']['arm'] == 'native'}
        on = {x['job']['seed']: x for x in W.L.E.read(R/'main/e3long_policy/shard1/RESULTS.json')}
        W.install()
        def state(status, **kw):
            W.atomic(OP/'DONE_STATUS.json', dict(status=status, t=time.time(), pid=os.getpid(),
                     n_results=len(done), charged_sim_s=sum(x['sim_s'] for x in charges), **kw))
        while W.checked_load()[0] > 8:
            assert not (R/'STOP').exists()
            state('PRIORITY_PAUSED')
            time.sleep(10)
        assert not (R/'STOP').exists()
        for job in todo:
            assert not (out/'jobs'/job['id']).exists()
            assert job['id'] not in {x['id'] for x in charges}
        # Charge each of the six original, never-attempted IDs exactly once.
        for job in todo:
            charge = dict(id=job['id'], sim_s=900, job_id=job['id'], task='S-E5DONE')
            with ledger.open('a') as f:
                f.write(json.dumps(charge)+'\n'); f.flush(); os.fsync(f.fileno())
            charges.append(charge)
        assert len(charges) == 120 and sum(x['sim_s'] for x in charges) == 108000
        state('RUNNING', in_flight=list(SEEDS))
        errors = []
        with cf.ProcessPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(W.L.worker, j, inputs, str(out/'jobs'/j['id'])): j for j in todo}
            for future in cf.as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                    F.verify_row(row, job, base[job['seed']], on[job['seed']])
                    done.append(row)
                    W.atomic(out/'RESULTS.json', done)
                except Exception:
                    errors.append(traceback.format_exc())
        if errors:
            state('FAILED', errors=errors)
            raise RuntimeError('completion failure retained; no retry authorized')
        assert len(done) == 114
        state('COMPLETE', new_charged_sim_s=5400)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (OP/'DONE_FAILURE.json').write_text(json.dumps(dict(traceback=traceback.format_exc(), t=time.time())))
        raise
