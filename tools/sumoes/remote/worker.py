"""S-SUMOES-RUN owned cpuQ worker: runs a list of PRECHARGED 900 s episodes. No retries.

Arms: ``ctrl0`` = byte-identical S-WMDATA CTRL0 path (tools/ctrlarm/zero.py run_e3);
``ours`` = candidate-B cap-relative residual on the frozen E3 C1 executor with the
137 residual-head scalars set from a theta file (tools/sumoes/remote/arm.py).
Fitness J600 = cumulative_600.json team_delay_vs (E3 team delay, veh-s; lower better).
One thread, nice 19, CUDA hidden. Any failure writes failure.json + STOP and raises.
"""
from pathlib import Path
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback

OWN = Path('/home/ubuntu/sumoes20261031')
R = Path('/home/ubuntu/e3final')
SOURCE = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
ZERO_SHA = '24095a1092eafb4c894804aa8142478a802eaea046142ec777e5f24f49f73596'
OBSERVER_SHA = 'b22af4623b5fe7fbb9a8920b8e8417c3591efae66a52bae7b72f4d899b4a7874'
CTRL0_POLICY_SHA = '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c'
DESIGN_SHA = '05b84910472db97619147b12df3bc2a28f82125ec2080ae333deb252d80105c7'
CAP_SIM_S = 650000
ARMS = ('ctrl0', 'ours')
sys.path[:0] = [str(OWN / 'pytools'), str(R / 'runtime/src'), str(R / 'runtime'),
                str(OWN / 'pytools/tools/speedhead')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def emit(kind, **data):
    print('@@SUMOES ' + json.dumps(dict(kind=kind, **data), allow_nan=False), flush=True)


def write(p, d):
    with p.open('x') as f:
        json.dump(d, f, indent=1, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def verify():
    m = R / 'runtime/SOURCE_MANIFEST.json'
    assert sha(m) == SOURCE
    entries = json.loads(m.read_text())
    assert len(entries) == 543
    assert all(sha(R / 'runtime' / n) == h for n, h in entries.items())
    assert sha(OWN / 'pytools/tools/ctrlarm/zero.py') == ZERO_SHA
    assert sha(OWN / 'pytools/tools/speedhead/observer.py') == OBSERVER_SHA
    assert sha(R / 'inputs/policy_latest.pt') == CTRL0_POLICY_SHA
    assert sha(OWN / 'design.json') == DESIGN_SHA


def charged_ids():
    rows = [json.loads(s) for s in (OWN / 'accounting.jsonl').read_text().splitlines() if s]
    assert len({r['id'] for r in rows}) == len(rows)
    assert sum(r['sim_s'] for r in rows) <= CAP_SIM_S
    return {r['id'] for r in rows}


def inputs():
    out = {k: str(R / 'inputs' / Path(v).name) for k, v in
           json.loads((R / 'inputs/INPUTS.json').read_text()).items()}
    return out


def load_theta(job):
    import torch
    data = json.loads(Path(job['theta_file']).read_text())
    theta = torch.tensor(data['theta'], dtype=torch.float64)
    assert theta.numel() == 137 and bool(torch.isfinite(theta).all())
    assert data['sha256'] == theta_sha(data['theta']) == job['theta_sha256']
    return theta


def theta_sha(values):
    return hashlib.sha256(json.dumps([float(v) for v in values]).encode()).hexdigest()


def main():
    jobs_file = Path(sys.argv[1])
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    os.environ.setdefault('C4_POLICY_STOCHASTIC', '0')
    os.environ.setdefault('C4_POLICY_ACCEL_NOISE', '0')
    assert os.environ['C4_POLICY_STOCHASTIC'] == '0'
    assert os.environ['C4_POLICY_ACCEL_NOISE'] == '0'
    lock = (OWN / 'locks' / f'{jobs_file.stem}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    torch.set_num_threads(1)
    verify()
    assert importlib.metadata.version('eclipse-sumo') == '1.27.1'
    assert shutil.which('sumo'), 'SUMO binary not on PATH'
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    from tools.ctrlarm.zero import run_e3 as run_ctrl0
    from tools.sumoes.remote.arm import run_ours
    from observer import Measurement, audit_lanes
    design = json.loads((OWN / 'design.json').read_text())
    by_seed = {int(w['seed']): w for w in design['worlds']}
    jobs = json.loads(jobs_file.read_text())
    charged = charged_ids()
    emit('preflight', pid=os.getpid(), load=os.getloadavg(), jobs=[j['id'] for j in jobs])
    for job in jobs:
        seed, arm, job_id = int(job['seed']), job['arm'], job['id']
        assert arm in ARMS
        w = by_seed[seed]
        assert (w['penetration'], w['demand']) == (job['penetration'], job['demand'])
        require_development_case(seed, w['demand'], 'exposed')
        check_case(seed, w['demand'])
        out = OWN / 'jobs' / job_id
        assert job_id in charged, 'attempt not precharged; refusing'
        assert not out.exists(), 'existing attempt; never retry'
        assert not (OWN / 'STOP').exists()
        while os.getloadavg()[0] > 30:
            emit('waiting_load', load=os.getloadavg())
            time.sleep(30)
            assert not (OWN / 'STOP').exists()
        theta = load_theta(job) if arm == 'ours' else None
        emit('start', job=job_id)
        out.mkdir(parents=True)
        spec = dict(id=job_id, seed=seed, arm=arm, penetration=w['penetration'],
                    demand=w['demand'], warmup=300, horizon=600, charge=900,
                    theta_sha256=job.get('theta_sha256'), tag=job.get('tag'))
        write(out / 'START.json', dict(job=spec, pid=os.getpid(), unix=time.time()))
        m = Measurement()
        m.path = out
        begin = time.monotonic()
        try:
            with (out / 'worker.log').open('x') as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                with audit_lanes() as lanes:
                    if arm == 'ctrl0':
                        result = run_ctrl0(dict(spec), inputs(), out, measurement=m)
                    else:
                        result = run_ours(dict(spec), inputs(), out, measurement=m, theta=theta)
            actual = float(m.progress)
            cumulative = json.loads((out / 'cumulative_600.json').read_text())
            result.update(job=spec, actual_sim_s=actual, wall_s=time.monotonic() - begin,
                          lane_audit=lanes, ok=True, source=SOURCE,
                          J600=float(cumulative['team_delay_vs']))
            assert actual == 900, actual
            assert all(v == 0 for v in lanes.values()), lanes
            assert not any(result['metrics'][k] for k in
                           ('collisions', 'teleports', 'dropped_vehicle_steps'))
            assert cumulative['initial_state_sha256'] == result['metrics']['initial_state_sha256']
            verify()
            write(out / 'result.json', result)
            emit('complete', job=job_id, J600=result['J600'], wall_s=result['wall_s'])
        except BaseException:
            failure = dict(job=spec, actual_sim_s=float(getattr(m, 'progress', 0) or 0),
                           charged_sim_s=900, traceback=traceback.format_exc())
            write(out / 'failure.json', failure)
            (OWN / 'STOP').write_text(job_id + '\n')
            emit('failed', **failure)
            raise
    emit('finished', jobs=len(jobs))


if __name__ == '__main__':
    main()
