"""Isolated job paths, atomic receipts, bounded owned children and result mirror."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

OUT = Path('/root/autodl-tmp/c4/longloss20261027')
OLD = Path('/root/autodl-tmp/c4/wmdeep20261027')
RT = OUT/'runtime'
CONFIG = RT/'configs/c4_wm_loop14_r16_ft.toml'
DEEP = ('lane_v5', 'merge_core', 'merge_pivot', 'ramp_speed_adapt',
        'driver_encoder', 'driver_updater', 'residual_net.lane')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(name, doc):
    p = OUT/name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    tmp.replace(p)


def event(**kw):
    with (OUT/'EVENTS.jsonl').open('a') as f:
        f.write(json.dumps({'t': now(), **kw}, default=str)+'\n')


def disk():
    free = shutil.disk_usage(OUT).free/2**30
    if free < 3.25:
        raise RuntimeError(f'disk stop: {free:.3f} GiB free')
    return free


def environment():
    env = dict(os.environ)
    env.update(PYTHONPATH=f'{OUT}/code:{RT}/src:{RT}', PYTHONDONTWRITEBYTECODE='1',
               C4_GPU_MEM_FRACTION='.85', WMS_GPU_MEM_FRACTION='.85',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               WMS_SPEED_PIVOT_DIAG_OFF='1', WMS_SPEED_BATCH_REPLAY='1',
               WMS_SPEED_FAST_CAV_DRIVE='1', WMS_SPEED_FAST_LONG_HISTORY='1',
               WMS_LOOPLONG_SPEED='1')
    return env


def mirror():
    # Mirror failures (e.g. gpuA rental end/gateway outage) must never kill training or
    # scoring; they are recorded and the next mirror retries. Results remain on gpuR.
    try:
        _mirror()
        return True
    except Exception as exc:
        event(mirror_error=repr(exc))
        return False


def _mirror():
    # No delete flags; inputs/runtime stay on gpuR, result artifacts go to gpuA.
    cmd = "/root/miniconda3/bin/python -c \"from pathlib import Path; import shutil; p=Path('/root/autodl-tmp/c4'); assert p.is_dir(); assert shutil.disk_usage(p).free>3.25*2**30; (p/'longloss20261027').mkdir(exist_ok=True)\""
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpuA', cmd],
                   check=True, timeout=35)
    subprocess.run(['rsync', '-a', '--exclude=inputs', '--exclude=views',
                    '--exclude=runtime', '--exclude=__pycache__', str(OUT)+'/',
                    'gpuA:'+str(OUT)+'/'], check=True, timeout=180)


def run(argv, label, timeout=43200):
    disk()
    event(stage=label, status='start')
    with (OUT/(label+'.log')).open('x') as log:
        p = subprocess.Popen(argv, cwd=RT, env=environment(), stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
        token = Path(f'/proc/{p.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write('CHILD.json', {'pid': p.pid, 'start_ticks': token, 'stage': label, 'argv': argv})
        try:
            deadline = time.monotonic()+timeout
            while True:
                disk()
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                try:
                    code = p.wait(timeout=min(10, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException as exc:
            # Popen object is our own unreaped child, never a name-based signal.
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            if not isinstance(exc, subprocess.TimeoutExpired):
                raise
            code = 124
    event(stage=label, exit=code)
    return code
