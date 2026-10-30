"""GPU-only exact-own CPU handoff once the current gpuR scorer finishes."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/c4')
OLD = BASE / 'candB_rankcheck_82dbd7bc_resume_20261030'
ROOT = BASE / 'candB_rankcheck_4d56f2a8_bias_20261030'


def write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, indent=1)


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
    except FileNotFoundError:
        return None


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def prepare():
    assert json.loads((ROOT / 'SOURCE.json').read_text())['commit'] == '4d56f2a8ef6efb6c0fa90504b9d1cc153e122974'
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest.items():
        assert not re.search(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', name, re.I)
        assert digest(ROOT / name) == expected, name
    write(ROOT / 'BIAS_PREFLIGHT.json', dict(source_commit='4d56f2a8ef6efb6c0fa90504b9d1cc153e122974',
          files_verified=len(manifest), time_unix=time.time(),
          gpuR_full_attention_CPU_CUDA_equivalence='2 passed; tolerance2e-6; verified before prepare',
          measurement_bias_rows=8, default_off=True, no_grad_only=True))
    print('Verified', len(manifest), 'immutable source files')


def retire_native_cpu():
    """Bound only our four new CPU attempts that cannot produce an arm by23:00."""
    for begin, end in ((0, 3), (3, 6), (6, 9), (9, 11)):
        directory = OLD / f'measurement_native_cpu_{begin}_{end}'
        assert not (directory / 'RAW.json').exists(), 'preserve any newly saved arm'
        owned = json.loads((directory / 'OWNERSHIP.json').read_text())
        pid = owned['pid']
        assert ticks(pid) == str(owned['start_ticks'])
        argv = [v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
        assert argv == owned['argv'] and argv[argv.index('--device') + 1] == 'cpu'
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, int(pid), 0)
        assert fd >= 0
        try:
            assert ticks(pid) == str(owned['start_ticks'])
            write(directory / 'OWNED_CPU_DEADLINE_STOP.json', dict(ownership=owned, time_unix=time.time(),
                  reason='First CPU future26-30min or unfinished after35min:4 futures exceed23:00 report deadline; no arm saved'))
            assert libc.syscall(424, fd, int(signal.SIGTERM), 0, 0) == 0
        finally:
            os.close(fd)
        print('Retired only owned bounded CPU native attempt', pid)


def start(arms, begin, end, *, handoff=False):
    assert (ROOT / 'BIAS_PREFLIGHT.json').is_file()
    snapshot = json.loads((ROOT / 'CIRCLES_GPUA_OOM_SNAPSHOT.json').read_text())
    circles = next(job for job in snapshot['jobs'] if job['directory'].endswith('/measurement_CIRCLES'))
    assert not circles['live'] and circles['completed_arms'] == {'CIRCLES': 11}
    assert 'OutOfMemoryError' in circles['log_tail'][-1]
    assert (arms, begin, end) in ((['PI', 'native', 'CIRCLES'], 11, 12), (['CIRCLES'], 12, 16))
    old_gpu = OLD / 'measurement_PI_native_cuda_12_16_a2'
    owned = json.loads((old_gpu / 'OWNERSHIP.json').read_text())
    assert ticks(owned['pid']) != str(owned['start_ticks']), 'existing gpuR scorer still running'
    assert json.loads((old_gpu / 'RAW.json').read_text())['shard_finished']
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], text=True).strip()
    used, total = map(int, memory.split(','))
    assert used + math.ceil(.40 * total) + 512 <= 22000, memory
    out = ROOT / ('measurement_' + '_'.join(arms) + f'_{begin}_{end}')
    assert not out.exists()
    if handoff:
        cpu = OLD / 'measurement_PI_native_cpu_11_12_a2'
        assert not (cpu / 'RAW.json').exists(), 'CPU has saved an arm; collect rather than repeat'
        owned = json.loads((cpu / 'OWNERSHIP.json').read_text())
        pid = owned['pid']
        assert ticks(pid) == str(owned['start_ticks'])
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        argv = [v.decode() for v in argv if v]
        assert argv == owned['argv'] and argv[argv.index('--device') + 1] == 'cpu'
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, int(pid), 0)
        assert fd >= 0
        try:
            assert ticks(pid) == str(owned['start_ticks'])
            write(cpu / 'OWNED_BIAS_GPU_HANDOFF.json', dict(ownership=owned, time_unix=time.time(),
                  reason='No complete future after>30min on CPU; same start moves to bounded-memory gpuR'))
            assert libc.syscall(424, fd, int(signal.SIGTERM), 0, 0) == 0
        finally:
            os.close(fd)
        for _ in range(100):
            if ticks(pid) != str(owned['start_ticks']):
                break
            # A defunct child has no active scoring work or GPU allocation.
            try:
                status = Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]
            except FileNotFoundError:
                break
            if status == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('CPU handoff did not finish; do not duplicate')
    else:
        large = ROOT / 'measurement_PI_native_CIRCLES_11_12'
        owned = json.loads((large / 'OWNERSHIP.json').read_text())
        assert ticks(owned['pid']) != str(owned['start_ticks'])
        assert json.loads((large / 'RAW.json').read_text())['shard_finished']
    template = json.loads((old_gpu / 'COMMAND.json').read_text())
    args = template['argv'][:template['argv'].index('--measurement-arms')]
    args[args.index('--out') + 1] = str(out / 'RAW.json')
    args += ['--measurement-arms', *arms, '--future-prefix', '4', '--input-starts',
             str(OLD / 'inputs/TRAIN_STARTS.pt'), '--start-index', str(begin),
             '--stop-after', str(end), '--host-controller-sensors', '--measurement-bias-rows', '8']
    env = dict(os.environ, **template['environment'])
    env['PYTHONPATH'] = str(ROOT / 'src') + ':' + str(ROOT)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    out.mkdir()
    write(out / 'COMMAND.json', dict(argv=args, environment={k: v for k, v in env.items()
          if k.startswith(('C4_', 'WMS_', 'OMP_', 'MKL_', 'OPENBLAS_', 'CUDA_', 'PYTHON', 'PYTORCH'))}))
    with (out / 'run.log').open('x') as stream:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: os.nice(19))
    write(out / 'OWNERSHIP.json', dict(pid=child.pid, start_ticks=ticks(child.pid),
          argv=args, time_unix=time.time(), nvidia_memory_before=memory,
          plan_indices=[begin, end], measurement_bias_rows=8, no_training=True, no_sumo=True))
    print('Started owned gpuR chunked measurement', child.pid, arms, [begin, end])


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'retire-native-cpu':
        retire_native_cpu()
    elif sys.argv[1] == 'large':
        start(['PI', 'native', 'CIRCLES'], 11, 12, handoff=True)
    elif sys.argv[1] == 'circles-tail':
        start(['CIRCLES'], 12, 16)
    else:
        raise ValueError(sys.argv[1])
