import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/c4')
ROOT = BASE/'candB_rankcheck_f396955a_tail_20261030'
OLD = BASE/'candB_rankcheck_3cd6481e_plan_20261030/measurement_MAPPO_native_lanes+MAPPO_speed_partial'
DEADLINE = datetime.fromisoformat('2026-10-30T21:20:00+08:00').timestamp()
def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
    except FileNotFoundError:
        return None
def write(path, doc):
    with path.open('x') as stream:
        json.dump(doc, stream, indent=1)
def stop(doc, receipt, reason):
    pid = doc['pid']
    if ticks(pid) != str(doc['start_ticks']):
        return
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return
    if not any(cmd):
        return
    assert b'tools.candB.rankcheck' in cmd
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, int(pid), 0)
    if fd < 0:
        return
    try:
        if ticks(pid) == str(doc['start_ticks']):
            write(receipt, dict(**doc, reason=reason, time_unix=time.time()))
            libc.syscall(424, fd, int(signal.SIGTERM), 0, 0)
    finally:
        os.close(fd)

if '--watch' in sys.argv:
    dispatch = json.loads((ROOT/'TAIL_DISPATCH.json').read_text())
    prefix_stopped = False
    memory_stopped = set()
    while time.time() < DEADLINE:
        if not prefix_stopped:
            try:
                doc = json.loads((OLD/'RAW.json').read_text())
                if len(doc['rows']) >= 4 and len(doc['rows'][3]['arms']) == 2:
                    stop(dispatch['prefix'], OLD/'OWNED_TAIL_HANDOFF_STOP.json',
                         'Keep completed plan indices0-3; disjoint tail workers own4-15')
                    prefix_stopped = True
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip())
        if memory > 45000:
            for item in reversed(dispatch['tails']):
                pid = item['ownership']['pid']
                if pid not in memory_stopped and ticks(pid) == str(item['ownership']['start_ticks']):
                    stop(item['ownership'], Path(item['directory'])/'OWNED_MEMORY_GUARD_STOP.json',
                         f'Total GPU memory{memory}MiB exceeds45000MiB reporting budget')
                    memory_stopped.add(pid)
                    break
        time.sleep(1)
    for item in dispatch['tails']:
        stop(item['ownership'], Path(item['directory'])/'OWNED_DEADLINE_STOP.json',
             'MAIN21:30 delivery; exact own-incarnation freeze at21:20')
    write(ROOT/'TAIL_GUARD_RESULT.json', dict(time_unix=time.time(),
        prefix_handoff=prefix_stopped, memory_guard_stopped=sorted(memory_stopped)))
else:
    assert ROOT.is_dir() and not (ROOT/'TAIL_DISPATCH.json').exists()
    template = json.loads((OLD/'COMMAND.json').read_text())
    prefix = json.loads((OLD/'OWNERSHIP.json').read_text())
    nvidia = subprocess.check_output(['nvidia-smi'], text=True)
    assert int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()) < 29000
    tails = []
    for arm in ('MAPPO_native_lanes', 'MAPPO_speed_partial'):
        out = ROOT/('measurement_'+arm+'_tail')
        assert not out.exists()
        out.mkdir()
        args = template['argv'][:template['argv'].index('--measurement-arms')]
        args[args.index('--out')+1] = str(out/'RAW.json')
        args[args.index('--gpu-memory-fraction')+1] = '.16'
        args += ['--measurement-arms', arm, '--future-prefix', '4', '--input-starts',
                 str(ROOT/'TRAIN_STARTS.pt'), '--start-index', '4']
        env = dict(os.environ, **template['environment'])
        env['PYTHONPATH'] = str(ROOT/'src')+':'+str(ROOT)
        write(out/'COMMAND.json', dict(argv=args, environment={**template['environment'],
            'PYTHONPATH': env['PYTHONPATH']}))
        with (out/'run.log').open('x') as log:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                preexec_fn=lambda: os.nice(19))
        owned = dict(pid=child.pid, start_ticks=ticks(child.pid), argv=args,
                     time_unix=time.time(), plan_indices=[4, 16])
        write(out/'OWNERSHIP.json', owned)
        tails.append(dict(directory=str(out), ownership=owned))
    write(ROOT/'TAIL_DISPATCH.json', dict(prefix=prefix, tails=tails, nvidia_smi=nvidia,
        deadline_unix=DEADLINE, no_optimizer_updates=True, no_sumo=True))
    with (ROOT/'TAIL_GUARD.log').open('x') as log:
        child = subprocess.Popen([sys.executable, __file__, '--watch'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT/'TAIL_GUARD_OWNERSHIP.json', dict(pid=child.pid, start_ticks=ticks(child.pid)))
    print('Disjoint tails launched', [(x['ownership']['pid'], x['ownership']['start_ticks']) for x in tails],
          'own guard', child.pid)
