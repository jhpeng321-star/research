"""Explicit owner-only continuation of the zero-dispatch B2 waiter; never a retry."""
from datetime import datetime
import json
import subprocess
import sys
import time

import watch

DEADLINE = '2026-11-01T11:30:00+08:00'
FREEZE = '2026-11-01T09:30:00+08:00'


def main():
    state = watch.WORK/'runs'/'rbrunb'
    claim = json.loads((watch.SHARED/'WM-PHYS-B2.claim').read_text())
    assert claim['owner'] == watch.OWNER
    assert claim['manifest']['commit'] == '4015b6412cdfa2a94168b7095a59287d0fddcd3b'
    deadline = datetime.fromisoformat(DEADLINE).timestamp()
    assert time.time() < datetime.fromisoformat(FREEZE).timestamp()
    # Capture the Windows process handle before comparing its exact incarnation.
    # Kill() uses that handle; a reused numeric PID cannot redirect the operation.
    ps = r'''
$ErrorActionPreference = 'Stop'
$p = Get-Process -Id 33076
$handle = $p.Handle
$c = Get-CimInstance Win32_Process -Filter "ProcessId = 33076"
if ($p.StartTime.ToUniversalTime().Ticks -ne 639290592919104196) { throw 'Wrong exact process incarnation' }
if ([math]::Abs($p.StartTime.ToUniversalTime().Ticks - $c.CreationDate.ToUniversalTime().Ticks) -ge 10) { throw 'Process changed' }
if ($c.CommandLine -notmatch 'python.exe -B -u E:\\agent-work\\c4-rb-runb\\tools\\rbrunb\\watch.py$') { throw 'Wrong owner command' }
if ($p.StartTime.ToString('yyyy-MM-dd HH:mm:ss') -ne '2026-11-01 00:01:31') { throw 'Wrong process incarnation' }
@{pid=$p.Id; command=$c.CommandLine; created=$p.StartTime.ToString('o'); operation='OWNER_NOON_WATCHER_CONTINUATION'} | ConvertTo-Json -Compress
$p.Kill()
if (-not $p.WaitForExit(10000)) { throw 'Owned watcher did not exit' }
'''
    receipt = dict(owner=watch.OWNER, state='OWNER_NOON_CONTINUATION',
                   deadline=DEADLINE, freeze_at=FREEZE, interim=['04:00', '08:00'],
                   requested_by='INT_NOON_SCHEDULE_REQUEST.md + MAIN 02:08',
                   automatic_retry=False, unix=time.time())
    watch.save_new(state/f'NOON_REQUEST_{time.time_ns()}.json', receipt)
    p = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stdout+'\n'+p.stderr
    receipt['old_watcher'] = json.loads(p.stdout)
    # Only a waiting, pristine supervisor is eligible. All old launch/source/
    # claim receipts remain in place. Both launch and admission locks are held.
    code = r'''
from pathlib import Path
import ctypes,fcntl,hashlib,json,os,signal,subprocess,time
r=Path('/root/rbeval20261031/runb/WM-PHYS-B2')
assert r.is_dir()
launch_lock=(r/'LAUNCH.lock').open('a'); fcntl.flock(launch_lock,fcntl.LOCK_EX)
admission=(r.parent.parent/'RB_GPU_DISPATCH.lock').open('a'); fcntl.flock(admission,fcntl.LOCK_EX)
for n in ('DISPATCH.json','FAILED.json','FINALIZED.json','DEADLINE.json','OWNER_NOON_CONTINUATION.json'):
 assert not (r/n).exists(), 'Not a pristine waiting supervisor: '+n
assert not list(r.glob('LAUNCHED_*.json')) and not list(r.glob('shard*'))
files=list(r.glob('SUPERVISOR_755633_*.json')); assert len(files)==1
owned=json.loads(files[0].read_text()); launch=json.loads((r/'LAUNCH.json').read_text())
assert owned['owner']=='S-RB-RUNB' and owned['pid']==launch['pid']==755633
assert owned['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert owned['manifest']==MANIFEST
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in owned['sources'].items())
p=Path('/proc/755633'); assert os.uname().machine=='x86_64'
libc=ctypes.CDLL(None,use_errno=True); fd=libc.syscall(434,755633,0)
assert fd>=0, ('pidfd_open',ctypes.get_errno())
assert (p/'stat').read_text().split()[21]==owned['start_ticks']
cmd=(p/'cmdline').read_bytes().split(b'\0'); argv=launch['argv']
assert cmd[0].decode()=='/root/miniconda3/bin/python'
assert 'supervise.py' in cmd[3].decode() and str(r).encode() in cmd
assert cmd[3].decode()==argv[6] and float(cmd[-2])==owned['deadline_unix']
note=dict(owner='S-RB-RUNB',state='OWNER_NOON_WAITING_CONTINUATION',old_supervisor=owned,
          old_launch=launch,tasks=0,deadline_unix=DEADLINE,automatic_retry=False,unix=time.time())
with (r/'OWNER_NOON_CONTINUATION_REQUEST.json').open('x') as f:
 json.dump(note,f,indent=2); f.flush(); os.fsync(f.fileno())
assert libc.syscall(424,fd,signal.SIGTERM,ctypes.c_void_p(),0)==0
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z': break
 time.sleep(.1)
else: raise RuntimeError('Owned waiting supervisor did not exit')
os.close(fd)
assert not (r/'DISPATCH.json').exists()
argv=list(argv); argv[-1]=str(DEADLINE)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'supervisor_noon.log').open('x') as log:
 child=subprocess.Popen(argv,cwd=r,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
note.update(pid=child.pid,argv=argv,old_exited=True)
with (r/'OWNER_NOON_CONTINUATION.json').open('x') as f:
 json.dump(note,f,indent=2); f.flush(); os.fsync(f.fileno())
fcntl.flock(admission,fcntl.LOCK_UN); fcntl.flock(launch_lock,fcntl.LOCK_UN)
time.sleep(2); assert child.poll() is None, 'Continuation failed; never automatically retry'
print(json.dumps(note))
'''
    code = 'MANIFEST='+repr(claim['manifest'])+'\nDEADLINE='+repr(deadline)+'\n'+code
    receipt['remote'] = json.loads(watch.ssh(code))
    watch.save_new(state/f'NOON_CONTINUATION_{time.time_ns()}.json', receipt)
    watch.save_new(watch.SHARED/'RUNB_NOON_CONTINUATION.json', receipt)
    subprocess.run([sys.executable, '-B', str(watch.HERE/'start.py'),
                    '--deadline', DEADLINE, '--freeze-at', FREEZE], check=True)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
