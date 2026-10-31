"""Single owned local collector/finalizer. Never launches inference or signals PIDs.

On full successful completion: sync, analyze, verify, append registration and
commit ONLY generated deliverables. A failed/dead dispatcher is surfaced as a
blocked receipt rather than silently changing samples or duplicating its work.
"""
from pathlib import Path
import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from probe import ssh
from operate import REMOTE


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,obj):
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=1);f.flush();os.fsync(f.fileno())


def creation(pid):
    from ctypes import wintypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    api.OpenProcess.restype=wintypes.HANDLE
    handle=api.OpenProcess(0x1000,False,pid)
    if not handle:return None
    try:
        c,e,k,u=(wintypes.FILETIME() for _ in range(4))
        api.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        status=wintypes.DWORD()
        if not api.GetProcessTimes(handle,ctypes.byref(c),ctypes.byref(e),ctypes.byref(k),ctypes.byref(u)):return None
        if not api.GetExitCodeProcess(handle,ctypes.byref(status)) or status.value!=259:return None
        return (c.dwHighDateTime<<32)|c.dwLowDateTime
    finally:
        api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle(handle)


def command(argv,timeout=1800):
    print('COMMAND',json.dumps(argv),flush=True)
    p=subprocess.run(argv,cwd=ROOT,text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=timeout)
    print(p.stdout,flush=True)
    if p.stderr:print(p.stderr,flush=True)
    if p.returncode:raise RuntimeError(f'command failed {p.returncode}: {argv}')
    return p.stdout


def state():
    return json.loads(ssh('gpuA',f'''from pathlib import Path
import json,time,os,subprocess
p=Path({REMOTE!r});d=json.loads((p/'DISPATCH.json').read_text())
try: ticks=Path('/proc/'+str(d['pid'])+'/stat').read_text().split(') ',1)[1].split()[19]
except FileNotFoundError: ticks=None
complete=p/'COMPLETE.json'
owned=set()
for q in (p/'evidence').glob('*_OWNED.json'):
 r=json.loads(q.read_text())
 if r.get('device')!='cuda':continue
 try: live=Path('/proc/'+str(r['pid'])+'/stat').read_text().split(') ',1)[1].split()[19]==r['start_ticks']
 except FileNotFoundError:live=False
 if live:owned.add(r['pid'])
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
mem=0
for line in raw.splitlines():
 pid,mib=[v.strip() for v in line.split(',')]
 if pid.isdigit() and int(pid) in owned and mib.isdigit():mem+=int(mib)
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
print(json.dumps(dict(alive=ticks==d['start_ticks'],pid=d['pid'],start_ticks=d['start_ticks'],
 completed=len(list((p/'panel').glob('*_h300.json'))),load=os.getloadavg(),
 owned_gpu_mib=mem,gpu_free_mib=free,
 done=json.loads(complete.read_text()) if complete.exists() else None)))
'''))


def finalize():
    expected=json.loads((HERE/'evidence/WATCH_LAUNCH.json').read_text())
    assert all(digest(HERE/name)==value for name,value in expected['source_hashes'].items()),'finalizer source changed; manual review required'
    command([sys.executable,'-B',str(HERE/'sync.py')])
    command([sys.executable,'-B',str(HERE/'verify_analysis.py')])
    command([sys.executable,'-B',str(HERE/'analyze.py')])
    command([sys.executable,'-B',str(HERE/'verify_delivery.py')])
    registration=ROOT/'runs/active/C4-V7FT-20261017/REGISTRATION.md'
    line='S-C1CLOSED COMPLETED full registered 480-task/40-world K8 panel'
    assert line not in registration.read_text(encoding='utf-8')
    with registration.open('a',encoding='utf-8') as f:
        f.write('\n- '+datetime.now().astimezone().isoformat()+' '+line+
            '; original frozen BBs only; endpoint/speed, paired whole-world CIs, E2 direction/ranking and aggregate/delay components exported in docs/paper_results/C1CLOSED.md + per_world/paired/bootstrap CSV + error_horizon PNG; DELIVERY_VERIFIED.json PASS. Primary WM2 unchanged; wall restoration, passing WM-fix verdict and MAIN floor remain separate pending slots. Zero SUMO/training/push.\n')
    generated=[f'docs/paper_results/{n}' for n in ('C1CLOSED.md','C1CLOSED_per_world.csv',
        'C1CLOSED_paired.csv','C1CLOSED_bootstrap.csv','C1CLOSED_error_horizon.png','C1CLOSED_ANALYSIS.json')]
    generated += ['tools/c1closed/evidence/DELIVERY_VERIFIED.json']
    # Preserve unrelated staged work; never accidentally commit it.
    assert not command(['git','diff','--cached','--name-only']).strip(),'index contains unrelated staged work'
    command(['git','status','--short','--branch'])
    command(['git','diff','--',str(registration)])
    command(['git','log','--oneline','-10'])
    command(['git','diff','--check'])
    command(['git','add','--',*generated])
    command(['git','add','-u','--',str(registration)])
    staged=command(['git','diff','--cached','--name-only']).splitlines()
    assert set(staged)==set(generated+['runs/active/C4-V7FT-20261017/REGISTRATION.md'])
    command(['git','diff','--cached','--stat'])
    command(['git','diff','--cached','--','docs/paper_results/C1CLOSED.md'])
    command(['git','commit','-m','S-C1CLOSED: report frozen closed-loop horizon and aggregate errors'])
    write(HERE/'evidence/WATCH_COMPLETE.json',dict(finished=datetime.now().astimezone().isoformat(),
        commit=command(['git','rev-parse','HEAD']).strip(),delivery_sha256=digest(HERE/'evidence/DELIVERY_VERIFIED.json')))


def watch():
    import msvcrt
    lock=(HERE/'evidence/WATCH.lock').open('a+b')
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        try:
            s=state()
            with (HERE/'evidence/WATCH_EVENTS.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(at=datetime.now().astimezone().isoformat(),**s))+'\n')
            if s['done'] is not None:
                if not s['done']['success']:
                    write(HERE/'evidence/WATCH_BLOCKED.json',dict(reason='dispatcher completed with failed/stopped tasks; inspect own receipts',state=s));return
                try:finalize()
                except Exception as exc:
                    write(HERE/'evidence/WATCH_BLOCKED.json',dict(reason='finalizer failed; preserve outputs and inspect before retry',error=repr(exc)))
                    print('FINALIZER BLOCKED',repr(exc),flush=True)
                return
            if not s['alive']:
                write(HERE/'evidence/WATCH_BLOCKED.json',dict(reason='dispatcher incarnation absent without completion; inspect before any resume',state=s));return
        except (subprocess.TimeoutExpired,ConnectionError,OSError,RuntimeError) as exc:
            # Connection/resource interruptions do not authorize new inference.
            print('WAIT/ERROR',repr(exc),flush=True)
            if (ROOT/'docs/paper_results/C1CLOSED_ANALYSIS.json').exists():
                write(HERE/'evidence/WATCH_BLOCKED.json',dict(reason='finalizer exception after analysis; preserve outputs for review',error=repr(exc)));return
        time.sleep(300)


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--probe',action='store_true');a=p.parse_args()
    receipt=HERE/'evidence/WATCH_LAUNCH.json'
    if a.probe:print(json.dumps(state()));return
    if a.status:
        x=json.loads(receipt.read_text());print(json.dumps(dict(receipt=x,alive=creation(x['pid'])==x['creation_time'])));return
    if not a.launch:watch();return
    assert not receipt.exists(),'existing collector receipt: inspect, never duplicate'
    assert (HERE/'evidence/DISPATCH.json').is_file()
    with (HERE/'evidence/WATCH.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
        identity=creation(child.pid);assert identity is not None
        sources=('watch.py','analyze.py','verify_analysis.py','verify_delivery.py','sync.py','labels.py','aggregates.py')
        x=dict(pid=child.pid,creation_time=identity,started=datetime.now().astimezone().isoformat(),
            source_hashes={n:digest(HERE/n) for n in sources},role='collector/finalizer only; no inference launches or PID signals')
        write(receipt,x);print(json.dumps(x))


if __name__=='__main__':main()
