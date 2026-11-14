"""Drain only unstarted, already charged u1 arms while the exact owned supervisor pauses.

Operational sidecar: never changes scientific sources, samples, charges or optimizer state.
cpuQ only; its finally block resumes only PID339173/start47835664. No peer signals.
"""
import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = '/home/ubuntu/coopfree20261101_build5'
PID = 339173
TICKS = '47835664'


def identity(stat, pid=PID, ticks=TICKS):
    # comm can contain spaces/parentheses; starttime is field22, state is field3.
    number, rest = stat.split(' (',1)
    _, fields = rest.rsplit(') ',1)
    values = fields.split()
    if int(number) != pid or values[19] != ticks:
        raise ValueError('owned supervisor process identity changed')
    return values[0]


def tasks(plan, charges, existing):
    rows = {r['id']:r for r in charges}
    selected=[]
    for i,w in enumerate(plan['windows']):
        members=w['sample']['context']['members']
        for arm,chosen in [('joint',members)]+[(f'solo{j}',[v]) for j,v in enumerate(members)]:
            ident=f'u001_w{i:03}_{arm}'
            expected=dict(id=ident,sim_s=w['t0']+600,phase='train')
            if rows.get(ident) != expected:
                raise ValueError('not exactly the precharged frozen plan: '+ident)
            if ident not in existing:
                selected.append(dict(id=ident,index=i,members=chosen))
    if len({r['id'] for r in selected})!=len(selected):
        raise ValueError('duplicate arm allocation')
    return selected


def owned_signal(sig):
    identity(Path(f'/proc/{PID}/stat').read_text())
    os.kill(PID,sig)


def resume(action=owned_signal):
    """Reusable finally action, unit-tested without real process signals."""
    action(getattr(signal,'SIGCONT',18))  # Linux value; permits no-signal Windows CPU tests.


WORKER = None


def worker_setup(root):
    global WORKER
    r=Path(root)
    sys.path.insert(0,str(r/'pytools'))
    import torch
    from tools.coopfree import backend
    from tools.coopfree.runner import verify
    from tools.coopfree.storage import read
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    binding=verify(r)
    backend.setup(binding)
    WORKER=(r,binding,read(r/'PLAN_001.json'),{})


def worker_run(task):
    from tools.coopfree.runner import branch
    from tools.coopfree.storage import read
    r,binding,plan,cache=WORKER
    w=plan['windows'][task['index']]
    seed=w['world']['seed']
    if seed not in cache:
        cache[seed]=read(r/'jobs'/f's{seed}_screen'/'RESULT.json')
    state=cache[seed]['states'][str(w['t0'])]
    # Original bound branch() owns admission, prefix equality, result/close and no-retry gates.
    result=branch(r,binding,task['id'],w['world'],w['t0'],state,w['sample'],task['members'])
    return dict(id=task['id'],actual_sim_s=result['actual_sim_s'],pid=os.getpid())


def run(root, workers):
    import fcntl
    r=Path(root)
    assert str(r)==ROOT and 1<=workers<=8
    sys.path.insert(0,str(r/'pytools'))
    from tools.coopfree.runner import verify
    from tools.coopfree.storage import read,write,ledger,admission
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if os.getpriority(os.PRIO_PROCESS,0)!=19:
        raise ValueError('nice19 required')
    with (r/'batch_sidecar.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify(r)
        admission(r)
        dispatch=read(r/'DISPATCH.json')
        assert dispatch['pid']==PID and dispatch['start_ticks']==TICKS
        assert not (r/'BATCH_PAUSE.json').exists(),'one-shot sidecar; reconcile previous attempt'
        assert not (r/'UPDATE_001.json').exists(),'optimizer already completed'
        state=identity(Path(f'/proc/{PID}/stat').read_text())
        assert state not in ('T','t','Z','X'),'supervisor not running normally'
        try:
            owned_signal(signal.SIGSTOP)
            for _ in range(100):
                if identity(Path(f'/proc/{PID}/stat').read_text()) in ('T','t'):
                    break
                time.sleep(.01)
            else:
                raise ValueError('supervisor failed to pause')
            assert not (r/'UPDATE_001.json').exists(),'optimizer completed during admission'
            write(r/'BATCH_PAUSE.json',dict(supervisor_pid=PID,supervisor_start_ticks=TICKS,
                  sidecar_pid=os.getpid(),sidecar_start_ticks=Path('/proc/self/stat').read_text().split()[21],
                  workers=workers,unix=time.time(),sidecar_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            plan=read(r/'PLAN_001.json')
            rows=ledger(r)
            existing={p.name for p in (r/'jobs').iterdir() if p.is_dir()}
            # The only unfinished existing attempt may be the original supervisor's active arm.
            unfinished=[]
            for name in existing:
                p=r/'jobs'/name
                if not (p/'CLOSE.json').exists():
                    unfinished.append(name)
                    if (p/'START.json').exists():
                        start=read(p/'START.json')
                        assert start['pid']==PID and start['start_ticks']==TICKS
            assert len(unfinished)<=1,'unreconciled pre-existing attempt'
            selected=tasks(plan,rows,existing)
            write(r/'BATCH_ALLOCATION.json',dict(plan_sha256=hashlib.sha256((r/'PLAN_001.json').read_bytes()).hexdigest(),
                  excluded_existing=sorted(existing),excluded_inflight=unfinished,tasks=selected,
                  charged_before=sum(x['sim_s'] for x in rows),new_charges=0,workers=workers))
            # Bounded submission: after an error no more than workers-1 arms remain in flight.
            iterator=iter(selected);done=[]
            with futures.ProcessPoolExecutor(max_workers=workers,initializer=worker_setup,initargs=(str(r),)) as pool:
                pending={pool.submit(worker_run,t):t for t in [next(iterator,None) for _ in range(workers)] if t}
                try:
                    while pending:
                        completed,_=futures.wait(pending,return_when=futures.FIRST_COMPLETED)
                        for result in completed:
                            task=pending.pop(result)
                            receipt=result.result()
                            assert receipt['id']==task['id']
                            done.append(receipt)
                            print(json.dumps(dict(completed=len(done),assigned=len(selected),**receipt)),flush=True)
                            next_task=next(iterator,None)
                            if next_task:
                                pending[pool.submit(worker_run,next_task)]=next_task
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise
            assert ledger(r)==rows,'sidecar must never reserve/alter charges'
            write(r/'BATCH_COMPLETE.json',dict(completed=done,charged_before=sum(x['sim_s'] for x in rows),
                                             charged_after=sum(x['sim_s'] for x in ledger(r)),unix=time.time()))
        except BaseException:
            import traceback
            if not (r/'STOP.json').exists():
                write(r/'STOP.json',dict(error=traceback.format_exc(),source='operational batch sidecar',unix=time.time()))
            raise
        finally:
            resume()
            write(r/'BATCH_RESUMED.json',dict(supervisor_pid=PID,supervisor_start_ticks=TICKS,unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('root')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    run(args.root,args.workers)
