"""Wait for owned source-guard exits, then resume local collection/reporting ONCE.

No remote calls, inference, PID signals, or retry of any forecast. This is the
explicit numerical-CI repair continuation; the same verified cache is retained.
"""
import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]

from tools.rbint.local_resources import child_incarnation, FileTime
from tools.rb2int.finish import PYTHON
from tools.rbeval.core import load, digest, save_new

OLD = (
    ('rb2int_finish_v2',9660,134379946028423282,
     "ValueError('collection/scoring source changed; preserve completed work and resume explicitly')"),
    ('rb2int_interim_v1',24320,134379964101425816,
     "ValueError('interim source/target changed; explicit recovery required')"))


def alive(record):
    if os.name != 'nt':
        raise ValueError('this recovery binds exact Windows creation-time receipts')
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000,False,record['pid'])
    if not handle:
        if ctypes.get_last_error() == 87:
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    handle = ctypes.c_void_p(handle)
    try:
        created, exited, ktime, utime = (FileTime() for _ in range(4))
        if not kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(ktime),ctypes.byref(utime)):
            raise ctypes.WinError(ctypes.get_last_error())
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return created.ticks() == record['create_filetime'] and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def recover(out):
    paths = [Path(__file__), ROOT/'tools/rb2int/interim_watch.py',ROOT/'tools/rb2int/interim.py',
        *[ROOT/'tools/rb2int'/n for n in ('uncertainty.py','credibility.py','gate_bootstrap.py','finish.py')]]
    hashes = {str(p):digest(p) for p in paths}
    save_new(out/'START_ONCE.json',dict(source_hashes=hashes,old=OLD,
        inference_dispatch=False,signals_sent=False,cache_reused=True))
    deadline = time.time()+3600
    while True:
        if any(digest(p) != h for p,h in hashes.items()):
            raise ValueError('recovery source changed; explicit recovery required')
        ready, states = True, []
        for name,pid,created,error in OLD:
            folder = ROOT/'runs'/name
            record = load(folder/'PROCESS.json')
            if (record['pid'],record['create_filetime']) != (pid,created):
                raise ValueError('old process ownership changed')
            stopped = folder/'STOP.json'
            if stopped.exists() and load(stopped)['error'] != error:
                raise ValueError('unexpected prior STOP; preserve for explicit recovery')
            live = alive(record)
            states.append(dict(root=str(folder),alive=live,expected_STOP=stopped.exists()))
            ready &= stopped.is_file() and not live
        print(datetime.now().astimezone().isoformat(),json.dumps(states),flush=True)
        if ready:
            break
        if time.time() >= deadline:
            raise TimeoutError('old source-guard exits not confirmed in1h; no successor launched')
        time.sleep(15)
    receipts = []
    for module,name in (('tools.rb2int.finish','rb2int_finish_v3'),
                        ('tools.rb2int.interim_watch','rb2int_interim_v2')):
        destination = ROOT/'runs'/name
        if destination.exists():
            raise ValueError('preserve existing successor receipt/root; never relaunch')
        argv = [str(PYTHON),'-B','-m',module,'start','--out',str(destination)]
        result = subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        process = load(destination/'PROCESS.json')
        if not alive(process):
            raise ValueError('successor exited; preserve its STOP and do not retry')
        receipts.append(dict(module=module,process=process,output=result.stdout))
    save_new(out/'COMPLETE.json',dict(successors=receipts,no_forecast_repeated=True,
        cache_reused=True,signals_sent=False,source_hashes=hashes))
    print('READ-ONLY SUCCESSORS READY',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','watch'))
    parser.add_argument('--out',default=str(ROOT/'runs/rb2int_ci_resume_v1'))
    args = parser.parse_args();out = Path(args.out)
    if not out.parent.is_dir():
        raise ValueError('existing parent required')
    out.mkdir(exist_ok=True)
    if args.action == 'start':
        save_new(out/'DISPATCH_ONCE.json',dict(source_sha256=digest(__file__),inference_dispatch=False))
        argv = [str(PYTHON),'-B','-u',str(Path(__file__).resolve()),'watch','--out',str(out)]
        with (out/'watch.log').open('x',encoding='utf-8') as log:
            child = subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8'))
        process = dict(pid=child.pid,argv=argv,**child_incarnation(child))
        save_new(out/'PROCESS.json',process);print(json.dumps(process));return
    try:
        recover(out)
    except Exception as error:
        save_new(out/'STOP.json',dict(error=repr(error),automatic_retry=False,signals_sent=False))
        raise


if __name__ == '__main__':
    main()
