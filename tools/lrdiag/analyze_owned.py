"""Run reducers inside the 24-active-worker budget, pausing one verified own job."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path.home()/'lrdiag'


def stat(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None


def own_slot():
    for owner_file,progress_file in [('OWNER.json','PROGRESS.json'),('PAIRS_OWNER.json','PAIRS_PROGRESS.json'),
                                      ('CONTINUATION_OWNER.json','PULSE_PROGRESS.json')]:
        if not (ROOT/owner_file).exists() or not (ROOT/progress_file).exists():continue
        owner=json.loads((ROOT/owner_file).read_text());s=stat(owner['pid'])
        if not s or s[19]!=owner['start_ticks'] or s[0]=='Z':continue
        for row in reversed(json.loads((ROOT/progress_file).read_text())['live']):
            cs=stat(row['pid'])
            if cs and int(cs[1])==owner['pid'] and cs[0]!='Z':
                return row['pid'],cs[19],owner['pid']
    return None


def collection():
    if (ROOT/'YIELD').exists():return True
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            if (p/'comm').read_text().strip() in ('sumo','sumo-gui'):return True
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return False


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('partial','complete','pairs','pulses','figures','verify','plotdeps','inputs'))
    a=ap.parse_args();assert ROOT.is_dir()
    out=ROOT/'evidence'
    if a.mode=='inputs':cmd=[sys.executable,str(ROOT/'inputs_verify.py')]
    elif a.mode=='plotdeps':
        cmd=[sys.executable,'-m','pip','install','--target',str(ROOT/'plotdeps'),'--only-binary=:all:',
             '--no-deps','matplotlib','contourpy','cycler','fonttools','kiwisolver','packaging',
             'pillow','pyparsing','python-dateutil','six']
    elif a.mode in ('partial','complete'):
        cmd=[sys.executable,str(ROOT/'summarize.py'),'--root',str(ROOT),'--out',str(out)]
        if a.mode=='partial':cmd.append('--partial')
    elif a.mode in ('pairs','pulses'):
        cmd=[sys.executable,str(ROOT/('pairs_summary.py' if a.mode=='pairs' else 'pulse_summary.py')),
             '--root',str(ROOT),'--out',str(out)]
    elif a.mode=='figures':cmd=[sys.executable,str(ROOT/'figures.py'),str(out)]
    else:cmd=[sys.executable,str(ROOT/'verify.py'),'--evidence',str(out)]
    paused=own_slot();child=None;stopped=False;paused_children=[]
    receipt={'pid':os.getpid(),'start_ticks':stat(os.getpid())[19],'mode':a.mode,
             'paused_owned_slot':paused,'cmd':cmd,'time':time.time()}
    (ROOT/('STATS_OWNER_'+a.mode+'.json')).write_text(json.dumps(receipt,indent=1))
    try:
        if paused:
            pid,token,parent=paused;s=stat(pid)
            assert s and s[19]==token and int(s[1])==parent
            os.kill(pid,signal.SIGSTOP)
            # A completed model worker may be lending its slot to a serial
            # pulse-repair child. Freeze that verified own subtree as well.
            path=Path(f'/proc/{pid}/task/{pid}/children')
            if path.exists():
                for c in path.read_text().split():
                    cs=stat(int(c))
                    if cs and int(cs[1])==pid:
                        os.kill(int(c),signal.SIGSTOP);paused_children.append((int(c),cs[19]))
            receipt['paused_owned_children']=paused_children
        while collection():time.sleep(10)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONDONTWRITEBYTECODE='1',PIP_CACHE_DIR=str(ROOT/'pipcache'),PIP_NO_INPUT='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
        with (ROOT/('stats_'+a.mode+'.log')).open('a') as f:
            child=subprocess.Popen(['nice','-n','10',*cmd],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            receipt['child_pid']=child.pid;receipt['child_start_ticks']=stat(child.pid)[19]
            (ROOT/('STATS_OWNER_'+a.mode+'.json')).write_text(json.dumps(receipt,indent=1))
            while child.poll() is None:
                busy=collection()
                if busy and not stopped:child.send_signal(signal.SIGSTOP);stopped=True
                elif not busy and stopped:child.send_signal(signal.SIGCONT);stopped=False
                time.sleep(2)
            receipt['returncode']=child.returncode
    finally:
        if child is not None and child.poll() is None:
            if stopped:child.send_signal(signal.SIGCONT)
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        if paused:
            for pid,token in paused_children:
                s=stat(pid)
                if s and s[19]==token:os.kill(pid,signal.SIGCONT)
            pid,token,parent=paused;s=stat(pid)
            if s and s[19]==token:os.kill(pid,signal.SIGCONT)
        receipt['finished']=time.time()
        (ROOT/('STATS_DONE_'+a.mode+'.json')).write_text(json.dumps(receipt,indent=1))
    if receipt.get('returncode')!=0:raise SystemExit(receipt.get('returncode',1))


if __name__=='__main__':
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    main()
