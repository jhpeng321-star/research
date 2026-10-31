"""cpuQ supervisor: bounded own children, no replacement/retry on failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .design import worlds
from .episode import ARMS
from .mainrun import ROOT
from .storage import exclusive, write


def completed(arm,seed):
    path = ROOT/'jobs'/f'{arm}_s{seed}_episode'
    if not path.exists():
        return False
    if not (path/'RESULT.json').exists() or not (path/'CLOSE.json').exists():
        raise ValueError('existing partial job: explicit reconciliation required')
    close = json.loads((path/'CLOSE.json').read_text())
    if close['actual_sim_s'] != 900 or close['charged_sim_s'] != 900:
        raise ValueError('episode accounting mismatch')
    return True


def main(phase):
    root = ROOT/phase
    root.mkdir(exist_ok=True)
    with exclusive(root):
        if (root/'START.json').exists() or (ROOT/'STOP').exists():
            raise ValueError('existing phase or durable STOP')
        write(root/'START.json',dict(pid=os.getpid(),unix=time.time(),phase=phase))
        panel = worlds('main')[:1] if phase == 'canary' else worlds('main')
        arms = ('wm','bb') if phase == 'canary' else ARMS
        tasks = [(arm,w['seed']) for arm in arms for w in panel]
        # Interleave switching arms so neither per-model wall allocation idles.
        tasks.sort(key=lambda t:(0 if t[0] in ('wm','bb') else 1,t[1],arms.index(t[0])))
        tasks = [t for t in tasks if not completed(*t)]
        active, processes, done = {}, {}, []
        limit = 2 if phase == 'canary' else 12  # <=24 active SUMO workers incl children.
        while tasks or active:
            if (ROOT/'STOP').exists():
                write(root/'STATUS.json',dict(status='STOP',active=list(active.values()),done=done))
                return  # Children see STOP before their next physics reservation.
            while tasks and len(active)<limit:
                arm,seed = tasks.pop(0)
                with (root/f'{arm}_s{seed}.log').open('x') as log:
                    p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.c1switch.mainrun',
                        'episode','--arm',arm,'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
                active[p.pid] = dict(pid=p.pid,arm=arm,seed=seed)
                processes[p.pid] = p
                write(root/f'OWN_{arm}_s{seed}.json',active[p.pid])
            for pid,row in list(active.items()):
                # Popen handles belong exclusively to this dispatcher.
                status = processes[pid].poll()
                if status is None:
                    continue
                if status != 0:
                    write(ROOT/'STOP',dict(reason='own episode failed',child=row,status=status))
                    return
                assert completed(row['arm'],row['seed'])
                done.append(row)
                active.pop(pid)
                processes.pop(pid)
            write(root/'STATUS.json',dict(status='RUNNING',active=list(active.values()),done=done,pending=len(tasks)))
            if phase == 'main':
                for arm in ('wm','bb'):
                    if all((ROOT/'jobs'/f'{arm}_s{w["seed"]}_episode'/'RESULT.json').exists() for w in worlds('main')):
                        if not (ROOT/'rpc'/arm/'COMPLETE').exists():
                            write(ROOT/'rpc'/arm/'COMPLETE',dict(episodes=16,unix=time.time()))
            time.sleep(.5)
        write(root/'COMPLETE.json',dict(phase=phase,done=done,unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('canary','main'))
    main(p.parse_args().phase)
