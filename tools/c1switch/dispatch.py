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
        arms = tuple(json.loads((ROOT/'BINDING.json').read_text()).get('canary_arms',('wm','bb'))) if phase == 'canary' else ARMS
        tasks = [(arm,w['seed']) for arm in arms for w in panel]
        # BB has the tighter cumulative GPU wall cap. Complete its remaining
        # worlds first, overlapping CPU branches; model draws are payload-keyed.
        priority = {'bb':0,'wm':1}
        tasks.sort(key=lambda t:(priority.get(t[0],2),t[1],arms.index(t[0])))
        tasks = [t for t in tasks if not completed(*t)]
        active, processes, done = {}, {}, []
        limit = 2 if phase == 'canary' else 14  # <=28 SUMO workers incl branch children.
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
                    if all((ROOT/'jobs'/f'{arm}_s{w["seed"]}_episode'/'pick_j9.json').exists() for w in worlds('main')):
                        if not (ROOT/'rpc'/arm/'COMPLETE').exists():
                            write(ROOT/'rpc'/arm/'COMPLETE',dict(all160_picks_committed=True,unix=time.time()))
            time.sleep(.5)
        if phase == 'main':
            from .analyze import summarize
            try:
                summarize(ROOT)
            except BaseException as exc:
                write(ROOT/'STOP',dict(reason='complete-panel analysis gate',error=str(exc)))
                raise
        write(root/'COMPLETE.json',dict(phase=phase,done=done,unix=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('canary','main'))
    main(p.parse_args().phase)
