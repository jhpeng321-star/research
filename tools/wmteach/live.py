"""WM-student paired efficacy through S-DISTILL's unchanged executor/replay code."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

from tools.coopbandit.protocol import checked,sha,digest
from tools.wmteach.portable import Student

PANEL_SHA='72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6'


def parent():
    import tools.distill
    sibling=Path('E:/agent-work/c4-distill-1101/tools/distill')
    if sibling.is_dir() and str(sibling) not in tools.distill.__path__:
        tools.distill.__path__.append(str(sibling))
    return importlib.import_module('tools.distill.live')


def budget():
    return dict(paired=57600,all_event_credit=0,mechanics_extra=1200,total=58800,
                unallocated=41200,cap=100000)


def approval(text):
    for line in reversed(text.splitlines()):
        if 'S-WMTEACH BUDGET APPROVED' in line and ('100,000' in line or '100000' in line):
            if not any(word in line.lower() for word in ('awaiting','not approved','pending')):
                return line
    raise PermissionError('exact MAIN S-WMTEACH BUDGET APPROVED/100000 required')


def verify(root):
    from tools.coopplan.storage import read
    b=read(root/'BINDING.json');a=read(root/'APPROVAL.json');approval(a['line'])
    if b['budget']!=budget() or b['design_sha256']!=sha(root/'WMTEACH_LIVE_DESIGN.md'):
        raise ValueError('frozen WM live budget/design mismatch')
    for name,expected in {**b['sources'],**b['artifacts']}.items():
        if sha(root/name)!=expected:raise ValueError('immutable live input changed: '+name)
    if digest(b['panel'])!=PANEL_SHA or read(root/'PANEL.json')!=b['panel'] or b['panel']!=parent().panel():
        raise ValueError('same exact S-DISTILL24world panel required')
    return b


def admit_event(event,spent):
    # This prospectively registered efficacy-only comparison has no credit budget.
    # Returning zero skips counterfactual branches, NEVER skips the policy action.
    event['proposed_action']=deepcopy(event['executed_action'])
    event['budget_pruned']=False
    return 0


def configure():
    p=parent()
    p.CAP=100000;p.WORLD_CREDIT=0
    p.budget=budget;p.approval=approval;p.verify=verify;p.Student=Student;p.admit_event=admit_event
    return p


def episode(root,world):
    return configure().episode(root,world)


def run(root,workers=2):
    import fcntl,multiprocessing,os,time,traceback
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from tools.coopplan.storage import read,write,stop
    p=configure();binding=p.setup(root)
    if not 1<=workers<=min(2,binding['workers']):raise ValueError('two-worldworker maximum')
    if not read(root/'CPU.json')['pass_'] or not read(root/'EXPOSURE.json')['pass_']:
        raise ValueError('CPU/shared exposure receipts required')
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'DISPATCH_ONCE.json',dict(pid=os.getpid(),unix=time.time(),budget=budget()))
        try:
            first=p.panel()[0]
            base=p.episode_baseline(root,first,f's{first["seed"]}_mappo')
            repeat=p.episode_baseline(root,first,'mechanics_all_O0',kind='mechanics',scripted=True)
            if base['final_prefix']!=repeat['final_prefix']:raise ValueError('BITWISE O0 mechanics mismatch')
            write(root/'MECHANICS.json',dict(pass_=True,actual_sim_s=2400,reused_first_baseline=True))
            with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for f in as_completed([pool.submit(episode,root,w) for w in p.panel()]):f.result()
            write(root/'COMPLETE.json',dict(worlds=24,unix=time.time(),new_sumo_cap=58800))
        except BaseException:
            stop(root,traceback.format_exc());raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();run(checked(args.root),args.workers)
