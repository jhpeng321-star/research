"""Single local read-only collector. Never launches/retries remote computation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from tools.wmteach.ops import ROOT,collect

HERE=ROOT/'docs/paper_results/wmteach'
REPORT=ROOT/'docs/paper_results/WMTEACH.md'
BEGIN='<!-- WMTEACH-WATCH:BEGIN -->'
END='<!-- WMTEACH-WATCH:END -->'


def run_git(*args):
    return subprocess.run(['git',*args],cwd=ROOT,check=True,capture_output=True,text=True).stdout


def commit_owned(paths,message):
    if run_git('branch','--show-current').strip()!='c4/wmteach':
        raise RuntimeError('unexpected branch; stop collector before staging')
    if run_git('diff','--cached','--name-only').strip():
        raise RuntimeError('foreign staging exists; stop collector without touching it')
    relative=[str(p.relative_to(ROOT)).replace('\\','/') for p in paths]
    for p in relative:
        if p!='docs/paper_results/WMTEACH.md' and not p.startswith('docs/paper_results/wmteach/'):
            raise ValueError('collector can stage only its own report/receipts')
    run_git('status','--short')
    run_git('diff','--',*relative)
    run_git('log','--oneline','-10')
    run_git('add','--',*relative)
    run_git('diff','--cached','--check')
    if run_git('diff','--cached','--name-only').strip():
        run_git('commit','-m',message)


def state_section(snapshot):
    files=snapshot['files']; process=snapshot['process']
    lines=[BEGIN,'## Collected GPU progress',
           f'- Snapshot Unix{int(snapshot["unix"])}; owned PID{process["pid"]}, '
           f'identity-match={process["own_identity_matches"]}, process-state={process["state"]}.',
           f'- Complete future tasks: {snapshot["completed_future_tasks"]}; '
           f'complete H120/H300 label contexts: {len(snapshot["labels"])}; new SUMO:0sim-s.']
    if 'QUALITY.json' in files:
        for h,q in files['QUALITY.json'].items():
            lines.append(f'- Exact diagnostic H{h}: {q["states"]}/{q["expected_states"]} states, '
                f'choice agreement {q["choice_agreement"]:.3f}, sign agreement {q["sign_agreement"]}, '
                f'within-state rank {q["mean_within_state_spearman"]}, '
                f'chosen exact saving {q["exact_chosen_saving"]:.3f}veh-s, regret {q["exact_regret"]:.3f}veh-s.')
    else:
        lines.append('- Full preregistered teacher-quality check pending; student fitting waits for it.')
    students=[(name,value) for name,value in files.items() if name.endswith('/SUMMARY.json')]
    for name,s in students:
        lines.append(f'- {name}: {s["states"]} WM-label contexts/{s["worlds"]} source worlds, '
            f'{s["selected"]["family"]}/confidence{s["selected"]["confidence"]}; '
            f'nested model-label saving {s["nested"]["saving"]["mean"]:.3f}veh-s/state '
            '(WM-target CV, not real SUMO efficacy).')
    if not students:lines.append('- No completed student snapshot yet.')
    if 'COMPLETE.json' in files:lines.append('- Registered initial teacher/student job COMPLETE; live test still requires frozen-policy design/budget binding.')
    elif not process['own_identity_matches'] or process['state']=='Z':
        lines.append('- GPU process ended before COMPLETE: preserve receipts and diagnose before any explicit continuation.')
    lines.extend(['- No live result or claim of beating MAPPO.',END])
    return '\n'.join(lines)


def main(interval=1200,hours=24):
    HERE.mkdir(exist_ok=True)
    lock=(HERE/'WATCH.lock').open('a+')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write('0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=os.getpid();receipt=HERE/f'WATCH_PROCESS_{pid}.json'
    with receipt.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=pid,started_unix=time.time(),interval_s=interval,max_hours=hours,
            role='read_only_collection_and_owned_receipt_commits',remote_dispatch=False),f)
    deadline=time.monotonic()+hours*3600; expected=None
    pending=[receipt]
    frozen={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('ops.py'))}
    while time.monotonic()<deadline:
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=s for p,s in frozen.items()):
            raise RuntimeError('collector source changed; stop without mutating new user work')
        result=json.loads(collect());path=Path(result['path']);snapshot=json.loads(path.read_text())
        current=REPORT.read_text(encoding='utf-8')
        if BEGIN in current:
            before,rest=current.split(BEGIN,1); old,after=rest.split(END,1)
            block=BEGIN+old+END
            if expected is not None and block!=expected:
                raise RuntimeError('collector section changed externally; preserve it and stop')
        else:
            before,after=current.rstrip()+'\n\n','\n'
        expected=state_section(snapshot)
        REPORT.write_text(before+expected+after,encoding='utf-8')
        pending.extend([path,REPORT])
        commit_owned(pending,'wmteach: collect owned GPU teacher and student progress')
        pending=[]
        print(json.dumps(result),flush=True)
        process=snapshot['process']
        if 'COMPLETE.json' in snapshot['files'] or not process['own_identity_matches'] or process['state']=='Z':
            return
        time.sleep(interval)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--interval',type=int,default=1200)
    parser.add_argument('--hours',type=float,default=24)
    args=parser.parse_args();main(args.interval,args.hours)
