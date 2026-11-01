"""Exclusive bounded scale collector and quality-gated approved live continuation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.wmteach.ops import ROOT,collect as collect_parent
from tools.wmteach.scale_ops import collect as collect_scale
from tools.wmteach.watch import HERE,REPORT,state_section,commit_owned

BEGIN='<!-- WMTEACH-SCALE:BEGIN -->'
END='<!-- WMTEACH-SCALE:END -->'


def section(data):
    files=data['files'];quality=files.get('HELPFUL_QUALITY.json');owner=data['process']
    lines=[BEGIN,'## OWNER12:53 scale continuation',
        f'- Snapshot Unix{int(data["unix"])}; GPU ownership {owner}.',
        f'- Generated {data["generated_states"]}/5120 distinct planned states; '
        f'labelled {data["labelled_states"]}/5120 (each K8/H120+H300); teacher SUMO0sim-s.']
    if quality:
        lines.extend([f'- Exact H300 helpful states: {quality["helpful_states"]}; '
            f'argmax agreement {quality["exact_argmax_matches"]}/{quality["helpful_states"]}, '
            f'positive-saving choices {quality["helpful_positive_choices"]}, '
            f'Wilson95lower {quality["wilson95_lower"]:.4f}; live quality gate PASS={quality["pass_"]}.',
            f'- Exact mean chosen saving on helpful states: {quality["helpful_exact_chosen_saving"]}veh-s. '
            'Inherited12state roster-only diagnostics; not full descendant qualification.'])
    else:lines.append('- Waiting for the existing exact-quality computation; no repeated forecasts or scale fitting.')
    if 'CANARY_PASS.json' in files:
        lines.append(f'- First batched K8 canary PASS; maximum paired-cost serial error '
                     f'{files["CANARY_PASS.json"]["max_team_cost_error"]}veh-s; result reused once.')
    if 'FAILURE.json' in files:
        lines.append('- Scale job FAILED; all partial work retained, no automatic retry. See snapshot traceback.')
    summary=files.get('student_scale_5120/SUMMARY.json')
    if summary:
        lines.append(f'- Scale student complete: {summary["states"]}states/{summary["worlds"]}worlds, '
            f'{summary["selected"]["family"]}/confidence{summary["selected"]["confidence"]}; '
            f'nested WM-target saving {summary["nested"]["saving"]["mean"]:.3f}veh-s/state '
            '(not real-SUMO efficacy).')
    else:lines.append('- Scale student pending. No claim of beating MAPPO.')
    lines.append(END)
    return '\n'.join(lines)


def update(text,begin,end,block):
    if begin not in text:return text.rstrip()+'\n\n'+block+'\n'
    before,rest=text.split(begin,1);_,after=rest.split(end,1)
    return before+block+after


def main(interval=1200,hours=168):
    lock=(HERE/'SCALE_WATCH.lock').open('a+')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write('0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=HERE/f'SCALE_WATCH_PROCESS_{os.getpid()}.json'
    with receipt.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),started_unix=time.time(),interval_s=interval,max_hours=hours,
            role='scale_collection_and_quality_gated_one_shot_live'),f)
    paths=[receipt]
    frozen={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),
        ROOT/'tools/wmteach/auto_live.py',ROOT/'tools/wmteach/scale_ops.py',ROOT/'tools/wmteach/scale.py')}
    deadline=time.monotonic()+hours*3600
    while time.monotonic()<deadline:
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in frozen.items()):
            raise RuntimeError('collector source changed; stop without touching new work')
        from tools.wmteach.auto_live import advance
        gate=advance();print(json.dumps(dict(live_gate=gate)),flush=True)
        parent=json.loads(collect_parent());parent_path=Path(parent['path'])
        old=json.loads(parent_path.read_text(encoding='utf-8'))
        scale=json.loads(collect_scale());scale_path=Path(scale['path'])
        data=json.loads(scale_path.read_text(encoding='utf-8'))
        text=REPORT.read_text(encoding='utf-8')
        text=update(text,'<!-- WMTEACH-WATCH:BEGIN -->','<!-- WMTEACH-WATCH:END -->',state_section(old))
        REPORT.write_text(update(text,BEGIN,END,section(data)),encoding='utf-8')
        paths.extend([parent_path,scale_path,REPORT])
        paths.extend(p for p in HERE.glob('LIVE_*.json') if p not in paths)
        commit_owned(paths,'wmteach: collect scale teacher quality and GPU progress')
        paths=[]
        print(json.dumps(scale),flush=True)
        files=data['files'];process=data['process'];live=old.get('live')
        terminal=live and (live.get('complete') or live.get('stopped') or live.get('terminal_incomplete'))
        if 'FAILURE.json' in files or not process['identity_matches'] or process['state']=='Z':
            if 'COMPLETE.json' not in files or gate['state']=='WITHHELD_TEACHER_QUALITY' or terminal:return
        if 'COMPLETE.json' in files and terminal:return
        time.sleep(interval)


def launch():
    # The child's OS lock enforces uniqueness, including cross-session races.
    path=HERE/f'SCALE_WATCH_{int(time.time())}.log'
    with path.open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.wmteach.scale_watch'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    return dict(pid=p.pid,log=str(path))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch:print(json.dumps(launch()))
    else:main()
