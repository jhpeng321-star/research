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
    backend=files['IDENTITY.json']['backend']
    lines=[BEGIN,f'## OWNER12:53 scale continuation — {backend}',
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
    if (HERE/'OWNER1735_PAUSE.json').exists():
        print(json.dumps(dict(state='PAUSED_BY_MAIN',resume_requires='S-WMTEACH RESUME')),flush=True)
        return
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
        if (HERE/'OWNER1735_PAUSE.json').exists():return
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in frozen.items()):
            raise RuntimeError('collector source changed; stop without touching new work')
        from tools.wmteach.auto_live import advance
        gate=advance();print(json.dumps(dict(live_gate=gate)),flush=True)
        parent=json.loads(collect_parent());parent_path=Path(parent['path'])
        old=json.loads(parent_path.read_text(encoding='utf-8'))
        from tools.wmteach import scale_ops,successor_ops
        snapshots=[]
        for profile in ('v4repair','v5'):
            scale_ops.PROFILE=profile
            scale_ops.SCALE,scale_ops.PARENT,scale_ops.BACKEND=scale_ops.PROFILES[profile]
            snapshot=json.loads(collect_scale());path=Path(snapshot['path'])
            snapshots.append((profile,path,json.loads(path.read_text(encoding='utf-8'))))
        q=json.loads(successor_ops.collect());qpath=Path(q['path'])
        quality=json.loads(qpath.read_text(encoding='utf-8'))
        text=REPORT.read_text(encoding='utf-8')
        text=update(text,'<!-- WMTEACH-WATCH:BEGIN -->','<!-- WMTEACH-WATCH:END -->',state_section(old))
        for profile,path,data in snapshots:
            begin,end=(BEGIN,END) if profile=='v4repair' else ('<!-- WMTEACH-V5-SCALE:BEGIN -->','<!-- WMTEACH-V5-SCALE:END -->')
            text=update(text,begin,end,section(data).replace(BEGIN,begin).replace(END,end))
            paths.append(path)
        qb,qe='<!-- WMTEACH-V5-QUALITY:BEGIN -->','<!-- WMTEACH-V5-QUALITY:END -->'
        lines=[qb,'## Published FULL-v5 teacher-quality rerun',
            f'- Owned process {quality["process"]}; {quality["completed_futures"]}/96futures, '
            f'{quality["completed_labels"]}/24horizon contexts; SUMO0sim-s.']
        for h,value in quality['files'].get('QUALITY.json',{}).items():
            lines.append(f'- H{h}: agreement {value["choice_agreement"]:.3f}, sign {value["sign_agreement"]}, '
                f'rank {value["mean_within_state_spearman"]}, exact chosen D {value["exact_chosen_saving"]:.3f}, '
                f'regret {value["exact_regret"]:.3f}veh-s/state.')
        if 'helpful_H300' in quality:lines.append('- Helpful H300: '+json.dumps(quality['helpful_H300'],sort_keys=True))
        lines.extend(['- Roster-only/cold-memory/source-prior diagnostics; model availability is not efficacy.',qe])
        REPORT.write_text(update(text,qb,qe,'\n'.join(lines)),encoding='utf-8')
        paths.extend([parent_path,qpath,REPORT])
        paths.extend(p for p in HERE.glob('LIVE_*.json') if p not in paths)
        commit_owned(paths,'wmteach: collect scale teacher quality and GPU progress')
        paths=[]
        print(json.dumps(dict(v5quality=q,scale=snapshots[-1][2]['process'])),flush=True)
        data=snapshots[-1][2]
        files=data['files'];process=data['process'];live=old.get('live')
        terminal=live and (live.get('complete') or live.get('stopped') or live.get('terminal_incomplete'))
        if 'FAILURE.json' in files or not process['identity_matches'] or process['state']=='Z':
            if 'COMPLETE.json' not in files or gate['state']=='WITHHELD_TEACHER_QUALITY' or terminal:return
        if 'COMPLETE.json' in files and terminal:return
        time.sleep(interval)


def launch():
    if (HERE/'OWNER1735_PAUSE.json').exists():
        return dict(state='PAUSED_BY_MAIN',resume_requires='S-WMTEACH RESUME')
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
