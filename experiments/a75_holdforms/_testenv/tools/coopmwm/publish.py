"""Owned bounded live receipt publisher; never dispatches or retries physics."""
import argparse
import contextlib
from datetime import datetime,timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import deploy,ops
from .storage import read,write


def summarize(name,value):
    if name=='CREDIT_CHECK.json':
        lines=[f"- E2 complete: diagnostic-only; passing horizons {list(value.get('weights',{}))}."]
        for h,v in value.get('horizons',{}).items():
            lines.append(f"- H{h}: C_G sign {v['C_G']['sign_agreement']},95CI {v['C_G']['ci95']}; "
                f"D_i sign {v['D_i']['sign_agreement']},95CI {v['D_i']['ci95']}; scale {v['scale']},95CI {v['scale_ci95']}.")
        return lines
    if name.startswith('CREDIT_WINDOW_'):
        lines=[f"- E2 window world{value['world']['seed']} t{value['time_s']}: exact quartet complete."]
        for h,v in value['noise'].items():
            lines.append(f"- H{h} conditional WM K8 variance: joint-team advantage {v['team_return_advantage_variance']:.6g}, "
                f"paired C_G {v['paired_credit_variance']:.6g} veh-s²; not empirical SUMO noise.")
            lines.append(f"- H{h} WM/exact C_G {value['wm'][h]['C_G']:.6g}/{value['exact'][h]['C_G']:.6g}; "
                f"D_i {value['wm'][h]['D_i']}/{value['exact'][h]['D_i']}; zero C_G is a tie, not sign evidence.")
        return lines
    if '_UPDATE_' in name:
        routing=value.get('proposal_routing',{});trust=value.get('trust',{})
        lines=[f"- {value['arm']} update{value['update']}, seed{value['training_seed']}; "
            f"SUMO-only reward; queries/accepted/executed {routing.get('queries')}/{routing.get('accepted')}/{routing.get('executed')}; "
            f"next pooled beta {trust.get('beta')}.",
            f"- Per-bucket trust: "+'; '.join(f"{k}={v['beta']:.6g}" for k,v in trust.get('buckets',{}).items())+'.',
            f"- TRAIN GAE variance {value['team_only_noise']['advantage_variance_veh_s_squared']:.6g} veh-s²; "
            f"cooperative-move residual/SNR and PPO details in `{name}`."]
        return lines
    if 'comparisons' in value:
        return [f"- {k}: {json.dumps(v,separators=(',',':'))}" for k,v in value['comparisons'].items()]
    return [f"- {name}: {json.dumps(value,separators=(',',':'))}"]


def commit(paths,message):
    # Respect any concurrent user-staged work; stage only generated task artifacts.
    def git(*args):return subprocess.run(['git',*args],cwd=ops.LOCAL,text=True,encoding='utf-8',errors='replace',capture_output=True,check=True).stdout
    git('status','--short');git('diff','--',*paths);git('log','--oneline','-10')
    staged=git('diff','--cached','--name-only').splitlines()
    if staged:raise ValueError('publisher found pre-existing staged work; preserved, commit deferred')
    git('add','--',*paths)
    return git('commit','-m',message)


def once():
    with contextlib.redirect_stdout(io.StringIO()):state=deploy.status()
    receipts=state['receipts'];fresh=[]
    for name,value in receipts.items():
        if name in ('SMOKE.json','R_SMOKE.json','CALIBRATION_WINDOWS.json'):continue
        marker=ops.EVIDENCE/'published'/(name+'.receipt.json')
        if marker.exists():continue
        fresh.append((name,value,marker))
    if not fresh:return state
    stamp=datetime.now(timezone.utc).isoformat()
    lines=[f'\n## Live publication — {stamp}',f"- Charged {state['charged_sim_s']:,}/6,000,000 sim-s; "
        f"closed jobs {state['finished']}/{state['jobs']}; pending WM {len(state['pending_WM'])}."]
    paths=['docs/paper_results/COOPMWM.md']
    for name,value,marker in fresh:
        lines.append(f'\n### {name}');lines.extend(summarize(name,value))
        paths.append('docs/paper_results/coopmwm/runtime/'+name)
    report=ops.LOCAL/paths[0]
    with report.open('a',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    for name,_,marker in fresh:
        write(marker,dict(published_utc=stamp,source=name))
        paths.append(str(marker.relative_to(ops.LOCAL)).replace('\\','/'))
    print(commit(paths,'coopmwm: publish '+','.join(name.removesuffix('.json') for name,_,_ in fresh)),flush=True)
    return state


def watch():
    import msvcrt
    claim=(ops.EVIDENCE/'PUBLISH.claim').open('a+b');claim.seek(0)
    if not claim.read(1):claim.write(b'1');claim.flush()
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    started=time.monotonic()
    write(ops.EVIDENCE/'PUBLISH_OWNER.json',dict(pid=os.getpid(),unix=time.time(),poll_s=30,max_wall_s=86400))
    try:
        while time.monotonic()-started<86400:
            try:
                state=once()
                if 'STOP.json' in state['receipts'] or 'COMPLETE.json' in state['receipts']:return
            except Exception as error:
                print(json.dumps(dict(unix=time.time(),error=repr(error),operation='monitor-only; no physical redispatch')),flush=True)
            time.sleep(30)
    finally:claim.close()


def start():
    assert not (ops.EVIDENCE/'PUBLISH_DISPATCH.json').exists(),'inspect existing publisher owner'
    log=(ops.EVIDENCE/'PUBLISH.log').open('x',encoding='utf-8')
    p=subprocess.Popen([sys.executable,'-B','-u','-m','tools.coopmwm.publish','watch'],cwd=ops.LOCAL,
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    value=dict(pid=p.pid,unix=time.time(),poll_s=30,max_wall_s=86400,mode='monitor/publish/commit only')
    write(ops.EVIDENCE/'PUBLISH_DISPATCH.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('start','watch','once'));a=p.parse_args()
    if a.mode=='start':start()
    elif a.mode=='watch':watch()
    else:once()
