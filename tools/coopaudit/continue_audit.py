"""Reviewed-release-only continuation: census, cap, audit, export.

No SUMO is started by a missing table, partial census, failed canary or cap refusal.
This process owns only its own SSH clients and the audit's recorded worker PIDs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.coopaudit.accounting import write
from tools.coopaudit.deploy import deploy
from tools.coopaudit.remote import run
from tools.coopaudit.design import CONTROLLERS
from tools.coopaudit.release import require_release

HERE=Path(__file__).resolve().parents[2]
EVIDENCE=HERE/'tools/coopaudit/evidence'
SOURCE=Path('E:/agent-work/c4-sumocf')
REMOTE=Path('/root/coopaudit20261031')


def selection(receipt):
    if receipt.get('status')!='PASS' or len(receipt.get('table',[]))!=27:
        raise ValueError('complete registered27-design TRAIN SNR table required')
    if {r['grid'] for r in receipt['table']} != set(range(27)):
        raise ValueError('unique complete probe grid required')
    eligible=[r for r in receipt['table'] if r.get('status')=='ok']
    for row in eligible:
        raw=row.get('raw',{})
        mean,sd=raw.get('mean'),raw.get('sd')
        if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (mean,sd)) or sd<0:
            raise ValueError('raw D_GROUP mean and SD required for owner-rule selection')
        score=abs(mean)/sd if sd else 0. if mean==0 else float('inf')
        if not math.isclose(row['selection_snr'],score,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('CV or other statistic substituted for raw D_GROUP SNR')
    if not eligible:
        raise ValueError('no eligible raw probe designs')
    chosen=sorted(eligible,key=lambda r:(-r['selection_snr'],r['sim_s_per_sample'],r['grid']))[0]
    if chosen!=receipt['selected']:
        raise ValueError('published selected design does not follow probe rule')
    size,window,rho=chosen['design']
    if size not in (2,4,'all') or window not in (10,30,60) or rho not in (0,.5,1):
        raise ValueError('unknown selected design')
    return dict(group_size=size,window_s=window,probe_rho=rho,probe_grid=chosen['grid'],
                audit_action='original deterministic controller, no sampled exploration',
                local_outcome='original on-road+source delay for reports at s[0,75.44] through window+120s',
                probe_cv_used=chosen['use_cv'],audit_credit='raw team and local C_G; no fitted CV')


def status():
    script=(HERE/'tools/coopaudit/status_host.py').read_text()
    result=run('gpuR',script)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def git_commit(message, paths):
    # Required inspection before every autonomous task-file commit.
    for args in (['status','--short'],['diff','--stat'],['log','--oneline','-10']):
        subprocess.run(['git',*args],cwd=HERE,check=True)
    if subprocess.check_output(['git','branch','--show-current'],cwd=HERE,text=True).strip()!='c4/coopmeasure':
        raise ValueError('unexpected branch')
    subprocess.run(['git','add','--',*paths],cwd=HERE,check=True)
    subprocess.run(['git','commit','-m',message],cwd=HERE,check=True)


def phase(name):
    require_release(EVIDENCE, artifact=SOURCE/'ARTIFACT_CHECK.md')
    s=status()
    if 'STOP' in s:
        raise ValueError('owned STOP prevents continuation')
    if name.upper()+'_COMPLETE.json' in s:
        return
    pid=s.get(name+'_PID.json')
    if not pid or not pid.get('live'):
        if pid:
            raise ValueError('previous owned phase exited without completion; no retry')
        deploy(name)
    while True:
        time.sleep(120)
        require_release(EVIDENCE, artifact=SOURCE/'ARTIFACT_CHECK.md')
        s=status()
        write(EVIDENCE/'STATUS.json',dict(phase=name,unix=time.time(),remote=s))
        if 'STOP' in s:
            raise ValueError('owned STOP / worker failure')
        if name.upper()+'_COMPLETE.json' in s:
            return
        if not s.get(name+'_PID.json',{}).get('live'):
            raise ValueError('owned phase exited without completion')


def export():
    require_release(EVIDENCE, artifact=SOURCE/'ARTIFACT_CHECK.md')
    script="""
import base64,json,sys
from pathlib import Path
root=Path('/root/coopaudit20261031')
pid=json.loads((root/'main_PID.json').read_text())
sys.path.insert(0,pid['code'])
from tools.coopaudit.analyze import analyze
analyze(root,root/'report')
files={p.name:base64.b64encode(p.read_bytes()).decode() for p in (root/'report').iterdir() if p.is_file()}
print(json.dumps(files))
"""
    r=run('gpuR',script)
    if r.returncode:
        raise RuntimeError(r.stderr)
    import base64
    for name,data in json.loads(r.stdout).items():
        if Path(name).name!=name or not name.startswith(('COOPAUDIT','SUMMARY')):
            raise ValueError('unexpected export path')
        (EVIDENCE/name).write_bytes(base64.b64decode(data))
    summary=json.loads((EVIDENCE/'SUMMARY.json').read_text())
    lines=['# COOPAUDIT — measured cooperation in real SUMO','',
           'S-COOPMEASURE. Complete fixed TRAIN/development panel; exploratory, not sealed evaluation.',
           'D=J(native)-J(branch); C_G=D_GROUP-sum D_ONLY:i; positive means beneficial interaction.',
           '28 worlds:24 high-share (.2/.4 x2400/2700) plus4 coverage worlds; frozen native t0=300.',
           'Primary controllers: speed-only MAPPO iter169 d9042cc5 and wall-v2.1; trained SUMOCF arm pending checkpoint.',
           'Disabled members and all continuations use native SUMO/IDM; native lanes throughout.',
           '95% stratified world-bootstrap CIs,5000draws/seed20261031; tiny/missing groups retained/reported.',
           'Raw outcomes below; no control-variate fitted on this audit. Units veh-s.','']
    by={(r['outcome'],r['controller'],r['metric']):r for r in summary['summary'] if r['cohort']=='high_share'}
    def number(x):
        return 'NA' if x is None else f'{x:.2f}'
    for outcome in ('600','300','local'):
        lines.append('## High-share24: '+outcome+' outcome')
        for c in CONTROLLERS:
            cg=by[outcome,c,'C_G']; dg=by[outcome,c,'D_GROUP']; si=by[outcome,c,'sum_D_ONLY']; fp=by[outcome,c,'frac_positive']
            lines.append(f"- {c}: C_G {number(cg['mean'])} [{number(cg['lo'])},{number(cg['hi'])}]; n={cg['n_worlds']}; share>0 {number(fp['mean'])}; D_G {number(dg['mean'])}; sumD_i {number(si['mean'])}.")
    lines.append('## All28:600s outcome (missing strata are never silently reweighted)')
    for c in CONTROLLERS:
        row=next(r for r in summary['summary'] if r['cohort']=='all' and r['outcome']=='600' and r['controller']==c and r['metric']=='C_G')
        lines.append(f"- {c}: C_G {number(row['mean'])} [{number(row['lo'])},{number(row['hi'])}]; n={row['n_worlds']}.")
    lines += ['',f"Charged {summary['charged_sim_s']} /500000; actual total {summary['actual_total_sim_s']} sim-s (all prefixes/canary/failures retained).",
              'Local cohort: reported in merge segment s[0,75.44] during window+120s; original road+source delay at that endpoint.',
              'Safety and branch costs: COOPAUDIT_branches.csv; all cohorts/cells and CIs: COOPAUDIT.csv.',
              f"Missing groups {summary['missing_groups']}/28; feasible wall requests {summary['wall_feasible_worlds']}/28; safety totals {summary['safety_totals']}.",
              'Coverage, frozen groups and raw credits: COOPAUDIT_worlds.csv; no outcome-dependent exclusions.',
              'Wall is a coordinated-mechanism control, not an assumed beneficial-C_G control; report feasibility/aborts.',
              'Traces: gpuR /root/coopaudit20261031/jobs/main_sSEED/traces/*.jsonl.gz.',
              'Schema: tools/coopaudit/TRACE_HANDOFF.md; members + nearby ramp vehicles, window+-30s.',
              'Design/binding: AUDIT_DESIGN.md and tools/coopaudit/evidence/SELECTED_DESIGN.json.',
              'Canary: live/load/native repeats and wall/MAPPO repeats bitwise; zero collision/teleport/drop.',
              'Exposure disclosure: initial seed scan opened legacy parameter JSON; no values used; see evidence/EXPOSURE_NOTE.md.',
              'Setup-only failed reservation300 retained (0actual); one documented pre-observation PATH repair.',
              'Interpret C_G and its CI directly; fleet improvement and visible yielding do not establish beneficial interaction.']
    assert len(lines)<=60
    text='\n'.join(lines)+'\n'
    (HERE/'docs/paper_results/COOPAUDIT.md').write_text(text,encoding='utf-8')
    for name in ('COOPAUDIT.md','COOPAUDIT.csv','COOPAUDIT_worlds.csv','COOPAUDIT_branches.csv'):
        src=HERE/'docs/paper_results'/name if name.endswith('.md') else EVIDENCE/name
        dest=Path('E:/research-dense/docs/paper_results')/name
        dest.write_bytes(src.read_bytes())
    git_commit('S-COOPMEASURE step 5: complete cooperation audit and world-bootstrap evidence',
               ['docs/paper_results/COOPAUDIT.md','tools/coopaudit/evidence'])


def main():
    # Check BEFORE taking/replacing PID receipts or considering any old selection.
    require_release(EVIDENCE, artifact=SOURCE/'ARTIFACT_CHECK.md')
    lock=(EVIDENCE/'CONTINUATION.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0)
        if lock.read(1)==b'':
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(EVIDENCE/'CONTINUATION_PID.json',dict(pid=os.getpid(),unix=time.time()))
    try:
        phase('census')
        phase('main')
        export()
        write(EVIDENCE/'STATUS.json',dict(phase='COMPLETE',unix=time.time()))
    except BaseException:
        write(EVIDENCE/'CONTINUATION_FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start',action='store_true')
    args=p.parse_args()
    if args.start:
        path=EVIDENCE/'CONTINUATION_PID.json'
        if path.exists():
            raise ValueError('continuation already has a recorded PID; inspect before relaunch')
        log=(EVIDENCE/'CONTINUATION.log').open('x')
        flags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS if os.name=='nt' else 0
        proc=subprocess.Popen([sys.executable,'-B','-m','tools.coopaudit.continue_audit'],
                              cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              creationflags=flags,start_new_session=os.name!='nt')
        print(json.dumps(dict(pid=proc.pid,log=str(EVIDENCE/'CONTINUATION.log'))))
    else:
        main()
