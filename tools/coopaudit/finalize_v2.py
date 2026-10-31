"""Finish the EXISTING v2 run: observe, export, verify, report and commit. Never SUMO."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.coopaudit.accounting import write
from tools.coopaudit.remote import run
from tools.coopaudit.design import sha

HERE=Path(__file__).resolve().parents[2]
EVIDENCE=HERE/'tools/coopaudit/evidence'
PROBE="""
import json,os
from pathlib import Path
r=Path('/root/coopaudit20261031')
s=Path('/root/autodl-tmp/coopaudit20261101_v2')
j=list(s.glob('v2_s*'))
print(json.dumps(dict(complete=(r/'V2_MAIN_COMPLETE.json').exists(),stop=(r/'STOP').exists(),
load=os.getloadavg(),complete_worlds=sum((p/'result.json').exists() for p in j),
complete_branches=sum(len(list(p.glob('branch_*.json'))) for p in j),
charged_sim_s=sum(json.loads(x)['sim_s'] for x in (r/'accounting.jsonl').read_text().splitlines()))))
"""


def git(*args):
    return subprocess.check_output(['git',*args],cwd=HERE,text=True).rstrip('\r\n')


def commit(message,paths):
    # Never absorb other work or staged files, even in an unexpectedly shared tree.
    assert not git('diff','--cached','--name-only'), 'pre-existing staged work'
    print(git('status','--short'),flush=True)
    print(git('diff','--stat'),flush=True)
    print(git('log','--oneline','-10'),flush=True)
    for line in git('status','--porcelain').splitlines():
        path=line[3:]
        if not (path.startswith('tools/coopaudit/evidence/') or path in
                ('docs/paper_results/COOPAUDIT.md','tools/coopaudit/RESUME.md')):
            raise ValueError('unexpected worktree change before automatic commit: '+path)
    subprocess.run(['git','add','--',*paths],cwd=HERE,check=True)
    subprocess.run(['git','commit','-m',message],cwd=HERE,check=True)
    return git('rev-parse','HEAD')


def report(summary):
    def lookup(cohort,outcome,c,k):
        return next(r for r in summary['summary'] if (r['cohort'],r['outcome'],r['controller'],r['metric'])==(cohort,outcome,c,k))
    def f(x):
        return 'NA' if x is None else f'{x:+.3f}'
    def ci(r):
        return f"{f(r['mean'])} [{f(r['lo'])},{f(r['hi'])}]"
    lines=['# COOPAUDIT — measured cooperation in real SUMO','',
           'S-COOPMEASURE v2 COMPLETE; exploratory TRAIN audit, owner2026-10-31 15:45/17:20.',
           'D(S)=J(native)-J(S); C_G=D_GROUP-sum_i D_ONLY:i; positive C_G is beneficial interaction.',
           'Native means no acceleration/lane request; disabled members use IDM. All vehicles released after60s.',
           'Each branch starts a fresh SUMO process and reruns the same native300s prefix; NO save/loadState.',
           'Frozen observable pair at t0=300s, zone[-150,75.44]m; corrected upstream grid7/pair60s adopted.',
           'MAPPO iter169 d9042cc5; wall-v2.1-closure; fixed final SUMOCF u15 e4b51f64 (deployed mean, no noise).',
           'TRAIN969650-969677:28 worlds,24 high-share(.2/.4 x2400/2700)+4 coverage;21 eligible pairs,20 high-share.',
           'Seven missing groups remain missing; no replacement/truncation. Only3/21 pairs have feasible wall geometry.',
           'All28 allocation-weighted CIs are UNDEFINED (missing strata); never treat missing cooperation as zero.',
           'Below: conditional eligible21-world means and95% stratified world-bootstrap CIs; eligible-cell weights.',
           '5000 paired world draws, seed20261031; units veh-s. Raw strict share C_G>0 is reported separately.',
           'J_H integrates fleet+source delay over H seconds AFTER t0 (J300/J600 end at absolute600/900s).','']
    for outcome in ('600','300','local'):
        lines.append('## '+('Merge-local J at window+120s' if outcome=='local' else 'Team J'+outcome))
        for c in ('mappo','wall','sumocf'):
            cg=lookup('eligible21',outcome,c,'C_G')
            dg=lookup('eligible21',outcome,c,'D_GROUP')
            si=lookup('eligible21',outcome,c,'sum_D_ONLY')
            fp=lookup('eligible21',outcome,c,'frac_positive')
            lines.append(f"- {c}: C_G {ci(cg)}; share>0={fp['mean']:.3f}; D_G {f(dg['mean'])}; sumD_i {f(si['mean'])}; n21.")
    lines+=['','## Prespecified high-share coverage and pairing']
    for c in ('mappo','wall','sumocf'):
        cg=lookup('eligible_high_share20','600',c,'C_G')
        lines.append(f'- High-share eligible20/24, {c}: C_G600 {ci(cg)} (conditional eligible worlds).')
    for p in summary['paired']:
        if p['metric']=='C_G':
            lines.append(f"- Paired SUMOCF-MAPPO delta C_G, {p['outcome']}: {ci(p)}; n21.")
    lines+=['','## Safety, mechanics and audit trail']
    for c,m in summary['safety'].items():
        lines.append(f"- {c} GROUP/B across21 worlds: TTC<3 {m['ttc_below_3_vs']:.1f}; hard-braking {m['hard_braking_vs']:.1f}veh-s; collision/teleport/drop=0/0/0.")
    lines+=['- Corrected canary16 full900s reruns: native/no-op/three-controller repeats bitwise; wall106commands, assembly307s/handback353s.',
            '- Canary wall C_G600+1658.239758 (D_G+1616.622358, sumD_i-41.617400); MAPPO/SUMOCF C_G0. One mechanics world, NOT panel inference.',
            '- Main231 full-prefix gates PASS;247 including canary. Historical saved-state outcomes excluded; all prior data retained.',
            f"- Budget: {summary['actual_total_sim_s']}actual/{summary['charged_sim_s']}charged of500000sim-s;300 unused setup reservation retained; no simulator retries/refunds.",
            '- Local cohort: vehicles observed in s[0,75.44] during window+120s, their road+source delay at that endpoint.',
            '- C_G, D_G, sumD_i, raw branches/safety, all cells/coverage, paired CIs and numerical-null counts: tools/coopaudit/evidence/v2_analysis/.',
            '- Traces: gpuR /root/autodl-tmp/coopaudit20261101_v2/v2_sSEED/traces/*.jsonl.gz; members+nearby ramp vehicles,(270,390]s/0.5s.',
            '- Portable SHA-verified trace copies: evidence/v2_complete/jobs/; TRACE_MANIFEST.csv and tools/coopaudit/TRACE_HANDOFF.md.',
            '- Design AUDIT_DESIGN.md; upstream ARTIFACT_PASS372d3326/probe420a4904; CPU26/26PASS; canary commitd58db191.',
            '- Exposure disclosure: initial legacy parameter scan opened forbidden truth JSON; no values used; evidence/EXPOSURE_NOTE.md. No such reads in v2.',
            '- Interpret the conditional C_G CIs directly: fleet gain alone is not synergy; a fixed-pair null does not exclude cooperation in other groups/windows.']
    assert len(lines)<=60
    return '\n'.join(lines)+'\n'


def finish(expected_head):
    assert git('rev-parse','HEAD')==expected_head, 'source HEAD changed during observation'
    result=run('gpuR',(HERE/'tools/coopaudit/final_verify_host.py').read_text(),timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr)
    write(EVIDENCE/'V2_FINAL_VERIFIED.json',json.loads(result.stdout))
    from tools.coopaudit.collect_v2 import collect
    collect('complete')
    raw_commit=commit('S-COOPMEASURE v2 step 4 complete: preserve fixed paired panel and exact budget',
                      ['tools/coopaudit/evidence'])
    from tools.coopaudit.analyze_v2 import analyze
    summary=analyze()
    text=report(summary)
    target=HERE/'docs/paper_results/COOPAUDIT.md'
    target.write_text(text,encoding='utf-8',newline='\n')
    from tools.coopaudit.qa_v2 import main as qa
    qa()
    public=Path('E:/research-dense/docs/paper_results')
    assert public.is_dir()
    (public/'COOPAUDIT.md').write_bytes(target.read_bytes())
    for name in ('COOPAUDIT.csv','COOPAUDIT_worlds.csv','COOPAUDIT_branches.csv','COOPAUDIT_paired.csv','TRACE_MANIFEST.csv'):
        (public/name).write_bytes((EVIDENCE/'v2_analysis'/name).read_bytes())
    registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    tag='S-COOPMEASURE v2 COMPLETE (2026-11-01'
    assert tag not in registry.read_text(encoding='utf-8'), 'completion already registered'
    cg=[r for r in summary['summary'] if r['cohort']=='eligible21' and r['outcome']=='600' and r['metric']=='C_G']
    stats=';'.join(f"{r['controller']}CG600{r['mean']:+.6f}[{r['lo']:+.6f},{r['hi']:+.6f}]" for r in cg)
    line=(f'- {tag}; full-prefix paired audit):28TRAIN969650-677/24highshare,21eligible20highshare/7missing/3wallfeasible;'
          f'{stats}; conditionaleligible21/5000stratifiedworldbootstrapseed20261031,full28CIundefined_missingstrata/nozeroimputation.'
          ' MAPPOiter169+wallv2.1+fixedfinalSUMOCFu15e4b51f64/deployedmeans. NativeIDMno-command60srelease/native lanes;'
          ' everybranchfreshsame-seed300prefix+600outcome,231main/247totalprefixgatesPASS; CPU26PASS/native-noop-controlledrepeatsbitwise.'
          ' CanarywallCG600+1658.239758/solosharm,jointhelps;oneworldnotpanel.0collision/teleport/drop.'
          ' EXACT243480actual/243780charged<500000,300unusedreservationretained/noSUMOretries/refunds/deletion/peerPIDactions/push.'
          f' Rawcommit{raw_commit[:8]},SUMMARYsha{sha(EVIDENCE/"v2_analysis/SUMMARY.json")},QA PASS/report<=60lines docs/paper_results/COOPAUDIT.md;'
          ' traces gpuR /root/autodl-tmp/coopaudit20261101_v2/v2_sSEED/traces plus SHA-verified E:copy/TRACE_MANIFEST for S-COOPBEHAV.'
          ' Oldsaved-stateoutcomesexcluded/preserved; supervisorinactive/RUNlockfree/no continuation.\n')
    with registry.open('a',encoding='utf-8',newline='\n') as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    write(EVIDENCE/'V2_REGISTRATION_RECEIPT.json',dict(line=line,summary_sha256=sha(EVIDENCE/'v2_analysis/SUMMARY.json')))
    resume=HERE/'tools/coopaudit/RESUME.md'
    resume.write_text('# S-COOPMEASURE v2 COMPLETE — do not relaunch\n\n'
                      'All28 worlds are accounted for (21eligible/7missing), all three primary arms paired.\n'
                      'Actual243480/charged243780 of500000;300unused setup reservation retained.\n'
                      'Main supervisor exited, RUN lock free, no SUMO continuation is armed.\n'
                      'Report docs/paper_results/COOPAUDIT.md; raw receipts+traces evidence/v2_complete/.\n'
                      'World/branch/paired/bootstrap CSVs and trace manifest evidence/v2_analysis/.\n'
                      'V2_FINAL_VERIFIED.json + V2_QA.json verify processes, budget, hashes, algebra and CIs.\n'
                      'The old saved-state data/charges/commits remain preserved but excluded.\n'
                      'Reproduce analysis only with python -m tools.coopaudit.analyze_v2; no simulator work.\n',encoding='utf-8',newline='\n')
    write(EVIDENCE/'STATUS.json',dict(phase='COMPLETE',actual_sim_s=243480,charged_sim_s=243780,
          cap_sim_s=500000,worlds=28,eligible_worlds=21,missing_groups=7,cpu_tests_passed=26,
          qa='PASS',run_receipt_commit=raw_commit,report='docs/paper_results/COOPAUDIT.md',
          no_sumo_continuation=True,own_main_supervisor_inactive=True,unix=time.time()))
    write(EVIDENCE/'V2_FINALIZED.json',dict(ok=True,run_receipt_commit=raw_commit,
          summary_sha256=sha(EVIDENCE/'v2_analysis/SUMMARY.json'),qa_sha256=sha(EVIDENCE/'V2_QA.json')))
    write(EVIDENCE/'V2_FINISHER_PROGRESS.json',dict(phase='COMPLETE',unix=time.time()))
    return commit('S-COOPMEASURE v2 step 5: publish cooperation audit, paired CIs and trace handoff',
                  ['docs/paper_results/COOPAUDIT.md','tools/coopaudit/RESUME.md','tools/coopaudit/evidence'])


def main():
    lock=(EVIDENCE/'V2_FINISHER.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected_head=git('rev-parse','HEAD')
    write(EVIDENCE/'V2_FINISHER_PID.json',dict(pid=os.getpid(),unix=time.time(),source_commit=expected_head,
          command=[sys.executable,'-B','-m','tools.coopaudit.finalize_v2'],role='observe existing run; never dispatch SUMO'))
    deadline=time.time()+86400
    try:
        while time.time()<deadline:
            result=run('gpuR',PROBE,timeout=30)
            if result.returncode:
                write(EVIDENCE/'V2_FINISHER_PROGRESS.json',dict(phase='NETWORK_WAIT',stderr=result.stderr,unix=time.time()))
                time.sleep(60)
                continue
            status=json.loads(result.stdout)
            write(EVIDENCE/'V2_FINISHER_PROGRESS.json',dict(phase='OBSERVING_EXISTING_MAIN',unix=time.time(),**status))
            if status['stop']:
                raise ValueError('owned STOP: no continuation, retries or panel estimates')
            if status['complete']:
                print(json.dumps(dict(ok=True,report_commit=finish(expected_head))),flush=True)
                return
            time.sleep(60)
        raise TimeoutError('24h read-only observer limit; existing main unchanged')
    except BaseException:
        write(EVIDENCE/'V2_FINISHER_FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start',action='store_true')
    a=p.parse_args()
    if a.start:
        marker=EVIDENCE/'V2_FINISHER_ONCE.json'
        with marker.open('x',encoding='utf-8') as f:
            json.dump(dict(unix=time.time(),source_commit=git('rev-parse','HEAD')),f)
        log=(EVIDENCE/'V2_FINISHER.log').open('x')
        flags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS if os.name=='nt' else 0
        proc=subprocess.Popen([sys.executable,'-B','-m','tools.coopaudit.finalize_v2'],cwd=HERE,
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              creationflags=flags,start_new_session=os.name!='nt')
        print(json.dumps(dict(pid=proc.pid,role='read-only observer + verified final export/commit; no SUMO dispatch')))
    else:
        main()
