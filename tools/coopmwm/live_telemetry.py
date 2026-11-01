"""TRAIN progress/utilisation observer only; no job submission or outcome opening.

The existing exact_watch owns status/learning-check publication. This companion
only records completed-search rates, admission progress and opaque final status.
"""
import argparse
import json
import subprocess
import sys
import time
from . import ops,exact_ops,hotadd_ops,final_ops
from .storage import read,write

TAG='LIVE_TELEMETRY'


def summarize(progress,monitors,dispatches):
    rates={}
    for q in progress['queries']:
        if q.get('superseded_engineering'):continue
        host=q['slot'].split('@')[0]
        pending_since=dispatches[host]['unix']
        admission=monitors['monitors'][host]['receipts'].get('WARM_REMOTE_ENDPOINT_ADMISSION_'+host+'_SECOND.json')
        admitted_at=admission['unix'] if admission else None
        start=q['unix']-q['inference_wall_s']
        if q['unix']<=pending_since:epoch='one_worker_before_qualification'
        elif admitted_at is not None and start>=admitted_at:epoch='two_workers_admitted'
        else:epoch='qualification_or_boundary_overlap'
        key=host+'|'+epoch+'|n'+str(q['group_size'])
        g=rates.setdefault(key,dict(queries=0,accepted=0,inference_worker_s=0.,
            imagined_futures=0,begin=start,end=q['unix']))
        g['queries']+=1;g['accepted']+=bool(q['accepted'])
        g['inference_worker_s']+=q['inference_wall_s']
        g['imagined_futures']+=q.get('futures') or 0
        g['begin']=min(g['begin'],start);g['end']=max(g['end'],q['unix'])
    for g in rates.values():
        elapsed=g['end']-g['begin']
        g['queries_per_inference_worker_hour']=g['queries']*3600/g['inference_worker_s']
        g['completed_queries_per_observed_span_hour']=g['queries']*3600/elapsed
        g['imagined_futures_per_observed_span_hour']=g['imagined_futures']*3600/elapsed
    return dict(unix=progress['unix'],rates=rates,executed_starts=len(progress['executed_starts']),
        grader=progress['grader'],charged_sim_s=progress['charged_sim_s'],
        interpretation='descriptive completed-search observations, not steady-state capacity or executed-plan benefit; span uses delivery completion minus inference wall; qualification-overlap separate; unfinished searches censored',
        outcomes_opened=False,new_sumo_sim_s=0)


def once():
    progress=exact_ops.progress()
    monitors=hotadd_ops.monitor_status()
    dispatches={h:read(ops.EVIDENCE/f'SECOND_DISPATCH_{h}.json') for h in ('gpuA','gpuR')}
    value=summarize(progress,monitors,dispatches)
    write(ops.EVIDENCE/f'LIVE_THROUGHPUT_{int(value["unix"])}.json',value)
    exact_ops.telemetry()
    final_ops.status()
    print(json.dumps(value),flush=True)
    return value


def run():
    import msvcrt
    from .warm_transport import retry
    lock=(ops.EVIDENCE/(TAG+'.claim')).open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    deadline=read(ops.EVIDENCE/(TAG+'_BINDING.json'))['deadline_unix']
    def verify():
        g,c=exact_ops.status()
        assert not c['stops'],'registered pause; no automatic resumption'
        for h,v in g.items():assert v['alive'] and v['ready'] and not v['failed'],h
    def event(v):write(ops.EVIDENCE/f'LIVE_TELEMETRY_TRANSPORT_{time.time_ns()}.json',v)
    try:
        while time.time()<deadline:
            retry(once,verify,event)
            # Publishing/committing these small metadata receipts is independent
            # of the original learning-check observer's status/publish loop.
            from .warm_v3_watch import commit
            paths=[str(p.relative_to(ops.LOCAL)) for pattern in ('LIVE_*.json','SECOND_*STATUS*.json','FINAL_PRECOMPUTE_STATUS_*.json','EXACT_PROGRESS_*.json','EXACT_TELEMETRY_*.json') for p in ops.EVIDENCE.glob(pattern)]
            try:commit(paths)
            except ValueError as error:
                if str(error)!='other staged work; preserve it':raise
            time.sleep(300)
    except BaseException as error:
        write(ops.EVIDENCE/f'LIVE_TELEMETRY_EXIT_{int(time.time())}.json',dict(
            unix=time.time(),error=repr(error),training_survives=True,physics_retry=False))
        raise
    finally:lock.close()


def dispatch():
    from .protocol import checked,REGISTRY,sha
    target=ops.EVIDENCE/(TAG+'_DISPATCH.json');assert not target.exists()
    binding=dict(deadline_unix=read(ops.EVIDENCE/'WARM_A48_BINDING.json')['gpu_deadline'],
        source_sha256=sha(ops.LOCAL/'tools/coopmwm/live_telemetry.py'),tests=4 if TAG.endswith('_V2') else 3,
        contract='TRAIN-only rate/utilisation, second-worker admission, hash-only final status; observation only, no policy/search/physics changes')
    write(ops.EVIDENCE/(TAG+'_BINDING.json'),binding)
    line='\nS-COOPMWM LIVE TELEMETRY OBSERVER REGISTERED '+TAG+' bindingSHA'+sha(ops.EVIDENCE/(TAG+'_BINDING.json'))+': disjoint completed-TRAIN-search rate/GPU utilisation/admission/opaque final-status observer at300s; CPU'+str(binding['tests'])+'PASS, zeroSUMO, no submission/selection/activation, originaldeadline retained; qualification-overlap and incomplete searches disclosed. V2 excludes superseded engineering queries from restart rates and retires only prior observer6216 after PID/creation/command verification; original receipts retained. Existing learning-check observer retained.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    if TAG=='LIVE_TELEMETRY_V2':
        # Only the receipt-owned local observer is retired; remote owners are
        # independent. Match PID, command and creation time before signalling.
        old=read(ops.EVIDENCE/'LIVE_TELEMETRY_DISPATCH.json')
        code=(f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {old['pid']}'; "
            "if ($null -ne $p) { "
            "if ($p.CommandLine -notlike '*tools.coopmwm.live_telemetry run*') { throw 'observer command changed' }; "
            f"$t=([DateTimeOffset]$p.CreationDate).ToUnixTimeMilliseconds()/1000; if ([Math]::Abs($t-{old['unix']}) -gt 2) {{ throw 'observer PID reused' }}; "
            f"Stop-Process -Id {old['pid']} -ErrorAction Stop }}")
        subprocess.run(['powershell','-NoProfile','-Command',code],capture_output=True,text=True,check=True)
    with (ops.EVIDENCE/(TAG+'.log')).open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.live_telemetry','run','--tag',TAG],
            cwd=ops.LOCAL,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    value=dict(pid=p.pid,unix=time.time(),binding=binding);write(target,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('once','run','dispatch'))
    p.add_argument('--tag',choices=('LIVE_TELEMETRY','LIVE_TELEMETRY_V2'),default=TAG)
    a=p.parse_args();TAG=a.tag;globals()[a.mode]()
