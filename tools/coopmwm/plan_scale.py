"""OWNER20:35 medium-CEM concurrency qualification; inference only, no SUMO."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
from .storage import read,write


def worker(root,count,index,connection,qualify):
    import os
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    import torch
    torch.set_num_threads(1)
    from tools.wmteach.runtime import bootstrap
    from .wm import cost_adapter
    from .plans import search,baseline_actions,family_seeds
    from .plan_wm import evaluate
    from .plan_service import configured_runtime
    rt=configured_runtime(bootstrap,min(.65,.80/count))
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'])
    source=read(root/'INPUTS.json')['records'][0]
    commands={r['vid']:0. for r in source['rows'] if r['is_cav']}
    record=dict(state_id=source['state_id'],world_id=source['world_id'],rows=source['rows'],group=source['group'],
        time_s=source['time_s'],commands=commands,policy_residuals={v:-.25 for v in commands})
    bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
    def ev(a,k,h,chunk=4):return evaluate(rt,record,a,k,h,chunk_actions=chunk,trim=False,pad_actions=4)
    if qualify:
        acts=baseline_actions(record)+family_seeds(record['rows'],record['group'],commands,bounds)[:2]
        for k,h in ((4,120),(8,300)):
            ref=ev(acts,k,h,1);batch=ev(acts,k,h)
            error=max(abs(a['team_delay']-b['team_delay']) for key in ref for a,b in zip(ref[key],batch[key]))
            item=dict(K=k,H=h,width=4,max_cost_error=error,pass_=error<=.001)
            write(root/f'FIXED4_K{k}_H{h}.json',item);assert item['pass_']
    connection.send(dict(event='ready',pid=os.getpid(),index=index))
    assert connection.recv()=='search'
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    result=search(record,bounds,ev);torch.cuda.synchronize()
    value=dict(index=index,pid=os.getpid(),wall_s=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),result=result)
    write(root/f'C{count}_W{index}.json',value);connection.send(value);connection.close()


def run(root,host):
    import fcntl
    import os
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    lock=(root/'SCALE.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=Path('/root/coopmwm20261101_planprobe3')
    owner=read(prior/'DISPATCH.json')
    while not (prior/'COMPLETE.json').exists():
        if (prior/'FAILED.json').exists():raise ValueError('original probe3 failed; preserve hold')
        p=Path('/proc')/str(owner['pid']);s=(p/'stat').read_text().split()
        assert s[21]==owner['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
        time.sleep(5)
    ctx=mp.get_context('spawn');levels=(1,2,4,8) if host=='gpuA' else (1,2,4)
    reference=None;measurements=[]
    for count in levels:
        inventory=subprocess.check_output(['nvidia-smi'],text=True)
        write(root/f'GPU_BEFORE_C{count}.json',dict(unix=time.time(),inventory=inventory))
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<count*4200:
            write(root/f'C{count}_NOT_ADMITTED.json',dict(free_MiB=free,required_MiB=count*4200,reason='memory headroom; no peer PID action'))
            continue
        processes=[];pipes=[]
        for i in range(count):
            parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(root,count,i,child,count==1))
            p.start();child.close();processes.append(p);pipes.append(parent)
        for p,c in zip(processes,pipes):
            while not c.poll(1):
                if p.exitcode is not None:raise ValueError(f'owned scale worker failed {p.pid}/{p.exitcode}')
            assert c.recv()['event']=='ready'
        start=time.monotonic()
        for c in pipes:c.send('search')
        results=[]
        for p,c in zip(processes,pipes):
            while not c.poll(1):
                if p.exitcode is not None:raise ValueError(f'owned scale worker failed {p.pid}/{p.exitcode}')
            value=c.recv();p.join();assert p.exitcode==0;c.close();results.append(value)
        wall=time.monotonic()-start
        if reference is None:reference=results[0]['result']
        # Identical source/record/CRN/search; each worker must reproduce the full
        # reference search, not merely the winning plan or rounded means.
        assert all(r['result']==reference for r in results),'cross-worker full-search parity failed'
        item=dict(pass_=True,host=host,workers=count,wall_s=wall,plans_per_hour=count*3600/wall,
            imagined_futures_per_hour=count*reference['imagined_futures']*3600/wall,
            imagined_futures=reference['imagined_futures'],peak_worker_allocated_bytes=max(r['peak_allocated_bytes'] for r in results),
            source_bound_full_search_parity=True,screen_width=4,final_width=4,new_sumo_sim_s=0,unix=time.time())
        write(root/f'CONCURRENCY_{count}.json',item);measurements.append(item);print(json.dumps(item),flush=True)
    assert measurements
    best=max(measurements,key=lambda v:(v['plans_per_hour'],-v['workers']))
    write(root/'COMPLETE.json',dict(pass_=True,host=host,best=best,measurements=measurements,new_sumo_sim_s=0))


def launch(host,cpu_name):
    from . import ops
    from .protocol import REGISTRY,checked,sha
    root='/root/coopmwm20261101_scale1';dest=ops.EVIDENCE/f'SCALE2035_{host}_DISPATCH.json'
    assert not dest.exists()
    cpu=read(ops.EVIDENCE/cpu_name);assert cpu['pass_'] and not cpu['skips']
    ops.stage_frozen(host,root,cpu)
    ops.stage(host,root,{'INPUTS.json':json.dumps(read(ops.EVIDENCE/'VOLUME_INPUTS_1854.json'))})
    line=(f'\nS-COOPMWM OWNER20:35 CONCURRENCY QUALIFICATION REGISTERED: {host}, '
        f'CPU{cpu_name}/SHA{sha(ops.EVIDENCE/cpu_name)}; medium32x3/elite8/K4H120/top8K8H300 unchanged; '
        'fixed4 batches full1600 roster; one source-bound fixed4 K4/K8 single-vs-batch parity, '
        'then simultaneous identical full-search CRN parity perworker. '
        f'Workers{[1,2,4,8] if host=="gpuA" else [1,2,4]}, >=4200MiB free perworker; '
        'best measured aggregate plans/hour selected, ties fewerworkers. Wait for originalprobe3 COMPLETE, '
        'never duplicate its running dispatch; inference only/noSUMO/nooutcome selection/nopeerPID action.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,os,subprocess,sys,time
r=pathlib.Path({root!r});assert r.is_dir();assert not (r/'DISPATCH.json').exists()
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'SCALE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.plan_scale','run','--root',str(r),'--host',{host!r}],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
value=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),host={host!r},root=str(r),new_sumo_sim_s=0)
with (r/'DISPATCH.json').open('x') as f:json.dump(value,f)
print(json.dumps(value))
''',timeout=30))
    write(dest,dict(value,registration=line,cpu=cpu_name));print(json.dumps(value))


def status(host):
    from . import ops
    d=read(ops.EVIDENCE/f'SCALE2035_{host}_DISPATCH.json')
    out=json.loads(ops.ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({d['root']!r});d=json.loads((r/'DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or (s[21]==d['start_ticks'] and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'])
receipts={{p.name:json.loads(p.read_text()) for pattern in ('CONCURRENCY_*.json','FIXED4_*.json','COMPLETE.json','FAILED.json','C*_NOT_ADMITTED.json') for p in r.glob(pattern)}}
print(json.dumps(dict(state=s[2] if s else 'exited',receipts=receipts,log=(r/'SCALE.log').read_text()[-4000:])))
''',timeout=25))
    for name,value in out['receipts'].items():
        p=ops.EVIDENCE/f'SCALE2035_{host}_{name}'
        if not p.exists():write(p,value)
    print(json.dumps(out));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','launch','status'));p.add_argument('--host',required=True);p.add_argument('--cpu');p.add_argument('--root',type=Path);a=p.parse_args()
    if a.mode=='launch':launch(a.host,a.cpu)
    elif a.mode=='status':status(a.host)
    else:
        try:run(a.root,a.host)
        except BaseException:
            import traceback
            write(a.root/'FAILED.json',dict(traceback=traceback.format_exc(),new_sumo_sim_s=0));raise
