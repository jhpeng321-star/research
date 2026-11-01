"""One-use release of CPU exact grading and already-running concurrent GPU owners."""
import argparse
import json
import time
from . import ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha
from .warm_deploy import ROOT

CPU='/home/ubuntu/coopmwm20261102_exactcpu3'
GPU='/root/coopmwm20261102_concurrent1'
ROUTES='/home/ubuntu/coopmwm20261102_remote_routes4'
GPU_R='/root/coopmwm20261102_concurrent2'


def authorize():
    from . import warm_remote_ops as w
    w.GPU=GPU;w.ROUTES=ROUTES
    w.authorize('EXACT_REMOTE_AUTHORIZATION_V2.json',roots={'gpuA':GPU,'gpuR':GPU_R})


def telemetry():
    for host,pids in [('gpuA','596572,588981'),('gpuR','39130,29230')]:
        value=json.loads(ops.ssh.remote(host,f'''import subprocess,json,time,os
print(json.dumps(dict(unix=time.time(),load=os.getloadavg(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True),processes=subprocess.run(['ps','-p',{pids!r},'-o','pid,ppid,stat,etime,time,pcpu,pmem,args'],capture_output=True,text=True).stdout)))''',timeout=30))
        write(ops.EVIDENCE/f'EXACT_TELEMETRY_{host}_{int(value["unix"])}.json',value);print(json.dumps(dict(host=host,**value)))


def retire_gpur():
    from . import warm_remote_ops as w
    from .concurrent_ops import status
    failed=status('gpuR');w.GPU='/root/coopmwm20261102_durable1';old=w.gpu_status('gpuR')
    assert failed['failed'] and failed['state']=='exited' and 'OutOfMemoryError' in failed['receipts']['SPOOL_FAILED.json']['error']
    assert old['alive'] and old['ready'] and not old['failed']
    owner=old['receipts']['SPOOL_OWNER.json']
    path=ops.EVIDENCE/'CONCURRENT_GPUR_MEMORY_AMENDMENT.json'
    value=dict(failed=failed,old=old,replacement_root=GPU_R,reason='old idle model occupies14.78GiB on24GiB host; graceful retirement then ONE same-source qualifier retry',new_sumo_sim_s=0)
    write(path,value);h=sha(path)
    line='\nS-COOPMWM GPU-R MEMORY ENGINEERING REGISTERED amendmentSHA'+h+': retire only verified original idle spool29230/start4218346888 via SPOOL_STOP, preserve receipts; failed concurrent1 OOM retained; one same-source concurrent2 qualification retry after original exits; unchanged search/H300/K8/memory flags; zero SUMO.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    print(ops.ssh.remote('gpuR',f'''import pathlib,json,sys
r=pathlib.Path('/root/coopmwm20261102_durable1');sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_spool import immutable
d=json.loads({json.dumps(owner)!r});assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
s=pathlib.Path('/proc')/str(d['pid'])/'stat';assert s.read_text().split()[21]==d['start_ticks']
for name in ('inbox','inference_claims','forecasts'):assert not list((r/name).glob('*.json')),'not idle'
immutable(r/'SPOOL_STOP.json',dict(amendment_sha256={h!r},reason='superseded idle model; free memory for concurrent qualified replacement'))
print(json.dumps(dict(graceful_stop_requested=True,pid=d['pid'],new_sumo_sim_s=0)))''',timeout=40))


def retry_gpur():
    from . import concurrent_ops as c
    from . import warm_remote_ops as w
    old=w.gpu_status('gpuR');assert not old['alive'],'wait for graceful exit'
    inventory=json.loads(ops.ssh.remote('gpuR',"import subprocess,json;p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True);print(json.dumps(dict(processes=p.stdout)))",timeout=30))
    assert not inventory['processes'].strip(),'inspect memory owners'
    stage=read(ops.EVIDENCE/'CONCURRENT_STAGE_gpuR.json')
    ops.stage_frozen('gpuR',GPU_R,stage['cpu'])
    inputs=ops.ssh.remote('gpuR',"import pathlib;print(pathlib.Path('/root/coopmwm20261102_concurrent1/INPUTS.json').read_text())",timeout=30)
    ops.stage('gpuR',GPU_R,{'INPUTS.json':inputs,'GPU_BINDING.json':json.dumps(stage['binding'],sort_keys=True)})
    # Source and all physics/search settings are byte-identical; only operational root differs.
    c.GPU=GPU_R
    original=c.read
    c.read=lambda p:dict(stage,root=GPU_R) if p.name=='CONCURRENT_STAGE_gpuR.json' else original(p)
    writer=c.write
    c.write=lambda p,v:writer(p.with_name('CONCURRENT_RETRY_DISPATCH_gpuR.json') if p.name=='CONCURRENT_DISPATCH_gpuR.json' else p,v)
    c.dispatch('gpuR')
    write(ops.EVIDENCE/'CONCURRENT_RETRY_gpuR.json',dict(root=GPU_R,inventory=inventory,new_sumo_sim_s=0))


def register():
    from .warm_protocol import budget
    from .resume_gate import check_budget
    from . import concurrent_ops as c
    check_budget(budget())
    target=ops.EVIDENCE/'EXACT_RUNTIME_REGISTRATION.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu3.json')
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    c.GPU=GPU;workers={'gpuA':c.status('gpuA')};c.GPU=GPU_R;workers['gpuR']=c.status('gpuR')
    assert all(v['alive'] and v['ready'] and not v['failed'] for v in workers.values()),'adopt running qualifications; no duplicate'
    for value in workers.values():
        gates=[r for name,r in value['receipts'].items() if name.startswith('STARTUP_')]
        assert len(gates)==8 and all(r['pass_'] and r['max_cost_error']==0. for r in gates)
    # CPU-only learner/controller/grade changes do not replace the running WM.
    numerical=('mppi_search.py','mppi_runtime.py','plan_wm.py','policy_wm.py','committed.py','overlap.py',
        'warm_runtime.py','warm_history.py','speed2_runtime.py','speed2_memory.py','speed2_transfers.py','plan_proposer.py')
    for host in workers:
        gpu=read(ops.EVIDENCE/f'CONCURRENT_STAGE_{host}.json')['binding']
        for name in numerical:
            key='pytools/tools/coopmwm/'+name
            assert gpu['hashes'][key]==cpu['source_hashes'][key],name
    endpoints=read(ops.EVIDENCE/'EXACT_REMOTE_AUTHORIZATION_V2.json')['endpoints']
    value=dict(cpu=cpu,workers=workers,endpoints=endpoints,budget=budget(),new_sumo_sim_s=0,
        learning='own-step PPO; critic all steps; only exact-positive full buffer imitation; weight1 until20 then frozen RMSfirst20 T,clip20,coefficient1',
        grades='marginal SUMO Dexecuted+120; all other concurrent plans replayed; prefix/CRN fault STOP; unresolved shared members excluded; no physics retries',
        library='same-B-arm/seed exact-positive complete plans only; partial/truncated plans have unlabelled tails and are excluded from search adaptation',
        supervision='detached B8/R8/grader8 plus originalA4; grading asynchronous and cannot block PPO; smoke is first actual successful B grade reused once',
        horizons='K4H300 ALWAYS; K8Hfinal=min(600,max(300,D+120)); registered MPPI unchanged; Nexec10',
        proposer='recording-only; activation requires offline check AND explicit OWNER approval in registry; R never',
        engineering='concurrent Controller preserves query draw metadata while old plans execute; grader replays all other groups and grades truncated portion',
        endpoints_fixed='A/B48,R24; TRAIN4/dev16; auto-continue under registered pauses; final96 only afterA/B48')
    path=ops.EVIDENCE/'EXACT_RUNTIME_AMENDMENT.json';write(path,value);h=sha(path)
    line='\nS-COOPMWM MAIN0303 EXACT-WARM CONCURRENT RELEASE REGISTERED amendmentSHA'+h+': '+json.dumps({k:v for k,v in value.items() if k not in ('cpu','workers','endpoints')},sort_keys=True)+'\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    operational=dict(health_guards=True,N_exec=10,workers=8,exact_grades=True)
    binding=dict(source_root=CPU,hashes=cpu['source_hashes'],cpu=cpu,amendment_sha256=h,operational=operational)
    broker=dict(endpoints=endpoints,deadline_unix=read(ops.EVIDENCE/'WARM_A48_BINDING.json')['gpu_deadline'],
        amendment_sha256=h,source_root=CPU,hashes=cpu['source_hashes'],search_sha256=cpu['source_hashes']['pytools/tools/coopmwm/mppi_search.py'])
    value.update(pass_=True,N_exec=10,release_amendment_sha256=h)
    reply=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running,safety_reference
from tools.coopmwm.accel_storage import read
assert_running(r,'B')
for a in ('B','R'):
 for s in ('','_V3','_V4'):assert not (r/f'WARM_{{a}}_DISPATCH{{s}}.json').exists()
assert not (r/'WARM_REMOTE_BROKER_DISPATCH.json').exists()
b=json.loads({json.dumps(binding)!r});b['operational']['safety_reference']=safety_reference(read(r/'MAPPO_CHECK.json')['records'])
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
for a in ('B','R'):immutable(r/f'WARM_{{a}}_RUNTIME_BINDING_V4.json',b)
immutable(r/'WARM_RELEASE.json',json.loads({json.dumps(value)!r}))
immutable(r/'WARM_REMOTE_BINDING.json',json.loads({json.dumps(broker)!r}))
original=read(r/'WARM_BINDING.json');immutable(r/'EXACT_OPERATIONAL_BINDING.json',original)
for name in ('centre_requests','centre_responses','broker_assignments','broker_completed'):(r/name).mkdir(exist_ok=True)
print('registered before B/R outcomes; no physics')''',timeout=45)
    write(target,dict(amendment_sha256=h,line=line,binding=binding,remote=reply,new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,amendment_sha256=h,tests=cpu['tests'],budget=budget())))


def dispatch(component):
    assert read(ops.EVIDENCE/'EXACT_RUNTIME_REGISTRATION.json')['amendment_sha256']
    modules={'B':('warm_runner',['--arm','B'],'WARM_B_DISPATCH_V4.json'),
        'R':('warm_runner',['--arm','R'],'WARM_R_DISPATCH_V4.json'),
        'grader':('exact_runtime',['--workers','8'],'EXACT_GRADER_DISPATCH.json'),
        'broker':('warm_broker',[],'WARM_REMOTE_BROKER_DISPATCH.json')}
    module,args,receipt=modules[component]
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,os,sys,subprocess,time,hashlib
r=pathlib.Path({ROOT!r});source=pathlib.Path({CPU!r});assert r.is_dir() and source.is_dir()
sys.path.insert(0,str(source/'pytools'))
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B');dest=r/{receipt!r};assert not dest.exists()
binding=json.loads((r/'WARM_B_RUNTIME_BINDING_V4.json').read_text())
for name,h in binding['hashes'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
if {component!r} in ('B','R'):
 from tools.coopmwm.warm_runner import verify
 verify(r,{component!r})
 from tools.coopmwm.exact_runtime import assert_live
 assert_live(r)
 assert (r/'WARM_REMOTE_BROKER_DISPATCH.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PYTHONPATH=str(source/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
cmd=['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m',{'tools.coopmwm.'+module!r},str(r)]+{args!r}
with (r/{('EXACT_'+component+'.log')!r}).open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(source),root=str(r),component={component!r})
with dest.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=45))
    write(ops.EVIDENCE/receipt,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('authorize','register','dispatch','retire-gpur','retry-gpur','telemetry'));p.add_argument('--component',choices=('B','R','grader','broker'))
    a=p.parse_args();dispatch(a.component) if a.mode=='dispatch' else globals()[a.mode.replace('-','_')]()
