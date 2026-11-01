"""Explicit durable warm release engineering; no implicit training dispatch."""
import argparse
import json
import subprocess
import time
from . import ops
from .storage import read,write
from .protocol import checked,sha,REGISTRY
from .warm_deploy import ROOT

CPU='/home/ubuntu/coopmwm20261102_durablecpu2'
BROKER='/home/ubuntu/coopmwm20261102_brokercpu3'
GPU='/root/coopmwm20261102_durable1'
ROUTES='/home/ubuntu/coopmwm20261102_remote_routes2'


def stage_gpu(host):
    from .warm_release_ops import collect
    data=collect()[host]
    assert data['state'] in ('exited','Z') and data['receipts']['COMPLETE.json']['pass_']
    inventory=json.loads(ops.ssh.remote(host,"""import subprocess,json
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not p.stdout.strip(),'GPU already has a compute process; inspect before admission'
print(json.dumps(dict(compute_processes=p.stdout,pass_=True)))""",timeout=30))
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_durablecpu2.json');assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    staged=json.loads(ops.stage_frozen(host,GPU,cpu))
    extra={'INPUTS.json':ops.ssh.remote(host,"import pathlib;print(pathlib.Path('/root/coopmwm20261101_planprobe3/INPUTS.json').read_text())",timeout=30)}
    additional=json.loads(ops.stage(host,GPU,extra));hashes=dict(staged['hashes'],**additional['hashes'])
    binding=dict(hashes=hashes,qualification=dict(pass_=True,receipts=data['receipts']),
        deadline_unix=read(ops.EVIDENCE/'WARM_A48_BINDING.json')['gpu_deadline'],
        memory_fraction=.80 if host=='gpuR' else .65,memory_mode='warm',startup_qualification=True,
        speed2=True,speed2_transfers=True,batch_width_by_K={'4':18 if host=='gpuR' else 34,'8':10},expandable_segments=host=='gpuR')
    ops.stage(host,GPU,{'GPU_BINDING.json':json.dumps(binding,sort_keys=True)})
    receipt=dict(host=host,root=GPU,binding=binding,cpu=cpu,inventory=inventory,new_sumo_sim_s=0)
    write(ops.EVIDENCE/f'WARM_DURABLE_STAGE_{host}.json',receipt)
    line=('\nS-COOPMWM DURABLE GPU SOURCE ADMISSION '+host+' receiptSHA'+sha(ops.EVIDENCE/f'WARM_DURABLE_STAGE_{host}.json')+
        ': detached disk-spool worker; CPU160PASS/zeroSUMO; unchanged warm FULLv4/speed2+transfers and registered MPPI search, '
        'new centre-only request router uses same helper/no candidate scores. Six source-bound exact startup gates before READY; '
        'owned original qual4 completed/adopted; GPU compute inventory empty. No SUMO dispatch; original deadline retained.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    print(json.dumps(dict(host=host,root=GPU,staged=True,new_sumo_sim_s=0)))


def dispatch_gpu(host):
    stage=read(ops.EVIDENCE/f'WARM_DURABLE_STAGE_{host}.json');assert stage['root']==GPU
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,subprocess,os,time
r=pathlib.Path({GPU!r});assert r.is_dir() and not (r/'DISPATCH.json').exists() and not (r/'SPOOL_OWNER.json').exists()
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
if {host!r}=='gpuR':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
cmd=['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.warm_spool',str(r)]
with (r/'SPOOL.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),root=str(r),host={host!r})
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=35))
    write(ops.EVIDENCE/f'WARM_DURABLE_DISPATCH_{host}.json',value);print(json.dumps(value))


def gpu_status(host):
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,sys
r=pathlib.Path({GPU!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_rpc import status
v=status(r);v['host']={host!r}
if v['failed']:v['log_tail']=(r/'SPOOL.log').read_text()[-5000:]
print(json.dumps(v))''',timeout=30))
    write(ops.EVIDENCE/f'WARM_DURABLE_STATUS_{host}_{int(time.time())}.json',value)
    print(json.dumps(value));return value


def authorize():
    """Task-only restricted public key; private material remains on cpuQ."""
    public=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,subprocess,json,os
r=pathlib.Path({ROUTES!r});assert r.parent.is_dir();r.mkdir(exist_ok=True);r.chmod(0o700)
p=r/'id_ed25519'
if not p.exists():subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-C','coopmwm-durable-rpc','-f',str(p)],check=True)
assert p.is_file() and p.with_suffix('.pub').is_file();p.chmod(0o600)
print(json.dumps(dict(public=p.with_suffix('.pub').read_text().strip())))''',timeout=35))['public']
    endpoints=[]
    for host in ('gpuA','gpuR'):
        config=dict(line.split(' ',1) for line in subprocess.check_output(['ssh','-G',host],text=True).splitlines() if ' ' in line)
        hostname=config['hostname'];port=int(config['port'])
        known=subprocess.check_output(['ssh-keygen','-F',f'[{hostname}]:{port}'],text=True);assert known.strip()
        ops.stage('cpuQ',ROUTES,{'known_hosts_'+host:known})
        command=f'env PYTHONPATH={GPU}/pytools /root/miniconda3/bin/python -B -m tools.coopmwm.warm_rpc {GPU}'
        entry='restrict,command="'+command+'" '+public+'\n'
        receipt=json.loads(ops.ssh.remote(host,f'''import pathlib,json,hashlib
r=pathlib.Path({GPU!r});assert (r/'GPU_BINDING.json').is_file()
p=pathlib.Path('/root/.ssh/authorized_keys');assert p.parent.is_dir()
entry={entry!r}
if not p.exists() or entry not in p.read_text():
  with p.open('a') as f:f.write(chr(10)+entry)
p.chmod(0o600)
print(json.dumps(dict(registered=True,public_key_sha256=hashlib.sha256({public!r}.encode()).hexdigest(),forced_root=str(r))))''',timeout=35))
        endpoints.append(dict(slot=host+'@0',destination=config['user']+'@'+hostname,port=port,
            identity_file=ROUTES+'/id_ed25519',known_hosts=ROUTES+'/known_hosts_'+host,authorization=receipt))
    ops.stage('cpuQ',ROUTES,{'ENDPOINTS.json':json.dumps(endpoints,sort_keys=True)})
    write(ops.EVIDENCE/'WARM_REMOTE_AUTHORIZATION.json',dict(endpoints=endpoints,private_material_copied=False,new_sumo_sim_s=0))
    print(json.dumps(dict(authorized=[e['slot'] for e in endpoints],private_material_copied=False)))


def probe_routes():
    """Real read-only RPC via the remote-only key; no forecast/physics."""
    endpoints=read(ops.EVIDENCE/'WARM_REMOTE_AUTHORIZATION.json')['endpoints']
    result=json.loads(ops.ssh.remote('cpuQ',f'''import sys,json,pathlib
sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.warm_broker import Remote
values={{}}
for e in json.loads({json.dumps(endpoints)!r}):
 remote=Remote({ROOT!r},e);v=remote.raw(dict(action='status'))
 values[e['slot']]=dict(alive=v['alive'],ready=v['ready'],failed=v['failed'],owner=v['receipts'].get('SPOOL_OWNER.json'),gates=list(v['receipts']))
print(json.dumps(values))''',timeout=60))
    write(ops.EVIDENCE/f'WARM_DURABLE_ROUTE_PROBE_{int(time.time())}.json',result);print(json.dumps(result));return result


def register_runtime():
    from .warm_protocol import budget
    from .resume_gate import check_budget
    check_budget(budget())
    target=ops.EVIDENCE/'WARM_DURABLE_RUNTIME_REGISTRATION.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_durablecpu2.json');assert cpu['pass_'] and cpu['tests']==160
    metadata=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.accel_storage import read,ledger,charged
from tools.coopmwm.accel_learner import load
from tools.coopmwm.warm_health import safety_reference
old=read(r/'WARM_A_DISPATCH.json');assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
p=pathlib.Path('/proc')/str(old['pid']);assert not (p/'stat').exists(),'A still exists; inspect rather than duplicate'
for arm in ('B','R'):assert not (r/f'WARM_{{arm}}_DISPATCH.json').exists() and not (r/f'WARM_{{arm}}_STARTED.json').exists()
warm=[c for c in ledger(r) if c['id'].startswith(('warm_','warmcheck','warmdev'))]
assert all((r/'jobs'/c['id']/'CLOSE.json').exists() for c in warm),'unfinished physics reservations'
policy,opt,rng,payload=load(r/'A_checkpoint_016.pt');assert payload['update']==16 and opt.state_dict()['state']
assert read(r/'A_UPDATE_016.json')['update']==16
print(json.dumps(dict(A_checkpoint_sha256=hashlib.sha256((r/'A_checkpoint_016.pt').read_bytes()).hexdigest(),A_resume_update=16,
 charged_sim_s=charged(r),closed_jobs=len(warm),original_stop_sha256=hashlib.sha256((r/'STOP.json').read_bytes()).hexdigest(),
 original_stop=read(r/'STOP.json'),safety_reference=safety_reference(read(r/'MAPPO_CHECK.json')['records']),
 pending_check16=not (r/'A_CHECK_016.json').exists(),pending_dev16=not (r/'A_DEV_016.json').exists())))''',timeout=45))
    assert metadata['original_stop']==dict(error='Connection closed by 1.15.42.207 port 22\n',source='owned warm bridge; no automatic retry')
    value=dict(cpu=cpu,source_root=CPU,metadata=metadata,authorization=read(ops.EVIDENCE/'WARM_REMOTE_AUTHORIZATION.json'),
        budget=budget(),AWR=read(ops.EVIDENCE/'MAIN0201_AWR_REGISTRATION.json'),
        transport='detached GPU spool and cpuQ supervisors/broker; same-input deliveries only, five bounded connection-only retries with original worker/gate reverification; no inference/physics retry',
        health='MAIN01:38: collision/teleport increase immediate; paired hard-brake95CI above A; >=30 executed and realized-advantageCI upper<0; two consecutive adverse TRAIN checks; starvation two updates or all trust0 beforeu8; pause B/R, A unaffected',
        new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM_DURABLE_RUNTIME_AMENDMENT.json',value);h=sha(ops.EVIDENCE/'WARM_DURABLE_RUNTIME_AMENDMENT.json')
    line=('\nS-COOPMWM MAIN01:29/01:33/01:38 DURABLE RUNTIME REGISTERED amendmentSHA'+h+
        ': exact original transportSTOP released by SHA only, preserved on disk; CPU160PASS/0skips/zeroSUMO. '
        'A resumes exactu16/Adam/RNG, missingu16 check/dev adopt completed jobs once. B/R remain MAPPOu0 until full GPU release. '
        'cpuQ remote detached supervisors and broker; GPU detached persistent spools, immutable claims/results, restricted task-root RPC key generated and retained only cpuQ. '
        'Bounded explicit SSH-connection-only redelivery5attempts/300s, owner/gates reverified, no physics/inference replay. '
        'Registered MAIN01:38 safety/harmful-plan/learning/starvation pauses integrated; B/R pause leaves A running; no final-panel access. '
        'MAPPO TRAIN-check zero-event safety reference, unmatched TRAIN conservative zero-event pause disclosed; paired whole-world hardbrake bootstrap vs same-update A. '
        'MAIN02:01 own PPO/positive-completed-plan AWR binding enforced. Same frozen warm MPPI search; Nexec10/u4 viability pause unchanged. '
        'Full6473440/cap6500000/free26560; all prior charges retained.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    binding=dict(source_root=CPU,hashes=cpu['source_hashes'],cpu=cpu,amendment_sha256=h,
        operational=dict(A_resume_update=16,health_guards=True,safety_reference=metadata['safety_reference']))
    release=dict(original_stop_sha256=metadata['original_stop_sha256'],amendment_sha256=h)
    ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'WARM_TRANSPORT_RELEASE.json',json.loads({json.dumps(release)!r}))
for arm in ('A','B','R'):immutable(r/f'WARM_{{arm}}_RUNTIME_BINDING_V3.json',json.loads({json.dumps(binding)!r}))
immutable(r/'WORKER_ADMISSION_V3.json',dict(workers=['gpuA@0','gpuR@0']))
for name in ('centre_requests','centre_responses','broker_assignments','broker_completed'):(r/name).mkdir(exist_ok=True)
print('registered; no physics')''',timeout=40)
    write(target,dict(line=line,binding=binding,amendment_sha256=h,metadata=metadata,new_sumo_sim_s=0))
    print(json.dumps(dict(registered=True,amendment_sha256=h,metadata=metadata)))


def dispatch_arm(arm):
    assert arm in ('A','B','R')
    registration=read(ops.EVIDENCE/'WARM_DURABLE_RUNTIME_REGISTRATION.json')
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,subprocess,os,time,sys
r=pathlib.Path({ROOT!r});source=pathlib.Path({CPU!r});assert r.is_dir() and source.is_dir()
sys.path.insert(0,str(source/'pytools'))
from tools.coopmwm.warm_runner import verify
b=verify(r,{arm!r});assert b['health_guards'] and b['A_resume_update']==16
p=r/{('WARM_'+arm+'_DISPATCH_V3.json')!r};assert not p.exists()
if {arm!r} in ('B','R'):assert not (r/{('WARM_'+arm+'_DISPATCH.json')!r}).exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PYTHONPATH=str(source/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];assert pathlib.Path(env['SUMO_HOME'],'bin/sumo').is_file()
cmd=[{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.warm_runner',str(r),'--arm',{arm!r}]
with (r/{('WARM_'+arm+'_V3.log')!r}).open('x') as log:
 child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path(f'/proc/{{child.pid}}/stat').read_text().split()
v=dict(pid=child.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),arm={arm!r},root=str(r),source=str(source),unix=time.time())
with p.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=45))
    write(ops.EVIDENCE/f'WARM_{arm}_DISPATCH_V3.json',value);print(json.dumps(value))


def status():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,shutil,os
r=pathlib.Path({ROOT!r});sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.accel_storage import read,ledger,charged
from tools.coopmwm.warm_health import active_stop
owners={{}}
for name in ['WARM_REMOTE_BROKER_DISPATCH.json']+[f'WARM_{{a}}_DISPATCH_V3.json' for a in ('A','B','R')]:
 p=r/name
 if not p.exists():continue
 d=read(p);assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 q=pathlib.Path('/proc')/str(d['pid']);s=(q/'stat').read_text().split() if (q/'stat').exists() else None
 assert not s or s[21]==d['start_ticks'];owners[name]=dict(owner=d,state=s[2] if s else 'exited')
receipts={{p.name:read(p) for pattern in ('PAUSE_*.json','WARM_*_FAILED_V3.json','WARM_GUIDED_STOP.json','WARM_*_PAUSED.json') for p in r.glob(pattern)}}
logs={{a:(r/f'WARM_{{a}}_V3.log').read_text()[-4000:] for a in ('A','B','R') if (r/f'WARM_{{a}}_FAILED_V3.json').exists()}}
charges=ledger(r);fresh=[c for c in charges if c['id'].startswith(('warm_','warmcheck','warmdev'))]
print(json.dumps(dict(unix=time.time(),owners=owners,active_transport_stop=active_stop(r),receipts=receipts,failed_logs=logs,
 updates={{a:max([0]+[int(p.stem.rsplit('_',1)[1]) for p in r.glob(a+'_UPDATE_*.json')]) for a in ('A','B','R')}},
 charged_sim_s=charged(r),reserved_jobs=len(fresh),closed_jobs=sum((r/'jobs'/c['id']/'CLOSE.json').exists() for c in fresh),
 pending_requests={{kind:sum(not (r/responses/p.name).exists() for p in (r/requests).glob('*.json')) for kind,requests,responses in [('plan','requests','responses'),('centre','centre_requests','centre_responses')]}},
 completed_forecasts=len(list((r/'broker_completed').glob('*.json'))),load=os.getloadavg(),disk_free=shutil.disk_usage(r).free,ram_free=shutil.disk_usage('/dev/shm').free)))''',timeout=40))
    write(ops.EVIDENCE/f'WARM_V3_STATUS_{int(value["unix"])}.json',value);print(json.dumps(value));return value


def release():
    from .warm_release_ops import collect
    from .warm_report import summary,publish
    from .warm_protocol import budget
    from .resume_gate import check_budget
    check_budget(budget())
    target=ops.EVIDENCE/'WARM_DURABLE_RELEASE_REGISTRATION.json';assert not target.exists()
    workers={h:gpu_status(h) for h in ('gpuA','gpuR')}
    assert all(w['alive'] and w['ready'] and not w['failed'] for w in workers.values()),'startup gates incomplete; adopt running workers'
    routes=probe_routes();assert all(v['alive'] and v['ready'] and not v['failed'] for v in routes.values())
    data=collect();assert all(v['receipts']['COMPLETE.json']['pass_'] for v in data.values())
    value=summary(data);runtime=read(ops.EVIDENCE/'WARM_DURABLE_RUNTIME_REGISTRATION.json')
    broker_cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_brokercpu3.json')
    assert broker_cpu['pass_'] and broker_cpu['tests']==161 and broker_cpu['skips']==broker_cpu['sim_s']==0
    value.update(pass_=True,N_exec=10,workers=workers,routes=routes,cpu_runtime=runtime,broker_cpu=broker_cpu,budget=budget(),
        AWR=read(ops.EVIDENCE/'MAIN0201_AWR_REGISTRATION.json'),
        quality_choice='full registered dimension-scaled medium MPPI retained; Nexec10, group-size benchmarks on different hosts/states; no acceptance-based ETA until first update',
        two_device_capacity_floor_hours_per_10_queries=[10/(2*max(v['plans_per_hour'] for v in value['timing'].values())),10/(2*min(v['plans_per_hour'] for v in value['timing'].values()))],
        R='same warm current-policy centre; no WM candidate scores/library/covariance learning; centre queue after B-u4 pause',
        original_deadline_unix=read(ops.EVIDENCE/'WARM_A48_BINDING.json')['gpu_deadline'])
    write(ops.EVIDENCE/'WARM_DURABLE_RELEASE_AMENDMENT.json',value);h=sha(ops.EVIDENCE/'WARM_DURABLE_RELEASE_AMENDMENT.json')
    line=('\nS-COOPMWM WARM MEMORY QUALIFIED / DURABLE B-R RESTART REGISTERED amendmentSHA'+h+
        ': CPU160PASS/0skips/zeroSUMO; rolling60s causal observable per-driver FULLv4 inference/shrinkage; emptyhistory==cold and '
        'both-host warm K4H120/K8H420 full-fast EXACT0, persistent workers six startup gates EXACT0 including warm batched-single. '
        'Original E2 cold/warm exact-labelled diagnostic and completed group2/4/6 timings adopted, no outcome-driven search change. '
        'Full discovery-pure policy-centred medium MPPI dimension-scaled population3iterations, top4 Thompson/K8 Hfinal=min(600,max(300,D+120)), '
        'groups2..6/knots<=10/D<=300; same-arm/seed SUMO-positive library/covariance only; Nexec10; bothGPU saturated, R count/regime/size/duration matched. '
        'MAIN0201 own-action PPO/positive completed-plan AWR/allstep critic; MAIN0138 health pauses enforced. '
        'Detached supervisors/broker/GPU spools, bounded SSH-only identical delivery retries, no physics/inference replay. '
        'Broker-onlyCPU3/161PASS additionally recognizes observed Read-from-remote-host Connection-reset and supports registered same-search endpoint hot-add; learner/GPU remain immutableCPU2. '
        'A exactu16 continuation; B/R MAPPOu0, firstu4 viability pause; R centre helper after B-u4, zero candidate scores. '
        'Full6473440/cap6500000/free26560; final panel unopened; fixed finalsA/B48,R24 and secondaryu24 insurance unchanged.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    value['release_amendment_sha256']=h
    endpoints=read(ops.EVIDENCE/'WARM_REMOTE_AUTHORIZATION.json')['endpoints']
    broker_binding=dict(endpoints=endpoints,deadline_unix=value['original_deadline_unix'],amendment_sha256=h,source_root=BROKER,
        hashes=broker_cpu['source_hashes'],search_sha256=runtime['binding']['hashes']['pytools/tools/coopmwm/mppi_search.py'])
    ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});sys.path.insert(0,{CPU!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B')
for arm in ('B','R'):assert not (r/f'WARM_{{arm}}_DISPATCH_V3.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'WARM_RELEASE.json',json.loads({json.dumps(value)!r}))
immutable(r/'WARM_REMOTE_BINDING.json',json.loads({json.dumps(broker_binding)!r}))
print('registered before B/R physics')''',timeout=45)
    publish(value,qualified=True)
    write(target,dict(line=line,amendment_sha256=h,new_sumo_sim_s=0));print(line.strip())


def dispatch_broker():
    assert read(ops.EVIDENCE/'WARM_DURABLE_RELEASE_REGISTRATION.json')['amendment_sha256']
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,os,subprocess,time,sys,hashlib
r=pathlib.Path({ROOT!r});assert r.is_dir() and (r/'WARM_REMOTE_BINDING.json').is_file()
assert not (r/'WARM_REMOTE_BROKER_DISPATCH.json').exists() and not (r/'WARM_REMOTE_BROKER_OWNER.json').exists()
b=json.loads((r/'WARM_REMOTE_BINDING.json').read_text());assert b['source_root']=={BROKER!r}
for name,h in b['hashes'].items():assert hashlib.sha256((pathlib.Path({BROKER!r})/name).read_bytes()).hexdigest()==h,name
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH={BROKER!r}+'/pytools',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
cmd=['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.warm_broker',str(r)]
with (r/'WARM_REMOTE_BROKER.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()
value=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),root=str(r))
with (r/'WARM_REMOTE_BROKER_DISPATCH.json').open('x') as f:json.dump(value,f)
print(json.dumps(value))''',timeout=40))
    write(ops.EVIDENCE/'WARM_REMOTE_BROKER_DISPATCH.json',value);print(json.dumps(value))


def routes():
    known=subprocess.check_output(['ssh-keygen','-F','[connect.bjb1.seetacloud.com]:11000'],text=True)
    assert known.strip()
    root='/home/ubuntu/coopmwm20261102_remote_routes1'
    ops.stage('cpuQ',root,{'known_hosts_gpuA':known})
    commands={
        'cpuQ_gpuA':['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
            '-o','UserKnownHostsFile='+root+'/known_hosts_gpuA',
            '-p','11000','root@connect.bjb1.seetacloud.com','true'],
    }
    result={}
    for name,command in commands.items():
        value=json.loads(ops.ssh.remote(name.split('_')[0],f'''import subprocess,json
p=subprocess.run({command!r},capture_output=True,text=True,timeout=20)
print(json.dumps(dict(code=p.returncode,stderr=p.stderr)))''',timeout=35))
        result[name]=value
    write(ops.EVIDENCE/f'WARM_REMOTE_ROUTES_{int(time.time())}.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('routes','stage-gpu','dispatch-gpu','gpu-status','authorize','probe-routes','register-runtime','dispatch-arm','status','release','dispatch-broker'));p.add_argument('--host',choices=('gpuA','gpuR'));p.add_argument('--arm',choices=('A','B','R'));a=p.parse_args()
    if a.mode=='dispatch-arm':dispatch_arm(a.arm)
    elif a.mode in ('routes','authorize','probe-routes','register-runtime','status','release','dispatch-broker'):globals()[a.mode.replace('-','_')]()
    else:globals()[a.mode.replace('-','_')](a.host)
