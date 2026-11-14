"""Explicit MAIN04:19 staging, registration and one-use second-worker dispatch."""
import argparse
import json
import time
from . import ops,exact_ops
from .storage import read,write
from .protocol import checked,REGISTRY,sha

ROOTS={'gpuA':'/root/coopmwm20261102_second1','gpuR':'/root/coopmwm20261102_second1'}
ORIGINAL={'gpuA':exact_ops.GPU,'gpuR':exact_ops.GPU_R}
MONITOR='/home/ubuntu/coopmwm20261102_hotadd1'
ROUTES='/home/ubuntu/coopmwm20261102_second_routes1'


def cpu():
    original=read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu5.json')
    ops.stage_frozen('cpuQ',MONITOR,original)
    files={'tools/coopmwm/'+name:checked(ops.LOCAL/'tools/coopmwm'/name).read_text(encoding='utf-8')
        for name in ('hotadd.py','hotadd_rpc.py','test_hotadd_cpu.py')}
    staged=json.loads(ops.stage('cpuQ',MONITOR,files))
    out=ops.ssh.remote('cpuQ',f'''import os,sys,time,unittest,json
os.nice(19);os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,{MONITOR!r}+'/pytools')
from unittest.mock import patch
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+name) for name in ('test_hotadd_cpu','test_warm_transport_cpu','test_warm_spool_cpu','test_warm_broker_cpu'))
begin=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('SUMO forbidden during hot-add CPU tests')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),wall_s=time.monotonic()-begin,sim_s=0)))''',timeout=60)
    print(out);value=json.loads(out.strip().splitlines()[-1]);value.update(staged=staged)
    write(ops.EVIDENCE/'SECOND_CPU_QUALIFICATION.json',value)
    assert value['pass_'] and value['tests']==26 and value['skips']==0


def stage(host):
    from .hotadd import binding
    target=ops.EVIDENCE/f'SECOND_STAGE_{host}.json';assert not target.exists()
    assert read(ops.EVIDENCE/'SECOND_CPU_QUALIFICATION.json')['pass_']
    original=json.loads(ops.ssh.remote(host,f'''import pathlib,json,sys,subprocess
r=pathlib.Path({ORIGINAL[host]!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_rpc import status
v=status(r);assert v['alive'] and v['ready'] and not v['failed']
print(json.dumps(dict(binding=json.loads((r/'GPU_BINDING.json').read_text()),inputs=(r/'INPUTS.json').read_text(),
 owner=v['receipts']['SPOOL_OWNER.json'],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True))))''',timeout=35))
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu5.json')
    assert cpu['pass_'] and cpu['tests']==204 and cpu['skips']==cpu['sim_s']==0
    staged=json.loads(ops.stage_frozen(host,ROOTS[host],cpu))
    extra=json.loads(ops.stage(host,ROOTS[host],{'INPUTS.json':original['inputs'],
        'tools/coopmwm/hotadd_rpc.py':checked(ops.LOCAL/'tools/coopmwm/hotadd_rpc.py').read_text(encoding='utf-8')}))
    hashes=dict(staged['hashes'],**extra['hashes'])
    b=binding(original['binding'],hashes,host)
    ops.stage(host,ROOTS[host],{'GPU_BINDING.json':json.dumps(b,sort_keys=True)})
    value=dict(host=host,root=ROOTS[host],binding=b,original=original,cpu=cpu,new_sumo_sim_s=0,
        purpose='MAIN04:19 second independently qualified search worker; existing owners/searches untouched',
        admission='ALL8EXACT0 per new process, unchanged search hash/deadline; hot-add existing broker',
        comparison='report 1vs2 completed queries/hour by group size and candidate futures; no outcome-based sizing')
    write(target,value);h=sha(target)
    line='\nS-COOPMWM MAIN0419 SECOND WORKER ENGINEERING REGISTERED '+host+' amendmentSHA'+h+': unchanged immutable exactcpu5 numerical/search source, fresh8EXACT0 gates before broker admission; memoryfraction'+str(b['memory_fraction'])+' widths'+json.dumps(b['batch_width_by_K'],sort_keys=True)+'; existing workers/searches untouched, H300always/K8Hfinal/Nexec10 unchanged, originaldeadline retained, no SUMO/inference replay. Compare throughput1vs2 by group size and report; logging-only proposer remains unactivated.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/f'SECOND_REGISTRATION_{host}.json',dict(amendment_sha256=h,line=line,new_sumo_sim_s=0))
    print(json.dumps(dict(host=host,staged=True,amendment_sha256=h,gpu=original['gpu'])))


def dispatch(host):
    from . import concurrent_ops as c
    stage=read(ops.EVIDENCE/f'SECOND_STAGE_{host}.json')
    assert stage['root']==ROOTS[host]
    assert read(ops.EVIDENCE/f'SECOND_REGISTRATION_{host}.json')['amendment_sha256']
    oldread=c.read;oldwrite=c.write;c.GPU=ROOTS[host]
    c.read=lambda p:stage if p.name==f'CONCURRENT_STAGE_{host}.json' else oldread(p)
    c.write=lambda p,v:oldwrite(p.with_name(f'SECOND_DISPATCH_{host}.json') if p.name==f'CONCURRENT_DISPATCH_{host}.json' else p,v)
    try:c.dispatch(host)
    finally:c.read=oldread;c.write=oldwrite


def authorize():
    from . import warm_remote_ops as w
    assert not (ops.EVIDENCE/'SECOND_AUTHORIZATION.json').exists()
    w.ROUTES=ROUTES
    w.authorize('SECOND_AUTHORIZATION.json',roots=ROOTS,rpc_module='tools.coopmwm.hotadd_rpc')


def monitor(host):
    target=ops.EVIDENCE/f'SECOND_MONITOR_DISPATCH_{host}.json';assert not target.exists()
    stage=read(ops.EVIDENCE/f'SECOND_STAGE_{host}.json')
    registration=read(ops.EVIDENCE/f'SECOND_REGISTRATION_{host}.json')
    endpoints=read(ops.EVIDENCE/'SECOND_AUTHORIZATION.json')['endpoints']
    endpoint=dict(next(e for e in endpoints if e['slot']==host+'@0'),slot=host+'@1')
    owner=json.loads(ops.ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({ROOTS[host]!r});print(json.dumps(json.loads((r/'SPOOL_OWNER.json').read_text())))''',timeout=30))
    config=dict(host=host,run_root=__import__('tools.coopmwm.warm_deploy',fromlist=['ROOT']).ROOT,
        endpoint=endpoint,expected_owner=owner,deadline_unix=stage['binding']['deadline_unix'],
        search_sha256=stage['binding']['hashes']['pytools/tools/coopmwm/mppi_search.py'],
        amendment_sha256=registration['amendment_sha256'])
    files={'tools/coopmwm/hotadd.py':checked(ops.LOCAL/'tools/coopmwm/hotadd.py').read_text(encoding='utf-8'),
        'CONFIG_'+host+'.json':json.dumps(config,sort_keys=True)}
    ops.stage_frozen('cpuQ',MONITOR,read(ops.EVIDENCE/'CPU_coopmwm20261102_exactcpu5.json'))
    staged=json.loads(ops.stage('cpuQ',MONITOR,files))
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,subprocess,os,time,sys
r=pathlib.Path({config['run_root']!r});source=pathlib.Path({MONITOR!r});sys.path.insert(0,{exact_ops.CPU!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
assert r.is_dir() and source.is_dir();name='HOTADD_'+{host!r}+'_DISPATCH.json';assert not (r/name).exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({registration['line']!r})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'pytools')+':'+{exact_ops.CPU!r}+'/pytools',OMP_NUM_THREADS='1')
with (r/('HOTADD_'+{host!r}+'.log')).open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.hotadd',str(source/('CONFIG_'+{host!r}+'.json'))],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),config=json.loads({json.dumps(config)!r}))
immutable(r/name,v);print(json.dumps(v))''',timeout=35))
    write(target,dict(value,staged=staged));print(json.dumps(value))


def status(host):
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,sys,subprocess,time
r=pathlib.Path({ROOTS[host]!r});assert r.is_dir();sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_rpc import status
v=status(r)
for name in ('STARTUP_COMMITTED_K4_H300.json','STARTUP_COMMITTED_K8_H420.json'):
 if (r/name).exists():v['receipts'][name]=json.loads((r/name).read_text())
v.update(unix=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True))
if v['failed']:v['log_tail']=(r/'SPOOL.log').read_text()[-4000:]
print(json.dumps(v))''',timeout=35))
    write(ops.EVIDENCE/f'SECOND_STATUS_{host}_{int(value["unix"])}.json',value)
    print(json.dumps(dict(host=host,alive=value['alive'],ready=value['ready'],failed=value['failed'],
        gates={n:dict(pass_=r['pass_'],max_cost_error=r['max_cost_error']) for n,r in value['receipts'].items() if n.startswith('STARTUP_')},
        gpu=value['gpu'],failure=value.get('log_tail'))))
    return value


def export(host):
    from .pool_ops import dispatch
    assert read(ops.EVIDENCE/f'SECOND_REGISTRATION_{host}.json')['amendment_sha256']
    target=ops.EVIDENCE/f'SECOND_POOL_REGISTRATION_{host}.json';assert not target.exists()
    line='\nS-COOPMWM MAIN0413/0419 SECOND-WORKER POOL EXPORT REGISTERED '+host+': existing CPU-qualified logging-only pool_export sidecar extended to '+ROOTS[host]+'; preserve forecast bytes/SHA, full accepted/rejected pools, no WM calls, no activation, original GPU deadline. Exporter may wait through qualification; cannot admit a GPU worker or submit requests.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(line=line,new_sumo_sim_s=0))
    dispatch(host,root=ROOTS[host],receipt_prefix='SECOND_POOL',require_ready=False)


def monitor_status():
    from .warm_deploy import ROOT
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r});result={{}}
for host in ('gpuA','gpuR'):
 d=json.loads((r/('HOTADD_'+host+'_DISPATCH.json')).read_text())
 assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 p=pathlib.Path('/proc',str(d['pid']),'stat');s=p.read_text().split() if p.exists() else None
 assert not s or s[21]==d['start_ticks']
 names=['HOTADD_'+host+'_FAILED.json','WARM_REMOTE_ENDPOINT_ADMISSION_'+host+'_SECOND.json']
 result[host]=dict(owner=d,state=s[2] if s else 'exited',receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}})
print(json.dumps(dict(unix=time.time(),monitors=result)))''',timeout=35))
    write(ops.EVIDENCE/f'SECOND_MONITORS_STATUS_{int(value["unix"])}.json',value)
    print(json.dumps(value));return value


def resource_replace_a():
    """New empty spool for the already-dead A second worker; no claim transfer."""
    root='/root/coopmwm20261102_deadlineA2'
    target=ops.EVIDENCE/'RESOURCE_SECOND_A_REGISTRATION.json';assert not target.exists()
    line='\nS-COOPMWM MAIN1651 SECOND-A REPLACEMENT REGISTERED: old617099 already exited at16:29 on the known unsupported posthoc-family label ValueError; failed original spool/claim retained, never replayed. New EMPTY spool deadlineA2 uses byte-identical qualified incident1 numerical files (label fix already CPU/GPU-qualified), same MPPI search and horizons, old second-worker memoryfraction/batch widths; new Nov5 resource deadline. All8freshEXACT0 gates before separate hot-add; no inference or SUMO replay, no outcome change.\n'
    write(target,dict(line=line,root=root,new_sumo_sim_s=0));h=sha(target)
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    out=ops.ssh.remote('gpuA',f'''import pathlib,json,hashlib,os,time,subprocess
source=pathlib.Path('/root/coopmwm20261102_incident1');old=pathlib.Path('/root/coopmwm20261102_second1');r=pathlib.Path({root!r})
d=json.loads((old/'SPOOL_OWNER.json').read_text());p=pathlib.Path('/proc',str(d['pid']))
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not p.exists() or (p/'stat').read_text().split()[21]==d['start_ticks'] and (p/'stat').read_text().split()[2]=='Z'
assert json.loads((old/'SPOOL_FAILED.json').read_text())['error']=="ValueError('complete observable group of two or three required')"
assert not r.exists() and r.parent.is_dir();r.mkdir()
b=json.loads((source/'GPU_BINDING.json').read_text());second=json.loads((old/'GPU_BINDING.json').read_text())
for n,h in b['hashes'].items():
 assert n in ('INPUTS.json','COOPMWM_DESIGN.md','FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py') or (n.startswith('pytools/tools/') and n.endswith('.py')),n
 assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h,n
 target=r/n;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source/n)
b.update(deadline_unix=1793894340.,gpu_deadline=1793894340.,memory_fraction=second['memory_fraction'],batch_width_by_K=second['batch_width_by_K'],resource_extension_amendment={h!r})
b.pop('adopted_startup_gates',None)
with (r/'GPU_BINDING.json').open('x') as f:json.dump(b,f)
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'SPOOL.log').open('x') as log:
 process=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.warm_spool',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(process.pid),'stat').read_text().split()
v=dict(pid=process.pid,start_ticks=st[21],boot_id=d['boot_id'],root=str(r),host='gpuA',unix=time.time(),search_sha256=b['hashes']['pytools/tools/coopmwm/mppi_search.py'],memory_fraction=b['memory_fraction'],batch_width_by_K=b['batch_width_by_K'])
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=60)
    value=json.loads(out);write(ops.EVIDENCE/'RESOURCE_SECOND_A_DISPATCH.json',value);print(out)


def _resource_r_wait(oldroot,newroot):
    import pathlib,json,os,time,signal,hashlib,sys
    oldroot=pathlib.Path(oldroot);newroot=pathlib.Path(newroot)
    d=json.loads((oldroot/'SPOOL_OWNER.json').read_text())
    assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    p=pathlib.Path('/proc',str(d['pid']))
    while True:
        alive=p.exists() and (p/'stat').read_text().split()[2]!='Z'
        if p.exists():assert (p/'stat').read_text().split()[21]==d['start_ticks']
        missing=[f.stem for f in (oldroot/'inbox').glob('*.json') if not (oldroot/'forecasts'/f.name).exists()]
        if not missing or not alive:break
        time.sleep(2)
    # The old spool is excluded from the live broker until this replacement is
    # independently admitted. No original inference/request is copied or replayed.
    if alive:
        argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
        assert argv[-3:]==['-m','tools.coopmwm.warm_spool',str(oldroot)] or argv[-4:-2]==[str(pathlib.Path('/root/coopmwm20261102_deadlineops2/restart.py')),str(oldroot)],argv
        os.kill(d['pid'],signal.SIGTERM)
        for _ in range(150):
            if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
            assert (p/'stat').read_text().split()[21]==d['start_ticks'];time.sleep(.1)
        else:raise ValueError('old idle R spool did not exit')
    with (newroot/'LEGACY_RETIREMENT.json').open('x') as f:json.dump(dict(old=d,unix=time.time(),unreplayed_claim_keys=missing),f)
    os.environ.update(PYTHONPATH=str(newroot/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    os.execv(sys.executable,[sys.executable,'-B','-u','-m','tools.coopmwm.warm_spool',str(newroot)])


def resource_replace_r():
    import inspect
    root='/root/coopmwm20261102_deadlineR2'
    target=ops.EVIDENCE/'RESOURCE_SECOND_R_REGISTRATION.json';assert not target.exists()
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_verifybeta13.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    names=('mppi_runtime','plan_service','query_errors')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({cpu['root']!r});hashes=json.loads({json.dumps(cpu['source_hashes'])!r});files={{}}
for n in {names!r}:
 name='pytools/tools/coopmwm/'+n+'.py';p=r/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[name]
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=35))
    line='\nS-COOPMWM MAIN1651 LEGACY-R POSTHOC REPLACEMENT REGISTERED: original concurrent3/second1 source still has unsupported4-6 posthoc labels. New EMPTY deadlineR2 spool clones old second1 numerical files except the ALREADY CPU155-qualified mppi_runtime posthoc fallback/plan_service code-error isolation/query_errors; no search/horizon changes. Old second1 migration helper alone is stopped while its original inference finishes; retire original GPU PID only after every forecast settles (or verified dead); preserve every missing failed claim, never copy/replay it. New helper execs warm_spool with Nov5 deadline/memory.34/widths4,4, requires8freshEXACT0 before new-slot hot-add. Zero SUMO.\n'
    write(target,dict(line=line,cpu=cpu,root=root,new_sumo_sim_s=0));h=sha(target)
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    out=ops.ssh.remote('gpuR',f'''import pathlib,json,hashlib,os,signal,time
old=pathlib.Path('/root/coopmwm20261102_second1');r=pathlib.Path({root!r});assert not r.exists() and r.parent.is_dir()
# Stop only the waiting/qualification migration helper; original inference PID
# remains untouched here and is drained by the new durable helper.
d=json.loads((old/'RESOURCE_EXTENSION_DISPATCH_1651_V2.json').read_text());p=pathlib.Path('/proc',str(d['pid']))
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
owner=json.loads((old/'SPOOL_OWNER.json').read_text())
if p.exists() and (p/'stat').read_text().split()[2]!='Z':
 assert (p/'stat').read_text().split()[21]==d['start_ticks']
 assert owner['pid']!=d['pid'],'helper already owns GPU; inspect before retirement'
 argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
 assert argv[-4:-2]==['/root/coopmwm20261102_deadlineops2/restart.py',str(old)],argv
 os.kill(d['pid'],signal.SIGTERM)
r.mkdir();b=json.loads((old/'GPU_BINDING.json').read_text());files=json.loads({json.dumps(files)!r})
for n,h in list(b['hashes'].items()):
 assert n in ('INPUTS.json','COOPMWM_DESIGN.md','FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py') or (n.startswith('pytools/tools/') and n.endswith('.py')),n
 assert hashlib.sha256((old/n).read_bytes()).hexdigest()==h,n
 if n.removeprefix('pytools/') in files:continue
 target=r/n;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(old/n)
for n,text in files.items():
 target=r/'pytools'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text);b['hashes'][str(target.relative_to(r))]=hashlib.sha256(target.read_bytes()).hexdigest()
b.update(deadline_unix=1793894340.,gpu_deadline=1793894340.,memory_fraction=.34,batch_width_by_K={{'4':4,'8':4}},resource_extension_amendment={h!r})
b.pop('adopted_startup_gates',None)
with (r/'GPU_BINDING.json').open('x') as f:json.dump(b,f)
print(json.dumps(dict(staged=True,root=str(r),old_owner=owner,search_sha256=b['hashes']['pytools/tools/coopmwm/mppi_search.py'])))''',timeout=60)
    write(ops.EVIDENCE/'RESOURCE_SECOND_R_STAGE.json',json.loads(out))
    script=inspect.getsource(_resource_r_wait)+f'\n_resource_r_wait({"/root/coopmwm20261102_second1"!r},{root!r})\n'
    ops.stage('gpuR',root,{'wait_old.py':script})
    out=ops.ssh.remote('gpuR',f'''import pathlib,json,subprocess,os,time
r=pathlib.Path({root!r});assert not (r/'DISPATCH.json').exists()
env=dict(os.environ,PYTHONPATH=str(r/'pytools'))
with (r/'SPOOL.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(r/'wait_old.py')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),host='gpuR',unix=time.time())
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=35)
    write(ops.EVIDENCE/'RESOURCE_SECOND_R_DISPATCH.json',json.loads(out));print(out)


def _resource_monitor(config):
    import pathlib,json,time,sys,os
    sys.path.insert(0,'/home/ubuntu/coopmwm20261102_verifybeta13/pytools')
    from tools.coopmwm.warm_broker import Remote
    from tools.coopmwm.warm_spool import immutable
    from tools.coopmwm.hotadd import qualified,GATES
    from tools.coopmwm.warm_health import assert_running
    from tools.coopmwm.warm_transport import retry
    root=pathlib.Path(config['run_root']);remote=Remote(root,config['endpoint'])
    def check_status(value):
        if value['failed']:raise ValueError('replacement qualification failed; preserve')
        owner=value['receipts'].get('SPOOL_OWNER.json')
        if owner:
            for key in ('pid','start_ticks','boot_id'):assert owner[key]==config['expected'][key],key
        return value
    def reverify():return check_status(remote.raw(dict(action='status')))
    def event(value):
        immutable(root/('RESOURCE_MONITOR_TRANSPORT_'+config['tag']+'_'+str(time.time_ns())+'.json'),value)
    try:
        while time.time()<1793894340.:
            if (root/('RESOURCE_ADMITTED_'+config['tag']+'.json')).exists():return
            assert_running(root,'B');status=check_status(retry(lambda:remote.raw(dict(action='status')),reverify,event))
            owner=status['receipts'].get('SPOOL_OWNER.json')
            if owner:
                for key in ('pid','start_ticks','boot_id'):assert owner[key]==config['expected'][key],key
                if qualified(status):
                    service=status['receipts']['SERVICE_OWNER.json'];assert service['pid']==owner['pid']
                    original=json.loads((root/'WARM_REMOTE_BINDING.json').read_text())
                    assert original['search_sha256']==config['search_sha256']
                    assert config['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
                    immutable(root/('WARM_REMOTE_ENDPOINT_ADMISSION_'+config['tag']+'_SECOND.json'),
                        dict(amendment_sha256=config['amendment_sha256'],search_sha256=config['search_sha256'],
                            endpoints=[config['endpoint']],owner=owner,gates={n:status['receipts'][n] for n in GATES},unix=time.time()))
                    immutable(root/('GPU_RESOURCE_READY_'+config['endpoint']['slot']+'_1651.json'),
                        dict(slot=config['endpoint']['slot'],owner=owner,service=service,amendment_sha256=config['amendment_sha256']))
                    immutable(root/('RESOURCE_ADMITTED_'+config['tag']+'.json'),dict(unix=time.time(),owner=owner,slot=config['endpoint']['slot']))
                    return
            elif status['failed']:raise ValueError('replacement failed before owner; preserve')
            time.sleep(15)
        raise ValueError('extended resource deadline')
    except BaseException as error:
        immutable(root/('RESOURCE_MONITOR_FAILED_'+config['tag']+'.json'),dict(error=repr(error),unix=time.time(),no_inference_retry=True));raise


def resource_authorize():
    from . import warm_remote_ops as w
    roots=dict(gpuA='/root/coopmwm20261102_deadlineA2',gpuR='/root/coopmwm20261102_deadlineR2')
    assert not (ops.EVIDENCE/'RESOURCE_SLOT_AUTHORIZATION_1651.json').exists()
    for host,root in roots.items():
        ops.stage(host,root,{'tools/coopmwm/hotadd_rpc.py':checked(ops.LOCAL/'tools/coopmwm/hotadd_rpc.py').read_text(encoding='utf-8')})
    w.ROUTES='/home/ubuntu/coopmwm20261102_resource_routes1651'
    w.authorize('RESOURCE_SLOT_AUTHORIZATION_1651.json',roots=roots,rpc_module='tools.coopmwm.hotadd_rpc')


def resource_monitor(host):
    import inspect
    from .warm_deploy import ROOT
    tag=host+'resource1651'
    target=ops.EVIDENCE/('RESOURCE_MONITOR_DISPATCH_'+host+'_1651_V2.json');assert not target.exists()
    dispatched=read(ops.EVIDENCE/('RESOURCE_SECOND_'+('A' if host=='gpuA' else 'R')+'_DISPATCH.json'))
    regfile=ops.EVIDENCE/('RESOURCE_SECOND_'+('A' if host=='gpuA' else 'R')+'_REGISTRATION.json')
    reg=read(regfile);h=sha(regfile)
    endpoints=read(ops.EVIDENCE/'RESOURCE_SLOT_AUTHORIZATION_1651.json')['endpoints']
    endpoint=dict(next(e for e in endpoints if e['slot']==host+'@0'),slot=host+'@resource1651')
    if host=='gpuA':search_sha=dispatched['search_sha256']
    else:search_sha=read(ops.EVIDENCE/'RESOURCE_SECOND_R_STAGE.json')['search_sha256']
    cfg=dict(run_root=ROOT,tag=tag,expected=dispatched,amendment_sha256=h,
        search_sha256=search_sha,endpoint=endpoint)
    source='/home/ubuntu/coopmwm20261102_resourcehotadd2'
    script=inspect.getsource(_resource_monitor)+'\nimport json,sys\n_resource_monitor(json.loads(sys.argv[1]))\n'
    ops.stage('cpuQ',source,{'monitor.py':script})
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,subprocess,os,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({source!r});name='RESOURCE_MONITOR_DISPATCH_'+{host!r}+'_1651_V2.json'
assert not (r/name).exists()
assert not (r/('RESOURCE_ADMITTED_'+{tag!r}+'.json')).exists(),'already admitted; adopt original monitor'
old=json.loads((r/('RESOURCE_MONITOR_DISPATCH_'+{host!r}+'_1651.json')).read_text());p=pathlib.Path('/proc',str(old['pid']))
assert old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if p.exists() and (p/'stat').read_text().split()[2]!='Z':
 assert (p/'stat').read_text().split()[21]==old['start_ticks']
 argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
 assert argv[:4]==['/home/ubuntu/miniconda3/bin/python','-B','-u','/home/ubuntu/coopmwm20261102_resourcehotadd1/monitor.py'] and json.loads(argv[4])==old['config']
 __import__('os').kill(old['pid'],__import__('signal').SIGTERM)
 for _ in range(150):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  assert (p/'stat').read_text().split()[21]==old['start_ticks'];time.sleep(.1)
 else:raise ValueError('old read-only monitor did not exit')
with (r/'AUTHORITY.md').open('a') as f:f.write({(reg['line']+'amendmentSHA'+h+chr(10))!r})
with (r/'AUTHORITY.md').open('a') as f:f.write('MAIN1651 monitor-only transport correction: bounded retries validate expected owner even before READY; incomplete qualification waits, never authorizes inference. No GPU/physics restart.\\n')
with (r/('RESOURCE_MONITOR_'+{host!r}+'_1651_V2.log')).open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u',str(s/'monitor.py'),{json.dumps(cfg)!r}],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),source=str(s),unix=time.time(),config=json.loads({json.dumps(cfg)!r}))
with (r/name).open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=35)
    write(target,json.loads(out));print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('cpu','stage','dispatch','authorize','monitor','status','export','monitor_status','resource_replace_a','resource_replace_r','resource_authorize','resource_monitor'))
    parser.add_argument('--host',choices=('gpuA','gpuR'));args=parser.parse_args()
    if args.mode in ('authorize','cpu','monitor_status','resource_replace_a','resource_replace_r','resource_authorize'):globals()[args.mode]()
    else:
        assert args.host;globals()[args.mode](args.host)
