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


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('cpu','stage','dispatch','authorize','monitor','status','export','monitor_status'))
    parser.add_argument('--host',choices=('gpuA','gpuR'));args=parser.parse_args()
    if args.mode in ('authorize','cpu','monitor_status'):globals()[args.mode]()
    else:
        assert args.host;globals()[args.mode](args.host)
