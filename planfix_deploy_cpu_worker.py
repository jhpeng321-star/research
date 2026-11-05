"""Stage or singly dispatch the CPU eight-gates/one-search worker; no SUMO."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from planfix_ops import remote,note,TEMP
from planfix_cpu_prepare import blob

HERE=Path(__file__).parent
ROOT='/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_wm_v1'

STAGE=r'''
import base64,hashlib,json,os,shutil,subprocess,sys,time,zlib
from pathlib import Path
w=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix');root=Path(__ROOT__)
packed=Path(__ARCHIVE__).read_bytes();assert hashlib.sha256(packed).hexdigest()==__ARCHIVE_SHA__
files=json.loads(zlib.decompress(packed))
assert not root.exists(),'fresh stage only; inspect existing receipt, never overwrite'
if root.name!='cpu_wm_v1':
 previous=w/('cpu_wm_v'+str(int(root.name.rsplit('v',1)[1])-1))
 assert (previous/'CPU_FAILED.json').is_file() or (previous/'CPU_OPERATOR_RETIRED.json').is_file() or (previous/'QUERY_ABI_CORRECTION.json').is_file()
 owner=json.loads((previous/'CPU_OWNER.json').read_text());p=Path('/proc',str(owner['pid']),'stat')
 f=p.read_text().split() if p.exists() else None
 active=bool(f and f[21]==owner['start_ticks'] and f[2]!='Z')
 if active:
  assert root.name=='cpu_wm_v7' and f[2]=='T' and (previous/'QUERY_ABI_CORRECTION.json').is_file() and (previous/'STOP.json').is_file(),'previous own worker still alive'
c=Path('/home/ubuntu/coopmwm20261105_clean2');launch=json.loads((c/'LAUNCHED.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(c/'CONFIG.json')==launch['config_file_sha256']
assert json.loads((c/'SMOKE_PASS.json').read_text())['receipt_sha256']==launch['smoke_sha256']
assert json.loads((c/'GATE_PASS.json').read_text())['receipt_sha256']==launch['gate_sha256']
assert not (c/'STOP.json').exists() and not (c/'FAULT.json').exists()
for owner in launch['owners'].values():
 f=Path('/proc',str(owner['pid']),'stat').read_text().split();assert f[21]==owner['start_ticks'] and f[2]!='Z'
root.mkdir();physical=root/'runtime';physical.mkdir()
prior=w/'cpu_priors_97fa5403964f';assert json.loads((prior/'VERIFIED.json').read_text())['all_file_shas_verified']
prepared=w/'cpu_prepare_ea559c952ff0';source_manifest=json.loads((prepared/'SOURCE_MANIFEST.json').read_text())
for name,h in source_manifest.items():
 p=prepared/name;assert sha(p)==h
 q=physical/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
posterior='artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz'
q=physical/posterior;q.symlink_to(prior/posterior)
recipe=physical/'configs/wm_phys.toml';text=recipe.read_text()
old='/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'
assert text.count(old)==1;text=text.replace(old,str(physical/'configs/c4_wm_loop14_r16_ft.toml'))
recipe.write_text(text)
(root/'pytools').symlink_to(physical,target_is_directory=True)
legacy=Path('/home/ubuntu/coopmwm20261104_launchcoop_src7')
for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):shutil.copyfile(legacy/name,root/name)
fix=root/'fix/configs';fix.mkdir(parents=True)
shutil.copyfile(w/'cpu_bootstrap_recovered/execparity_cav_idm_v1.json',fix/'execparity_cav_idm_v1.json')
for name,value in files.items():
 target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as out:out.write(base64.b64decode(value));out.flush();os.fsync(out.fileno())
pace={}
for role in ('A','B','R'):
 values=[json.loads(p.read_text()) for p in sorted(c.glob(role+'_UPDATE_*.json'))[-4:]]
 assert len(values)>=4
 intervals=[b['finished_unix']-a['finished_unix'] for a,b in zip(values,values[1:])]
 assert all(t>0 for t in intervals)
 pace[role]=dict(last_update=values[-1]['update'],intervals_s=intervals,ceiling_s=max(intervals))
if int(root.name.rsplit('v',1)[1])>=7:
 inherited=json.loads((w/'cpu_wm_v6/CLEAN2_BASELINE.json').read_text())
 assert inherited['launch_file_sha256']==sha(c/'LAUNCHED.json')
 pace=inherited['pace']  # preserve the original limits; never reset after a hold
with (root/'CLEAN2_BASELINE.json').open('x') as out:json.dump(dict(unix=time.time(),launch_file_sha256=sha(c/'LAUNCHED.json'),pace=pace),out)
# Model-free fixture extraction in the qualified existing CPU environment.
code="""
import hashlib,json,os,sys,time,torch,numpy as np
from pathlib import Path
root=Path(__FIXTURE_ROOT__);physical=root/'runtime';prior=root.parent/'cpu_priors_97fa5403964f'
os.nice(19);torch.set_num_threads(1)
sys.path[:0]=[str(physical),'/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools']
import tools;tools.__path__.insert(0,str(physical/'tools'))
from tools.coopmwm.protocol import digest
from tools.coopmwm.fresh_worker import pending
from tools.coopmwm.storage import write
from tools.coopmwm.varcollect import search_config
p=prior/'data/jobs/s969420_ctrl0/start.pt';st=torch.load(p,map_location='cpu',weights_only=False)
with np.load(p.with_name('start_history.npz'),allow_pickle=False) as raw:ids=torch.as_tensor(raw['vid_index'][-1])[st['state']['present'][0].bool()].long().tolist()
selected=st['state']['present'][0].bool();state=st['state']
fields=('s_m','v_mps','lane','is_cav','length_m')
rows=[dict(vid='observed.'+str(vid),**{k:state[k][0,selected][i].item() for k in fields}) for i,vid in enumerate(ids)]
cavs=[r['vid'] for r in rows if r['is_cav']];assert len(cavs)>=2
fixture=dict(world_id=969420,time_s=300,rows=rows,group=cavs[:2]);fixture['state_id']=digest(fixture)
write(root/'INPUTS.json',dict(records=[fixture],source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),new_sim_s=0))
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'runtime').rglob('*') if p.is_file()}
for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','INPUTS.json','planfix_cpu_runtime.py','planfix_cpu_worker.py'):
 hashes[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
pin=dict(host='cpuV',python_executable=sys.executable,python_version='.'.join(map(str,sys.version_info[:2])),
 torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
 variables={k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
binding=dict(hashes=hashes,qualification=pending(hashes),worker_env=pin,memory_mode='warm',startup_qualification=True,
 speed2=True,speed2_transfers=True,concurrent_plans=True,batch_widths={'4':4,'8':2},
 memory_fraction=.65,deadline_unix=time.time()+(86400 if int(root.name.rsplit('v',1)[1])>=7 else 21600),search_config={'variety_families':True},
 device='cpu',phase='plan_fix',sumo_allowed=False,source_commit='f631ce90cfd2f23213f7c57a828c848e2db10ecd')
# The original collector uses its registered it2/K2 niche boundary and retains
# all eight final K8 candidates, including negatives, for all8 verification.
binding.update(search_config=search_config({'variety':{'variety_families':True}}),
               label_list8=True,niche_comparison=True)
write(root/'GPU_BINDING.json',binding)
print('CPU_BINDING',digest(binding),len(hashes),flush=True)
""".replace('__FIXTURE_ROOT__',repr(str(root)))
env=dict(os.environ,PYTHONPATH=str(physical/'src')+':'+str(physical)+':/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools',
 LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-c',code],env=env,capture_output=True,text=True,timeout=25)
print(r.stdout,r.stderr,flush=True)
with (root/'STAGE_ENV.json').open('x') as out:json.dump(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,env=env),out)
assert r.returncode==0,'retain staged failure, no dispatch'
receipt=dict(root=str(root),source_files=len(source_manifest),binding_file_sha256=sha(root/'GPU_BINDING.json'),
 launch_file_sha256=sha(c/'LAUNCHED.json'),pace=pace,new_sim_s=0,dispatch_started=False)
with (root/'STAGE_PASS.json').open('x') as out:json.dump(receipt,out,sort_keys=True)
print('CPU_STAGE_PASS',json.dumps(receipt),flush=True)
'''

DISPATCH=r'''
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__ROOT__);assert (root/'STAGE_PASS.json').is_file()
assert (root/'CPU_PORT_QUALIFIED.json').is_file()
reg=root/'CPU_EXECUTION_REGISTRATION.md';assert reg.is_file()
assert 'PLANFIX CPU execution registration SHA '+hashlib.sha256(reg.read_bytes()).hexdigest() in (root.parent.parent/'AUTHORITY.md').read_text()
assert not (root/'DISPATCH_STARTED.json').exists() and not (root/'CPU_OWNER.json').exists()
with (root/'DISPATCH_STARTED.json').open('x') as out:json.dump(dict(unix=time.time(),new_sim_s=0),out);out.flush();os.fsync(out.fileno())
env=json.loads((root/'STAGE_ENV.json').read_text())['env'];env['PYTHONPATH']=str(root/'compat')+':'+str(root)+':'+env['PYTHONPATH']
with (root/'CPU_WORKER.log').open('xb') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(root/'planfix_cpu_worker.py'),str(root)],
  env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
fields=Path('/proc',str(p.pid),'stat').read_text().split()
receipt=dict(root=str(root),pid=p.pid,start_ticks=fields[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 unix=time.time(),new_sim_s=0,max_cpu_cores=1,nice=19,procedure='eight same-process gates then ONE WM search; no SUMO; no expansion')
with (root/'DISPATCH.json').open('x') as out:json.dump(receipt,out,sort_keys=True);out.flush();os.fsync(out.fileno())
print('CPU_DISPATCH',json.dumps(receipt),flush=True)
'''

QUALIFY=r'''
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__ROOT__);assert (root/'STAGE_PASS.json').is_file()
compat=root/'compat';compat.mkdir(exist_ok=True)
shim=compat/'tomllib.py';content='from pip._vendor.tomli import *\n'
if shim.exists():assert shim.read_text()==content
else:
 with shim.open('x') as out:out.write(content)
code="""
import ast,hashlib,json,sys,tomllib
from pathlib import Path
root=Path(__QUALIFY_ROOT__);sys.path.insert(0,str(root))
from planfix_cpu_runtime import cpu_tree,RUNTIME_SHA,execute_tree
original=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools/tools/wmteach/runtime.py')
assert hashlib.sha256(original.read_bytes()).hexdigest()==RUNTIME_SHA
tree=cpu_tree(original.read_text(),{'cuda':'cpu'},recovered_config=True)
compile(tree,'<cpu-port-abi-qualification>','exec')
import inspect
scope={};source=chr(10).join(['def prepare_batch(rt, source):',"    return with_entry_slots(source,300,960,device='cuda')",''])
execute_tree(cpu_tree(source,{'cuda':'cpu'}),'<cpu-port-source-qualification>',scope)
assert 'cpu' in inspect.getsource(scope['prepare_batch'])
scale=original.with_name('scale.py');raw=scale.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='98c63848a054cf2ec99f57acd3cdbb6975217317e80e0ec142827007741f698c'
scale_tree=cpu_tree(raw.decode(),{'cuda':'cpu'})
for f in scale_tree.body:
 if isinstance(f,ast.FunctionDef) and f.name in ('futures','policy_trajectory'):
  assert not any(isinstance(n,ast.Constant) and n.value=='cuda' for n in ast.walk(f))
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cuda' for n in ast.walk(tree))
for name in ('planfix_cpu_runtime.py','planfix_cpu_worker.py'):
 compile((root/name).read_text(),name,'exec')
from planfix_cpu_worker import native_requests
if hasattr(__import__('planfix_cpu_worker'),'prepare_query'):
 import numpy as np
 class Export:
  def __call__(self,state):raise AssertionError('bare export returns acceleration only')
  def act_with_lane(self,state):return np.array([[1.2,np.nan]]),None
 assert native_requests(Export(),{},[dict(vid='c',is_cav=True),dict(vid='h',is_cav=False)])=={'c':1.2}
old=tomllib.loads((root.parent/'cpu_prepare_ea559c952ff0/configs/wm_phys.toml').read_text())
new=tomllib.loads((root/'runtime/configs/wm_phys.toml').read_text());old['source']['config']=new['source']['config'];assert old==new
recovered=json.loads((root.parent/'cpu_bootstrap_recovered/RECOVERED.json').read_text())
assert recovered['independent_config_observations']==4 and len(recovered['config'])==75
assert recovered['config']['cav_exec_cap']=='idm'
binding=json.loads((root/'GPU_BINDING.json').read_text())
for name,sha in binding['hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
if (root/'runtime/tools/rbfit/idm_fit.py').exists():
 sys.path[:0]=[str(root/'runtime'),'/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools']
 import tools;tools.__path__.insert(0,str(root/'runtime/tools'))
 from tools.coopmwm.varcollect import search_config
 assert binding['search_config']==search_config({'variety':{'variety_families':True}})
 assert binding['label_list8'] and binding['niche_comparison']
print('CPU_PORT_AST_PATHS_CONFIG_SOURCE_PASS',len(binding['hashes']),flush=True)
""".replace('__QUALIFY_ROOT__',repr(str(root)))
env=json.loads((root/'STAGE_ENV.json').read_text())['env'];env['PYTHONPATH']=str(compat)+':'+env['PYTHONPATH']
r=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-c',code],env=env,capture_output=True,text=True,timeout=20)
receipt=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,new_sim_s=0,model_bootstrap=False,
 tomllib_compat_sha256=hashlib.sha256(shim.read_bytes()).hexdigest())
print('CPU_PORT_QUALIFICATION',json.dumps(receipt),flush=True)
assert r.returncode==0
with (root/'CPU_PORT_QUALIFIED.json').open('x') as out:json.dump(receipt,out,sort_keys=True)
'''

REGISTER=r'''
import base64,fcntl,hashlib,json,os,time
from pathlib import Path
root=Path(__ROOT__);assert (root/'CPU_PORT_QUALIFIED.json').is_file()
raw=base64.b64decode(__REG__);sha=hashlib.sha256(raw).hexdigest()
p=root/'CPU_EXECUTION_REGISTRATION.md'
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
ledger=root.parent.parent;marker='PLANFIX CPU execution registration SHA '+sha
with (ledger/'ACCOUNTING.claim').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if marker not in (ledger/'AUTHORITY.md').read_text():
  with (ledger/'AUTHORITY.md').open('a') as out:out.write('\n'+marker+'\n'+raw.decode()+'\n');out.flush();os.fsync(out.fileno())
rows=[json.loads(x) for x in (ledger/'CHARGES.jsonl').read_text().splitlines()]
spent=sum(x['sim_s'] for x in rows if x['phase']=='plan_fix');assert spent==33600
receipt=dict(root=str(root),registration_sha256=sha,authority=str(ledger/'AUTHORITY.md'),phase_charged_sim_s=spent,new_sim_s=0,unix=time.time())
print('CPU_REGISTERED',json.dumps(receipt),flush=True)
'''

STATUS=r'''
import json,time,os
from pathlib import Path
root=Path(__ROOT__);result=dict(root=str(root),exists=root.exists(),unix=time.time())
if root.exists():
 for name in ('STAGE_PASS.json','STAGE_ENV.json','DISPATCH_STARTED.json','DISPATCH.json','CPU_OWNER.json','BOOTSTRAP_PASS.json','CPU_FAILED.json','FRESH_QUALIFICATION.json','SPEED_SEARCH_STARTED.json','SPEED_SEARCH_RESULT.json','PACE_CONTROL_STATE.json','PACE_SUPERVISOR_OWNER.json','PACE_SUPERVISOR_TERMINAL.json','CPU_OPERATOR_RETIRED.json','GATED_COORDINATOR_OWNER.json','GATED_COORDINATOR_TERMINAL.json','GATED_PHASE.json','GATED_POLL_LATEST.json'):
  p=root/name
  if p.exists():
   d=json.loads(p.read_text())
   if name=='STAGE_ENV.json':d={k:d[k] for k in ('returncode','stdout','stderr')}
   if name=='BOOTSTRAP_PASS.json':d={k:v for k,v in d.items() if k!='source_receipt'}
   if name=='SPEED_SEARCH_RESULT.json':d={k:v for k,v in d.items() if k!='result'}
   result[name]=d
 for name in ('CPU_OWNER.json','DISPATCH.json','PACE_SUPERVISOR_OWNER.json','GATED_COORDINATOR_OWNER.json'):
  if name in result:
   d=result[name];p=Path('/proc',str(d['pid']),'stat');f=p.read_text().split() if p.exists() else None
   d['alive_same_identity']=bool(f and f[21]==d['start_ticks'] and f[2]!='Z')
   if d['alive_same_identity']:
    d['process_state']=f[2]
    d['cpu_seconds']=(int(f[13])+int(f[14]))/os.sysconf('SC_CLK_TCK')
    d['elapsed_s']=time.time()-d['unix']
    status=Path('/proc',str(d['pid']),'status').read_text().splitlines()
    d['resources']={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status if line.split(':',1)[0] in ('VmRSS','Threads','Cpus_allowed_list')}
 result['gates']={p.name:json.loads(p.read_text()) for p in sorted(root.glob('STARTUP_*.json'))}
 p=root/'SPEED_QUERY.json'
 if p.exists():
  q=json.loads(p.read_text());record=q['record']
  result['query_preflight']=dict(prepared=True,prepared_before_gates_without_wm_search=q.get('prepared_before_gates_without_wm_search',False),
   world_id=record['world_id'],time_s=record['time_s'],group=record['group'],observable_rows=len(record['rows']),
   snapshot_sha256=q['snapshot_sha256'],new_sim_s=q['new_sim_s'])
 p=root/'CPU_WORKER.log'
 if p.exists():result['log_tail']=p.read_bytes()[-14000:].decode(errors='replace')
 p=root/'PACE_SUPERVISOR.log'
 if p.exists():result['pace_supervisor_log_tail']=p.read_bytes()[-6000:].decode(errors='replace')
 p=root/'GATED_COORDINATOR.log'
 if p.exists():result['gated_log_tail']=p.read_bytes()[-10000:].decode(errors='replace')
 control=root/'gated_control_v2'
 if control.exists():
  attempt={}
  for name in ('GATED_COORDINATOR_OWNER.json','GATED_COORDINATOR_TERMINAL.json','GATED_PHASE.json','GATED_POLL_LATEST.json','GATED_COORDINATOR_DISPATCH.json'):
   p=control/name
   if p.exists():attempt[name]=json.loads(p.read_text())
  if 'GATED_COORDINATOR_OWNER.json' in attempt:
   d=attempt['GATED_COORDINATOR_OWNER.json'];p=Path('/proc',str(d['pid']),'stat');f=p.read_text().split() if p.exists() else None
   d['alive_same_identity']=bool(f and f[21]==d['start_ticks'] and f[2]!='Z')
   if d['alive_same_identity']:d['process_state']=f[2];d['cpu_seconds']=(int(f[13])+int(f[14]))/os.sysconf('SC_CLK_TCK')
  p=control/'GATED_COORDINATOR.log'
  if p.exists():attempt['log_tail']=p.read_bytes()[-10000:].decode(errors='replace')
  result['gated_control_v2']=attempt
print('CPU_STATUS',json.dumps(result),flush=True)
'''

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stage','qualify','register','dispatch','status'))
    p.add_argument('--generation',type=int,default=1);a=p.parse_args();assert 1<=a.generation<=8
    ROOT=ROOT.rsplit('v',1)[0]+'v'+str(a.generation)
    files={n:base64.b64encode((HERE/n).read_bytes()).decode() for n in ('planfix_cpu_runtime.py','planfix_cpu_worker.py')}
    if a.action=='stage':
        for name in ('entry_learned.json','merge_core_init.json','ramp_speed_adapt_init.json'):
            relative='src/wms/model/'+name
            files['runtime/'+relative]=base64.b64encode(blob('22cc32f6',relative)).decode()
        files['runtime/tools/rbfit/idm_fit.py']=base64.b64encode(blob('22cc32f6','tools/rbfit/idm_fit.py')).decode()
    code=(STAGE if a.action=='stage' else STATUS if a.action=='status' else QUALIFY if a.action=='qualify' else REGISTER if a.action=='register' else DISPATCH).replace('__ROOT__',repr(ROOT))
    if a.action=='stage':
        import zlib,subprocess
        packed=zlib.compress(json.dumps(files,sort_keys=True).encode(),9);digest=hashlib.sha256(packed).hexdigest()
        package=TEMP/f'planfix_cpu_worker_{digest[:12]}.z';assert TEMP.is_dir()
        if package.exists():assert package.read_bytes()==packed
        else:
            with package.open('xb') as out:out.write(packed)
        destination='/home/ubuntu/coopmwm20261105_clean1/plan_fix/'+package.name
        checked=remote('cpuV',f"from pathlib import Path;import hashlib;p=Path({destination!r});print('PACKAGE',p.exists(),hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else '')",timeout=25)
        if 'PACKAGE True '+digest not in checked:
            assert 'PACKAGE False' in checked
            result=subprocess.run(['wsl','-d','Ubuntu-22.04','--','python3','-B',
                '/mnt/e/agent-work/c4-variety-collect/planfix_cpuv_transport.py','--timeout','35',
                '--put','/mnt/e/Temp/opencode/'+package.name,destination],capture_output=True,timeout=95)
            print(result.stdout.decode(errors='replace'),result.stderr.decode(errors='replace'),flush=True)
            assert result.returncode==0,'inspect package SHA before any repeat'
        code=code.replace('__ARCHIVE__',repr(destination)).replace('__ARCHIVE_SHA__',repr(digest))
    if a.action=='register':
        name='PLANFIX_CPU_EXECUTION_REGISTRATION.md' if a.generation==1 else f'PLANFIX_CPU_V{a.generation}_REGISTRATION.md'
        code=code.replace('__REG__',repr(base64.b64encode((HERE/name).read_bytes()).decode()))
    text=remote('cpuV',code,timeout=50)
    if a.action=='status':
        import time
        value=json.loads(next(x[len('CPU_STATUS '):] for x in text.splitlines() if x.startswith('CPU_STATUS ')))
        with (HERE/f'PLANFIX_CPU_STATUS_v{a.generation}_{time.time_ns()}.json').open('x') as out:json.dump(value,out,indent=2,sort_keys=True)
        resource_owner=value.get('gated_control_v2',{}).get('GATED_COORDINATOR_OWNER.json',value.get('GATED_COORDINATOR_OWNER.json',{}))
        owner=value.get('CPU_OWNER.json',value.get('DISPATCH.json',resource_owner))
        component='WM' if 'CPU_OWNER.json' in value else 'guarded coordinator'
        complete=sum(g.get('pass_') is True for g in value.get('gates',{}).values())
        note('cpuV',f"CPU v{a.generation} {component} pid={owner.get('pid')} alive={owner.get('alive_same_identity',False)} state={owner.get('process_state')} bootstrap_pass={'BOOTSTRAP_PASS.json' in value} gates_pass={complete}/8 speed_complete={'SPEED_SEARCH_RESULT.json' in value} failure={value.get('CPU_FAILED.json',{}).get('error')} new_sim_s=0")
        raise SystemExit(0)
    prefix='CPU_STAGE_PASS ' if a.action=='stage' else 'CPU_PORT_QUALIFICATION ' if a.action=='qualify' else 'CPU_REGISTERED ' if a.action=='register' else 'CPU_DISPATCH '
    receipt=json.loads(next(x[len(prefix):] for x in text.splitlines() if x.startswith(prefix)))
    path=HERE/('PLANFIX_CPU_STAGE_PASS.json' if a.action=='stage' else 'PLANFIX_CPU_PORT_QUALIFIED.json' if a.action=='qualify' else 'PLANFIX_CPU_EXECUTION_REGISTERED.json' if a.action=='register' else 'PLANFIX_CPU_DISPATCH.json')
    if a.generation!=1:path=path.with_name(path.stem+f'_v{a.generation}'+path.suffix)
    with path.open('x') as out:json.dump(receipt,out,indent=2,sort_keys=True)
    note('cpuV',f'CPU {a.action} receipt verified; original WM/scoring, eight fresh gates then one speed search, no SUMO or bank switch')
