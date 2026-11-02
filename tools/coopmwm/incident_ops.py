"""Explicit MAIN09:44 incident qualification and read-only recovery census."""
import argparse
import json
from . import ops
from .storage import read,write
from .warm_deploy import ROOT

SOURCE='/home/ubuntu/coopmwm20261102_incidentcpu2'
OVERLAYS=('mppi_runtime','query_errors','plan_service','capacity_broker','incident_ops','test_incident_cpu')
GPU='/root/coopmwm20261102_incident1'


def register():
    from .protocol import REGISTRY,sha
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_incidentcpu1.json')
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    target=ops.EVIDENCE/'INCIDENT0944_AMENDMENT.json';assert not target.exists()
    value=dict(cpu=cpu,new_sumo_sim_s=0,diagnosis='posthoc legacy prototypes rejected accepted group>3',
        change='unsupported group labels new; any prototype-label failure unlabelled with traceback; per-plan lookup/type/name Python code faults produce record-bound declined result and traceback; physics/parity/accounting/validation/CUDA faults remain fatal',
        gpu='replace ONLY failed gpuA primary596572 after boot/start/death verification, fresh isolated root, all8 startup exact-zero gates before READY/admission; preserve other3live workers and all original inference claims; never replay interrupted query',
        invariants='same warm history/WM/search/population/horizons/CRN/library/Thompson/acceptance; deadline unchanged; no SUMO dispatch or budget reallocation authorised by this amendment',
        budget_block=dict(allocated=7495040,cap=7500000,spare=4960,interrupted_B=10,interrupted_R=11,replacement_sim_s=25200,additional_room_needed=20240,
            action='MAIN allocation decision required before replacing interrupted physics; retain original charges/results; never silently resume claimed incomplete episodes'))
    write(target,value);h=sha(target)
    line='\nS-COOPMWM MAIN0944 ENGINEERING REPAIR REGISTERED amendmentSHA'+h+': '+json.dumps({k:v for k,v in value.items() if k!='cpu'},sort_keys=True)+'\n'
    with ops.checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'INCIDENT0944_REGISTRATION.json',dict(amendment_sha256=h,line=line))
    print(json.dumps(dict(registered=True,amendment_sha256=h,new_sumo_sim_s=0,budget_block=value['budget_block'])))


def gpu_stage():
    from .exact_ops import GPU as original
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261102_incidentcpu1.json')
    reg=read(ops.EVIDENCE/'INCIDENT0944_REGISTRATION.json')
    target=ops.EVIDENCE/'INCIDENT0944_GPU_STAGE.json';assert not target.exists()
    old=json.loads(ops.ssh.remote('gpuA',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({original!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_rpc import status
v=status(r);assert not v['alive'] and v['failed']
assert v['receipts']['SPOOL_OWNER.json']['pid']==596572
assert v['receipts']['SPOOL_FAILED.json']['error']=="ValueError('complete observable group of two or three required')"
print(json.dumps(dict(status=v,binding=json.loads((r/'GPU_BINDING.json').read_text()),inputs=(r/'INPUTS.json').read_text(),failure_traceback=(r/'SPOOL.log').read_text()[-6000:],unfinished=[p.stem for p in (r/'inference_claims').glob('*.json') if not (r/'forecasts'/p.name).exists()])))''',timeout=35))
    staged=json.loads(ops.stage_frozen('gpuA',GPU,cpu))
    extra=json.loads(ops.stage('gpuA',GPU,{'INPUTS.json':old['inputs']}))
    binding=dict(old['binding'],hashes=dict(staged['hashes'],**extra['hashes']),qualification=cpu,
        incident_amendment_sha256=reg['amendment_sha256'])
    binding.pop('adopted_startup_gates',None)
    ops.stage('gpuA',GPU,{'GPU_BINDING.json':json.dumps(binding,sort_keys=True)})
    write(target,dict(root=GPU,binding=binding,old=old,new_sumo_sim_s=0))
    print(json.dumps(dict(staged=True,root=GPU,unfinished_original_claims=old['unfinished'],deadline_unix=binding['deadline_unix'])))


def gpu_dispatch():
    from . import concurrent_ops as c
    stage=read(ops.EVIDENCE/'INCIDENT0944_GPU_STAGE.json')
    assert read(ops.EVIDENCE/'INCIDENT0944_REGISTRATION.json')['amendment_sha256']
    # Reuse the existing identity-checked, exclusive, one-use detached launcher.
    oldread=c.read;oldwrite=c.write;c.GPU=GPU
    c.read=lambda p:stage if p.name=='CONCURRENT_STAGE_gpuA.json' else oldread(p)
    c.write=lambda p,v:oldwrite(p.with_name('INCIDENT0944_GPU_DISPATCH.json') if p.name=='CONCURRENT_DISPATCH_gpuA.json' else p,v)
    try:c.dispatch('gpuA')
    finally:c.read=oldread;c.write=oldwrite


def gpu_status():
    value=json.loads(ops.ssh.remote('gpuA',f'''import pathlib,json,sys,time,subprocess
r=pathlib.Path({GPU!r});sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_rpc import status
v=status(r)
for n in ('STARTUP_COMMITTED_K4_H300.json','STARTUP_COMMITTED_K8_H420.json'):
 if (r/n).exists():v['receipts'][n]=json.loads((r/n).read_text())
v.update(unix=time.time(),log_tail=(r/'SPOOL.log').read_text()[-4000:],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu,power.draw','--format=csv,noheader'],text=True))
print(json.dumps(v))''',timeout=40))
    write(ops.EVIDENCE/f'INCIDENT0944_GPU_STATUS_{int(value["unix"])}.json',value)
    print(json.dumps(dict(alive=value['alive'],ready=value['ready'],failed=value['failed'],gpu=value['gpu'],
        gates={n:dict(pass_=v['pass_'],max_cost_error=v['max_cost_error']) for n,v in value['receipts'].items() if n.startswith('STARTUP_')},log_tail=value['log_tail'] if value['failed'] else None)))
    return value


def cpu():
    prior=read(ops.EVIDENCE/'CPU_coopmwm20261102_tailcpu3.json')
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({prior['root']!r});files={{}}
for name,h in json.loads({json.dumps(prior['source_hashes'])!r}).items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=40))
    for name in OVERLAYS:
        files['tools/coopmwm/'+name+'.py']=ops.checked(ops.LOCAL/'tools/coopmwm'/(name+'.py')).read_text(encoding='utf-8')
    staged=json.loads(ops.stage('cpuQ',SOURCE,files))
    out=ops.ssh.remote('cpuQ',f'''import sys,os,json,unittest,time
from unittest.mock import patch
sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',COOPMWM_COST_SOURCE={SOURCE!r}+'/FROZEN_COSTS.json',COOPMWM_MOTION_SOURCE={SOURCE!r}+'/FROZEN_MOTION_SOURCE.py')
import torch;torch.set_num_threads(1)
names=('test_incident_cpu','test_mppi_runtime_cpu','test_mppi_search_cpu','test_capacity_broker_cpu','test_parallel_groups_cpu','test_tail_runtime_cpu','test_warm_health_cpu','test_warm_spool_cpu','test_exact_grade_cpu','test_awr_cpu')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.'+n) for n in names);start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('no SUMO/inference during CPU qualification')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),sim_s=0,root={SOURCE!r},wall_s=time.monotonic()-start)))''',timeout=90)
    print(out);value=json.loads(out.strip().splitlines()[-1]);value['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'),value)
    assert value['pass_'] and value['sim_s']==value['skips']==0


def census():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os
r=pathlib.Path({ROOT!r});result={{}}
for arm,u in [('B',3),('R',2)]:
 jobs=[]
 for p in sorted((r/'jobs').glob(f'warm_u{{u:03}}_{{arm}}_*')):
  jobs.append(dict(id=p.name,closed=(p/'RESULT.json').exists(),claim=json.loads((p/'CLAIM.json').read_text()) if (p/'CLAIM.json').exists() else None,files=[x.name for x in p.iterdir() if x.is_file() and x.suffix in ('.json','.jsonl','.pt')]))
 result[arm]=jobs
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(bytes([0]))
  if any(b'tools.coopmwm.' in a for a in args):processes.append(dict(pid=int(p.name),args=[a.decode(errors='replace') for a in args if a]))
 except OSError:pass
print(json.dumps(dict(unix=time.time(),jobs=result,processes=processes,new_sumo_sim_s=0)))''',timeout=40))
    write(ops.EVIDENCE/f'INCIDENT0944_CENSUS_{int(value["unix"])}.json',value)
    print(json.dumps(dict(unix=value['unix'],jobs={a:dict(total=len(j),closed=sum(x['closed'] for x in j),interrupted=sum(x['claim'] is not None and not x['closed'] for x in j),unclaimed=sum(x['claim'] is None for x in j)) for a,j in value['jobs'].items()},processes=value['processes'])))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cpu','census','register','gpu_stage','gpu_dispatch','gpu_status'));globals()[p.parse_args().mode]()
