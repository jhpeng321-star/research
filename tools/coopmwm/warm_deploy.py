"""Explicit source-bound warm namespace admission; one-use per-arm dispatch."""
import argparse
import json
from . import ops
from .protocol import checked,sha,REGISTRY
from .storage import read,write
from .warm_protocol import budget

ROOT='/home/ubuntu/coopmwm20261102_warm1'
PRIOR='/home/ubuntu/coopmwm20261101_accel1'
CPU='/home/ubuntu/coopmwm20261102_mppicpu6'


def prepare():
    cpu=read(ops.EVIDENCE/('CPU_'+CPU.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and not cpu['skips'] and not cpu['sim_s']
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({CPU!r});hashes=json.loads({json.dumps(cpu['source_hashes'])!r});files={{}}
for name,h in hashes.items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 assert name.startswith('pytools/tools/') or name in ('COOPMWM_DESIGN.md','FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py')
 files[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(files))''',timeout=35))
    staged=json.loads(ops.stage('cpuQ',ROOT,files))
    assert staged['hashes']==cpu['source_hashes']
    amendment=dict(scope=read(ops.EVIDENCE/'WARM2354_REGISTRATION.json'),cpu=cpu,
        source_hashes=staged['hashes'],root=ROOT,prior=PRIOR,A='resume exact A u4/Adam/RNG to fixedu48',
        B_R='hold until warm release; newu0 checkpoints only',budget=budget(),new_sumo_sim_s=0)
    write(ops.EVIDENCE/'WARM_A48_AMENDMENT.json',amendment)
    h=sha(ops.EVIDENCE/'WARM_A48_AMENDMENT.json')
    line=('\nS-COOPMWM WARM A48 CONTINUATION REGISTERED amendmentSHA'+h+
        ': new isolated source-bound warm1 namespace, exact A u4 checkpoint/Adam/RNG preserved, '
        'only A continuesu5..48 under MAIN2354 allocation6473440/cap6500000. All988040currentcharges copied/reclassified '
        'without refund; old coldB28800/failedA14400 retained separately; old supervisor/children stayheld. '
        'TRAIN50%focus,checks4/dev16; no final panel dispatch; B/R require additional warm GPU release. No retry/duplicate SUMO.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    authority=checked(REGISTRY).read_text(encoding='utf-8')
    write(ops.EVIDENCE/'WARM_A48_REGISTRATION.json',dict(line=line,amendment_sha256=h))
    metadata=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib,shutil,sys
r=pathlib.Path({ROOT!r});old=pathlib.Path({PRIOR!r});assert r.is_dir() and old.is_dir()
sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.accel_storage import read,write
assert not (r/'WARM_BINDING.json').exists() and not (r/'CHARGES.jsonl').exists()
assert read(old/'A_WARM5_CONTINUATION_PAUSED.json')['update']==4
d=read(old/'TRAIN_DISPATCH.json');p=pathlib.Path('/proc')/str(d['pid'])
s=(p/'stat').read_text().split();assert s[21]==d['start_ticks'] and s[2]=='T'
proof={{}}
for name in ['A_checkpoint_004.pt','SMOKE.json','MAPPO_CHECK.json','MAPPO_DEV.json']+[f'A_UPDATE_{{i:03}}.json' for i in range(1,5)]:
 source=old/name;target=r/name;assert not target.exists()
 shutil.copyfile(source,target);proof[name]=hashlib.sha256(target.read_bytes()).hexdigest()
 assert proof[name]==hashlib.sha256(source.read_bytes()).hexdigest()
charges=[json.loads(x) for x in (old/'CHARGES.jsonl').read_text().splitlines()]
for c in charges:
 if c['phase']=='train' and '_B_' in c['id']:c['phase']='cold_B_retained'
 elif c['phase']=='train' and '_A_' in c['id'] and (old/'jobs'/c['id']/'CLOSE.json').exists() and not (old/'jobs'/c['id']/'RESULT.json').exists():c['phase']='failed_A_retained'
with (r/'CHARGES.jsonl').open('x') as f:
 for c in charges:f.write(json.dumps(c,sort_keys=True)+'\\n')
assert 811640+sum(c['sim_s'] for c in charges)==988040
records=[]
for seed in range(986072,986084):
 e=read(old/'jobs'/f'check004_A_A_{{seed}}'/'RESULT.json')
 records.append({{k:e[k] for k in ('id','world','arm','metrics')}})
write(r/'A_CHECK_RECORDS_004.json',records)
for name in ('requests','responses','guided_starts','random_match','worker_claims'):(r/name).mkdir(exist_ok=True)
oldbinding=read(old/'BINDING_TRAINING.json')
value=dict(proof=proof,charged_sim_s=988040,oldbinding=oldbinding,reclassified_charges=charges,new_sumo_sim_s=0)
write(r/'WARM_PRESERVED.json',value);print(json.dumps(value))''',timeout=45))
    old=metadata['oldbinding']
    b=dict(budget=budget(),cpu=cpu,hashes=staged['hashes'],workers=4,
        runtime=old['runtime'],checkpoint=old['checkpoint'],accel_bounds=old['accel_bounds'],
        gpu_deadline=old['gpu_deadline'],N_exec=10,warm_history=True,
        cpu_host='cpuQ',scope_amendment_sha256=amendment['scope']['amendment_sha256'],
        release_amendment_sha256=h,authority=old.get('authority'),
        admission=old.get('admission',{}))
    # Preserve the inherited admission fields used by runner.admission.
    for name in ('workers','runtime','checkpoint','accel_bounds','gpu_deadline'):
        assert name in b
    b.update({k:v for k,v in old.items() if k in ('limits','load_max','ram_min_bytes','disk_min_bytes')})
    ops.stage('cpuQ',ROOT,{'WARM_BINDING.json':json.dumps(b,sort_keys=True),'AUTHORITY.md':authority})
    write(ops.EVIDENCE/'WARM_A48_BINDING.json',b)
    write(ops.EVIDENCE/'WARM_A48_PRESERVED.json',metadata)
    print(json.dumps(dict(root=ROOT,amendment_sha256=h,charged_sim_s=988040,new_sumo_sim_s=0)))


def dispatch(arm):
    b=read(ops.EVIDENCE/'WARM_A48_BINDING.json')
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,os,subprocess,time,sys
r=pathlib.Path({ROOT!r});assert r.is_dir()
assert not (r/{('WARM_'+arm+'_DISPATCH.json')!r}).exists()
sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_runner import verify
verify(r,{arm!r})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
assert pathlib.Path(env['SUMO_HOME'],'bin/sumo').is_file()
cmd=[{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.warm_runner',str(r),'--arm',{arm!r}]
with (r/{('WARM_'+arm+'.log')!r}).open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),arm={arm!r},root=str(r),unix=time.time())
with (r/{('WARM_'+arm+'_DISPATCH.json')!r}).open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=35))
    write(ops.EVIDENCE/f'WARM_{arm}_DISPATCH2.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','dispatch'));p.add_argument('--arm',choices=('A','B','R'),default='A')
    a=p.parse_args();prepare() if a.mode=='prepare' else dispatch(a.arm)
