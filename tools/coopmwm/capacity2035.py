"""Outcome-independent capacity admission; retain completed measurements, defer slower trials."""
import argparse
import json
from . import ops
from .storage import read,write
from .protocol import REGISTRY,checked,sha


def register():
    wide={h:read(ops.EVIDENCE/f'PLAN_PROBE3_{h}_COMPLETE.json') for h in ('gpuA','gpuR')}
    assert all(v['pass_'] for v in wide.values())
    one=read(ops.EVIDENCE/'SCALE2035_gpuA_CONCURRENCY_1.json')
    two=read(ops.EVIDENCE/'SCALE2035_gpuA_CONCURRENCY_2.json')
    assert two['plans_per_hour']<one['plans_per_hour']<wide['gpuA']['plans_per_hour']
    value=dict(hosts={h:dict(pass_=True,best=dict(v,workers=1,source_bound_full_search_parity=True),
        batch_widths={'4':34,'8':10},qualification=read(ops.EVIDENCE/f'PLAN_PROBE3_{h}_PARITY.json')) for h,v in wide.items()},
        reason='OWNER20:35 restart now + early viability priority: measured two-worker regression, use faster already-qualified wide batches',
        deferred='uncompleted fixed4 concurrency4/8 trials resume as NEW inference-only measurements after u4 owner-review pause; completed1/2 results reused',
        selected_by='completed plans/hour, never reward/advantage/learning outcome',
        original_gpu_deadline_not_extended=True,new_sumo_sim_s=0)
    write(ops.EVIDENCE/'CAPACITY2035_ADMISSION.json',value)
    h=sha(ops.EVIDENCE/'CAPACITY2035_ADMISSION.json')
    line=(f'\nS-COOPMWM EARLY VIABILITY CAPACITY AMENDMENT REGISTERED BEFORE TRAINING: admissionSHA{h}; '
        'OWNER20:35 restartnow/qualitymedium/earlyevidence priority; measured GPUA fixed4 concurrency2 '
        f'{two["plans_per_hour"]:.8f}plans/h <one{one["plans_per_hour"]:.8f}; '
        f'qualified wide1GPUA{wide["gpuA"]["plans_per_hour"]:.8f}/GPUR{wide["gpuR"]["plans_per_hour"]:.8f}plans/h, '
        'same96candidates/3CEMiterations/488futures/K4H120/K8H300, full1600/source/controller; '
        'both K4width34/K8width10 single-vs-batch parity EXACT0. '
        'Admit one wide persistent worker/GPU now; once/newworker K4/K8 qualification and <=65%memory AFTER bootstrap. '
        'Defer unfinished slower fixed4 scale trials to AFTER u4 viability pause: verified own scale parent/descendant identities '
        'stopped, incomplete inference receipts retained as deferred; completed1/2 retained/no rerun; '
        'no SUMO job killed/repeated, no other PIDs touched. Complete deferred4/8 measurements under new inference dispatches '
        'after viability, not a training gate. Nexec10minimum/R count-regime-duration matching/mediumsearch/trust/reward '
        'unchanged; actual sampling still queue-capacity only; no efficacy outcome used. '
        'Primary A/Bu1..4 thenRu1..4/pause, all budget4280840/6000000/free1719160, original GPUdeadline and sealed96 rules unchanged.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'CAPACITY2035_REGISTRATION.json',dict(line=line,admission_sha256=h));print(line.strip())


def defer(host):
    assert (ops.EVIDENCE/'CAPACITY2035_REGISTRATION.json').exists()
    d=read(ops.EVIDENCE/f'SCALE2035_{host}_DISPATCH.json')
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,os,signal,time,subprocess
r=pathlib.Path({d['root']!r});d=json.loads((r/'DISPATCH.json').read_text());receipt=r/'DEFERRED_FOR_VIABILITY.json';assert not receipt.exists()
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==d['boot_id']
p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split()
assert s[21]==d['start_ticks'] and 'tools.coopmwm.plan_scale' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
os.kill(d['pid'],signal.SIGSTOP)
stats={{}}
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try:stats[int(p.name)]=(p/'stat').read_text().split()
  except (OSError,PermissionError):pass
owned={{d['pid']}};changed=True
while changed:
 previous=len(owned)
 owned.update(pid for pid,s in stats.items() if int(s[3]) in owned)
 changed=len(owned)!=previous
identities=[]
for pid in sorted(owned):
 p=pathlib.Path('/proc')/str(pid)
 if not (p/'stat').exists():continue
 s=(p/'stat').read_text().split()
 assert int(s[21])>=int(d['start_ticks'])
 assert pid==d['pid'] or int(s[3]) in owned
 if pid!=d['pid']:os.kill(pid,signal.SIGSTOP)
 identities.append(dict(pid=pid,start_ticks=s[21],ppid=int(s[3]),cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()))
v=dict(unix=time.time(),host={host!r},owner=d,identities=identities,action='SIGSTOP identity-verified own tree, then SIGKILL inference-only owned children/parent; no physics',
 reason='registered early viability priority; completed timings preserved; unfinished scale trials deferred',
 completed={{p.name:json.loads(p.read_text()) for p in r.glob('CONCURRENCY_*.json')}},new_sumo_sim_s=0)
with receipt.open('x') as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
for vpid in sorted(identities,key=lambda v:v['pid']==d['pid']):
 p=pathlib.Path('/proc')/str(vpid['pid'])
 if not (p/'stat').exists():continue
 assert (p/'stat').read_text().split()[21]==vpid['start_ticks']
 os.kill(vpid['pid'],signal.SIGKILL)
time.sleep(1)
v['gpu_inventory_after']=subprocess.check_output(['nvidia-smi'],text=True)
print(json.dumps(v))
''',timeout=30))
    write(ops.EVIDENCE/f'CAPACITY2035_{host}_DEFERRED.json',value)
    for name,result in value['completed'].items():
        p=ops.EVIDENCE/f'SCALE2035_{host}_{name}'
        if not p.exists():write(p,result)
    print(json.dumps(value))


def audit_wide(cpu_name):
    cpu=read(ops.EVIDENCE/cpu_name)
    from .viability2035 import dependency_names
    expected={name:h for name,h in cpu['source_hashes'].items() if name in dependency_names(cpu)}
    results={}
    for host in ('gpuA','gpuR'):
        results[host]=json.loads(ops.ssh.remote(host,f'''import pathlib,json,hashlib
r=pathlib.Path('/root/coopmwm20261101_planprobe3');expected=json.loads({json.dumps(expected)!r});different=[]
for name,h in expected.items():
 p=r/name;data=p.read_bytes()
 if name=='FROZEN_COSTS.json':data=(p.read_text().strip()+'\\n').encode()
 if hashlib.sha256(data).hexdigest()!=h:different.append(name)
print(json.dumps(dict(pass_=not different,different=different,dependencies=len(expected))))
''',timeout=20))
    value=dict(cpu=cpu_name,hosts=results,pass_=all(v['pass_'] for v in results.values()),new_sumo_sim_s=0)
    write(ops.EVIDENCE/'CAPACITY2035_WIDE_AUDIT.json',value);print(json.dumps(value));assert value['pass_']


def deferred_after_pause(cpu_name):
    """No new physics. Start only after PPO is paused and owned services release GPUs."""
    dispatches={h:ops.EVIDENCE/f'CAPACITY_DEFERRED_{h}_DISPATCH.json' for h in ('gpuA','gpuR')}
    if not all(p.exists() for p in dispatches.values()):
        for host in ('gpuA','gpuR'):
            ready=read(ops.EVIDENCE/f'ACCEL_GPU_{host}@0_READY.json')
            ids=ops.ssh.remote(host,"import subprocess;print(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True))",timeout=20)
            if str(ready['pid']) in ids.split():return 'WAIT_OWN_GPU_SERVICE_RELEASE'
        cpu=read(ops.EVIDENCE/cpu_name)
        for host,dest in dispatches.items():
            if dest.exists():continue
            root='/root/coopmwm20261101_scaledeferred1';ops.stage_frozen(host,root,cpu)
            old=read(ops.EVIDENCE/f'CAPACITY2035_{host}_DEFERRED.json')
            reference=ops.ssh.remote(host,"import pathlib,json;print(json.dumps(json.loads(pathlib.Path('/root/coopmwm20261101_scale1/C1_W0.json').read_text())['result']))",timeout=20).strip()
            ops.stage(host,root,{'REFERENCE.json':reference,'INPUTS.json':json.dumps(read(ops.EVIDENCE/'VOLUME_INPUTS_1854.json')),
                'MODE.json':json.dumps(dict(levels=[4,8] if host=='gpuA' else [4],completed=list(old['completed'].values()),
                    deadline_unix=read(ops.EVIDENCE/'DUAL_1755_DISPATCH.json')['unix']+86400))})
            value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,os,subprocess,sys,time
r=pathlib.Path({root!r});assert r.is_dir();assert not (r/'DISPATCH.json').exists()
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'SCALE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.plan_scale','run','--root',str(r),'--host',{host!r}],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
v=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),root=str(r),host={host!r},new_sumo_sim_s=0)
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))
''',timeout=25))
            write(dest,value)
    from .viability2035 import publish_receipts
    done=True
    for host,dest in dispatches.items():
        d=read(dest)
        receipts=json.loads(ops.ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({d['root']!r});d=json.loads((r/'DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or (s[21]==d['start_ticks'] and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'])
out={{p.name:json.loads(p.read_text()) for pattern in ('CONCURRENCY_*.json','COMPLETE.json','FAILED.json','C*_NOT_ADMITTED.json') for p in r.glob(pattern)}}
if not s and not any(k in out for k in ('COMPLETE.json','FAILED.json')):out['UNFINISHED_EXIT.json']={{'owner':d,'reason':'no automatic retry'}}
print(json.dumps(out))
''',timeout=25))
        publish_receipts({f'DEFERRED_{host}_{n}':v for n,v in receipts.items()},'accel_scale')
        done=done and any(k in receipts for k in ('COMPLETE.json','FAILED.json','UNFINISHED_EXIT.json'))
    return 'PAUSED' if done else 'PAUSED_CAPACITY_MEASUREMENT'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','defer','audit-wide'));p.add_argument('--host');p.add_argument('--cpu');a=p.parse_args()
    register() if a.mode=='register' else audit_wide(a.cpu) if a.mode=='audit-wide' else defer(a.host)
