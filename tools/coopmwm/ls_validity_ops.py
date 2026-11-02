"""Receipt-adopting MAIN21:27 recovery: model bounds, query isolation, fresh gates."""
import argparse
import base64
import json
import zlib
from . import ops
from .protocol import checked, sha, REGISTRY
from .storage import read, write
from .ls_collect_ops import COLLECTION, ROOT
from .ls_recover import RELEASE as BASE

SOURCE='/home/ubuntu/coopmwm20261102_lsvalidcpu3'
SUFFIX='_VALIDITY1'


def remote(host, code, timeout=40):
    from .warm_transport import retry
    return retry(lambda:ops.ssh.remote(host,code,timeout=timeout),lambda:None,
        lambda v:write(ops.EVIDENCE/f'LS_VALIDITY_TRANSPORT_{__import__("time").time_ns()}.json',v))


def payload(names):
    files={n:checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8') for n in names}
    return base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()


def qualify():
    packed=payload(('ls_collection','ls_gpu','ls_broker','ls_validity','test_ls_validity_cpu','test_incident_cpu','plan_service','query_errors'))
    out=remote('cpuQ',f'''import pathlib,json,os,time,shutil,hashlib,subprocess,zlib,base64
r=pathlib.Path({SOURCE!r});base=pathlib.Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 shutil.copytree(base,r,ignore=shutil.ignore_patterns('LS_CPU*','LS_OVERLAY*','LS_SOURCE_HASHES.json','__pycache__'))
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['test_ls_validity_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('w') as f:json.dump(sorted(set(tests)),f)
for n,text in files.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(json.dumps(dict(result=json.loads(done.read_text()))))
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
 with (r/'LS_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_cpu_test',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))''')
    print(out)


def cpu_result():
    out=remote('cpuQ',f'''from pathlib import Path
import json
r=Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
print(p.read_text() if p.exists() else json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-4000:])))''')
    v=json.loads(out)
    if v.get('pending'):print(out);return
    p=ops.EVIDENCE/('LS_VALIDITY_CPU_'+SOURCE.rsplit('/',1)[1]+'.json')
    if not p.exists():write(p,v)
    print(json.dumps({k:v[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
    if not v['pass_']:print(remote('cpuQ',f'from pathlib import Path; print(Path({SOURCE!r},"LS_CPU.log").read_text()[-16000:])'))


def register():
    cpu=read(ops.EVIDENCE/('LS_VALIDITY_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'));assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    # Read ONLY the observable-arrival fit metadata, never any hidden assignment.
    out=remote('gpuR',f'''from pathlib import Path
import hashlib,json
p=Path('/root/rbeval20261031/phys_code_22cc32f65a80/docs/paper_results/rbunc/observed_arrivals.json');a=json.loads(p.read_text())
print(json.dumps(dict(artifact={{k:a[k] for k in ('schema','generator_family','conditioning_end_s','observed_phase_end_s')}},path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))''')
    model=json.loads(out)
    from .ls_validity import contract
    from .mppi_search import REGISTERED
    admission=contract(model['artifact'],REGISTERED)
    target=ops.EVIDENCE/'LS_VALIDITY_REGISTRATION.json'
    if not target.exists():
        line=('\nS-COOPMWM MAIN21:27/21:29 MODEL-VALIDITY ENGINEERING AMENDMENT: '
            'all four recovery workers exited on forecast beyond fitted arrival phase; '
            'read observed arrival artifact bound, admit only t0+max(Hscreen,Hfinal)<=phase_end. '
            'No horizon/population/K/iterations changes; K4H300 screen/K8Hfinal retained. '
            'Existing72episodes/14534snapshots adopted, no episode replay. Interrupted inference '
            'claims terminally declined with original failure traceback, never replayed; all charges retained. '
            'Per-query ValueError/AssertionError/OOM decline; safety/physics/accounting/parity remain fatal. '
            'Overlay error boundary SHA-bound; original GPU bindings/sources/receipts immutable; '
            'ALL8fresh startup gates per new process before admission, four slots. '
            'Collection only; B/Rremainu3, no B-REPLAY update until separate qualified integration '
            'and20positiveBbank entries. Existing LS envelope/phases/cap11900000 unchanged, '
            'restricted k8training/interim not spent. No new episode physics.\n')
        write(target,dict(line=line,cpu=cpu,model=model,arrival_admission=admission,source_root=SOURCE,
            authority='MAIN2026-11-02 21:27/21:29',new_episode_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(target)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(target),arrival_admission=admission)))


def gpu(host):
    reg=read(ops.EVIDENCE/'LS_VALIDITY_REGISTRATION.json');h=sha(ops.EVIDENCE/'LS_VALIDITY_REGISTRATION.json')
    rows=read(ops.EVIDENCE/('LS_GPU_RECOVERY_DISPATCH_'+host+'.json'))['owners']
    packed=payload(('ls_gpu','plan_service','query_errors'))
    import hashlib
    for n in ('ls_gpu','plan_service','query_errors'):
        text=checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8')
        assert hashlib.sha256(text.encode()).hexdigest()==reg['cpu']['source_hashes']['pytools/tools/coopmwm/'+n+'.py']
    out=remote(host,f'''import pathlib,json,os,time,hashlib,subprocess,base64,zlib
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
base=pathlib.Path('/root/coopmwm20261102_lsvalidops1');assert base.parent.is_dir();base.mkdir(exist_ok=True)
for n,text in files.items():
 p=base/(n+'.py')
 if p.exists():assert p.read_text()==text,n
 else:p.write_text(text)
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();owners=[]
for d in {rows!r}:
 r=pathlib.Path(d['root']);dest=r/'LS_GPU_DISPATCH_VALIDITY1.json'
 if dest.exists():owners.append(json.loads(dest.read_text()));continue
 assert boot==d['boot_id'];p=pathlib.Path('/proc',str(d['pid']))
 if p.exists():
  st=(p/'stat').read_text().split();assert st[21]==d['start_ticks'] and st[2]=='Z','existing worker must be adopted'
 failed=json.loads((r/'LS_GPU_FAILED_RECOVERY1.json').read_text())
 assert 'forecast exceeds observed/documented fitted demand phase' in failed['error']
 for name in ('inbox','inference_claims'):
  for q in (r/name).glob('*.json'):
   target=r/'forecasts'/q.name
   if target.exists():continue
   record=json.loads((r/'inbox'/q.name).read_text())
   import sys;sys.path.insert(0,str(r/'pytools'))
   from tools.coopmwm.protocol import digest
   value=dict(accepted=False,record_sha256=digest(record),compute_counts_unknown=True,
    reason='terminal interrupted query; never replayed',query_error=dict(kind='code',error=failed['error'],
    traceback=(r/'LS_GPU_RECOVERY1.log').read_text()[-6000:],inference_retry=False))
   with target.open('x') as f:json.dump(dict(key=q.stem,result=value,wall_s=None,unix=time.time(),interrupted=True,original_failure=failed),f)
 overlay_hashes={{n:hashlib.sha256((base/(n+'.py')).read_bytes()).hexdigest() for n in files}}
 with (r/'LS_VALIDITY_OVERLAY.json').open('x') as f:json.dump(dict(source_root=str(base),hashes=overlay_hashes,amendment_sha256={h!r}),f)
 old=dict(d,argv='')
 code='import sys;sys.path.insert(0,'+repr(str(base))+');from ls_gpu import run;run('+repr(str(r))+','+repr(old)+',1793894340.,'+repr({h!r})+',recover=True,generation="_VALIDITY1",overlay='+repr(str(base))+')'
 env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 if {host!r}=='gpuR':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
 with (r/'LS_GPU_VALIDITY1.log').open('x') as log:
  proc=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=pathlib.Path('/proc',str(proc.pid),'stat').read_text().split()
 v=dict(pid=proc.pid,start_ticks=st[21],boot_id=boot,root=str(r),unix=time.time(),host={host!r},overlay_hashes=overlay_hashes)
 with dest.open('x') as f:json.dump(v,f)
 owners.append(v)
print(json.dumps(dict(host={host!r},owners=owners)))''')
    v=json.loads(out);p=ops.EVIDENCE/('LS_VALIDITY_GPU_DISPATCH_'+host+'.json')
    if not p.exists():write(p,v)
    print(out)


def gpu_status(host):
    rows=read(ops.EVIDENCE/('LS_VALIDITY_GPU_DISPATCH_'+host+'.json'))['owners']
    out=remote(host,f'''from pathlib import Path
import json,time,subprocess
workers=[]
for d in {rows!r}:
 r=Path(d['root']);p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 assert not st or st[21]==d['start_ticks']
 gates={{p.name:json.loads(p.read_text()) for p in r.glob('STARTUP_*_LSCOLLECT_VALIDITY1.json')}}
 service=json.loads((r/'SERVICE_OWNER.json').read_text());failure=r/'LS_GPU_FAILED_VALIDITY1.json'
 ready=st is not None and st[2] in ('S','R','D') and not failure.exists() and len(gates)==8 and all(x['pass_'] and x['max_cost_error']==0 for x in gates.values()) and all(service[k]==d[k] for k in ('pid','start_ticks','boot_id'))
 completed=[json.loads(p.read_text()) for p in (r/'forecasts').glob('*.json') if p.stat().st_mtime>=d['unix']]
 workers.append(dict(owner=d,state=st[2] if st else 'exited',ready=ready,gates=gates,
  failure=json.loads(failure.read_text()) if failure.exists() else None,
  new_forecasts=len(completed),successful_searches=sum(x['result'].get('proposer_receipt') is not None for x in completed),
  log_tail=(r/'LS_GPU_VALIDITY1.log').read_text()[-1800:]))
print(json.dumps(dict(host={host!r},unix=time.time(),workers=workers,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True))))''')
    v=json.loads(out);write(ops.EVIDENCE/f'LS_VALIDITY_GPU_{host}_{int(v["unix"])}.json',v)
    print(json.dumps(dict(host=host,gpu=v['gpu'],workers=[dict(owner=w['owner'],state=w['state'],ready=w['ready'],gates=len(w['gates']),failure=w['failure'],successful_searches=w['successful_searches'],log_tail=w['log_tail']) for w in v['workers']])))
    return v


def cpu():
    reg=read(ops.EVIDENCE/'LS_VALIDITY_REGISTRATION.json');h=sha(ops.EVIDENCE/'LS_VALIDITY_REGISTRATION.json')
    # Metadata owners only: no episode/job process is signalled or restarted.
    out=remote('cpuQ',f'''from pathlib import Path
import os,json,sys,time,signal,subprocess,hashlib
c=Path({COLLECTION!r});r=Path({ROOT!r});s=Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
reg=json.loads({json.dumps(reg)!r})
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
assert all(not (r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R'))
assert all((r/'jobs'/('lscollect1_B3_'+str(w))/'RESULT.json').exists() for w in range(986000,986072))
cfg=dict(run_root=str(r),source_root=str(s),cpu=reg['cpu'],amendment_sha256={h!r},receipt_suffix='_VALIDITY1',gpu_deadline=1793894340.,training_paused=True,arrival_admission=reg['arrival_admission'])
immutable(c/'LS_VALIDITY_CONFIG.json',cfg)
registered=c/'LS_VALIDITY_REGISTERED.json'
if not registered.exists():
 with (r/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+'\n')
 immutable(registered,dict(amendment_sha256={h!r},unix=time.time()))
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for component,module in (('BROKER','ls_broker'),('SUPERVISOR','ls_collection')):
 dest=c/(component+'_DISPATCH_VALIDITY1.json')
 if dest.exists():continue
 d=json.loads((c/(component+'_DISPATCH_RECOVERY1.json')).read_text());p=Path('/proc',str(d['pid']))
 assert boot==d['boot_id']
 if p.exists():
  st=(p/'stat').read_text().split();assert st[21]==d['start_ticks']
  if st[2]!='Z':
   argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
   assert 'tools.coopmwm.'+module in argv and str(c) in argv
   # Metadata daemons must have no live child processes before replacement.
   children=(p/'task'/str(d['pid'])/'children').read_text().split()
   for child in children:
    q=Path('/proc',child)
    assert 'multiprocessing.resource_tracker' in (q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(),child
   os.kill(d['pid'],signal.SIGTERM)
   for _ in range(100):
    if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
    time.sleep(.1)
   else:raise ValueError('metadata owner failed to exit')
 immutable(c/(component+'_RETIRED_VALIDITY1.json'),dict(old=d,unix=time.time(),no_physics_signals=True))
# Settle only interrupted assignments and invalid unassigned snapshots; keep receipts/charges.
from tools.coopmwm.ls_validity import feasible
from tools.coopmwm.protocol import digest
for p in (c/'admissions').glob('*.json'):
 target=r/'responses'/p.name
 if target.exists():continue
 record=json.loads((c/'requests'/p.name).read_text());assigned=c/'broker_assignments'/p.name
 if not assigned.exists() and feasible(record['time_s'],cfg['arrival_admission']):continue
 why='interrupted inference; no replay' if assigned.exists() else 'outside documented arrival phase; not submitted'
 value=dict(accepted=False,record_sha256=digest(record),compute_counts_unknown=True,reason=why,
  query_error=dict(kind='code',error=why,traceback='MAIN21:27 admission recovery; original GPU failure receipts retained',inference_retry=False))
 (c/'terminal_declines').mkdir(exist_ok=True)
 immutable(target,value);immutable(c/'terminal_declines'/(p.stem+'.json'),dict(key=p.stem,reason=why,charges_retained=True,unix=time.time()))
if not (c/'BROKER_DISPATCH_VALIDITY1.json').exists():
 tmp=c/'LS_VALIDITY_EMPTY_ENDPOINTS.publishing'
 with tmp.open('x') as f:json.dump(dict(workers=[],amendment_sha256={h!r},unix=time.time()),f)
 os.replace(tmp,c/'LS_ENDPOINTS.json')
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
owners={{}}
for component,module in (('BROKER','ls_broker'),('SUPERVISOR','ls_collection')):
 dest=c/(component+'_DISPATCH_VALIDITY1.json')
 if dest.exists():owners[component]=json.loads(dest.read_text());continue
 with (c/(component+'_VALIDITY1.log')).open('x') as log:
  proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.'+module,str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=Path('/proc',str(proc.pid),'stat').read_text().split();d=dict(pid=proc.pid,start_ticks=st[21],boot_id=boot,unix=time.time(),source_root=str(s),root=str(c),component=component)
 immutable(dest,d);owners[component]=d
print(json.dumps(dict(owners=owners,episodes_adopted=72,arrival_admission=cfg['arrival_admission'],training_paused=True)))''',timeout=60)
    v=json.loads(out);p=ops.EVIDENCE/'LS_VALIDITY_CPU_DISPATCH.json'
    if not p.exists():write(p,v)
    print(out)


def admit():
    workers=[]
    for host in ('gpuA','gpuR'):
        p=max(ops.EVIDENCE.glob('LS_VALIDITY_GPU_'+host+'_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        workers.extend(w for w in read(p)['workers'] if w['ready'])
    assert workers,'no eight-gate-qualified slots yet'
    h=sha(ops.EVIDENCE/'LS_VALIDITY_REGISTRATION.json')
    out=remote('cpuQ',f'''from pathlib import Path
import json,sys,time,os
c=Path({COLLECTION!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_broker import Remote
from tools.coopmwm.ls_broker import owner_matches
endpoints=json.loads((c/'LS_ENDPOINT_CANDIDATES.json').read_text())['endpoints'];admitted=[]
for w in {workers!r}:
 d=w['owner'];slot={{'/root/coopmwm20261102_incident1':'gpuA@incident1','/root/coopmwm20261102_deadlineA2':'gpuA@resource1651','/root/coopmwm20261102_concurrent3':'gpuR@0','/root/coopmwm20261102_deadlineR2':'gpuR@resource1651'}}[d['root']]
 e=next(e for e in endpoints if e['slot']==slot)
 assert owner_matches(Remote(c,e).call(dict(action='status')),d),slot
 admitted.append(dict(endpoint=e,owner=d,fresh_gates=w['gates']))
value=dict(workers=admitted,amendment_sha256={h!r},unix=time.time());target=c/'LS_ENDPOINTS.json';tmp=c/'LS_VALIDITY_ENDPOINTS.publishing'
with tmp.open('x') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
os.replace(tmp,target)
print(json.dumps(dict(admitted_slots=[w['endpoint']['slot'] for w in admitted],unix=value['unix'],amendment_sha256={h!r})))''',timeout=120)
    v=json.loads(out);write(ops.EVIDENCE/f'LS_VALIDITY_ADMISSION_{int(v["unix"])}.json',v);print(out)


def status():
    out=remote('cpuQ',f'''from pathlib import Path
import json,time,os,shutil
c=Path({COLLECTION!r});r=Path({ROOT!r});v=dict(unix=time.time(),owners={{}},disk_free=shutil.disk_usage(c).free,load=os.getloadavg(),training_paused=all(not (r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R')))
for name in ('BROKER','SUPERVISOR'):
 d=json.loads((c/(name+'_DISPATCH_VALIDITY1.json')).read_text());p=Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert not st or st[21]==d['start_ticks']
 v['owners'][name]=dict(d,state=st[2] if st else 'exited',log_tail=(c/(name+'_VALIDITY1.log')).read_text()[-1800:])
v['collection']=json.loads((c/'LS_COLLECT_STATUS.json').read_text())
v['broker_completed']=len(list((c/'broker_completed').glob('*.json')))
v['successful_searches']=sum(json.loads(p.read_text()).get('proposer_receipt') is not None for p in (r/'responses').glob('*.json') if (c/'admissions'/p.name).exists())
v['terminal_declines']=len(list((c/'terminal_declines').glob('*.json')))
v['first_searches']=[dict(key=p.stem,**json.loads(p.read_text())) for p in sorted((c/'broker_completed').glob('*.json'),key=lambda p:p.stat().st_mtime)][:2]
print(json.dumps(v))''')
    v=json.loads(out);write(ops.EVIDENCE/f'LS_VALIDITY_STATUS_{int(v["unix"])}.json',v)
    print(out);return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qualify','cpu_result','register','gpu','gpu_status','cpu','admit','status']);p.add_argument('--host',choices=['gpuA','gpuR'])
    a=p.parse_args();globals()[a.action](a.host) if a.action in ('gpu','gpu_status') else globals()[a.action]()
