"""Registered, one-slot-at-a-time joint niche/nullfix rollout; no SUMO dispatch."""
import argparse
import base64
import json
import time
import zlib
from . import ops,niche_ops
from .ls_collect_ops import ROOT,COLLECTION
from .ls_validity_ops import remote
from .protocol import sha,digest,checked
from .storage import read,write

REG='NICHE_NULLFIX_COMPARISON_REGISTRATION.json'
SLOTS={'incident1':'gpuA@incident1','deadlineA2':'gpuA@resource1651',
       'concurrent3':'gpuR@0','deadlineR2':'gpuR@resource1651'}


def worker_command(root,generation='NICHENULL1'):
    # Pure geometry is absent from the GPU WM-only package. Validate the additive
    # source-only dependency package before importing the unchanged worker.
    return ('from pathlib import Path;import json,hashlib;'
        'r=Path('+repr(root)+');'
        'h=json.loads((r/"NICHENULL_DEPENDENCY_HASHES.json").read_text());'
        'assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==s for n,s in h.items());'
        'from tools.coopmwm.niche_worker import run;run('+repr(root)+','+repr(generation)+')')


def dependency_files():
    return {'pytools/'+n:checked(ops.LOCAL/n).read_text(encoding='utf-8') for n in (
        'traffic_shepherd/__init__.py','traffic_shepherd/c4/__init__.py',
        'traffic_shepherd/c4/geometry.py')} | {
        'pytools/tools/coopmwm/'+n+'.py':checked(ops.LOCAL/('tools/coopmwm/'+n+'.py')).read_text(encoding='utf-8')
         for n in ('ls_fast_worker','resource_config','query_errors')}


def frozen_rollout_files(names):
    """Deploy the registered CPU snapshot, never an evolving integration checkout."""
    paths=['pytools/tools/coopmwm/'+n+'.py' for n in names]+['FROZEN_GEOMETRY_SOURCE.py']
    return json.loads(remote('cpuQ',f'''from pathlib import Path
import json,base64,hashlib
r=Path('/home/ubuntu/coopmwm20261103_nichenullcpu7')
manifest=json.loads((r/'LS_SOURCE_HASHES.json').read_text())
names={paths!r}+['FROZEN_NETWORK.net.xml.gz']
for n in names:assert hashlib.sha256((r/n).read_bytes()).hexdigest()==manifest[n],n
print(json.dumps(dict(files={{n:(r/n).read_text() for n in {paths!r}}},network=base64.b64encode((r/'FROZEN_NETWORK.net.xml.gz').read_bytes()).decode())))''',timeout=40))


def broker():
    reg=read(ops.EVIDENCE/REG);h=sha(ops.EVIDENCE/REG)
    out=remote('cpuQ',f'''from pathlib import Path
import os,json,time,sys,signal,subprocess,hashlib
c=Path({COLLECTION!r});s=Path({niche_ops.SOURCE!r});dest=c/'BROKER_DISPATCH_NICHENULL1.json'
if dest.exists():print(dest.read_text());raise SystemExit
reg=json.loads({json.dumps(reg)!r})
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
old=json.loads((c/'BROKER_DISPATCH_FASTK2V2.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
if p.exists():
 st=(p/'stat').read_text().split();assert st[21]==old['start_ticks']
 if st[2]!='Z':
  argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();assert 'tools.coopmwm.ls_broker' in argv and str(c) in argv
  # Only owned SSH metadata clients may be children; inference is remote/durable.
  for child in (p/'task'/str(old['pid'])/'children').read_text().split():
   q=Path('/proc',child);a=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();assert a.startswith('ssh '),a
  os.kill(old['pid'],signal.SIGTERM)
  for _ in range(100):
   if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
   time.sleep(.1)
  else:raise ValueError('old metadata broker still alive')
with (Path({ROOT!r})/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+chr(10))
env=dict(os.environ,PYTHONPATH=str(s/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'BROKER_NICHENULL1.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_broker',str(c),'_NICHENULL1'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],unix=time.time(),source_root=str(s),root=str(c),old=old,amendment_sha256={h!r})
with dest.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60)
    value=json.loads(out);target=ops.EVIDENCE/'LS_NICHENULL_BROKER_DISPATCH.json'
    if not target.exists():write(target,value)
    print(out)


def drain(host,root):
    slot=SLOTS[root.rsplit('_',1)[1]]
    remembered=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
c=Path({COLLECTION!r});p=c/'niche_roll/drains'/({digest(slot)!r}+'.json')
if p.exists():
 d=json.loads(p.read_text());done=c/'gpu_drained_v2'/(d['drain_key']+'.json')
 print(json.dumps(dict(adopted=True,drained=done.exists(),**d)))
else:print('{{}}')''',timeout=40))
    if remembered:
        print(json.dumps(remembered));return
    owner=json.loads(remote(host,f'''from pathlib import Path
import json
r=Path({root!r});d=json.loads((r/'SPOOL_OWNER.json').read_text());p=Path('/proc',str(d['pid']))
assert (p/'stat').read_text().split()[21]==d['start_ticks']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
print(json.dumps(dict(d,root=str(r),host={host!r})))'''))
    print(remote('cpuQ',f'''from pathlib import Path
import json,sys,time
c=Path({COLLECTION!r});sys.path.insert(0,{niche_ops.SOURCE!r}+'/pytools')
from tools.coopmwm.protocol import digest
from tools.coopmwm.ls_fast_worker import owner_key
from tools.coopmwm.warm_spool import immutable
owner={owner!r};slot={slot!r}
d=c/'niche_roll/drains';d.mkdir(parents=True,exist_ok=True)
p=d/(digest(slot)+'.json')
if p.exists():owner=json.loads(p.read_text())['owner']
key=digest(dict(slot=slot,owner=owner_key(owner)))
request=c/'gpu_drains_v2'/(key+'.json');request.parent.mkdir(exist_ok=True)
if not request.exists():immutable(request,dict(slot=slot,owner=owner,unix=time.time()))
if not p.exists():immutable(p,dict(slot=slot,owner=owner,drain_key=key))
done=c/'gpu_drained_v2'/request.name
print(json.dumps(dict(owner=owner,drained=done.exists(),drain_key=key)))'''))


def gpu(host,root):
    import inspect
    from .source_install import install_source
    installer=inspect.getsource(install_source)
    for path in ops.EVIDENCE.glob('LS_NICHENULL_DISPATCH_*.json'):
        d=read(path)
        if d['root']==root:
            print(json.dumps(dict(adopted=d)));return
        state=json.loads(remote(d['host'],f'''from pathlib import Path
import json
r=Path({d['root']!r});p=Path('/proc',{str(d['pid'])!r})
st=(p/'stat').read_text().split() if p.exists() else None
assert st is None or st[21]=={d['start_ticks']!r}
g=[json.loads(p.read_text()) for p in r.glob('STARTUP_*_LSCOLLECT_'+{d.get('generation','NICHENULL1')!r}+'.json')]
print(json.dumps(dict(qualifying=st is not None and st[2] in ('R','S','D') and not(len(g)==8 and all(x['pass_'] and x['max_cost_error']==0 for x in g)))))'''))
        if state['qualifying']:
            print(json.dumps(dict(wait=True,reason='another slot qualifying',owner=d)));return
    slot=SLOTS[root.rsplit('_',1)[1]]
    old=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
c=Path({COLLECTION!r});p=c/'niche_roll/drains'/({digest(slot)!r}+'.json')
if not p.exists():print('{{"wait":true}}')
else:
 d=json.loads(p.read_text());done=c/'gpu_drained_v2'/(d['drain_key']+'.json')
 print(json.dumps(d['owner']) if done.exists() else '{{"wait":true}}')'''))
    if old.get('wait'):print(json.dumps(old));return
    recovery=read(ops.EVIDENCE/'NICHE_DEPENDENCY_RECOVERY3.json')
    reg=read(ops.EVIDENCE/REG);expected=dict(reg['cpu']['source_hashes'],**recovery['source_hashes'])
    names=('mppi_search','mppi_runtime','plan_service','ls_gpu','niche_collect','feasible',
           'sumo_verify','nullfix_geometry','niche_worker')
    frozen=frozen_rollout_files(names)
    files=frozen['files']
    import hashlib
    for n,text in files.items():assert hashlib.sha256(text.encode()).hexdigest()==expected[n],n
    files.update(dependency_files())
    net=base64.b64decode(frozen['network'])
    assert hashlib.sha256(net).hexdigest()==expected['FROZEN_NETWORK.net.xml.gz']
    packed=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    value=json.loads(remote(host,f'''from pathlib import Path
import json,time,os,signal,subprocess,hashlib,base64,zlib,shutil,fcntl
r=Path({root!r});dest=r/'LS_GPU_DISPATCH_NICHENULL3.json'
if dest.exists():print(dest.read_text());raise SystemExit
lock=(r/'NICHENULL_DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old={old!r};p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert all((r/'forecasts'/p.name).exists() for p in (r/'inbox').glob('*.json'))
if p.exists():
 st=(p/'stat').read_text().split();assert st[21]==old['start_ticks']
 if st[2]!='Z':
  a=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();assert str(r) in a and any(x in a for x in ('ls_fast_worker','warm_spool','ls_gpu')),a
  os.kill(old['pid'],signal.SIGTERM)
 for _ in range(300):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.1)
 else:raise ValueError('retired owned GPU process still alive')
probe=(r/'SPOOL_WORKER.claim').open('a');fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(probe,fcntl.LOCK_UN);probe.close()
binding=json.loads((r/'GPU_BINDING.json').read_text());backup=r/'PRE_NICHENULL1'
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
exec({installer!r})
source_records=[]
for n,text in files.items():
 target=r/n;prior=backup/n
 target.parent.mkdir(parents=True,exist_ok=True)
 source_records.append(install_source(target,prior,text.encode(),binding['hashes'].get(n)))
 binding['hashes'][n]=hashlib.sha256(target.read_bytes()).hexdigest()
net=r/'FROZEN_NETWORK.net.xml.gz';source_records.append(install_source(net,backup/net.name,base64.b64decode({base64.b64encode(net).decode()!r}),binding['hashes'].get(net.name)));binding['hashes'][net.name]=hashlib.sha256(net.read_bytes()).hexdigest()
with (r/'NICHENULL_SOURCE_INSTALL.json').open('x') as f:json.dump(source_records,f)
deps={{n:hashlib.sha256(text.encode()).hexdigest() for n,text in files.items() if n.startswith('pytools/traffic_shepherd/') or any(n.endswith('/'+q+'.py') for q in ('ls_fast_worker','resource_config','query_errors'))}}
with (r/'NICHENULL_DEPENDENCY_HASHES.json').open('x') as f:json.dump(deps,f)
binding.pop('adopted_startup_gates',None)
binding.update(deadline_unix=1793894340.,gpu_deadline=1793894340.,niche_comparison=True,effective_amendment_sha256={sha(ops.EVIDENCE/'NICHE_DEPENDENCY_RECOVERY3.json')!r})
if {host!r}=='gpuR':binding['batch_widths']={{'1':1,'2':1,'4':1,'8':1}}
for n in ('SPOOL_OWNER.json','SERVICE_OWNER.json'):
 if (r/n).exists():shutil.copy2(r/n,backup/n)
with (r/'GPU_BINDING_NICHENULL3.json').open('x') as f:json.dump(binding,f)
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
if {host!r}=='gpuR':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
with (r/'LS_GPU_NICHENULL1.log').open('x') as log:
  proc=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-c',{worker_command(root,'NICHENULL3')!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(r),host={host!r},unix=time.time(),generation='NICHENULL3',log_name='LS_GPU_NICHENULL1.log',amendment_sha256={sha(ops.EVIDENCE/'NICHE_DEPENDENCY_RECOVERY3.json')!r})
with dest.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    path=ops.EVIDENCE/f'LS_NICHENULL_DISPATCH_{host}_{root.rsplit("_",1)[1]}.json'
    if not path.exists():write(path,value)
    print(json.dumps(value))


def recover(host,root):
    """Adopt or replace only the exited, pre-gate first qualifier; no query replay."""
    import time
    from .protocol import REGISTRY,checked
    receipt=ops.EVIDENCE/'NICHE_STARTUP_PATH_RECOVERY.json'
    if not receipt.exists():
        import hashlib
        deps=dependency_files()
        hashes={n:hashlib.sha256(t.encode()).hexdigest() for n,t in deps.items()}
        # CPU-only dependency check in the actual frozen GPU environment.
        probe=json.loads(remote(host,f'''from pathlib import Path
import subprocess,os,json,hashlib
r=Path({root!r})
files={deps!r};hashes={hashes!r}
assert hashlib.sha256((r/'FROZEN_GEOMETRY_SOURCE.py').read_bytes()).hexdigest()==hashes['pytools/traffic_shepherd/c4/geometry.py']
for n,text in files.items():
 p=r/n
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n],n
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
manifest=r/'NICHENULL_DEPENDENCY_HASHES.json'
if manifest.exists():assert json.loads(manifest.read_text())==hashes
else:
 with manifest.open('x') as f:json.dump(hashes,f)
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',CUDA_VISIBLE_DEVICES='')
code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(r/'FROZEN_GEOMETRY_SOURCE.py'))+','+repr(str(r/'FROZEN_NETWORK.net.xml.gz'))+');import torch;assert not torch.cuda.is_initialized();print("PASS")'
p=subprocess.run(['/root/miniconda3/bin/python','-B','-c',code],env=env,text=True,capture_output=True,timeout=45)
print(json.dumps(dict(pass_=p.returncode==0,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,new_sumo_sim_s=0)))''',timeout=60))
        assert probe['pass_'],probe
        line=('\nS-COOPMWM Nov3 startup-path engineering amendment: first niche/nullfix '
            'qualifier exited BEFORE gates/owner/search on missing traffic_shepherd import. '
            'GPU WM-only sources lack traffic_shepherd entirely. Add ONLY its two '
            'package initializers and the already SHA-bound pure geometry source; '
            'verify all three against an immutable dependency manifest before worker import. '
            'Actual GPU-environment CPU dependency probe PASS; '
            'no CUDA initialization/SUMO. Numerical sources/binding unchanged; ALL8fresh '
            'gates still required. Preserve failed dispatch/log; replace only exited '
            'pre-gate owner once, no inference replay; other3 workers continue. '
            'Fresh6M guard unchanged.\n')
        write(receipt,dict(line=line,probe=probe,dependency_hashes=hashes,parent_sha256=sha(ops.EVIDENCE/REG),
            worker_command=worker_command(root),unix=time.time(),new_sumo_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(receipt)+'\n')
    value=json.loads(remote(host,f'''from pathlib import Path
import json,time,os,subprocess,hashlib,fcntl
r=Path({root!r});dest=r/'LS_GPU_DISPATCH_NICHENULL1_PATH2.json'
if dest.exists():print(dest.read_text());raise SystemExit
assert r.parent.is_dir()
lock=(r/'NICHENULL_DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((r/'LS_GPU_DISPATCH_NICHENULL1.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
st=(p/'stat').read_text().split() if p.exists() else None
assert st is None or (st[21]==old['start_ticks'] and st[2]=='Z'),'adopt live qualifier'
assert not list(r.glob('STARTUP_*_LSCOLLECT_NICHENULL1.json'))
assert not (r/'NICHENULL_OWNER.json').exists()
assert not (r/'LS_GPU_FAILED_NICHENULL1.json').exists()
assert 'ModuleNotFoundError: No module named' in (r/'LS_GPU_NICHENULL1.log').read_text()
binding=json.loads((r/'GPU_BINDING_NICHENULL1.json').read_text())
for name,h in binding['hashes'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
probe=(r/'SPOOL_WORKER.claim').open('a');fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(probe,fcntl.LOCK_UN);probe.close()
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'LS_GPU_NICHENULL1_PATH2.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-c',{worker_command(root)!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(r),host={host!r},unix=time.time(),generation='NICHENULL1',log_name='LS_GPU_NICHENULL1_PATH2.log',amendment_sha256={sha(receipt)!r},previous=old)
with dest.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    target=ops.EVIDENCE/f'LS_NICHENULL_DISPATCH_{host}_{root.rsplit("_",1)[1]}_PATH2.json'
    if not target.exists():write(target,value)
    print(json.dumps(value))


def recover2(host,root,version=2):
    """Complete source package after the pre-gate ls_fast_worker omission."""
    import hashlib
    from .protocol import REGISTRY
    generation='NICHENULL'+str(version)
    previous='NICHENULL'+str(version-1)
    cpu=read(ops.EVIDENCE/f'LS_LABEL_CPU_coopmwm20261103_nichenullcpu{version+4}.json')
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    files=dependency_files()
    files['pytools/tools/coopmwm/niche_worker.py']=checked(ops.LOCAL/'tools/coopmwm/niche_worker.py').read_text(encoding='utf-8')
    hashes={n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()}
    for n in ('pytools/tools/coopmwm/niche_worker.py','pytools/tools/coopmwm/ls_fast_worker.py','pytools/tools/coopmwm/resource_config.py','pytools/tools/coopmwm/query_errors.py'):
        assert hashes[n]==cpu['source_hashes'][n],n
    receipt=ops.EVIDENCE/f'NICHE_DEPENDENCY_RECOVERY{version}.json'
    if not receipt.exists():
        line=('\nS-COOPMWM Nov3 pre-gate dependency recovery2: original PATH2 qualifier '
            'exited before all gates/search on missing tools.coopmwm.ls_fast_worker. '
            'Copy the SAME CPU-qualified it2_K2 wrapper plus pinned pure geometry '
            'package; all dependency bytes verified. Metadata-only worker generation '
            'parameter preserves earlier owner/failure/gate files. New additive '
            'GPU_BINDING_NICHENULL2 records actual source hashes; original bindings '
            'and failed logs retained. Full frozen Linux CPU suite PASS/noSUMO. '
            'Run actual GPU-environment complete dependency smoke BEFORE launch; '
            'ALL8fresh source-bound gates before admission, ONE qualifier globally. '
            'No query/episode replay, horizon/K/N/iterations or6Mguard change.\n')
        if version==3:
            line=('\nS-COOPMWM metadata-dependency recovery3: NICHENULL2 passed four '
                'exact checks then exited on missing resource_config (legacy overlay omission). '
                'Package resource_config+query_errors too; smoke their lazy service imports '
                'and clocks before dispatch. Numerical sources unchanged. Preserve/reuse '
                'the FOUR completed same-source exact checks by SHA+complete NUMERICAL '
                'manifest; new process runs FOUR fresh cold/warm full/fast gates. This '
                'supersedes redundant all8fresh for this metadata-only recovery; no '
                'completed inference or qualification repeated, all8checks required '
                'before admission. New untouched slots still run all8fresh. CPU7 full '
                'release qualified; original failures/receipts/bindings kept, newgen3 '
                'binding only. Same horizons/K/N/iterations/6Mguard; no SUMO/PPO.\n')
        write(receipt,dict(line=line,cpu=cpu,source_hashes=hashes,parent_sha256=sha(ops.EVIDENCE/REG),new_sumo_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(receipt)+'\n')
    value=json.loads(remote(host,f'''from pathlib import Path
import os,json,hashlib,time,subprocess,fcntl,shutil
r=Path({root!r});dest=r/('LS_GPU_DISPATCH_'+{generation!r}+'.json')
if dest.exists():print(dest.read_text());raise SystemExit
assert r.parent.is_dir()
lock=(r/'NICHENULL_DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((r/{('LS_GPU_DISPATCH_NICHENULL1_PATH2.json' if version==2 else 'LS_GPU_DISPATCH_NICHENULL2.json')!r}).read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
st=(p/'stat').read_text().split() if p.exists() else None
assert st is None or (st[21]==old['start_ticks'] and st[2]=='Z'),'adopt live qualifier'
assert len(list(r.glob('STARTUP_*_LSCOLLECT_'+{previous!r}+'.json')))=={0 if version==2 else 4}
assert not list(r.glob('STARTUP_*_LSCOLLECT_'+{generation!r}+'.json'))
assert 'No module named' in (r/{('LS_GPU_NICHENULL1_PATH2.log' if version==2 else 'LS_GPU_NICHENULL2.log')!r}).read_text()
binding=json.loads((r/('GPU_BINDING_'+{previous!r}+'.json')).read_text())
for n,h in binding['hashes'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
files={files!r};hashes={hashes!r};backup=r/('PRE_'+{generation!r})
for n,text in files.items():
 target=r/n;prior=backup/n
 if target.exists() and not prior.exists():
  prior.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,prior)
  assert hashlib.sha256(prior.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==hashes[n]
 binding['hashes'][n]=hashes[n]
old_manifest=r/'NICHENULL_DEPENDENCY_HASHES.json';prior=backup/old_manifest.name
if not prior.exists():shutil.copy2(old_manifest,prior)
with old_manifest.open('w') as f:json.dump(hashes,f)
binding['effective_amendment_sha256']={sha(receipt)!r}
if {version}==3:
 import sys;sys.path.insert(0,str(r/'pytools'))
 from tools.coopmwm.warm_service import ADOPTABLE,NUMERICAL
 binding['adopted_joint_checks']=dict(numerical_hashes={{n:binding['hashes'][n] for n in NUMERICAL}},receipts={{}})
 for name in ADOPTABLE:
  path=r/(Path(name).stem+'_LSCOLLECT_'+{previous!r}+'.json');q=json.loads(path.read_text());assert q['pass_'] and q['max_cost_error']==0
  binding['adopted_joint_checks']['receipts'][name]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with (r/('GPU_BINDING_'+{generation!r}+'.json')).open('x') as f:json.dump(binding,f)
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
smoke='from pathlib import Path;import json;from tools.coopmwm.resource_config import clocks;from tools.coopmwm.query_errors import execute;clocks(Path('+repr(str(r))+'),json.loads(Path('+repr(str(r/('GPU_BINDING_'+{generation!r}+'.json')))+').read_text()));from tools.coopmwm.ls_fast_worker import configured_search;from tools.coopmwm.ls_gpu import capture_search;from tools.coopmwm import warm_service,plan_service,mppi_search,mppi_runtime;from tools.coopmwm.nullfix_geometry import install;install('+repr(str(r/'FROZEN_GEOMETRY_SOURCE.py'))+','+repr(str(r/'FROZEN_NETWORK.net.xml.gz'))+');import torch;assert not torch.cuda.is_initialized();print("PASS")'
check=subprocess.run(['/root/miniconda3/bin/python','-B','-c',smoke],env=env,text=True,capture_output=True,timeout=45)
probe=dict(pass_=check.returncode==0,stdout=check.stdout,stderr=check.stderr,new_sumo_sim_s=0)
with (r/({generation!r}+'_DEPENDENCY_SMOKE.json')).open('x') as f:json.dump(probe,f)
assert probe['pass_'],probe
probe_lock=(r/'SPOOL_WORKER.claim').open('a');fcntl.flock(probe_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(probe_lock,fcntl.LOCK_UN);probe_lock.close()
with (r/('LS_GPU_'+{generation!r}+'.log')).open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-c',{worker_command(root,generation)!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(r),host={host!r},unix=time.time(),generation={generation!r},log_name='LS_GPU_'+{generation!r}+'.log',amendment_sha256={sha(receipt)!r},previous=old,dependency_smoke=probe)
with dest.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    target=ops.EVIDENCE/f'LS_NICHENULL_DISPATCH_{host}_{root.rsplit("_",1)[1]}_PATH{version+1}.json'
    if not target.exists():write(target,value)
    print(json.dumps(value))


def recover3(host,root):recover2(host,root,version=3)


def roll_target(gpus):
    workers=[w for g in gpus.values() for w in g['workers']]
    if any(w['owner'].get('generation','').startswith('NICHENULL') and
            w['state'] in ('S','R','D') and not w['failure'] and not w['ready'] for w in workers):
        return None
    for host,suffix in (('gpuA','incident1'),('gpuA','deadlineA2'),
                        ('gpuR','concurrent3'),('gpuR','deadlineR2')):
        root='/root/coopmwm20261102_'+suffix
        w=next(w for w in gpus[host]['workers'] if w['owner']['root']==root)
        if w['owner'].get('generation')=='NICHENULL3':
            if not w['ready']:raise ValueError('joint-release qualifier failed; diagnose retained receipt before replacement')
        else:return host,root
    return None


def continue_roll():
    """Finish only registered drained-boundary handovers; never dispatch SUMO/PPO."""
    import contextlib,io,msvcrt
    from .ls_validity_ops import gpu_status,admit
    lock=(ops.EVIDENCE/'LS_NICHENULL_ROLL.claim').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while time.time()<1793894340:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                gpus={h:gpu_status(h) for h in ('gpuA','gpuR')}
                target=roll_target(gpus)
                if target:
                    host,root=target
                    drain(host,root);gpu(host,root)
                ready=[w for g in gpus.values() for w in g['workers']
                    if w['ready'] and w['owner'].get('generation')=='NICHENULL3']
                if ready:admit()
            print(json.dumps(dict(unix=time.time(),ready=[w['owner']['root'] for w in ready],
                target=target,training_paused=True,no_sumo_dispatch=True)),flush=True)
            if len(ready)==4:
                path=ops.EVIDENCE/'NICHE_NULLFIX_FOUR_WORKERS_READY.json'
                if not path.exists():write(path,dict(unix=time.time(),workers=gpus,
                    amendment_sha256=sha(ops.EVIDENCE/'NICHE_DEPENDENCY_RECOVERY3.json'),
                    new_sumo_sim_s=0,collector_activation_pending=True))
                return
        except Exception as error:
            write(ops.EVIDENCE/f'NICHE_ROLL_ERROR_{time.time_ns()}.json',dict(error=repr(error),unix=time.time(),no_inference_replay=True))
            raise
        time.sleep(60)


def roll_dispatch():
    import subprocess,sys,psutil
    from .protocol import REGISTRY
    target=ops.EVIDENCE/'LS_NICHENULL_ROLL_DISPATCH_V5.json'
    if target.exists():
        d=read(target)
        if psutil.pid_exists(d['pid']):
            p=psutil.Process(d['pid']);assert abs(p.create_time()-d['unix'])<5
            assert p.cmdline()[-3:]==['-m','tools.coopmwm.niche_roll','continue_roll']
        print(json.dumps(dict(adopted=d)));return
    reg=ops.EVIDENCE/'NICHE_ROLL_CONTROLLER_REGISTRATION_V5.json'
    if not reg.exists():
        line=('\nS-COOPMWM joint-release controller V5 engineering amendment: V4 retired '
            'the drained Asecond then refused stale binding hashes: its per-file '
            'source symlinks point at the already-qualified Afirst package. Install '
            'owned destination entries ATOMICALLY (detach aliases, never write through '
            'to a serving peer); preserve original bytes and link metadata with SHA. '
            'Accept prior-binding bytes OR exact incoming CPU7-qualified bytes only. '
            'Numerical files/network still from immutable CPU7 with full hashes. '
            'Same qualified source/all8fresh gates, no numerical change. '
            'Retain all earlier controller faults and original workers/requests. Adopt live '
            'source-qualified NICHENULL3 successor; retain stopped V1 fault receipt; finish registered Asecond/Rfirst/Rsecond drained '
            'handovers ONE qualifier globally. Preserve requests/claims/forecasts; '
            'no inference replay or SUMO/PPO dispatch. Source-bound dependency package '
            'and all8 checks per slot before admission (first metadata recovery adopts4 same-source; untouched slots all8fresh). Stop controller on code '
            'fault; bounded existing transport retries. Collector activation remains '
            'separate after all4ready; other slots keep serving.\n')
        write(reg,dict(line=line,source_sha256=sha(ops.LOCAL/'tools/coopmwm/niche_roll.py'),new_sumo_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(reg)+'\n')
    old=read(ops.EVIDENCE/'LS_NICHENULL_ROLL_DISPATCH_V4.json')
    assert not psutil.pid_exists(old['pid']),'adopt original controller until exited'
    with (ops.EVIDENCE/'LS_NICHENULL_ROLL_V5.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.niche_roll','continue_roll'],
            cwd=ops.LOCAL,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    value=dict(pid=p.pid,unix=time.time(),source_sha256=sha(ops.LOCAL/'tools/coopmwm/niche_roll.py'),
        amendment_sha256=sha(reg),mode='registered sequential GPU rollout only; no SUMO/PPO')
    write(target,value);print(json.dumps(value))


def activate():
    """All-four-gate collector handover, at a verified no-grade boundary only."""
    import contextlib,io
    from .ls_validity_ops import gpu_status
    from .protocol import REGISTRY
    with contextlib.redirect_stdout(io.StringIO()):
        gpus={h:gpu_status(h) for h in ('gpuA','gpuR')}
    workers=[w for g in gpus.values() for w in g['workers']]
    if len(workers)!=4 or not all(w['ready'] and w['owner'].get('generation')=='NICHENULL3' for w in workers):
        print(json.dumps(dict(wait=True,reason='all four source-bound GPU gates required')));return False
    cpu=read(ops.EVIDENCE/'LS_LABEL_CPU_coopmwm20261103_nichenullcpu7.json')
    transfer=read(ops.EVIDENCE/'B_REPLAY_PHASE_TRANSFER_0706.json')
    assert transfer['funding_complete'] and transfer['budget']['total']==14500000
    reg=ops.EVIDENCE/'NICHE_COLLECTOR_ACTIVATION_REGISTRATION_0706.json'
    if not reg.exists():
        line=('\nS-COOPMWM joint niche/nullfix collector activation: all4 source-bound '
            'eight-gate GPU workers required. Metadata supervisor handover only at '
            'no-active-grade boundary: freeze receipt-verified parent; require all '
            'its verification claims have final labels and only metadata tracker '
            'children; otherwise CONT unchanged and wait. Never interrupt/replay '
            'physics or claims. Adopt72episodes/allsnapshots/labels/requests/charges; '
            'new immutable LS_NULLFIX_CONFIG from271CPU-qualified source6 enables '
            'observable actionable gate before verification and B-only prospective '
            'niches1/4 admission alternation. Same it2_K2/K8 final/horizons/N, '
            'same existing precharges; MAIN06:27/07:06 operationalcap14500000 and '
            'authorized checks->train transfer loaded explicitly in new metadata '
            'config, original immutable bindings retained. R/PPO unchanged.\n')
        write(reg,dict(line=line,cpu=cpu,parent_sha256=sha(ops.EVIDENCE/'NICHE_DEPENDENCY_RECOVERY3.json'),new_sumo_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(reg)+'\n')
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,os,signal,subprocess,hashlib
c=Path({COLLECTION!r});s=Path('/home/ubuntu/coopmwm20261103_nichenullcpu7');dest=c/'SUPERVISOR_DISPATCH_NICHENULL2.json'
if dest.exists():print(dest.read_text());raise SystemExit
assert s.is_dir() and c.is_dir()
cpu={cpu!r}
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
old=json.loads((c/'SUPERVISOR_DISPATCH_SUMOFOCUS1.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert p.exists() and (p/'stat').read_text().split()[21]==old['start_ticks']
argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
assert 'tools.coopmwm.ls_collection' in argv and str(c) in argv
os.kill(old['pid'],signal.SIGSTOP)
try:
 for _ in range(100):
  if (p/'stat').read_text().split()[2]=='T':break
  time.sleep(.01)
 else:raise ValueError('metadata parent did not freeze')
 children=(p/'task'/str(old['pid'])/'children').read_text().split()
 active_children=[k for k in children if 'multiprocessing.resource_tracker' not in (Path('/proc',k)/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()]
 labelled={{json.loads(q.read_text())['verification']['verification_id'] for q in (c/'labels').glob('*.json')}}
 attempts=c/'label_attempts/SUMOENV1/verification/claims'
 open_claims=[q.stem for q in attempts.glob('*.json') if json.loads(q.read_text()).get('pid')==old['pid'] and q.stem not in labelled]
 if active_children or open_claims:
  os.kill(old['pid'],signal.SIGCONT)
  print(json.dumps(dict(wait=True,active_children=active_children,open_claims=open_claims,no_physics_signals=True)));raise SystemExit
 os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT)
except BaseException:
 if p.exists() and (p/'stat').read_text().split()[2]=='T':os.kill(old['pid'],signal.SIGCONT)
 raise
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise ValueError('old metadata supervisor did not exit')
cfg=dict(source_root=str(s),receipt_suffix='_NICHENULL2',nullfix_active=True,
 niche_alternation=dict(start_admission_count=len(list((c/'admissions').glob('*.json')))),
  operational_spend_cap=14500000,budget={transfer['budget']!r},
  budget_amendment_sha256={sha(ops.EVIDENCE/'B_REPLAY_PHASE_TRANSFER_0706.json')!r},amendment_sha256={sha(reg)!r})
with (c/'LS_NULLFIX_CONFIG.json').open('x') as f:json.dump(cfg,f)
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
code='from pathlib import Path;from tools.coopmwm.nullfix_geometry import install;install('+repr(str(s/'FROZEN_GEOMETRY_SOURCE.py'))+','+repr(str(s/'FROZEN_NETWORK.net.xml.gz'))+');from tools.coopmwm.ls_collection import supervise;supervise(Path('+repr(str(c))+'))'
with (c/'SUPERVISOR_NICHENULL2.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],unix=time.time(),source_root=str(s),root=str(c),old=old,amendment_sha256={sha(reg)!r},config=cfg,no_physics_signals=True)
with dest.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    if value.get('wait'):print(json.dumps(value));return False
    target=ops.EVIDENCE/'NICHE_NULLFIX_COLLECTOR_ACTIVE.json'
    if not target.exists():write(target,value)
    from .ls_validity_ops import fast_watch_dispatch
    fast_watch_dispatch(13)
    print(json.dumps(value));return True


def activation_watch():
    """Existing rollout completion event only; no GPU/episode dispatch."""
    import msvcrt
    lock=(ops.EVIDENCE/'NICHE_ACTIVATION_WATCH.claim').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while time.time()<1793894340:
        if (ops.EVIDENCE/'NICHE_NULLFIX_FOUR_WORKERS_READY.json').exists() and activate():return
        time.sleep(60)


def activation_dispatch():
    import subprocess,sys,psutil
    target=ops.EVIDENCE/'NICHE_ACTIVATION_WATCH_DISPATCH_V3.json'
    if target.exists():print(json.dumps(dict(adopted=read(target))));return
    reg=ops.EVIDENCE/'NICHE_ACTIVATION_BUDGET_WATCH_0706.json'
    if not reg.exists():
        from .protocol import REGISTRY
        line=('\nS-COOPMWM MAIN06:27/07:06 activation-waiter metadata amendment: '
            'new collector config explicitly loads approved14.5M cap and SHA-bound '
            'checks->train transfer after all4workersREADY/no-active-grade boundary. '
            'Retain old bindings and precharges; no new search/physics/PPO, same '
            'qualified numerical source. Replace only identity-verified local waiter.\n')
        write(reg,dict(line=line,budget_sha256=sha(ops.EVIDENCE/'B_REPLAY_PHASE_TRANSFER_0706.json'),new_sumo_sim_s=0))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(reg)+'\n')
    old=read(ops.EVIDENCE/'NICHE_ACTIVATION_WATCH_DISPATCH_V2.json')
    if psutil.pid_exists(old['pid']):
        p=psutil.Process(old['pid']);assert abs(p.create_time()-old['unix'])<5
        assert p.cmdline()[-3:]==['-m','tools.coopmwm.niche_roll','activation_watch']
        p.terminate();p.wait(timeout=15)
    with (ops.EVIDENCE/'NICHE_ACTIVATION_WATCH_V3.log').open('x',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.niche_roll','activation_watch'],
            cwd=ops.LOCAL,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    value=dict(pid=proc.pid,unix=time.time(),source_sha256=sha(ops.LOCAL/'tools/coopmwm/niche_roll.py'),
        mode='wait four qualified slots; metadata-only no-grade-boundary collector activation')
    write(target,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('broker','drain','gpu','recover','recover2','recover3','continue_roll','roll_dispatch','activate','activation_watch','activation_dispatch'))
    p.add_argument('--host',choices=('gpuA','gpuR'));p.add_argument('--root')
    a=p.parse_args()
    if a.action in ('broker','continue_roll','roll_dispatch','activate','activation_watch','activation_dispatch'):globals()[a.action]()
    else:globals()[a.action](a.host,a.root)
