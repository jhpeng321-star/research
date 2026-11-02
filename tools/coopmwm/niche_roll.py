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


def worker_command(root):
    # Pure geometry is absent from the GPU WM-only package. Validate the additive
    # source-only dependency package before importing the unchanged worker.
    return ('from pathlib import Path;import json,hashlib;'
        'r=Path('+repr(root)+');'
        'h=json.loads((r/"NICHENULL_DEPENDENCY_HASHES.json").read_text());'
        'assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==s for n,s in h.items());'
        'from tools.coopmwm.niche_worker import run;run('+repr(root)+')')


def dependency_files():
    return {'pytools/'+n:checked(ops.LOCAL/n).read_text(encoding='utf-8') for n in (
        'traffic_shepherd/__init__.py','traffic_shepherd/c4/__init__.py',
        'traffic_shepherd/c4/geometry.py')}


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
    for path in ops.EVIDENCE.glob('LS_NICHENULL_DISPATCH_*.json'):
        d=read(path)
        if d['root']==root:
            print(json.dumps(dict(adopted=d)));return
        state=json.loads(remote(d['host'],f'''from pathlib import Path
import json
r=Path({d['root']!r});p=Path('/proc',{str(d['pid'])!r})
st=(p/'stat').read_text().split() if p.exists() else None
assert st is None or st[21]=={d['start_ticks']!r}
g=[json.loads(p.read_text()) for p in r.glob('STARTUP_*_LSCOLLECT_NICHENULL1.json')]
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
    reg=read(ops.EVIDENCE/REG);expected=reg['cpu']['source_hashes']
    names=('mppi_search','mppi_runtime','plan_service','ls_gpu','niche_collect','feasible',
           'sumo_verify','nullfix_geometry','niche_worker')
    files={f'pytools/tools/coopmwm/{n}.py':(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    files['FROZEN_GEOMETRY_SOURCE.py']=(ops.LOCAL/'traffic_shepherd/c4/geometry.py').read_text(encoding='utf-8')
    import hashlib
    for n,text in files.items():assert hashlib.sha256(text.encode()).hexdigest()==expected[n],n
    files.update(dependency_files())
    net=(ops.LOCAL/'assets/research/highway_merge_shepherd_long/osm.net.xml.gz').read_bytes()
    assert hashlib.sha256(net).hexdigest()==expected['FROZEN_NETWORK.net.xml.gz']
    packed=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    value=json.loads(remote(host,f'''from pathlib import Path
import json,time,os,signal,subprocess,hashlib,base64,zlib,shutil,fcntl
r=Path({root!r});dest=r/'LS_GPU_DISPATCH_NICHENULL1.json'
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
for n,text in files.items():
 target=r/n;prior=backup/n
 if target.exists() and not prior.exists():
  old_hash=hashlib.sha256(target.read_bytes()).hexdigest()
  if n in binding['hashes']:assert old_hash==binding['hashes'][n],n
  prior.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,prior)
  assert hashlib.sha256(prior.read_bytes()).hexdigest()==old_hash,n
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
 binding['hashes'][n]=hashlib.sha256(target.read_bytes()).hexdigest()
net=r/'FROZEN_NETWORK.net.xml.gz';net.write_bytes(base64.b64decode({base64.b64encode(net).decode()!r}));binding['hashes'][net.name]=hashlib.sha256(net.read_bytes()).hexdigest()
deps={{n:hashlib.sha256(text.encode()).hexdigest() for n,text in files.items() if n.startswith('pytools/traffic_shepherd/')}}
with (r/'NICHENULL_DEPENDENCY_HASHES.json').open('x') as f:json.dump(deps,f)
binding.pop('adopted_startup_gates',None)
binding.update(deadline_unix=1793894340.,gpu_deadline=1793894340.,niche_comparison=True,effective_amendment_sha256={sha(ops.EVIDENCE/REG)!r})
if {host!r}=='gpuR':binding['batch_widths']={{'1':1,'2':1,'4':1,'8':1}}
for n in ('SPOOL_OWNER.json','SERVICE_OWNER.json'):
 if (r/n).exists():shutil.copy2(r/n,backup/n)
with (r/'GPU_BINDING_NICHENULL1.json').open('x') as f:json.dump(binding,f)
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
if {host!r}=='gpuR':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
with (r/'LS_GPU_NICHENULL1.log').open('x') as log:
  proc=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-c',{worker_command(root)!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(r),host={host!r},unix=time.time(),generation='NICHENULL1',log_name='LS_GPU_NICHENULL1.log',amendment_sha256={sha(ops.EVIDENCE/REG)!r})
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('broker','drain','gpu','recover'))
    p.add_argument('--host',choices=('gpuA','gpuR'));p.add_argument('--root')
    a=p.parse_args()
    if a.action=='broker':broker()
    else:globals()[a.action](a.host,a.root)
