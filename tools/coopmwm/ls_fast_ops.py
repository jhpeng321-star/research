"""Registered rolling, drain-before-retire it2_K2 collection deployment."""
import argparse
import json
from . import ops
from .ls_collect_ops import COLLECTION, ROOT
from .ls_validity_ops import remote, payload
from .protocol import checked, REGISTRY, sha, digest
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_lsfastcpu1'
OVERLAY='/root/coopmwm20261103_it2k2ops1'
REG='LS_FAST_IT2K2_REGISTRATION.json'
SLOTS={'incident1':'gpuA@incident1','deadlineA2':'gpuA@resource1651',
    'concurrent3':'gpuR@0','deadlineR2':'gpuR@resource1651'}


def qualify():
    from .ls_focus_ops import SOURCE as BASE
    packed=payload(('ls_fast_worker','ls_broker','test_ls_fast_worker_cpu'))
    print(remote('cpuQ',f'''from pathlib import Path
import json,os,shutil,hashlib,subprocess,time,base64,zlib
r=Path({SOURCE!r});base=Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 r.mkdir();shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):shutil.copy2(base/name,r/name)
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['test_ls_fast_worker_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('x') as f:json.dump(sorted(set(tests)),f)
for n,text in files.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
 with (r/'LS_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_cpu_test',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))'''))


def result():
    from . import ls_label_ops
    ls_label_ops.SOURCE=SOURCE
    ls_label_ops.result()


def register():
    cpu=read(ops.EVIDENCE/('LS_LABEL_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    p=ops.EVIDENCE/REG
    if not p.exists():
        line=('\nS-COOPMWM MAIN00:48 IT2_K2 ENGINEERING DEPLOYMENT: authorized iterations2/'
            'screenK2H300, full dimension-scaled population; finalK8/Hfinal unchanged. '
            'No learned seeds, warm start, horizon reduction or mirror activation. '
            'Current workers cache config with no reload hook: rolling boundary replacement '
            'required for NEW parameter-size qualification. Immutable per-owner broker drain '
            'finishes/delivers original inference before retiring only that worker; other '
            'slots and grading continue. Each successor must pass NEW exact K2H300 warm '
            'full/fast and batch/single plus ALL8fresh inherited gates before admission. '
            'Metadata broker handover adopts assigned/in-flight queries; immutable submit '
            'redelivery is transport-only, never an inference retry. Every result records '
            'search_config. All failed receipts/charges retained. Zero new SUMO for '
            'qualification; conservative6000000 operational cap and phase caps unchanged.\n')
        write(p,dict(line=line,cpu=cpu,authority='MAIN2026-11-03 00:48',
            config=dict(name='it2_K2',iterations=2,screen_K=2,screen_H=300,final_K=8),
            new_sumo_sim_s=0,operational_spend_cap=6000000))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(p)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(p),cpu_tests=cpu['tests'])))


def broker():
    reg=read(ops.EVIDENCE/REG);h=sha(ops.EVIDENCE/REG)
    out=remote('cpuQ',f'''from pathlib import Path
import json,os,sys,time,signal,subprocess,hashlib
c=Path({COLLECTION!r});s=Path({SOURCE!r});r=Path({ROOT!r});dest=c/'BROKER_DISPATCH_FASTK2.json'
if dest.exists():print(dest.read_text());raise SystemExit
reg=json.loads({json.dumps(reg)!r})
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
old=json.loads((c/'BROKER_DISPATCH_VALIDITY1.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert (p/'stat').read_text().split()[21]==old['start_ticks']
argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
assert 'tools.coopmwm.ls_broker' in argv and str(c) in argv
os.kill(old['pid'],signal.SIGSTOP)
try:
 for child in (p/'task'/str(old['pid'])/'children').read_text().split():
  q=Path('/proc',child);a=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  assert 'ssh ' in a and 'sumo' not in a.lower(),a
 # Existing remote searches are independent; the replacement adopts assignments.
 os.kill(old['pid'],signal.SIGTERM)
finally:os.kill(old['pid'],signal.SIGCONT)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise ValueError('old metadata broker did not exit')
sys.path.insert(0,str(s/'pytools'));from tools.coopmwm.warm_spool import immutable
immutable(c/'BROKER_RETIRED_FASTK2.json',dict(owner=old,unix=time.time(),no_inference_or_physics_signals=True))
with (r/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+chr(10))
env=dict(os.environ,PYTHONPATH=str(s/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'BROKER_FASTK2.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_broker',str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(c),source_root=str(s),unix=time.time(),amendment_sha256={h!r})
immutable(dest,d);print(json.dumps(d))''',timeout=60)
    v=json.loads(out);p=ops.EVIDENCE/'LS_FAST_BROKER_DISPATCH.json'
    if not p.exists():write(p,v)
    print(out)


def drain(host, root):
    """Drain one current owner; call again only to observe the same request."""
    from .ls_validity_ops import gpu_status
    import contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):g=gpu_status(host)
    w=next(w for w in g['workers'] if w['owner']['root']==root)
    d=w['owner'];assert d.get('generation')!='FASTK2'
    slot=SLOTS[root.rsplit('_',1)[1]]
    out=remote('cpuQ',f'''from pathlib import Path
import json,sys,time
c=Path({COLLECTION!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.protocol import digest
broker=json.loads((c/'BROKER_DISPATCH_FASTK2.json').read_text());p=Path('/proc',str(broker['pid']))
assert (p/'stat').read_text().split()[21]==broker['start_ticks']
(c/'gpu_drains').mkdir(exist_ok=True)
target=c/'gpu_drains'/(digest({slot!r})+'.json')
if not target.exists():immutable(target,dict(slot={slot!r},owner={d!r},unix=time.time(),amendment_sha256={sha(ops.EVIDENCE/REG)!r}))
done=c/'gpu_drained'/target.name
print(json.dumps(dict(requested=json.loads(target.read_text()),drained=json.loads(done.read_text()) if done.exists() else None)))''')
    print(out)


def gpu(host, root):
    """Exactly one detached successor after the broker has delivered its last search."""
    slot=SLOTS[root.rsplit('_',1)[1]];h=sha(ops.EVIDENCE/REG)
    d=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
p=Path({COLLECTION!r})/'gpu_drained'/({digest(slot)!r}+'.json')
print(p.read_text() if p.exists() else json.dumps(dict(wait=True)))'''))
    if d.get('wait'):print(json.dumps(d));return
    old=d['owner']
    from .protocol import checked
    packed=__import__('base64').b64encode(checked(ops.LOCAL/'tools/coopmwm/ls_fast_worker.py').read_bytes()).decode()
    out=remote(host,f'''from pathlib import Path
import json,os,time,subprocess,hashlib,base64,shutil
r=Path({root!r});dest=r/'LS_GPU_DISPATCH_FASTK2.json';overlay=Path({OVERLAY!r})
if dest.exists():print(dest.read_text());raise SystemExit
assert overlay.parent.is_dir()
if not overlay.exists():shutil.copytree('/root/coopmwm20261102_lsvalidops1',overlay,ignore=shutil.ignore_patterns('__pycache__'))
text=base64.b64decode({packed!r});target=overlay/'ls_fast_worker.py'
if target.exists():assert target.read_bytes()==text
else:target.write_bytes(text)
old={old!r};assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
p=Path('/proc',str(old['pid']));assert (p/'stat').read_text().split()[21]==old['start_ticks']
old['argv']=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
assert 'ls_gpu' in old['argv'] and str(r) in old['argv']
assert all((r/'forecasts'/q.name).exists() for q in (r/'inbox').glob('*.json'))
code='import sys;sys.path.insert(0,'+repr(str(overlay))+');from ls_fast_worker import run;run('+repr(str(r))+','+repr(old)+',1793894340.,'+repr({h!r})+','+repr(str(overlay))+')'
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
if {host!r}=='gpuR':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
with (r/'LS_GPU_FASTK2.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(r),host={host!r},unix=time.time(),generation='FASTK2',amendment_sha256={h!r},wrapper_sha256=hashlib.sha256(text).hexdigest())
with dest.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=60)
    v=json.loads(out);p=ops.EVIDENCE/('LS_FAST_DISPATCH_'+host+'_'+root.rsplit('_',1)[1]+'.json')
    if not p.exists():write(p,v)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','result','register','broker','drain','gpu'))
    p.add_argument('--host',choices=('gpuA','gpuR'));p.add_argument('--root')
    a=p.parse_args()
    globals()[a.action](a.host,a.root) if a.action in ('drain','gpu') else globals()[a.action]()
