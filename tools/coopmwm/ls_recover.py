"""Receipt-adopting LS-COLLECT recovery; never starts training."""
import argparse
import base64
import json
import zlib
from . import ops
from .storage import read, write
from .protocol import checked, sha, REGISTRY
from .ls_collect_ops import COLLECTION, ROOT, SOURCE

RELEASE='/home/ubuntu/coopmwm20261102_lsrecoverycpu1'


def qualify():
    names=('ls_collection','ls_gpu','ls_broker','test_ls_collection_cpu')
    texts={n:checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8') for n in names}
    payload=base64.b64encode(zlib.compress(json.dumps(texts).encode())).decode()
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,shutil,hashlib,zlib,base64,subprocess,os,time
r=pathlib.Path({RELEASE!r});base=pathlib.Path({SOURCE!r});assert r.parent.is_dir()
texts=json.loads(zlib.decompress(base64.b64decode({payload!r})))
if not r.exists():
 shutil.copytree(base,r,ignore=shutil.ignore_patterns('LS_CPU*','LS_OVERLAY*','__pycache__'))
 for n,text in texts.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('w') as f:json.dump(hashes,f)
for n,text in texts.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
 with (r/'LS_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_cpu_test',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))''',timeout=40)
    print(out)


def cpu_result():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({RELEASE!r});p=r/'LS_CPU_RESULT.json'
if p.exists():print(p.read_text())
else:print(json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-3000:])))''',timeout=30)
    v=json.loads(out)
    if not v.get('pending'):
        dest=ops.EVIDENCE/'LS_RECOVERY_CPU.json'
        if not dest.exists():write(dest,v)
        print(json.dumps({k:v[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
        if not v['pass_']:print(ops.ssh.remote('cpuQ',f'from pathlib import Path\nprint(Path({RELEASE!r},"LS_CPU.log").read_text()[-16000:])',timeout=30))
    else:print(out)


def census(host):
    roots = dict(cpuQ=[COLLECTION],
        gpuA=['/root/coopmwm20261102_incident1','/root/coopmwm20261102_deadlineA2'],
        gpuR=['/root/coopmwm20261102_concurrent3','/root/coopmwm20261102_deadlineR2'])[host]
    out=ops.ssh.remote(host,f'''import pathlib,json,time
out=[]
for name in {roots!r}:
 r=pathlib.Path(name);v=dict(root=name,receipts={{}})
 names=('DISPATCH.json','LS_RECOVERY_CONFIG.json','LS_GPU_DISPATCH.json','LS_GPU_ORIGINALS.json','LS_GPU_RESTART.json','LS_GPU_RESTART_RECOVERY1.json','LS_GPU_FAILED_RECOVERY1.json','SPOOL_OWNER.json','SERVICE_OWNER.json')
 for n in names:
  p=r/n
  if not p.exists():continue
  x=json.loads(p.read_text())
  if n=='LS_GPU_ORIGINALS.json':x=dict(names=list(x),binding_hash_count=len(x['GPU_BINDING.json']['hashes']))
  v['receipts'][n]=x
  if 'pid' in x:
   q=pathlib.Path('/proc',str(x['pid']))
   if q.exists():
    st=(q/'stat').read_text().split();v[n+'_process']=dict(state=st[2],same_start=st[21]==x['start_ticks'],argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode())
 if {host!r}=='cpuQ':
  v['broker_assignments']=len(list((r/'broker_assignments').glob('*.json')))
  v['completed']=len(list((r/'broker_completed').glob('*.json')))
 out.append(v)
print(json.dumps(dict(host={host!r},unix=time.time(),roots=out)))''',timeout=35)
    value=json.loads(out)
    write(ops.EVIDENCE/f'LS_RECOVERY_CENSUS_{host}_{int(value["unix"])}.json',value)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['census','qualify','cpu_result']);p.add_argument('--host',choices=['cpuQ','gpuA','gpuR'])
    a=p.parse_args()
    if a.action=='census':census(a.host)
    else:globals()[a.action]()
