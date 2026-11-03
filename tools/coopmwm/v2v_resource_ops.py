"""Qualify/register/start a durable watcher under MAIN16:47 additive authority."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .protocol import checked,sha,REGISTRY
from .storage import read,write

CONTROL='/home/ubuntu/coopmwm20261103_v2v_resource1647'
ROOT='/home/ubuntu/coopmwm20261103_v2v2'


def qualify():
    files={n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8')
        for n in ('v2v_resource_watch.py','test_v2v_resource_watch_cpu.py')}
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,hashlib,subprocess,time
r=Path({CONTROL!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
for n,text in json.loads({json.dumps(files)!r}).items():
 p=r/n
 if p.exists():assert p.read_text()==text
 else:
  with p.open('x') as f:f.write(text)
p=r/'CPU_RESULT.json'
if not p.exists():
 start=time.time();x=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-m','unittest','discover','-s',str(r),'-p','test_v2v_resource_watch_cpu.py','-v'],capture_output=True,text=True,timeout=30)
 v=dict(pass_=x.returncode==0,stdout=x.stdout,stderr=x.stderr,wall_s=time.time()-start,new_sim_s=0,
  source_sha256=hashlib.sha256((r/'v2v_resource_watch.py').read_bytes()).hexdigest())
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())''',timeout=45))
    p=ops.EVIDENCE/'V2V_RESOURCE_CPU_1647.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    cpu=read(ops.EVIDENCE/'V2V_RESOURCE_CPU_1647.json');assert cpu['pass_']
    original=read(ops.EVIDENCE/'V2V_RUNTIME_REGISTRATION_1640_V2.json')
    assert original['cpu']['pass_'] and original['cpu']['root'].endswith('_v2vcpu7')
    p=ops.EVIDENCE/'V2V_RESOURCE_REGISTRATION_1647.json'
    if not p.exists():
        line=('\nS-COOPMWM MAIN2026-11-03 16:27/16:47 ADDITIVE V2V RESOURCE AMENDMENT: '
            'launch free-disk threshold6GiB supersedes10GiB; rolling mover maintenance '
            'target8GiB/gpuRfloor4GiB. Durable qualified wrapper changes ONLY watcher '
            'resource admission; original runtime/config/source/769Linux CPU qualification '
            'remain immutable and hash-checked. Require live registered rolling mover '
            'and no gpuRcapacity STOP. Existing all3 commonMAPPOu0 real one-update '
            'smokes then exact u2..24 continuation, pacing/safety/diagnostic schedules, '
            'restricted phases and17471840cap unchanged. Watcher starts now and spends '
            'no SUMO before>=6GiB and original health gates; no duplicate watcher or '
            'numerical release. Original WARM_BINDING/V2V_RUNTIME never overwritten.\n')
        write(p,dict(line=line,cpu=cpu,source_sha256=cpu['source_sha256'],new_sim_s=0,
            original_registration_sha256=sha(ops.EVIDENCE/'V2V_RUNTIME_REGISTRATION_1640_V2.json')))
    value=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(value['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({CONTROL!r});root=Path({ROOT!r});v=json.loads({json.dumps(dict(value,amendment_sha256=h))!r})
p=r/'AUTHORITY.json'
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
text=v['line']+'amendmentSHA'+{h!r}+'\\n'
if {h!r} not in (root/'AUTHORITY.md').read_text():
 with (root/'AUTHORITY.md').open('a') as f:f.write(text)
q=root/'RESOURCE_ADMISSION_1647.json'
if not q.exists():
 with q.open('x') as f:json.dump(dict(required_free_bytes=6*1024**3,amendment_sha256={h!r}),f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=40))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,subprocess,time
r=Path({CONTROL!r});root=Path({ROOT!r});p=root/'WATCH_DISPATCH.json'
assert (r/'AUTHORITY.json').is_file()
if p.exists():print(p.read_text());raise SystemExit
assert not (root/'WATCH_OWNER.json').exists(),'adopt existing V2V owner'
cfg=json.loads((root/'V2V_RUNTIME.json').read_text())['config'];source=Path(cfg['source_root'])
env=dict(os.environ,PYTHONPATH=str(source/'pytools')+':'+str(source/'runtime_src')+':/home/ubuntu/e3final/runtime',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PATH='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo/bin:'+os.environ['PATH'])
command=['/home/ubuntu/miniconda3/bin/python','-B','-u',str(r/'v2v_resource_watch.py')]
with (r/'WATCH.log').open('x') as log:
 child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),argv=command,
 source_root=str(source),root=str(root),resource_gate_bytes=6*1024**3)
with p.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=45))
    p=ops.EVIDENCE/'V2V_RESOURCE_WATCH_DISPATCH_1647.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
r=Path({CONTROL!r});root=Path({ROOT!r});out=dict(unix=time.time(),cpu_free=shutil.disk_usage('/home/ubuntu').free)
for name in ('WATCH_DISPATCH','WATCH_OWNER','V2V_WATCH_STATUS','V2V_STARTED','V2V_REAL_SMOKES_PASS'):
 p=root/(name+'.json');out[name]=json.loads(p.read_text()) if p.exists() else None
p=r/'FAILED.json';out['failure']=json.loads(p.read_text()) if p.exists() else None
d=out['WATCH_DISPATCH']
if d:
 p=Path('/proc',str(d['pid']));s=(p/'stat').read_text().split() if p.exists() else None
 out['alive']=bool(s and s[21]==d['start_ticks'] and s[2]!='Z')
print(json.dumps(out))''',timeout=40))
    write(ops.EVIDENCE/f'V2V_RESOURCE_STATUS_{time.time_ns()}.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','register','launch','status'))
    globals()[p.parse_args().action]()
