"""Separate, one-use B/native supplement; original analysis stays immutable."""
import argparse
import base64
import json
import time
from pathlib import Path

from . import ops
from .ls_validity_ops import remote
from .ls_collect_ops import ROOT
from .protocol import checked, sha, REGISTRY
from .storage import read, write
from .interim_u8_ops import BASE, SOURCE as ORIGINAL

SOURCE = '/home/ubuntu/coopmwm20261103_interimnative1'
CONTROL = ROOT+'/interim_native_1335_v1'
REG = 'INTERIM_U8_NATIVE_REGISTRATION_1335.json'


def qualify():
    files = {n: checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8') for n in
             ('interim_u8_native_supplement.py', 'test_interim_u8_native_supplement_cpu.py')}
    v = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,os,subprocess,time,hashlib
r=Path({SOURCE!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for n,text in files.items():
 p=r/n
 if p.exists():assert p.read_text()==text,n
 else:
  with p.open('x') as f:f.write(text)
p=r/'CPU.json'
if not p.exists():
 env=dict(os.environ,PYTHONPATH={SOURCE!r}+':'+{ORIGINAL!r}+':'+{BASE!r}+'/pytools',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 start=time.time();run=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-m','unittest','discover','-s',str(r),'-p','test_interim_u8_native_supplement_cpu.py','-v'],env=env,capture_output=True,text=True,timeout=60)
 v=dict(pass_=run.returncode==0,stdout=run.stdout,stderr=run.stderr,wall_s=time.time()-start,new_sim_s=0,new_inference=0,source_hashes={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files}})
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())''', timeout=75))
    write(ops.EVIDENCE/'INTERIM_U8_NATIVE_CPU_1335.json', v)
    print(json.dumps(v))


def register():
    p = ops.EVIDENCE/REG
    if not p.exists():
        cpu = read(ops.EVIDENCE/'INTERIM_U8_NATIVE_CPU_1335.json')
        assert cpu['pass_'] and cpu['new_sim_s'] == cpu['new_inference'] == 0
        old = read(ops.EVIDENCE/'INTERIM_U8_ANALYSIS_REGISTRATION_1235.json')
        delivered = read(ops.EVIDENCE/'INTERIM_U8_ANALYSIS_DELIVERED_1235.json')
        assert delivered['pass_'] and delivered['new_sim_s'] == 0
        line = ('\nS-COOPMWM MAIN2026-11-03 13:35 B-NATIVE INTERIM SUPPLEMENT: '
            'fixed B u8 vs reused native on the same completed 96-world checked interim grid; '
            'J600/J900 all and focus, existing episode safety, unchanged whole-world paired '
            'bootstrap10000/RNG20261101/95CI. Secondary descriptive native contrast; '
            'no new Holm family or adjusted confirmatory claim. Original resultSHA '
            +delivered['result_sha256']+' retained immutable. MAIN-only delivery. '
            'Zero new SUMO/inference/reservation; no training/search/checkpoint changes.\n')
        write(p, dict(authority='MAIN2026-11-03 13:35', line=line, cpu=cpu,
            driver_sha256=cpu['source_hashes']['interim_u8_native_supplement.py'],
            original_driver=ORIGINAL+'/interim_u8_analysis.py',
            original_driver_sha256=old['driver_sha256'],
            original_registration=ROOT+'/INTERIM_U8_ANALYSIS_REGISTRATION_1235.json',
            panel_sha256=old['panel_sha256'], public_manifest_sha256=old['public_manifest_sha256'],
            original_result_sha256=delivered['result_sha256'], new_sim_s=0))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a', encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ', f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/{REG!r};v=json.loads({json.dumps(dict(v,amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''', timeout=40))


def launch():
    raise RuntimeError('SUPERSEDED before dispatch: adopt interim_native_ops / INTERIM_U8_ANALYSIS_DISPATCH_1335_V2.json; never launch a duplicate supplement')
    # Original unlaunched draft retained under the no-deletion rule.
    v=json.loads(remote('cpuQ', f'''from pathlib import Path
import json,os,subprocess,fcntl,time,hashlib
r=Path({ROOT!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/{REG!r}).read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
assert hashlib.sha256(Path({SOURCE!r},'interim_u8_native_supplement.py').read_bytes()).hexdigest()==v['driver_sha256']
env=dict(os.environ,PYTHONPATH={SOURCE!r}+':'+{ORIGINAL!r}+':'+{BASE!r}+'/pytools',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (c/'RUN.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u',{SOURCE!r}+'/interim_u8_native_supplement.py',str(r),str(c),str(r/{REG!r})],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),control_root=str(c),new_sim_s=0)
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=45))
    p=ops.EVIDENCE/'INTERIM_U8_NATIVE_DISPATCH_1335.json'
    if not p.exists():write(p,v)
    print(json.dumps(v))


def status():
    v=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({CONTROL!r});d=json.loads((c/'DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
s=(p/'stat').read_text().split() if p.exists() else None
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert s is None or s[21]==d['start_ticks']
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'ABSENT',complete=json.loads((c/'COMPLETE.json').read_text()) if (c/'COMPLETE.json').exists() else None,failed=json.loads((c/'FAILED.json').read_text()) if (c/'FAILED.json').exists() else None)))''',timeout=40))
    write(ops.EVIDENCE/f'INTERIM_U8_NATIVE_STATUS_{time.time_ns()}.json',v)
    print(json.dumps(v))


def deliver():
    v=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,base64,hashlib
c=Path({CONTROL!r});v=json.loads((c/'COMPLETE.json').read_text());assert v['pass_'] and not (c/'FAILED.json').exists()
b=(c/'RESULT.json').read_bytes();assert hashlib.sha256(b).hexdigest()==v['result_sha256']
print(json.dumps(dict(complete=v,payload=base64.b64encode(b).decode())))''',timeout=40))
    target=checked(Path('E:/agent-work/_sol/COOPMWM_INTERIM_U8_NATIVE_MAIN.json'))
    assert target.parent.is_dir()
    blob=base64.b64decode(v['payload'])
    if target.exists():assert target.read_bytes()==blob
    else:
        with target.open('xb') as f:f.write(blob)
    receipt=dict(v['complete'],main_report=str(target),local_sha256=sha(target))
    p=ops.EVIDENCE/'INTERIM_U8_NATIVE_DELIVERED_1335.json'
    if not p.exists():write(p,receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','register','launch','status','deliver'))
    p.parse_args()
    raise SystemExit('SUPERSEDED before dispatch: use tools.coopmwm.interim_native_ops; original draft/CPU/registration retained, zero SUMO and no duplicate analysis job')
