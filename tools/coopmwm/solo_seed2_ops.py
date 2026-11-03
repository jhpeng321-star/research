"""Qualify, quote, register then dispatch missing second-seed solo evidence."""
import argparse
import json
import time
from . import ops
from .ls_validity_ops import remote
from .storage import write,read
from .protocol import sha,checked,REGISTRY

CONTROL='/home/ubuntu/coopmwm20261103_solo1755_v2'
SOURCE='/home/ubuntu/coopmwm20261103_compositereleasecpu1'
ROOT='/home/ubuntu/coopmwm20261102_warm1'


def qualify():
    files={n:checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8') for n in ('solo_seed2.py','test_solo_seed2_cpu.py')}
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import os,sys,json,subprocess,time,hashlib
r=Path({CONTROL!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
for n,text in json.loads({json.dumps(files)!r}).items():
 p=r/n
 if p.exists():assert p.read_text()==text
 else:
  with p.open('x') as f:f.write(text)
p=r/'CPU_QUOTE.json'
if not p.exists():
 env=dict(os.environ,PYTHONPATH={SOURCE!r}+'/pytools:'+{SOURCE!r}+'/runtime_src:/home/ubuntu/e3final/runtime',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 x=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-m','unittest','discover','-s',str(r),'-p','test_solo_seed2_cpu.py','-v'],capture_output=True,text=True,env=env,timeout=30)
 assert x.returncode==0,x.stderr
 code="import sys,json;sys.path.insert(0,"+repr(str(r))+");import solo_seed2 as m;print(json.dumps(m.quote("+repr({ROOT!r})+")))"
 q=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-c',code],capture_output=True,text=True,env=env,timeout=40)
 assert q.returncode==0,q.stderr
 v=dict(pass_=True,stdout=x.stdout,stderr=x.stderr,quote=json.loads(q.stdout),new_sim_s=0,source_sha256=hashlib.sha256((r/'solo_seed2.py').read_bytes()).hexdigest())
 with p.open('x') as f:json.dump(v,f)
print(p.read_text())''',timeout=85))
    p=ops.EVIDENCE/'SOLO_SEED2_CPU_QUOTE_1755.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def register():
    cpu=read(ops.EVIDENCE/'SOLO_SEED2_CPU_QUOTE_1755.json');assert cpu['pass_']
    record=ops.EVIDENCE/'SOLO_SEED2_REGISTRATION_1755.json'
    line=('\nS-COOPMWM MAIN17:55/18:00b second-seed solo supplement: B every first-seed '
        'positive cooperative candidate passing existing replication+later-harm checks, '
        'including986066/048/003 and qualifying newer snapshots. Frozen issue list, '
        'same exact snapshot and SAME CRN seed as existing group replication, n member-only '
        'solos + one shared policy branch; original group-replication snapshotSHA and '
        'policy-branchJ must match exactly or own supplement STOP (never imputed). '
        'No new world seeds/no policy update/no WM calls. Charge every full-prefix '
        'branch before physics to bank_verification_ext with distinct '
        'confirm_solo_seed2_1755 charge kind. Preserve original receipts; new raw evidence '
        'does NOT itself certify a CI: BANK_STAT_V2 bootstrap required. '
        f"Exact quote {cpu['quote']['total_sim_s']}sim-s, {len(cpu['quote']['entries'])}decisions. "
        'Inside existing17471840cap/restricted phase; fail before physics on insufficient '
        'room. Durable own cpuQ supervisor/nice19/two branches max; no replay on failure.\n')
    if not record.exists():write(record,dict(line=line,cpu=cpu,quote=cpu['quote'],source_sha256=cpu['source_sha256'],new_sim_s=0))
    h=sha(record);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({CONTROL!r});p=r/'AUTHORITY.json';v=json.loads({json.dumps(dict(read(record),amendment_sha256=h))!r})
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0)))''',timeout=35))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,subprocess,time
r=Path({CONTROL!r});assert (r/'AUTHORITY.json').is_file();p=r/'DISPATCH.json'
if p.exists():print(p.read_text());raise SystemExit
assert not (r/'OWNER.json').exists(),'adopt existing solo supervisor'
env=dict(os.environ,PYTHONPATH={SOURCE!r}+'/pytools:'+{SOURCE!r}+'/runtime_src:/home/ubuntu/e3final/runtime',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',PATH='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo/bin:'+os.environ['PATH'])
command=['/home/ubuntu/miniconda3/bin/python','-B','-u',str(r/'solo_seed2.py'),{ROOT!r},str(r)]
with (r/'RUN.log').open('x') as log:
 child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),argv=command)
with p.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=40))
    p=ops.EVIDENCE/'SOLO_SEED2_DISPATCH_1755.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
r=Path({CONTROL!r});out=dict(unix=time.time())
for name in ('DISPATCH','OWNER','COMPLETE','FAILED'):
 p=r/(name+'.json');out[name]=json.loads(p.read_text()) if p.exists() else None
out['supplements']=[json.loads(p.read_text()) for p in r.glob('sv_*.json')]
out['log']=(r/'RUN.log').read_text()[-4000:] if (r/'RUN.log').exists() else None
print(json.dumps(out))''',timeout=35))
    write(ops.EVIDENCE/f'SOLO_SEED2_STATUS_{time.time_ns()}.json',value)
    print(json.dumps(dict({k:v for k,v in value.items() if k!='supplements'},supplements=[
        {k:v for k,v in r.items() if k!='verification'} for r in value['supplements']])) )


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','register','launch','status'))
    globals()[p.parse_args().action]()
