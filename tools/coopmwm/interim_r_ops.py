"""Qualified fixed-u8 R supplement; public status contains no outcome values."""
import argparse
import json
import time
from . import ops, replay_start_ops
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote
from .protocol import REGISTRY, checked, sha
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_interimrcpu2'
BASE='/home/ubuntu/coopmwm20261103_compositereleasecpu1'
CONTROL=ROOT+'/interim_r_1235_v2'
REG='INTERIM_R_REGISTRATION_1235_V2.json'


def qualify():
    names=('interim_r','interim_r_ops','test_interim_r_cpu')
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    replay_start_ops.BASE=BASE
    print(remote('cpuQ',replay_start_ops.qualification_payload(files,source=SOURCE,
        selection=['test_interim_r_cpu','test_final_endpoint_cpu','test_replay_start_cpu']),timeout=50))


def cpu_result():
    replay_start_ops.SOURCE=SOURCE;replay_start_ops.cpu_result()


def register():
    p=ops.EVIDENCE/REG
    if not p.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
        line=('\nS-COOPMWM MAIN2026-11-03 12:35 fixed-u8 interim R supplement V2: '
            'V1 failed before accounting/reservation/physics because the training config '
            'asserts its immutable source identity; V2 validates that source separately. '
            'Original failure/registration retained; no job/charge was created by V1. '
            'adopt running A/B interim3442884, never replay or interrupt it. '
            'R8 completed after the original trigger; add exactly96 R8 episode jobs, '
            '115200sim-s in restricted interim_u8_test; A/B230400 plusR115200 '
            '=345600 within registered quote. Frozen checkpoint8 and original96world '
            'panel; MAPPO/native reused, no branches. Single SCHED_IDLE/nice19 worker, '
            'no training handover; exact complete-set precharge before physics, '
            'one-use jobs/no retries. Only hashes/identities/completion counts public; '
            'outcomes remain closed pending authorised completed interim analysis.\n')
        write(p,dict(line=line,authority='MAIN2026-11-03 12:35',cpu=cpu,
            source_root=SOURCE,control_root=CONTROL,sim_s=115200,
            phase='interim_u8_test',checkpoint_update=8,arm='R',outcomes_closed=True))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});v=json.loads({json.dumps(dict(v,amendment_sha256=h))!r});p=r/{REG!r}
assert (r/'R_UPDATE_008.json').is_file()
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,sha256={h!r},quote_sim_s=115200)))''',timeout=40))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,subprocess,os,fcntl,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/{REG!r}).read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
sys.path.insert(0,str(s/'pytools'));from tools.coopmwm.replay_start import owner
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH'];assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
with (c/'RUN.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.interim_r',str(r),str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(owner(p.pid),source_root=str(s),control_root=str(c),amendment_sha256=v['amendment_sha256'])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=50))
    p=ops.EVIDENCE/'INTERIM_R_DISPATCH_1235_V2.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,time
r=Path({ROOT!r});c=Path({CONTROL!r});ab=r/'replay_composite_0957'
sys.path.insert(0,{SOURCE!r}+'/pytools');from tools.coopmwm.replay_start import alive
out=dict(unix=time.time(),outcomes_closed=True)
for key,base,name in [('AB',ab,'interim_DISPATCH.json'),('R',c,'DISPATCH.json')]:
 d=json.loads((base/name).read_text()) if (base/name).exists() else None
 out[key]=dict(owner=d,alive=alive(d) if d else False)
out['completed_by_arm']={{a:len(list((r/'final_receipts').glob('interim008_'+a+'_*.json'))) for a in ('A','B','R')}}
out['R_complete']=(c/'COMPLETE.json').exists();out['R_failed']=(c/'FAILED.json').exists()
out['AB_failed']=(ab/'interim_FAILED.json').exists()
print(json.dumps(out))''',timeout=40))
    p=ops.EVIDENCE/f'INTERIM_U8_PUBLIC_STATUS_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),**value)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cmd',choices=('qualify','cpu_result','register','launch','status'))
    globals()[p.parse_args().cmd]()
