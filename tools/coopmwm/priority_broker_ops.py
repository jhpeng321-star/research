"""Receipt-driven metadata-only centre fairness; adopt all existing GPU claims."""
import argparse
import json
from . import ops,replay_start_ops as start
from .ls_collect_ops import ROOT,COLLECTION
from .ls_validity_ops import remote
from .protocol import checked,sha,REGISTRY
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261103_rprioritycpu1'
REG='LS_CENTRE_PRIORITY_REGISTRATION_0906.json'


def qualify():
    start.BASE='/home/ubuntu/coopmwm20261103_replaylesson_cpu3'
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8')
           for n in ('ls_broker','test_ls_priority_cpu')}
    code=start.qualification_payload(files,SOURCE,
        selection=['test_ls_priority_cpu','test_ls_collection_cpu','test_niche_drain_cpu'])
    print(remote('cpuQ',code,timeout=60))


def cpu_result():
    start.SOURCE=SOURCE;start.cpu_result()


def register():
    p=ops.EVIDENCE/REG
    if not p.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
        line=('\nS-COOPMWM MAIN09:06 R bank anti-starvation engineering amendment: '
            'queue-only priority for centre-mode requests over not-yet-started full '
            'population searches; stable FIFO within both classes. Each R centre is '
            'the existing qualified K1/H300 current-policy rollout, zero R candidate '
            'scoring. Preserve all horizons/K/populations/search outputs, existing '
            'GPU workers/startup gates/claims/CRN and original saved requests. '
            'Adopt existing assignments/forecasts; idempotent submit only, no inference '
            'replay or withdrawal. Metadata broker handover only, verified own PID/'
            'boot/start/argv; children must be SSH metadata clients, never physics. '
            'No SUMO/reservations/budget change. Numerical sources unchanged from '
            '713-check base; targeted Linux queue/LS/drain checks required.\n')
        write(p,dict(line=line,cpu=cpu,source_root=SOURCE,new_sim_s=0,
            base_cpu_sha256=sha(ops.EVIDENCE/'B_REPLAY_START_CPU_coopmwm20261103_replaylesson_cpu3.json')))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(json.dumps(dict(registered=True,amendment_sha256=h,tests=v['cpu']['tests'],new_sim_s=0)))


def launch():
    p=ops.EVIDENCE/REG;v=read(p);h=sha(p)
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,sys,os,signal,subprocess,fcntl,hashlib
c=Path({COLLECTION!r});r=Path({ROOT!r});s=Path({SOURCE!r});target=c/'BROKER_DISPATCH_RPRIORITY0906.json'
lock=(c/'RPRIORITY_DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if target.exists():print(target.read_text());raise SystemExit
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
old=json.loads((c/'BROKER_DISPATCH_NICHENULL1.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert (p/'stat').read_text().split()[21]==old['start_ticks']
argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
assert 'tools.coopmwm.ls_broker' in argv and str(c) in argv
for child in (p/'task'/str(old['pid'])/'children').read_text().split():
 a=(Path('/proc',child)/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert a.startswith('ssh '),'only SSH metadata clients may be children'
with (r/'AUTHORITY.md').open('a') as f:f.write({v['line']!r}+'amendmentSHA'+{h!r}+chr(10))
os.kill(old['pid'],signal.SIGTERM)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 time.sleep(.1)
else:raise ValueError('old metadata broker did not exit')
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'BROKER_RPRIORITY0906.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_broker',str(c),'_RPRIORITY0906'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],unix=time.time(),source_root=str(s),root=str(c),old=old,amendment_sha256={h!r})
with target.open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    p=ops.EVIDENCE/'LS_CENTRE_PRIORITY_DISPATCH_0906.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','cpu_result','register','launch'))
    globals()[p.parse_args().action]()
