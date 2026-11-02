"""Grading-only recovery; GPU/broker owners and all saved searches are adopted."""
import argparse
import json

from . import ops
from .ls_collect_ops import COLLECTION, ROOT
from .ls_validity_ops import SOURCE as BASE, remote, payload
from .protocol import REGISTRY, checked, sha
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261102_lslabelcpu3'
GENERATION='SUMOENV1'
FILES=('ls_collection','ls_label_runtime','sumo_verify','test_ls_label_runtime_cpu')


def qualify():
    packed=payload(FILES)
    print(remote('cpuQ',f'''from pathlib import Path
import json,os,shutil,hashlib,subprocess,time,base64,zlib
r=Path({SOURCE!r});base=Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 r.mkdir()
 shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):shutil.copy2(base/name,r/name)
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['test_ls_label_runtime_cpu']
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
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
print(p.read_text() if p.exists() else json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-4000:])))'''))
    if value.get('pending'):print(json.dumps(value));return
    p=ops.EVIDENCE/('LS_LABEL_CPU_'+SOURCE.rsplit('/',1)[1]+'.json')
    if not p.exists():write(p,value)
    print(json.dumps({k:value[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
    if not value['pass_']:print(remote('cpuQ',f'from pathlib import Path;print(Path({SOURCE!r},"LS_CPU.log").read_text()[-14000:])'))


def register():
    cpu=read(ops.EVIDENCE/('LS_LABEL_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'));assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    target=ops.EVIDENCE/'LS_LABEL_DISK_REGISTRATION.json'
    line=('\nS-COOPMWM MAIN23:32 SUMO ENVIRONMENT ENGINEERING RECOVERY: preserve all GPU/broker owners, '
        '72 completed episodes, pools and candidates. Replace collection metadata/grading supervisor only '
        'after boot/start/argv/no-branch-child verification. Explicit existing SUMO_HOME/PATH; '
        'real policy branch must pass prefix smoke before candidate branches, reused without extra physics. '
        'Zero-physics missing-binary labels marked void/no-outcome, original files/charges retained; '
        'fresh SUMOENV1 attempts precharged in sumo_verification, no new WM searches for retries. '
        'Fresh-message conservative total cap6000000 includes retained superseded811640; '
        'reserve hard guard applies to collection admissions and retry charges. No training restart, '
        'mirror/learned activation, horizon/population/K change or final outcome access. '
        'DISK FOLLOW-UP: existing2GiB guard queues labels BEFORE verification claims, no failed labels '
        'or new charges while waiting; original SUMOENV1 smoke/labels/claims adopted unchanged.\n')
    if not target.exists():
        write(target,dict(line=line,cpu=cpu,source_root=SOURCE,generation=GENERATION,
            operational_spend_cap=6000000,sumo_home='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
            authority='MAIN2026-11-02 23:32 plus fresh resume cap',original_failed_charges_retained=True))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(target)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(target),cpu_tests=cpu['tests'],cap=6000000)))


def deploy():
    reg=read(ops.EVIDENCE/'LS_LABEL_DISK_REGISTRATION.json');h=sha(ops.EVIDENCE/'LS_LABEL_DISK_REGISTRATION.json')
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,sys,time,signal,subprocess,hashlib
c=Path({COLLECTION!r});r=Path({ROOT!r});s=Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.ls_label_runtime import sumo_environment
from tools.coopmwm.accel_storage import charged
reg=json.loads({json.dumps(reg)!r})
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
dest=c/'SUPERVISOR_DISPATCH_SUMODISK1.json'
if dest.exists():print(dest.read_text());raise SystemExit
assert all(not (r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R'))
assert all((r/'jobs'/('lscollect1_B3_'+str(w))/'RESULT.json').exists() for w in range(986000,986072))
assert charged(r)<reg['operational_spend_cap']
env=sumo_environment(reg['sumo_home']);version=subprocess.check_output([env['SUMO_HOME']+'/bin/sumo','--version'],env=env,text=True,timeout=20)
old=json.loads((c/'SUPERVISOR_DISPATCH_SUMOENV1.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
if p.exists() and (p/'stat').read_text().split()[2]!='Z':
 assert (p/'stat').read_text().split()[21]==old['start_ticks']
 argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert 'tools.coopmwm.ls_collection' in argv and str(c) in argv
 assert json.loads((c/'LS_COLLECT_STATUS.json').read_text())['grading_inflight']==0,'wait for grading boundary'
 os.kill(old['pid'],signal.SIGSTOP)
 children=(p/'task'/str(old['pid'])/'children').read_text().split()
 for child in children:
  q=Path('/proc',child);a=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  if 'multiprocessing.resource_tracker' not in a:
   os.kill(old['pid'],signal.SIGCONT);raise ValueError('live branch child; wait')
 os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT)
 for _ in range(100):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.1)
 else:raise ValueError('old metadata supervisor did not exit')
immutable(c/'SUPERVISOR_RETIRED_SUMODISK1.json',dict(owner=old,unix=time.time(),no_gpu_broker_or_physics_signals=True))
config=dict(source_root=str(s),receipt_suffix='_SUMODISK1',label_generation='SUMOENV1',
 sumo_home=reg['sumo_home'],operational_spend_cap=reg['operational_spend_cap'],amendment_sha256={h!r})
immutable(c/'LS_LABEL_DISK_CONFIG.json',config)
bound=c/'LABEL_DISK_REGISTERED.json'
if not bound.exists():
 with (r/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+chr(10))
 immutable(bound,dict(amendment_sha256={h!r},unix=time.time()))
env.update(PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'SUPERVISOR_SUMODISK1.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_collection',str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=Path('/proc',str(proc.pid),'stat').read_text().split()
d=dict(pid=proc.pid,start_ticks=st[21],boot_id=old['boot_id'],root=str(c),source_root=str(s),unix=time.time(),amendment_sha256={h!r},sumo_version=version.splitlines()[0])
immutable(dest,d);print(json.dumps(d))''',timeout=60))
    target=ops.EVIDENCE/'LS_LABEL_DISK_DISPATCH.json'
    if not target.exists():write(target,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
c=Path({COLLECTION!r});suffix='SUMODISK1' if (c/'SUPERVISOR_DISPATCH_SUMODISK1.json').exists() else 'SUMOENV1'
d=json.loads((c/('SUPERVISOR_DISPATCH_'+suffix+'.json')).read_text());p=Path('/proc',str(d['pid']))
assert not p.exists() or (p/'stat').read_text().split()[21]==d['start_ticks']
labels=[json.loads(p.read_text()) for p in (c/'labels').glob('*_SUMOENV1.json')]
print(json.dumps(dict(unix=time.time(),owner=d,state=(p/'stat').read_text().split()[2] if p.exists() else 'exited',
 smoke=json.loads((c/'LABEL_SUMO_SMOKE_SUMOENV1.json').read_text()) if (c/'LABEL_SUMO_SMOKE_SUMOENV1.json').exists() else None,
 disk_wait=json.loads((c/'LABEL_DISK_WAIT_SUMOENV1.json').read_text()) if (c/'LABEL_DISK_WAIT_SUMOENV1.json').exists() else None,
 labels=[dict(search_id=x['search_id'],status=x['verification']['status'],branches=len(x['verification'].get('branches',[])),actual_sim_s=x['verification'].get('actual_sim_s'),grade=x['verification'].get('grade_veh_s'),reason=x['verification'].get('reason')) for x in labels],
 grading_health=dict(disk_blocked=shutil.disk_usage(c).free<2*1024**3,
  owner_identity_verified=bool(p.exists() and (p/'stat').read_text().split()[21]==d['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']),
  owner_argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode() if p.exists() else None,
  children=(p/'task'/str(d['pid'])/'children').read_text().split() if p.exists() else [],
  threads=[dict(tid=t.name,wchan=(t/'wchan').read_text()) for t in (p/'task').iterdir()] if p.exists() else [],
  inflight_semantics='thread futures include queued disk waiters; not a count of running SUMO branches',
  automatic_resume='existing grading_disk_ready loop rechecks disk every 10 seconds'),
 collection=json.loads((c/'LS_COLLECT_STATUS.json').read_text()),disk_free=shutil.disk_usage(c).free,
 log_tail=(c/('SUPERVISOR_'+suffix+'.log')).read_text()[-2200:])))'''))
    write(ops.EVIDENCE/f'LS_LABEL_STATUS_{int(value["unix"])}.json',value);print(json.dumps(value));return value


def publish():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({COLLECTION!r});labels=[json.loads(p.read_text()) for p in (c/'labels').glob('*_SUMOENV1.json')]
labels=[x for x in labels if x['verification']['status'] in ('accepted','rejected')]
assert labels,'no complete real label yet'
first=min(labels,key=lambda x:x['verification']['timing']['finished_unix'])
print(json.dumps(dict(unix=time.time(),first_label=first,smoke=json.loads((c/'LABEL_SUMO_SMOKE_SUMOENV1.json').read_text()),
 voids=[json.loads(p.read_text()) for p in (c/'label_voids').glob('*.json')],new_searches_for_regrading=0)))'''))
    p=ops.EVIDENCE/'LS_FIRST_REAL_LABEL_SUMOENV1.json'
    if not p.exists():write(p,value)
    print(json.dumps(dict(published=str(p),branches=len(value['first_label']['verification']['branches']),
        status=value['first_label']['verification']['status'],smoke_pass=value['smoke']['pass_'])))


def bank_status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,sys
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.exact_grade import _bootstrap_mean
c=Path({COLLECTION!r});all_labels=[json.loads(p.read_text()) for p in (c/'labels').glob('*.json')]
labels=[x for x in all_labels if x['verification']['status'] in ('accepted','rejected')]
assert len({{x['search_id'] for x in labels}})==len(labels),'duplicate usable labels'
positive=[x for x in labels if x['verification']['status']=='accepted' and x['verification']['grade_veh_s']>0]
values=[x['verification']['best_margin_veh_s'] for x in labels]
mean,ci=_bootstrap_mean(values,[x['world_id'] for x in labels],10000,20261101) if labels else (None,None)
focus=lambda x:x['verification']['episode']['world']['demand'] in (2400,2700) and x['verification']['episode']['world']['penetration'] in (.2,.4)
print(json.dumps(dict(unix=time.time(),arm='B',policy_update=3,entries=len(labels),positives=len(positive),
 mean_exact_advantage_veh_s=mean,world_bootstrap_ci95=ci,
 estimand='best of eight exact candidate margins vs policy; includes nonpositive decisions',
 rank5_8_winners=sum(x['verification']['chosen']>=4 for x in positive),
 rank5_8_share_of_positive_winners=sum(x['verification']['chosen']>=4 for x in positive)/len(positive) if positive else None,
 worlds=sorted(set(x['world_id'] for x in labels)),focus_share=sum(focus(x) for x in labels)/len(labels) if labels else None,
 void_attempts=len(list((c/'label_voids').glob('*.json'))),training_paused=True,
 prerequisite_positive_entries=20,prerequisite_met=len(positive)>=20,
 R_bank_entries=0,training_changes_from_statistics=False)))'''))
    target=ops.EVIDENCE/'BANK_STATUS.json';tmp=target.with_suffix('.publishing')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,sort_keys=True)
    __import__('os').replace(tmp,target)
    write(ops.EVIDENCE/f'BANK_STATUS_{int(value["unix"])}.json',value)
    print(json.dumps(value));return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','result','register','deploy','status','publish','bank_status'))
    globals()[p.parse_args().action]()
