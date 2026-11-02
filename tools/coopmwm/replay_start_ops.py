"""Qualify, register and dispatch MAIN07:35's durable remote start-watcher once."""
import argparse
import base64
import json
import time
import zlib
from . import ops
from .ls_collect_ops import ROOT, COLLECTION
from .ls_validity_ops import remote
from .protocol import checked, sha, REGISTRY
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_replaystartcpu2'
BASE='/home/ubuntu/coopmwm20261103_replayreleasecpu6'
CONTROL=ROOT+'/replay_start_0735'
REG='B_REPLAY_START_REGISTRATION_0735.json'


def qualification_payload(files, source=SOURCE):
    packed=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    return f'''from pathlib import Path
import os,json,shutil,hashlib,subprocess,time,base64,zlib
r=Path({source!r});base=Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 r.mkdir()
 for n in ('pytools','WM_PACKAGE'):shutil.copytree(base/n,r/n,ignore=shutil.ignore_patterns('__pycache__'))
 for n in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','FROZEN_GEOMETRY_SOURCE.py','FROZEN_NETWORK.net.xml.gz','FROZEN_PRE_NICHE_SEARCH.py'):shutil.copy2(base/n,r/n)
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in json.loads((base/'LS_SOURCE_HASHES.json').read_text())}}
 for n in files:hashes['pytools/tools/coopmwm/'+n+'.py']=hashlib.sha256((r/'pytools/tools/coopmwm'/(n+'.py')).read_bytes()).hexdigest()
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['test_replay_start_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('x') as f:json.dump(sorted(set(tests)),f)
 shutil.copy2(base/'LS_CPU_SELECTION.json',r/'LS_CPU_SELECTION.json')
for n,text in files.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
if (r/'LS_CPU_RESULT.json').exists():print((r/'LS_CPU_RESULT.json').read_text())
elif (r/'LS_CPU_DISPATCH.json').exists():print((r/'LS_CPU_DISPATCH.json').read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':'+str(r/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',COOPMWM_NICHE_BASELINE=str(r/'FROZEN_PRE_NICHE_SEARCH.py'),COOPMWM_NULLFIX_GEOMETRY=str(r/'FROZEN_GEOMETRY_SOURCE.py'),COOPMWM_NULLFIX_NETWORK=str(r/'FROZEN_NETWORK.net.xml.gz'))
 code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(r/'FROZEN_GEOMETRY_SOURCE.py'))+');from tools.coopmwm.ls_cpu_test import run;run('+repr(str(r))+')'
 with (r/'LS_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with (r/'LS_CPU_DISPATCH.json').open('x') as f:json.dump(d,f)
 print(json.dumps(d))'''


def qualify():
    names=('replay_start','replay_start_ops','replay_bank','ls_collection','test_replay_start_cpu')
    files={n:checked(ops.LOCAL/f'tools/coopmwm/{n}.py').read_text(encoding='utf-8') for n in names}
    for n,text in files.items():compile(text,n+'.py','exec')
    print(remote('cpuQ',qualification_payload(files),timeout=60))


def cpu_result():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
print(p.read_text() if p.exists() else json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-3000:])))''',timeout=40))
    if not value.get('pending'):
        p=ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json')
        if not p.exists():write(p,value)
        print(json.dumps({k:value[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
        if not value['pass_']:print(remote('cpuQ',f'from pathlib import Path;print(Path({SOURCE!r},"LS_CPU.log").read_text()[-12000:])',timeout=40))
    else:print(json.dumps(value))


def inspect_live():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,time,hashlib,sys
r=Path({ROOT!r});c=Path({COLLECTION!r});sys.path.insert(0,{BASE!r}+'/pytools')
from tools.coopmwm.warm_health import assert_running
stops={{}}
for a in ('B','R'):
 try:assert_running(r,a);stops[a]=None
 except Exception as e:stops[a]=repr(e)
print(json.dumps(dict(unix=time.time(),load=os.getloadavg(),stops=stops,
 checkpoint_sha256={{a:hashlib.sha256((r/(a+'_checkpoint_003.pt')).read_bytes()).hexdigest() for a in ('B','R')}},
 current_updates={{a:[p.stem for p in r.glob(a+'_UPDATE_[0-9][0-9][0-9].json')] for a in ('B','R')}},
 paused={{a:dict(value=json.loads((r/('WARM_'+a+'_PAUSED.json')).read_text()),sha256=hashlib.sha256((r/('WARM_'+a+'_PAUSED.json')).read_bytes()).hexdigest()) if (r/('WARM_'+a+'_PAUSED.json')).exists() else None for a in ('B','R')}},
 watcher_dispatched=(Path({CONTROL!r})/'DISPATCH.json').exists(),
 transfer=json.loads((r/'B_REPLAY_PHASE_TRANSFER_0706.json').read_text()),
 demand_phase_end_s=json.loads((c/'LS_VALIDITY_CONFIG.json').read_text())['arrival_admission']['observed_phase_end_s'],
 nullfix_activated=(c/'LS_NULLFIX_CONFIG.json').exists())))''',timeout=40))
    p=ops.EVIDENCE/f'B_REPLAY_START_PREFLIGHT_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),**{k:v for k,v in value.items() if k!='transfer'})))
    return value


def register():
    path=ops.EVIDENCE/REG
    if not path.exists():
        cpu=read(ops.EVIDENCE/('B_REPLAY_START_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
        assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
        live=inspect_live();assert not any(live['stops'].values()),live['stops']
        assert all(len(live['current_updates'][a])==3 for a in ('B','R'))
        assert all(p is None or p['value']['update']<24 for p in live['paused'].values())
        budget=live['transfer']['budget'];assert budget['total']==14500000
        assert live['transfer']['funding_complete']
        # Same 24 PPO episodes capture bank snapshots: zero extra collection train cost.
        quote=dict(PPO_remaining_sim_s=1209600,additional_collection_episode_sim_s=0,
            interim_max_sim_s=345600,interim_phase='interim_u8_test',
            bank_branch_phase='sumo_verification',confirmation_lesson_phase='bank_verification_ext',
            new_reserved_sim_s=0)
        line=('\nS-COOPMWM MAIN2026-11-03 07:35 AUTOMATIC B-REPLAY START: durable cpuQ '
            'nohup-equivalent detached watcher, ownedPID/boot/start receipts, flock; poll120s. '
            'Gate ten confirmed/admitted B entries, full Linux CPU PASS, four joint-release '
            'eight-gate GPU owners and qualified collector boundary activation. Gracefully '
            'handover original bank builder only after gate; no physics interrupted. '
            'B/R exactu3 Adam/RNG restore, NEW replay0735 job namespace/no old job replay; '
            'Relocate any SHA-checked old sub24 WARM_arm_PAUSED certificate to '
            'WARM_arm_PAUSED_BEFORE_REPLAY0735.json (bytes preserved, no overwrite) '
            'only after start gate, permitting fresh final24 endpoint certificate. '
            'B/R to24, R lag1, 24own-policy PPO episodes/update, no executedplans/WMwait. '
            'Use SAME PPO episodes for nonblocking current-policy bank snapshots (zero '
            'extra train sim-s); inherited50% focus episode mix. Single existing LSbroker '
            'retained. Frozen-u3 collector parent handed over ONLY at no-active-grade '
            'boundary; new drain-only supervisor finishes saved labels without any new '
            'frozen-u3 admissions, original claims/labels/precharges retained. Broker '
            'delivers Bsearch and Rcentre-only requests; R candidates never WMscored. '
            'B it2_K2 niches1/4 alternating, unchanged finalK8/Hfinal, actionablegate both. '
            'Whole-decision precharge/one-use confirmations; all exact-positive safe '
            'same-arm bank entries AWR/cooperative exposure; per-update lessons. '
            'Fixed compute pacing minimum1800s between update starts, not outcome-based. '
            'PPO1209600 fully funded checks->train144000 transfer; total14500000 unchanged. '
            'Interim fixedu8 A/B(+R if reached at B8 trigger),96worlds episodeonly max345600 '
            'in restricted900000 interim phase; checkpoint8/no selection, MAPPO/native '
            'reused, hash-only receipts/no outcomes during training. Existing isolated '
            'final coordinator owns final24 evaluation; never duplicate it. Health/STOP '
            'rules active; dead child alerts/preserves jobs, no physics auto-retry. '
            'Register before any newphysics; unchanged failed mirror remainsOFF.\n')
        write(path,dict(line=line,authority='MAIN2026-11-03 07:35',source_root=SOURCE,
            control_root=CONTROL,collection_root=COLLECTION,budget=budget,cpu=cpu,
            checkpoints=live['checkpoint_sha256'],demand_phase_end_s=live['demand_phase_end_s'],
            old_endpoints=live['paused'],
            gpu_deadline=1793894340,pace_s=1800,interim_quote_sim_s=345600,quote=quote))
    value=read(path);h=sha(path)
    registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(value['line']+'amendmentSHA'+h+'\n')
    binding=dict(value,amendment_sha256=h)
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/'B_REPLAY_START_0735.json';v=json.loads({json.dumps(binding)!r})
assert r.is_dir() and v['demand_phase_end_s'] is not None
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,amendment_sha256={h!r},new_reserved_sim_s=0)))''',timeout=40))


def launch():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import sys,json,os,time,subprocess,fcntl,hashlib
r=Path({ROOT!r});s=Path({SOURCE!r});c=Path({CONTROL!r});assert c.parent.is_dir()
c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/'B_REPLAY_START_0735.json').read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
cpu=json.loads((s/'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':'+str(s/'WM_PACKAGE')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
assert Path(env['SUMO_HOME'],'bin/sumo').is_file()
with (c/'WATCH.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.replay_start',str(r),'watch'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_root=str(s),control_root=str(c),amendment_sha256=v['amendment_sha256'])
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''',timeout=60))
    p=ops.EVIDENCE/'B_REPLAY_START_WATCH_DISPATCH_0735.json'
    if not p.exists():write(p,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({CONTROL!r});d=json.loads((c/'DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
st=(p/'stat').read_text().split() if p.exists() else None
assert st is None or st[21]==d['start_ticks']
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
print(json.dumps(dict(unix=time.time(),owner=d,state=st[2] if st else 'ABSENT',
 status=json.loads((c/'STATUS.json').read_text()) if (c/'STATUS.json').exists() else None,
 log=(c/'WATCH.log').read_text()[-5000:],started=(Path({ROOT!r})/'B_REPLAY_STARTED.json').exists())))''',timeout=40))
    p=ops.EVIDENCE/f'B_REPLAY_START_WATCH_STATUS_{time.time_ns()}.json';write(p,value)
    print(json.dumps(dict(receipt=str(p),**value)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','cpu_result','inspect_live','register','launch','status'))
    globals()[p.parse_args().action]()
