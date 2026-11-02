"""Registered rolling, drain-before-retire it2_K2 collection deployment."""
import argparse
import json
from . import ops
from .ls_collect_ops import COLLECTION, ROOT
from .ls_validity_ops import remote, payload
from .protocol import checked, REGISTRY, sha, digest
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261103_lsfastcpu5'
OVERLAY='/root/coopmwm20261103_it2k2ops5'
REG='LS_FAST_IT2K2_REGISTRATION_V5.json'
GPU_GENERATION='FASTK2V5'
ATTEMPT='FASTK2V5'
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
        line+=('V2 metadata correction: first broker replacement failed BEFORE serving '
            'on an immutable BROKER_OWNER_VALIDITY1 collision. Retain that failure; '
            'explicit new receipt suffix_FASTK2V2, no original owner-file overwrite.\n')
        line+=('V3 wrapper correction: existing roots already contain LS_GPU_ORIGINALS, '
            'so inherited ls_gpu.run requires recover=True. First gpuR successor571419 '
            'failed BEFORE signalling its original500795; no query replay. Keep that '
            'failure and use new FASTK2V3 receipt generation/overlay, same search. '
            'Read-only observer/controller history receipts use unique nanosecond names.\n')
        line+=('V4 transport/ownership correction: inherited owned() treats empty argv '
            'as exit before flock is released. First V3 successors107959/573879 failed '
            'before qualification on SPOOL_WORKER lock contention. Same CPU231-qualified '
            'wrapper/math retained; launcher now waits for actual PID exit/zombie and '
            'probes lock release BEFORE starting successor. Original source/bindings and '
            'all completed queries retained; new RECOVERY dispatch/log names. No inference '
            'retry, no new SUMO, no gate or tolerance weakened.\n')
        line+=('V5 qualification-memory amendment: gpuR576918 passed K2/H300 and '
            'committedK4/H300, then full1600-slot H420 trace recapture OOM alongside '
            'its live peer. Use EXISTING warm_service SHA-bound adoption for four '
            'unchanged committed/single-batch gates (including K8/H420); run fresh '
            'K2/H300 plus four cold/warm full-vs-fast gates per successor. No search '
            'horizon/K/population/action change. Source mismatches or nonzero errors '
            'still FAIL. Preserve OOM and completed receipts. Release unused cache '
            'between qualification stages. Adopt any already-live earlier successor; '
            'never interrupt its inference or restart a successful qualifier.\n')
        write(p,dict(line=line,cpu=cpu,authority='MAIN2026-11-03 00:48',
            config=dict(name='it2_K2',iterations=2,screen_K=2,screen_H=300,final_K=8),
            new_sumo_sim_s=0,operational_spend_cap=6000000))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(p)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(p),cpu_tests=cpu['tests'])))


def broker():
    reg=read(ops.EVIDENCE/REG);h=sha(ops.EVIDENCE/REG)
    out=remote('cpuQ',f'''from pathlib import Path
import json,os,sys,time,signal,subprocess,hashlib
c=Path({COLLECTION!r});s=Path({SOURCE!r});r=Path({ROOT!r});dest=c/'BROKER_DISPATCH_FASTK2V2.json'
if dest.exists():print(dest.read_text());raise SystemExit
reg=json.loads({json.dumps(reg)!r})
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
old=json.loads((c/'BROKER_DISPATCH_FASTK2.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert not p.exists() or ((p/'stat').read_text().split()[21]==old['start_ticks'] and (p/'stat').read_text().split()[2]=='Z'),'adopt live owner'
assert 'conflicting immutable spool receipt BROKER_OWNER_VALIDITY1.json' in (c/'BROKER_FASTK2.log').read_text()
sys.path.insert(0,str(s/'pytools'));from tools.coopmwm.warm_spool import immutable
immutable(c/'BROKER_RETIRED_FASTK2V2.json',dict(owner=old,unix=time.time(),no_inference_or_physics_signals=True,reason='startup owner-receipt collision; no queries claimed'))
with (r/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+chr(10))
env=dict(os.environ,PYTHONPATH=str(s/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'BROKER_FASTK2V2.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_broker',str(c),'_FASTK2V2'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(c),source_root=str(s),unix=time.time(),amendment_sha256={h!r})
immutable(dest,d);print(json.dumps(d))''',timeout=60)
    v=json.loads(out);p=ops.EVIDENCE/'LS_FAST_BROKER_DISPATCH_V2.json'
    if not p.exists():write(p,v)
    print(out)


def drain(host, root):
    """Drain one current owner; call again only to observe the same request."""
    slot=SLOTS[root.rsplit('_',1)[1]]
    prior=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
c=Path({COLLECTION!r});p=c/'gpu_drains'/({digest(slot)!r}+'.json');done=c/'gpu_drained'/p.name
print(json.dumps(dict(requested=json.loads(p.read_text()),drained=json.loads(done.read_text()) if done.exists() else None)) if p.exists() else '{{}}')'''))
    if prior:print(json.dumps(prior));return
    from .ls_validity_ops import gpu_status
    import contextlib,io
    with contextlib.redirect_stdout(io.StringIO()):g=gpu_status(host)
    w=next(w for w in g['workers'] if w['owner']['root']==root)
    d=w['owner'];assert d.get('generation')!='FASTK2'
    out=remote('cpuQ',f'''from pathlib import Path
import json,sys,time
c=Path({COLLECTION!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.protocol import digest
broker=json.loads((c/'BROKER_DISPATCH_FASTK2V2.json').read_text());p=Path('/proc',str(broker['pid']))
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
    qualified=read(ops.EVIDENCE/REG)['cpu']['source_hashes']['pytools/tools/coopmwm/ls_fast_worker.py']
    assert sha(ops.LOCAL/'tools/coopmwm/ls_fast_worker.py')==qualified,'wrapper changed since CPU qualification'
    out=remote(host,f'''from pathlib import Path
import json,os,time,subprocess,hashlib,base64,shutil,fcntl,signal
r=Path({root!r});dest=r/'LS_GPU_DISPATCH_FASTK2V5.json';overlay=Path({OVERLAY!r})
if dest.exists():print(dest.read_text());raise SystemExit
dispatch_lock=(r/'FAST_UPGRADE_DISPATCH.claim').open('a');fcntl.flock(dispatch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert overlay.parent.is_dir()
if not overlay.exists():shutil.copytree('/root/coopmwm20261102_lsvalidops1',overlay,ignore=shutil.ignore_patterns('__pycache__'))
text=base64.b64decode({packed!r});target=overlay/'ls_fast_worker.py'
if target.exists():assert target.read_bytes()==text
else:target.write_bytes(text)
old={old!r};assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
assert all((r/'forecasts'/q.name).exists() for q in (r/'inbox').glob('*.json'))
p=Path('/proc',str(old['pid']));old['argv']=''
if p.exists():
 st=(p/'stat').read_text().split();assert st[21]==old['start_ticks']
 if st[2]!='Z':
  old['argv']=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  assert 'ls_gpu' in old['argv'] and str(r) in old['argv']
  os.kill(old['pid'],signal.SIGTERM)
for _ in range(300):
 try:st=(p/'stat').read_text().split()
 except FileNotFoundError:break
 assert st[21]==old['start_ticks']
 if st[2]=='Z':break
 time.sleep(.1)
else:raise ValueError('retired GPU PID still alive; no successor launched')
probe=(r/'SPOOL_WORKER.claim').open('a')
for _ in range(300):
 try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:time.sleep(.1)
else:raise ValueError('retired GPU lock still owned; no successor launched')
fcntl.flock(probe,fcntl.LOCK_UN);probe.close()
retired=r/'LS_FAST_RETIREMENT_RECOVERY.json'
if not retired.exists():
 with retired.open('x') as f:json.dump(dict(old=old,unix=time.time(),actual_exit_and_lock_release_verified=True,no_inference_replay=True),f)
code='import sys;sys.path.insert(0,'+repr(str(overlay))+');from ls_fast_worker import run;run('+repr(str(r))+','+repr(old)+',1793894340.,'+repr({h!r})+','+repr(str(overlay))+')'
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
if {host!r}=='gpuR':env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
with (r/'LS_GPU_FASTK2V5.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/root/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=proc.pid,start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],boot_id=old['boot_id'],root=str(r),host={host!r},unix=time.time(),generation='FASTK2V5',log_name='LS_GPU_FASTK2V5.log',variant_name='STARTUP_VARIANT_IT2K2_'+str(proc.pid)+'.json',amendment_sha256={h!r},wrapper_sha256=hashlib.sha256(text).hexdigest())
with dest.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=60)
    v=json.loads(out);p=ops.EVIDENCE/('LS_FAST_DISPATCH_'+host+'_'+root.rsplit('_',1)[1]+'_'+ATTEMPT+'.json')
    if not p.exists():write(p,v)
    print(out)


def progress(host,root):
    """Read-only evidence of the original search being drained, not a retry."""
    out=remote(host,f'''from pathlib import Path
import json,time,os
r=Path({root!r});owner=json.loads((r/'SERVICE_OWNER.json').read_text());p=Path('/proc',str(owner['pid']))
before=(p/'stat').read_text().split() if p.exists() else None
pending=[]
for q in (r/'inbox').glob('*.json'):
 if (r/'forecasts'/q.name).exists():continue
 record=json.loads(q.read_text());claim=r/'inference_claims'/q.name
 c=json.loads(claim.read_text()) if claim.exists() else None
 pending.append(dict(key=q.stem,world=record['world_id'],time_s=record['time_s'],group_size=len(record['group']),sensed_rows=len(record['rows']),claim=c,age_s=time.time()-c['unix'] if c else None))
time.sleep(2);after=(p/'stat').read_text().split() if p.exists() else None
assert not after or after[21]==owner['start_ticks']
cpu_s=(sum(int(after[i])-int(before[i]) for i in (13,14))/os.sysconf('SC_CLK_TCK')) if before and after else None
print(json.dumps(dict(unix=time.time(),root=str(r),owner=owner,state=after[2] if after else 'exited',pending=pending,cpu_seconds_over_2s=cpu_s)))''')
    v=json.loads(out);write(ops.EVIDENCE/f'LS_FAST_DRAIN_PROGRESS_{host}_{int(v["unix"])}.json',v)
    print(out)


def publish_variant():
    """Adopt completed K2 gates without rerunning qualification or inference."""
    target=ops.EVIDENCE/'IT2_K2_VARIANT_GPU_QUALIFIED.json'
    if target.exists():print(json.dumps(read(target)));return
    workers=[]
    for host in ('gpuA','gpuR'):
        p=max(ops.EVIDENCE.glob('LS_VALIDITY_GPU_'+host+'_*.json'),key=lambda p:p.stat().st_mtime_ns)
        row=next(w for w in read(p)['workers'] if w.get('variant_gate') is not None)
        gate=row['variant_gate']
        assert gate['pass_'] and gate['max_cost_error']==gate['full_fast']['max_cost_error']==0
        assert gate['identical_draw_ids'] and gate['full_fast']['identical_draw_ids']
        assert all(x==0 for x in gate['full_fast']['trajectory_errors'].values())
        workers.append(dict(host=host,owner=row['owner'],variant_gate=gate,
            source_receipt=str(p),source_sha256=sha(p),startup_gates_complete=len(row['gates'])))
    value=dict(unix=__import__('time').time(),pass_=True,search_config='it2_K2',
        workers=workers,new_sumo_sim_s=0,new_wm_calls=0,activation=False,
        note='parameter-size gates passed on both GPUs; wait for all eight fresh startup gates and admission',
        amendment_sha256=sha(ops.EVIDENCE/REG))
    write(target,value);print(json.dumps(value))


def continue_roll():
    """Durable local deployment observer; only already-registered rolling steps."""
    import contextlib,io,msvcrt,time
    lock=(ops.EVIDENCE/'LS_FAST_ROLL.claim').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'1');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    from .ls_validity_ops import gpu_status
    tiers=[(('gpuA','/root/coopmwm20261102_incident1'),('gpuR','/root/coopmwm20261102_concurrent3')),
        (('gpuA','/root/coopmwm20261102_deadlineA2'),('gpuR','/root/coopmwm20261102_deadlineR2'))]
    try:
        for tier in tiers:
            requested=set()
            while __import__('time').time()<1793894340.:
                with contextlib.redirect_stdout(io.StringIO()):
                    gpus={h:gpu_status(h) for h in ('gpuA','gpuR')}
                    live={w['owner']['root'] for g in gpus.values() for w in g['workers']
                        if w['owner'].get('generation','').startswith('FASTK2')
                        and w['state'] in ('S','R','D') and not w['failure']}
                    # Idempotent drain/gpu operations adopt immutable remote receipts.
                    for host,root in tier:
                        p=ops.EVIDENCE/('LS_FAST_DISPATCH_'+host+'_'+root.rsplit('_',1)[1]+'_'+ATTEMPT+'.json')
                        if not p.exists() and root not in live:
                            if root not in requested:drain(host,root);requested.add(root)
                            gpu(host,root)
                    gpus={h:gpu_status(h) for h in ('gpuA','gpuR')}
                ready={w['owner']['root'] for g in gpus.values() for w in g['workers']
                    if w['ready'] and w['owner'].get('generation','').startswith('FASTK2')}
                print(json.dumps(dict(unix=time.time(),it2_K2_ready=sorted(ready),
                    current_tier=[r for h,r in tier],no_training_or_sumo_dispatch=True)),flush=True)
                if all(root in ready for host,root in tier):break
                time.sleep(60)
            else:raise ValueError('registered resource deadline')
        value=dict(unix=time.time(),search_config='it2_K2',workers=gpus,
            amendment_sha256=sha(ops.EVIDENCE/REG),new_sumo_sim_s=0,
            mirror_active=False,training_paused=True)
        write(ops.EVIDENCE/'IT2_K2_FOUR_WORKERS_READY.json',value)
    except BaseException as error:
        import traceback
        write(ops.EVIDENCE/f'LS_FAST_ROLL_FAILED_{time.time_ns()}.json',dict(unix=time.time(),
            error=repr(error),traceback=traceback.format_exc(),inference_replayed=False))
        raise


def continue_dispatch(generation=1):
    import subprocess,sys,time
    assert generation in (1,2,3,4)
    suffix='' if generation==1 else f'_V{generation}'
    p=ops.EVIDENCE/('LS_FAST_ROLL_DISPATCH'+suffix+'.json')
    if p.exists():print(json.dumps(read(p)));return
    assert (ops.EVIDENCE/'LS_FAST_BROKER_DISPATCH_V2.json').exists()
    with (ops.EVIDENCE/('LS_FAST_ROLL'+suffix+'.log')).open('x',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','tools.coopmwm.ls_fast_ops','continue_roll'],
            cwd=ops.LOCAL,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    value=dict(pid=child.pid,unix=time.time(),source_sha256=sha(ops.LOCAL/'tools/coopmwm/ls_fast_ops.py'),
        amendment_sha256=sha(ops.EVIDENCE/REG),
        mode='registered rolling metadata/GPU qualification only; no PPO/SUMO dispatch')
    write(p,value);print(json.dumps(value))


def continue_recover():
    import psutil
    previous=ops.EVIDENCE/'LS_FAST_ROLL_DISPATCH_V3.json'
    old=read(previous if previous.exists() else ops.EVIDENCE/'LS_FAST_ROLL_DISPATCH.json')
    try:owner=psutil.Process(old['pid'])
    except psutil.NoSuchProcess:owner=None
    if owner:
        assert abs(owner.create_time()-old['unix'])<5
        assert owner.cmdline()[-3:]==['-m','tools.coopmwm.ls_fast_ops','continue_roll']
        print(json.dumps(dict(adopted_running_controller=old)));return
    continue_dispatch(4)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','result','register','broker','drain','gpu','progress','publish_variant','continue_roll','continue_dispatch','continue_recover'))
    p.add_argument('--host',choices=('gpuA','gpuR'));p.add_argument('--root')
    a=p.parse_args()
    globals()[a.action](a.host,a.root) if a.action in ('drain','gpu','progress') else globals()[a.action]()
