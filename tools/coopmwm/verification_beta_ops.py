"""MAIN16:07 fixed verified guidance; adopt the current wave, never replay physics."""
import argparse
import copy
import json
import time
from pathlib import Path
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261102_verifybeta13'
RESOURCE_DEADLINE=1793894340.0  # 2026-11-05 23:59:00 +08, MAIN16:34


def extended_config(value):
    """Replace only resource-clock fields; preserve every scientific setting."""
    if isinstance(value,dict):
        return {k:RESOURCE_DEADLINE if k in ('gpu_deadline','deadline_unix') else extended_config(v)
                for k,v in value.items()}
    if isinstance(value,list):return [extended_config(v) for v in value]
    return value


def replace_resource_config(path,value):
    """Atomic mutable operational pointer; caller first preserves the old receipt."""
    import os
    from .warm_spool import canonical
    path=Path(path);temporary=path.with_name(path.name+'.deadline1634.publishing')
    with temporary.open('xb') as f:
        f.write(canonical(value));f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def _resource_gpu_main(root,deadline,amendment):
    """Fresh process after the old worker's last forecast, never replay a claim."""
    import os,sys,time,json,signal,hashlib,fcntl
    from pathlib import Path
    root=Path(root);sys.path.insert(0,str(root/'pytools'))
    from tools.coopmwm.warm_spool import immutable,SpoolInput
    def replace(path,value):
        tmp=path.with_name(path.name+'.deadline1634.publishing')
        with tmp.open('x') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    old=json.loads((root/'SPOOL_OWNER.json').read_text())
    p=Path('/proc',str(old['pid']))
    def owned():
        assert old['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if not p.exists():return False
        st=(p/'stat').read_text().split();assert st[21]==old['start_ticks']
        if st[2]=='Z':return False
        argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        assert argv[-3:]==['-m','tools.coopmwm.warm_spool',str(root)],argv
        return True
    assert owned()
    # cpuQ's broker is receipt-verified and frozen before this controller starts.
    quiet=0
    while quiet<3:
        assert owned(),'old worker exited before its forecasts settled'
        missing=[f.stem for f in (root/'inbox').glob('*.json') if not (root/'forecasts'/f.name).exists()]
        quiet=0 if missing else quiet+1
        time.sleep(2)
    names=['GPU_BINDING.json','SPOOL_OWNER.json','SERVICE_OWNER.json']+[f.name for f in root.glob('STARTUP_*.json')]
    original={n:json.loads((root/n).read_text()) for n in names}
    b=original['GPU_BINDING.json']
    for n,h in b['hashes'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    immutable(root/'RESOURCE_EXTENSION_ORIGINALS_1634.json',original)
    assert owned();os.kill(old['pid'],signal.SIGTERM)
    for _ in range(150):
        if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
        assert (p/'stat').read_text().split()[21]==old['start_ticks'];time.sleep(.1)
    else:raise ValueError('old idle worker did not exit; no second signal')
    claim=(root/'SPOOL_WORKER.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Binding and completed qualification receipts remain byte-for-byte unchanged.
    b=dict(b,deadline_unix=deadline,gpu_deadline=deadline,resource_extension_amendment=amendment)
    owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=old['boot_id'],unix=time.time(),new_sumo_sim_s=0)
    replace(root/'SPOOL_OWNER.json',owner)
    immutable(root/'RESOURCE_EXTENSION_RESTART_1634.json',dict(old=old,new=owner,
        gpu_deadline=deadline,amendment_sha256=amendment,all_inbox_completed=True,inference_replayed=False))
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    sys.stdin=SpoolInput(root,deadline)
    from tools.coopmwm import warm_service,plan_service
    # Four source-identical gates adopted; the four full/fast cold/warm gates
    # still run fresh in the new process before SERVICE_OWNER is published.
    b['adopted_startup_gates']=dict(
        receipts={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in warm_service.ADOPTABLE},
        numerical_hashes={n:b['hashes'][n] for n in warm_service.NUMERICAL})
    previous_read=plan_service.read
    def config_read(path):
        return b if path==root/'GPU_BINDING.json' else previous_read(path)
    warm_service.read=plan_service.read=config_read
    exclusive=plan_service.write
    def metadata_write(path,value):
        # Only the snapshotted startup/owner pointers may be replaced. Inference
        # forecasts retain the original exclusive-create semantics.
        if path.parent==root and path.name in original:
            if path.name.startswith('STARTUP_'):
                return exclusive(root/(path.stem+'_RESOURCE1651.json'),value)
            if path.name=='SERVICE_OWNER.json':return replace(path,value)
        return exclusive(path,value)
    warm_service.write=plan_service.write=metadata_write
    try:warm_service.main(root)  # Same numerics; four adopted + four fresh gates.
    except BaseException as error:
        immutable(root/'SPOOL_FAILED.json',dict(owner=owner,error=repr(error),unix=time.time(),
            inference_retry=False,new_sumo_sim_s=0));raise


def extend_register():
    """Register MAIN16:34 before changing any resource clocks or process owners."""
    # MAIN16:51: superseded. This path replaced immutable bindings in place
    # (V1 failed with FileExistsError on WARM_BINDING.json). The extension is the
    # additive RESOURCE_EXTENSION_1651 record read by resource_config.clocks.
    raise ValueError('superseded by additive RESOURCE_EXTENSION_1651 (MAIN16:51); never overwrite immutable bindings')
    from . import ops
    from .protocol import REGISTRY,checked,sha
    from .warm_deploy import ROOT
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    line='\nS-COOPMWM MAIN1634 RESOURCE EXTENSION/DESIGN FREEZE REGISTERED: OWNER-delegated MAIN extends all component resource clocks to 2026-11-05 23:59:00 +08 (unix1793894340). Cap9500000/latestMAIN1556 allocation unchanged. NO design/search/horizon/arm/loss/selection change. Original clocks/configs/owners preserved in append-only receipts. Explicit gpu_deadline on active bindings and nested operational configs. Frozen broker prevents new submissions while each owned GPU completes ALL original inbox claims; restart only after every forecast settles, same numerical source and eight fresh startup parity gates per new process. Replace cached-clock broker/controller and child-free waiting R; preserve current B4 physics, adopt u3/Adam/RNG/library/results once, beta1 for fresh B/R decisions. No inference/episode replay. Original B4 live children still have their original cached deadline; monitor completion before expiry and disclose any migration blocker. No new sim-s. Design frozen for rest of run; engineering uptime repairs only. CPU'+str(cpu['tests'])+'PASS.\n'
    line+='Engineering V2: original registration/deployment retained; exclusive-create storage guard stopped BEFORE the first binding/owner change. Atomic operational pointer replacement only after original configuration snapshot; CPU-qualified repair, no physics replay.\n'
    target=ops.EVIDENCE/'RESOURCE_EXTENSION_REGISTRATION_1634_V2.json'
    assert not target.exists(),'adopt existing extension before repeating registration'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(line=line,authority='MAIN2026-11-02 16:34',gpu_deadline=RESOURCE_DEADLINE,
        cpu=cpu,source_root=SOURCE,new_sim_s=0))
    h=sha(target)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,os,signal,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.verification_beta_ops import extended_config,replace_resource_config
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.storage import write
assert not (r/'RESOURCE_EXTENSION_1634.json').exists(),'adopt existing extension'
cpu=json.loads({json.dumps(cpu)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
names=('WARM_BINDING.json','EXACT_OPERATIONAL_BINDING.json','WARM_B_RUNTIME_BINDING_V4.json','WARM_R_RUNTIME_BINDING_V4.json','VERIFICATION_BETA_BINDING.json','WARM_REMOTE_BINDING.json','MATCH_BROKER_CONFIG.json')
original={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}
immutable(r/'RESOURCE_EXTENSION_ORIGINALS_1634.json',original)
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r}+'amendmentSHA'+{h!r}+'\\n')
for n,v in original.items():
 v=extended_config(v);v['gpu_deadline']={RESOURCE_DEADLINE!r}
 if n=='VERIFICATION_BETA_BINDING.json':v.update(source_root=str(s),cpu=cpu,receipt_suffix='_DL1')
 replace_resource_config(r/n,v)
v=dict(unix=time.time(),gpu_deadline={RESOURCE_DEADLINE!r},deadline_unix={RESOURCE_DEADLINE!r},amendment_sha256={h!r},source_root=str(s))
immutable(r/'RESOURCE_EXTENSION_1634.json',v)
print(json.dumps(v))''',timeout=40)
    write(ops.EVIDENCE/'RESOURCE_EXTENSION_1634.json',json.loads(out));print(out)


def extend_dispatch():
    """Start only the durable migration controller; it owns subsequent handovers."""
    from . import ops
    from .warm_deploy import ROOT
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,os,signal,subprocess
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.verification_beta_ops import identity
assert not (r/'RESOURCE_EXTENSION_DISPATCH_1634.json').exists(),'adopt existing migration'
v=json.loads((r/'RESOURCE_EXTENSION_1634.json').read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
old=json.loads((r/'VERIFICATION_BETA_TRANSITION_DISPATCH_V3.json').read_text())
p=pathlib.Path('/proc',str(old['pid']));st=(p/'stat').read_text().split();assert st[21]==old['start_ticks'] and old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
assert argv==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.verification_beta_ops','serve','--root',str(r),'--config','VERIFICATION_BETA_TRANSITION_CONFIG_V3.json']
assert not (p/'task'/str(old['pid'])/'children').read_text().strip()
os.kill(old['pid'],signal.SIGTERM)
broker=json.loads((r/'MATCH_BROKER_DISPATCH.json').read_text());assert identity(broker) in ('S','R')
os.kill(broker['pid'],signal.SIGSTOP)
cfg=json.loads((r/'VERIFICATION_BETA_TRANSITION_CONFIG_V3.json').read_text())
cfg.update(controller_source=str(s),source_root=str(s),receipt_suffix='_V4',binding_sha256=__import__('hashlib').sha256((r/'VERIFICATION_BETA_BINDING.json').read_bytes()).hexdigest(),old_broker=broker)
immutable(r/'VERIFICATION_BETA_TRANSITION_CONFIG_V4.json',cfg)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'RESOURCE_EXTENSION_1634.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.verification_beta_ops','extend_serve','--root',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();d=dict(pid=p.pid,start_ticks=st[21],boot_id=broker['boot_id'],root=str(r),source=str(s),unix=time.time())
immutable(r/'RESOURCE_EXTENSION_DISPATCH_1634.json',d);print(json.dumps(d))''',timeout=40)
    write(ops.EVIDENCE/'RESOURCE_EXTENSION_DISPATCH_1634.json',json.loads(out));print(out)


def extend_gpu():
    import inspect
    from . import ops
    reg=read(ops.EVIDENCE/'RESOURCE_EXTENSION_1634.json')
    # Explicit owned roots: no directory discovery, no other project's process.
    roots=dict(gpuA=('/root/coopmwm20261102_incident1','/root/coopmwm20261102_second1'),
               gpuR=('/root/coopmwm20261102_concurrent3','/root/coopmwm20261102_second1'))
    script=inspect.getsource(_resource_gpu_main)+'\nimport sys\n_resource_gpu_main(sys.argv[1],float(sys.argv[2]),sys.argv[3])\n'
    for host,paths in roots.items():
        stage='/root/coopmwm20261102_deadlineops1'
        ops.stage(host,stage,{'restart.py':script})
        out=ops.ssh.remote(host,f'''import pathlib,json,os,sys,time,subprocess
s=pathlib.Path({stage!r});out=[]
for name in {paths!r}:
 r=pathlib.Path(name);assert r.is_dir();assert not (r/'RESOURCE_EXTENSION_DISPATCH_1634.json').exists(),'adopt existing GPU restart'
 b=json.loads((r/'GPU_BINDING.json').read_text())
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(r/'pytools'))
 with (r/'RESOURCE_EXTENSION_1634.log').open('x') as log:
  p=subprocess.Popen([sys.executable,'-B','-u',str(s/'pytools/restart.py'),str(r),str({RESOURCE_DEADLINE!r}),{reg['amendment_sha256']!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();d=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),unix=time.time(),host={host!r})
 with (r/'RESOURCE_EXTENSION_DISPATCH_1634.json').open('x') as f:json.dump(d,f)
 out.append(d)
print(json.dumps(out))''',timeout=40)
        write(ops.EVIDENCE/f'RESOURCE_EXTENSION_GPU_{host}_1634.json',json.loads(out));print(out)


def extend_serve(root):
    import os,signal,subprocess,sys
    from .warm_spool import immutable
    from .warm_broker import Remote,worker_endpoints
    cfg=read(root/'VERIFICATION_BETA_TRANSITION_CONFIG_V4.json')
    v=read(root/'RESOURCE_EXTENSION_1634.json')
    assert v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    try:
        endpoints=worker_endpoints(root,read(root/'WARM_REMOTE_BINDING.json'),{})
        ready={}
        while len(ready)<len(endpoints):
            for slot,e in endpoints.items():
                if slot in ready:continue
                s=Remote(root,e).call(dict(action='status'))
                owner=s['receipts'].get('SPOOL_OWNER.json',{});service=s['receipts'].get('SERVICE_OWNER.json',{})
                if s['alive'] and s['ready'] and not s['failed'] and owner.get('unix',0)>v['unix'] and service.get('pid')==owner.get('pid'):
                    ready[slot]=owner
            if time.time()>RESOURCE_DEADLINE:raise ValueError('extended resource deadline')
            time.sleep(5)
        old=cfg['old_broker'];assert identity(old)=='T';os.kill(old['pid'],signal.SIGKILL)
        for _ in range(150):
            if identity(old) is None:break
            time.sleep(.1)
        else:raise ValueError('old broker did not exit')
        bc=read(root/'MATCH_BROKER_CONFIG.json');bc['receipt_tag']='_DL1'
        write(root/'RESOURCE_BROKER_CONFIG_1634.json',bc)
        with (root/'RESOURCE_BROKER_1634.log').open('x') as log:
            p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.capacity_broker',str(root),str(root/'RESOURCE_BROKER_CONFIG_1634.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        st=Path('/proc',str(p.pid),'stat').read_text().split()
        immutable(root/'RESOURCE_BROKER_DISPATCH_1634.json',dict(pid=p.pid,start_ticks=st[21],boot_id=old['boot_id'],root=str(root),source=cfg['source_root'],unix=time.time(),component='broker'))
        immutable(root/'RESOURCE_GPU_HANDOVER_COMPLETE_1634.json',dict(unix=time.time(),workers=ready,gpu_deadline=RESOURCE_DEADLINE))
        serve(root,'VERIFICATION_BETA_TRANSITION_CONFIG_V4.json')
    except BaseException as e:
        immutable(root/'RESOURCE_EXTENSION_FAILED_1634.json',dict(unix=time.time(),error=repr(e)))
        raise


def status():
    from . import ops
    from .warm_deploy import ROOT
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time
r=pathlib.Path({ROOT!r});out={{}}
for a in ('B','R'):
 name='BETA_'+a+'_DISPATCH.json' if (r/('BETA_'+a+'_DISPATCH.json')).exists() else 'VERIFIED_'+a+'_DISPATCH.json'
 d=json.loads((r/name).read_text());p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert not st or st[21]==d['start_ticks']
 children=[]
 if st:
  for c in (p/'task'/str(d['pid'])/'children').read_text().split():
   q=pathlib.Path('/proc',c);children.append(dict(pid=int(c),argv=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(),children=(q/'task'/c/'children').read_text()))
 b=json.loads((r/'VERIFICATION_CONTINUATION_BINDING.json').read_text())
 jobs=[r/'jobs'/('warm_u004_'+a+'_'+str(w['seed'])) for w in b['plans'][a]['worlds']]
 out[a]=dict(owner=d,state=st[2] if st else 'exited',children=children,claimed=sum((j/'CLAIM.json').exists() for j in jobs),results=sum((j/'RESULT.json').exists() for j in jobs),closed=sum((j/'CLOSE.json').exists() for j in jobs))
names=('VERIFICATION_BETA_BINDING.json','VERIFICATION_BETA_TRANSITION_DISPATCH.json','VERIFICATION_BETA_TRANSITION_DISPATCH_V2.json','VERIFICATION_BETA_TRANSITION_DISPATCH_V3.json','VERIFICATION_BETA_TRANSITION_FAILED.json','VERIFICATION_BETA_TRANSITION_FAILED_V2.json','VERIFICATION_BETA_TRANSITION_FAILED_V3.json','VERIFICATION_BETA_HANDOVER_COMPLETE.json','VERIFICATION_BETA_HANDOVER_COMPLETE_V2.json','VERIFICATION_BETA_HANDOVER_COMPLETE_V3.json')
receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}
for n,d in receipts.items():
 if '_TRANSITION_DISPATCH' not in n:continue
 p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert not st or st[21]==d['start_ticks']
 d['observed_state']=st[2] if st else 'exited'
if 'VERIFICATION_BETA_BINDING.json' in receipts:
 b=receipts['VERIFICATION_BETA_BINDING.json'];receipts['VERIFICATION_BETA_BINDING.json']={{k:b[k] for k in ('source_root','resume','sumo_verify','receipt_suffix')}}
logs={{n:(r/n).read_text()[-5000:] for n in ('VERIFICATION_BETA_TRANSITION.log','VERIFICATION_BETA_TRANSITION_V2.log','VERIFICATION_BETA_TRANSITION_V3.log','BETA_R.log','BETA_B.log') if (r/n).exists()}}
deadlines={{n:json.loads((r/n).read_text()).get('gpu_deadline') for n in ('EXACT_OPERATIONAL_BINDING.json','WARM_BINDING.json','WARM_B_RUNTIME_BINDING_V4.json') if (r/n).exists()}}
print(json.dumps(dict(unix=time.time(),owners=out,receipts=receipts,logs=logs,deadlines=deadlines)))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'VERIFICATION_BETA_STATUS_{int(v["unix"])}.json',v)
    print(json.dumps(dict(unix=v['unix'],owners={a:dict(pid=x['owner']['pid'],state=x['state'],
        claimed=x['claimed'],results=x['results'],closed=x['closed'],children=len(x['children'])) for a,x in v['owners'].items()},
        receipts=v['receipts'],logs=v['logs'],deadlines=v['deadlines'])))


def cpu():
    from . import verify_ops as v
    v.SOURCE=SOURCE
    v.OVERLAYS=tuple(dict.fromkeys(v.OVERLAYS+('warm_protocol','warm_runner','warm_health',
        'r_matching','test_r_matching_cpu','test_warm_health_cpu','incident_recovery',
        'capacity_broker','verification_smoke','verification_beta_ops','warm_transport',
        'warm_broker','resource_config','quota_admission','accel_runner',
        'test_warm_transport_cpu','test_warm_broker_cpu','test_resource_config_cpu')))
    v.cpu()


def identity(d, proc=Path('/proc')):
    """Only a receipt-owned incident supervisor with exact source may be controlled."""
    assert d['boot_id']==(proc/'sys/kernel/random/boot_id').read_text().strip()
    p=proc/str(d['pid'])
    if not p.exists():return None
    st=(p/'stat').read_text().split();assert st[21]==d['start_ticks'],'PID reused'
    if st[2]=='Z':return None
    argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
    if not argv:return 'exiting'  # same boot/start; observe only, never an extra signal
    assert argv==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m',
        'tools.coopmwm.incident_recovery',d['root'],'--component',d['component']]
    env=(p/'environ').read_bytes().split(bytes([0]))
    assert ('PYTHONPATH='+d['source']+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime').encode() in env
    return st[2]


def children(d):
    p=Path('/proc',str(d['pid']));out=[]
    for c in (p/'task'/str(d['pid'])/'children').read_text().split():
        q=Path('/proc',c);st=(q/'stat').read_text().split()
        argv=[x.decode() for x in (q/'cmdline').read_bytes().split(bytes([0])) if x]
        assert not (q/'task'/c/'children').read_text().strip(),'physics descendants remain'
        tracker=len(argv)==4 and argv[1:3]==['-B','-c'] and argv[3].startswith('from multiprocessing.resource_tracker import main;main(')
        pool=len(argv)==5 and argv[1:3]==['-B','-c'] and argv[3].startswith('from multiprocessing.spawn import spawn_main; spawn_main(') and argv[4]=='--multiprocessing-fork'
        assert argv[0]=='/home/ubuntu/miniconda3/bin/python' and (tracker or pool),argv
        out.append(dict(pid=int(c),start_ticks=st[21],argv=argv,tracker=tracker))
    return out


def dispatch():
    from . import ops
    from .protocol import REGISTRY,checked,sha
    from .warm_deploy import ROOT
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    line='\nS-COOPMWM MAIN1607 VERIFIED GUIDE CORRECTION REGISTERED: beta=1 for both B/R verification; adaptive trust reporting only, early-zero-trust starvation disabled ONLY for fixed verified beta1; count/safety/harmful/learning guards retained. Preserve one query/beta RNG draw and exact branch continuation. Current B4 legacy episodes adopted as explicitly mixed-timing engineering deviation; R4 fresh. Freeze ONLY verified supervisor parents, existing SUMO/GPU children continue, wait all24 B4 RESULT+CLOSE before terminating idle pool/parent; resume from u3 Adam/RNG and adopt B4 results exactly once. No new SUMO or inference dispatch in handover. CPU'+str(cpu['tests'])+'PASS; sourceSHA'+sha(ops.LOCAL/'tools/coopmwm/sumo_verify.py')+'; original deadline retained.\n'
    target=ops.EVIDENCE/'VERIFICATION_BETA_REGISTRATION.json'
    reg=dict(line=line,authority='MAIN2026-11-02 16:07',cpu_tests=cpu['tests'],source_root=SOURCE)
    if target.exists():assert read(target)==reg,'registration changed; inspect before continuation'
    else:
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
        write(target,reg)
    amendment=sha(target)
    budget=read(ops.EVIDENCE/'FINAL_RECOVERY_MAIN1556_PLAN.json')['budget']
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,os,signal,subprocess,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.verification_beta_ops import identity
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_health import assert_running
assert_running(r,'B');assert not (r/'VERIFICATION_BETA_TRANSITION_CONFIG.json').exists(),'adopt existing handover'
cpu=json.loads({json.dumps(cpu)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
owners={{a:json.loads((r/('VERIFIED_'+a+'_DISPATCH.json')).read_text()) for a in ('B','R')}}
for a,d in owners.items():
 assert identity(d) in ('S','R')
 assert not (r/(a+'_UPDATE_004.json')).exists()
b=json.loads((r/'VERIFICATION_CONTINUATION_BINDING.json').read_text())
for a in ('B','R'):assert hashlib.sha256((r/(a+'_checkpoint_003.pt')).read_bytes()).hexdigest()==b['checkpoints'][a]
for w in b['plans']['R']['worlds']:assert not (r/'jobs'/('warm_u004_R_'+str(w['seed']))/'CLAIM.json').exists(),'R4 already attempted'
assert not (pathlib.Path('/proc',str(owners['R']['pid']))/'task'/str(owners['R']['pid'])/'children').read_text().strip(),'R not waiting child-free'
for w in b['plans']['B']['worlds']:assert (r/'jobs'/('warm_u004_B_'+str(w['seed']))/'CLAIM.json').exists(),'B4 not fully claimed'
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r}+'amendmentSHA'+{amendment!r}+'\\n')
b.update(cpu=cpu,source_root=str(s),amendment_sha256={amendment!r},receipt_suffix='_SB1',budget=json.loads({json.dumps(budget)!r}))
b['sumo_verify']['guide_beta']=1.
immutable(r/'VERIFICATION_BETA_BINDING.json',b)
cfg=dict(owners=owners,source_root=str(s),amendment_sha256={amendment!r},cpu_tests=cpu['tests'],binding_sha256=hashlib.sha256((r/'VERIFICATION_BETA_BINDING.json').read_bytes()).hexdigest())
immutable(r/'VERIFICATION_BETA_TRANSITION_CONFIG.json',cfg)
for a,d in owners.items():
 assert identity(d) in ('S','R');os.kill(d['pid'],signal.SIGSTOP)
immutable(r/'VERIFICATION_BETA_PARENTS_FROZEN.json',dict(unix=time.time(),owners=owners,signals='SIGSTOP parents ONLY; children continue'))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'VERIFICATION_BETA_TRANSITION.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.verification_beta_ops','serve','--root',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
d=dict(pid=p.pid,start_ticks=st[21],boot_id=owners['B']['boot_id'],root=str(r),source=str(s),unix=time.time())
immutable(r/'VERIFICATION_BETA_TRANSITION_DISPATCH.json',d);print(json.dumps(d))''',timeout=45)
    write(ops.EVIDENCE/'VERIFICATION_BETA_TRANSITION_DISPATCH.json',json.loads(out));print(out)


def recover():
    """Adopt the verified partial handover; never repeat the old R signal."""
    from . import ops
    from .protocol import REGISTRY,checked
    from .warm_deploy import ROOT
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    line='\nS-COOPMWM MAIN1607 HANDOVER EXIT-RACE REPAIR REGISTERED: first controller failed after SIGKILL of child-free R parent on transient same-PID empty argv (pre-existing Linux exit race). No R4 claims/optimizer/physics. Empty argv with matching boot/start is observing exiting, never permission to signal; wait for disappearance/zombie. Adopt original R handover without re-signalling; original B parent remains held and its SUMO/GPU work continues. Controller-only source3 CPU qualification; numerical binding/source2 unchanged. One new detached controller, receipts V2; old failure/charges retained.\n'
    target=ops.EVIDENCE/'VERIFICATION_BETA_EXIT_REPAIR_REGISTRATION.json'
    assert not target.exists(),'adopt existing repair before retry'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(line=line,cpu_tests=cpu['tests'],source_root=SOURCE,new_physics=0))
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,os,subprocess,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.verification_beta_ops import identity
from tools.coopmwm.warm_spool import immutable
old=json.loads((r/'VERIFICATION_BETA_TRANSITION_DISPATCH.json').read_text());p=pathlib.Path('/proc',str(old['pid']))
assert old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if p.exists():
 st=(p/'stat').read_text().split();assert st[21]==old['start_ticks'] and st[2]=='Z','old controller live'
assert not (r/'VERIFICATION_BETA_TRANSITION_CONFIG_V2.json').exists()
cfg=json.loads((r/'VERIFICATION_BETA_TRANSITION_CONFIG.json').read_text());b=json.loads((r/'VERIFICATION_BETA_BINDING.json').read_text())
assert identity(cfg['owners']['B'])=='T' and identity(cfg['owners']['R']) is None
assert (r/'BETA_R_HANDOVER.json').exists() and not (r/'BETA_R_DISPATCH.json').exists()
assert not (r/'B_UPDATE_004.json').exists()
assert all(not (r/'jobs'/('warm_u004_R_'+str(w['seed']))/'CLAIM.json').exists() for w in b['plans']['R']['worlds'])
cpu=json.loads({json.dumps(cpu)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
cfg.update(controller_source=str(s),receipt_suffix='_V2',controller_cpu_tests=cpu['tests'])
immutable(r/'VERIFICATION_BETA_TRANSITION_CONFIG_V2.json',cfg)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'VERIFICATION_BETA_TRANSITION_V2.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.verification_beta_ops','serve','--root',str(r),'--config','VERIFICATION_BETA_TRANSITION_CONFIG_V2.json'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=old['boot_id'],root=str(r),source=str(s),numerical_source=b['source_root'],unix=time.time())
immutable(r/'VERIFICATION_BETA_TRANSITION_DISPATCH_V2.json',v);print(json.dumps(v))''',timeout=45)
    write(ops.EVIDENCE/'VERIFICATION_BETA_TRANSITION_DISPATCH_V2.json',json.loads(out));print(out)


def resource_deadline(root,b):
    extension=root/'RESOURCE_EXTENSION_1634.json'
    if extension.exists():
        v=read(extension)
        assert v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
        assert v['gpu_deadline']==RESOURCE_DEADLINE
        return v['gpu_deadline']
    if 'gpu_deadline' in b:return b['gpu_deadline']
    return read(root/'EXACT_OPERATIONAL_BINDING.json')['gpu_deadline']


def repair_wait():
    from . import ops
    from .protocol import REGISTRY,checked
    from .warm_deploy import ROOT
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    line='\nS-COOPMWM MAIN1607 CONTROLLER WAIT-KEY REPAIR V2 REGISTERED: R source2 fixed-beta supervisor restored u3 and waits for B4 quota, no R4 claims. Controller V2 failed on missing top-level deadline; source4 preflight caught deadline absent from operational override too, BEFORE control or dispatch. Read actual base EXACT_OPERATIONAL_BINDING.gpu_deadline1793614022.4598482, identical WARM_BINDING; do not extend it. Controller-only V3 adopts existing R owner and waits B4; B4 children continue. Numerical source2/binding unchanged, no optimizer/episode replay, no R redispatch or repeated signals.\n'
    target=ops.EVIDENCE/'VERIFICATION_BETA_WAIT_REPAIR_REGISTRATION_V2.json';assert not target.exists()
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(target,dict(line=line,cpu_tests=cpu['tests'],source_root=SOURCE,new_physics=0))
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,time,os,subprocess,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.verification_beta_ops import identity,resource_deadline
from tools.coopmwm.warm_spool import immutable
old=json.loads((r/'VERIFICATION_BETA_TRANSITION_DISPATCH_V2.json').read_text());p=pathlib.Path('/proc',str(old['pid']))
assert old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if p.exists():
 st=(p/'stat').read_text().split();assert st[21]==old['start_ticks'] and st[2]=='Z'
assert not (r/'VERIFICATION_BETA_TRANSITION_CONFIG_V3.json').exists()
cfg=json.loads((r/'VERIFICATION_BETA_TRANSITION_CONFIG_V2.json').read_text());b=json.loads((r/'VERIFICATION_BETA_BINDING.json').read_text())
assert identity(cfg['owners']['B'])=='T'
rd=json.loads((r/'BETA_R_DISPATCH.json').read_text());assert identity(rd) in ('S','R')
assert time.time()<resource_deadline(r,b)
cpu=json.loads({json.dumps(cpu)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
cfg.update(controller_source=str(s),receipt_suffix='_V3',controller_cpu_tests=cpu['tests'])
immutable(r/'VERIFICATION_BETA_TRANSITION_CONFIG_V3.json',cfg)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'VERIFICATION_BETA_TRANSITION_V3.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',{ops.ssh.PYTHON['cpuQ']!r},'-B','-u','-m','tools.coopmwm.verification_beta_ops','serve','--root',str(r),'--config','VERIFICATION_BETA_TRANSITION_CONFIG_V3.json'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=old['boot_id'],root=str(r),source=str(s),numerical_source=b['source_root'],unix=time.time(),original_deadline=resource_deadline(r,b))
immutable(r/'VERIFICATION_BETA_TRANSITION_DISPATCH_V3.json',v);print(json.dumps(v))''',timeout=45)
    write(ops.EVIDENCE/'VERIFICATION_BETA_TRANSITION_DISPATCH_V3.json',json.loads(out));print(out)


def serve(root,config='VERIFICATION_BETA_TRANSITION_CONFIG.json'):
    import os
    import signal
    import subprocess
    import sys
    from .warm_spool import immutable
    from .warm_health import assert_running
    from .protocol import sha
    cfg=json.loads((root/config).read_text());suffix=cfg.get('receipt_suffix','')
    b=json.loads((root/'VERIFICATION_BETA_BINDING.json').read_text())
    assert sha(root/'VERIFICATION_BETA_BINDING.json')==cfg['binding_sha256']
    assert cfg['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
    try:
        for a in ('R','B'):
            existing=root/f'BETA_{a}_DISPATCH.json'
            if existing.exists():
                current=read(existing)
                assert identity(current) in ('S','R'),'inspect existing replacement owner'
                if suffix!='_V4' or current['source']==b['source_root']:continue
                # R is still waiting for B4's quota: no live physics may be stopped.
                assert a=='R' and not children(current)
                assert not any((root/'jobs'/('warm_u004_R_'+str(w['seed']))/'CLAIM.json').exists()
                               for w in b['plans']['R']['worlds'])
                immutable(root/'BETA_R_PRE_EXTENSION_DISPATCH_1634.json',current)
                os.kill(current['pid'],signal.SIGSTOP)
                assert identity(current)=='T';os.kill(current['pid'],signal.SIGKILL)
                for _ in range(150):
                    if identity(current) is None:break
                    time.sleep(.1)
                else:raise ValueError('old waiting R did not exit')
                # Preserve the original dispatch above; publish the new owner below.
                d=current;partial=True
            else:
                d=cfg['owners'][a]
                partial=(root/f'BETA_{a}_HANDOVER.json').exists()
            assert identity(d) is None if partial else identity(d)=='T','old parent must remain frozen or verified handover complete'
            if a=='B':
                while True:
                    assert_running(root,a)
                    jobs=[root/'jobs'/('warm_u004_B_'+str(w['seed'])) for w in b['plans']['B']['worlds']]
                    if all((j/'RESULT.json').exists() and (j/'CLOSE.json').exists() for j in jobs):break
                    if time.time()>resource_deadline(root,b):raise ValueError('original resource deadline')
                    time.sleep(5)
            if not partial:
                cs=children(d)
                immutable(root/f'BETA_{a}_HANDOVER.json',dict(unix=time.time(),old=d,children=cs,
                    resume_update=3,adopt_current_wave=a=='B',signals='TERM verified idle pool; KILL held parent, never resume old code'))
                for c in cs:
                    if not c['tracker']:
                        assert Path('/proc',str(c['pid']),'stat').read_text().split()[21]==c['start_ticks']
                        os.kill(c['pid'],signal.SIGTERM)
                assert identity(d)=='T';os.kill(d['pid'],signal.SIGKILL)
            for _ in range(150):
                if identity(d) is None:break
                time.sleep(.1)
            else:raise ValueError('old parent did not exit')
            # install.verify will recheck all claims/results, checkpoint SHA and source hashes.
            import tools.coopmwm
            tools.coopmwm.__path__.insert(0,str(Path(b['source_root'])/'pytools/tools/coopmwm'))
            from .incident_recovery import install
            wr,_=install(root,a);pre=wr.verify(root,a)
            assert pre['resume_update']==3 and pre['sumo_verify']['guide_beta']==1.
            immutable(root/f'BETA_{a}_PREFLIGHT.json',dict(unix=time.time(),resume_update=3,
                tracked_beta=pre['resume_progress']['beta'],effective_guide_beta=1.,
                restored_state_sha256=__import__('tools.coopmwm.protocol',fromlist=['digest']).digest(pre['resume_progress']),new_physics=0))
            replacing=a=='R' and suffix=='_V4' and existing.exists()
            assert replacing or not existing.exists()
            env=dict(os.environ);env['PYTHONPATH']=cfg['source_root']+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime'
            with (root/f'BETA_{a}.log').open('x') as log:
                p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.incident_recovery',str(root),'--component',a],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            st=Path('/proc',str(p.pid),'stat').read_text().split()
            dispatched=dict(pid=p.pid,start_ticks=st[21],boot_id=d['boot_id'],root=str(root),source=cfg['source_root'],component=a,first_update=4,sumo_verification=True,guide_beta=1.,unix=time.time(),gpu_deadline=resource_deadline(root,b))
            if replacing:
                immutable(root/'BETA_R_RESOURCE_DISPATCH_1634.json',dispatched)
                replace_resource_config(existing,dispatched)
            else:immutable(existing,dispatched)
        immutable(root/('VERIFICATION_BETA_HANDOVER_COMPLETE'+suffix+'.json'),dict(unix=time.time(),guide_beta=1.,source_root=cfg['source_root'],adopted_B4_legacy_wave=True,R4_fresh=True))
    except BaseException as e:
        immutable(root/('VERIFICATION_BETA_TRANSITION_FAILED'+suffix+'.json'),dict(unix=time.time(),error=repr(e)))
        raise


def resume_register():
    """MAIN16:51 additive binding; preserve every original config and failed job."""
    from . import ops
    from .warm_deploy import ROOT
    from .protocol import REGISTRY,checked,sha
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    target=ops.EVIDENCE/'RESOURCE_RECOVERY_PLAN_1651.json'
    assert not target.exists(),'adopt existing recovery registration'
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib,time
r=pathlib.Path({ROOT!r});b=json.loads((r/'VERIFICATION_BETA_BINDING.json').read_text());failed=[];complete=[]
assert not (r/'RESOURCE_EXTENSION_1651.json').exists()
assert not list(r.glob('PAUSE_*.json'))
for a in ('B','R'):assert not (r/(a+'_UPDATE_004.json')).exists()
for w in b['plans']['B']['worlds']:
 key='warm_u004_B_'+str(w['seed']);p=r/'jobs'/key
 assert (p/'CLOSE.json').exists()
 (complete if (p/'RESULT.json').exists() else failed).append(key)
assert len(complete)==14 and len(failed)==10
stop=r/'WARM_GUIDED_STOP_V4.json';v=json.loads(stop.read_text())
assert v['source']=='capacity broker' and 'Temporary failure in name resolution' in v['error']
originals={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ('WARM_BINDING.json','EXACT_OPERATIONAL_BINDING.json','VERIFICATION_BETA_BINDING.json','WARM_REMOTE_BINDING.json')}}
print(json.dumps(dict(binding=b,failed=failed,complete=complete,stop_sha256=hashlib.sha256(stop.read_bytes()).hexdigest(),originals=originals,unix=time.time())))''',timeout=40)
    state=json.loads(out);b=state['binding'];cost=1200*len(state['failed'])
    assert cost<50000
    b['budget']['phases']['discovery']-=cost;b['budget']['phases']['train']+=cost
    assert b['budget']['phases']['discovery']>=120000
    b['budget']['phases']['sumo_verification_k8']=1500000
    b['budget'].update(total=11000000,allocated=11000000,unallocated=0)
    assert sum(b['budget']['phases'].values())==b['budget']['allocated']
    b.update(source_root=SOURCE,cpu=cpu,gpu_deadline=RESOURCE_DEADLINE,receipt_suffix='_DNS1651',
        job_id_overrides={k:k.replace('warm_u','warmdns1651_u',1) for k in state['failed']},
        originals=state['originals'],stop_sha256=state['stop_sha256'])
    assert b['sumo_verify']['k']==4 and b['sumo_verify']['guide_beta']==1.
    value=dict(binding=b,state={k:v for k,v in state.items() if k!='binding'},new_sim_s=cost,
        k8_active=False,learned_search_active=False,authority='MAIN16:34/16:43/16:51; MAIN09:58 delegated reserve rule')
    write(target,value);h=sha(target)
    line='\nS-COOPMWM MAIN1651 DNS ENGINEERING RECOVERY REGISTERED planSHA'+h+': additive RESOURCE_EXTENSION_1651 only, no immutable binding replacement; all clocks2026-11-05 23:59+08, beta1 both verified guides. Ten closed/no-result B4 attempts replaced with NEW IDs, same registered TRAIN worlds/no fresh seeds, identical noise wave and u3 Adam/RNG/library;14 completed results adopted. Cost12000 transferred discovery150200->138200/train2840400->2852400; all original charges retained. Cap11000000; extra1500000 restricted sumo_verification_k8 INACTIVE pending activation addendum, existing9500000 phases otherwise unchanged. Exact DNS-stop SHA release only; V5 faults fatal. SSH DNS/closed/reset/timeout/exit255 metadata retries6attempts/300s, identity/gates checked before identical delivery; no inference/physics replay. ScreenK4/H300 and finalK8/Hfinal unchanged, search/learned seeds/k8 verification inactive. CPU'+str(cpu['tests'])+'PASS. No sealed outcome access.\n'
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/'RESOURCE_RECOVERY_REGISTRATION_1651.json',dict(line=line,amendment_sha256=h))
    value['binding']['amendment_sha256']=h
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.incident_recovery import retired_key
v=json.loads({json.dumps(value)!r});b=v['binding']
for n,h in b['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
for n,h in b['originals'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
keys=[]
for folder in ('requests','centre_requests'):
 for p in (r/folder).glob('*.json'):
  if retired_key(json.loads(p.read_text()),v['state']['failed']):keys.append(p.stem)
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
immutable(r/'DNS_RETIRED_JOBS_1651.json',dict(jobs=v['state']['failed'],query_keys=keys,amendment_sha256=b['amendment_sha256']))
immutable(r/'RESOURCE_EXTENSION_1651.json',b)
immutable(r/'DNS_GUIDED_RELEASE_1651.json',dict(original_stop_sha256=b['stop_sha256'],amendment_sha256=b['amendment_sha256']))
old=json.loads((r/'MATCH_BROKER_CONFIG.json').read_text())
old.update(source_root=str(s),gpu_deadline=b['gpu_deadline'],deadline_unix=b['gpu_deadline'],receipt_tag='_DNS1651',amendment_sha256=b['amendment_sha256'])
immutable(r/'DNS_BROKER_CONFIG_1651.json',old)
print(json.dumps(dict(bound=True,unix=time.time(),retired_queries=len(keys),new_sim_s=v['new_sim_s'],gpu_deadline=b['gpu_deadline'],budget=b['budget'])))''',timeout=45)
    write(ops.EVIDENCE/'RESOURCE_RECOVERY_BOUND_1651.json',json.loads(out));print(out)


def resume_dispatch():
    """One durable recovery controller, never duplicate existing dispatch receipts."""
    from . import ops
    from .warm_deploy import ROOT
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.incident_recovery import binding
assert not (r/'DNS_RECOVERY_DISPATCH_1651.json').exists(),'adopt existing controller'
b=binding(r)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
with (r/'DNS_RECOVERY_1651.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.verification_beta_ops','resume_serve','--root',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();d=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),unix=time.time())
immutable(r/'DNS_RECOVERY_DISPATCH_1651.json',d);print(json.dumps(d))''',timeout=40)
    write(ops.EVIDENCE/'DNS_RECOVERY_DISPATCH_1651.json',json.loads(out));print(out)


def resume_serve(root):
    """Reconcile exited owners, stop ONLY the verified idle B pool, restore u3."""
    import os,sys,signal,subprocess
    from .incident_recovery import binding,install
    from .warm_spool import immutable
    from .warm_health import assert_running
    b=binding(root);assert_running(root,'B')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def gone(d):
        assert d['boot_id']==boot
        p=Path('/proc',str(d['pid']))
        if not p.exists():return True
        st=(p/'stat').read_text().split();assert st[21]==d['start_ticks']
        return st[2]=='Z'
    try:
        assert not (root/'DNS_RECOVERY_STARTED_1651.json').exists()
        old=read(root/'VERIFIED_B_DISPATCH.json')
        assert identity(old)=='T'
        cs=children(old)  # refuses physics descendants, checks exact pool argv
        immutable(root/'DNS_B_IDLE_HANDOVER_1651.json',dict(old=old,children=cs,unix=time.time()))
        for c in cs:
            if not c['tracker']:
                assert Path('/proc',str(c['pid']),'stat').read_text().split()[21]==c['start_ticks']
                os.kill(c['pid'],signal.SIGTERM)
        assert identity(old)=='T';os.kill(old['pid'],signal.SIGKILL)
        for _ in range(150):
            if gone(old):break
            time.sleep(.1)
        else:raise ValueError('held B did not exit')
        for name in ('BETA_R_DISPATCH.json','VERIFICATION_BETA_TRANSITION_DISPATCH_V3.json','MATCH_BROKER_DISPATCH.json','VERIFIED_GRADER_DISPATCH.json'):
            assert gone(read(root/name)),name+' still live; adopt before recovery'
        for a in ('B','R'):
            wr,_=install(root,a);v=wr.verify(root,a)
            assert v['resume_update']==3 and v['gpu_deadline']==RESOURCE_DEADLINE and v['sumo_verify']['guide_beta']==1.
            immutable(root/f'DNS_{a}_PREFLIGHT_1651.json',dict(unix=time.time(),resume_update=3,
                restored_state_sha256=__import__('tools.coopmwm.protocol',fromlist=['digest']).digest(v['resume_progress']),new_physics=0))
        # Install must not leak B/R schedule closures into this controller's child processes.
        env=dict(os.environ,SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo')
        env['PATH']=env['SUMO_HOME']+'/bin:'+env['PATH']
        for component in ('broker','grader','B','R'):
            name='DNS_'+component.upper()+'_DISPATCH_1651.json'
            assert not (root/name).exists(),'adopt existing component'
            cmd=[sys.executable,'-B','-u','-m']
            if component=='broker':cmd+=['tools.coopmwm.capacity_broker',str(root),str(root/'DNS_BROKER_CONFIG_1651.json')]
            else:cmd+=['tools.coopmwm.incident_recovery',str(root),'--component',component]
            with (root/('DNS_'+component.upper()+'_1651.log')).open('x') as log:
                p=subprocess.Popen(['nice','-n','19']+cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            st=Path('/proc',str(p.pid),'stat').read_text().split()
            immutable(root/name,dict(pid=p.pid,start_ticks=st[21],boot_id=boot,source=b['source_root'],root=str(root),component=component,unix=time.time()))
        immutable(root/'DNS_RECOVERY_STARTED_1651.json',dict(unix=time.time(),gpu_deadline=RESOURCE_DEADLINE,
            beta=1.,resume=dict(B=3,R=3),adopt_completed_B4=14,replacement_attempts=10,k8_active=False))
    except BaseException as e:
        immutable(root/'DNS_RECOVERY_FAILED_1651.json',dict(unix=time.time(),error=repr(e)));raise


def gpu_register():
    from . import ops
    from .protocol import checked,REGISTRY,sha
    from .warm_deploy import ROOT
    target=ops.EVIDENCE/'GPU_RESOURCE_REGISTRATION_1651.json'
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    line='\nS-COOPMWM MAIN1651 GPU CACHED-CLOCK HANDOVER REGISTERED: controller freezes only receipt-verified DNS broker; each of four owned spools finishes ALL inbox claims/forecasts before old idle process exits by verified PID. Preserve original GPU_BINDING and startup receipts byte-identically; additive restart/qualification receipts, snapshot original owner pointers, replace owner pointers only. Same frozen numeric source, four exact source-bound startup gates adopted and four cold/warm full-fast gates rerun per fresh process before service READY. Explicit in-memory deadline2026-11-05 23:59+08; no inference replay, zero SUMO. Broker restarted once with fresh owner baseline after all four services READY. CPU'+str(cpu['tests'])+'PASS.\n'
    value=dict(line=line,cpu=cpu,new_sim_s=0)
    if target.exists():assert read(target)==value,'registration differs'
    else:
        write(target,value)
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(target)+'\n')
    h=sha(target)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os,signal
r=pathlib.Path({ROOT!r});d=json.loads((r/'DNS_BROKER_DISPATCH_1651.json').read_text())
assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split();assert st[21]==d['start_ticks'] and st[2] in ('S','R')
argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
assert argv==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.capacity_broker',str(r),str(r/'DNS_BROKER_CONFIG_1651.json')]
assert not (r/'GPU_RESOURCE_BINDING_1651.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({(line+'amendmentSHA'+h+chr(10))!r})
v=dict(amendment_sha256={h!r},gpu_deadline={RESOURCE_DEADLINE!r},broker=d,unix=time.time(),controller_source={SOURCE!r})
with (r/'GPU_RESOURCE_BINDING_1651.json').open('x') as f:json.dump(v,f)
os.kill(d['pid'],signal.SIGSTOP)
print(json.dumps(v))''',timeout=30)
    write(ops.EVIDENCE/'GPU_RESOURCE_BINDING_1651.json',json.loads(out));print(out)


def gpu_workers(host):
    """Fresh-process qualification only; always adopt an existing dispatch first."""
    import inspect
    from . import ops
    reg=read(ops.EVIDENCE/'GPU_RESOURCE_BINDING_1651.json')
    roots=dict(gpuA=('/root/coopmwm20261102_incident1','/root/coopmwm20261102_second1'),
               gpuR=('/root/coopmwm20261102_concurrent3','/root/coopmwm20261102_second1'))
    stage='/root/coopmwm20261102_deadlineops2'
    script=inspect.getsource(_resource_gpu_main)+'\nimport sys\n_resource_gpu_main(sys.argv[1],float(sys.argv[2]),sys.argv[3])\n'
    ops.stage(host,stage,{'restart.py':script})
    out=ops.ssh.remote(host,f'''import pathlib,json,os,sys,time,subprocess
s=pathlib.Path({stage!r});out=[]
for name in {roots[host]!r}:
 r=pathlib.Path(name);assert r.is_dir()
 old=json.loads((r/'RESOURCE_EXTENSION_DISPATCH_1634.json').read_text())
 p=pathlib.Path('/proc',str(old['pid']))
 assert old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert not p.exists() or (p/'stat').read_text().split()[21]==old['start_ticks'] and (p/'stat').read_text().split()[2]=='Z','first helper still live'
 assert "can't open file" in (r/'RESOURCE_EXTENSION_1634.log').read_text()
 assert not (r/'RESOURCE_EXTENSION_DISPATCH_1651_V2.json').exists(),'adopt existing second helper'
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(r/'pytools'))
 assert (s/'restart.py').is_file()
 with (r/'RESOURCE_EXTENSION_1651_V2.log').open('x') as log:
  p=subprocess.Popen([sys.executable,'-B','-u',str(s/'restart.py'),str(r),str({RESOURCE_DEADLINE!r}),{reg['amendment_sha256']!r}],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();d=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),unix=time.time(),host={host!r})
 with (r/'RESOURCE_EXTENSION_DISPATCH_1651_V2.json').open('x') as f:json.dump(d,f)
 out.append(d)
print(json.dumps(out))''',timeout=40)
    write(ops.EVIDENCE/f'GPU_RESOURCE_DISPATCH_{host}_1651_V2.json',json.loads(out));print(out)


def gpu_dispatch():
    from . import ops
    from .warm_deploy import ROOT
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,sys,time,subprocess
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
assert not (r/'GPU_RESOURCE_CONTROLLER_1651_V3.json').exists()
old=json.loads((r/'GPU_RESOURCE_CONTROLLER_1651_V2.json').read_text());p=pathlib.Path('/proc',str(old['pid']))
assert old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
st=(p/'stat').read_text().split();assert st[21]==old['start_ticks']
argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
assert argv==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.verification_beta_ops','gpu_serve','--root',str(r)]
__import__('os').kill(old['pid'],__import__('signal').SIGTERM)
for _ in range(150):
 if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
 assert (p/'stat').read_text().split()[21]==old['start_ticks'];time.sleep(.1)
else:raise ValueError('old observer controller did not exit')
env=dict(os.environ,PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'AUTHORITY.md').open('a') as handle:handle.write('MAIN1651 resource handover incremental admission: admit only fresh READY owner/service pair; resume broker once BOTH GPU hosts have a ready worker, hot-add remaining migrated slots at search boundaries; dead slots remain quarantined, no old claim replay; same horizons/search/grades. CPU154PASS.\\n')
with (r/'GPU_RESOURCE_1651_V3.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.verification_beta_ops','gpu_serve','--root',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();d=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),unix=time.time())
immutable(r/'GPU_RESOURCE_CONTROLLER_1651_V3.json',d);print(json.dumps(d))''',timeout=30)
    write(ops.EVIDENCE/'GPU_RESOURCE_CONTROLLER_1651_V3.json',json.loads(out));print(out)


def gpu_serve(root):
    import os,signal,sys,subprocess
    from .warm_broker import Remote,worker_endpoints
    from .warm_spool import immutable
    try:
        v=read(root/'GPU_RESOURCE_BINDING_1651.json');old=v['broker']
        assert v['amendment_sha256'] in (root/'AUTHORITY.md').read_text()
        endpoints=worker_endpoints(root,read(root/'WARM_REMOTE_BINDING.json'),{})
        ready={};broker_started=False
        while len(ready)<len(endpoints):
            for slot,e in endpoints.items():
                if slot in ready:continue
                s=Remote(root,e).call(dict(action='status'))
                owner=s['receipts'].get('SPOOL_OWNER.json',{});service=s['receipts'].get('SERVICE_OWNER.json',{})
                if s['failed'] and not s['alive']:
                    # An already-dead worker/claim is never replayed or resurrected.
                    # capacity_broker will quarantine it while healthy slots serve.
                    ready[slot]=dict(quarantined=True,original_owner=owner,
                        failure=s['receipts'].get('SPOOL_FAILED.json'));continue
                # SERVICE_OWNER is published only after all fresh gates pass.
                if s['alive'] and s['ready'] and not s['failed'] and owner.get('unix',0)>v['unix'] and service.get('pid')==owner.get('pid'):
                    ready[slot]=owner
                    immutable(root/('GPU_RESOURCE_READY_'+slot+'_1651.json'),dict(slot=slot,
                        owner=owner,service=service,amendment_sha256=v['amendment_sha256']))
            hosts={slot.split('@')[0] for slot,x in ready.items() if not x.get('quarantined')}
            if not broker_started and hosts=={'gpuA','gpuR'}:
                start_resource_broker(root,v);broker_started=True
            if time.time()>RESOURCE_DEADLINE:raise ValueError('extended resource deadline')
            time.sleep(5)
        assert broker_started,'no healthy worker on each GPU'
        immutable(root/'GPU_RESOURCE_COMPLETE_1651.json',dict(unix=time.time(),workers=ready,gpu_deadline=RESOURCE_DEADLINE,no_inference_replay=True))
    except BaseException as e:
        immutable(root/'GPU_RESOURCE_FAILED_1651.json',dict(unix=time.time(),error=repr(e)));raise


def start_resource_broker(root,v):
        import os,signal,sys,subprocess
        from .warm_spool import immutable
        old=v['broker']
        p=Path('/proc',str(old['pid']));st=(p/'stat').read_text().split()
        assert old['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and st[21]==old['start_ticks'] and st[2]=='T'
        argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
        assert argv[-3:]==['tools.coopmwm.capacity_broker',str(root),str(root/'DNS_BROKER_CONFIG_1651.json')]
        os.kill(old['pid'],signal.SIGKILL)
        for _ in range(150):
            if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
            assert (p/'stat').read_text().split()[21]==old['start_ticks'];time.sleep(.1)
        else:raise ValueError('old broker failed to exit')
        cfg=read(root/'DNS_BROKER_CONFIG_1651.json');cfg.update(receipt_tag='_DNS1651_DL',resource_ready_only=True,source_root=SOURCE)
        immutable(root/'GPU_RESOURCE_BROKER_CONFIG_1651.json',cfg)
        source=SOURCE
        env=dict(os.environ,PYTHONPATH=source+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
        with (root/'GPU_RESOURCE_BROKER_1651.log').open('x') as log:
            p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.capacity_broker',str(root),str(root/'GPU_RESOURCE_BROKER_CONFIG_1651.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        st=Path('/proc',str(p.pid),'stat').read_text().split()
        immutable(root/'GPU_RESOURCE_BROKER_DISPATCH_1651.json',dict(pid=p.pid,start_ticks=st[21],boot_id=old['boot_id'],source=source,root=str(root),component='broker',unix=time.time()))


def broker_repair():
    """Keep policy episodes live when a legacy posthoc worker dies mid-query."""
    from . import ops
    from .warm_deploy import ROOT
    from .protocol import checked,REGISTRY,sha
    cpu=read(ops.EVIDENCE/('CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
    line='\nS-COOPMWM MAIN1651 LEGACY POSTHOC WORKER ISOLATION REGISTERED: unsupported-family bug persists in legacy gpuR spools; old failed gpuA second claim retained. A per-query dead worker terminally declines its assigned PLAN request on CPU (policy continues; no inference retry/no positive score), quarantines only that slot; diagnostics never fabricated. All other assigned GPU inferences adopted through source-bound broker replacement, no SUMO restart. Stop only receipt-verified migration controller so legacy R second is not hot-admitted; fresh fixed-code replacement spools qualify8EXACT0 then hot-add. Same search/horizons/Nexec/trust/SUMO-only loss, zero new sim-s. CPU'+str(cpu['tests'])+'PASS.\n'
    target=ops.EVIDENCE/'BROKER_ISOLATION_REGISTRATION_1651.json';assert not target.exists()
    write(target,dict(line=line,cpu=cpu,new_sim_s=0));h=sha(target)
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,sys,time,signal,subprocess,hashlib
r=pathlib.Path({ROOT!r});s=pathlib.Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
assert not (r/'GPU_RESOURCE_BROKER_DISPATCH_1651_V2.json').exists()
cpu=json.loads({json.dumps(cpu)!r})
for n,h in cpu['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h
for name,module,tail in [('GPU_RESOURCE_CONTROLLER_1651_V3.json','tools.coopmwm.verification_beta_ops',['gpu_serve','--root',str(r)]),('GPU_RESOURCE_BROKER_DISPATCH_1651.json','tools.coopmwm.capacity_broker',[str(r),str(r/'GPU_RESOURCE_BROKER_CONFIG_1651.json')])]:
 d=json.loads((r/name).read_text());p=pathlib.Path('/proc',str(d['pid']))
 assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if not p.exists():continue
 st=(p/'stat').read_text().split();assert st[21]==d['start_ticks']
 if st[2]=='Z':continue
 argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
 assert argv==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m',module]+tail,argv
 os.kill(d['pid'],signal.SIGTERM)
 for _ in range(150):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  assert (p/'stat').read_text().split()[21]==d['start_ticks'];time.sleep(.1)
 else:raise ValueError('old broker/controller did not exit')
with (r/'AUTHORITY.md').open('a') as handle:handle.write({(line+'amendmentSHA'+h+chr(10))!r})
cfg=json.loads((r/'GPU_RESOURCE_BROKER_CONFIG_1651.json').read_text());cfg.update(source_root=str(s),receipt_tag='_DNS1651_FIX',amendment_sha256={h!r})
immutable(r/'GPU_RESOURCE_BROKER_CONFIG_1651_V2.json',cfg)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'GPU_RESOURCE_BROKER_1651_V2.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.capacity_broker',str(r),str(r/'GPU_RESOURCE_BROKER_CONFIG_1651_V2.json')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split();v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=str(s),root=str(r),component='broker',unix=time.time())
immutable(r/'GPU_RESOURCE_BROKER_DISPATCH_1651_V2.json',v);print(json.dumps(v))''',timeout=50)
    write(ops.EVIDENCE/'GPU_RESOURCE_BROKER_DISPATCH_1651_V2.json',json.loads(out));print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('status','cpu','dispatch','serve','recover','repair_wait','resume_register','resume_dispatch','resume_serve','gpu_register','gpu_workers','gpu_dispatch','gpu_serve','broker_repair'));p.add_argument('--root',type=Path);p.add_argument('--config',default='VERIFICATION_BETA_TRANSITION_CONFIG.json');p.add_argument('--host',choices=('gpuA','gpuR'))
    a=p.parse_args()
    if a.mode=='serve':serve(a.root,a.config)
    elif a.mode=='extend_serve':extend_serve(a.root)
    elif a.mode=='resume_serve':resume_serve(a.root)
    elif a.mode=='gpu_serve':gpu_serve(a.root)
    elif a.mode=='gpu_workers':gpu_workers(a.host)
    else:globals()[a.mode]()
