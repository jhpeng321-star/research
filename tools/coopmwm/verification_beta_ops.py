"""MAIN16:07 fixed verified guidance; adopt the current wave, never replay physics."""
import argparse
import copy
import json
import time
from pathlib import Path
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261102_verifybeta5'


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
        'capacity_broker','verification_smoke','verification_beta_ops')))
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
                assert identity(read(existing)) in ('S','R'),'inspect existing replacement owner'
                continue
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
            assert not (root/f'BETA_{a}_DISPATCH.json').exists()
            env=dict(os.environ);env['PYTHONPATH']=cfg['source_root']+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime'
            with (root/f'BETA_{a}.log').open('x') as log:
                p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.incident_recovery',str(root),'--component',a],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            st=Path('/proc',str(p.pid),'stat').read_text().split()
            immutable(root/f'BETA_{a}_DISPATCH.json',dict(pid=p.pid,start_ticks=st[21],boot_id=d['boot_id'],root=str(root),source=cfg['source_root'],component=a,first_update=4,sumo_verification=True,guide_beta=1.,unix=time.time()))
        immutable(root/('VERIFICATION_BETA_HANDOVER_COMPLETE'+suffix+'.json'),dict(unix=time.time(),guide_beta=1.,source_root=cfg['source_root'],adopted_B4_legacy_wave=True,R4_fresh=True))
    except BaseException as e:
        immutable(root/('VERIFICATION_BETA_TRANSITION_FAILED'+suffix+'.json'),dict(unix=time.time(),error=repr(e)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('status','cpu','dispatch','serve','recover','repair_wait'));p.add_argument('--root',type=Path);p.add_argument('--config',default='VERIFICATION_BETA_TRANSITION_CONFIG.json')
    a=p.parse_args()
    if a.mode=='serve':serve(a.root,a.config)
    else:globals()[a.mode]()
