"""Explicit once-only acceleration planner engineering probes; no physics dispatch."""
import argparse
import json
from . import ops
from .storage import read,write


TAG='PLAN_PROBE3'
ROOT='/root/coopmwm20261101_planprobe3'


def register():
    from .protocol import REGISTRY,checked
    line=('\nS-COOPMWM ACCELERATION PLAN GPU QUALIFICATION3 REGISTERED BEFORE TRAINING: allarmsheld; '
        'probe2full1600/fixed4K8 parityEXACT0 bothGPUs; complete32x3search635.987s(gpuA)/653.082s(gpuR), '
        '5.6605+5.5123=11.1728queries/hour, peak2.98GB each; original receipts retained. '
        'Widen batches to USE memory: whole34candidate(K4H120)screen and10candidate(K8H300)final, same32x3/elite8 search. '
        'Samefirst registered TRAIN geometry, nonzero syntheticownresidual-.25m/s2 on observable CAVs to test nongroup/postcommit own continuation; '
        'source/driver/controller private1600 roster preserved; compare singlecandidate paddedtofixed34/10 vs heterogeneousbatch, '
        'maxcosterror<=.001veh-s at BOTH screen/final stages, then one fullmediumsearch perGPU,65%memory. '
        'Finite/CRN/conservation gates unchanged; originalprobes finished, no duplicate dispatch; noSUMO. '
        'E2onfirst12registeredexecutedplans uses MAPPO background in BOTH WM/exact after decision, joint/memberreverts/all, '
        'H60/120/300/600, same committed rawprofiles then MAPPO; asynchronous diagnostics never reward/gate. '
        'Worldcluster bootstrap handles multiple windows perworld; one-world signCI[0,1], tiesseparate, scale learned notset; '
        'exacttwins stay within240000discoveryreserve. No outcome-dependent algorithm/reward change. '
        'Queryqueue pauses eligibleSUMO windows until worker slot free (p=.95 once admitted),densepriority,ownroutepositive; '
        'Rmatches logged perregime query frequency plus executed count/duration, sameEBtrust. '
        'NumericalNexec and GPU-time requote still REQUIRED before guided restart; originalGPUdeadline retained until registered amendment.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/(TAG+'_REGISTRATION.json'),dict(line=line,new_sumo_sim_s=0));print(line.strip())


def launch(host):
    root=ROOT
    assert not (ops.EVIDENCE/f'{TAG}_{host}_DISPATCH.json').exists()
    files=ops.sources()
    files['FROZEN_COSTS.json']=ops.ssh.remote('gpuA',"import pathlib;print(pathlib.Path('/root/coopmwm20261101_dual2/FROZEN_COSTS.json').read_text())",timeout=20).strip()
    files['INPUTS.json']=json.dumps(read(ops.EVIDENCE/'VOLUME_INPUTS_1854.json'))
    ops.stage(host,root,files)
    v=json.loads(ops.ssh.remote(host,f'''import os,pathlib,json,subprocess,sys,time
r=pathlib.Path({root!r});assert r.is_dir();assert not (r/'DISPATCH.json').exists()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>16000
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'PROBE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.plan_qualify',str(r)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
v=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),host={host!r},unix=time.time(),new_sumo_sim_s=0)
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/f'{TAG}_{host}_DISPATCH.json',v);print(json.dumps(v))


def status(host):
    tag=next(t for t in ('PLAN_PROBE3','PLAN_PROBE2','PLAN_PROBE1') if (ops.EVIDENCE/f'{t}_{host}_DISPATCH.json').exists())
    d=read(ops.EVIDENCE/f'{tag}_{host}_DISPATCH.json')
    v=json.loads(ops.ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({d['root']!r});d=json.loads((r/'DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or (s[21]==d['start_ticks'] and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'])
out={{n:json.loads((r/n).read_text()) for n in ('PARITY.json','PARITY_K4_H120.json','PARITY_K8_H300.json','COMPLETE.json','FAILED.json') if (r/n).exists()}}
out.update(process_state=s[2] if s else 'exited',log=(r/'PROBE.log').read_text()[-6000:]);print(json.dumps(out))
''',timeout=25))
    for name,value in v.items():
        if name.endswith('.json') and not (ops.EVIDENCE/f'{tag}_{host}_{name}').exists():write(ops.EVIDENCE/f'{tag}_{host}_{name}',value)
    print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','launch','status'));p.add_argument('--host',default='gpuA');a=p.parse_args()
    register() if a.mode=='register' else launch(a.host) if a.mode=='launch' else status(a.host)
