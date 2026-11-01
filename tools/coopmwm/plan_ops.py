"""Explicit once-only acceleration planner engineering probes; no physics dispatch."""
import argparse
import json
from . import ops
from .storage import read,write


TAG='PLAN_PROBE2'
ROOT='/root/coopmwm20261101_planprobe2'


def register():
    from .protocol import REGISTRY,checked
    line=('\nS-COOPMWM ACCELERATION PLAN GPU QUALIFICATION2 REGISTERED: allarmsheld; '
        'probe7two-windowK8 passes both GPUs but4-windowK4 fails3.506veh-s; not admitted. '
        'Qualify vector acceleration profile decoder at inheritedWMTEACH pre-safety bridge, '
        'samefirst registered observable TRAIN geometry; noSUMO, syntheticzero policyresidual/rawseed anchors for engineering only. '
        'Probe1 compact-vs-full parityfailed318.494veh-s (actionresponse nonzero); retained/rejected. '
        'Isolate batchshape by keeping full1600 forward AND fixed4candidatewidth; singlecandidate is padded with3private duplicates. '
        'Compare wholeK8 eachcandidatealone/padded4 vs fourdifferentcandidates/K8 at full1600forward, '
        'maxpairedcosterror<=.001veh-s and nonzero large-action sensitivity; keep full source/controller virginroster. '
        'Then time registered medium32x3CEM K4H120/top8K8H300 search on BOTH GPUs,65%memory. '
        'No perquery canaries after workerqualification; no new physics/training/reward/outcome selection. '
        'Allfailedreceipts retained; originalGPUdeadline.\n')
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
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.plan_probe',str(r)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
v=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),host={host!r},unix=time.time(),new_sumo_sim_s=0)
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/f'{TAG}_{host}_DISPATCH.json',v);print(json.dumps(v))


def status(host):
    tag=next(t for t in ('PLAN_PROBE2','PLAN_PROBE1') if (ops.EVIDENCE/f'{t}_{host}_DISPATCH.json').exists())
    d=read(ops.EVIDENCE/f'{tag}_{host}_DISPATCH.json')
    v=json.loads(ops.ssh.remote(host,f'''import pathlib,json
r=pathlib.Path({d['root']!r});d=json.loads((r/'DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or (s[21]==d['start_ticks'] and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'])
out={{n:json.loads((r/n).read_text()) for n in ('PARITY.json','COMPLETE.json','FAILED.json') if (r/n).exists()}}
out.update(process_state=s[2] if s else 'exited',log=(r/'PROBE.log').read_text()[-6000:]);print(json.dumps(out))
''',timeout=25))
    for name,value in v.items():
        if name.endswith('.json') and not (ops.EVIDENCE/f'{tag}_{host}_{name}').exists():write(ops.EVIDENCE/f'{tag}_{host}_{name}',value)
    print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','launch','status'));p.add_argument('--host',default='gpuA');a=p.parse_args()
    register() if a.mode=='register' else launch(a.host) if a.mode=='launch' else status(a.host)
