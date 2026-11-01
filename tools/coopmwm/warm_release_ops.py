"""Bounded warm-service admission and release, adopting completed qualifications."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from . import ops
from .storage import read,write
from .protocol import sha,checked,REGISTRY
from .warm_deploy import ROOT,CPU
from .warm_bridge import GPU_ROOT


def collect():
    combined={}
    for host,tag in (('gpuA','warmmppiqual4'),('gpuR','warmmppiqual4')):
        root='/root/coopmwm20261102_'+tag
        value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,time
r=pathlib.Path({root!r});assert r.is_dir()
names=['CAUSAL_MEMORY.json','E2_COLD.json','E2_WARM.json','PARITY_K4_H120.json','PARITY_K8_H420.json','EMPTY_COLD_PARITY.json','SEARCH_GROUP2.json','SEARCH_GROUP4.json','SEARCH_GROUP6.json','FAILED.json','COMPLETE.json']
d=json.loads((r/'DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or s[21]==d['start_ticks']
receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}
if {host!r}=='gpuA':
 prior=pathlib.Path('/root/coopmwm20261102_warmmppiqual1')
 receipts['SEARCH_GROUP2.json']=json.loads((prior/'SEARCH_GROUP2.json').read_text())
print(json.dumps(dict(root=str(r),unix=time.time(),state=s[2] if s else 'exited',receipts=receipts)))''',timeout=35))
        combined[host]=value
    from .protocol import digest
    target=ops.EVIDENCE/('WARM_RELEASE_STATE_'+digest({h:v['receipts'] for h,v in combined.items()})[:16]+'.json')
    if not target.exists():write(target,combined)
    timing={str(size):dict(host=host,wall_s=value['receipts'][f'SEARCH_GROUP{size}.json']['wall_s'],
        plans_per_hour=value['receipts'][f'SEARCH_GROUP{size}.json']['plans_per_hour'],
        centre_open_loop_gap=value['receipts'][f'SEARCH_GROUP{size}.json']['result']['centre_open_loop_gap'])
        for host,value in combined.items() for size in (2,4,6) if f'SEARCH_GROUP{size}.json' in value['receipts']}
    print(json.dumps(dict(states={h:v['state'] for h,v in combined.items()},timing=timing)))
    return combined


def stage_worker(host):
    from . import mppi_search
    root=GPU_ROOT+'_w0'
    dest=ops.EVIDENCE/f'WARM_SERVICE_STAGE_{host}.json';assert not dest.exists()
    data=collect()[host]
    assert data['state'] in ('exited','Z'),'existing qualifier still runs; no duplicate GPU work'
    for name in ('PARITY_K4_H120.json','PARITY_K8_H420.json','EMPTY_COLD_PARITY.json'):
        assert data['receipts'][name]['pass_']
    cpu=read(ops.EVIDENCE/('CPU_'+CPU.rsplit('/',1)[1]+'.json'))
    files=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib
r=pathlib.Path({CPU!r});hashes=json.loads({json.dumps(cpu['source_hashes'])!r});out={{}}
for name,h in hashes.items():
 p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;out[name.removeprefix('pytools/')]=p.read_text()
print(json.dumps(out))''',timeout=35))
    # The only extra worker source is a startup gate; the tested core is immutable.
    files['tools/coopmwm/warm_service.py']=checked(ops.LOCAL/'tools/coopmwm/warm_service.py').read_text(encoding='utf-8')
    files['INPUTS.json']=ops.ssh.remote('gpuA',"import pathlib;print(pathlib.Path('/root/coopmwm20261101_planprobe3/INPUTS.json').read_text())",timeout=25)
    staged=json.loads(ops.stage(host,root,files))
    dependencies=[n for n in cpu['source_hashes'] if n.startswith('pytools/tools/wmteach/')]
    dependencies += ['pytools/tools/coopmwm/'+n+'.py' for n in ('wm','plan_wm','speed2_runtime','speed2_transfers',
        'mppi_search','mppi_runtime','policy_wm','warm_runtime','speed2_memory','fused','plans')]
    audit=json.loads(ops.ssh.remote(host,f'''import pathlib,json,hashlib
r=pathlib.Path({root!r});prior=pathlib.Path('/root/coopmwm20261102_warmmppiqual4')
names=json.loads({json.dumps(dependencies)!r});matches={{}}
for name in names:
 a=r/name;b=prior/name
 # GPU A's qualifier predates the native FIT-selector extension only.
 if name.endswith('/speed2_memory.py') and {host!r}=='gpuA':
  import ast
  def cls(p):return ast.dump(next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='WarmQuery'))
  assert cls(a)==cls(b);matches[name]='WarmQuery source-identical; fixture-selector-only extension'
 else:
  assert hashlib.sha256(a.read_bytes()).hexdigest()==hashlib.sha256(b.read_bytes()).hexdigest(),name
  matches[name]=hashlib.sha256(a.read_bytes()).hexdigest()
print(json.dumps(dict(pass_=True,matches=matches,new_sumo_sim_s=0)))''',timeout=30))
    b=read(ops.EVIDENCE/'WARM_A48_BINDING.json')
    binding=dict(hashes=staged['hashes'],qualification=dict(pass_=True,receipt=data['receipts']['PARITY_K8_H420.json'],audit=audit),
        deadline_unix=b['gpu_deadline'],memory_fraction=.80 if host=='gpuR' else .65,memory_mode='warm',startup_qualification=True,
        speed2=True,speed2_transfers=True,batch_width_by_K={'4':18 if host=='gpuR' else 34,'8':10},
        expandable_segments=host=='gpuR',search_config=mppi_search.REGISTERED)
    ops.stage(host,root,{'GPU_BINDING.json':json.dumps(binding,sort_keys=True)})
    write(dest,dict(root=root,binding=binding,audit=audit,host=host,new_sumo_sim_s=0))
    print(json.dumps(dict(host=host,root=root,qualified_dependencies=len(dependencies))))


def admit(host):
    stage=read(ops.EVIDENCE/f'WARM_SERVICE_STAGE_{host}.json');assert stage['audit']['pass_']
    line=('\nS-COOPMWM WARM SERVICE ENGINEERING ADMISSION '+host+' root'+stage['root']+
        ', exact-source auditSHA'+sha(ops.EVIDENCE/f'WARM_SERVICE_STAGE_{host}.json')+
        '; tested coreCPU6, extra source-bound startup warm batch/single gate; speed2+transfers/warm, widths34/10, one worker/GPU, '
        'original deadline retained. Pure inference admission only; B/R remain held until full release registration.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir()
if {host!r}=='gpuA':
 p=r/'WORKERS.json';v=dict(workers=['gpuA@0']);assert not p.exists()
else:
 assert json.loads((r/'WORKERS.json').read_text())['workers']==['gpuA@0']
 p=r/'WARM_WORKER_ADMISSION_0002.json';v=dict(workers=['gpuA@0','gpuR@0']);assert not p.exists()
with p.open('x') as f:json.dump(v,f)
print(json.dumps(v))''',timeout=25))
    write(ops.EVIDENCE/f'WARM_SERVICE_ADMISSION_{host}.json',dict(line=line,**value));print(json.dumps(value))


def bridge():
    target=ops.EVIDENCE/'WARM_BRIDGE_DISPATCH.json';assert not target.exists() and ops.EVIDENCE.is_dir()
    with (ops.EVIDENCE/'WARM_BRIDGE.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-m','tools.coopmwm.warm_bridge'],cwd=ops.LOCAL,stdout=log,stderr=subprocess.STDOUT)
    value=dict(pid=p.pid,unix=time.time(),purpose='new warm bridge, per-worker startup-gated, no training dispatch')
    write(target,value);print(json.dumps(value))


def worker_status(host):
    root=GPU_ROOT+'_w0'
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,time
r=pathlib.Path({root!r});assert r.is_dir()
names=['STARTUP_WARM_BATCH_SINGLE_K4_H120.json','STARTUP_WARM_BATCH_SINGLE_K8_H300.json',
 'STARTUP_PARITY_K4_H120.json','STARTUP_PARITY_K8_H300.json',
 'STARTUP_WARM_PARITY_K4_H120.json','STARTUP_WARM_PARITY_K8_H300.json','SERVICE_OWNER.json']
receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}
alive=False
if 'SERVICE_OWNER.json' in receipts:
 d=receipts['SERVICE_OWNER.json'];p=pathlib.Path('/proc')/str(d['pid']);s=(p/'stat').read_text().split() if (p/'stat').exists() else None
 alive=bool(s and s[21]==d['start_ticks'] and s[2] in ('R','S','D') and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id'])
print(json.dumps(dict(unix=time.time(),receipts=receipts,identity_verified_alive=alive)))''',timeout=30))
    from .protocol import digest
    target=ops.EVIDENCE/f'WARM_WORKER_STATE_{host}_{digest(value["receipts"])[:16]}.json'
    if not target.exists():write(target,value)
    print(json.dumps(dict(host=host,receipts=list(value['receipts']))));return value


def release():
    from .warm_report import summary,publish
    from .warm_protocol import budget
    target=ops.EVIDENCE/'WARM_RELEASE_REGISTRATION.json';assert not target.exists()
    data=collect()
    assert all(v['state'] in ('exited','Z') and v['receipts'].get('COMPLETE.json',{}).get('pass_') for v in data.values())
    assert 'SEARCH_GROUP2.json' in data['gpuA']['receipts']
    assert 'SEARCH_GROUP4.json' in data['gpuR']['receipts'] and 'SEARCH_GROUP6.json' in data['gpuA']['receipts']
    workers={h:worker_status(h) for h in ('gpuA','gpuR')}
    for host,w in workers.items():
        assert len(w['receipts'])==7 and w['identity_verified_alive'] and 'SERVICE_OWNER.json' in w['receipts']
        for name,v in w['receipts'].items():
            if name=='SERVICE_OWNER.json':continue
            assert v['pass_'] and v['max_cost_error']==0.,host+name
    cpu=read(ops.EVIDENCE/'WARM_BR_CPU8_REGISTRATION.json')
    assert read(ops.EVIDENCE/'WARM_R_CENTRE_REGISTRATION.json')['runtime']['cpu']['pass_']
    value=summary(data)
    # Use the full dimension-scaled medium search for quality, the owner-selected
    # minimum10starts; report the measured capacity, not an acceptance-based ETA.
    fastest=value['timing']['2']['plans_per_hour']
    slowest=min(t['plans_per_hour'] for t in value['timing'].values())
    value.update(pass_=True,N_exec=10,workers=workers,cpu_runtime=cpu,budget=budget(),
        capacity_note='group-size benchmarks use different hosts/states; no end-to-end execution ETA until an update closes',
        per_update_single_device_capacity_floor_hours=[10/fastest,10/slowest],
        size_choice='retain full quality-first dimension-scaled medium MPPI; Nexec10 minimum, positive acceptance unknown')
    write(ops.EVIDENCE/'WARM_RELEASE_AMENDMENT.json',value);h=sha(ops.EVIDENCE/'WARM_RELEASE_AMENDMENT.json')
    line=('\nS-COOPMWM WARM MEMORY QUALIFIED / B-R RESTART REGISTERED amendmentSHA'+h+
        ': rolling60s causal per-driver FULL-v4 posteriors/shrinkage, empty==cold, both-host warm batched/single and cold/warm full-fast EXACT0 gates; '
        'registered E2 cold/warm decision/sign/timing diagnostic reused exact labels. CPU8 learner126PASS; source-bound CPU8 B/R u0, '
        'R baseline-only same-policy centre and zero candidate scoring. Full medium dimension-scaled MPPI, K4H300 screen, '
        'K8Hfinal=min(600,max(300,D+120)), top4 posterior choice; Nexec10/update, R count/regime/size/duration matched. '
        'Quality-first/no hard deadline; actual executed throughput unknown; measured group2/4/6 timings in amendment. '
        'Both GPUs, widthsA34/10,R18/10, allocatorA65%/R80%+expandable segments, original deadline retained. A continues preserved CPU6; '
        'B/Ru4 viability pause, no final outcome access; full fixed A/B48/R24 and6473440/cap6500000 retained.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    value['release_amendment_sha256']=h
    # Authorization precedes every new B/R reservation and does not alter sources.
    ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert r.is_dir() and not (r/'STOP.json').exists()
for a in ('B','R'):assert not (r/f'WARM_{{a}}_DISPATCH.json').exists()
with (r/'AUTHORITY.md').open('a') as f:f.write({line!r})
with (r/'WARM_RELEASE.json').open('x') as f:json.dump(json.loads({json.dumps(value)!r}),f,sort_keys=True)
print('registered before B/R physics')''',timeout=40)
    publish(value,qualified=True)
    write(target,dict(line=line,release_amendment_sha256=h,new_sumo_sim_s=0));print(line.strip())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('collect','stage','admit','bridge','worker-status','release'));p.add_argument('--host',choices=('gpuA','gpuR'))
    a=p.parse_args()
    if a.mode=='stage':stage_worker(a.host)
    elif a.mode=='admit':admit(a.host)
    elif a.mode=='worker-status':worker_status(a.host)
    else:globals()[a.mode]()
