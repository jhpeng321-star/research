"""Registered four-update viability release and owned, bounded receipt observer."""
import argparse
import contextlib
from datetime import datetime,timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from . import ops
from .storage import read,write
from .protocol import REGISTRY,checked,sha,authority,digest
from .accel_protocol import budget
from .plans import SEARCH
from .plan_bridge import ROOT,GPU_ROOT

SOURCE='/home/ubuntu/coopmwm20261101_acceltrain1'
CPU='CPU_coopmwm20261101_cpu37.json'
TAG='VIABILITY2035v4'


def dependency_names(previous):
    return [name for name in previous['source_hashes'] if name.startswith('pytools/tools/wmteach/')]+['FROZEN_COSTS.json']+[
        'pytools/tools/coopmwm/'+n for n in ('plan_wm.py','plans.py','wm.py','fused.py')]


def audit_dependencies():
    current=read(ops.EVIDENCE/CPU);results={}
    for host in ('gpuA','gpuR'):
        dispatch=read(ops.EVIDENCE/f'SCALE2035_{host}_DISPATCH.json')
        previous=read(ops.EVIDENCE/dispatch['cpu']);names=dependency_names(previous)
        different=[n for n in names if previous['source_hashes'][n]!=current['source_hashes'].get(n)]
        results[host]=dict(pass_=not different,dependencies=len(names),different=different,original_cpu=dispatch['cpu'])
    value=dict(pass_=all(v['pass_'] for v in results.values()),cpu=CPU,hosts=results,new_sumo_sim_s=0)
    write(ops.EVIDENCE/f'{TAG}_DEPENDENCY_AUDIT.json',value);print(json.dumps(value))
    assert value['pass_'],'forecast dependencies changed; qualification remains held'


def register():
    cpu=read(ops.EVIDENCE/CPU);assert cpu['pass_'] and not cpu['skips'] and cpu['sim_s']==0
    value=dict(cpu=CPU,cpu_sha256=sha(ops.EVIDENCE/CPU),budget=budget(),search=SEARCH,N_exec=10,viability_updates=4,
        order='A/B primary u1..4 in parallel, then matched R u1..4, durable pause; no sealed outcomes',
        subset_precharge_max_sim_s=dict(train=345600,checks=43200,discovery=57600,mechanics=3600),
        including_superseded_max_sim_s=1261640,full_allocation_sim_s=budget()['allocated'],
        service_size='CAPACITY2035 admission: fastest completed qualified measurement, one wide34/10 worker/GPU; unfinished4/8 trials after viability',
        quota_basis=dict(measured_plans_per_hour_gpuA=5.6605,measured_plans_per_hour_gpuR=5.5123,
            minimum_executed_starts=10,capacity_lower_bound_hours_per_update=10/(5.6605+5.5123),
            acceptance='unknown; >=10 executed may require more than one hour; no fabricated acceptance ETA'),
        gpu_quote=dict(original_deadline_unix=read(ops.EVIDENCE/'DUAL_1755_DISPATCH.json')['unix']+86400,
            concurrency_estimated_hours_per_host=[1,3],viability_service_estimated_hours_per_host=[4,12],
            deadline_not_extended=True,actual_worker_time_and_throughput_reported=True),
        infrastructure='opaque verified archive; CPU37 lossless hashed gzip results; committed-plan continuation atom preserved even on eligible10s ticks; R query rate estimated only over ex-ante eligible uncommitted windows; persistent GPU caps applied after inherited bootstrap; immutable new training source, old smoke reused',
        reporting='each UPDATE/VIABILITY/CHECK/E2 immediately; pertype/regime/rank/new-family SUMO advantage/CI; after u4 pause',
        no_new_sumo_sim_s=0)
    write(ops.EVIDENCE/f'{TAG}_AMENDMENT.json',value);h=sha(ops.EVIDENCE/f'{TAG}_AMENDMENT.json')
    line=(f'\nS-COOPMWM OWNER20:35 VIABILITY NUMERIC RESTART REGISTERED BEFORE ACCEL TRAINING OUTCOMES: '
        f'amendmentSHA{h}; CPU{CPU}/SHA{value["cpu_sha256"]}; '
        'A/B/R allMAPPOu0/common1sacceleration residuals, SUMO-only J, safety/ownership unchanged; '
        'medium32x3/elite8/K4H120/top8K8H300/Thompson unchanged. '
        'Nexec=10 executed plan-starts minimum per B update from measured5.6605+5.5123queries/hour; '
        'acceptanceunknown/capacity floor.895h perupdate, no fabricated execution ETA. '
        'Live free-slot sampling p.95/densepriority, bothGPUs saturated beyondminimum; R count/regime/duration matched. '
        'CAPACITY2035 amendment admits fastest completed source-qualified timing: wide34/10 one worker/GPU now; '
        'fixed4 uncompleted4/8 trials deferred until after viability, completed1/2 never rerun; '
        'once/persistentworker K4H120/K8H300 parity; hot-add only admitted frozen-source same-search workers. '
        'A/B primaryu1..4 then Ru1..4/checks; durablePAUSE for MAIN/owner viability review, no sealed-test access. '
        'First12executedB windows t<=600 sampled RNG20261101 for exact complete H60/120/300/600 twins, '
        'asynchronous2nice19CPUworkers/no training gate; failed twins retained/no retries; unchanged240000discovery reserve. '
        'Viability max450000 including3600already-reserved smoke; with811640superseded max1261640sim-s. '
        f'Full registeredallocation{budget()["allocated"]}/6000000/free{budget()["unallocated"]} unchanged; '
        '96sealedworlds and fixedfullu24 checkpoint/test rules protected; seed2 remainsdropped. '
        'GPU quote concurrency1..3h/host +viability4..12h/host estimates, original24hGPUdeadline notreset; '
        'actual times published. Owner-approved v2 opaque tar/hashmatched archive then literal root removal only; '
        'CPU37 lossless digest-checked gzip 1s tapes/newimmutable training namespace; '
        'CPU regressions fix continuation plan-route atom on eligible10s ticks and R sampling denominator '
        '(eligible uncommitted decisions only, not all committed1s steps); R exhausted matched tickets routeown/p0, '
        'with no fictitious empty-proposal atoms, before ANY acceleration update. '
        'Persistent-worker memory fraction min(.65,.80/N) applied AFTER inherited bootstrap overrides; '
        'original CPU33 scaling retains measured actual peaks and inherited8GiB cap, not a promised customfraction; '
        'final persistent workers qualify under their actual admitted cap. '
        'Supersedes prior release metadata only; all original probes/smoke/charges reused. No outcome-dependent tuning.\n')
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line)
    write(ops.EVIDENCE/f'{TAG}_REGISTRATION.json',dict(line=line,amendment_sha256=h));print(line.strip())


def bind():
    cpu=read(ops.EVIDENCE/CPU);amend=read(ops.EVIDENCE/f'{TAG}_AMENDMENT.json')
    assert amend['cpu_sha256']==sha(ops.EVIDENCE/CPU)
    registration=read(ops.EVIDENCE/f'{TAG}_REGISTRATION.json')
    text=checked(REGISTRY).read_text(encoding='utf-8');assert registration['amendment_sha256'] in text
    smoke=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({ROOT!r});assert not (r/'STOP.json').exists();assert not (r/'BINDING_TRAINING.json').exists()
v=json.loads((r/'SMOKE.json').read_text());assert v['pass_'];print(json.dumps(v))
''',timeout=20))
    assert read(ops.EVIDENCE/'ARCHIVE2035_COMPLETE.json')['source_removed']
    admission=read(ops.EVIDENCE/'CAPACITY2035_ADMISSION.json')
    scale=admission['hosts']
    assert all(v['pass_'] and v['best']['source_bound_full_search_parity'] and v['qualification']['pass_'] for v in scale.values())
    staged=json.loads(ops.stage_frozen('cpuQ',SOURCE,cpu))
    hashes={SOURCE+'/'+name:h for name,h in staged['hashes'].items()}
    slots=[];qual={}
    for host,v in scale.items():
        n=v['best']['workers']
        # Qualification and final service must share the forecast/search source.
        dispatch=read(ops.EVIDENCE/f'SCALE2035_{host}_DISPATCH.json')
        previous=read(ops.EVIDENCE/dispatch['cpu'])
        for name in dependency_names(previous):
            assert previous['source_hashes'][name]==cpu['source_hashes'][name]
        gpu_source=json.loads(ops.stage_frozen(host,GPU_ROOT,cpu))
        inputs=json.dumps(read(ops.EVIDENCE/'VOLUME_INPUTS_1854.json'))
        frozen=ops.ssh.remote(host,f'import pathlib;print(pathlib.Path({(GPU_ROOT+"/FROZEN_COSTS.json")!r}).read_text())',timeout=20).strip()
        for index in range(n):
            slot=f'{host}@{index}';slots.append(slot);worker=GPU_ROOT+f'_w{index}'
            binding=dict(hashes={GPU_ROOT+'/'+k:h for k,h in gpu_source['hashes'].items()},qualification=v['best'],
                deadline_unix=amend['gpu_quote']['original_deadline_unix'],memory_fraction=min(.65,.80/n),
                batch_widths=v['batch_widths'],startup_qualification=True,search=SEARCH,slot=slot)
            ops.stage(host,worker,{'GPU_BINDING.json':json.dumps(binding),'FROZEN_COSTS.json':frozen,'INPUTS.json':inputs})
            qual[slot]=v['best']
    throughput=sum(v['best']['plans_per_hour'] for v in scale.values())
    b=dict(runtime='/home/ubuntu/e3final',checkpoint='/home/ubuntu/mappospeed/production2/policy_00169.pt',
        authority=authority(text),authority_file='AUTHORITY_TRAINING.md',budget=budget(),search=SEARCH,
        cpu=cpu,workers=12,load_max=20,hashes=hashes,amendment_sha256=registration['amendment_sha256'],
        N_exec=10,viability_updates=4,eta_primary_hours=dict(capacity_lower_bound=240/throughput,acceptance_unknown=True),
        accel_bounds=[-3.,2.6],gpu_deadline=amend['gpu_quote']['original_deadline_unix'],smoke_only=False,gpu_qualification=qual)
    out=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.storage import write
b=json.loads({json.dumps(b)!r})
with (r/'AUTHORITY_TRAINING.md').open('x') as f:f.write({ascii(text)})
write(r/'WORKER_ADMISSION_000.json',dict(workers={slots!r},search=b['search']))
for p in (r/'MAPPO_DEV.json',r/'MAPPO_CHECK.json',pathlib.Path(b['checkpoint']),pathlib.Path('/home/ubuntu/e3final/runtime/SOURCE_MANIFEST.json')):b['hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
write(r/'BINDING_TRAINING.json',b)
from tools.coopmwm.accel_runner import verify
verify(r)
print(json.dumps(dict(binding_sha256=hashlib.sha256((r/'BINDING_TRAINING.json').read_bytes()).hexdigest(),slots={slots!r},source={SOURCE!r},new_sumo_sim_s=0)))
''',timeout=30))
    out.update(throughput_plans_per_hour=throughput,scale=scale,smoke=smoke)
    write(ops.EVIDENCE/f'{TAG}_BINDING.json',out);print(json.dumps(out))


def launch():
    assert (ops.EVIDENCE/f'{TAG}_BINDING.json').exists()
    assert not (ops.EVIDENCE/f'{TAG}_TRAIN_DISPATCH.json').exists()
    d=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,os,subprocess,time
r=pathlib.Path({ROOT!r});assert r.is_dir();sys.path[:0]=[{SOURCE!r}+'/pytools','/home/ubuntu/e3final/runtime/src','/home/ubuntu/e3final/runtime']
from tools.coopmwm.storage import write
from tools.coopmwm.accel_runner import verify
verify(r);assert not (r/'TRAIN_DISPATCH.json').exists()
import sumo
home=str(pathlib.Path(sumo.__file__).parent)
env=dict(os.environ,PYTHONPATH={SOURCE!r}+'/pytools:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SUMO_HOME=home,PATH=home+'/bin:'+str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''))
with (r/'TRAIN.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.accel_runner',str(r)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
d=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),root=str(r),source={SOURCE!r})
write(r/'TRAIN_DISPATCH.json',d);print(json.dumps(d))
''',timeout=30))
    write(ops.EVIDENCE/f'{TAG}_TRAIN_DISPATCH.json',d)
    with (ops.EVIDENCE/'ACCEL_BRIDGE.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u','-m','tools.coopmwm.plan_bridge'],cwd=ops.LOCAL,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    write(ops.EVIDENCE/f'{TAG}_BRIDGE_DISPATCH.json',dict(pid=p.pid,unix=time.time(),root=ROOT))
    print(json.dumps(d))


def summarize(name,value):
    from .publish import summarize as original
    if name.startswith('ACCEL_CREDIT_CHECK_'):return original('CREDIT_CHECK.json',value)
    if name.startswith('ACCEL_CREDIT_WINDOW_'):return original(name.removeprefix('ACCEL_'),value)
    if '_VIABILITY_' in name:
        return ['- SUMO-realized critic-baselined proposal advantage; selection-conditional, not exact causal grading.',
            '- '+json.dumps(value,separators=(',',':'))]
    return original(name,value)


def collect_diagnostics():
    """Only registered executed-plan diagnostic receipts, never final episodes."""
    ops.ssh.remote('cpuQ',f'''import pathlib,json,sys
r=pathlib.Path({ROOT!r});source=pathlib.Path({SOURCE!r})
if source.is_dir():
 sys.path.insert(0,str(source/'pytools'))
 from tools.coopmwm.storage import read,write
 from tools.coopmwm.accel_diagnostic import agreement
 records=[]
 for p in sorted(r.glob('ACCEL_EXACT_PLAN_*.json')):
  if p.name.endswith('_FAILED.json'):continue
  exact=read(p);request=exact.get('wm_request')
  if not request:continue
  response=r/'responses'/(request['state_id']+'.json')
  if not response.exists():continue
  wm=read(response)
  assert wm['record_sha256']==__import__('tools.coopmwm.protocol',fromlist=['digest']).digest(request)
  record=dict(world=exact['world'],time_s=exact['time_s'],wm=wm['credits'],exact=exact['credits'],noise=wm['noise'])
  records.append(record);target=r/p.name.replace('ACCEL_EXACT_PLAN_','ACCEL_CREDIT_WINDOW_')
  if not target.exists():write(target,record)
 if records:
  target=r/f'ACCEL_CREDIT_CHECK_{{len(records):03}}.json'
  if not target.exists():write(target,agreement(records))
print('diagnostic receipts only')
''',timeout=40)


def publish_receipts(receipts,folder='accel_runtime'):
    from .publish import commit
    fresh=[]
    for name,value in receipts.items():
        marker=ops.EVIDENCE/'accel_published'/(name+'.receipt.json')
        if not marker.exists():fresh.append((name,value,marker))
    if not fresh:return
    stamp=datetime.now(timezone.utc).isoformat();lines=[f'\n## Acceleration viability — {stamp}'];paths=[]
    for name,value,marker in fresh:
        target=ops.EVIDENCE/folder/name
        if not target.exists():write(target,value)
        lines.append(f'\n### {name}');lines.extend(summarize(name,value))
        write(marker,dict(published_utc=stamp,source=str(target.relative_to(ops.LOCAL))))
        paths += [str(target.relative_to(ops.LOCAL)).replace('\\','/'),str(marker.relative_to(ops.LOCAL)).replace('\\','/')]
    report=ops.LOCAL/'docs/paper_results/COOPMWM.md'
    with report.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths.append('docs/paper_results/COOPMWM.md')
    print(commit(paths,'coopmwm: publish acceleration '+','.join(n.removesuffix('.json') for n,_,_ in fresh)),flush=True)


def once():
    from . import recovery2035,accel_deploy,plan_scale
    complete=(ops.EVIDENCE/'CAPACITY2035_ADMISSION.json').exists()
    if not complete and not (ops.EVIDENCE/f'{TAG}_BINDING.json').exists():
        complete=True
        for host in ('gpuA','gpuR'):
            with contextlib.redirect_stdout(io.StringIO()):v=plan_scale.status(host)
            publish_receipts({f'SCALE_{host}_{n}':r for n,r in v['receipts'].items()},'accel_scale')
            if 'FAILED.json' in v['receipts']:return 'STOP_SCALE'
            complete=complete and 'COMPLETE.json' in v['receipts']
    if not (ops.EVIDENCE/'ARCHIVE2035_VERIFIED.json').exists():return 'WAIT_ARCHIVE'
    if not (ops.EVIDENCE/'ARCHIVE2035_COMPLETE.json').exists():recovery2035.remove_verified_source()
    publish_receipts({'ARCHIVE2035_COMPLETE.json':read(ops.EVIDENCE/'ARCHIVE2035_COMPLETE.json')})
    collect_diagnostics()
    with contextlib.redirect_stdout(io.StringIO()):state=accel_deploy.status()
    publish_receipts(state['receipts'])
    if 'STOP.json' in state['receipts']:return 'STOP'
    if 'VIABILITY_PAUSED.json' in state['receipts']:
        if state['pending_WM']:return 'DRAIN_DIAGNOSTICS'
        from .capacity2035 import deferred_after_pause
        return deferred_after_pause(CPU)
    if 'SMOKE.json' not in state['receipts']:return 'WAIT_EXISTING_SMOKE'
    if not (ops.EVIDENCE/f'{TAG}_BINDING.json').exists():
        if not complete:return 'WAIT_SCALE'
        publish_receipts({'CAPACITY2035_ADMISSION.json':read(ops.EVIDENCE/'CAPACITY2035_ADMISSION.json')})
        bind()
        publish_receipts({f'{TAG}_BINDING.json':read(ops.EVIDENCE/f'{TAG}_BINDING.json')})
    if not (ops.EVIDENCE/f'{TAG}_TRAIN_DISPATCH.json').exists():launch()
    publish_receipts({f'{TAG}_{kind}_DISPATCH.json':read(ops.EVIDENCE/f'{TAG}_{kind}_DISPATCH.json') for kind in ('TRAIN','BRIDGE')})
    return 'RUNNING'


def watch():
    import msvcrt
    claim=(ops.EVIDENCE/f'{TAG}.claim').open('a+b');claim.seek(0)
    if not claim.read(1):claim.write(b'1');claim.flush()
    claim.seek(0);msvcrt.locking(claim.fileno(),msvcrt.LK_NBLCK,1)
    deadline=read(ops.EVIDENCE/f'{TAG}_AMENDMENT.json')['gpu_quote']['original_deadline_unix']
    last=None
    while time.time()<deadline:
        # A failure is durable. Never reattempt a partially completed bind/dispatch.
        try:status=once()
        except Exception as error:
            write(ops.EVIDENCE/f'{TAG}_WATCH_FAILED.json',dict(unix=time.time(),error=repr(error),automatic_retry=False))
            raise
        if status!=last:print(json.dumps(dict(unix=time.time(),status=status)),flush=True);last=status
        if status in ('STOP','STOP_SCALE','PAUSED'):return
        time.sleep(30)
    write(ops.EVIDENCE/f'{TAG}_WATCH_DEADLINE.json',dict(unix=time.time(),status=last,deadline_reached=True))


def start_watch():
    target=ops.EVIDENCE/f'{TAG}_WATCH_DISPATCH.json';assert not target.exists()
    assert (ops.EVIDENCE/f'{TAG}_REGISTRATION.json').exists()
    with (ops.EVIDENCE/f'{TAG}_WATCH.log').open('x',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u','-m','tools.coopmwm.viability2035','watch'],cwd=ops.LOCAL,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    value=dict(pid=p.pid,unix=time.time(),poll_s=30,original_gpu_deadline=True,
        mode='one-use registered archive/smoke adoption/scale qualification/training release; publish+commit; fixedu4pause')
    write(target,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','audit-dependencies','bind','launch','once','watch','start-watch'));a=p.parse_args()
    {'register':register,'audit-dependencies':audit_dependencies,'bind':bind,'launch':launch,'once':once,'watch':watch,'start-watch':start_watch}[a.mode]()
