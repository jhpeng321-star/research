"""Owner 18:34 action-response investigation; no physical dispatch."""
import argparse
import json
from . import ops, deploy
from .storage import write


def hold():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import os,signal,pathlib,json,time
r=pathlib.Path({deploy.ROOT!r});assert r.is_dir()
d=json.loads((r/'TRAIN_DISPATCH_1742.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert (p/'stat').read_text().split()[21]==d['start_ticks']
assert 'tools.coopmwm.runner' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
receipt=r/'HOLD_1834.json'
if receipt.exists():
 v=json.loads(receipt.read_text());assert (p/'stat').read_text().split()[2]=='T'
else:
 os.kill(d['pid'],signal.SIGSTOP)
 charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
 incomplete=[x['id'] for x in charges if not all((r/'jobs'/x['id']/f).exists() for f in ('RESULT.json','CLOSE.json'))]
 assert not incomplete,'supervisor held; inspect outstanding owned jobs before proceeding'
 assert not (r/'SAMPLING_1755.json').exists(),'inspect any post-u1 outcomes'
 v=dict(owner=d,unix=time.time(),reason='OWNER18:34 WM action-response wiring investigation',
  arms_held=['B','R'],shared_supervisor_paused=True,A_update=1,B_update=1,
  charged_sim_s=sum(x['sim_s'] for x in charges),closed_jobs=len(charges),new_sumo_sim_s=0)
 with receipt.open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/'HOLD_1834.json',value)
    print(json.dumps(value))


def fetch():
    values=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({deploy.ROOT!r});out=[]
windows=json.loads((r/'CALIBRATION_WINDOWS.json').read_text())['windows'][:2]
for w in windows:
 found=[]
 for p in (r/'requests').glob('*.json'):
  v=json.loads(p.read_text())
  if v.get('world_id')==w['world']['seed'] and v.get('time_s')==w['time_s'] and v.get('mode')!='proposal':found.append(v)
 assert len(found)==1
 out.append(found[0])
print(json.dumps(out))
''',timeout=25))
    for i,v in enumerate(values):
        write(ops.EVIDENCE/f'ACTION_PROBE_INPUT_{i:02}.json',v)
    print(json.dumps([dict(world=v['world_id'],time_s=v['time_s'],group=v['group'],
        members=[r for r in v['rows'] if r['vid'] in v['group']],
        targets={k:v['targets'].get(k) for k in v['group']}) for v in values]))


def launch_probe(window=1):
    from .storage import read
    root=f'/root/coopmwm20261101_actionprobe2_{window:02}'
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261101_cpu21.json')
    ops.stage_frozen('gpuA',root,cpu)
    ops.stage('gpuA',root,{'tools/coopmwm/action_probe.py':(ops.LOCAL/'tools/coopmwm/action_probe.py').read_text(),
        'INPUT.json':json.dumps(read(ops.EVIDENCE/f'ACTION_PROBE_INPUT_{window:02}.json'))})
    value=json.loads(ops.ssh.remote('gpuA',f'''import os,pathlib,json,subprocess,sys
r=pathlib.Path({root!r});assert r.is_dir();assert not (r/'DISPATCH.json').exists()
print_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
assert int(print_gpu.strip())>12000
env=dict(os.environ,PYTHONPATH=str(r/'pytools'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (r/'PROBE.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19',sys.executable,'-B','-u','-m','tools.coopmwm.action_probe',str(r)],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
v=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),host='gpuA',new_sumo_sim_s=0)
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/f'ACTION_PROBE2_DISPATCH_{window:02}.json',value);print(json.dumps(value))


def collect_probe(window):
    from .storage import read
    d=read(ops.EVIDENCE/f'ACTION_PROBE2_DISPATCH_{window:02}.json')
    value=json.loads(ops.ssh.remote(d['host'],f'''import pathlib,json
r=pathlib.Path({d['root']!r})
if not (r/'COMPLETE.json').exists():
 print(json.dumps(dict(complete=False,log=(r/'PROBE.log').read_text())))
else:
 print(json.dumps(dict(complete=True,credit=json.loads((r/'CREDIT.json').read_text()),options=json.loads((r/'OPTIONS.json').read_text()))))
''',timeout=30))
    if value['complete']:write(ops.EVIDENCE/f'ACTION_PROBE2_RESULT_{window:02}.json',value)
    print(json.dumps(dict(complete=value['complete'],log=value.get('log'))))


def deltas():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,math,sys
r=pathlib.Path({deploy.ROOT!r});sys.path.insert(0,'/home/ubuntu/coopmwm20261101_v3/pytools')
from tools.coopfree.policy import requests
windows=json.loads((r/'CALIBRATION_WINDOWS.json').read_text())['windows'][:2];out=[]
for w in windows:
 matches=[]
 for p in (r/'jobs').glob('s0_u001_*/RESULT.json'):
  e=json.loads(p.read_text())
  if e['world']['seed']!=w['world']['seed'] or e['arm']!='B':continue
  matches.extend(s for s in e['steps'] if s['time_s']==w['time_s'])
 assert len(matches)==1
 s=matches[0];sample=s['sample'];members=[]
 req=requests(s['targets'],s['rows'],zone=None)
 for vid in s['group']:
  i=sample['context']['members'].index(vid);a=sample['context']['anchors'][i];t=s['targets'][vid]
  members.append(dict(vid=vid,baseline_speed=a[0],baseline_tau=math.exp(a[1]),target=t,
   delta_speed=t['speed']-a[0],delta_tau=t['tau']-math.exp(a[1]),action=sample['action'][i],request=req.get(vid)))
 out.append(dict(world=w['world']['seed'],time_s=w['time_s'],members=members))
print(json.dumps(dict(windows=out,new_sumo_sim_s=0)))
''',timeout=30))
    write(ops.EVIDENCE/'ACTION_DELTAS_1838.json',value);print(json.dumps(value))


def wm_cpu():
    root='/root/coopmwm20261101_wmcpu2'
    staged=json.loads(ops.stage('gpuA',root))
    out=ops.ssh.remote('gpuA',f'''import os,sys,json,unittest,time
from unittest.mock import patch
os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt')
sys.path[:0]=[{root!r}+'/pytools','/root/rbeval20261031/phys_code_22cc32f65a80/src','/root/rbeval20261031/phys_code_22cc32f65a80']
start=time.monotonic()
with patch('subprocess.Popen',side_effect=AssertionError('physics forbidden')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.coopmwm.test_wm_cpu'))
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),new_sumo_sim_s=0,wall_s=time.monotonic()-start,root={root!r},host='gpuA',CUDA_VISIBLE_DEVICES='')))
''',timeout=60)
    print(out)
    v=json.loads(out.strip().splitlines()[-1]);v['source_hashes']=staged['hashes']
    write(ops.EVIDENCE/'CPU_WM_ACTION_RESPONSE2_1838.json',v)
    assert v['pass_'] and not v['skips']


def report():
    from .storage import read
    out=[]
    for i in (0,1):
        v=read(ops.EVIDENCE/f'ACTION_PROBE2_RESULT_{i:02}.json');modes={}
        for mode in ('credit','options'):
            item=v[mode];tr=item['traces'];baseline='revert:0' if mode=='credit' else 'MAPPO'
            by={(t['stage'],t['action'],t['vid'],t['tick']):t for t in tr}
            differences={}
            for a in item['actions']:
                differences[a['id']]={}
                for vid in read(ops.EVIDENCE/f'ACTION_PROBE_INPUT_{i:02}.json')['group']:
                    values={}
                    for stage,fields in [('safety',('bounded_raw','safe')),('trajectory',('next_s_m','next_v_mps','accel_mps2'))]:
                        for field in fields:
                            values[field]=max(abs(t[field]-by[stage,baseline,vid,t['tick']][field]) for t in tr
                                if t['stage']==stage and t['action']==a['id'] and t['vid']==vid)
                    differences[a['id']][vid]=values
            modes[mode]=dict(trajectory_max_abs_vs_baseline=differences,
                H60_one_draw_cost={a:fs[0]['team_delay'] for a,fs in item['result']['futures']['60'].items()},
                first_tick_safety=[t for t in tr if t['stage']=='safety' and t['tick']==0])
        out.append(dict(window=i,modes=modes))
    value=dict(verdict='NO_ACTION_WIRING_BUG: safety/plant-cap saturation of original positive requests; inherited proposals sensitive',
        windows=out,regression=read(ops.EVIDENCE/'CPU_WM_ACTION_RESPONSE2_1838.json'),
        exact_Sumo_reused=True,new_sumo_sim_s=0,scope='same original windows; one indexed H60 WM future/variant; first12s traces, no efficacy selection')
    write(ops.EVIDENCE/'DIAGNOSIS_1838.json',value)
    print(json.dumps(dict(verdict=value['verdict'],windows=[dict(window=x['window'],modes={m:v['trajectory_max_abs_vs_baseline'] for m,v in x['modes'].items()}) for x in out])))


def amendment():
    from .storage import read
    from .protocol import digest,sha,budget
    import time
    d=read(ops.EVIDENCE/'DIAGNOSIS_1838.json')
    assert d['regression']['pass_'] and not d['regression']['skips']
    sampling=read(ops.EVIDENCE/'SAMPLING_1755.json')
    v=dict(unix=time.time(),owner='18:34/18:38',verdict=d['verdict'],
        diagnosis_sha256=sha(ops.EVIDENCE/'DIAGNOSIS_1838.json'),
        numeric_sampling_payload_sha256=digest(sampling),probabilities=sampling['probabilities'],
        action_response_CPU_sha256=sha(ops.EVIDENCE/'CPU_WM_ACTION_RESPONSE2_1838.json'),
        change='release same owned supervisor from engineering hold; no training/inference code, weights, optimizer or RNG change; bind preregistered timing-derived numeric rates',
        retained='u1 A/B checkpoints and all70 closed jobs/83200sim-s; original exact twins/K8 predictions; GPU services and queue',
        budget=budget(),new_sumo_sim_s=0)
    write(ops.EVIDENCE/'ENGINEERING_AMENDMENT_1838.json',v);print(json.dumps(v))


def release():
    from .storage import read
    from .protocol import REGISTRY,checked,digest
    text=checked(REGISTRY).read_text(encoding='utf-8')
    a=read(ops.EVIDENCE/'ENGINEERING_AMENDMENT_1838.json')
    assert any('S-COOPMWM 18:38 ENGINEERING RELEASE REGISTERED' in s and a['diagnosis_sha256'] in s for s in text.splitlines())
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,os,signal,time
r=pathlib.Path({deploy.ROOT!r});assert r.is_dir()
d=json.loads((r/'TRAIN_DISPATCH_1742.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert (p/'stat').read_text().split()[21]==d['start_ticks']
assert 'tools.coopmwm.runner' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
assert (p/'stat').read_text().split()[2]=='T'
assert not (r/'STOP.json').exists()
charges=[json.loads(s) for s in (r/'CHARGES.jsonl').read_text().splitlines()]
assert len(charges)==70 and sum(c['sim_s'] for c in charges)==83200
assert all((r/'jobs'/c['id']/'CLOSE.json').exists() and (r/'jobs'/c['id']/'RESULT.json').exists() for c in charges)
s=json.loads((r/'SAMPLING_1755.json').read_text())
assert s['registration_sha256_of_payload']=={a['numeric_sampling_payload_sha256']!r}
assert not (r/'RELEASE_1838.json').exists(),'one-use release; inspect existing receipt'
with (r/'ENGINEERING_AUTHORITY_1838.md').open('x') as f:f.write({ascii(text)})
v=dict(unix=time.time(),owner=d,amendment=json.loads({json.dumps(a)!r}),charged_sim_s=83200,closed_jobs=70,new_sumo_sim_s=0)
with (r/'RELEASE_INTENT_1838.json').open('x') as f:json.dump(v,f,sort_keys=True)
os.kill(d['pid'],signal.SIGCONT)
v['state_after_signal']=(p/'stat').read_text().split()[2]
with (r/'RELEASE_1838.json').open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''',timeout=30))
    write(ops.EVIDENCE/'RELEASE_1838.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['hold','fetch','probe','deltas','collect','wm-cpu','report','amendment','release']);p.add_argument('--window',type=int,choices=(0,1),default=1);a=p.parse_args()
    if a.mode=='hold':hold()
    elif a.mode=='fetch':fetch()
    elif a.mode=='deltas':deltas()
    elif a.mode=='collect':collect_probe(a.window)
    elif a.mode=='wm-cpu':wm_cpu()
    elif a.mode=='report':report()
    elif a.mode=='amendment':amendment()
    elif a.mode=='release':release()
    else:launch_probe(a.window)
