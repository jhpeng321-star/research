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


def launch_probe():
    from .storage import read
    root='/root/coopmwm20261101_actionprobe1'
    cpu=read(ops.EVIDENCE/'CPU_coopmwm20261101_cpu21.json')
    ops.stage_frozen('gpuA',root,cpu)
    ops.stage('gpuA',root,{'tools/coopmwm/action_probe.py':(ops.LOCAL/'tools/coopmwm/action_probe.py').read_text(),
        'INPUT.json':json.dumps(read(ops.EVIDENCE/'ACTION_PROBE_INPUT_01.json'))})
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
    write(ops.EVIDENCE/'ACTION_PROBE_DISPATCH_01.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['hold','fetch','probe']);a=p.parse_args()
    if a.mode=='hold':hold()
    elif a.mode=='fetch':fetch()
    else:launch_probe()
