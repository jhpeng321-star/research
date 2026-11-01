"""One development-only frozen-baseline orchestration replay; never final worlds."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from tools.rbint.remote import ssh
from tools.rb2int.audit import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status', 'verify'))
    parser.add_argument('--host', choices=('gpuA', 'gpuR'), required=True)
    parser.add_argument('--model', choices=('WM2', 'FULL-v4'), required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'/root/autodl-tmp/c4/rb2int20261101/phys_code_[a-f0-9]{12}', args.control):
        raise ValueError('immutable INT control stage required')
    local = Path(args.receipt)
    if not local.parent.is_dir() or local.exists():
        raise ValueError('exclusive local bridge operation receipt required')
    audit_name = 'BASELINE_WM2_SOURCE_AUDIT.json' if args.model == 'WM2' else 'BASELINE_V4_SOURCE_AUDIT.json'
    raw = (ROOT/'docs/rb2int'/audit_name).read_bytes()
    audit = json.loads(raw)
    if audit['host'] != args.host or not audit['passed']:
        raise ValueError('preserve the original verified baseline host/source')
    model_name = audit['original_identity']['model']
    original_task = str(Path(audit['identity_path']).parent).replace('\\', '/')+'/w969420_k0.json'
    name = 'wm2' if args.model == 'WM2' else 'v4'
    root = args.control.rsplit('/', 1)[0]+'/bridge_'+args.control.rsplit('_', 1)[1]+'_'+name
    common = f'''
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
root=Path({root!r});control=Path({args.control!r})
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
 with path.open('x') as f: json.dump(data,f,indent=1)
def alive(process):
 try:
  f=Path('/proc/'+str(process['pid'])+'/stat').read_text().split()
  return f[2]!='Z' and f[21]==process['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']
 except FileNotFoundError: return False
'''
    if args.action == 'launch':
        code = f'''
assert (control/'INT_STAGE.json').is_file() and not root.exists(), 'preserve old bridge claim/job'
assert shutil.disk_usage(control.parent).free>512*2**20 and os.getloadavg()[0]<os.cpu_count()/2
cg=Path('/sys/fs/cgroup');maximum=cg/'cpu.max';stat=cg/'cpu.stat'
if maximum.is_file() and maximum.read_text().split()[0]!='max':
 quota,period=map(int,maximum.read_text().split());capacity=quota/period
 def usage(): return int(dict(line.split() for line in stat.read_text().splitlines())['usage_usec'])
 begin=time.monotonic();before=usage();time.sleep(1);used=(usage()-before)/1e6/(time.monotonic()-begin)
 assert used+1<.8*capacity, 'wait for baseline-bridge CPU quota'
memory=cg/'memory.max';current=cg/'memory.current'
if memory.is_file() and memory.read_text().strip()!='max': assert int(memory.read_text())-int(current.read_text())>8*2**30, 'wait for baseline-bridge memory'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
assert int(gpu.split(',')[0])>13*1024, '5GiB bridge plus8GiB shared GPU headroom'
root.mkdir();save(root/'CLAIM_ONCE.json',dict(owner='S-RB2-INT',model={args.model!r},worlds=[969420],future_indices=[0],validation_only=True,automatic_retry=False,audit_sha256={hashlib.sha256(raw).hexdigest()!r},gpu=gpu.strip()))
with (root/'BASELINE_AUDIT.json').open('xb') as f: f.write({raw!r})
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(control/'tools/rb2int/evaluate.py'),'--variant',{audit['variant']!r},'--runtime-root',{audit['runtime_root']!r},'--wm-config',{audit['config']!r},'--wm','/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt','--model-name',{model_name!r},'--development-direct','--baseline-audit',str(root/'BASELINE_AUDIT.json'),'--out',str(root/'tasks'),'--device','cuda','--memory-gib','5']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
with (root/'worker.log').open('x') as log:
 child=subprocess.Popen(argv,env=env,cwd=control,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
fields=Path('/proc/'+str(child.pid)+'/stat').read_text().split()
result=dict(root=str(root),pid=child.pid,start_ticks=fields[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),argv=argv)
save(root/'PROCESS.json',result);print(json.dumps(result))
'''
    elif args.action == 'status':
        code = '''
result=dict(root=str(root),exists=root.is_dir())
if root.is_dir():
 process=json.loads((root/'PROCESS.json').read_text());result['process']=process;result['alive']=alive(process)
 result['tasks']=len(list((root/'tasks').glob('w*_k*.json')))
 for name in ('BRIDGE_PASS.json','tasks/COMPLETE.json'):
  if (root/name).is_file(): result[name]=json.loads((root/name).read_text())
 if (root/'worker.log').is_file(): result['log_tail']=(root/'worker.log').read_text()[-2500:]
print(json.dumps(result))
'''
    else:
        code = f'''
import math
process=json.loads((root/'PROCESS.json').read_text());assert not alive(process), 'bridge still computing'
assert not (root/'BRIDGE_PASS.json').exists(), 'do not repeat a completed bridge verification'
new_path=root/'tasks/w969420_k0.json';old_path=Path({original_task!r})
new=json.loads(new_path.read_text());old=json.loads(old_path.read_text())
new_id=json.loads((new_path.parent/'IDENTITY.json').read_text());old_id=json.loads((old_path.parent/'IDENTITY.json').read_text())
assert (root/'tasks/COMPLETE.json').is_file() and new['identity_sha256']==sha(new_path.parent/'IDENTITY.json') and old['identity_sha256']==sha(old_path.parent/'IDENTITY.json')
for key in ('model','wm_sha256','config_sha256','candidate_sha256','export_sha256','K','horizon','t0','arms','mix','native_lanes','seed','update','switches'): assert new_id[key]==old_id[key], key
assert new['seed']==old['seed']==969420 and new['future']==old['future']==0
assert len(list((root/'tasks').glob('w*_k*.json')))==1
assert len(new['episodes'])==len(old['episodes'])==5 and {{e['arm'] for e in new['episodes']}}==set(new_id['arms'])
differences=[];regions=0;events=0
for episode in new['episodes']:
 prior=next(e for e in old['episodes'] if e['arm']==episode['arm'])
 assert episode['initial_sha256']==prior['initial_sha256']
 for h in ('120','300','600'):
  for key,value in episode['clocks'][h].items():
   delta=abs(value-prior['clocks'][h][key]);differences.append(delta);assert math.isclose(value,prior['clocks'][h][key],rel_tol=0,abs_tol=1e-8),(episode['arm'],h,key,delta)
 for region in ('upstream','merge','ramp'):
  a,b=episode['bins'][region],prior['bins'][region]
  assert a['post_counts']==b['post_counts'];regions+=1
  for x,y in zip(a['speed'],b['speed']):
   assert (x is None)==(y is None)
   if x is not None: differences.append(abs(x-y));assert math.isclose(x,y,rel_tol=0,abs_tol=1e-8)
  for h in ('120','300','600'):
   for key in ('event','onset_s','preexisting'): assert a['events'][h].get(key)==b['events'][h].get(key)
   events+=1
result=dict(passed=True,development_only=True,world=969420,future=0,arms=5,horizons_s=[120,300,600],max_absolute_error=max(differences),absolute_tolerance=1e-8,region_fields=regions,event_decisions=events,new_task_sha256=sha(new_path),original_task_sha256=sha(old_path),new_identity_sha256=sha(new_path.parent/'IDENTITY.json'),original_identity_sha256=sha(old_path.parent/'IDENTITY.json'),final_worlds_used=False,factual_future_targets_used=False)
save(root/'BRIDGE_PASS.json',result);print(json.dumps(result))
'''
    result = json.loads(ssh(args.host, common+code, timeout=60))
    with local.open('x', encoding='utf-8') as stream:
        json.dump(dict(result, host=args.host, model=args.model, action=args.action), stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
