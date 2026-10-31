"""Own immutable candidate canary/continuation; never dispatch a peer's claim.

A successful one-task canary is retained in shard0 and reused by the panel.
An unsuccessful canary/worker is terminal: no automatic scientific retry.
"""
import argparse
import json
from pathlib import Path
import re

from tools.rbint.remote import ssh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('canary', 'continue', 'monitor', 'status'))
    parser.add_argument('--host', choices=('gpuA', 'gpuR'), default='gpuA')
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'WM-PHYS-FULL-[A-Za-z0-9_.-]+', args.name):
        raise ValueError('INT owns FULL candidates only; A/B/C belong to runners')
    if not re.fullmatch(r'/root/rbeval20261031/phys_code_[a-f0-9]{12}', args.runtime):
        raise ValueError('use an immutable INT stage')
    root = '/root/rbeval20261031/int/'+args.name
    common = f'''
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path({root!r});runtime=Path({args.runtime!r})
def save(path,data):
 with path.open('x') as f: json.dump(data,f,indent=2)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def alive(r):
 try: return Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id'] and Path('/proc/'+str(r['pid'])+'/stat').read_text().split()[21]==r['start_ticks']
 except FileNotFoundError: return False
def incarnation(pid): return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=Path('/proc/'+str(pid)+'/stat').read_text().split()[21])
def launch(i,canary):
 out=root/f'shard{{i}}';out.mkdir(exist_ok=True)
 argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(runtime/'tools/rbint/evaluate_FULL.py'),'--out',str(out),'--runtime-root',str(runtime),'--wm','/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt','--wm-config',str(runtime/'configs/wm_phys.toml'),'--model-name',{args.name!r},'--shard',str(i),'--shards','3','--memory-gib','5']
 if canary: argv+=['--max-tasks','1']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
 with (root/('canary.log' if canary else f'shard{{i}}.log')).open('x') as log:
  p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=runtime,start_new_session=True)
 return dict(**incarnation(p.pid),argv=argv,unix=time.time())
'''
    if args.action == 'canary':
        code = '''
assert runtime.is_dir() and (runtime/'INT_STAGE.json').is_file()
assert os.getloadavg()[0]<16, 'wait for a free CPU'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>6500, 'wait for a free GPU allocation'
smoke=runtime.parent/('SMOKE_'+runtime.name.removeprefix('phys_code_')+'.json')
assert json.loads(smoke.read_text())['passed']
assert root.parent.parent.is_dir();root.parent.mkdir(exist_ok=True);root.mkdir()
save(root/'SPEC.json',dict(owner='S-RB-INT',host='''+repr(args.host)+''',runtime=str(runtime),stage=json.loads((runtime/'INT_STAGE.json').read_text()),smoke_sha256=sha(smoke),variant='FULL',tasks=160,world_order='reserved-first',training=False))
save(root/'CANARY_CLAIM.json',dict(time=time.time(),automatic_retry=False))
r=launch(0,True);save(root/'CANARY_PROCESS.json',r);print(json.dumps(r))
'''
    elif args.action == 'continue':
        code = '''
spec=json.loads((root/'SPEC.json').read_text());assert spec['owner']=='S-RB-INT' and spec['runtime']==str(runtime)
assert not alive(json.loads((root/'CANARY_PROCESS.json').read_text())), 'canary still live'
assert not (root/'shard0/STOP').exists(), 'terminal canary failure'
files=list((root/'shard0').glob('w*_k*.json'));assert len(files)==1, 'need successful one-task canary'
task=json.loads(files[0].read_text());assert task['future']==0 and len(task['episodes'])==5
assert task['future_sampling']['population_audit']['conserved']
assert os.getloadavg()[0]<16, 'wait for free CPU'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>19500, 'wait for free aggregate3x5GiB allocation'
save(root/'PANEL_CLAIM.json',dict(time=time.time(),reuse=files[0].name,automatic_retry=False))
workers=[launch(i,False) for i in range(3)]
save(root/'PANEL_PROCESS.json',dict(workers=workers));print(json.dumps(workers))
'''
    elif args.action == 'monitor':
        source = Path(__file__).with_name('panel_monitor.py').read_text(encoding='utf-8')
        code = '''
assert (root/'PANEL_PROCESS.json').is_file()
assert not (root/'MONITOR_ONCE.json').exists(), 'preserve existing monitor'
save(root/'MONITOR_DISPATCH_CLAIM.json',dict(unix=time.time(),automatic_retry=False))
source='''+repr(source)+'''
path=root/'PANEL_MONITOR.py'
with path.open('x') as f: f.write(source)
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(path),'--root',str(root)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'monitor.log').open('x') as log:
 p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root,start_new_session=True)
r=dict(**incarnation(p.pid),argv=argv,source_sha256=sha(path));save(root/'MONITOR_PROCESS.json',r);print(json.dumps(r))
'''
    else:
        code = '''
print('LOAD',os.getloadavg())
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv'],text=True))
print('ROOT',str(root),'exists',root.is_dir())
if root.is_dir():
 for name in ('CANARY_PROCESS.json','PANEL_PROCESS.json','MONITOR_PROCESS.json'):
  p=root/name
  if p.exists():
   r=json.loads(p.read_text());print(name,[(v['pid'],alive(v)) for v in r.get('workers',[r])])
 for i in range(3):
  out=root/f'shard{i}';print(out.name,len(list(out.glob('w*_k*.json'))),'complete',(out/'COMPLETE.json').exists(),'STOP',(out/'STOP').exists())
 for name in ('MONITOR_FAILED.json','MONITOR_COMPLETE.json','ANALYSIS_full.json','ANALYSIS_interim04.json','ANALYSIS_interim08.json','canary.log','shard0.log','shard1.log','shard2.log','monitor.log'):
  p=root/name
  if p.exists(): print(name,p.read_text()[-5000:])
'''
    print(ssh(args.host, common+code, timeout=90))


if __name__ == '__main__':
    main()
