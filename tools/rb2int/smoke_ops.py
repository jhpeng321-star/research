"""Exclusive one-use development smoke launch and read-only incarnation status."""
import argparse
import json
from pathlib import Path
import re

from tools.rbint.remote import ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status'))
    parser.add_argument('--host', choices=('gpuA', 'gpuR'), required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--recipe', choices=('wm_phys_v5.toml', 'wm_phys_v5_lp.toml'), required=True)
    parser.add_argument('--kind', choices=('closed60', 'fivearm600'), default='closed60')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--receipt', help='exclusive local launch receipt')
    args = parser.parse_args()
    if not re.fullmatch(r'/root/(?:rbeval20261031|autodl-tmp/c4/rb2int20261101)/phys_code_[a-f0-9]{12}', args.runtime):
        raise ValueError('immutable INT stage required')
    target = Path(args.receipt) if args.receipt else None
    if args.action == 'launch' and (target is None or not target.parent.is_dir() or target.exists()):
        raise ValueError('new local launch receipt in existing parent required')
    name = args.kind+'_'+('lp' if '_lp.' in args.recipe else 'v5')+'_'+args.device
    common = f'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
runtime=Path({args.runtime!r});stage=json.loads((runtime/'INT_STAGE.json').read_text())
assert runtime.name=='phys_code_'+stage['commit'][:12]
root=runtime.parent/('dev_'+stage['commit'][:12]+'_'+{name!r})
def save(path,data):
 with path.open('x') as f: json.dump(data,f,indent=1)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def incarnation(pid):
 fields=Path('/proc/'+str(pid)+'/stat').read_text().split()
 return dict(pid=pid,start_ticks=fields[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def alive(r):
 try:
  now=incarnation(r['pid'])
  return now=={{k:r[k] for k in ('pid','start_ticks','boot_id')}} and Path('/proc/'+str(r['pid'])+'/stat').read_text().split()[2]!='Z'
 except FileNotFoundError: return False
'''
    if args.action == 'status':
        code = '''
result=dict(root=str(root),exists=root.is_dir())
if root.is_dir():
 p=root/'PROCESS.json'
 if p.is_file():
  process=json.loads(p.read_text());result.update(process=process,alive=alive(process))
 result['tasks']=len(list((root/'tasks').glob('w*_k*.json')))
 result['receipts']={name:json.loads((root/name).read_text()) for name in ('CLOSED_SMOKE.json','tasks/DEV_CANARY_PASS.json','tasks/COMPLETE.json') if (root/name).is_file()}
 log=root/'worker.log'
 if log.is_file(): result['log_tail']=log.read_text()[-3500:]
print(json.dumps(result))
'''
    else:
        code = f'''
assert not root.exists(), 'preserve existing development smoke; never relaunch'
assert root.parent.is_dir()
assert shutil.disk_usage(root.parent).free>512*2**20, 'wait for free own output disk'
assert os.getloadavg()[0]<20, 'leave CPU headroom'
cg=Path('/sys/fs/cgroup');maximum=cg/'cpu.max';stat=cg/'cpu.stat'
resources=dict(load=os.getloadavg(),disk_free=shutil.disk_usage(root.parent).free)
if maximum.is_file() and maximum.read_text().split()[0]!='max':
 quota,period=map(int,maximum.read_text().split());capacity=quota/period
 def usage(): return int(dict(line.split() for line in stat.read_text().splitlines())['usage_usec'])
 begin=time.monotonic();before=usage();time.sleep(1);used=(usage()-before)/1e6/(time.monotonic()-begin)
 assert used+1<.8*capacity, 'wait for free effective CPU quota'
 resources.update(cpu_capacity=capacity,cpu_used=used)
memory=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
if memory.is_file() and memory.read_text().strip()!='max':
 free=int(memory.read_text())-int(current.read_text());assert free>8*2**30, 'wait for effective memory'
 resources['memory_free']=free
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
assert gpu.returncode==0
resources['gpu']=gpu.stdout.strip()
if {args.device!r}=='cuda':
 free,util=map(int,gpu.stdout.strip().split(','));assert free>13*1024, '5GiB own cap plus8GiB shared headroom'
 assert {args.kind!r}=='fivearm600', 'closed smoke is CPU only'
root.mkdir();save(root/'CLAIM_ONCE.json',dict(owner='S-RB2-INT',stage=stage,kind={args.kind!r},recipe={args.recipe!r},device={args.device!r},resources=resources,automatic_retry=False))
entry=runtime/('tools/rbint/smoke.py' if {args.kind!r}=='closed60' else 'tools/rb2int/dev_canary.py')
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(entry),'--out',str(root/'CLOSED_SMOKE.json' if {args.kind!r}=='closed60' else root/'tasks'),'--wm-config',str(runtime/'configs'/{args.recipe!r})]
if {args.kind!r}=='closed60': argv+=['--variants','FULL']
else: argv+=['--runtime-root',str(runtime),'--model-name','RB2-INT-DEV-'+{name!r},'--device',{args.device!r},'--memory-gib','5']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if {args.device!r}=='cuda' else '',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH='/root/autodl-tmp/c4/speedhead20261030/pythondeps')
with (root/'worker.log').open('x') as log:
 child=subprocess.Popen(argv,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(**incarnation(child.pid),argv=argv,root=str(root),stage=stage,entry_sha256=sha(entry),resources=resources)
save(root/'PROCESS.json',receipt);print(json.dumps(receipt))
'''
    result = json.loads(ssh(args.host, common+code, timeout=60))
    if args.action == 'launch':
        with target.open('x', encoding='utf-8') as stream:
            json.dump(dict(host=args.host, **result), stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
