"""Stage/start once or inspect the owner17:35 claim-preserving continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tools.rb2int.audit import ROOT
from tools.rb2int.drain_ops import CONTROL, EXPECTED
from tools.rb2int.final_ops import REMOTE_ROOT
from tools.rbint.remote import ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status'))
    parser.add_argument('--host', choices=tuple(EXPECTED), required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    out = Path(args.receipt)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('exclusive local receipt in existing parent required')
    common = f'''
import hashlib,json,os,subprocess,time
from datetime import datetime
from pathlib import Path
root=Path({REMOTE_ROOT!r});control=Path({CONTROL!r});expected={EXPECTED[args.host]!r}
stage=root/'owner1735_control'
def save(path,value):
 with path.open('x') as stream: json.dump(value,stream,indent=1)
def alive(process):
 try:
  fields=Path('/proc/'+str(process['pid'])+'/stat').read_text().split()
  return fields[2]!='Z' and fields[21]==process['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==process['boot_id']
 except FileNotFoundError: return False
'''
    if args.action == 'start':
        source = ROOT/'tools/rb2int/capped_worker.py'
        committed = subprocess.run(['git', 'show', 'HEAD:tools/rb2int/capped_worker.py'],
            cwd=ROOT, capture_output=True, check=True).stdout
        if source.read_bytes() != committed:
            raise ValueError('only exact committed capped source may be staged')
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
        code = f'''
assert not stage.exists(), 'preserve prior coordinator deployment'
assert root.is_dir() and (root/'OWNER_1735_DRAIN_APPLIED.json').is_file()
original=json.loads((root/'SUPERVISOR_PROCESS.json').read_text())
assert {{k:original[k] for k in expected}}==expected and alive(original)
assert Path('/proc/'+str(original['pid'])+'/stat').read_text().split()[2]=='T'
raw={committed!r};compile(raw,'capped_worker.py','exec')
stage.mkdir()
with (stage/'capped_worker.py').open('xb') as stream: stream.write(raw)
save(stage/'DEPLOYED.json',dict(source_commit={head!r},source_sha256={hashlib.sha256(committed).hexdigest()!r},model_source_changed=False))
save(stage/'START_ONCE.json',dict(owner='S-RB2-INT',original=original,evaluate_signals=False,redispatched_tasks=0))
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',str(stage/'capped_worker.py'),'--root',str(root),'--host',{args.host!r}]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (stage/'coordinator.log').open('x') as log:
 child=subprocess.Popen(argv,cwd=control,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
fields=Path('/proc/'+str(child.pid)+'/stat').read_text().split()
result=dict(host={args.host!r},pid=child.pid,start_ticks=fields[21],boot_id=expected['boot_id'],argv=argv,
 phase='wait for all original children; then exclusive capped continuation',
 cap={2 if args.host == 'gpuA' else 1},source_commit={head!r},source_sha256={hashlib.sha256(committed).hexdigest()!r},
 started_at=datetime.now().astimezone().isoformat())
save(stage/'PROCESS.json',result)
print(json.dumps(result))
'''
    else:
        code = '''
result=dict(exists=stage.is_dir(),time=datetime.now().astimezone().isoformat())
if stage.is_dir():
 process=json.loads((stage/'PROCESS.json').read_text());result.update(process=process,alive=alive(process))
 result['log_tail']=(stage/'coordinator.log').read_text()[-5000:]
 for name in ('OWNER_1735_CAPPED_START_ONCE.json','OWNER_1735_CAPPED_HANDOVER.json','OWNER_1735_CAPPED_STOP.json','PANEL_COMPLETE.json','PANEL_TERMINAL_INCOMPLETE.json'):
  if (root/name).is_file(): result[name]=json.loads((root/name).read_text())
 original=json.loads((root/'SUPERVISOR_PROCESS.json').read_text());result['original_alive']=alive(original)
 if alive(original): result['original_state']=Path('/proc/'+str(original['pid'])+'/stat').read_text().split()[2]
 result['gpu']=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
print(json.dumps(result))
'''
    result = json.loads(ssh(args.host, common+code, timeout=90))
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
