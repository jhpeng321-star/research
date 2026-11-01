"""Read only existing complete option-task receipts; no remote file writes/imports."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/paper_results/bdrank/option_cache_audit.json'


def main():
    if OUT.exists():
        print('Reusing existing task-owned option cache audit:', OUT)
        return
    receipt = ROOT.parent/'c4-wmteach/docs/paper_results/wmteach/SNAPSHOT_1793521615.json'
    snapshot = json.loads(receipt.read_text(encoding='utf-8'))
    states = snapshot['files']['IDENTITY.json']['quality_states']
    source = f'''import os,json,pathlib,hashlib,time
os.nice(10)
bad={('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params','vtype')!r}
def read(p):
 assert not any(s in str(p).lower() for s in bad)
 b=p.read_bytes()
 return json.loads(b),hashlib.sha256(b).hexdigest()
result={{'collected_unix':time.time(),'new_inference':False,'remote_writes':False,'versions':{{}}}}
for model,name in [('FULL-v4','wmteach20261101_v3'),('FULL-v5','wmteach_v5quality20261101_v2')]:
 root=pathlib.Path('/root/autodl-tmp')/name
 ident,idhash=read(root/'IDENTITY.json')
 entry={{'source':ident['source'],'identity_sha256':idhash,'tasks':[],'labels':[]}}
 for state in {states!r}:
  for k in range(8):
   p=root/'tasks'/state/f'k{{k}}.json'
   t,digest=read(p)
   flags=[a.get('breakdown') for actions in t['results'].values() for a in actions.values()]
   entry['tasks'].append({{'state_id':state,'future':k,'sha256':digest,'path':str(p),'keys':list(t),
     'wall_s':t.get('wall_s'),'all_breakdown_missing':all(v is None for v in flags),
     'saved_outcome_keys':sorted({{key for actions in t['results'].values() for a in actions.values() for key in a}})}})
  for h in (120,300):
   p=root/'tasks'/state/f'LABEL_H{{h}}.json'
   t,digest=read(p)
   entry['labels'].append({{'state_id':state,'horizon_s':h,'sha256':digest,'expected_delay':t['paired_gate']['expected_delay'],
     'future_ids':t['future_ids'],'available':t['available']}})
 result['versions'][model]=entry
print(json.dumps(result,allow_nan=False))
'''
    command = '/root/miniconda3/bin/python -B -c "import sys;exec(compile(sys.stdin.read(),\'<bdrank-read-only>\',\'exec\'))"'
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'gpuA', command],
                       input=source, text=True, capture_output=True, encoding='utf-8', timeout=90)
    if r.returncode:
        raise RuntimeError(r.stdout+r.stderr)
    result = json.loads(r.stdout)
    result['local_snapshot_sha256'] = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert OUT.parent.is_dir()
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=1, allow_nan=False)
        f.write('\n')
    for model, entry in result['versions'].items():
        times = [t['wall_s'] for t in entry['tasks'] if t['wall_s'] is not None]
        print(model, 'tasks', len(entry['tasks']), 'labels', len(entry['labels']),
              'all breakdown missing', all(t['all_breakdown_missing'] for t in entry['tasks']),
              'recorded worker GPU minutes', sum(times)/60)


if __name__ == '__main__':
    main()
