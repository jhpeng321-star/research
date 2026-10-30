"""Read-only, CPU-only inspection of the named validation runtime."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/wmrequal'))
from remote import remote

SCRIPT = r'''
import os, sys, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  name=os.fsdecode(args[0]).lower()
  if any(x in name for x in bad) or name.endswith('.xml'): raise RuntimeError('prohibited input')
sys.addaudithook(guard)
from pathlib import Path
p=Path('/root/autodl-tmp/c4/wmrequal20261030')
sys.path[:0]=[str(p),str(p/'tools/wmrequal'),str(p/'src'),'/root/autodl-tmp/c4/restart20261026/runtime_longint']
from real_panel import entries_for
root,entries=entries_for('validation')
design=Path('/root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json')
import torch
torch.set_num_threads(2)
from wms.control.policy import load_policy_components
for arm,spec in json.loads(design.read_text())['controllers'].items():
 path=Path(spec['checkpoint'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
 policy,encoder,rules,payload=load_policy_components(path)
 print('POLICY',arm,json.dumps(dict(spec=spec,format=payload.get('format'),
   lane_control=payload.get('lane_control'),has_lane_head=getattr(policy,'has_lane_head',None),
   residual_on_native=getattr(policy,'residual_on_native',None),lane_rules=str(rules),
   payload_scalar={k:v for k,v in payload.items() if isinstance(v,(str,int,float,bool))})))
for arm in sorted(set(e['arm'] for e in entries)):
 rows=[e for e in entries if e['arm']==arm]
 print('ARM_METADATA',arm,len(rows),json.dumps(rows[0]))
 ep=root/rows[0]['dir']/'meta.json'
 if ep.is_file():print('EPISODE_META',ep,ep.read_text())
'''

if __name__ == '__main__':
    remote(SCRIPT, timeout=180)
