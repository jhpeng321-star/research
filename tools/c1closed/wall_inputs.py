"""Named safe historical cache schema audit; no forecast and no raw-path opens."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
GAP=Path('E:/agent-work/c4-c1gap/docs/paper_results/c1gap')
ids=json.loads((GAP/'wall_same_identities.json').read_text())
seeds=sorted({int(re.match(r'w(\d+)m',r['world']).group(1)) for r in ids})
assert all(not (964000<=s<=964199 or str(s).startswith(('962','977'))) for s in seeds)
recovery=json.loads((GAP/'recovery.json').read_text())
cache=Path('E:/research-dense/data_cache/wallc1_states_v2/wall_same.json')
expected=recovery['local'][str(cache).replace('/','\\')]['sha256']
data=cache.read_bytes();assert hashlib.sha256(data).hexdigest()==expected
x=json.loads(data)
print('top keys',list(x) if isinstance(x,dict) else 'list')
rows=x.get('windows') if isinstance(x,dict) else x
assert rows is not None
print('state type',type(rows).__name__)
if isinstance(rows,dict): rows=list(rows.values())
first=rows[0]
def schema(value):
    if isinstance(value,dict): return {k:schema(v) for k,v in value.items()}
    if isinstance(value,list):
        shape=[]
        while isinstance(value,list):
            shape.append(len(value));value=value[0] if value else None
        return dict(shape=shape,leaf=type(value).__name__)
    return type(value).__name__
out=dict(cache=str(cache),sha256=expected,states=len(rows),safe_world_seeds=seeds,
         first_state_schema=schema(first),top_keys=list(x) if isinstance(x,dict) else [],
         h60_named_cache_exists=Path('E:/research-dense/data_cache/wallc1_states_h60_real/wall_same.json').is_file(),
         note='schema-only eligibility audit; raw episode paths and future arrays are never supplied to a controller')
with (HERE/'evidence/WALL_INPUT_SCHEMA.json').open('x') as f:json.dump(out,f,indent=1)
print(json.dumps(out,indent=1))
