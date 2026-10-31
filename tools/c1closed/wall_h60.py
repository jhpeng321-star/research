"""Check the named full-history wall cache's causal executor prerequisites."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
root=Path('E:/research-dense/data_cache/wallc1_states_h60_real')
receipt=json.loads((root/'BUILD_RECEIPT.json').read_text())
text=json.dumps(receipt)
seeds={int(s) for s in re.findall(r'w(\d+)m\d+p\d+t',text)}
assert seeds and all(8000<=s<9000 for s in seeds)
cache=root/'wall_same.json'
b=cache.read_bytes();x=json.loads(b)
rows=x['windows']
def shape(value):
    result=[]
    while isinstance(value,list):
        result.append(len(value));value=value[0] if value else None
    return result
details=[]
for r in rows:
    assert 8000<=int(re.match(r'w(\d+)m',r['world']).group(1))<9000
    timeline=r.get('wall_timeline',{})
    details.append(dict(world=r['world'],key=r['key'],t0=r['t0_s'],
        history_keys=list(r.get('history',{})),history_kin_shape=shape(r.get('history',{}).get('kin')),
        driver_history_shape=shape(r.get('driver_history',{}).get('kin')) if isinstance(r.get('driver_history'),dict) else [],
        controller_map_nonempty=bool(r.get('cav_controllers')),wall_members=len(r.get('wall_members',[])),
        timeline_keys=list(timeline),request_pre_t0=timeline.get('request_s') is not None and timeline['request_s']<=r['t0_s'],
        has_executor_snapshot=any(k in r for k in ('wall_executor_state','executor_state','wall_state','controller_state')),
        max_recorded_horizon_s=r['horizon_s'],keys=list(r)))
out=dict(cache=str(cache),sha256=hashlib.sha256(b).hexdigest(),states=len(rows),details=details,
    conclusion='Full-history named cache survives; causal executor restoration must still be established. No wall inference registered or launched.')
with (HERE/'evidence/WALL_H60_AUDIT.json').open('x') as f:json.dump(out,f,indent=1)
print(json.dumps(dict(states=len(rows),first=details[0] if details else None,
    executor_snapshots=sum(d['has_executor_snapshot'] for d in details),
    controller_maps=sum(d['controller_map_nonempty'] for d in details)),indent=1))
