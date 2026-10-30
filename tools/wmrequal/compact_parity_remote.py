"""Wait boundedly for the owned full K8 probe, then compare complete row metrics."""
from pathlib import Path
import json
import time

p=Path('/root/autodl-tmp/c4/wmrequal20261030')
q=p/'compact/validation/batch_000.json'
deadline=time.monotonic()+900
while not q.is_file():
    if time.monotonic()>deadline:
        raise TimeoutError('owned compact full-batch probe still incomplete')
    time.sleep(30)
a=json.loads((p/'fast/validation/batch_000.json').read_text())
b=json.loads(q.read_text())
maximum={}
for x,y in zip(a['rows'],b['rows']):
    assert (x['pair_key'],x['draw'],x['mix'],x['arm'])==(y['pair_key'],y['draw'],y['mix'],y['arm'])
    for h in x['values']:
        for k in x['values'][h]:
            maximum[k]=max(maximum.get(k,0.),abs(x['values'][h][k]-y['values'][h][k]))
doc=dict(rows=len(a['rows']),horizons=[60,90,120,180,300],
         maximum_absolute_difference=maximum,dense_seconds=a['seconds'],compact_seconds=b['seconds'],
         pass_numeric=max(maximum.values())<.02)
(p/'COMPACT_PARITY.json').write_text(json.dumps(doc,indent=1))
print(json.dumps(doc,indent=1))
assert doc['pass_numeric'], 'compact numerical parity failed; do not adopt this runtime'
