"""Write-once local preservation of label provenance and dispatch identity."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
labels=HERE/'out/labels.json'
x=json.loads(labels.read_text())
assert len(x['rows'])==200
for r in x['rows']:
    assert 969400<=r['seed']<=969439
    for h,v in r['values'].items():
        assert abs(v['J']-sum(v[k] for k in ('source','upstream','merge','discharge')))<1e-7
receipt=dict(task='S-C1CLOSED',state='RUNNING_NOT_A_RESULT',
    dispatch=json.loads((HERE/'evidence/DISPATCH.json').read_text()),
    labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),label_rows=200,
    label_source_sha256=x['source_sha256'],label_design_sha256=x['design_sha256'],
    label_boundary='observed genuine t300 state + recorded post-step tape starting t300.5',
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
with (HERE/'evidence/PANEL_LAUNCHED.json').open('x') as f:json.dump(receipt,f,indent=1)
print(json.dumps(dict(label_rows=200,dispatch_pid=receipt['dispatch']['pid'],task_count=receipt['dispatch']['task_count'])))
