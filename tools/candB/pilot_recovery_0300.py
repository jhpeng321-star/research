"""Read only named B receipts and owned process metadata; never query CUDA."""
import json
from pathlib import Path
import subprocess
import time


CODE = r'''
import hashlib,json,time
from pathlib import Path
root=Path('/root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031')
doc=dict(epoch=time.time(),owners={},files={},outputs={})
for name in ('OWNERSHIP.json','repair_TRAINER_OWNERSHIP.json','continue_TRAINER_OWNERSHIP.json',
             'WMFAST_CONTROL_OWNERSHIP.json','WMFAST4_CONTROL_OWNERSHIP.json'):
    p=root/name
    if not p.exists(): continue
    owner=json.loads(p.read_text()); proc=Path('/proc')/str(owner['pid'])
    owner['alive']=False
    if (proc/'stat').exists():
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        owner.update(alive=fields[19]==str(owner['start_ticks']),state=fields[0],
            argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    doc['owners'][name]=owner
for name in ('EXIT.json','repair_EXIT.json','continue_EXIT.json','GPU.jsonl',
             'WMFAST_CONTROL_FAILURE.json','WMFAST4_CONTROL_FAILURE.json'):
    if (root/name).exists(): doc['files'][name]=(root/name).read_text()[-6000:]
out=root/'pilot_EXPLORATORY-LEAN'
for name in ('RUN_SUMMARY.json','frozen_scoring_baseline.json','updates.jsonl'):
    p=out/name
    if p.exists(): doc['outputs'][name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text=p.read_text())
for name in ('policy_native_best.pt','policy_latest.pt'):
    p=out/name
    if p.exists(): doc['outputs'][name]=dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(doc))
'''


def main():
    result = subprocess.run(['ssh','-o','ConnectTimeout=10','gpuA','/root/miniconda3/bin/python','-'],
        input=CODE,text=True,capture_output=True,timeout=35,check=True)
    doc = json.loads(result.stdout)
    doc['tracked_numstat'] = subprocess.run(['git','diff','--numstat'],text=True,
        capture_output=True,check=True).stdout
    target = Path(__file__).with_name(f'pilot_recovery_0300_{int(time.time())}.json')
    with target.open('x') as stream:
        json.dump(doc,stream,indent=1)
    summary = json.loads(doc['outputs']['RUN_SUMMARY.json']['text'])
    print(json.dumps(dict(receipt=str(target),owners=doc['owners'],files=doc['files'],
        status=summary['status'],error=summary.get('error'),updates=len(summary['updates']),
        baseline=doc['outputs']['frozen_scoring_baseline.json']['sha256'],
        checkpoints={k:v for k,v in doc['outputs'].items() if k.endswith('.pt')},
        tracked_numstat=doc['tracked_numstat'])))


if __name__ == '__main__':
    main()
