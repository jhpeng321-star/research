"""Read named owned receipts and process progress; never query GPU or dispatch."""
import json
from pathlib import Path
import subprocess
import time


ROOT = '/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030'
CODE = r'''
import json,time
from pathlib import Path
root=Path('/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030')
doc=dict(epoch=time.time(),root=str(root),owners={},files={})
for name in ('OWNERSHIP.json','dry_OWNERSHIP.json','continue_OWNERSHIP.json','continue_trainer_OWNERSHIP.json'):
    path=root/name
    if not path.is_file(): continue
    owner=json.loads(path.read_text())
    proc=Path('/proc')/str(owner['pid'])
    if (proc/'stat').is_file():
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        owner.update(same_incarnation=fields[19]==str(owner['start_ticks']),state=fields[0],cpu_ticks=int(fields[11])+int(fields[12]),argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    doc['owners'][name]=owner
for name in ('GPU_ADMISSION.jsonl','continue_GPU.jsonl','continue_supervisor.log','dry.log','continue.log','dry_EXPLORATORY/frozen_scoring_baseline.json','dry_EXPLORATORY/updates.jsonl','dry_EXPLORATORY/STOP','continue_EXIT.json'):
    path=root/name
    if path.is_file():
        doc['files'][name]=dict(size=path.stat().st_size,mtime=path.stat().st_mtime,tail=path.read_text()[-6000:])
print(json.dumps(doc))
'''


def main():
    result = subprocess.run(['ssh', 'gpuA', '/root/miniconda3/bin/python', '-'],
                            input=CODE, text=True, capture_output=True, check=True, timeout=25)
    doc = json.loads(result.stdout)
    target = Path(__file__).parent / f'pilot_recovery_2305_{int(time.time())}.json'
    with target.open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc))


if __name__ == '__main__':
    main()
