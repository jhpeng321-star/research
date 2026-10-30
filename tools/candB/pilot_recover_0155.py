"""Read only the named current pilot receipts and their owned process identities."""
import json
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_wmfast_dispatch import ROOT


def main():
    code = f'''
import json,time
from pathlib import Path
root=Path({ROOT!r})
doc=dict(epoch=time.time(),root=str(root),owners={{}},files={{}})
for name in ('OWNERSHIP.json','repair_TRAINER_OWNERSHIP.json',
             'WMFAST_CONTROL_OWNERSHIP.json','continue_TRAINER_OWNERSHIP.json'):
    p=root/name
    if not p.exists(): continue
    owner=json.loads(p.read_text()); proc=Path('/proc')/str(owner['pid'])
    owner['alive']=False
    if (proc/'stat').exists():
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        owner.update(alive=fields[19]==str(owner['start_ticks']),state=fields[0],
            cpu_ticks=int(fields[11])+int(fields[12]),
            argv=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode())
    doc['owners'][name]=owner
for name in ('WMFAST_CONTROL_FAILURE.json','WMFAST_CLEAN_BOUNDARY.json',
             'WMFAST_CONTROL_STOP','STOP','EXIT.json','repair_EXIT.json',
             'continue_EXIT.json','GPU.jsonl','supervisor.log','WMFAST_CONTROL.log'):
    p=root/name
    if p.exists(): doc['files'][name]=p.read_text()[-12000:]
print(json.dumps(doc))
'''
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=10', 'gpuA',
        '/root/miniconda3/bin/python', '-'], input=code, text=True,
        capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr)
    doc = json.loads(result.stdout)
    receipt = Path(__file__).parent / f'pilot_recovery_0155_{int(time.time())}.json'
    with receipt.open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(dict(receipt=str(receipt), **doc)))


if __name__ == '__main__':
    main()
