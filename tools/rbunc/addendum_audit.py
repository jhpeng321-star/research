"""Read only explicit observable FIT starts and documented scenario source."""
import json
from pathlib import Path

from remote import ssh


def main():
    result = ssh('''
import os,json,hashlib,torch
from pathlib import Path
torch.set_num_threads(1)
root=Path('/home/ubuntu/rbunc20261031/fit_probe_v2')
terminal=json.loads((root/'PANEL_TERMINAL.json').read_text())
assert all(rc==0 for _,rc in terminal['results'])
p=Path('/home/ubuntu/wmdata20261031/jobs/s969400_native/start.pt')
s=torch.load(p,map_location='cpu',weights_only=False)
source=Path('/home/ubuntu/e3final/runtime/traffic_shepherd/scenarios.py')
text=source.read_text().splitlines()
print(json.dumps(dict(load=os.getloadavg(),cpus=os.cpu_count(),
 terminal=terminal,keys=list(s),state_keys=list(s['state']),
 queue_first=s['observed_queue_history'][:4],queue_last=s['observed_queue_history'][-4:],
 entry_first=s['past_entries'][:2],source=str(source),
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 source_excerpt='\\n'.join(text[35:90])),default=lambda x:x.tolist()))
''')
    out = Path(__file__).resolve().parents[2]/'docs/paper_results/rbunc/addendum_input_audit.json'
    with out.open('x', encoding='utf-8') as stream:
        stream.write(result)
    print(json.dumps(json.loads(result), indent=1))


if __name__ == '__main__':
    main()
