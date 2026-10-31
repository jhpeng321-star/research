"""Mirror only allowlisted compact task receipts; no large/raw simulator states."""
from pathlib import Path
import base64
import hashlib
import json
from remote import ssh

SOURCE=r'''
from pathlib import Path
import base64,json,hashlib
r=Path('/home/ubuntu/sumocore20261031')
paths=[r/n for n in ('plan.json','ledger.jsonl','input_audit.json','helper_crosscheck.json','padding_audit.json',
                     'STATUS_AMENDMENT.json','preflight/receipt.json','preflight_all/receipt.json')]
for pat in ('AMENDMENT_*.json','DISPATCH_*.json','STOP*','scores/*.json','scores/*.csv','jobs/*/failure.json','jobs/*/START.json'):
 paths.extend(r.glob(pat))
out={}
for p in paths:
 if p.is_file():
  data=p.read_bytes();out[str(p.relative_to(r))]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
print(json.dumps(out))
'''


def main():
    root=Path(__file__).resolve().parents[2]/'docs/paper_results/sumocore'
    assert root.parent.is_dir()
    root.mkdir(exist_ok=False)
    files=json.loads(ssh(SOURCE))
    hashes={}
    for name,record in files.items():
        rel=Path(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
        data=base64.b64decode(record['data']);assert hashlib.sha256(data).hexdigest()==record['sha256']
        with p.open('xb') as f:f.write(data)
        hashes[name]=record['sha256']
    with (root/'MIRROR_SHA256.json').open('x') as f:json.dump(hashes,f,indent=1)
    print(json.dumps(dict(files=len(files),local=str(root),
        summary_sha256=hashes.get('scores/summary.json'),verification_sha256=hashes.get('scores/verification.json'))))


if __name__=='__main__':main()
