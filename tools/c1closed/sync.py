"""Fetch only S-C1CLOSED own outputs/receipts; no dispatch or inference."""
from pathlib import Path
from operate import scp, REMOTE, HERE
from probe import ssh
import json
import hashlib

for name in ('evidence','out/panel'):
    (HERE/name).mkdir(parents=True,exist_ok=True)
manifest = json.loads(ssh('gpuA',f'''from pathlib import Path
import json,hashlib
p=Path({REMOTE!r})
names=['DISPATCH.json','COMPLETE.json','evidence/LAUNCH.json','evidence/PROGRESS.jsonl','evidence/DISPATCH.log']
names += [str(q.relative_to(p)) for q in (p/'panel').glob('*_h300.json')]
names += [str(q.relative_to(p)) for q in (p/'evidence').glob('*_attempt*.json')]
print(json.dumps({{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names if (p/n).is_file()}}))
'''))
for name,digest in manifest.items():
    if name.startswith('panel/'):
        target=HERE/'out'/name
    else:
        target=HERE/'evidence'/Path(name).name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        continue
    if target.exists():
        # Mutable logs are retained by content hash before refreshed copies.
        backup=target.with_name(target.name+'.'+hashlib.sha256(target.read_bytes()).hexdigest()[:12])
        if not backup.exists(): backup.write_bytes(target.read_bytes())
    scp(f'gpuA:{REMOTE}/{name}',target)
print(json.dumps(dict(downloaded_manifest=len(manifest),panel_files=len(list((HERE/'out/panel').glob('*.json'))))))
