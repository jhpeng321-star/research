"""Package exact cpuV environment and named parity closure. Zero physics."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

h=Path('/home/ubuntu')
r=h/'coopmwm20261106_s_mining/parity_stage_01'
assert r.parent.is_dir()
r.mkdir(exist_ok=False)
os.nice(19)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
deny=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
def guard(p):
    text=str(p).lower()
    assert not any(x in text for x in deny), text
    # Static implementation source required by the frozen manifest is opaque;
    # population/parameter/truth datasets are never included.
    assert p.suffix=='.py' or not any(x in text for x in ('driver_truth','vtype_truth','vehicle_params'))
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): digest.update(b)
    return digest.hexdigest()
entries={}
env=json.loads((h/'cpuv_setup_20261105/ENV_MANIFEST.json').read_bytes())
for row in env:
    p=h/row['dst'];guard(p)
    if '__pycache__' in p.parts or p.suffix=='.pyc':continue
    if 'link' in row:
        assert p.is_symlink() and os.readlink(p)==row['link']
        entries[row['dst']]=dict(link=row['link'])
    else:
        actual=sha(p);assert actual==row['sha256'],str(p)
        entries[row['dst']]=dict(size=p.stat().st_size,sha256=actual)
def add(p):
    guard(p);guard(p.resolve())
    if p.name in ('__pycache__','.pytest_cache','.git') or p.suffix=='.pyc':return
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if any(x in str(child).lower() for x in deny):continue
            if child.suffix!='.py' and any(x in str(child).lower() for x in ('truth','vehicle_params')):continue
            add(child)
    elif p.is_file():
        entries[p.relative_to(h).as_posix()]=dict(size=p.stat().st_size,sha256=sha(p))
for rel in ('coopmwm20261104_launchcoop_src7','e3final/runtime',
            'mappospeed/production2/policy_00169.pt','cpus_env/libmkl_intel_shim.so',
            'cpuv_parity_20261105/PREFLIGHT.json','cpuv_parity_20261105/PARITY_PASS.json'):
    add(h/rel)
pre=json.loads((h/'cpuv_parity_20261105/PREFLIGHT.json').read_bytes())
for case in pre['inputs']:
    root=Path(case['source'])
    for name in ('SNAPSHOT.pt','cand1.spec.json','cand1.result.json','cand1/RESULT.json','cand1/RESULT.json.gz'):
        add(root/name)
manifest=r/'MANIFEST.json'
manifest.write_text(json.dumps(entries,sort_keys=True,separators=(',',':')))
archive=r/'cpuv_parity.tar.gz'
with tarfile.open(archive,'x:gz',compresslevel=1,dereference=False) as tar:
    tar.add(manifest,arcname='PARITY_CLONE_MANIFEST.json',recursive=False)
    for rel,record in sorted(entries.items()):
        p=h/rel
        if 'link' in record:tar.add(p,arcname=rel,recursive=False)
        else:
            # Materialize source links only after both path guards pass.
            info=tar.gettarinfo(str(p.resolve()),arcname=rel)
            assert info.isfile(),rel
            with p.open('rb') as f:tar.addfile(info,f)
record=dict(schema='sol.mining.parity-package.v1',archive=str(archive),
    archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
    manifest_sha256=sha(manifest),files=len(entries),
    unpacked_bytes=sum(v.get('size',0) for v in entries.values()),unix=time.time(),sim_s=0)
(r/'PACKAGED.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
