"""Snapshot only owned results/traces/ledger; never include generated SUMO inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from tools.coopaudit.accounting import write
from tools.coopaudit.remote import run
from tools.coopaudit.design import checked


def collect(phase):
    evidence=Path(__file__).resolve().parent/'evidence'
    dest=evidence/('v2_'+phase)
    if dest.exists():
        raise ValueError('immutable collection already exists; do not overwrite')
    dest.mkdir()
    code=f'''
import hashlib,json,tarfile,time
from pathlib import Path
root=Path('/root/coopaudit20261031')
selection=json.loads((root/'CORRECTED_DESIGN.json').read_text())
storage=Path(selection['storage_root'])
phase={phase!r}
if phase=='complete':
    assert (root/'V2_MAIN_COMPLETE.json').exists()
else:
    assert (root/'V2_CANARY_COMPLETE.json').exists()
archive=storage/('EXPORT_'+phase+'_'+str(time.time_ns())+'.tgz')
files=[]
pid=json.loads((root/('v2-canary_PID.json' if phase=='canary' else 'v2-main_PID.json')).read_text())
scientific_files={{name:hashlib.sha256((Path(pid['code'])/'tools/coopaudit'/name).read_bytes()).hexdigest()
                  for name in ('full_rerun.py','controllers.py','trained.py','runner.py','design.py','release.py','accounting.py')}}
with tarfile.open(archive,'w:gz') as tf:
    for name in ('accounting.jsonl','CORRECTED_DESIGN.json','ARTIFACT_RELEASE.json',
                 'V2_CANARY_COMPLETE.json','V2_CANARY_VERIFIED.json','V2_MAIN_COMPLETE.json',
                 'V2_PANEL_ADMITTED.json','CPU_TESTS.json','v2-canary_PID.json','v2-main_PID.json'):
        p=root/name
        if p.is_file():
            tf.add(p,arcname=name,recursive=False)
            files.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for directory in sorted(storage.glob('v2_*')):
        if not directory.is_dir() or (phase=='canary' and not directory.name.startswith('v2_canary_')):
            continue
        assert (directory/'result.json').exists()
        for p in sorted(list(directory.glob('*.json'))+list((directory/'traces').glob('*.jsonl.gz'))):
            source_rel=str(p.relative_to(storage))
            rel=source_rel.replace(':','_')  # portable Windows filenames, data unchanged
            assert not __import__('re').search(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype|assignment',rel,__import__('re').I)
            tf.add(p,arcname='jobs/'+rel,recursive=False)
            files.append(dict(path='jobs/'+rel,source_remote=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=files,scientific_files=scientific_files)))
'''
    result=run('gpuR',code,timeout=120)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)
    receipt=json.loads(result.stdout)
    archive=dest/'receipts.tgz'
    # Destination parent is explicitly created/verified before SCP.
    assert dest.is_dir()
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',
                    'gpuR:'+receipt['archive'],str(archive)],check=True,timeout=120)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=receipt['sha256']:
        raise ValueError('archive transport SHA mismatch')
    with tarfile.open(archive,'r:gz') as tf:
        for item in tf.getmembers():
            if item.name.startswith('/') or ':' in item.name or '..' in Path(item.name).parts or not item.isfile():
                raise ValueError('unsafe archive member')
            checked(dest/item.name)
        tf.extractall(dest)
    for item in receipt['files']:
        if hashlib.sha256((dest/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('extracted receipt/trace SHA mismatch')
    write(dest/'COLLECTION.json',receipt)
    print(json.dumps(dict(ok=True,phase=phase,files=len(receipt['files']),sha256=receipt['sha256'],destination=str(dest))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('canary','complete'))
    collect(p.parse_args().phase)
