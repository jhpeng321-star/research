"""Fetch only this task's named offline receipts; never checkpoints or traffic data."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

SCRIPT = r'''
import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path('/root/autodl-tmp/c4/nativeinit_s1')
names=['MEASURE_PID.json','MEASURE_CRN_PID.json','MEASURE_FINAL_PID.json','TRACE_ZERO.json','DELIVERED_RESCORE.json']
for tag in ('measure30','measure30_crn','measure30_final','trace0','trace0_crn'):
    names += [tag+'.log',tag+'/RUN_SUMMARY.json',tag+'/native_curve.jsonl',tag+'/gate_panel_result.json',
              tag+'/native_gate_result.json']
payload=io.BytesIO()
with tarfile.open(fileobj=payload,mode='w:gz') as tar:
    for name in names:
        path=root/name
        if path.is_file(): tar.add(path,arcname=name)
    receipt=json.loads((root/'DELIVERED_RESCORE.json').read_text())
    normalized={}
    for name,expected in receipt['delivered_runtime_source_sha256'].items():
        source=(root/'runtime_verified'/name).read_bytes()
        assert hashlib.sha256(source).hexdigest()==expected
        normalized[name]=hashlib.sha256(source.replace(b'\r\n',b'\n')).hexdigest()
    text=(json.dumps(normalized,indent=1)+'\n').encode()
    member=tarfile.TarInfo('SOURCE_NORMALIZED_SHA256.json')
    member.size=len(text)
    tar.addfile(member,io.BytesIO(text))
sys.stdout.buffer.write(payload.getvalue())
'''

if __name__ == '__main__':
    result = subprocess.run(['ssh','-o','BatchMode=yes','gpuR','/root/miniconda3/bin/python -'],
                            input=SCRIPT.encode(),capture_output=True,check=True)
    root = Path(__file__).resolve().parent / 'results'
    root.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as tar:
        assert all(not m.issym() and not m.islnk() and not m.name.startswith('/')
                   and '..' not in Path(m.name).parts for m in tar.getmembers())
        tar.extractall(root,filter='data')
    # Receipts are text: canonical LF digests survive Git's Windows CRLF checkout.
    hashes = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
              for p in root.rglob('*') if p.is_file() and p.name != 'FETCH_SHA256.json'}
    (root/'FETCH_SHA256.json').write_text(json.dumps(hashes,indent=1)+'\n')
    print(json.dumps({'receipt_files_hashed':len(hashes),'sha256':hashes}))
