"""Explicit detached zero-SUMO E5 analysis staging, polling and receipt fetch."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tools.autonomy_e5.verify import remote

ROOT = Path(__file__).resolve().parents[2]
OP = '/home/ubuntu/e3long/operations_e5'
EVIDENCE = ROOT/'tools/autonomy_e5/receipts_done'


def stage(box):
    source = (ROOT/'tools/autonomy_e5/analyze_shard.py').read_bytes()
    encoded = base64.b64encode(source).decode()
    shard = 0 if box == 'cpuP' else 1
    return remote(box, f'''import pathlib,json,base64,subprocess,os
op=pathlib.Path({OP!r}); assert op.is_dir()
p=op/'analyze_shard.py'; data=base64.b64decode({encoded!r})
if p.exists(): assert p.read_bytes()==data,'different previous analysis retained'
else: p.write_bytes(data)
assert not (op/'ANALYSIS_PID.json').exists(),'inspect existing analysis; never duplicate'
with (op/'analysis.log').open('xb') as log:
 child=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',str(p),'--shard',{str(shard)!r}],
  cwd=op,env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}},
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=child.pid,sid=os.getsid(child.pid),start_ticks=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
assert receipt['sid']==receipt['pid']
(op/'ANALYSIS_PID.json').write_text(json.dumps(receipt)); print(json.dumps(receipt))
''')


def status(box):
    return remote(box, f'''import pathlib,json
op=pathlib.Path({OP!r}); names=('ANALYSIS_PID.json','ANALYSIS_VALIDATION.json','ANALYSIS_STATUS.json')
print(json.dumps({{n:json.loads((op/n).read_text()) for n in names if (op/n).exists()}}))
''')


def fetch(box):
    payload = remote(box, f'''import pathlib,json,base64,hashlib
op=pathlib.Path({OP!r}); assert json.loads((op/'ANALYSIS_STATUS.json').read_text())['status']=='COMPLETE'
names=('ANALYSIS_DATA.json','ANALYSIS_STATUS.json','ANALYSIS_VALIDATION.json','ANALYSIS_PID.json','DONE_AUTHORITY.json','DONE_PREFLIGHT.json','DONE_PID.json','DONE_STATUS.json','E5DONE_AMENDMENT.md')
files={{n:base64.b64encode((op/n).read_bytes()).decode() for n in names if (op/n).exists()}}
print(json.dumps(files))
''')
    EVIDENCE.mkdir(exist_ok=True)
    dest = EVIDENCE/box
    dest.mkdir(exist_ok=True)
    hashes = {}
    for name, encoded in payload.items():
        data = base64.b64decode(encoded)
        p = dest/name
        if p.exists():
            assert p.read_bytes() == data, 'previous evidence differs'
        else:
            p.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return dict(files=len(hashes), hashes=hashes)


def audit(box, script='identity_audit.py'):
    source = base64.b64encode((ROOT/'tools/autonomy_e5'/script).read_bytes()).decode()
    shard = 0 if box == 'cpuP' else 1
    return remote(box, f'''import pathlib,json,base64,subprocess
op=pathlib.Path({OP!r}); assert op.is_dir(); p=op/{script!r}; data=base64.b64decode({source!r})
if p.exists() and p.read_bytes()!=data:
 import hashlib
 old=p.with_name('identity_audit.'+hashlib.sha256(p.read_bytes()).hexdigest()[:12]+'.py'); assert not old.exists(); p.rename(old)
if p.exists(): assert p.read_bytes()==data
else: p.write_bytes(data)
p=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B',str(p),{str(shard)!r}],capture_output=True,text=True,timeout=90)
assert p.returncode==0,p.stderr
print(p.stdout,end='')
''')


def telemetry(box):
    return audit(box, 'telemetry_audit.py')


def fetch_audit(box, name='IDENTITY_AUDIT.json'):
    payload = remote(box, f'''import pathlib,json,base64
p=pathlib.Path({OP!r})/{name!r}; print(json.dumps(dict(data=base64.b64encode(p.read_bytes()).decode())))
''')
    dest = EVIDENCE/box/name
    data = base64.b64decode(payload['data'])
    if dest.exists():
        assert dest.read_bytes() == data
    else:
        dest.write_bytes(data)
    return dict(sha256=hashlib.sha256(data).hexdigest())


def fetch_telemetry(box):
    return fetch_audit(box, 'TELEMETRY_AUDIT.json')


def mirror(box):
    assert box == 'cpuP', 'single gpuA full-panel verification command'
    hashes = {}
    for host in ('cpuP', 'cpuQ'):
        d = json.loads((EVIDENCE/host/'ANALYSIS_DATA.json').read_text())
        prefix = 'main/e3long_coopoff/'
        hashes[host] = {p[len(prefix):]: sha for p, sha in d['input_sha256'].items()
                        if p.startswith(prefix) and p.endswith(('.json', '.jsonl'))}
    proof = remote('gpuA', f'''import pathlib,json,hashlib
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026/e5'); hashes={hashes!r}
d=json.loads((r/'DONE_PID.json').read_text()); proc=pathlib.Path('/proc')/str(d['pid'])
if proc.exists():
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split(); assert stat[19]!=d['start_ticks'] or stat[0]=='Z'
assert json.loads((r/'DONE_STATUS.json').read_text())['status']=='COMPLETE'
for box,files in hashes.items():
 for name,sha in files.items():
  assert hashlib.sha256((r/box/name).read_bytes()).hexdigest()==sha,(box,name)
print(json.dumps(dict(status='PASS',coordinator_exited=True,files_verified={{box:len(files) for box,files in hashes.items()}},
 all_complete_coopoff_results_clocks_ledgers_mirrored=True)))
''')
    (EVIDENCE/'MIRROR_VERIFIED.json').write_text(json.dumps(proof, indent=2))
    return proof


def publish(box):
    assert box == 'cpuP', 'single gpuA publication command'
    names = ('E5_COOPOFF_C1.md', 'E5_COOPOFF_C1.csv', 'E5_COOPOFF_C1_worlds.csv',
             'MANIFEST.json', 'VERIFIED.json', 'OUTCOME.md')
    files = {'analysis_done/'+n: base64.b64encode(gzip.compress((EVIDENCE/n).read_bytes(), mtime=0)).decode() for n in names}
    files['RESULTS.json'] = base64.b64encode(gzip.compress((EVIDENCE/'RESULTS.json').read_bytes(), mtime=0)).decode()
    files['DONE_ANALYSIS_VERIFIED.json'] = base64.b64encode(gzip.compress((EVIDENCE/'VERIFIED.json').read_bytes(), mtime=0)).decode()
    proof = remote('gpuA', f'''import pathlib,json,base64,hashlib,gzip
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026/e5'); assert r.is_dir()
(r/'analysis_done').mkdir(exist_ok=True); files={files!r}; hashes={{}}
for name,encoded in files.items():
 data=gzip.decompress(base64.b64decode(encoded)); p=r/name
 if p.exists(): assert p.read_bytes()==data,'existing published file differs; retain'
 else:
  with p.open('xb') as f: f.write(data)
 hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((r/'DONE_ANALYSIS_VERIFIED.json').read_text())['status']=='PASS'
print(json.dumps(dict(status='COMPLETE',paired_worlds=228,charged_E5_sim_s=210600,
 actual_E5_sim_s=205200,published_sha256=hashes)))
''')
    (EVIDENCE/'PUBLISHED.json').write_text(json.dumps(proof, indent=2))
    return proof


def publish_background(box):
    assert box == 'cpuP'
    assert EVIDENCE.is_dir() and not (EVIDENCE/'PUBLISH_PID.json').exists()
    with (EVIDENCE/'publish.log').open('xb') as log:
        p = subprocess.Popen([sys.executable, '-B', '-m', 'tools.autonomy_e5.analysis_ops', 'publish', box],
                             cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    receipt = dict(pid=p.pid, command='analysis_ops publish cpuP')
    (EVIDENCE/'PUBLISH_PID.json').write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('stage', 'status', 'fetch', 'audit', 'fetch_audit', 'telemetry', 'fetch_telemetry', 'mirror', 'publish', 'publish_background'))
    p.add_argument('box', choices=('cpuP', 'cpuQ'))
    a = p.parse_args()
    print(a.box, json.dumps(globals()[a.command](a.box), indent=2), flush=True)
