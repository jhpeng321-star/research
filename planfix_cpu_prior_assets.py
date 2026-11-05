"""Package only exact FIT priors and the frozen inferred-posterior loader asset.

No tensor deserialization, model invocation or simulation. Original SHA metadata
is checked before packaging; only the bootstrap's 32 registered FIT donors enter.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib
from planfix_ops import TEMP, note, remote
from planfix_cpu_prepare import blob

HERE = Path(__file__).parent
ORIGINAL = Path('E:/agent-work/c4-wmcalib/tools/wmcalib')
DEST = '/home/ubuntu/coopmwm20261105_clean1/plan_fix'

def package():
    expected_path = ORIGINAL/'evidence/gpuR_expected_hashes.json'
    expected = json.loads(expected_path.read_text())
    design_path = ORIGINAL/'data/starts/design.json'
    files = {'data/design.json': design_path.read_bytes()}
    hashes = {}
    for seed in range(969400, 969432):
        for name in ('start.pt','start_history.npz','result.json'):
            relative = f'jobs/s{seed}_ctrl0/{name}'
            path = ORIGINAL/'data/starts'/relative
            if name == 'result.json' and not path.exists():
                path = ORIGINAL/'data'/f's{seed}_ctrl0'/name
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            assert sha == expected[relative], (relative, sha, expected[relative])
            files['data/'+relative] = raw
            hashes[relative] = sha
    files['expected_hashes.json'] = json.dumps(hashes,sort_keys=True).encode()
    posterior = 'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz'
    files[posterior] = blob('22cc32f6', posterior)
    manifest = {name: dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in files.items()}
    provenance = dict(original_expected_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        original_design_sha256=hashlib.sha256(files['data/design.json']).hexdigest(),
        donor_seeds=list(range(969400,969432)),original_expected_source=str(expected_path),
        posterior_git_object='22cc32f6:'+posterior,source_files=manifest,new_sim_s=0)
    files['ASSET_PROVENANCE.json'] = json.dumps(provenance,sort_keys=True).encode()
    packed = zlib.compress(json.dumps({n:base64.b64encode(v).decode() for n,v in files.items()},sort_keys=True).encode(),6)
    digest = hashlib.sha256(packed).hexdigest()
    assert TEMP.is_dir()
    path = TEMP/f'planfix_cpu_priors_{digest[:12]}.z'
    if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    else:
        with path.open('xb') as out:out.write(packed)
    return path, packed, digest, provenance

ADOPT = r'''
import base64,hashlib,json,os,zlib
from pathlib import Path
os.nice(19)
w=Path(__DEST__);chunks=w/('prior_chunks_'+__SHA__[:12]);chunks.mkdir(exist_ok=True)
manifest=__PARTS__
present=[]
for i,h in enumerate(manifest):
 p=chunks/('%04d_%s.part'%(i,h))
 if p.is_file():assert hashlib.sha256(p.read_bytes()).hexdigest()==h;present.append(i)
if len(present)!=len(manifest):
 print('PART_STATUS',json.dumps(dict(present=present,total=len(manifest))),flush=True)
else:
 packed=b''.join((chunks/('%04d_%s.part'%(i,h))).read_bytes() for i,h in enumerate(manifest))
 assert hashlib.sha256(packed).hexdigest()==__SHA__
 files=json.loads(zlib.decompress(packed));root=w/('cpu_priors_'+__SHA__[:12]);root.mkdir(exist_ok=True)
 for name,data in files.items():
  assert '..' not in Path(name).parts and not Path(name).is_absolute()
  raw=base64.b64decode(data);p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:
   with p.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
 prov=json.loads((root/'ASSET_PROVENANCE.json').read_text())
 for name,r in prov['source_files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==r['sha256']
 receipt=dict(root=str(root),package_sha256=__SHA__,files=len(files),all_file_shas_verified=True,
              donor_seeds=prov['donor_seeds'],new_sim_s=0,model_bootstrap=False)
 p=root/'VERIFIED.json'
 if p.exists():assert json.loads(p.read_text())==receipt
 else:
  with p.open('x') as out:json.dump(receipt,out,sort_keys=True)
 print('PRIOR_ASSETS_VERIFIED',json.dumps(receipt),flush=True)
'''

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--upload',action='store_true')
    parser.add_argument('--max-chunks',type=int,default=4);a=parser.parse_args()
    assert 1 <= a.max_chunks <= 8
    path,packed,sha,prov=package()
    parts=[packed[i:i+1048576] for i in range(0,len(packed),1048576)]
    hashes=[hashlib.sha256(p).hexdigest() for p in parts]
    local=dict(package=str(path),sha256=sha,bytes=len(packed),chunks=len(parts),provenance=prov)
    print(json.dumps({k:v for k,v in local.items() if k!='provenance'}),flush=True)
    receipt_path=HERE/'PLANFIX_CPU_PRIOR_INVENTORY.json'
    if receipt_path.exists():assert json.loads(receipt_path.read_text())==local
    else:
        with receipt_path.open('x') as out:json.dump(local,out,indent=2,sort_keys=True)
    if not a.upload:raise SystemExit(0)
    code=ADOPT.replace('__DEST__',repr(DEST)).replace('__SHA__',repr(sha)).replace('__PARTS__',repr(hashes))
    text=remote('cpuV',code,timeout=30)
    if 'PRIOR_ASSETS_VERIFIED ' not in text:
        status=json.loads(next(x[len('PART_STATUS '):] for x in text.splitlines() if x.startswith('PART_STATUS ')))
        uploaded=0
        for i,(raw,h) in enumerate(zip(parts,hashes)):
            if i in status['present']:continue
            if uploaded >= a.max_chunks:break
            part=TEMP/f'planfix_cpu_priors_{sha[:12]}_{i:04d}.part'
            if part.exists():assert hashlib.sha256(part.read_bytes()).hexdigest()==h
            else:
                with part.open('xb') as out:out.write(raw)
            destination=f'{DEST}/prior_chunks_{sha[:12]}/{i:04d}_{h}.part'
            note('cpuV',f'exact FIT-prior chunk {i+1}/{len(parts)} upload attempt 1/1; SHA {h}')
            result=subprocess.run(['wsl','-d','Ubuntu-22.04','--','python3','-B',
                '/mnt/e/agent-work/c4-variety-collect/planfix_cpuv_transport.py','--timeout','35',
                '--put','/mnt/e/Temp/opencode/'+part.name,destination],capture_output=True,timeout=95)
            print(result.stdout.decode(errors='replace'),result.stderr.decode(errors='replace'),flush=True)
            note('cpuV',f'exact FIT-prior chunk {i+1} upload outcome rc={result.returncode}')
            if result.returncode:raise SystemExit('inspect SHA-present chunks on next invocation; no blind retransmission')
            uploaded+=1
        text=remote('cpuV',code,timeout=40)
    if 'PRIOR_ASSETS_VERIFIED ' not in text:
        note('cpuV','bounded prior asset upload batch complete; retained SHA-checked chunks, zero inference/SUMO')
        raise SystemExit(0)
    receipt=json.loads(next(x[len('PRIOR_ASSETS_VERIFIED '):] for x in text.splitlines() if x.startswith('PRIOR_ASSETS_VERIFIED ')))
    with (HERE/'PLANFIX_CPU_PRIORS_VERIFIED.json').open('x') as out:json.dump(receipt,out,indent=2,sort_keys=True)
    note('cpuV',f'original 32 FIT donor priors and frozen inferred-posterior asset SHA-verified; package {sha}; zero inference/SUMO')
