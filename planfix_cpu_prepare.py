"""Stage immutable CPU-port dependencies only; never bootstrap, search or simulate.

No GPU connection. Missing bootstrap inputs are reported, never substituted.
All scientific source/learned artifacts come from exact Git objects; existing
cpuV WMTEACH code and frozen WM2 weights remain at their original paths.
"""
import base64
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

from planfix_ops import remote, note, TEMP

HERE = Path(__file__).parent
PHYS = '22cc32f6'
COLLECT = 'f631ce90cfd2f23213f7c57a828c848e2db10ecd'
DENY = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
        'hetdata_truth', 'vehicle_params', 'vtype')


def blob(commit, name):
    if any(word in name.lower() for word in DENY):
        raise ValueError('prohibited source path')
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=HERE)


def collect():
    files = {}
    requests = []
    for commit, prefix, dest in ((PHYS, 'src/wms', 'src/wms'),
                                 (PHYS, 'tools/rbint', 'tools/rbint'),
                                 (PHYS, 'tools/rbeval', 'tools/rbeval'),
                                 (COLLECT, 'tools/coopmwm', 'tools/coopmwm')):
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
                                         '--', prefix], cwd=HERE, text=True).splitlines()
        for name in names:
            if not name.endswith('.py') or '/test_' in name or name.endswith('/conftest.py'):
                continue
            if any(word in name.lower() for word in DENY):
                continue
            target = dest + name[len(prefix):]
            requests.append((commit, name, target))
    for name in ('tools/rbeval/references.json', 'configs/wm_phys.toml',
                 'docs/paper_results/rbunc/observed_arrivals.json',
                 *('artifacts/rbfit/shrinkage_v1/' + n for n in
                   ('idm_population.json', 'idm_shrinkage.json',
                    'driver_shrinkage.json', 'provenance.json'))):
        requests.append((PHYS, name, name))
    # One Git plumbing process, rather than hundreds of Windows process starts.
    response = subprocess.check_output(['git', 'cat-file', '--batch'], cwd=HERE,
        input=''.join(commit + ':' + name + '\n' for commit, name, _ in requests).encode())
    at = 0
    for _, _, target in requests:
        end = response.index(b'\n', at)
        _, kind, size = response[at:end].split()
        assert kind == b'blob'
        begin = end + 1
        data = response[begin:begin + int(size)]
        assert response[begin + int(size):begin + int(size) + 1] == b'\n'
        at = begin + int(size) + 1
        files[target] = base64.b64encode(data).decode()
    assert at == len(response)
    # Preserve the original model-loading TOML chain, including exact relative bases.
    queue = ['configs/c4_wm_loop14_r16_ft.toml']
    import re
    while queue:
        name = queue.pop()
        if name in files:
            continue
        data = blob(PHYS, name)
        files[name] = base64.b64encode(data).decode()
        match = re.search(r'^extends\s*=\s*"([^"]+)"', data.decode(), re.M)
        if match:
            queue.append('configs/' + match.group(1))
    return files


CODE = r'''
import base64,hashlib,json,os,sys,time,zlib
from pathlib import Path
w=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix');assert w.is_dir()
data=Path(__ARCHIVE__).read_bytes();assert hashlib.sha256(data).hexdigest()==__SHA__
files=json.loads(zlib.decompress(data));manifest={n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in files.items()}
root=w/('cpu_prepare_'+__SHA__[:12]);root.mkdir(exist_ok=True)
for name,value in files.items():
 assert not any(x in name.lower() for x in __DENY__) and '..' not in Path(name).parts
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(value)
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[name]
 else:
  with p.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[name]
p=root/'SOURCE_MANIFEST.json'
if p.exists():assert json.loads(p.read_text())==manifest
else:
 with p.open('x') as out:json.dump(manifest,out,sort_keys=True);out.flush();os.fsync(out.fileno())
wm=Path('/home/ubuntu/c4/main14/restart20261026/work/w968195/C4-LOOP14-COOPON-20261026/c03.a2/inputs/wm/wm_s.pt')
assert hashlib.sha256(wm.read_bytes()).hexdigest()=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
source=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa')
inherited={}
for name in ('runtime.py','scale.py','marginal.py'):
 p=source/'pytools/tools/wmteach'/name;assert p.is_file();inherited[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
# Check names only. These exact frozen bootstrap inputs cannot be fabricated
# from the live training outcomes or replaced by another model/source prior.
required={
 'fit_start_priors':'/root/wmcalib/expected_hashes.json',
 'fit_start_design':'/root/wmcalib/data/design.json',
 'dreamer_config_checkpoint':'/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',
 'physical_export_binding':'/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json',
 'fitted_posterior_artifact':str(root/'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz'),
}
missing={}
for key,name in required.items():
 try:
  if not Path(name).is_file():missing[key]=dict(path=name,reason='absent')
 except PermissionError:missing[key]=dict(path=name,reason='inaccessible legacy GPU-only path; cpuV mapping required')
receipt=dict(status='SOURCE_PREPARED_BOOTSTRAP_INPUTS_MISSING' if missing else 'SOURCE_PREPARED',
 source_root=str(root),collector_commit=__COLLECT__,physical_commit='22cc32f6',source_files=len(manifest),
 source_bytes=sum((root/n).stat().st_size for n in manifest),package_sha256=__SHA__,
 wm_checkpoint=str(wm),wm_sha256=hashlib.sha256(wm.read_bytes()).hexdigest(),
 inherited_wmteach_hashes=inherited,unavailable_original_bootstrap_inputs=missing,
 clean2_exists=Path('/home/ubuntu/coopmwm20261105_clean2').exists(),
 cpu_search_started=False,new_sim_s=0,phase_charged_sim_s=33600,
 note='Preparation only. No startup gates, model bootstrap, search, SUMO or bank switch. CPU <=19/48 cores after clean2 launch; one speed search before any expansion.')
p=root/'PREPARED.json'
if not p.exists():
 with p.open('x') as out:json.dump(receipt,out,indent=2,sort_keys=True);out.flush();os.fsync(out.fileno())
else:
 old=json.loads(p.read_text());assert old['package_sha256']==receipt['package_sha256']
print('CPU_PREPARED',json.dumps(receipt),flush=True)
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage-only',action='store_true',help='inspect/adopt prior uploaded archive by exact SHA; do not transfer again')
    args=parser.parse_args()
    files = collect()
    packed = zlib.compress(json.dumps(files, sort_keys=True).encode(), 9)
    sha = hashlib.sha256(packed).hexdigest()
    assert TEMP.is_dir()
    package = TEMP / ('planfix_cpu_source_' + sha[:12] + '.z')
    if package.exists():
        assert hashlib.sha256(package.read_bytes()).hexdigest() == sha
    else:
        with package.open('xb') as out:
            out.write(packed)
    destination = '/home/ubuntu/coopmwm20261105_clean1/plan_fix/' + package.name
    code = CODE.replace('__SHA__', repr(sha)).replace('__DENY__', repr(DENY)).replace(
        '__COLLECT__', repr(COLLECT)).replace('__ARCHIVE__', repr(destination))
    note('cpuV', f'CPU source preparation only: {len(files)} immutable files, {len(packed)} compressed bytes, SHA {sha}; zero search/SUMO')
    assert len(packed)<4000000, 'keep preparation small; bulky learned artifacts need separate resumable staging'
    # Pooled scp uses an explicit file, avoiding interrupted inherited-stdin
    # sessions in the Windows->WSL->F6 supervisor chain. No direct SSH fallback.
    if not args.stage_only:
        copied=subprocess.run(['wsl','-d','Ubuntu-22.04','--','python3','-B',
            '/mnt/e/agent-work/c4-variety-collect/planfix_cpuv_transport.py','--timeout','45',
            '--put','/mnt/e/Temp/opencode/'+package.name,destination],capture_output=True,timeout=100)
        print(copied.stdout.decode(errors='replace'),copied.stderr.decode(errors='replace'),flush=True)
        note('cpuV','pooled CPU source file upload rc='+str(copied.returncode))
        assert copied.returncode==0, 'inspect archive/receipt before another upload'
    text = remote('cpuV', code, timeout=45)
    receipt = json.loads(next(x[len('CPU_PREPARED '):] for x in text.splitlines()
                              if x.startswith('CPU_PREPARED ')))
    path = HERE / 'PLANFIX_CPU_PREPARED.json'
    if path.exists():
        assert json.loads(path.read_text()) == receipt
    else:
        with path.open('x', encoding='utf-8') as out:
            json.dump(receipt, out, indent=2, sort_keys=True)
