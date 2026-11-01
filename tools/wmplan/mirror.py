"""Byte-identical gpuR inference-only mirror; no unscoped tree reads or overwrite."""
import json
from pathlib import Path
import subprocess

from tools.wmplan import ops
from tools.wmplan.bridge import GROOT


def stage():
    staged = json.loads((ops.OUT/'STAGED.json').read_text())
    selected = staged['binding']['model']['selected']
    result = json.loads(ops.remote(f'''import pathlib,json,hashlib,tarfile,re
root=pathlib.Path({ops.ROOT!r});groot=pathlib.Path({GROOT!r})
b=json.loads((root/'BINDING.json').read_text());items={{}}
def safe(name):
 assert not any(x in name.lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params'))
 assert not re.search(r'(?<![0-9])(?:964[0-9]{{3}}|962[0-9]{{3}}|977[0-9]{{3}})(?![0-9])',name)
def add(source,target,expected=None):
 source=pathlib.Path(source);safe(str(source));safe(str(target))
 data=source.read_bytes();h=hashlib.sha256(data).hexdigest()
 if expected:assert h==expected,str(source)
 items[str(target)]=dict(source=str(source),sha256=h,size=len(data))
for name,h in b['sources'].items():
 if name.startswith('pywm/'):add(root/name,groot/name,h)
for name,h in b['external_sources'].items():
 if name.startswith({ops.BASE!r}):continue
 add(name,name,h)
phys=('/root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f' if {selected!r}=='FULL-v5' else '/root/rbeval20261031/phys_code_22cc32f65a80')
extra=[phys+'/tools/rbeval/references.json','/root/wmcalib/expected_hashes.json','/root/wmcalib/data/design.json',
 '/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt',
 '/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt',
 '/root/autodl-tmp/c4/setupaudit20261030/fix/configs/execparity_cav_idm_v1.json',
 '/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml',
 '/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json']
export=json.loads(pathlib.Path(extra[-1]).read_text());extra.append(export['path'])
for seed in range(969400,969432):
 for n in ('result.json','start.pt','start_history.npz'):extra.append(f'/root/wmcalib/data/jobs/s{{seed}}_ctrl0/{{n}}')
base='/root/autodl-tmp/c4/'
private=('wmcombined_wm2/wm_s.pt','setupaudit20261030/policy_000190.pt','speedhead20261030/EXPORT_BOUND.json')
for name in extra:
 target=groot/'base'/name[len(base):] if name.startswith(base) and name[len(base):] in private else name
 add(name,target)
deps=pathlib.Path(b['wm_deps'])
for p in deps.rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts or 'tests' in p.parts:continue
 rel=str(p.relative_to(deps))
 if any(x in rel.lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vehicle_params','driver_params')):continue
 add(p,groot/'deps'/rel)
archive=pathlib.Path('/root/wmplan_gpuR_transfer_v1.tar')
assert archive.parent.is_dir() and not archive.exists()
with tarfile.open(archive,'x',dereference=True) as tar:
 for target,info in items.items():tar.add(info['source'],arcname=target.lstrip('/'),recursive=False)
print(json.dumps(dict(model={selected!r},items=items,archive=str(archive),bytes=archive.stat().st_size)))
''',timeout=180))
    # Compare existing named files before transfer; no existing byte is replaced.
    check = json.loads(ops.remote(f'''import pathlib,json,hashlib,shutil
items=json.loads({json.dumps(result['items'])!r});missing=[];conflicts=[]
for name,info in items.items():
 p=pathlib.Path(name)
 if p.exists():
  if hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:conflicts.append(name)
 else:missing.append(name)
free=shutil.disk_usage('/root').free
print(json.dumps(dict(missing=missing,conflicts=conflicts,root_free=free,
 required=sum(items[n]['size'] for n in missing))))
''',host='gpuR',timeout=90))
    ops.save('MIRROR_PREFLIGHT.json',dict(**check,archive_bytes=result['bytes']))
    if check['conflicts']:
        raise ValueError('gpuR existing-file conflict; preserve: '+str(check['conflicts']))
    if check['root_free'] < result['bytes']+check['required']+2*2**30:
        raise ValueError('gpuR root disk admission')
    local = Path('E:/Temp/opencode/wmplan_gpuR_transfer_v1.tar')
    if not local.parent.is_dir() or local.exists():
        raise ValueError('fresh approved transfer target required')
    subprocess.run(['scp','-o','BatchMode=yes',f'gpuA:{result["archive"]}',str(local)],check=True)
    subprocess.run(['scp','-o','BatchMode=yes',str(local),'gpuR:/root/wmplan_gpuR_transfer_v1.tar'],check=True)
    receipt = json.loads(ops.remote(f'''import pathlib,json,tarfile,hashlib,shutil
items=json.loads({json.dumps(result['items'])!r});root=pathlib.Path({GROOT!r})
assert root.parent.is_dir() and not root.exists()
with tarfile.open('/root/wmplan_gpuR_transfer_v1.tar') as tar:
 for member in tar:
  name='/'+member.name;assert name in items and member.isfile()
  p=pathlib.Path(name);h=items[name]['sha256']
  if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==h;continue
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:shutil.copyfileobj(tar.extractfile(member),f)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==h
manifest=dict(model={selected!r},base=str(root/'base'),hashes={{n:v['sha256'] for n,v in items.items()}},
 new_sumo_sim_s=0,inference_only=True)
with (root/'MIRROR.json').open('x') as f:json.dump(manifest,f,sort_keys=True)
print(json.dumps(dict(root=str(root),manifest=manifest,files=len(items),new_sumo_sim_s=0)))
''',host='gpuR',timeout=180))
    ops.save('MIRROR.json',receipt)
    print(json.dumps(dict(root=receipt['root'],files=receipt['files'],new_sumo_sim_s=0)))


def check():
    result = json.loads(ops.remote(f'''import pathlib,json,sys,os,hashlib,contextlib
root=pathlib.Path({GROOT!r});m=json.loads((root/'MIRROR.json').read_text());os.nice(19)
for name,h in m['hashes'].items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==h,name
sys.path[:0]=[str(root/'pywm'),str(root/'deps')]
os.environ.update(WMTEACH_BACKEND=m['model'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with contextlib.redirect_stdout(sys.stderr):
 from tools.wmteach import runtime
 runtime.BASE=pathlib.Path(m['base'])
 rt=runtime.bootstrap()
print(json.dumps(dict(pass_=True,host='gpuR',model=m['model'],fit_worlds=len(rt.batches),
 export=rt.export,new_sumo_sim_s=0,new_forecasts=0)))
''',host='gpuR',timeout=240))
    ops.save('MIRROR_CPU_GPU_GATE.json',result)
    print(json.dumps(result))


def repair_base():
    """Relocate identical private assets inside the loader's existing allowlist.

    Preserve the initial manifest and all initial copies; no loader/code change.
    """
    initial = json.loads((ops.OUT/'MIRROR.json').read_text())['manifest']
    result = json.loads(ops.remote(f'''import pathlib,json,hashlib,shutil
root=pathlib.Path({GROOT!r});p=root/'MIRROR.json';old=json.loads(p.read_text())
assert old==json.loads({json.dumps(initial)!r})
assert not (root/'MIRROR_INITIAL.json').exists()
with (root/'MIRROR_INITIAL.json').open('x') as f:f.write(p.read_text())
base=pathlib.Path('/root/rbeval20261031/wmplan20261101_v1/base')
assert base.parent.parent.is_dir() and not base.exists()
new=dict(old);new['hashes']=dict(old['hashes']);new['base']=str(base)
copies=[]
for name,h in old['hashes'].items():
 source=pathlib.Path(name)
 if not source.is_relative_to(pathlib.Path(old['base'])):continue
 target=base/source.relative_to(old['base']);target.parent.mkdir(parents=True,exist_ok=True)
 assert hashlib.sha256(source.read_bytes()).hexdigest()==h
 with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==h
 new['hashes'][str(target)]=h;copies.append(dict(source=str(source),target=str(target),sha256=h))
assert len(copies)==3
with (root/'MIRROR_REPAIRED.json').open('x') as f:json.dump(new,f,sort_keys=True)
# This is our own pre-dispatch manifest; its original bytes remain above.
with p.open('w') as f:json.dump(new,f,sort_keys=True)
print(json.dumps(dict(pass_=True,initial_manifest=old,repaired_manifest=new,copies=copies,
 new_sumo_sim_s=0,new_forecasts=0,runtime_source_changed=False)))
''',host='gpuR',timeout=90))
    ops.save('MIRROR_BASE_REPAIR.json',result)
    print(json.dumps(dict(pass_=result['pass_'],copies=len(result['copies']),new_sumo_sim_s=0)))


def repair_assets():
    """Bind the three package JSON initializers required during checkpoint load."""
    assets = json.loads(ops.remote('''import pathlib,json,base64,hashlib
root=pathlib.Path('/root/rbeval20261031/phys_code_22cc32f65a80/src/wms/model')
out={}
for name in ('entry_learned.json','merge_core_init.json','ramp_speed_adapt_init.json'):
 p=root/name;data=p.read_bytes()
 out[str(p)]=dict(sha256=hashlib.sha256(data).hexdigest(),content=base64.b64encode(data).decode())
print(json.dumps(out))
'''))
    result = json.loads(ops.remote(f'''import pathlib,json,base64,hashlib
assets=json.loads({json.dumps(assets)!r});root=pathlib.Path({GROOT!r})
p=root/'MIRROR.json';m=json.loads(p.read_text())
for name,value in assets.items():
 target=pathlib.Path(name);data=base64.b64decode(value['content']);h=value['sha256']
 assert target.parent.is_dir() and hashlib.sha256(data).hexdigest()==h
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==h
 else:
  with target.open('xb') as f:f.write(data)
 m['hashes'][name]=h
with (root/'MIRROR_ASSETS.json').open('x') as f:json.dump(m,f,sort_keys=True)
with p.open('w') as f:json.dump(m,f,sort_keys=True)
print(json.dumps(dict(pass_=True,assets={{n:v['sha256'] for n,v in assets.items()}},
 manifest=m,new_sumo_sim_s=0,new_forecasts=0,runtime_source_changed=False)))
''',host='gpuR'))
    ops.save('MIRROR_PACKAGE_ASSETS.json',result)
    print(json.dumps(dict(pass_=True,files=len(assets),new_sumo_sim_s=0)))
