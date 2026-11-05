"""Immutable accepted-G5 overlay, qualification and one own collection dispatch."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib
from .remote import run


def main():
    files = {}
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'b903c6c2',
                                    'tools/coopmwm', 'tools/coopfree', 'traffic_shepherd'], text=True).splitlines()
    for name in names:
        if not name.endswith('.py') or '/test' in name:
            continue
        if any(x in name.lower() for x in ('sealed', 'holdout', 'heldout', 'hetdata_truth', 'vehicle_params', 'vtype')):
            continue
        files[name] = subprocess.check_output(['git', 'show', 'b903c6c2:' + name])
    for name in ('__init__.py', 'collect_contexts.py', 'wm_contract.py'):
        files['tools/mining/' + name] = (Path(__file__).parent / name).read_bytes()
    payload = base64.b64encode(zlib.compress(json.dumps({k: base64.b64encode(v).decode()
                                                      for k, v in files.items()}).encode(), 9)).decode()
    code = r'''
import base64,hashlib,json,os,subprocess,time,zlib
from pathlib import Path
parent=Path('/root/mining20261106');root=parent/'executor_b903c6c2'
assert parent.is_dir() and not root.exists(), 'adopt existing dispatch; never duplicate'
root.mkdir();source=root/'source';source.mkdir()
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for name,encoded in files.items():
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(encoded))
old='/home/ubuntu/coopmwm20261104_launchcoop_src7'
fallback=[old+'/pytools/tools', '/home/ubuntu/e3final/pytools/tools']
(source/'tools/__init__.py').write_text('__path__.extend('+repr(fallback)+')\n'.replace('\\n','\n'))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONDONTWRITEBYTECODE='1',LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PYTHONPATH=':'.join([str(source),old+'/runtime_src',old+'/WM_PACKAGE',old+'/pytools']))
env['PATH']=env['SUMO_HOME']+'/bin:/home/ubuntu/miniconda3/bin:'+env['PATH']
manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
(root/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True))
authority=dict(g5_commit='b903c6c2',gpu_cap_sim_s=300000,cpuv_cap_sim_s=200000,
 total_mining_cap_sim_s=500000,phase='mining',full_collection_worlds=4,full_sim_s_per_world=1200,
 selection='max local ramp count, mainline occupancy, inverse ramp speed, proximity; no outcome input',
 verification='K8 H600 WM ranking only; native group/base/each solo both CRN seeds common H600; G5 exact fields/tape',
 clean3='periodic current B/R training contexts after launch; same procedure and effort; registered boundaries only',
 source_sha256=hashlib.sha256((root/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),unix=time.time())
(root/'COLLECTION_AUTHORITY.json').write_text(json.dumps(authority,sort_keys=True))
py='/home/ubuntu/miniconda3/bin/python'
check="from tools.coopmwm import accel_plant,group_lane; from tools.mining.collect_contexts import opportunities; import traffic_shepherd.c4.env as e; print(group_lane.__file__,e.__file__); assert hasattr(group_lane,'executable_fields'); compile(open('"+str(source/'tools/mining/collect_contexts.py')+"').read(),'<collector>','exec')"
r=subprocess.run([py,'-B','-c',check],env=env,capture_output=True,text=True,timeout=90)
(root/'QUALIFICATION.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),sort_keys=True))
assert r.returncode==0, r.stderr
cmd=[py,'-B','-u','-m','tools.mining.collect_contexts',str(root)]
with (root/'COLLECTION.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],argv=cmd,unix=time.time(),root=str(root))
(root/'COLLECTION_DISPATCH.json').write_text(json.dumps(record,sort_keys=True))
print(json.dumps(dict(dispatch=record,authority=authority,qualification=r.stdout)),flush=True)
'''
    result = run('gpuA', 'PAYLOAD=' + repr(payload) + '\n' + code,
                 '/root/miniconda3/bin/python -B -', timeout=110)
    receipt = Path('mining_receipts/EXECUTOR_STAGE_01.json')
    assert not receipt.exists()
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(result['returncode'])


if __name__ == '__main__':
    main()
