"""Prepare the matched cycle1 E5 arm, without dispatching simulation."""
import argparse
import base64
import hashlib
from pathlib import Path

from tools.loopresume.operate import remote

ROOT = Path(__file__).resolve().parents[2]
POLICY = 'e1fa4881bbc5c89d05298e979d98b23284308331ad1bbb12f9608fa67e105675'


def prepare():
    register = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    markdown_sha = hashlib.sha256(register.read_bytes()).hexdigest()
    for host in ('cpuP', 'cpuQ'):
        remote(host, """import pathlib
r=pathlib.Path('/home/ubuntu/e3long'); assert r.is_dir()
(r/'inputs/coopoff_cycle1').mkdir(exist_ok=True)
(r/'operations_e5').mkdir(exist_ok=True)
print('E5 directories ready')
""")
    remote('gpuR', f"""import pathlib,subprocess,hashlib,json
r=pathlib.Path('/root/autodl-tmp/c4/coopoff220261027')
p=r/'arm/loop/policy/c01.a2/policy_latest.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest()=={POLICY!r}
gate=p.with_name('native_gate_result.json'); g=json.loads(gate.read_text())
print('C1_GATE',json.dumps({{k:g[k] for k in ('passed','update','mean_delta','checkpoint_sha256')}}))
for host in ('cpuP','cpuQ'):
 for src,name in ((p,'policy_latest.pt'),(gate,'native_gate_result.json')):
  subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=10',str(src),
   host+':/home/ubuntu/e3long/inputs/coopoff_cycle1/'+name],check=True,timeout=120)
print('Coop-off cycle1 policy/gate staged; no simulation')
""", timeout=300)
    for host in ('cpuP', 'cpuQ'):
        remote(host, f"""import pathlib,sys,os,json,hashlib,shutil
r=pathlib.Path('/home/ubuntu/e3long'); root=r/'runtime'; op=r/'operations_e5'; target=r/'inputs/coopoff_cycle1'
sys.path[:0]=[str(root),str(root/'src'),str(r/'operations')]; os.chdir(root)
os.environ['PATH']='/home/ubuntu/miniconda3/bin:/usr/local/bin:/usr/bin:/bin:/home/ubuntu/.local/bin'
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
for name in ('wm_s_best.pt','config.resolved.json','physics.json'):
 src=r/'inputs/cycle1'/name; dest=target/name
 if not dest.exists(): dest.symlink_to(src)
base=json.loads((r/'inputs/cycle1/INPUTS.json').read_text())
for key,name in (('policy','policy_latest.pt'),('wm','wm_s_best.pt'),('wm_config','config.resolved.json'),('wm_physics','physics.json')):
 base[key]=str(target/name)
(target/'INPUTS.json').write_text(json.dumps(base,indent=2))
source=(r/'operations_ours/e3ours_runner.py').read_text()
assert source.count("OP = R/'operations_ours'")==1
source=source.replace("OP = R/'operations_ours'", "OP = R/'operations_e5'")
source=source.replace('64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c',{POLICY!r})
source=source.replace("task='S-E3OURS'", "task='S-E5-COOPOFF-C1'")
source=source.replace("assert L.E.digest(R/'runtime/SOURCE_MANIFEST.json') == SOURCE", "assert go.get('arm_alias') == 'ours-coopoff'; assert L.E.digest(R/'runtime/SOURCE_MANIFEST.json') == SOURCE")
runner=op/'e5_runner.py'; runner.write_text(source)
import runpy
ns=runpy.run_path(str(runner)); h=ns['hashes'](base)
from traffic_shepherd.c4 import e3long as L
from traffic_shepherd.config import load_config
from traffic_shepherd.damping import CirclesController
from wms.control.policy import load_policy_fn
CirclesController(load_config()); load_policy_fn(target/'policy_latest.pt',device='cpu')
for shard in ({0 if host == 'cpuP' else 1},):
 for phase,arms in (('e3long_baseline',{{'native','mappo'}}),('e3long_policy',{{'ours'}})):
  results=L.E.read(r/'main'/phase/('shard'+str(shard))/'RESULTS.json')
  assert len(results)==114*len(arms) and all(x['ok'] for x in results)
go=dict(experiment=L.ID,task='S-E5-COOPOFF-C1',go=True,latest_native_gate_passing=True,
 issued_by='main owner approval13:25',phase='policy',arms=['ours'],arm_alias='ours-coopoff',
 cap_sim_s=205200,worlds=228,production_seeds=[973000,973227],workers_per_box=6,
 input_hashes=h,source_manifest_sha256=ns['SOURCE'],design_sha256=L.canonical_sha(L.plan()),
 operational_adapter_sha256=L.E.digest(runner),markdown_registration_sha256={markdown_sha!r},
 funding='205200 of unused410400 policy-phase allocation; gated arms not run',
 comparison='ours-coopON(cycle1) minus ours-coopOFF(cycle1); world-cluster bootstrap120/300/600s')
ns['validate'](base,go)
(target/'GO_E5.json').write_text(json.dumps(go,indent=2))
proof={{'status':'PREPARED_NOT_LAUNCHED','charged_sim_s':0,'policy_sha256':h['policy'],
 'go_sha256':L.E.digest(target/'GO_E5.json'),'runner_sha256':L.E.digest(runner),
 'arm_alias':'ours-coopoff','internal_runtime_arm':'ours','source_unchanged':True}}
(op/'PREFLIGHT.json').write_text(json.dumps(proof,indent=2)); print(json.dumps(proof))
""", timeout=240)
    payload = base64.b64encode((ROOT/'tools/autonomy_e5/remote.py').read_bytes()).decode()
    remote('gpuA', f"""import pathlib,base64,hashlib,json
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026'); assert r.is_dir()
p=r/'e5'; p.mkdir(exist_ok=True); script=p/'remote.py'; script.write_bytes(base64.b64decode({payload!r}))
print(json.dumps({{'status':'PREPARED_NOT_LAUNCHED','command':'/root/miniconda3/bin/python -B '+str(script)+' launch',
 'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}}))
""")


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
