"""Verify scientific source identity across read-only GPU reference and own CPU copies."""
import ast
import hashlib
import json
from pathlib import Path

from operate import ROOT, remote

BODY = '''import hashlib,json,os,pathlib
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')
manifest={}
for relative in ('src/wms','traffic_shepherd'):
 for parent,dirs,files in os.walk(root/relative):
  dirs[:]=[d for d in dirs if d!='__pycache__' and not any(t in d.lower() for t in bad)]
  for name in files:
   p=pathlib.Path(parent)/name
   if p.suffix!='.py' or any(t in str(p).lower() for t in bad): continue
   manifest[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for relative in ('tools/coopw/horizon_gate.py','tools/loop14_checks/accept.py','tools/r10_attribution.py','tools/wm_eval_common.py'):
 p=root/relative; manifest[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(manifest,sort_keys=True))
'''
proof = {}
for host in ('gpuA','cpuP','cpuQ'):
    prefix = ("import pathlib; root=pathlib.Path('/root/autodl-tmp/c4/restart20261026/runtime_c1')\n"
              if host=='gpuA' else "import pathlib; root=pathlib.Path.home()/'longhzn/runtime'\n")
    proof[host] = json.loads(remote(host,prefix+BODY,echo=False))
differences = {host: {key: [proof['gpuA'].get(key), proof[host].get(key)]
                     for key in set(proof['gpuA']) | set(proof[host])
                     if proof['gpuA'].get(key) != proof[host].get(key)} for host in ('cpuP','cpuQ')}
print(json.dumps({'differences':differences},indent=1))
unrelated = {'src/wms/control/policy14_cli.py','src/wms/control/policy14_recovery.py'}
assert all(not (set(d)-unrelated) for d in differences.values())
assert proof['cpuP'] == proof['cpuQ']
source = remote('cpuP', "from pathlib import Path; print((Path.home()/'longhzn/run_cpu.py').read_text(),end='')",echo=False)
assert ast.dump(ast.parse(source)) == ast.dump(ast.parse((Path(__file__).with_name('run_cpu.py')).read_text()))
out = {'scientific_files':{k:v for k,v in proof['gpuA'].items() if k not in unrelated},
       'exact_inference_source_identity':True,'unrelated_policy_source_differences':differences,
       'deployed_runner_sha256':hashlib.sha256(source.encode()).hexdigest(),
       'committed_runner_ast_identical':True}
(ROOT/'tools/longhzn/evidence/source.json').write_text(json.dumps(out,indent=1)+'\n')
print(json.dumps({'exact_inference_source_identity':True,'files':len(out['scientific_files']),
                  'committed_runner_ast_identical':True}))
