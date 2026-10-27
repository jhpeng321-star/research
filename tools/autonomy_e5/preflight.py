"""Exercise the staged E5 dispatcher without simulation, sequentially on both CPUs."""
from tools.loopresume.operate import remote


SOURCE = r'''
import hashlib,json,pathlib,sys,os,runpy
r=pathlib.Path('/home/ubuntu/e3long'); op=r/'operations_e5'
sys.path[:0]=[str(op),str(r/'runtime'),str(r/'runtime/src'),str(r/'operations')]
os.chdir(r/'runtime')
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import e5_runner as W
manifest=json.loads((r/'runtime/SOURCE_MANIFEST.json').read_text())
print('MANIFEST_SCHEMA',type(manifest).__name__,list(manifest)[:8])
files=manifest.get('files',manifest)
assert isinstance(files,dict)
for name,sha in files.items():
 p=pathlib.PurePosixPath(name)
 assert not p.is_absolute() and '..' not in p.parts
 assert not any(s in name.lower() for s in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype'))
 assert hashlib.sha256((r/'runtime'/name).read_bytes()).hexdigest()==sha,name
assert len(files)==543
inputs=json.loads((r/'inputs/coopoff_cycle1/INPUTS.json').read_text())
go=json.loads((r/'inputs/coopoff_cycle1/GO_E5.json').read_text())
W.validate(inputs,go)
gate=json.loads((r/'inputs/coopoff_cycle1/native_gate_result.json').read_text())
assert gate['passed'] and gate['update']==180 and gate['checkpoint_sha256']==W.POLICY
print('GATE',json.dumps(gate))
test=(r/'operations_ours/e3ours_verify.py').read_text()
test=test.replace('import e3ours_runner as W','import e5_runner as W')
test=test.replace('/inputs/cycle1/INPUTS.json','/inputs/coopoff_cycle1/INPUTS.json')
test=test.replace('/inputs/cycle1/GO_OURS_MAIN_20261027T0720.json','/inputs/coopoff_cycle1/GO_E5.json')
exec(compile(test,'e5_no_sumo_dispatch_preflight','exec'),{'__name__':'__main__'})
from e3base_priority import collection_kind,probe
assert collection_kind(['python','-m','traffic_shepherd.c4.collect','--policy-pairs'])
assert not collection_kind(['python','-m','metervis'])
assert not (op/'CANARY_PID.json').exists() and not (op/'MAIN_PID.json').exists()
assert not (r/'main/e3long_coopoff').exists()
doc=dict(status='PASS',no_SUMO=True,source_files_verified=len(files),priority=probe(),
         go_sha256=hashlib.sha256((r/'inputs/coopoff_cycle1/GO_E5.json').read_bytes()).hexdigest(),
         tests=['GO arm/cap','actual dispatcher: load/YIELD/stale mirror/live collection','source543','native gate'])
(op/'DISPATCH_PREFLIGHT.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(doc))
'''


if __name__ == '__main__':
    for host in ('cpuP', 'cpuQ'):
        remote(host, SOURCE, timeout=240)
