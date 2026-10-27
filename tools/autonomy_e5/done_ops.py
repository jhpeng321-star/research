"""Sequential, explicit S-E5DONE staging, zero-charge checks and inspection."""
import base64
import hashlib
import json
from pathlib import Path
import sys

from tools.autonomy_e5.verify import remote

ROOT = Path(__file__).resolve().parents[2]
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
GPU = '/root/autodl-tmp/c4/restart20261026/e5'
CPU = '/home/ubuntu/e3long/operations_e5'
SEEDS = [973217, 973219, 973221, 973223, 973225, 973227]


def stage():
    amendment = '## S-E5DONE'+REG.read_text(encoding='utf-8').split('## S-E5DONE', 1)[1]
    runner = (ROOT/'tools/autonomy_e5/done_runner.py').read_bytes()
    coordinator = (ROOT/'tools/autonomy_e5/remote_done.py').read_bytes()
    authority = dict(task='S-E5DONE', issued_by='main 2026-10-27 16:00 act-on-recommendation',
                     seeds=SEEDS, additional_sim_s=5400, E5_cap_sim_s=210600,
                     cpuQ_cap_sim_s=108000, policy_phase_used_sim_s=415800, policy_phase_cap_sim_s=615600,
                     amendment_sha256=hashlib.sha256(amendment.encode()).hexdigest(),
                     registration_sha256=hashlib.sha256(REG.read_bytes()).hexdigest(),
                     done_runner_sha256=hashlib.sha256(runner).hexdigest(),
                     remote_done_sha256=hashlib.sha256(coordinator).hexdigest())
    verifier = base64.b64encode((ROOT/'tools/autonomy_e5/fix_runner.py').read_bytes()).decode()
    for box, shard in (('cpuP', 0), ('cpuQ', 1)):
        proof = remote(box, f'''import pathlib,json,sys,hashlib,os,shutil,subprocess
r=pathlib.Path('/home/ubuntu/e3long'); op=r/'operations_e5'; sys.path.insert(0,str(op))
import base64,types
ns={{'__name__':'e5done_preflight'}}; exec(compile(base64.b64decode({verifier!r}),'e5done_preflight','exec'),ns)
F=types.SimpleNamespace(**ns)
import e5_runner as W
os.environ.update(SUMO_HOME=F.SUMO_HOME,PATH=F.PATH,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
os.chdir(r/'runtime')
name='MAIN_PID.json' if {shard}==0 else 'FIX_REMAINDER_PID.json'
d=json.loads((op/name).read_text()); proc=pathlib.Path('/proc')/str(d['pid'])
if proc.exists():
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split(); assert stat[19]!=d['start_ticks'] or stat[0]=='Z','owned predecessor live'
out=r/'main/e3long_coopoff'/('shard'+str({shard}))
rows=json.loads((out/'RESULTS.json').read_text()); charges=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
assert len(rows)=={114 if shard==0 else 108}; assert F.ledger_total(charges)==102600
base={{x['job']['seed']:x for x in json.loads((r/'main/e3long_baseline'/('shard'+str({shard}))/'RESULTS.json').read_text()) if x['job']['arm']=='native'}}
on={{x['job']['seed']:x for x in json.loads((r/'main/e3long_policy'/('shard'+str({shard}))/'RESULTS.json').read_text())}}
for row in rows:
 job=row['job']; F.verify_row(row,job,base[job['seed']],on[job['seed']]); assert sum(x['id']==F.attempt(job) for x in charges)==1
 path=out/('retry_jobs' if job['id'] in F.FAILED else 'jobs')/job['id']/'result.json'; assert json.loads(path.read_text())==row
missing=sorted(set(base)-{{x['job']['seed'] for x in rows}}); assert missing=={[] if shard==0 else SEEDS!r}
inputs=json.loads((r/'inputs/coopoff_cycle1/INPUTS.json').read_text()); W.validate(inputs,json.loads((r/'inputs/coopoff_cycle1/GO_E5.json').read_text()))
files=json.loads((r/'runtime/SOURCE_MANIFEST.json').read_text()); files=files.get('files',files)
assert len(files)==543
for name,sha in files.items():
 assert not pathlib.Path(name).is_absolute() and '..' not in pathlib.Path(name).parts
 assert not any(s in name.lower() for s in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype'))
 assert hashlib.sha256((r/'runtime'/name).read_bytes()).hexdigest()==sha
gate=json.loads((r/'inputs/coopoff_cycle1/native_gate_result.json').read_text()); assert gate['passed'] and gate['checkpoint_sha256']==W.POLICY and gate['update']==180
binary=shutil.which('sumo'); assert binary
version=subprocess.run([binary,'--version'],text=True,capture_output=True,check=True).stdout
from traffic_shepherd.config import load_config
from traffic_shepherd.damping import CirclesController
from wms.control.policy import load_policy_fn
CirclesController(load_config()); load_policy_fn(r/'inputs/coopoff_cycle1/policy_latest.pt',device='cpu')
print(json.dumps(dict(status='PASS',no_simulation=True,n_results=len(rows),charged_sim_s=102600,missing_worlds=missing,source_files=543,priority=W.probe(),sumo=binary,version=version)))
''')
        print(box, json.dumps(proof), flush=True)
        files = {'E5DONE_AMENDMENT.md': amendment.encode(), 'DONE_AUTHORITY.json': json.dumps(authority, indent=2).encode(),
                 'DONE_PREFLIGHT.json': json.dumps(proof, indent=2).encode()}
        if box == 'cpuQ':
            files['done_runner.py'] = runner
        put(box, CPU, files)
    proof = remote('gpuA', f'''import pathlib,json
r=pathlib.Path({GPU!r}); d=json.loads((r/'FIX_PID.json').read_text()); p=pathlib.Path('/proc')/str(d['pid'])
if p.exists():
 s=(p/'stat').read_text().rsplit(')',1)[1].split(); assert s[19]!=d['start_ticks'] or s[0]=='Z'
assert json.loads((r/'FIX_STATUS.json').read_text())['status']=='WAIT_OWNER_BUDGET'
print(json.dumps(dict(status='PASS',predecessor_exited=True)))
''')
    print('gpuA', json.dumps(proof), flush=True)
    put('gpuA', GPU, {'E5DONE_AMENDMENT.md': amendment.encode(), 'DONE_AUTHORITY.json': json.dumps(authority, indent=2).encode(), 'remote_done.py': coordinator})


def put(box, root, files):
    payload = {k: base64.b64encode(v).decode() for k, v in files.items()}
    print(box, remote(box, f'''import pathlib,json,base64
r=pathlib.Path({root!r}); assert r.is_dir(); files={payload!r}
assert all(not (r/name).exists() for name in files)
for name,data in files.items():
 with (r/name).open('xb') as f: f.write(base64.b64decode(data))
print(json.dumps(dict(staged=list(files))))
'''), flush=True)


def status():
    for box, root in (('cpuQ', CPU), ('gpuA', GPU)):
        print(box, remote(box, f'''import pathlib,json
r=pathlib.Path({root!r}); names=('DONE_PID.json','DONE_STATUS.json','DONE_FAILURE.json','PAUSE.json')
print(json.dumps({{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}))
'''), flush=True)


if __name__ == '__main__':
    assert sys.argv[1] in ('stage', 'status')
    globals()[sys.argv[1]]()
