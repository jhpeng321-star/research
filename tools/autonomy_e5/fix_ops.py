"""Sequential deployment and verification for the one authorized S-E5FIX repair."""
import base64
import hashlib
import json
from pathlib import Path
import time

from tools.autonomy_e5.verify import remote

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
GPU = '/root/autodl-tmp/c4/restart20261026/e5'
CPU = '/home/ubuntu/e3long/operations_e5'


def stage():
    full = REGISTRATION.read_text(encoding='utf-8')
    amendment = '## S-E5FIX'+full.split('## S-E5FIX', 1)[1]
    runner = (ROOT/'tools/autonomy_e5/fix_runner.py').read_bytes()
    coordinator = (ROOT/'tools/autonomy_e5/remote_fix.py').read_bytes()
    authority = dict(task='S-E5FIX', issued_by='main 2026-10-27 15:00 act-on-recommendation',
                     remaining_sim_s=182700, historical_charged_sim_s=22500,
                     cap_per_box=102600, total_cap_sim_s=205200,
                     amendment_sha256=hashlib.sha256(amendment.encode()).hexdigest(),
                     registration_sha256=hashlib.sha256(REGISTRATION.read_bytes()).hexdigest(),
                     fix_runner_sha256=hashlib.sha256(runner).hexdigest(),
                     coordinator_sha256=hashlib.sha256(coordinator).hexdigest())
    # Read-only proof of the exact adopted live cpuP, exited cpuQ and idle boundaries.
    for box, charge, count in (('cpuP', 17100, 19), ('cpuQ', 5400, 0)):
        proof = remote(box, f'''import pathlib,json,os,sys
r=pathlib.Path('/home/ubuntu/e3long'); op=r/'operations_e5'; shard={0 if box=='cpuP' else 1}
sys.path.insert(0,str(op)); import e5_runner as W
W.validate(json.loads((r/'inputs/coopoff_cycle1/INPUTS.json').read_text()),json.loads((r/'inputs/coopoff_cycle1/GO_E5.json').read_text()))
out=r/'main/e3long_coopoff'/('shard'+str(shard))
d=json.loads((op/'MAIN_PID.json').read_text()); proc=pathlib.Path('/proc')/str(d['pid'])
alive=False
if proc.exists():
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 alive=stat[19]==d['start_ticks'] and stat[0]!='Z'
 if alive:
  assert int(stat[3])==d['sid']==d['pid'] and int(stat[1])==1
  assert os.readlink(proc/'fd/0')=='/dev/null'
assert alive=={box=='cpuP'!r}
rows=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else []
ledger=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
assert len(rows)=={count} and sum(x['sim_s'] for x in ledger)=={charge}
assert len(ledger)==len(rows)+(0 if shard==0 else 6)
assert not (op/'FIX_AUTHORITY.json').exists()
print(json.dumps(dict(pid=d,alive=alive,charged_sim_s={charge},n_results={count})))
''')
        print(box, json.dumps(proof), flush=True)
    for box, root in (('cpuP', CPU), ('cpuQ', CPU), ('gpuA', GPU)):
        files = {'E5FIX_AMENDMENT.md': amendment.encode(),
                 'FIX_AUTHORITY.json': json.dumps(authority, indent=2).encode()}
        if box == 'cpuQ':
            files['fix_runner.py'] = runner
        if box == 'gpuA':
            files['remote_fix.py'] = coordinator
        payload = {k: base64.b64encode(v).decode() for k, v in files.items()}
        print(box, remote(box, f'''import pathlib,json,base64
r=pathlib.Path({root!r}); assert r.is_dir()
files={payload!r}
assert all(not (r/name).exists() for name in files)
for name,data in files.items():
 with (r/name).open('xb') as f: f.write(base64.b64decode(data))
print(json.dumps(dict(staged=list(files))))
'''), flush=True)


def preflight():
    receipt = remote('gpuA', f"import sys,json; sys.path.insert(0,{GPU!r}); import remote_fix as F; print(json.dumps(F.start('preflight')))")
    print('DETACHED_PREFLIGHT', json.dumps(receipt), flush=True)
    for _ in range(30):
        doc = remote('cpuQ', f'''import pathlib,json
r=pathlib.Path({CPU!r}); p=r/'FIX_PREFLIGHT.json'
print(json.dumps(json.loads(p.read_text()) if p.exists() else dict(waiting=True)))
''')
        if not doc.get('waiting'):
            assert doc['status'] == 'PASS' and doc['new_charged_sim_s'] == 0
            print(json.dumps(doc), flush=True)
            return
        time.sleep(5)
    raise RuntimeError('zero-charge detached preflight not ready; inspect its exact PID/log')


def verify():
    for _ in range(60):
        proof = remote('cpuQ', f'''import pathlib,json,sys,hashlib,os,time
op=pathlib.Path({CPU!r}); r=op.parent; sys.path.insert(0,str(op))
import fix_runner as F
out=r/'main/e3long_coopoff/shard1'
rows=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else []
state=json.loads((op/'FIX_RUNNER_STATUS.json').read_text()) if (op/'FIX_RUNNER_STATUS.json').exists() else {{}}
assert state.get('status')!='failed',state
if not set(F.FAILED)<={{x['job']['id'] for x in rows}}:
 print(json.dumps(dict(ready=False,n_results=len(rows),state=state)))
else:
 base={{x['job']['seed']:x for x in json.loads((r/'main/e3long_baseline/shard1/RESULTS.json').read_text()) if x['job']['arm']=='native'}}
 on={{x['job']['seed']:x for x in json.loads((r/'main/e3long_policy/shard1/RESULTS.json').read_text())}}
 ledger=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
 charged=F.ledger_total(ledger)
 files={{}}
 for row in rows:
  job=row['job']; F.verify_row(row,job,base[job['seed']],on[job['seed']])
  aid=F.attempt(job); assert sum(x['id']==aid for x in ledger)==1
  path=out/('retry_jobs' if job['id'] in F.FAILED else 'jobs')/job['id']/'result.json'
  assert json.loads(path.read_text())==row
  files[str(path.relative_to(out.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
 canary=json.loads((op/'FIX_CANARY_VERIFIED.json').read_text()); assert canary['passed']
 pid=json.loads((op/'FIX_REMAINDER_PID.json').read_text()); proc=pathlib.Path('/proc')/str(pid['pid'])
 if state['status'] not in ('complete','WAIT_OWNER_BUDGET'):
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  assert stat[19]==pid['start_ticks'] and int(stat[3])==pid['sid']==pid['pid'] and int(stat[1])==1
  assert os.readlink(proc/'fd/0')=='/dev/null'
 doc=dict(ready=True,t=time.time(),pid=pid,n_results=len(rows),charged_sim_s=charged,
  historical_failures_retained=6,authorized_repairs_verified=list(F.FAILED),canary=canary,
  CRN_clocks_conservation='PASS',result_files=files,max_world_wall_s=max(x['wall_s'] for x in rows),
  pause=json.loads((op/'PAUSE.json').read_text()))
 (op/'FIX_VERIFIED.json').write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')
        print(json.dumps({k: v for k, v in proof.items() if k != 'result_files'}), flush=True)
        if proof['ready']:
            break
        time.sleep(15)
    else:
        raise RuntimeError('cpuQ verified outputs not ready within15min')
    result = remote('gpuA', f'''import pathlib,json,hashlib,os,time
import sys; sys.path.insert(0,{GPU!r}); import remote_fix as F
r=pathlib.Path({GPU!r}); F.B.pull('cpuQ'); F.B.pull('cpuP')
proof={proof!r}
for name,sha in proof['result_files'].items():
 assert hashlib.sha256((r/'cpuQ'/name).read_bytes()).hexdigest()==sha,name
pid=json.loads((r/'FIX_PID.json').read_text()); state=json.loads((r/'FIX_STATUS.json').read_text())
assert state['status'] in ('RUNNING','WAIT_OWNER_BUDGET','COMPLETE'),state
if state['status']=='RUNNING':
 p=pathlib.Path('/proc')/str(pid['pid']); stat=(p/'stat').read_text().rsplit(')',1)[1].split()
 assert stat[19]==pid['start_ticks'] and int(stat[3])==pid['sid']==pid['pid'] and int(stat[1])==1
 assert os.readlink(p/'fd/0')=='/dev/null'
doc=dict(status='CPUQ_REPAIRS_VERIFIED',t=time.time(),coordinator=pid,cpuQ=proof,mirror_sha256='PASS',state=state)
(r/'FIX_VERIFIED.json').write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')
    print('VERIFIED', json.dumps(result), flush=True)


if __name__ == '__main__':
    import sys
    assert sys.argv[1] in ('stage', 'preflight', 'verify')
    globals()[sys.argv[1]]()
