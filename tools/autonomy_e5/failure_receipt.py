"""Preserve first-batch failure and pause only E5 dispatch through its own heartbeat."""
import json
from tools.autonomy_e5.verify import CPU_SOURCE, remote


if __name__ == '__main__':
    proofs = {}
    for shard, box in enumerate(('cpuP', 'cpuQ')):
        proofs[box] = remote(box, f'''
import pathlib,json,os,time,shutil
r=pathlib.Path('/home/ubuntu/e3long'); op=r/'operations_e5'; out=r/'main/e3long_coopoff/shard{shard}'
heartbeat=op/'MIRROR_HEARTBEAT.json'
if heartbeat.exists(): os.utime(heartbeat,(time.time()-600,time.time()-600))
ledger=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
rows=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else []
state=json.loads((op/'RUNNER_STATUS.json').read_text())
pid=json.loads((op/'MAIN_PID.json').read_text())
failures=[]
for charge in ledger:
 p=out/'jobs'/charge['id']/'failure.json'
 if p.exists(): failures.append(json.loads(p.read_text()))
assert len(ledger)==len({{x['id'] for x in ledger}})
doc=dict(t=time.time(),pid=pid,state=state,charged_sim_s=sum(x['sim_s'] for x in ledger),
 n_charges=len(ledger),n_results=len(rows),n_failures=len(failures),failures=failures,
 idle_boundary=len(ledger)==len(rows)+len(failures),
 pause=json.loads((op/'PAUSE.json').read_text()),
 detached_PATH=os.environ.get('PATH'),sumo_in_detached_PATH=shutil.which('sumo'),
 prepared_sumo_exists=(pathlib.Path('/home/ubuntu/miniconda3/bin/sumo').exists()))
(op/'FAILURE_RECEIPT.json').write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')
        print(box, json.dumps({k: v for k, v in proofs[box].items() if k != 'failures'}), flush=True)
    assert all(p['idle_boundary'] for p in proofs.values()), 'wait for owned in-flight batch before handoff'
    assert proofs['cpuQ']['n_failures'] == 6
    first_batch = remote('cpuP', CPU_SOURCE.replace('SHARD', '0'))
    assert first_batch['ready'] and first_batch['pause']['stale_mirror_pause']
    # No heartbeat is refreshed: the independent failed coordinator stays stopped.
    result = remote('gpuA', f'''
import pathlib,json,subprocess,time
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026/e5')
for box in ('cpuP','cpuQ'):
 dest=r/box; dest.mkdir(exist_ok=True)
 subprocess.run(['rsync','-az','--timeout=30','-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
  '--include','*/','--include','*.json','--include','*.jsonl','--exclude','*',
  box+':/home/ubuntu/e3long/main/e3long_coopoff/',str(dest)+'/'],check=True,timeout=90)
proofs={proofs!r}
first_batch={first_batch!r}
for name,sha in first_batch['first_result_files'].items():
 import hashlib
 assert hashlib.sha256((r/'cpuP'/name).read_bytes()).hexdigest()==sha
doc=dict(status='WAIT_OWNER_NO_SIMULATION_RETRY',t=time.time(),boxes=proofs,
 charged_sim_s=sum(p['charged_sim_s'] for p in proofs.values()),
 reason='cpuQ detached PATH omitted SUMO; six pre-environment failures fully charged',
 results_unavailable=True,cpuP_first_batch_CRN_clocks_conservation_mirror='PASS')
(r/'FAILURE_RECEIPT.json').write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')
    print('PRESERVED', json.dumps({k: v for k, v in result.items() if k != 'boxes'}), flush=True)
