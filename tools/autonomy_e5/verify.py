"""Read-only polling, then write owned first-batch verification receipts."""
import json
import subprocess
import time


def remote(host, source):
    python = '/home/ubuntu/miniconda3/bin/python' if host.startswith('cpu') else '/root/miniconda3/bin/python'
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
                             python + ' -B -'], input=source, text=True, capture_output=True,
                            encoding='utf-8', timeout=120)
    if result.returncode:
        raise RuntimeError(f'{host}: {result.stderr} {result.stdout}')
    return json.loads(result.stdout.splitlines()[-1])


CPU_SOURCE = r'''
import pathlib,json,os,time,hashlib
r=pathlib.Path('/home/ubuntu/e3long'); op=r/'operations_e5'; shard=SHARD
out=r/'main/e3long_coopoff'/('shard'+str(shard))
state=json.loads((op/'RUNNER_STATUS.json').read_text()) if (op/'RUNNER_STATUS.json').exists() else {}
assert state.get('status')!='failed',state
rows=json.loads((out/'RESULTS.json').read_text()) if (out/'RESULTS.json').exists() else []
main=op/'MAIN_PID.json'
if not main.exists() or len(rows)<(7 if shard==0 else 6):
 print(json.dumps(dict(ready=False,status=state,n_results=len(rows))))
else:
 receipt=json.loads(main.read_text()); proc=pathlib.Path('/proc')/str(receipt['pid'])
 if state['status']!='complete':
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
  assert stat[19]==receipt['start_ticks'] and int(stat[3])==receipt['sid']==receipt['pid'] and int(stat[1])==1
  assert os.readlink(proc/'fd/0')=='/dev/null'
 baseline=json.loads((r/'main/e3long_baseline'/('shard'+str(shard))/'RESULTS.json').read_text())
 on=json.loads((r/'main/e3long_policy'/('shard'+str(shard))/'RESULTS.json').read_text())
 hashes={x['job']['seed']:x['metrics']['initial_state_sha256'] for x in baseline}
 on_hashes={x['job']['seed']:x['metrics']['initial_state_sha256'] for x in on}
 ledger=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
 assert len(ledger)==len({x['id'] for x in ledger}) and sum(x['sim_s'] for x in ledger)<=102600
 assert all(x['sim_s']==900 for x in ledger)
 for row in rows:
  seed=row['job']['seed']; assert 973000<=seed<=973227 and row['ok'] and row['job']['arm']=='ours'
  assert row['metrics']['initial_state_sha256']==hashes[seed]==on_hashes[seed]
  assert set(row['metrics']['windows'])=={'120','300','600'}
  for h in (120,300,600):
   w=row['metrics']['windows'][str(h)]
   assert w['t0_s']==300 and w['t_end_s']==300+h and not w['dropped_vehicle_steps']
   assert abs(w['team_delay_vs']-w['road_delay_vs']-w['source_wait_vs'])<1e-3
   assert abs(w['team_delay_vs']-w['mainline_delay_vs']-w['ramp_delay_vs'])<1e-3
 if shard==0:
  assert sum(x['id']=='s973000_ours' for x in ledger)==1
 canary=json.loads((op/'CANARY_VERIFIED.json').read_text()); assert canary['passed']
 pause=json.loads((op/'PAUSE.json').read_text())
 proof=dict(ready=True,t=time.time(),pid=receipt,n_results=len(rows),charged_sim_s=sum(x['sim_s'] for x in ledger),
  canary=canary,CRN_clocks_conservation='PASS',pause=pause,
  max_world_wall_s=max(x['wall_s'] for x in rows),
  first_result_files={str(pathlib.Path('shard'+str(shard))/'jobs'/x['job']['id']/'result.json'):
   hashlib.sha256((out/'jobs'/x['job']['id']/'result.json').read_bytes()).hexdigest() for x in rows[:7 if shard==0 else 6]})
 (op/'FIRST_BATCH_VERIFIED.json').write_text(json.dumps(proof,indent=2)); print(json.dumps(proof))
'''


def verify():
    deadline = time.monotonic() + 900
    proofs = {}
    while time.monotonic() < deadline:
        coordinator = remote('gpuA', "import pathlib,json; p=pathlib.Path('/root/autodl-tmp/c4/restart20261026/e5/STATUS.json'); print(json.dumps(json.loads(p.read_text()) if p.exists() else {}))")
        assert coordinator.get('status') != 'WAIT_OWNER', coordinator
        for shard, box in enumerate(('cpuP', 'cpuQ')):
            if box not in proofs:
                proof = remote(box, CPU_SOURCE.replace('SHARD', str(shard)))
                print(box, json.dumps(proof), flush=True)
                if proof['ready']:
                    proofs[box] = proof
        if len(proofs) == 2:
            break
        time.sleep(20)
    assert len(proofs) == 2, 'first batch not ready within15min; inspect status before handoff'
    # Give the independent coordinator a mirror turn; inspect only its owned PID.
    time.sleep(20)
    result = remote('gpuA', f'''
import pathlib,json,os,hashlib,time
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026/e5')
receipt=json.loads((r/'PID.json').read_text()); proc=pathlib.Path('/proc')/str(receipt['pid'])
state=json.loads((r/'STATUS.json').read_text()); assert state['status'] in ('RUNNING','COMPLETE'),state
if state['status']=='RUNNING':
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 assert stat[19]==receipt['start_ticks'] and int(stat[3])==receipt['sid']==receipt['pid'] and int(stat[1])==1
 assert os.readlink(proc/'fd/0')=='/dev/null'
proofs={proofs!r}
for box,proof in proofs.items():
 for name,sha in proof['first_result_files'].items():
  assert hashlib.sha256((r/box/name).read_bytes()).hexdigest()==sha,(box,name)
doc=dict(status='FIRST_BATCH_VERIFIED',t=time.time(),coordinator=receipt,boxes=proofs,mirror_sha256='PASS')
(r/'FIRST_BATCH_VERIFIED.json').write_text(json.dumps(doc,indent=2)); print(json.dumps(doc))
''')
    print('VERIFIED', json.dumps(result), flush=True)


if __name__ == '__main__':
    verify()
