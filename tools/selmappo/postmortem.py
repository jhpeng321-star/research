"""Freeze the exact production-transport stop; never close unknown debits or retry."""
from .launch import save_tag
from .remote import run


def record():
    result = run('cpuQ', '''
import hashlib,json,resource
from pathlib import Path
root=Path.home()/'selmappo20261031/build_b3d13ccfae77de71'
out=root/'training';assert root.is_dir() and out.is_dir()
dispatch=json.loads((root/'DISPATCH.json').read_text())
assert dispatch['pid']==503239 and not (Path('/proc')/'503239').exists()
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  except (OSError,ProcessLookupError):continue
  assert str(root) not in cmd or '-c import base64' in cmd,'owned descendant still live'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
rows=[json.loads(s) for s in (out/'ledger.jsonl').read_text().splitlines()]
starts=[r for r in rows if r['event']=='START'];ends=[r for r in rows if r['event']=='END']
assert sum(r['reserved_sim_s'] for r in starts)==38700
assert len(starts)==43 and len(ends)==34
assert not (out/'training.jsonl').exists() and not (out/'checkpoint_001.pt').exists()
log=(root/'training.log').read_text()
failures=(out/'failure.jsonl').read_text()
assert 'received 0 items of ancdata' in failures and 'BrokenProcessPool' in failures
assert 'incomplete production wave' in log
actuals=[];unknown=[];successful=[]
for s in starts:
 seed=int(s['id'].split(':')[1])
 if seed>=18007998:continue
 path=out/str(seed)/'actual.jsonl'
 if path.exists():
  records=[json.loads(line) for line in path.read_text().splitlines()];assert len(records)==1
  actuals.append(dict(seed=seed,actual_sim_s=records[0]['actual_sim_s'],receipt_sha256=sha(path)))
 else:unknown.append(seed)
 result=out/str(seed)/'result.jsonl'
 if result.exists():successful.append(dict(seed=seed,receipt_sha256=sha(result)))
assert len(actuals)==len(successful)==33 and len(unknown)==8
record=dict(status='STOPPED_AT_PRODUCTION_TRANSPORT',root=str(root),own_pid=503239,
 own_pid_exited=True,own_descendants_remaining=0,charged_sim_s=38700,
 known_actual_sim_s=900+sum(a['actual_sim_s'] for a in actuals),
 actual_upper_sim_s=900+sum(a['actual_sim_s'] for a in actuals)+900*len(unknown),
 actual_unknown_worlds=unknown,successful_production_episodes=33,worker_actual_receipts=actuals,
 successful_result_receipts=successful,ledger_open_reservations=9,
 ledger_sha256=sha(out/'ledger.jsonl'),log_sha256=sha(root/'training.log'),
 failure_sha256=sha(out/'failure.jsonl'),initial_checkpoint_sha256=sha(out/'checkpoint_000.pt'),
 source_binding=dispatch['binding'],training_updates=0,final_checkpoint=False,evaluation_episodes=0,
 mechanics_replayed=False,retry_authorized=False,automatic_launch_armed=False,
 error='PyTorch tensor IPC storage FD exhaustion / received 0 items of ancdata',
 current_host_nofile_limits=resource.getrlimit(resource.RLIMIT_NOFILE),
 note='33 completed result/actual receipts; first32 supervisor-closed, ninth unresolved has full receipt; eight killed workers lack terminal actual, retain full900 each. Original ledger unchanged.')
with (root/'STOP.json').open('x') as f:json.dump(record,f,indent=2)
print('STOP_RECEIPT',json.dumps(record))
''')
    save_tag(result.stdout, 'STOP_RECEIPT', 'STOP_build_b3d13ccfae77de71.json')


if __name__ == '__main__':
    record()
