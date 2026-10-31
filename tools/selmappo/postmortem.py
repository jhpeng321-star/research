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


def history_record():
    result = run('cpuQ', '''
import hashlib,json
from pathlib import Path
root=Path.home()/'selmappo20261031/build_3351504cece84e4b'
out=root/'training';assert root.is_dir() and out.is_dir()
dispatch=json.loads((root/'DISPATCH.json').read_text())
assert dispatch['pid']==606965 and not (Path('/proc')/'606965').exists()
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
  except (OSError,ProcessLookupError):continue
  assert str(root) not in cmd or '-c import base64' in cmd,'owned descendant remains'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ledger=out/'ledger.jsonl';rows=[json.loads(s) for s in ledger.read_text().splitlines()]
assert len(rows)==6 and rows[0]['event']=='HISTORY' and rows[0]['charged_sim_s']==38700
assert [r['event'] for r in rows[1:]]==['START']*5
assert {r['id'] for r in rows[1:]}=={f'fleet:{s}' for s in range(18000041,18000046)}
assert all(r['reserved_sim_s']==900 for r in rows[1:])
assert sha(out/'inherited_ledger.jsonl')=='497e0c292a1acb757f4bd733fa4a60b9ad7e2853e6b16d16e8fe50c6c570ec88'
failures=[json.loads(s) for s in (out/'failure.jsonl').read_text().splitlines()]
assert len(failures)==5
assert {r['world']['seed'] for r in failures}==set(range(18000041,18000046))
for row in failures:
 assert row['actual'] is None and not row['ok']
 assert "KeyError: 'id'" in row['error'] and 'storage.py", line 90, in <listcomp>' in row['error']
 assert 'runtime.py", line 135, in __init__' in row['error']
for seed in range(18000041,18000046):
 assert not list((out/str(seed)).iterdir()),'unexpected physics/trajectory artifact'
assert not (out/'training.jsonl').exists()
assert [p.name for p in out.glob('checkpoint_*.pt')]==['checkpoint_000.pt']
assert (out/'STOP_REQUESTED').exists()
record=dict(status='STOPPED_BEFORE_PHYSICS_HISTORY_READER',root=str(root),own_pid=606965,
 own_pid_exited=True,own_descendants_remaining=0,charged_sim_s=43200,
 inherited_charge_sim_s=38700,new_charge_sim_s=4500,new_actual_sim_s=0,
 known_actual_sim_s=30600,actual_upper_sim_s=37800,new_open_reservations=5,
 failed_seeds=list(range(18000041,18000046)),training_updates=0,final_checkpoint=False,
 ledger_sha256=sha(ledger),inherited_ledger_sha256=sha(out/'inherited_ledger.jsonl'),
 log_sha256=sha(root/'training.log'),failure_sha256=sha(out/'failure.jsonl'),
 initial_checkpoint_sha256=sha(out/'checkpoint_000.pt'),source_binding=dispatch['binding'],
 failure_before_env_constructor=True,retry_authorized=False,automatic_launch_armed=False,
 error='PrechargedEpisode reserved-row scan assumed id on immutable HISTORY debit',
 note='All five tracebacks are at runtime budget.reserve before CavSumoEnv construction/reset; empty episode roots. Null worker actual is retained; new actual0 is independently established by the failure location. No invented END rows/refund/replay.')
with (root/'STOP.json').open('x') as stream:json.dump(record,stream,indent=2)
print('STOP_RECEIPT',json.dumps(record))
''')
    save_tag(result.stdout, 'STOP_RECEIPT', 'STOP_build_3351504cece84e4b.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('transport', 'history'), default='transport', nargs='?')
    if parser.parse_args().kind == 'history':
        history_record()
    else:
        record()
