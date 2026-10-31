"""Read-only task receipt snapshot; never dispatch, retry, or touch simulator files."""
import argparse

from .launch import save_tag
from .remote import run


def snapshot(build):
    if not build.startswith('build_') or not build.removeprefix('build_').isalnum():
        raise ValueError('exact task build name required')
    result = run('cpuQ', f'''
import datetime,hashlib,json,os,time,statistics
from pathlib import Path
root=Path.home()/'selmappo20261031'/{build!r};assert root.is_dir()
out=root/'training';assert out.is_dir()
dispatch=json.loads((root/'DISPATCH.json').read_text())
def rows(path):
 if not path.exists():return []
 raw=path.read_bytes()
 # A live append may be between write and newline; capture complete rows only.
 return [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(bytes([10]))]
ledger=rows(out/'ledger.jsonl');starts=[r for r in ledger if r['event']=='START']
history=[r for r in ledger if r['event']=='HISTORY'];assert len(history)<=1
historical_charge=history[0]['charged_sim_s'] if history else 1800
historical_low=history[0]['known_actual_sim_s'] if history else 900
historical_high=history[0]['actual_upper_sim_s'] if history else 900
ends=[r for r in ledger if r['event']=='END'];ended={{r['id'] for r in ends}}
production=[r for r in starts if int(r['id'].split(':')[1])<18007998]
closed=[r for r in production if r['id'] in ended]
results=[]
for r in production:
 seed=int(r['id'].split(':')[1]);data=rows(out/str(seed)/'result.jsonl')
 if data:
  assert len(data)==1 and not data[0]['smoke'] and not data[0]['evaluation']
  results.append(data[0])
updates=rows(out/'training.jsonl');status=rows(out/'status.jsonl');failures=rows(out/'failure.jsonl')
worker_receipts=[]
for r in production:
 seed=int(r['id'].split(':')[1]);actual=rows(out/str(seed)/'actual.jsonl');result=rows(out/str(seed)/'result.jsonl')
 assert len(actual)<=1 and len(result)<=1
 worker_receipts.append(dict(seed=seed,actual=actual[0]['actual_sim_s'] if actual else None,
                             result_complete=bool(result),supervisor_closed=r['id'] in ended))
own_processes=[]
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:
   cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
   if str(root) in cmd and '-c import base64' not in cmd:own_processes.append(dict(pid=int(proc.name),command=cmd))
  except (OSError,ProcessLookupError):pass
checkpoints=[]
for path in sorted(out.glob('checkpoint_*.pt')):
 index=int(path.stem.split('_')[-1])
 # A completed update row precedes saving. A matching later checkpoint or
 # final trainer completion attests write completion; do not publish live partials.
 stable=(out/f'checkpoint_{{index+1:03d}}.pt').exists() or index==0
 if index==169:stable=any(r['status'].startswith('TRAIN_COMPLETE') for r in status)
 if stable:checkpoints.append(dict(iteration=index,path=str(path),size=path.stat().st_size,
     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
pid=dispatch['pid'];alive=(Path('/proc')/str(pid)).exists()
record=dict(schema='selmappo.progress/1',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 root=str(root),source_binding=dispatch['binding'],own_pid=pid,own_pid_exists=alive,
 own_nice=os.getpriority(os.PRIO_PROCESS,pid) if alive else None,
 load_1m=os.getloadavg()[0],cores=os.cpu_count(),max_load_fraction=.8,
  charged_sim_s=sum(r['reserved_sim_s'] for r in starts)+sum(r['charged_sim_s'] for r in history),
 closed_actual_sim_s=sum(r['actual_sim_s'] for r in ends),
 outstanding_episodes=len(starts)-len(ends),outstanding_actual_not_yet_final=True,
  historical_charged_sim_s=historical_charge,historical_actual_sim_s_lower=historical_low,
  historical_actual_sim_s_upper=historical_high,
  inherited_unresolved_reservations=history[0]['unresolved_reservations'] if history else 0,
 production_reserved_episodes=len(production),production_closed_episodes=len(closed),
 production_completed_episodes=len(results),production_completed_sim_s=sum(r['actual_sim_s'] for r in results),
 completed_updates=len(updates),latest_update=updates[-1] if updates else None,
 initial_and_stable_checkpoints=checkpoints,latest_priority_status=status[-1] if status else None,
 failure_count=len(failures),failed_worlds=[r['world']['seed'] for r in failures],
 first_failure=failures[0] if failures else None,worker_receipts=worker_receipts,
 own_processes=own_processes,
  actual_sim_s_lower_bound=historical_low+sum(r['actual'] or 0 for r in worker_receipts),
  actual_sim_s_upper_bound=historical_high+sum(r['actual'] if r['actual'] is not None else 900 for r in worker_receipts),
 completed_result_totals={{key:sum(r[key] for r in results) for key in
   ('road_delay_vs','source_wait_vs','team_delay_vs','collisions','teleports','ttc_below_3_vs','hard_brake_vs')}},
 completed_episode_wall_s_mean=statistics.mean(r['initialization_warmup_wall_s']+r['control_wall_s'] for r in results) if results else None,
 prospective_local_evaluation_sim_s=14400,local_evaluation_started=False,
 final169_eligible=any(c['iteration']==169 for c in checkpoints),
 training_budget_upper_sim_s=7012800,mechanics_replayed=False)
print('PROGRESS_RECEIPT',json.dumps(record))
print('SNAPSHOT_NAME',str(time.time_ns()))
''')
    stamp = next(line.split()[1] for line in result.stdout.splitlines()
                 if line.startswith('SNAPSHOT_NAME '))
    save_tag(result.stdout, 'PROGRESS_RECEIPT', f'PROGRESS_{build}_{stamp}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    args = parser.parse_args()
    snapshot(args.build)
