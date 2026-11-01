"""Read-only fresh-session receipts for exact owned roots and resource accounting."""
import json
import time

from tools.c1switch.storage import write
from .operations import EVIDENCE, ssh
from .watch import collect, local_process


CPUQ = '''import json,os,shutil,time
from pathlib import Path
from tools.c1switch.design import FORBIDDEN
out=dict(unix=time.time(),uptime_s=float(Path('/proc/uptime').read_text().split()[0]),
         disk=shutil.disk_usage('/home/ubuntu')._asdict(),processes={},own_outputs={})
for pid in (649272,645235,606968):
 p=Path('/proc',str(pid))
 if p.exists():
  fields=(p/'stat').read_text().split()
  out['processes'][str(pid)]=dict(state=fields[2],start_ticks=int(fields[21]),uid=p.stat().st_uid,
      command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace'))
 else:out['processes'][str(pid)]=dict(state='EXITED')
for item in os.scandir('/home/ubuntu'):
 if not item.name.startswith('c1switch2_') or not item.is_dir(follow_symlinks=False):continue
 stack=[item.path];size=count=0
 while stack:
  current=stack.pop()
  for entry in os.scandir(current):
   if entry.is_symlink() or FORBIDDEN.search(entry.path):continue
   if entry.is_dir(follow_symlinks=False):stack.append(entry.path)
   elif entry.is_file(follow_symlinks=False):
    size+=entry.stat(follow_symlinks=False).st_size;count+=1
 out['own_outputs'][item.path]=dict(bytes=size,files=count)
print(json.dumps(out))
'''


def current_status(snapshot, receipt):
    """Missing hosts remain unknown; closed actual excludes open-job physics."""
    coverage = all(host in snapshot['hosts'] for host in ('cpuQ', 'gpuR', 'gpuA'))
    cpus = {host: data['cpu'] for host, data in snapshot['hosts'].items()}
    gpu = {host: data['gpu'] for host, data in snapshot['hosts'].items()}
    charged = 4680 + sum(row.get('charged_sim_s', 0) for rows in cpus.values() for row in rows.values())
    closed = 4680 + sum(row.get('closed_actual_sim_s', 0) for rows in cpus.values() for row in rows.values())
    allocations = [row for host, rows in gpu.items() if host in ('gpuA', 'gpuR')
                   for row in rows.values() if row.get('started') or row.get('finished')]
    terminals = [row.get('finished') for row in allocations]
    gpu_complete = coverage and len(terminals) >= 4 and all(terminals)
    return dict(protocol='S-C1SWITCH2/v2', verified_unix=snapshot['unix'],
        status='CPU_COMPLETE_WM_TERMINAL_REGISTERED_PANEL_PARTIAL_OWNED_SCOPES_SUPERVISED',
        receipt=str(receipt), base_sim_s=468380, conditional_sim_s=548180,
        base_gpu_hours_cap=19, conditional_gpu_hours_cap=35,
        design_sha256='95aa84cf801aea773a0c7f9a208396ed4898e5cacdf8b6177940c7d9e1f5a273',
        panel_sha256='7c780c25237dde73a657937ab577b8e48f71766817f4363f3298982e90144827',
        prediction_objective='on_road_fleet_delay', evaluation_objective='full_team_delay_fleet_plus_source',
        host_coverage_complete=coverage, charged_sim_s_at_latest_receipts=charged if coverage else None,
        charged_verified_lower_bound_sim_s=charged, closed_actual_verified_lower_bound_sim_s=closed,
        closed_actual_sim_s_at_latest_receipts=closed if coverage else None,
        registered_base_remaining_sim_s=468380-charged if coverage else None,
        actual_note='Closed actual excludes currently open jobs; historical charges are never refunded.',
        new_gpu_s=sum(row['elapsed_s'] for row in terminals) if gpu_complete else None,
        finalized_gpu_s=sum(row['elapsed_s'] for row in terminals if row),
        gpu_accounting_note='Every original and continuation allocation retained; exact total unknown while any allocation is live. Finalized monotonic receipts include loading/parity/idle.',
        canary=dict(state='COMPLETE', charged_sim_s=4680, closed_actual_sim_s=4680, never_redispatch=True),
        cpu=cpus, gpu=gpu, alerts=snapshot['alerts'],
        cpuQ_resources=snapshot.get('cpuQ_resources'), observer=snapshot['observer'],
        gpu_hold_owner2014=snapshot.get('gpu_hold_owner2014', False),
        bb_startup_stop=snapshot['bb_startup_stop'],
        final_checkpoint_receipts=snapshot['final_checkpoint_receipts'],
        optional_wm_phys='OFF: MAIN11:48 final report says registered switching rule FAIL; keep WM2',
        efficacy_status='Primary full paired WM-vs-RL panel incomplete; no primary efficacy CI')


def run():
    snapshot = collect()
    # Install only the already-frozen source path to obtain the path-pruning regex.
    snapshot['cpuQ_resources'] = ssh('cpuQ',
        "import sys\nsys.path.insert(0,'/home/ubuntu/c1switch2_run_292bfffe94b13ed8/pytools')\n" + CPUQ,
        timeout=45)
    observer = EVIDENCE/'WATCH_OWNER2014/DISPATCH.json'
    if not observer.exists():
        observer = EVIDENCE/'WATCH_MAIN1214/DISPATCH.json'
    snapshot['observer'] = local_process(json.loads(observer.read_text())['pid'] if observer.exists() else 25348)
    snapshot['bb_startup_stop'] = json.loads((EVIDENCE/'BB_RELOCATION_STOP.json').read_text())
    snapshot['read_only'] = True
    from .gpuhold import active
    snapshot['gpu_hold_owner2014'] = active()
    path = EVIDENCE/('RESUME_RECONCILIATION_'+str(int(time.time()))+'.json')
    write(path, snapshot)
    write(EVIDENCE/'STATUS_CURRENT.json', current_status(snapshot, path))
    resources = snapshot['cpuQ_resources']
    print(json.dumps(dict(receipt=str(path), observer=snapshot['observer'],
        uptime_days=resources['uptime_s']/86400, disk=resources['disk'],
        own_output_bytes=sum(row['bytes'] for row in resources['own_outputs'].values()),
        processes=resources['processes'], bb_stop=snapshot['bb_startup_stop'],
        alerts=snapshot['alerts']), indent=2))


if __name__ == '__main__':
    run()
