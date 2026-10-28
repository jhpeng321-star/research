"""STOP only this task's recorded processes; no signals or name matching."""
from pathlib import Path
import json
import subprocess
import time

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
assert base.is_dir()
(base/'STOP').touch()
pids, pools = set(), set()
for pattern in ('out/*/future_workers/*/PIDS.json',
                'qualification*_tmp/**/future_workers/*/PIDS.json',
                'failed*/**/future_workers/*/PIDS.json'):
    for path in base.glob(pattern):
        pids.update(json.loads(path.read_text()))
        pools.add(path.parent)
        path.with_name('STOP').touch()
for pattern in ('out/*/PID', 'failed*/**/PID'):
    for path in base.glob(pattern):
        pids.add(int(path.read_text()))
for filename in ('COORDINATOR_PID', 'FOLLOWUP_PID'):
    path = base/filename
    if path.exists():
        pids.add(int(path.read_text()))
path = base/'QUALIFICATION_PID.json'
if path.exists():
    pids.add(json.loads(path.read_text())['pid'])
for path in (base/'out').glob('*'):
    (path/'STOP').touch()


def running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


deadline = time.monotonic()+20
while any(running(pid) for pid in pids) and time.monotonic() < deadline:
    time.sleep(1)
live = sorted(pid for pid in pids if running(pid))
text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader,nounits'], text=True)
gpu = []
for line in text.strip().splitlines():
    pid, memory = (int(x.strip()) for x in line.split(','))
    if pid in pids:
        gpu.append(dict(pid=pid, memory_mib=memory))
receipt = dict(t=time.time(), owned_pids=sorted(pids), stopped_pools=len(pools),
               live_owned_pids=live, owned_gpu_allocations=gpu, stopped=not live and not gpu)
(base/'STOP_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
raise SystemExit(0 if receipt['stopped'] else 2)
