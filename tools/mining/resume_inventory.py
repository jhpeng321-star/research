"""Read-only recovery of named mining owners, inputs, ledgers and stop gates."""
import hashlib
import json
import os
from pathlib import Path
import time

gpu = os.geteuid() == 0
root = Path('/root/mining20261106') if gpu else Path('/home/ubuntu/coopmwm20261106_s_mining')
out = dict(unix=time.time(), host='gpuA' if gpu else 'cpuV', root=str(root))
out['files'] = [p.name for p in root.iterdir()] if root.is_dir() else []
out['receipts'] = {}
names = ('CHARGES.jsonl', 'PARITY_PASS.json', 'BUDGET_ADMISSION.json')
if not gpu:
    names += ()
for name in names:
    p = root / name
    if p.is_file():
        out['receipts'][name] = p.read_text()
if gpu:
    worker = root / 'wm_worker01'
    out['worker_files'] = [p.name for p in worker.iterdir()]
    out['queries'] = [p.name for p in (worker / 'queries').glob('*.json')]
    out['results'] = [p.name for p in (worker / 'results').iterdir()]
    for name in ('MINING_WM_DISPATCH.json', 'MINING_WM_OWNER.json', 'MINING_WM_EXIT.json',
                 'MINING_WM_FAULT.json', 'GPU_MODEL_READY.json', 'FRESH_QUALIFICATION.json'):
        p = worker / name
        if p.is_file():
            v = json.loads(p.read_bytes()); out['receipts'][name] = v
            if name == 'MINING_WM_OWNER.json':
                proc = Path('/proc') / str(v['pid'])
                out['worker_identity'] = dict(alive=proc.exists())
                if proc.exists():
                    out['worker_identity'].update(stat=(proc/'stat').read_text(),
                        argv=(proc/'cmdline').read_bytes().decode().replace('\x00', ' '),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    out['u0_present'] = (root/'frozen_u0.pt').is_file()
    out['input_dirs'] = [str(p) for p in Path('/home/ubuntu/cpuv_parity_20261105').iterdir()]
    for p in (worker/'results').glob('*.result.json'):
        out['receipts'][p.name] = json.loads(p.read_bytes())
else:
    for rel in ('coopmwm20261105_clean2/STOPPED_BY_OWNER.json',
                'coopmwm20261105_clean2/CHAIN_STATUS.json',
                'coopmwm20261106_clean3/LAUNCHED.json',
                'coopmwm20261106_clean3/CONFIG.json'):
        p = Path('/home/ubuntu')/rel
        out['receipts'][rel] = json.loads(p.read_bytes()) if p.is_file() else None
    out['cores'] = len(os.sched_getaffinity(0))
out['free_bytes'] = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
print(json.dumps(out, sort_keys=True))
