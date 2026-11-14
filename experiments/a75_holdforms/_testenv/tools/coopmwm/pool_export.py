"""Receipt-only export of full pools ALREADY embedded in completed GPU forecasts.

Standalone stdlib sidecar: never imports the WM, submits requests, or edits a
forecast. This preserves immutable running search sources and in-flight work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def immutable(path, value):
    data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if path.exists():
        if json.loads(path.read_bytes())!=value:raise ValueError('conflicting export '+path.name)
        return False
    temporary=path.with_name(path.name+'.publishing')
    with temporary.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    return True


def export_one(root, path):
    root=Path(root);path=Path(path)
    if path.parent!=root/'forecasts':raise ValueError('completed forecasts only')
    before=path.read_bytes();payload=json.loads(before)
    key=payload['key']
    if len(key)!=64 or any(c not in '0123456789abcdef' for c in key) or path.name!=key+'.json':
        raise ValueError('forecast identity mismatch')
    result=payload['result'];rec=result.get('proposer_receipt')
    if rec is None:
        if 'candidates_searched' in result:raise ValueError('scored search lacks full pool')
        return None  # Random-arm centres and diagnostic calls are never proposer data.
    if rec['schema']!='coopmwm.plan_proposer.receipt/1' or rec['arm']!='B':
        raise ValueError('only registered B search receipts')
    screen=sum(c['stage']=='screen' for c in rec['candidates'])
    if screen!=result['candidates_searched']:raise ValueError('incomplete screened pool')
    output=root/'proposer_receipts';output.mkdir(exist_ok=True)
    audits=root/'proposer_export_audits';audits.mkdir(exist_ok=True)
    created=immutable(output/path.name,rec)
    if path.read_bytes()!=before:raise ValueError('completed forecast changed')
    audit=dict(key=key,forecast_sha256=sha(before),export_sha256=sha((output/path.name).read_bytes()),
        screen_candidates=screen,candidate_records=len(rec['candidates']),accepted=result['accepted'],
        search_output_byte_identical=True,new_wm_calls=0,new_sumo_sim_s=0)
    immutable(audits/path.name,audit)
    return dict(audit,created=created)


def scan(root):
    root=Path(root)
    if not root.is_dir() or not (root/'forecasts').is_dir():raise ValueError('existing GPU root required')
    rows=[]
    for p in sorted((root/'forecasts').glob('*.json')):
        try:v=export_one(root,p)
        except json.JSONDecodeError:continue  # Producer may still be flushing its completed JSON.
        if v is not None:rows.append(v)
    return rows


def run(root, deadline):
    import fcntl
    root=Path(root)
    with (root/'PROPOSER_EXPORT.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),deadline_unix=deadline)
        immutable(root/'PROPOSER_EXPORT_OWNER.json',owner)
        while time.time()<deadline:
            rows=scan(root)
            for row in rows:
                if row['created']:print(json.dumps(row),flush=True)
            if (root/'SPOOL_FAILED.json').exists() or (root/'SPOOL_STOP.json').exists():return
            time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--deadline',type=float)
    a=p.parse_args()
    if a.deadline:run(a.root,a.deadline)
    else:print(json.dumps(scan(a.root)))
