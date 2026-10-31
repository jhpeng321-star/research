"""Read-only own-dispatch progress; exact recorded worker PID metadata only."""
import argparse
import json
import subprocess
import time

from .audit import digest, safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out')
    a = p.parse_args()
    root = safe(a.root)
    launch_path = root/('RESUME.json' if (root/'RESUME.json').exists() else 'LAUNCH.json')
    launch = json.loads(launch_path.read_text())
    claims = [p for p in (root/'claims').glob('*.json') if p.name.count('.') == 1]
    workers = {}
    for path in claims:
        d = json.loads(path.read_text())
        key = path.stem
        remote = path.with_name(key+'.remote.json')
        if remote.exists() and not path.with_name(key+'.complete.json').exists():
            pid = json.loads(remote.read_text())['pid']
            workers.setdefault(d['host'], []).append(int(pid))
    meta = {}
    for host, ids in workers.items():
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
            'ps -p '+','.join(map(str, sorted(ids)))+' -o pid,ppid,etime,pcpu,rss,args'],
            capture_output=True, text=True, timeout=30)
        meta[host] = dict(returncode=r.returncode, own_process_metadata=r.stdout)
    completed = {p.name for p in (root/'tasks').glob('*.json')}
    original_completed = set()
    if launch.get('original'):
        original_completed = {p.name for p in (safe(launch['original'])/'tasks').glob('*.json')}
    errors = list((root/'claims').glob('*.error.json'))
    result = dict(schema='wmvalue_progress/1', time=time.time(),
        dispatcher_pid=launch['pid'], claimed=len(claims),
        completed=len(completed | original_completed), short_completed=len(completed),
        original_completed=len(original_completed), expected=320,
        failures=len(errors), unresolved_failures=sum(p.name.replace('.error.json', '.json')
            not in completed for p in errors), workers=meta,
        launch_sha256=digest(launch_path), active_launch_receipt=launch_path.name,
        complete=(root/'COMPLETE.json').exists(),
        frozen_test_written=safe(__file__).parent.joinpath('evidence/frozen_test.json').exists(),
        amended_test_written=safe(__file__).parent.joinpath('evidence/short_test.json').exists())
    if a.out:
        with safe(a.out).open('x') as stream:
            json.dump(result, stream, indent=1, allow_nan=False)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
