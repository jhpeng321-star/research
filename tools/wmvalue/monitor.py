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
    launch = json.loads((root/'LAUNCH.json').read_text())
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
    result = dict(schema='wmvalue_progress/1', time=time.time(),
        dispatcher_pid=launch['pid'], claimed=len(claims),
        completed=len(list((root/'tasks').glob('*.json'))), expected=320,
        failures=len(list((root/'claims').glob('*.error.json'))), workers=meta,
        launch_sha256=digest(root/'LAUNCH.json'),
        complete=(root/'COMPLETE.json').exists(),
        frozen_test_written=safe(__file__).parent.joinpath('evidence/frozen_test.json').exists())
    if a.out:
        with safe(a.out).open('x') as stream:
            json.dump(result, stream, indent=1, allow_nan=False)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
