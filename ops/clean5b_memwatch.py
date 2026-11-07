"""Log-only memory/disk watchdog (MAIN 2026-11-07): every 60 s append MemAvailable, disk free and owner/child RSS + liveness
to <root>/MEMORY_WATCH.jsonl (report-only; never acts on the run)."""
import json, shutil, sys, time
from pathlib import Path

root = Path(sys.argv[1])


def rss_kb(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def children(pid):
    out = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if int((p/'stat').read_text().split(')')[-1].split()[1]) == pid:
                    out.append(int(p.name))
            except (OSError, ValueError, IndexError):
                pass
    return out


while not (root/'LAUNCHED.json').exists():
    if (root/'STOP.json').exists() or (root/'OWNER_STOP.json').exists():
        sys.exit(0)
    time.sleep(30)
peak = 0.
while not ((root/'MONITOR_COMPLETE.json').exists() or (root/'STOP.json').exists() or (root/'OWNER_STOP.json').exists()):
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    avail = int(mem['MemAvailable'].split()[0])/1048576
    used = (int(mem['MemTotal'].split()[0])-int(mem['MemAvailable'].split()[0]))/1048576
    peak = max(peak, used)
    row = dict(unix=time.time(), t=time.strftime('%H:%M:%S'), mem_available_gb=round(avail, 1),
               used_gb=round(used, 1), peak_used_gb=round(peak, 1), disk_free_gb=round(shutil.disk_usage(root).free/1024**3, 1), owners={})
    for f in sorted((root/'owners').glob('*.json')):
        try:
            pid = json.loads(f.read_text())['pid']
        except (OSError, ValueError, KeyError):
            continue
        kids = children(pid)
        row['owners'][f.stem] = dict(pid=pid, alive=Path(f'/proc/{pid}').exists(), rss_gb=round((rss_kb(pid) or 0)/1048576, 2),
                                     children=len(kids), children_rss_gb=round(sum(rss_kb(k) or 0 for k in kids)/1048576, 2),
                                     grandchildren_rss_gb=round(sum(rss_kb(g) or 0 for k in kids for g in children(k))/1048576, 2))
    with (root/'MEMORY_WATCH.jsonl').open('a') as fh:
        fh.write(json.dumps(row)+'\n')
    time.sleep(60)
