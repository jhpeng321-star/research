"""Record system/container memory counters after observed worker SIGKILLs."""
from pathlib import Path
import json
import time

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
record = {'t': time.time()}
for name in ('memory.max', 'memory.current', 'memory.peak', 'memory.events'):
    path = Path('/sys/fs/cgroup')/name
    if path.exists():
        record[name] = path.read_text().strip()
record['host_memory_kib'] = {line.split(':')[0]: int(line.split()[1])
    for line in Path('/proc/meminfo').read_text().splitlines()
    if line.split(':')[0] in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}
(base/'HOST_MEMORY.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
