"""Public environment and named mining prerequisite inventory, no simulation."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

home = Path.home()
out = dict(python=sys.version, executable=sys.executable, affinity=sorted(os.sched_getaffinity(0)),
           disk=shutil.disk_usage(home)._asdict(), load=os.getloadavg())
out['packages'] = {}
for name in ('eclipse-sumo','sumolib','traci','libsumo','numpy','scipy','pandas','torch'):
    try: out['packages'][name] = metadata.version(name)
    except metadata.PackageNotFoundError: out['packages'][name] = None
quota = Path('/sys/fs/cgroup/cpu.max')
if quota.exists(): out['quota'] = quota.read_text()
out['roots'] = [str(p) for p in home.iterdir() if p.is_dir() and p.name.startswith(
    ('coopmwm','cpuv','cpus_env','mining'))]
if home == Path('/root'):
    data = Path('/root/autodl-tmp')
    out['data_disk'] = shutil.disk_usage(data)._asdict()
    out['data_roots'] = [p.name for p in data.iterdir() if p.is_dir()]
    out['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu',
                                         '--format=csv,noheader'], text=True)
print(json.dumps(out, sort_keys=True))
