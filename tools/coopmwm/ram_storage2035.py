"""Own-run storage telemetry and prospective lossless RAM staging; no policy changes."""
import argparse
import json
from . import ops
from .storage import write
from .plan_bridge import ROOT


def inventory():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import pathlib,json,shutil,time
r=pathlib.Path({ROOT!r});assert r.is_dir()
cases=[]
for p in (r/'jobs').iterdir():
 if not p.is_dir():continue
 files={{n:(p/n).stat().st_size for n in ('RESULT.json','RESULT.json.gz','observable.jsonl.gz') if (p/n).exists()}}
 cases.append(dict(id=p.name,claimed=(p/'CLAIM.json').exists(),closed=(p/'CLOSE.json').exists(),files=files))
print(json.dumps(dict(unix=time.time(),disk_free=shutil.disk_usage(r).free,ram_fs_free=shutil.disk_usage('/dev/shm').free,cases=cases,new_sumo_sim_s=0)))
''',timeout=25))
    write(ops.EVIDENCE/f'STORAGE_TELEMETRY_{int(value["unix"])}.json',value)
    completed=[c for c in value['cases'] if c['closed']]
    print(json.dumps(dict(unix=value['unix'],disk_free=value['disk_free'],ram_fs_free=value['ram_fs_free'],
        completed=completed,claimed=sum(c['claimed'] for c in value['cases']),new_sumo_sim_s=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('inventory',));a=p.parse_args();inventory()
