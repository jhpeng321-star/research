"""One resident GPU model, original eight startup gates, bounded query inbox."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from .wm_contract import check_sources, file_sha, grid, read_json, validate_query
from .wm_prescreen import ASSET_PINS, load_base, qualify, rankings, score_grid

def write(p,v):
 with p.open('x') as f:
  json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())

def main(root):
 binding=read_json(root/'GPU_BINDING.json')
 check_sources(read_json(root/'MINING_SOURCE_MANIFEST.json'))
 check_sources(ASSET_PINS)
 write(root/'MINING_WM_OWNER.json',dict(pid=os.getpid(),
  start_ticks=Path('/proc/self/stat').read_text().split()[21],
  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time()))
 try:
  rt,q=qualify(root)
  write(root/'GPU_MODEL_READY.json',dict(qualification=q,unix=time.time(),new_sumo_sim_s=0))
  seen=set()
  while time.time()<binding['deadline_unix']:
   if (root/'MINING_STOP.json').exists():break
   pending=sorted((root/'queries').glob('*.json'))
   for path in pending:
    if path.name in seen:continue
    claim=root/'results'/(path.stem+'.claim')
    write(claim,dict(query=str(path),sha256=file_sha(path),unix=time.time()))
    query=validate_query(read_json(path))
    # A fixed, prospectively bounded longitudinal grid. Lane ABI refuses explicitly.
    actions,rejected=grid(query,onsets=(0,10),holds=(20,),depths=(.3,1.))
    weights=load_base(query)
    scores,stats=score_grid(rt,query,actions,weights)
    write(root/'results'/(path.stem+'.result.json'),dict(query_sha256=file_sha(path),
     actions=actions,geometry_rejections=rejected,scores=scores,ranked=rankings(scores,actions),
     telemetry=stats,K=8,H=600,unix=time.time(),native_admissions=0,
     lane_status='NOT_SCORED: no qualified candidate lane ABI',new_sumo_sim_s=0))
    seen.add(path.name)
   time.sleep(5)
  write(root/'MINING_WM_EXIT.json',dict(reason='stop_or_deadline',completed=sorted(seen),unix=time.time()))
 except BaseException as e:
  write(root/'MINING_WM_FAULT.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()))
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root)
