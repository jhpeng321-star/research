"""Append-only MAIN03:19 grading dependency diagnosis and recovery."""
import argparse
import json
import time

from . import ops
from .ls_collect_ops import COLLECTION
from .ls_validity_ops import remote
from .storage import write


def diagnose():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,os,sys,traceback,shutil,ast
from unittest.mock import patch
c=Path({COLLECTION!r});cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text())
s=Path(json.loads((c/'LS_FOCUS_CONFIG.json').read_text())['source_root'])
sys.path[:0]=[str(s/'pytools'),cfg['runtime']+'/runtime/src',cfg['runtime']+'/runtime']
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from tools.coopmwm.ls_label_runtime import sumo_environment
os.environ.update(sumo_environment(json.loads((c/'LS_LABEL_CONFIG.json').read_text())['sumo_home']))
from traffic_shepherd.c4.env import CavSumoEnv
from tools.leverprobe import backend
out=dict(unix=__import__('time').time(),new_sumo_sim_s=0,source_root=str(s),binary=shutil.which('sumo'))
try:
 backend.setup(cfg['runtime'],cfg['checkpoint'])
 # Constructor/configuration probe cannot start physics. Keep the full traceback,
 # including paths hidden by the branch runner's repr-only error boundary.
 with patch('subprocess.Popen',side_effect=AssertionError('diagnostic forbids physics')):
  env=CavSumoEnv(k_slots=192,controller='direct',ttc=True,physics_log=False,population='textbook')
 out['constructor_pass']=True
 from traffic_shepherd.c4.env import build_scenario
 import xml.etree.ElementTree as ET
 # Reproduce the failing dependency open while prohibiting template/truth reads.
 # An absent file yields its genuine FileNotFoundError; an existing file is
 # refused before any content can be read. No request_schedule is evaluated.
 template=env.config.scenario_template_path
 def missing_only_open(path,*args,**kwargs):
  if Path(path).exists():raise AssertionError('diagnostic refuses assignment content')
  return original_open(path,*args,**kwargs)
 import builtins
 original_open=builtins.open
 with patch('builtins.open',side_effect=missing_only_open):
  ET.parse(template)
except Exception as e:
 out.update(constructor_pass=False,error=repr(e),filename=getattr(e,'filename',None),traceback=traceback.format_exc())
paths=[Path(out['binary'])] if out['binary'] else []
out['binary_metadata']=[dict(path=str(p),resolved=str(p.resolve()),exists=p.exists(),executable=os.access(p,os.X_OK)) for p in paths]
owner=json.loads((c/'SUPERVISOR_DISPATCH_SUMOFOCUS1.json').read_text());p=Path('/proc',str(owner['pid']))
out['owner']=owner
out['owner_identity_verified']=p.exists() and (p/'stat').read_text().split()[21]==owner['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
out['collection']={{k:v for k,v in json.loads((c/'LS_COLLECT_STATUS.json').read_text()).items() if k in ('requested','finished','verified_lists','grading_inflight','charged_collection_sim_s')}}
print(json.dumps(out))''', timeout=60))
    path=ops.EVIDENCE/f'LS_PATH_DIAGNOSIS_{time.time_ns()}.json'
    write(path,value)
    print(json.dumps(value))


def sources():
    print(remote('cpuQ', f'''from pathlib import Path
import json,sys,inspect
c=Path({COLLECTION!r});cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text())
sys.path[:0]=[cfg['runtime']+'/runtime/src',cfg['runtime']+'/runtime']
from traffic_shepherd.c4.env import build_scenario
print(inspect.getsource(build_scenario))
from traffic_shepherd.simulation import start_sumo
print(inspect.getsource(start_sumo))
from traffic_shepherd.simulation import launch_lock
print(inspect.getsource(launch_lock))
from traffic_shepherd.config import load_config
config=load_config()
print(json.dumps({{k:dict(path=str(v),exists=Path(v).exists()) for k,v in vars(config).items() if isinstance(v,Path)}}))
'''))


def candidates():
    value=json.loads(remote('cpuQ', f'''from pathlib import Path
import json,time
c=Path({COLLECTION!r});cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text())
r=Path(cfg['runtime'])/'runtime';manifest=json.loads((r/'SOURCE_MANIFEST.json').read_text())
names=[n for n in manifest if Path(n).name=='driver_routes.rou.xml']
print(json.dumps(dict(unix=time.time(),candidates=[dict(path=str(r/n),exists=(r/n).exists(),expected_sha256=manifest[n]) for n in names],assignment_content_opened=False)))'''))
    write(ops.EVIDENCE/f'LS_PATH_CANDIDATES_{time.time_ns()}.json',value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=('diagnose','sources','candidates'))
    globals()[parser.parse_args().action]()
