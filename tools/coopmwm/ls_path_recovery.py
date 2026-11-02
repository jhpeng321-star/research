"""Append-only MAIN03:19 grading dependency diagnosis and recovery."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time

from . import ops
from .ls_collect_ops import COLLECTION
from .ls_validity_ops import remote
from .storage import write
from .protocol import REGISTRY, checked, sha

RESTORE=ops.EVIDENCE/'LS_TEMPLATE_RESTORE_LF_REGISTRATION.json'
ARCHIVE=Path('E:/Temp/opencode/coopmwm_template_65cc714e_lf.tar')
TEMPLATE_ROOT='/home/ubuntu/aq_code/code_r13e_65cc714e'
BLOB='9bbbe41c79eb4df4903be80753001a1b960b4199'


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


def register_restore():
    import tarfile
    with tarfile.open(ARCHIVE) as archive:
        raw=archive.extractfile('scenarios/idm/driver_routes.rou.xml').read()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==BLOB
    archive_sha=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if not RESTORE.exists():
        line=('\nS-COOPMWM MAIN03:19 ENGINEERING DEPENDENCY RESTORATION: deleted aq_code '
              'scenario template restored opaquely from original git revision65cc714e; '
              'git blob9bbbe41c79eb4df4903be80753001a1b960b4199 equals current HEAD. '
              'No assignment content inspection, policy/training/search change or new WM searches. '
              'Existing supervisor/broker/GPU owners and failed receipts/charges retained. '
              'Archive digest then git-blob digest checked before recovery declared. '
              'Initial Windows archive had CRLF conversion (blob e2b027a8e39f5f188a7c28d763fbaa0eeae4d572), '
              'failed digest gate before recovery; retained as .windows-crlf backup. '
              'LF archive generated with invocation-only core.autocrlf=false; no git settings changed. '
              'Fresh operational6M total guard unchanged; regrading separately precharged '\
              'as new attempts from saved snapshots under sumo_verification.\n')
        write(RESTORE,dict(line=line,unix=time.time(),archive_sha256=archive_sha,
            git_revision='65cc714e',git_blob=BLOB,root=TEMPLATE_ROOT,
            member='scenarios/idm/driver_routes.rou.xml',new_sumo_sim_s=0,
            authority='MAIN2026-11-03 03:19',content_inspection=False))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:
            f.write(line+'amendmentSHA'+sha(RESTORE)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(RESTORE),archive_sha256=archive_sha)))


def restore():
    reg=json.loads(RESTORE.read_text());data=ARCHIVE.read_bytes()
    assert hashlib.sha256(data).hexdigest()==reg['archive_sha256']
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,base64,hashlib,subprocess
c=Path({COLLECTION!r});reg=json.loads({json.dumps(reg)!r});dest=Path(reg['root'])
receipt=c/'TEMPLATE_RESTORED_0319.json'
if receipt.exists():print(receipt.read_text());raise SystemExit
assert Path('/home/ubuntu').is_dir()
archive=c/'TEMPLATE_RESTORE_65cc714e_LF.tar';data=base64.b64decode({base64.b64encode(data).decode()!r})
assert hashlib.sha256(data).hexdigest()==reg['archive_sha256']
if archive.exists():assert hashlib.sha256(archive.read_bytes()).hexdigest()==reg['archive_sha256']
else:
 with archive.open('xb') as f:f.write(data)
names=subprocess.check_output(['tar','-tf',str(archive)],text=True).splitlines()
assert set(names)=={{'scenarios/','scenarios/idm/',reg['member']}},names
target=dest/reg['member']
if target.exists():
 existing=subprocess.check_output(['git','hash-object','--',str(target)],text=True).strip()
 if existing!=reg['git_blob']:
  assert existing=='e2b027a8e39f5f188a7c28d763fbaa0eeae4d572'
  target.rename(target.with_name(target.name+'.windows-crlf'))
if not target.exists():
 dest.mkdir(parents=True,exist_ok=True)
 subprocess.check_call(['tar','-xf',str(archive),'-C',str(dest),reg['member']])
blob=subprocess.check_output(['git','hash-object','--',str(target)],text=True).strip()
assert blob==reg['git_blob'],blob
sha256=subprocess.check_output(['sha256sum','--',str(target)],text=True).split()[0]
out=dict(unix=time.time(),path=str(target),git_blob=blob,sha256=sha256,
 archive_sha256=reg['archive_sha256'],amendment_sha256={sha(RESTORE)!r},
 source_revision=reg['git_revision'],opaque_transfer=True,content_inspection=False,
 new_sumo_sim_s=0,no_process_signals=True)
with receipt.open('x') as f:json.dump(out,f)
with (Path(json.loads((c/'LS_COLLECT_BINDING.json').read_text())['run_root'])/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{sha(RESTORE)!r}+chr(10))
print(json.dumps(out))''',timeout=60))
    path=ops.EVIDENCE/'LS_TEMPLATE_RESTORED_0319.json'
    if not path.exists():write(path,value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=('diagnose','sources','candidates','register_restore','restore'))
    globals()[parser.parse_args().action]()
