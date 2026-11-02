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
SOURCE='/home/ubuntu/coopmwm20261103_lspathcpu1'
REGRADE=ops.EVIDENCE/'LS_SAVED_REGRADE_REGISTRATION_0319.json'


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


def qualify():
    from .ls_validity_ops import payload
    packed=payload(('ls_saved_regrade','test_ls_saved_regrade_cpu'))
    print(remote('cpuQ',f'''from pathlib import Path
import json,os,shutil,hashlib,subprocess,time,base64,zlib
c=Path({COLLECTION!r});base=Path(json.loads((c/'LS_FOCUS_CONFIG.json').read_text())['source_root'])
r=Path({SOURCE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 r.mkdir();shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):shutil.copy2(base/name,r/name)
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['test_ls_saved_regrade_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('x') as f:json.dump(sorted(set(tests)),f)
for n,text in files.items():assert (r/'pytools/tools/coopmwm'/(n+'.py')).read_text()==text,n
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
 with (r/'LS_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_cpu_test',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))''',timeout=60))


def cpu_result():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
print(p.read_text() if p.exists() else json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-4000:])))''',timeout=60))
    if value.get('pending'):print(json.dumps(value));return
    path=ops.EVIDENCE/'LS_SAVED_REGRADE_CPU_0319.json'
    if not path.exists():write(path,value)
    print(json.dumps({k:value[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))


def regrade_plan():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,time,gzip
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.ls_saved_regrade import select,GENERATION
from tools.coopmwm.protocol import digest
from tools.coopmwm.accel_storage import read,charged
c=Path({COLLECTION!r});cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text())
for name in ('LS_RECOVERY_CONFIG','LS_VALIDITY_CONFIG','LS_LABEL_CONFIG','LS_LABEL_DISK_CONFIG','LS_FOCUS_CONFIG'):
 p=c/(name+'.json')
 if p.exists():cfg.update(json.loads(p.read_text()))
labels=[dict(json.loads(p.read_text()),_file=p.name) for p in (c/'labels').glob('*.json')]
items=[];m=Path(cfg['run_root'])
for x in select(labels):
 file=x.pop('_file');key=x['search_id'];p=c/'pending'/(key+'.json.gz')
 assert p.exists() and (m/'responses'/(key+'.json')).exists(),key
 reservation=json.loads((c/'admissions'/(key+'.json')).read_text())
 assert not (c/'labels'/(key+'.json')).exists()
 items.append(dict(search_id=key,file=file,world_id=x['world_id'],original_label_sha256=digest(x),
  pending_path=str(p),response_sha256=digest(read(m/'responses'/(key+'.json'))),sim_s=reservation['charged_sim_s']))
print(json.dumps(dict(unix=time.time(),items=items,total_sim_s=sum(x['sim_s'] for x in items),
 charged_before=charged(m),operational_spend_cap=6000000,generation=GENERATION,config=cfg)))''',timeout=60))
    path=ops.EVIDENCE/'LS_SAVED_REGRADE_PLAN_0319.json'
    if path.exists():assert json.loads(path.read_text())['items']==value['items']
    else:write(path,value)
    print(json.dumps({k:value[k] for k in ('unix','items','total_sim_s','charged_before','operational_spend_cap')}))


def register_regrade():
    cpu=json.loads((ops.EVIDENCE/'LS_SAVED_REGRADE_CPU_0319.json').read_text())
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    plan=json.loads((ops.EVIDENCE/'LS_SAVED_REGRADE_PLAN_0319.json').read_text())
    assert plan['charged_before']+plan['total_sim_s']<=6000000
    if not REGRADE.exists():
        line=('\nS-COOPMWM MAIN03:19 SAVED-SEARCH REGRADE ENGINEERING AMENDMENT: '
            'missing aq_code template restored to original65cc714e blob; '
            'ten prephysics FileNotFoundError labels get NEW SUMOPATH1 attempts, '
            'same saved observable snapshots and original8 candidates, ZEROnewWMsearches. '
            'Originalfailedlabels/chargesretained; no successful label replay. '
            'Precharge upper bound64800sim-s in sumo_verification before any physics; '
            'current4972430+64800=5037230<fresh operational6000000 cap, '
            'also recheck live atomic total+phase guard. Serial9branchlists/nice19, '
            'each policybranch real smoke/prefix passes before candidates; new claim IDs '
            'never replayed if interrupted. All metadata/broker/GPU owners retained. '
            'Frozen lspathcpu1:229Linux CPU PASS0skips43.132s/zeroSUMO. '
            'B/Rremainu3, mirrorOFF, no training or search change.\n')
        write(REGRADE,dict(plan,line=line,cpu=cpu,source_root=SOURCE,restore_sha256=sha(RESTORE)))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:
            f.write(line+'amendmentSHA'+sha(REGRADE)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(REGRADE),cpu_tests=cpu['tests'],retry_sim_s=plan['total_sim_s'])))


def deploy_regrade():
    reg=json.loads(REGRADE.read_text());h=sha(REGRADE)
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,sys,time,subprocess,hashlib
c=Path({COLLECTION!r});s=Path({SOURCE!r});reg=json.loads({json.dumps(reg)!r})
sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.warm_protocol import install_accounting
from tools.coopmwm import accel_storage
from tools.coopmwm.accel_protocol import SUPERSEDED_CHARGE
from tools.coopmwm.ls_label_runtime import sumo_environment
dest=c/'SAVED_REGRADE_DISPATCH_0319.json'
if dest.exists():print(dest.read_text());raise SystemExit
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
assert json.loads((c/'TEMPLATE_RESTORED_0319.json').read_text())['git_blob']=={BLOB!r}
assert all(not (Path(reg['config']['run_root'])/(a+'_checkpoint_004.pt')).exists() for a in ('B','R'))
config=reg['config'];main=Path(config['run_root']);install_accounting(main,config['budget'])
accel_storage._reserve.__globals__['CAP']=min(accel_storage._reserve.__globals__['CAP'],6000000-SUPERSEDED_CHARGE)
jobs=[dict(id='lslabel_'+x['search_id']+'_'+reg['generation'],sim_s=x['sim_s']) for x in reg['items']]
existing={{x['id'] for x in accel_storage.ledger(main)}}
if not any(j['id'] in existing for j in jobs):accel_storage.reserve(main,jobs,'sumo_verification')
else:assert all(j['id'] in existing for j in jobs),'partial charge, never retry blindly'
immutable(c/'SAVED_REGRADE_REGISTRATION_0319.json',reg)
bound=c/'SAVED_REGRADE_REGISTERED_0319.json'
if not bound.exists():
 with (main/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+chr(10))
 immutable(bound,dict(amendment_sha256={h!r},unix=time.time()))
env=sumo_environment(config['sumo_home'])
env.update(PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'SAVED_REGRADE_0319.log').open('x') as log:
 p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_saved_regrade',str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_root=str(s),amendment_sha256={h!r},charged_total_sim_s=accel_storage.charged(main),new_wm_searches=0)
immutable(dest,d);print(json.dumps(d))''',timeout=60))
    path=ops.EVIDENCE/'LS_SAVED_REGRADE_DISPATCH_0319.json'
    if not path.exists():write(path,value)
    print(json.dumps(value))


def regrade_status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({COLLECTION!r});d=json.loads((c/'SAVED_REGRADE_DISPATCH_0319.json').read_text());p=Path('/proc',str(d['pid']))
identity=p.exists() and (p/'stat').read_text().split()[21]==d['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
rows=[]
for path in (c/'labels').glob('*_SUMOPATH1.json'):
 v=json.loads(path.read_text());r=v['verification'];rows.append(dict(key=v['search_id'],status=r['status'],branches=len(r.get('branches',[])),actual_sim_s=r.get('actual_sim_s'),reason=r.get('reason')))
print(json.dumps(dict(unix=time.time(),owner=d,identity_verified=identity,state=(p/'stat').read_text().split()[2] if p.exists() else 'exited',
 labels=rows,complete=(c/'saved_regrade_0319/COMPLETE.json').exists(),
 smoke=json.loads((c/'LABEL_SUMO_SMOKE_SUMOPATH1.json').read_text()) if (c/'LABEL_SUMO_SMOKE_SUMOPATH1.json').exists() else None,
 log=(c/'SAVED_REGRADE_0319.log').read_text()[-4000:])))''',timeout=60))
    write(ops.EVIDENCE/f'LS_SAVED_REGRADE_STATUS_{time.time_ns()}.json',value)
    print(json.dumps(value))


def gpu_status(host):
    import contextlib
    import io
    from .ls_validity_ops import gpu_status as status
    with contextlib.redirect_stdout(io.StringIO()):
        value=status(host)
    print(json.dumps(dict(host=host,unix=value['unix'],gpu=value.get('gpu'),workers=[
        dict(owner=w['owner'],state=w['state'],ready=w['ready'],gates=len(w['gates']),failure=w['failure'],
             variant_pass=(w.get('variant_gate') or {}).get('pass_')) for w in value['workers']])))


def bank_summary():
    import contextlib
    import io
    from .ls_label_ops import bank_status
    with contextlib.redirect_stdout(io.StringIO()):
        value=bank_status()
    print(json.dumps({k:value[k] for k in ('unix','entries','selected_positive_lists',
        'confirmed_entries','admitted_positive_entries','prerequisite_met','exact_delay_null_lists',
        'focus_share','training_paused')}))


def publish_regrade():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,sys,time,hashlib
c=Path({COLLECTION!r});sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.protocol import digest
from tools.coopmwm.accel_storage import charged
reg=json.loads((c/'SAVED_REGRADE_REGISTRATION_0319.json').read_text())
assert (c/'saved_regrade_0319/COMPLETE.json').exists()
rows=[]
for item in reg['items']:
 old=json.loads((c/'labels'/item['file']).read_text())
 assert digest(old)==item['original_label_sha256']
 path=c/'labels'/(item['search_id']+'_SUMOPATH1.json');x=json.loads(path.read_text());v=x['verification']
 assert v['status'] in ('accepted','rejected') and len(v['branches'])==9
 assert all(b['status']=='graded' for b in v['branches']),[b['status'] for b in v['branches']]
 rows.append(dict(key=item['search_id'],world_id=item['world_id'],status=v['status'],
  branches=9,actual_sim_s=v['actual_sim_s'],receipt_sha256=v['receipt_sha256'],
  label_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_retained=True))
smoke=json.loads((c/'LABEL_SUMO_SMOKE_SUMOPATH1.json').read_text());assert smoke['pass_']
out=dict(unix=time.time(),pass_=True,lists=len(rows),branches=sum(r['branches'] for r in rows),
 actual_sim_s=sum(r['actual_sim_s'] for r in rows),precharged_sim_s=reg['total_sim_s'],
 charged_total_sim_s=charged(Path(reg['config']['run_root'])),operational_cap=6000000,
 new_wm_searches=0,all_originals_retained=True,smoke=smoke,results=rows,
 amendment_sha256={sha(REGRADE)!r})
print(json.dumps(out))''',timeout=60))
    path=ops.EVIDENCE/'LS_SAVED_REGRADE_COMPLETE_0319.json'
    if not path.exists():write(path,value)
    print(json.dumps({k:value[k] for k in ('unix','pass_','lists','branches','actual_sim_s',
        'precharged_sim_s','charged_total_sim_s','new_wm_searches','all_originals_retained')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=('diagnose','sources','candidates','register_restore','restore','qualify','cpu_result','regrade_plan','register_regrade','deploy_regrade','regrade_status','gpu_status','bank_summary','publish_regrade'))
    parser.add_argument('--host',choices=('gpuA','gpuR'))
    args=parser.parse_args()
    globals()[args.action](args.host) if args.action=='gpu_status' else globals()[args.action]()
