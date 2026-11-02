"""Grading-only recovery; GPU/broker owners and all saved searches are adopted."""
import argparse
import json

from . import ops
from .ls_collect_ops import COLLECTION, ROOT
from .ls_validity_ops import SOURCE as BASE, remote, payload
from .protocol import REGISTRY, checked, sha
from .storage import read, write

SOURCE='/home/ubuntu/coopmwm20261102_lslabelcpu3'
GENERATION='SUMOENV1'
FILES=('ls_collection','ls_label_runtime','sumo_verify','test_ls_label_runtime_cpu')


def qualify():
    packed=payload(FILES)
    print(remote('cpuQ',f'''from pathlib import Path
import json,os,shutil,hashlib,subprocess,time,base64,zlib
r=Path({SOURCE!r});base=Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 r.mkdir()
 shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):shutil.copy2(base/name,r/name)
 for n,text in files.items():(r/'pytools/tools/coopmwm'/(n+'.py')).write_text(text)
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['test_ls_label_runtime_cpu']
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
 print(json.dumps(d))'''))


def result():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
print(p.read_text() if p.exists() else json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-4000:])))'''))
    if value.get('pending'):print(json.dumps(value));return
    p=ops.EVIDENCE/('LS_LABEL_CPU_'+SOURCE.rsplit('/',1)[1]+'.json')
    if not p.exists():write(p,value)
    print(json.dumps({k:value[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
    if not value['pass_']:print(remote('cpuQ',f'from pathlib import Path;print(Path({SOURCE!r},"LS_CPU.log").read_text()[-14000:])'))


def register():
    cpu=read(ops.EVIDENCE/('LS_LABEL_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'));assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    target=ops.EVIDENCE/'LS_LABEL_DISK_REGISTRATION.json'
    line=('\nS-COOPMWM MAIN23:32 SUMO ENVIRONMENT ENGINEERING RECOVERY: preserve all GPU/broker owners, '
        '72 completed episodes, pools and candidates. Replace collection metadata/grading supervisor only '
        'after boot/start/argv/no-branch-child verification. Explicit existing SUMO_HOME/PATH; '
        'real policy branch must pass prefix smoke before candidate branches, reused without extra physics. '
        'Zero-physics missing-binary labels marked void/no-outcome, original files/charges retained; '
        'fresh SUMOENV1 attempts precharged in sumo_verification, no new WM searches for retries. '
        'Fresh-message conservative total cap6000000 includes retained superseded811640; '
        'reserve hard guard applies to collection admissions and retry charges. No training restart, '
        'mirror/learned activation, horizon/population/K change or final outcome access. '
        'DISK FOLLOW-UP: existing2GiB guard queues labels BEFORE verification claims, no failed labels '
        'or new charges while waiting; original SUMOENV1 smoke/labels/claims adopted unchanged.\n')
    if not target.exists():
        write(target,dict(line=line,cpu=cpu,source_root=SOURCE,generation=GENERATION,
            operational_spend_cap=6000000,sumo_home='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
            authority='MAIN2026-11-02 23:32 plus fresh resume cap',original_failed_charges_retained=True))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(target)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(target),cpu_tests=cpu['tests'],cap=6000000)))


def deploy():
    reg=read(ops.EVIDENCE/'LS_LABEL_DISK_REGISTRATION.json');h=sha(ops.EVIDENCE/'LS_LABEL_DISK_REGISTRATION.json')
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,os,sys,time,signal,subprocess,hashlib
c=Path({COLLECTION!r});r=Path({ROOT!r});s=Path({SOURCE!r});sys.path.insert(0,str(s/'pytools'))
from tools.coopmwm.warm_spool import immutable
from tools.coopmwm.ls_label_runtime import sumo_environment
from tools.coopmwm.accel_storage import charged
reg=json.loads({json.dumps(reg)!r})
for n,h in reg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
dest=c/'SUPERVISOR_DISPATCH_SUMODISK1.json'
if dest.exists():print(dest.read_text());raise SystemExit
assert all(not (r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R'))
assert all((r/'jobs'/('lscollect1_B3_'+str(w))/'RESULT.json').exists() for w in range(986000,986072))
assert charged(r)<reg['operational_spend_cap']
env=sumo_environment(reg['sumo_home']);version=subprocess.check_output([env['SUMO_HOME']+'/bin/sumo','--version'],env=env,text=True,timeout=20)
old=json.loads((c/'SUPERVISOR_DISPATCH_SUMOENV1.json').read_text());p=Path('/proc',str(old['pid']))
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==old['boot_id']
if p.exists() and (p/'stat').read_text().split()[2]!='Z':
 assert (p/'stat').read_text().split()[21]==old['start_ticks']
 argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert 'tools.coopmwm.ls_collection' in argv and str(c) in argv
 assert json.loads((c/'LS_COLLECT_STATUS.json').read_text())['grading_inflight']==0,'wait for grading boundary'
 os.kill(old['pid'],signal.SIGSTOP)
 children=(p/'task'/str(old['pid'])/'children').read_text().split()
 for child in children:
  q=Path('/proc',child);a=(q/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  if 'multiprocessing.resource_tracker' not in a:
   os.kill(old['pid'],signal.SIGCONT);raise ValueError('live branch child; wait')
 os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT)
 for _ in range(100):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.1)
 else:raise ValueError('old metadata supervisor did not exit')
immutable(c/'SUPERVISOR_RETIRED_SUMODISK1.json',dict(owner=old,unix=time.time(),no_gpu_broker_or_physics_signals=True))
config=dict(source_root=str(s),receipt_suffix='_SUMODISK1',label_generation='SUMOENV1',
 sumo_home=reg['sumo_home'],operational_spend_cap=reg['operational_spend_cap'],amendment_sha256={h!r})
immutable(c/'LS_LABEL_DISK_CONFIG.json',config)
bound=c/'LABEL_DISK_REGISTERED.json'
if not bound.exists():
 with (r/'AUTHORITY.md').open('a') as f:f.write(reg['line']+'amendmentSHA'+{h!r}+chr(10))
 immutable(bound,dict(amendment_sha256={h!r},unix=time.time()))
env.update(PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (c/'SUPERVISOR_SUMODISK1.log').open('x') as log:
 proc=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_collection',str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
st=Path('/proc',str(proc.pid),'stat').read_text().split()
d=dict(pid=proc.pid,start_ticks=st[21],boot_id=old['boot_id'],root=str(c),source_root=str(s),unix=time.time(),amendment_sha256={h!r},sumo_version=version.splitlines()[0])
immutable(dest,d);print(json.dumps(d))''',timeout=60))
    target=ops.EVIDENCE/'LS_LABEL_DISK_DISPATCH.json'
    if not target.exists():write(target,value)
    print(json.dumps(value))


def status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,shutil
c=Path({COLLECTION!r});suffix=next(s for s in ('SUMOFOCUS1','SUMODISK1','SUMOENV1') if (c/('SUPERVISOR_DISPATCH_'+s+'.json')).exists())
d=json.loads((c/('SUPERVISOR_DISPATCH_'+suffix+'.json')).read_text());p=Path('/proc',str(d['pid']))
assert not p.exists() or (p/'stat').read_text().split()[21]==d['start_ticks']
labels=[json.loads(p.read_text()) for p in (c/'labels').glob('*_SUMOENV1.json')]
print(json.dumps(dict(unix=time.time(),owner=d,state=(p/'stat').read_text().split()[2] if p.exists() else 'exited',
 smoke=json.loads((c/'LABEL_SUMO_SMOKE_SUMOENV1.json').read_text()) if (c/'LABEL_SUMO_SMOKE_SUMOENV1.json').exists() else None,
 disk_wait=json.loads((c/'LABEL_DISK_WAIT_SUMOENV1.json').read_text()) if (c/'LABEL_DISK_WAIT_SUMOENV1.json').exists() else None,
 labels=[dict(search_id=x['search_id'],status=x['verification']['status'],branches=len(x['verification'].get('branches',[])),actual_sim_s=x['verification'].get('actual_sim_s'),grade=x['verification'].get('grade_veh_s'),reason=x['verification'].get('reason')) for x in labels],
 grading_health=dict(disk_blocked=shutil.disk_usage(c).free<2*1024**3,
  owner_identity_verified=bool(p.exists() and (p/'stat').read_text().split()[21]==d['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']),
  owner_argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode() if p.exists() else None,
  children=(p/'task'/str(d['pid'])/'children').read_text().split() if p.exists() else [],
  threads=[dict(tid=t.name,wchan=(t/'wchan').read_text()) for t in (p/'task').iterdir()] if p.exists() else [],
  inflight_semantics='thread futures include queued disk waiters; not a count of running SUMO branches',
  automatic_resume='existing grading_disk_ready loop rechecks disk every 10 seconds'),
 collection=json.loads((c/'LS_COLLECT_STATUS.json').read_text()),disk_free=shutil.disk_usage(c).free,
 log_tail=(c/('SUPERVISOR_'+suffix+'.log')).read_text()[-2200:])))'''))
    write(ops.EVIDENCE/f'LS_LABEL_STATUS_{int(value["unix"])}.json',value);print(json.dumps(value));return value


def publish():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({COLLECTION!r});labels=[json.loads(p.read_text()) for p in (c/'labels').glob('*_SUMOENV1.json')]
labels=[x for x in labels if x['verification']['status'] in ('accepted','rejected')]
assert labels,'no complete real label yet'
first=min(labels,key=lambda x:x['verification']['timing']['finished_unix'])
print(json.dumps(dict(unix=time.time(),first_label=first,smoke=json.loads((c/'LABEL_SUMO_SMOKE_SUMOENV1.json').read_text()),
 voids=[json.loads(p.read_text()) for p in (c/'label_voids').glob('*.json')],new_searches_for_regrading=0)))'''))
    p=ops.EVIDENCE/'LS_FIRST_REAL_LABEL_SUMOENV1.json'
    if not p.exists():write(p,value)
    print(json.dumps(dict(published=str(p),branches=len(value['first_label']['verification']['branches']),
        status=value['first_label']['verification']['status'],smoke_pass=value['smoke']['pass_'])))


def bank_status():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,sys
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.exact_grade import _bootstrap_mean,verify_sealed
c=Path({COLLECTION!r});all_labels=[json.loads(p.read_text()) for p in (c/'labels').glob('*.json')]
labels=[x for x in all_labels if x['verification']['status'] in ('accepted','rejected')]
assert len({{x['search_id'] for x in labels}})==len(labels),'duplicate usable labels'
positive=[x for x in labels if x['verification']['status']=='accepted' and x['verification']['grade_veh_s']>0]
values=[x['verification']['best_margin_veh_s'] for x in labels]
mean,ci=_bootstrap_mean(values,[x['world_id'] for x in labels],10000,20261101) if labels else (None,None)
focus=lambda x:x['verification']['episode']['world']['demand'] in (2400,2700) and x['verification']['episode']['world']['penetration'] in (.2,.4)
confirmed=[]
for p in (Path({ROOT!r})/'replay_bank/B/confirmations').glob('*.json'):
 x=json.loads(p.read_text());verify_sealed(x,'confirmation_sha256');confirmed.append(x)
legacy_confirmed=list(confirmed)
overlays=Path({ROOT!r})/'replay_bank/B/confirmations_safety0426'
for i,x in enumerate(confirmed):
 p=overlays/(x['verification_id']+'.json')
 if p.exists():
  v=json.loads(p.read_text());verify_sealed(v,'confirmation_sha256')
  assert v['original_confirmation_sha256']==x['confirmation_sha256']
  confirmed[i]=v
admitted=[x for x in confirmed if x['admitted']]
null_lists=[dict(search_id=x['search_id'],world_id=x['world_id'],
 all_eight_delay_equal_policy=len(x['verification'].get('J_candidates_veh_s') or [])==8 and
 all(v is not None and v==x['verification']['J_policy_veh_s'] for v in x['verification']['J_candidates_veh_s']),
 mirror_active=False,mirror_source='current collection binding; command mirror gate failed') for x in labels]
print(json.dumps(dict(unix=time.time(),arm='B',policy_update=3,entries=len(labels),positives=len(positive),
 mean_exact_advantage_veh_s=mean,world_bootstrap_ci95=ci,
 estimand='best of eight exact candidate margins vs policy; includes nonpositive decisions',
 rank5_8_winners=sum(x['verification']['chosen']>=4 for x in positive),
 rank5_8_share_of_positive_winners=sum(x['verification']['chosen']>=4 for x in positive)/len(positive) if positive else None,
 worlds=sorted(set(x['world_id'] for x in labels)),focus_share=sum(focus(x) for x in labels)/len(labels) if labels else None,
 void_attempts=len(list((c/'label_voids').glob('*.json'))),training_paused=True,
   prerequisite_positive_entries=10,prerequisite_met=len(admitted)>=10,
   prerequisite_contract='MAIN00:40 ten confirmed/admitted positive B entries; raw best-of-eight positives are not admissions; qualified release also required',
    selected_positive_lists=len(positive),confirmed_entries=len(confirmed),admitted_positive_entries=len(admitted),
    safety_rule='MAIN2026-11-03 04:26',legacy_admitted_positive_entries=sum(x['admitted'] for x in legacy_confirmed),
  exact_delay_null_lists=sum(x['all_eight_delay_equal_policy'] for x in null_lists),
  exact_delay_null_rate=sum(x['all_eight_delay_equal_policy'] for x in null_lists)/len(null_lists) if null_lists else None,
  per_label_null_flags=null_lists,null_definition='all eight complete candidate team delays exactly equal policy; no tolerance or trajectory-equivalence inference',
 R_bank_entries=0,training_changes_from_statistics=False)))'''))
    target=ops.EVIDENCE/'BANK_STATUS.json';stamp=__import__('time').time_ns();tmp=target.with_name(f'.BANK_STATUS.{stamp}.publishing')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,sort_keys=True)
    __import__('os').replace(tmp,target)
    write(ops.EVIDENCE/f'BANK_STATUS_{stamp}.json',value)
    print(json.dumps(value));return value


def variation_summary(rows):
    """Describe exact labelled slates; missing post-release data stays missing."""
    groups={key:[] for key in ('before_release','niches1','niches4')}
    for row in rows:
        mode=row.get('niches')
        key='niches'+str(mode) if row.get('search_clamp_enabled') and mode in (1,4) else 'before_release'
        groups[key].append(row)
    result={}
    for key,xs in groups.items():
        complete=[x for x in xs if x.get('candidate_count')==8 and x.get('exact_null') is not None]
        nulls=sum(x['exact_null'] for x in complete)
        distances=[x['mean_pairwise_distance'] for x in xs if x.get('mean_pairwise_distance') is not None]
        occupancy=[x['occupied_niches'] for x in xs if x.get('occupied_niches') is not None]
        signs=[x['share_with_net_deceleration'] for x in xs if x.get('share_with_net_deceleration') is not None]
        result[key]=dict(labelled_lists=len(xs),complete_eight_lists=len(complete),
            exact_null_lists=nulls,exact_null_rate=nulls/len(complete) if complete else None,
            partial_or_unmeasured_lists=len(xs)-len(complete),
            inactionable_skipped=sum(x['status']=='inactionable' for x in xs),
            mean_pairwise_distance=sum(distances)/len(distances) if distances else None,
            diversity_observed_lists=len(distances),
            mean_occupied_niches=sum(occupancy)/len(occupancy) if occupancy else None,
            mean_share_with_net_deceleration=sum(signs)/len(signs) if signs else None,
            mirror_active=any(x.get('mirror_active',False) for x in xs))
    return result


def variation():
    """Read saved TRAIN labels/responses only; zero new search/SUMO or control."""
    import time
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time,hashlib
c=Path({COLLECTION!r});r=Path({ROOT!r});rows=[]
for p in (c/'labels').glob('*.json'):
 x=json.loads(p.read_text());v=x['verification']
 if v['status'] not in ('accepted','rejected','inactionable'):continue
 response=r/'responses'/(x['search_id']+'.json')
 a=json.loads(response.read_text()) if response.exists() else {{}}
 variation=x.get('variation') or a.get('variation') or {{}}
 config=x.get('search_config') or a.get('search_config') or {{}}
 req=c/'requests'/(x['search_id']+'.json')
 request=json.loads(req.read_text()).get('search_config',{{}}) if req.exists() else {{}}
 costs=v.get('J_candidates_veh_s') or [];policy=v.get('J_policy_veh_s')
 null=(all(j==policy for j in costs) if len(costs)==8 and policy is not None and all(j is not None for j in costs) else None)
 rows.append(dict(search_id=x['search_id'],world_id=x['world_id'],status=v['status'],
  label_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
  search_config=config,niches=variation.get('niches',request.get('niches')),
  search_clamp_enabled=(x.get('nullfix') or a.get('nullfix') or {{}}).get('enabled',False),
  label_gate_enabled='feasibility' in x,mirror_active=False,candidate_count=len(costs),
  exact_null=null,best_margin_veh_s=v.get('best_margin_veh_s'),
  mean_pairwise_distance=variation.get('mean_pairwise_distance'),occupied_niches=variation.get('occupied_niches'),
  share_with_net_deceleration=variation.get('share_with_net_deceleration')))
print(json.dumps(dict(unix=time.time(),rows=rows,new_sumo_sim_s=0,new_wm_calls=0)))'''))
    value['cohorts']=variation_summary(value['rows'])
    value['contract']='exact delay equality only for complete eight-candidate lists; partial/skipped lists excluded from null-rate denominator; no post-release observations means null, not zero'
    write(ops.EVIDENCE/f'NULL_DIVERSITY_{time.time_ns()}.json',value)
    target=ops.EVIDENCE/'NULL_DIVERSITY_BY_CONFIG.json'
    tmp=target.with_name(f'.NULL_DIVERSITY.{time.time_ns()}.publishing')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f)
    __import__('os').replace(tmp,target)
    print(json.dumps(dict(unix=value['unix'],cohorts=value['cohorts'],new_sumo_sim_s=0)));return value


def failures():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import json,time
c=Path({COLLECTION!r});out=[]
for p in (c/'labels').glob('*.json'):
 x=json.loads(p.read_text());v=x['verification']
 if v['status']!='failed':continue
 out.append(dict(file=p.name,world=x['world_id'],reason=v.get('reason'),actual_sim_s=v.get('actual_sim_s'),
  timing=v.get('timing'),branches=[{{k:b.get(k) for k in ('id','status','reason','actual_sim_s','job_path')}} for b in v.get('branches',[])]))
targets=['/home/ubuntu/mappospeed/production2/policy_00169.pt','/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo/bin/sumo','/home/ubuntu/e3final/runtime']
print(json.dumps(dict(unix=time.time(),failed_labels=out,required_path_exists={{p:Path(p).exists() for p in targets}})))'''))
    write(ops.EVIDENCE/f'LS_LABEL_FAILURES_{__import__("time").time_ns()}.json',value)
    print(json.dumps(value));return value


def failure_sources():
    print(remote('cpuQ',f'''from pathlib import Path
import ast,json
c=Path({COLLECTION!r});s=Path(json.loads((c/'LS_FOCUS_CONFIG.json').read_text())['source_root'])
p=s/'pytools/tools/leverprobe/backend.py';text=p.read_text();tree=ast.parse(text)
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name=='setup':print(ast.get_source_segment(text,node))
 if isinstance(node,ast.ClassDef) and node.name=='Backend':
  for child in node.body:
   if isinstance(child,ast.FunctionDef) and child.name=='__init__':print(ast.get_source_segment(text,child))
'''))


def failure_paths():
    print(remote('cpuQ',f'''from pathlib import Path
import json,re
c=Path({COLLECTION!r});cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text())
r=Path(cfg['runtime']);p=r/'runtime/SOURCE_MANIFEST.json'
blocked=re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype',re.I)
entries=json.loads(p.read_text()) if p.exists() else {{}}
missing=[name for name in entries if not blocked.search(name) and not (r/'runtime'/name).exists()]
print(json.dumps(dict(runtime=str(r),manifest=str(p),manifest_exists=p.exists(),checkpoint=cfg['checkpoint'],checkpoint_exists=Path(cfg['checkpoint']).exists(),missing_allowed_runtime_paths=missing,metadata_only=True)))'''))


def policy_load_probe():
    print(remote('cpuQ',f'''from pathlib import Path
import os,sys,json,traceback
from unittest.mock import patch
sys.dont_write_bytecode=True
c=Path({COLLECTION!r});cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text());r=Path(cfg['runtime'])
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(r/'runtime/src'),str(r/'runtime')]
try:
 with patch('subprocess.Popen',side_effect=AssertionError('diagnostic cannot launch physics')):
  from wms.control.policy import load_policy_fn
  policy=load_policy_fn(cfg['checkpoint'],device='cpu');policy.reset()
 print(json.dumps(dict(pass_=True,new_sumo_sim_s=0)))
except Exception as error:print(json.dumps(dict(pass_=False,error=repr(error),filename=getattr(error,'filename',None),traceback=traceback.format_exc(),new_sumo_sim_s=0)))'''))


def env_failure_sources():
    print(remote('cpuQ',f'''from pathlib import Path
import ast,json
c=Path({COLLECTION!r});r=Path(json.loads((c/'LS_COLLECT_BINDING.json').read_text())['runtime'])/'runtime'
paths=[r/'traffic_shepherd/c4/env.py',r/'src/traffic_shepherd/c4/env.py']
p=next(p for p in paths if p.exists());text=p.read_text();tree=ast.parse(text)
for node in tree.body:
 if isinstance(node,ast.ClassDef) and node.name=='CavSumoEnv':
  for child in node.body:
   if isinstance(child,ast.FunctionDef) and child.name in ('__init__','reset','_start_sumo','_launch_sumo'):
    print(ast.get_source_segment(text,child))
'''))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','result','register','deploy','status','publish','bank_status','variation','failures','failure_sources','failure_paths','policy_load_probe','env_failure_sources'))
    globals()[p.parse_args().action]()
