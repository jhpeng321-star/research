"""Sequential, receipt-bound S-C1COLLECT2 operations; no simulation on import."""
import argparse
from pathlib import Path

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import remote


def inspect():
    remote('gpuA', PRELUDE+'''
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
j=loop.read_json(a/'loop/worlds_c00.json')
print(json.dumps({'config':cfg,'cycle0':j,'current_state':state,
 'driver_live':previous.live(loop.read_json(a/'PID.json'))}))
''')


def discover():
    remote('gpuA', PRELUDE+'''
cfg=loop.load_config(a/'CONFIG.json'); j=loop.read_json(a/'loop/worlds_c00.json')
for box in ('cpuP','cpuQ'):
 h=loop.host(cfg,box)
 world=next(w for w in j['worlds'].values() if w['box']==box)
 code="""import pathlib,json,os
b=pathlib.Path('/home/ubuntu/c4/main14/restart20261026'); old=b/'runtime'; new=b/'runtime_c1'
work=pathlib.Path(WORK)
print('CYCLE0_METADATA',json.dumps({'work':str(work),'files':[p.name for p in (work/'collect').iterdir() if p.is_file()]}))
for name in ('COMMAND.json','COLLECT_CONFIG.json','COLLECT_SUMMARY.json'):
 p=work/'collect'/name
 if p.is_file(): print(name,p.read_text()[:12000])
for tree in (old,new):
 print('TREE',str(tree),'TOP',[(p.name, p.is_symlink()) for p in tree.iterdir()])
 for sub in ('src/traffic_shepherd','traffic_shepherd','baselines/circles'):
  p=tree/sub
  if p.is_dir(): print('SUB',str(p),[q.name for q in p.iterdir()])
for p in (new/'src/traffic_shepherd').glob('**/*.py'):
 if any(x in str(p).lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype')): continue
 if p.name in ('config.py','settings.py') or p.name=='collect.py':
  print('SOURCE',str(p)); print(p.read_text()[:35000])
"""
 result=subprocess.run([*h['ssh'],h['python']+' -B -'],input='WORK='+repr(world['collect']['work'])+'\\n'+code,text=True,capture_output=True,check=True,timeout=120)
 print('BOX',box,result.stdout,result.stderr)
''', timeout=280)


def sources():
    remote('gpuA', PRELUDE+'''
p=a/'loop/worlds/c00/w968144.a1/logs/c00_collect.log'
print('CYCLE0_COMMAND',p.read_text()[:3000])
''')


def dependencies():
    remote('cpuP', '''
import pathlib,json
r=pathlib.Path('/home/ubuntu/c4/main14/restart20261026/runtime_c1')
print('ADAPTER',(r/'baselines/circles/adapter.json').read_text())
for n in ('traffic_shepherd/c4/env.py','traffic_shepherd/c4/scenarios.py','traffic_shepherd/scenarios.py','traffic_shepherd/damping.py'):
 p=r/n
 if p.is_file(): print('SOURCE',n,p.read_text()[:40000])
''')


def parity_plan():
    source = Path(__file__).with_name('c1collect2_cpu.py').read_text()
    for box in ('cpuP', 'cpuQ'):
        remote(box, "__name__='__main__'\n"+source, timeout=300)


def parity_install():
    source = Path(__file__).with_name('c1collect2_cpu.py').read_text()
    for box in ('cpuP', 'cpuQ'):
        remote(box, "__name__='c1collect2_cpu'\n"+source+'\nparity(install=True)\n', timeout=300)


def preflight():
    import base64
    source = Path(__file__).with_name('c1collect2_cpu.py').read_text()
    remote('gpuA', PRELUDE+f'''
import base64
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
assert state['sim_s_charged']==139540 and not previous.live(loop.read_json(a/'PID.json'))
j=loop.read_json(a/'loop/worlds_c01.json'); acc=state['policy_wm']; pol=state['policy']
for box in ('cpuP','cpuQ'):
 one=json.loads(json.dumps(cfg)); one['hosts']['cpu']=one['hosts'][box]
 stage='/home/ubuntu/c4/main14/restart20261026/preflight_c1collect2'
 files=[('policy.pt',pol['checkpoint']),('native_gate_result.json',str(pathlib.Path(pol['checkpoint']).with_name('native_gate_result.json'))),
  ('wm/wm_s.pt',acc['checkpoint']),('wm/config.json',acc['config']),('REGISTRATION.json',cfg['registration'])]
 for name in ('physics.json','TRAIN_RECEIPT.json','lane_calib.json'):
  p=pathlib.Path(acc['checkpoint']).with_name(name)
  if p.is_file(): files.append(('wm/'+name,str(p)))
 for name,src in files: loop.sync(one,'gpu',src,'cpu',stage+'/'+name,r/'C1COLLECT2_STAGE.log')
 binding={{'policy_sha256':pol['sha256'],'wm_sha256':acc['sha256'],'wm_config_sha256':loop.sha256(acc['config'])}}
 h=loop.host(one,'cpu')
 code="import pathlib,json; pathlib.Path("+repr(stage+'/BINDINGS.json')+").write_text("+repr(json.dumps(binding))+")"
 result=subprocess.run([*h['ssh'],h['python']+' -B -'],input=code,text=True,capture_output=True,check=True,timeout=60)
 code="__name__='c1collect2_cpu'\\n"+base64.b64decode({base64.b64encode(source.encode()).decode()!r}).decode()+"\\npreflight("+repr(j['cases'])+")\\n"
 env="import os; os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')\\n"
 result=subprocess.run([*h['ssh'],h['python']+' -B -'],input=env+code,text=True,capture_output=True,check=True,timeout=180)
 print(box,result.stdout,result.stderr)
 doc=json.loads(result.stdout); loop.write_atomic(r/('C1COLLECT2_PREFLIGHT_'+box+'.json'),doc)
 assert doc['status']=='PASS', box+': '+doc.get('error','preflight failed')
assert loop.load_state(cfg)['sim_s_charged']==139540
''', timeout=500)


def pause():
    """Record the failed preflight and authorized allowance; never dispatch/retry."""
    import base64
    from tools.loopresume.operate import ROOT
    names = ['tools/loopresume/c1collect2.py', 'tools/loopresume/c1collect2_cpu.py']
    payload = {n: base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
    remote('gpuA', PRELUDE+f'''
import base64
with (a/'cycle1.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
 assert not previous.live(loop.read_json(a/'PID.json'))
 assert state['sim_s_charged']==139540
 assert loop.read_json(r/'C1COLLECT2_PREFLIGHT_cpuP.json')['status']=='FAIL'
 assert not (r/'C1COLLECT2_RESULT.json').exists()
 backup=r/'c1collect2_before'; backup.mkdir(exist_ok=False)
 for n in ('REGISTRATION.json','loop/LOOP14_STATE.json','OPERATIONAL_PAUSED.json'):
  dest=backup/n; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(a/n,dest)
 source={{}}
 for n,encoded in {payload!r}.items():
  dest=root/n; assert not dest.exists(); dest.write_bytes(base64.b64decode(encoded)); source[n]=loop.sha256(dest)
 loop.write_atomic(r/'C1COLLECT2_SOURCE.json',source)
 boxes={{}}
 for box in ('cpuP','cpuQ'):
  h=loop.host(cfg,box)
  code="""import pathlib,json,hashlib
b=pathlib.Path('/home/ubuntu/c4/main14/restart20261026'); rel='assets/research/migration_manifest.json'
p=b/'C1COLLECT2_PARITY.json'; parity=json.loads(p.read_text())
missing={{'successor_path':str(b/'runtime_c1'/rel),'successor_exists':(b/'runtime_c1'/rel).exists(),
 'original_path':str(b/'runtime'/rel),'original_exists':(b/'runtime'/rel).exists()}}
print(json.dumps({{'parity_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'copied_count':len(parity['copied']),'copied_bytes':sum(r['bytes'] for r in parity['copied']),
 'differences':parity['different'],'template_identity':parity['template_identity'],
 'missing_metadata':missing,'preflight_executed':(b/'C1COLLECT2_PREFLIGHT.json').exists()}}))
"""
  result=subprocess.run([*h['ssh'],h['python']+' -B -'],input=code,text=True,capture_output=True,check=True,timeout=60)
  boxes[box]=json.loads(result.stdout)
 reg=loop.read_json(cfg['registration'])
 assert reg['budget']['cycle_attempt_allowances']['1']==71025
 amendment={{'id':'S-C1COLLECT2','authority':'main decision 2026-10-27 05:45 UTC+8; act-on-recommendation',
  'registered_at':loop.now(),'cycle':1,'attempt_allowance_sim_s':73866,'increase_sim_s':2841,
  'long_credit_allowance_sim_s':15000,'total_cap_sim_s':210000,'charged_before':139540,
  'no_refunds':True,'seeds':list(range(968157,968170)),'new_collection_attempts':1,
  'launch_conditions':['full asset parity','zero-charge preflight PASS on cpuP and cpuQ','verified canary before remainder'],
  'failure':'OPERATIONAL_PAUSED; no further launch after failed preflight or canary',
  'policy_sha256':state['policy']['sha256'],'previous_registration_sha256':loop.sha256(cfg['registration'])}}
 reg.setdefault('amendments',[]).append(amendment)
 reg['budget']['cycle_attempt_allowances']['1']=73866
 previous.event('S-C1COLLECT2_amendment_installed',**amendment)
 loop.append(a/'loop/cycle_ledger.jsonl',{{'event':'S-C1COLLECT2_amendment_installed',**amendment}})
 doc={{'id':'S-C1COLLECT2','t':loop.now(),'status':'OPERATIONAL_PAUSED',
  'preflight':{{'cpuP':'FAIL','cpuQ':'NOT_RUN_AFTER_cpuP_FAILURE'}},'boxes':boxes,
  'failure_missing_path':boxes['cpuP']['missing_metadata']['successor_path'],
  'failure_classification':'preflight checker incorrectly required an unused migration manifest; direct adapter path does not invoke migration resolver',
  'collector_asset_missing_proven':False,'collector_import_completed':True,'onnx_load_reached':False,
  'canary_dispatched':False,'remaining_dispatched':False,'additional_charged':0,'charged_total':139540,
  'cycle_charged':36933,'cycle_attempt_allowance':73866,'long_credit_charged_included':7800,
  'long_credit_attempt_allowance':15000,'total_cap':210000,'remaining':70460,
  'source':source,'main_priority_active':False,'guard':None,
  'convert_wm1_check':'pending; blocked before dispatch','simulation_retry_armed':False,
  'wm0_exact_incarnations':{{str(pid):loop.fast_module().process_token(pid) for pid in (62645,62910)}},
  'next_action':'main decision required after failed preflight; checker must not require a nonexistent unused manifest'}}
 loop.write_atomic(r/'C1COLLECT2_RESULT.json',doc)
 pause={{'t':doc['t'],'task':'S-C1COLLECT2','error':doc['failure_classification'],
  'missing_path':doc['failure_missing_path'],'result':str(r/'C1COLLECT2_RESULT.json'),'no_launch':True}}
 loop.write_atomic(a/'OPERATIONAL_PAUSED.json',pause)
 reg.setdefault('operational_events',[]).append({{'id':'S-C1COLLECT2_result','t':doc['t'],
  'status':'OPERATIONAL_PAUSED','result_sha256':loop.sha256(r/'C1COLLECT2_RESULT.json'),'additional_charged':0}})
 loop.write_atomic(pathlib.Path(cfg['registration']),reg)
 state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
 previous.event('S-C1COLLECT2_preflight_failed',**doc)
 loop.append(a/'loop/cycle_ledger.jsonl',{{'event':'S-C1COLLECT2_preflight_failed',**doc}})
 print(json.dumps({{'result':doc,'result_sha256':loop.sha256(r/'C1COLLECT2_RESULT.json'),
  'registration_sha256':state['registration_sha256']}}))
''', timeout=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'discover', 'sources', 'dependencies', 'parity-plan', 'parity-install', 'preflight', 'pause'])
    globals()[parser.parse_args().action.replace('-', '_')]()
