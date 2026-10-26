"""Read-only result extraction and durable WAIT_OWNER receipt for S-MAINFIX."""
import argparse
import json

import collectfix as cf

A = '/root/autodl-tmp/c4/v5/mainstart/mainr16/restart'
OUT = cf.ROOT / 'tools/mainstart/mainfix'


def save(name, text):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(text, encoding='utf-8')
    print(text)


def extract():
    code = f'''
import pathlib,json,sys
a=pathlib.Path({A!r});r=a/'runtime';sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
import loop14 as loop
root=pathlib.Path('/root/main14_loop');state=loop.read_json(root/'LOOP14_STATE.json')
cyc=state['cycles'][0];conv=cyc['phases']['convert']['receipt']
manifest=loop.read_json(pathlib.Path(conv['pairs'])/'C4_EPISODES_MANIFEST.json');rows=[]
for entry in manifest['episodes']:
 p=pathlib.Path(loop.guard(entry['source']))/'summary.json'
 rows.append({{'entry':entry,'summary':loop.read_json(p),'summary_sha256':loop.sha256(p)}})
credits=[]
for world in cyc['phases']['collect']['receipt']['worlds']:
 p=pathlib.Path(world['artifact'])/'SUMOCREDIT_POINTS.jsonl'
 credits.append({{'world':world['case'],'path':str(p),'sha256':loop.sha256(p),
 'rows':[json.loads(s) for s in p.read_text().splitlines() if s.strip()]}})
print(json.dumps({{'rows':rows,'credits':credits,'collection':cyc['phases']['collect']['receipt'],
 'conversion':conv,'returns_receipt':loop.read_json(pathlib.Path(conv['returns']).with_suffix('.receipt.json')),
 'returns_sha256':loop.sha256(conv['returns']),'ledger_sha256':loop.sha256(root/'cycle_ledger.jsonl'),
 'policy':state['policy'],'ledger':[json.loads(s) for s in (root/'cycle_ledger.jsonl').read_text().splitlines()]}}))
'''
    text=cf.remote('gpuR',code)
    OUT.mkdir(exist_ok=True)
    (OUT/'CYCLE0_INPUTS.json').write_text(text,encoding='utf-8')
    print(json.dumps({'rows':len(json.loads(text)['rows'])}))


def stop():
    code = f'''
import pathlib,json,datetime,hashlib
a=pathlib.Path({A!r});root=pathlib.Path('/root/main14_loop')
state=json.loads((root/'LOOP14_STATE.json').read_text())
assert state['cycles'][-1]['phases']['wm_ft']['status']=='failed'
assert (a/'WAIT_OWNER.json').is_file()
doc={{'task':'S-MAINFIX','status':'WAIT_OWNER','t':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'failures':['Driver-owned WM timeout after5400s at2026-10-26T10:59:03Z',
 'Local staging WinError206 before SSH process creation: command length exceeded'],
 'new_remote_source_config_state_changes':False,'new_sumo_sim_s':0,'new_background_pids':[],
 'continuation':'Explicit owner direction required before shorter-transport staging, testing, operational amendment, official wm_ft retry and WM/check overlap',
 'registration_sha256':state['registration_sha256'],
 'config_sha256':hashlib.sha256((a/'CONFIG.json').read_bytes()).hexdigest(),
 'preserved_wait_owner':json.loads((a/'WAIT_OWNER.json').read_text())}}
target=a/'MAINFIX_WAIT_OWNER.json';assert not target.exists()
target.write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
'''
    save('WAIT_OWNER.json', cf.remote('gpuR', code))


def safety():
    code = f'''
import pathlib,json,sys,hashlib,os,numpy as np
a=pathlib.Path({A!r});r=a/'runtime';sys.path[:0]=[str(r/'scripts/c4'),str(r),str(r/'src')]
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import loop14 as loop
from traffic_shepherd.metrics import braking_effort
from traffic_shepherd.c4.env import SPEED_LIMIT_MPS
root=pathlib.Path('/root/main14_loop');state=loop.read_json(root/'LOOP14_STATE.json')
cyc=state['cycles'][0];conv=cyc['phases']['convert']['receipt'];pairs=pathlib.Path(conv['pairs'])
manifest=loop.read_json(pairs/'C4_EPISODES_MANIFEST.json');rows=[]
for entry in manifest['episodes']:
 assert not 964000<=int(entry['seed'])<=964199
 ep=pathlib.Path(loop.guard(pairs/entry['dir']));summary=loop.read_json(pathlib.Path(entry['source'])/'summary.json')
 times=np.load(ep/'time_s.npy',allow_pickle=False)
 kin=np.load(ep/'veh_kin.npy',mmap_mode='r',allow_pickle=False)
 mask=np.load(ep/'veh_mask.npy',mmap_mode='r',allow_pickle=False).astype(bool)
 vid=np.load(ep/'vid.npy',mmap_mode='r',allow_pickle=False)
 selected=np.flatnonzero((times>summary['warmup_s']) & (times<=summary['warmup_s']+summary['horizon_s']))
 assert len(selected)==600 and np.allclose(np.diff(times),0.5)
 brakes=sum(braking_effort(kin[t,:,5][mask[t]],0.5)['braking_vehicle_seconds'] for t in selected)
 road=float(sum(0.5*np.maximum(0,1-kin[t,:,4][mask[t]].astype(np.float64)/SPEED_LIMIT_MPS).sum() for t in selected))
 present_seconds=float(mask[selected].sum()*0.5)
 assert np.isclose(road,summary['ledger']['speed_deficit_vehicle_seconds'],rtol=1e-6,atol=1e-3),(entry['episode'],road)
 row={{'pair_key':entry['pair_key'],'kind':entry['branch']['kind'],'seed':entry['seed'],
 'braking_vehicle_seconds':float(brakes),'road_reconstructed':road,'road_authoritative':summary['ledger']['speed_deficit_vehicle_seconds'],
 'observed_vehicle_seconds':present_seconds,'postbranch_frames':len(selected),
 'array_sha256':{{name:loop.sha256(ep/name) for name in ('time_s.npy','veh_kin.npy','veh_mask.npy','vid.npy')}}}}
 rows.append(row)
with np.load(conv['returns'],allow_pickle=False) as z:
 returns={{name:z[name].tolist() for name in ('returns','seeds','requested_population','horizon_s','accounting','terminal_semantics','policy_checkpoint')}}
receipt=loop.read_json(pathlib.Path(conv['returns']).with_suffix('.receipt.json'))
assert np.allclose(-np.array(returns['returns']),[e['J'] for e in receipt['entries']],rtol=0,atol=0)
summaries={{entry['summary']:entry['sha256'] for entry in receipt['entries']}}
checks=[]
for world in cyc['phases']['collect']['receipt']['worlds']:
 artifact=pathlib.Path(world['artifact']);hashes=loop.read_json(artifact/'HASHES.json')
 for rel,sha in hashes.items():
  if rel.endswith('/summary.json') or rel in ('SUMOCREDIT_POINTS.jsonl','GATE_CALIBRATION.jsonl'):
   assert loop.sha256(artifact/rel)==sha;checks.append(str(artifact/rel))
print(json.dumps({{'rows':rows,'returns':returns,'returns_sha256':loop.sha256(conv['returns']),
 'speed_limit_mps':SPEED_LIMIT_MPS,'brake_metric_source_sha256':loop.sha256(r/'traffic_shepherd/metrics.py'),
 'verified_artifact_files':checks,'manifest_sha256':loop.sha256(pairs/'C4_EPISODES_MANIFEST.json')}}))
'''
    text = cf.remote('gpuR', code, timeout=300)
    OUT.mkdir(exist_ok=True)
    (OUT / 'CYCLE0_SAFETY.json').write_text(text,encoding='utf-8')
    doc=json.loads(text)
    print(json.dumps({'rows':len(doc['rows']),'example':doc['rows'][0],
                      'verified_artifact_files':len(doc['verified_artifact_files'])}))


def status():
    code = f'''
import pathlib,json,subprocess,sys
a=pathlib.Path({A!r});r=a/'runtime';sys.path[:0]=[str(r/'scripts/c4')]
import loop14 as loop,loop14_fast as fast
cfg=loop.load_config(a/'CONFIG.json');s=loop.load_state(cfg)
print(json.dumps({{'wm_timeout_h':cfg['wm']['timeout_h'],'overlap':cfg['fastloop']['overlap_policy'],
 'wm_phase':s['cycles'][-1]['phases']['wm_ft'],'accepted_wm':s['accepted_wm'],
 'charged_sim_s':s['sim_s_charged'],'state_sha256':loop.sha256(loop.state_path(cfg)),
 'driver93301':fast.process_token(93301),'trainer94892':fast.process_token(94892),
 'wait_owner':loop.read_json(a/'WAIT_OWNER.json'),'mainfix_wait_owner':loop.read_json(a/'MAINFIX_WAIT_OWNER.json'),
 'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)}}))
'''
    save('FINAL_STATUS.json', cf.remote('gpuR',code))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['extract','stop','safety','status'])
    globals()[ap.parse_args().action]()
