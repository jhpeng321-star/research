"""Sequential transport, SHA-bound launch, and20-minute S-C2H60B verification."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import ROOT, remote

REGISTER = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
FILES = ['tools/loopresume/c2h60b.py', 'tools/loopresume/c2h60b_box.py',
         'configs/c4_policy14_c2h60b.toml', 'src/wms/control/policy14_cli.py',
         'src/wms/control/policy14_recovery.py', 'tests/unit/test_policy14_recovery.py']


def stage():
    payload = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in FILES}
    remote('gpuA', PRELUDE + f'''
import base64
assert not (r/'C2H60B_LAUNCH.json').exists()
old=loop.read_json(r/'C2H60_LAUNCH.json'); assert not previous.live(old)
backup=r/'c2h60b_before'; backup.mkdir(exist_ok=False)
for name in ('CONFIG.json','REGISTRATION.json','loop/LOOP14_STATE.json'):
 target=backup/'arm'/name; target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes((a/name).read_bytes())
source={{}}
for name,encoded in {payload!r}.items():
 p=root/name; assert p.parent.is_dir()
 if p.exists():
  target=backup/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(p.read_bytes())
 p.write_bytes(base64.b64decode(encoded)); source[name]=loop.sha256(p)
for name in ('tools/loopresume/c2h60_box.py','configs/c4_policy14_c2h60.toml',
             'configs/c4_policy14_c2prep.toml','src/wms/control/paircrit.py'):
 source[name]=loop.sha256(root/name)
loop.write_atomic(r/'C2H60B_SOURCE.json',source)
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'CUDA_VISIBLE_DEVICES':'',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
 'NATIVEINIT_BASELINE':str(r/'nativeinit_baseline')}}
env.pop('C4_ALLOWED_ROOTS',None)
cmd=[sys.executable,'-B','-m','pytest','-q','-ra','-p','no:cacheprovider','--import-mode=importlib',
 '--basetemp',str(r/'pytest_c2h60b'),'tests/unit/test_policy14_recovery.py',
 'tests/unit/test_c2h60.py','tests/unit/test_nativeinit.py','tests/unit/test_c2prep.py',
 'tests/unit/test_loop14.py']
result=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True)
(r/'C2H60B_TESTS.log').write_text(result.stdout+result.stderr)
doc={{'t':loop.now(),'exit':result.returncode,'source_sha256':loop.sha256(r/'C2H60B_SOURCE.json'),
 'log_sha256':loop.sha256(r/'C2H60B_TESTS.log')}}
loop.write_atomic(r/'C2H60B_TESTS.json',doc)
print(result.stdout+result.stderr); print(json.dumps(doc)); raise SystemExit(result.returncode)
''', timeout=240)


def launch():
    text = REGISTER.read_text(encoding='utf-8')
    assert '## S-C2H60B — prospective' in text
    markdown_sha = hashlib.sha256(REGISTER.read_bytes()).hexdigest()
    remote('gpuA', PRELUDE + f'''
import torch
assert not (r/'C2H60B_LAUNCH.json').exists()
source=loop.read_json(r/'C2H60B_SOURCE.json'); tests=loop.read_json(r/'C2H60B_TESTS.json')
assert tests['exit']==0 and tests['source_sha256']==loop.sha256(r/'C2H60B_SOURCE.json')
for name,sha in source.items(): assert loop.sha256(root/name)==sha,name
old=loop.read_json(r/'C2H60_LAUNCH.json')
samples=[json.loads(line) for line in (r/'C2H60_MEMORY.jsonl').read_text().splitlines()]
child=samples[-1]['child']
assert not previous.live(old) and not previous.live(child)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
failed=a/'loop/policy/c02.h60'; cause=loop.read_json(failed/'RUN_SUMMARY.json')
assert cause['status']=='failed' and 'OutOfMemoryError: CUDA out of memory' in cause['error']
interrupted=a/'loop/policy/c02.h60b'
summary=loop.read_json(interrupted/'RUN_SUMMARY.json')
stop=loop.read_json(r/'C2H60B_STOP_REQUEST.json')
assert not previous.live(stop['coordinator']) and not previous.live(stop['child'])
assert summary['error']=='RuntimeError: own-run STOP requested; checkpoint saved'
saved=interrupted/'policy_latest.pt'; ck=torch.load(saved,map_location='cpu',weights_only=True)
assert ck['update_count']>=240 and ck['coopw']['anchor']==0
saved_update=ck['update_count']
assert summary['updates'][-1]['update']==saved_update
best=interrupted/'policy_native_best.pt'
assert loop.sha256(best)=='21c04bdc0f3821c4c05e9ce0bf85db919848fb39ca57d2cfff1fd80082ceead4'
required=('policy','group_encoder','critic','target_critic','actor_optimizer','critic_optimizer','rng','sampler','paircrit')
def finite(v):
 if isinstance(v,torch.Tensor): return bool(torch.isfinite(v).all())
 if isinstance(v,dict): return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)): return all(finite(x) for x in v)
 return True
assert all(k in ck and finite(ck[k]) for k in required)
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); loop.check_registration(cfg)
assert state['sim_s_charged']==176473 and state['cycles'][2]['phases']['policy']['status']=='failed'
recovery={{'summary':str(interrupted/'RUN_SUMMARY.json'),'checkpoint':str(saved),
 'config':str(root/'configs/c4_policy14_c2h60b.toml'),'gate_panel':str(a/'gate_panel.json'),
 'saved_update':saved_update,'phase_updates':150,'seed':965142,'checkpoint_every_update':True,
 'retry_outputs':[str(a/'loop/policy/c02.h60b.r2')],
 'cause_summary':str(failed/'RUN_SUMMARY.json'),'cause_summary_sha256':loop.sha256(failed/'RUN_SUMMARY.json'),
 'best_update':240,'best_checkpoint':str(best),'best_checkpoint_sha256':loop.sha256(best),
 'real_returns_sha256':loop.sha256(state['real_returns'])}}
for key in ('summary','checkpoint','config','gate_panel'): recovery[key+'_sha256']=loop.sha256(recovery[key])
loop.write_atomic(r/'C2H60B_RECOVERY.json',recovery)
amendment={{'id':'S-C2H60B','t':loop.now(),'authority':'main2026-10-27 12:45 fix-and-retry',
 'markdown_registration_sha256':{markdown_sha!r},'failed_processes':[old,child,stop['coordinator'],stop['child']],
 'failed_summary_sha256':recovery['cause_summary_sha256'],'failure':'allocator-cap CUDA OOM; operational',
 'saved_checkpoint':str(saved),'saved_checkpoint_sha256':recovery['checkpoint_sha256'],
 'saved_update':saved_update,'retained_updates':saved_update-233,'original_unsaved_updates_replayed':[241,249],
 'original_phase_start':233,'original_phase_end':383,'logical_updates':150,'remaining_updates':383-saved_update,
 'gpu_cap':0.92,'allocator':'expandable_segments:True','batch_starts':ck['config']['batch_size'],
 'checkpoint_every_update':True,'warmup_repeated':False,'share_selection_repeated':False,
 'horizon_s':60,'paircrit':True,'native_gate':'unchanged','charged':176473,'cap':210000,
 'after_gate':'WAIT_OWNER_BUDGET unless owner-approved SHA-bound registered raise',
 'fallback':'if resource OOM persists, record/reduce starts batch; never horizon'}}
binding={{'t':loop.now(),'source':source,'tests':tests,'amendment':amendment,
 'recovery_sha256':loop.sha256(r/'C2H60B_RECOVERY.json'),
 'previous_resume':summary['resumed_from']['checkpoint'],
 'failed_processes':[old,child,stop['coordinator'],stop['child']],
 'wm_sha256':state['accepted_wm']['sha256'],'gate_sha256':loop.sha256(a/'horizon_gate_current.json'),
 'pairing_source_policy_sha256':ck['paircrit']['policy_checkpoint_sha256']}}
loop.write_atomic(r/'C2H60B_BINDINGS.json',binding)
with (a/'driver_c2h60b.log').open('xb') as log:
 proc=subprocess.Popen([sys.executable,'-B','-u',str(root/'tools/loopresume/c2h60b_box.py')],cwd=root,
 env={{**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}},
 start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
launch={{'pid':proc.pid,'start_ticks':loop.fast_module().process_token(proc.pid),'t':loop.now(),
 'task':'S-C2H60B','bindings_sha256':loop.sha256(r/'C2H60B_BINDINGS.json')}}
loop.write_atomic(r/'C2H60B_LAUNCH.json',launch); print(json.dumps(launch))
''')


def progress():
    remote('gpuA', PRELUDE + '''
import datetime
from tools.loopresume.c2h60_box import verify_updates
launch=loop.read_json(r/'C2H60B_LAUNCH.json'); doc=loop.read_json(r/'C2H60B_PROGRESS.json')
print('PROGRESS',json.dumps(doc)); print('COORDINATOR_LIVE',previous.live(launch))
out=pathlib.Path(doc['output'])
rows=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()] if (out/'updates.jsonl').exists() else []
if rows:
 verify_updates(rows,doc['saved_update'],len(rows))
 samples=[json.loads(line) for line in (r/'C2H60B_MEMORY.jsonl').read_text().splitlines()]
 mem=[s['used_memory_mib'] for s in samples if 'used_memory_mib' in s]
 peaks=[json.loads(line) for line in (out/'cuda_peaks.jsonl').read_text().splitlines()]
 elapsed=time.time()-doc['started_unix']; rate=len(rows)*3600/elapsed
 observation={'t':loop.now(),'new_updates':len(rows),'phase_updates':doc['saved_update']-233+len(rows),
  'last_update':rows[-1]['update'],'finite_losses_and_gradients':True,'elapsed_s':elapsed,
  'updates_per_hour':rate,'seconds_per_update':elapsed/len(rows),
  'eta_utc':datetime.datetime.fromtimestamp(time.time()+(383-rows[-1]['update'])*elapsed/len(rows),datetime.timezone.utc).isoformat(),
  'sampled_process_peak_mib':max(mem),'memory_watch_s':samples[-1]['unix']-samples[0]['unix'],
  'memory_samples':len(samples),'cuda_peak_allocated_bytes':max(p['cuda_peak_allocated_bytes'] for p in peaks),
  'cuda_peak_reserved_bytes':max(p['cuda_peak_reserved_bytes'] for p in peaks),
  'gate':loop.read_json(out/'gate_panel_result.json') if (out/'gate_panel_result.json').exists() else None,
  'coordinator':launch,'child':doc['child'],'coordinator_live':previous.live(launch),
  'child_live':previous.live(doc['child']),'charged':loop.load_state(loop.load_config(a/'CONFIG.json'))['sim_s_charged']}
 loop.write_atomic(r/'C2H60B_OBSERVATION.json',observation); print('OBSERVATION',json.dumps(observation))
print('DRIVER_LOG',(a/'driver_c2h60b.log').read_text()[-2500:])
if (out/'RUN_SUMMARY.json').exists():
 summary=loop.read_json(out/'RUN_SUMMARY.json'); print('SUMMARY',summary['status'],summary.get('error'))
''')


def verify():
    folder = ROOT / 'tools/loopresume/receipts_c2h60b'
    folder.mkdir(exist_ok=True)
    remote('gpuA', PRELUDE + '''
obs=loop.read_json(r/'C2H60B_OBSERVATION.json')
assert obs['new_updates']>=15 and obs['memory_watch_s']>=1200
assert obs['coordinator_live'] and obs['child_live'] and obs['charged']==176473
assert obs['finite_losses_and_gradients'] and obs['gate']['passed']
for name in ('SOURCE','TESTS','RECOVERY','BINDINGS','LAUNCH','OBSERVATION'):
 p=r/('C2H60B_'+name+'.json'); print(name,loop.sha256(p))
loop.write_atomic(r/'C2H60B_VERIFIED.json',obs)
loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C2H60B_verified',**obs})
loop.append(r/'C1_EVENTS.jsonl',{'event':'S-C2H60B_verified',**obs})
print('VERIFIED',json.dumps(obs))
''')
    # Copy only explicit compact receipts; no data trees are traversed.
    import subprocess
    for name in ('SOURCE', 'TESTS', 'RECOVERY', 'BINDINGS', 'LAUNCH', 'VERIFIED'):
        subprocess.run(['scp', 'gpuA:/root/autodl-tmp/c4/restart20261026/C2H60B_' + name + '.json',
                        str(folder / (name + '.json'))], check=True)
    (folder / 'REGISTRATION_AMENDMENT.md').write_text(
        REGISTER.read_text(encoding='utf-8').split('## S-C2H60B — prospective', 1)[1], encoding='utf-8')
    print(json.dumps({'receipt_folder': str(folder)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'launch', 'progress', 'verify'])
    globals()[parser.parse_args().action]()
