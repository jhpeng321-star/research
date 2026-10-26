"""Capture/verify bounded S-C1COLLECT3 handoff receipts without dispatch or process signals."""
import argparse
import ast
import hashlib
import json

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import ROOT, remote


def running():
    remote('gpuA', PRELUDE+'''
doc=loop.read_json(r/'C1COLLECT3_STATUS.json')
assert doc['live'] and doc['main_priority_active'] and doc['canaries']['cpuP']['verified']
assert ({p['box'] for p in doc['probes'].values() if p['verified']}
 | {p['box'] for p in doc['canaries'].values() if p['verified']})=={'cpuP','cpuQ'}
assert not (r/'C1COLLECT3_RUNNING.json').exists()
loop.write_atomic(r/'C1COLLECT3_RUNNING.json',doc)
print(json.dumps({'t':doc['t'],'running_evidence_sha256':loop.sha256(r/'C1COLLECT3_RUNNING.json'),
 'driver':doc['pid'],'charged':doc['charged'],'main_priority_active':True,'numerical_output_both_cpus':True}))
''')


def capture():
    remote('gpuA', PRELUDE+'''
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg)
j=loop.read_json(a/'loop/worlds_c01.json'); observed=loop.read_json(r/'C1COLLECT3_STATUS.json')
assert time.time()-__import__('datetime').datetime.fromisoformat(observed['t'].replace('Z','+00:00')).timestamp()<120
assert not (r/'C1COLLECT3_RESULT.json').exists()
binding=loop.read_json(r/'C1COLLECT3_RECOVERY_BINDINGS.json')
assert state['sim_s_charged']<=210000 and j['charged']<=73866
ledger=[json.loads(line) for line in (a/'loop/seed_ledger.jsonl').read_text().splitlines()]
new=[row for row in ledger if row.get('event')=='world_attempt' and row.get('cycle')==1
 and row.get('attempt')==cfg['collect']['startup_canary']['next_attempts'][str(row['seed'])]]
assert all(968157<=row['seed']<=968169 for row in new)
assert len({row['seed'] for row in new})==len(new)
long=sum(row.get('long_credit_sim_s_charged',0) for row in ledger if row.get('cycle')==1)
assert long<=15000
ph=state['cycles'][1]['phases']
pause=loop.read_json(a/'OPERATIONAL_PAUSED.json') if (a/'OPERATIONAL_PAUSED.json').exists() else None
from wms.control.nativeinit import artifact_world_delta
deltas={seed:artifact_world_delta(world['artifact'],int(cfg['collect']['branch_points'])) for seed,world in j['worlds'].items()}
doc={'id':'S-C1COLLECT3','observed_utc':loop.now(),'status':('OPERATIONAL_PAUSED' if pause else ('WAIT_OWNER' if state.get('stopped') else ph['collect']['status'])),
 'preflight':observed['preflight'],'manifest_sha256':'fe722b8941f50ff23463f3d673d0e07fa5e826c2a6e85ea46eac8108167f7fcb',
 'manifest_git_blob':'dd6dc191f0de71d6056a0ccb820a808ab77fe403',
 'collector_import_onnx_and_learned_policy_load_both_pass':True,
 'checker_change':binding['checker_change'],'driver':loop.read_json(a/'PID.json'),
 'driver_live':previous.live(loop.read_json(a/'PID.json')),'canary':j.get('startup_canaries',{}),
 'new_world_attempts':new,'remaining_dispatched':len(new)>1,'completed_worlds':list(j['worlds']),
 'charged_total':state['sim_s_charged'],'additional_charged':state['sim_s_charged']-139540,
 'total_cap':210000,'remaining':210000-state['sim_s_charged'],'cycle_charged':j['charged'],
 'cycle_attempt_allowance':73866,'long_credit_charged_included':long,'long_credit_attempt_allowance':15000,
 'main_priority_active':any(c['phases'][n]['status']=='running' for c in state['cycles'] for n in ('collect','check')),
 'priority_basis':'existing main-loop watcher predicate: any collect/check phase running',
 'guard':j.get('native_smoke'),'per_world_policy_minus_native_veh_s':deltas,
 'phase_status':{k:v['status'] for k,v in ph.items()},
 'projected_long_credit_for_all13':15600,'long_credit_shortfall_for_all13':600,
 'pause':pause,'exact_collection_refusal':ph['collect'].get('refused'),
 'stopped':state.get('stopped'),'simulation_retry_armed':False,
 'next_action':'Main decision required: long-credit cap would need 15600 (+600) for all13. '
 'Retain four completed worlds and guard; remaining nine never dispatched. No renewed launch armed.',
 'downstream':'driver executes convert -> wm_ft -> check after successful collection; no actor launch',
 'source':binding['source'],'tests':binding['tests'],
 'source_manifest_sha256':loop.sha256(r/'C1COLLECT3_RECOVERY_SOURCE.json'),
 'bindings_sha256':loop.sha256(r/'C1COLLECT3_RECOVERY_BINDINGS.json'),
 'registration_sha256':state['registration_sha256'],'config_sha256':loop.sha256(a/'CONFIG.json'),
 'preflight_sha256':binding['preflight'],'zero_dispatch_repair':binding['zero_dispatch_repair'],
 'running_evidence_sha256':loop.sha256(r/'C1COLLECT3_RUNNING.json'),
 'collection_running_evidence':loop.read_json(r/'C1COLLECT3_RUNNING.json'),
 'wm0_exact_incarnations_observed_only':{str(pid):loop.fast_module().process_token(pid) for pid in (62645,62910)}}
assert not doc['driver_live'] and doc['status']=='OPERATIONAL_PAUSED'
assert 'long credit attempts exceed 15000' in doc['exact_collection_refusal']
assert len(doc['completed_worlds'])==4 and len(new)==4 and doc['guard']['passed']
assert not any(p['collectors'] or p['sumo'] for p in observed['probes'].values())
reg=loop.check_registration(cfg)
reg.setdefault('operational_events',[]).append({'id':'S-C1COLLECT3_result','t':doc['observed_utc'],
 'status':doc['status'],'result':str(r/'C1COLLECT3_RESULT.json'),'charged':doc['charged_total'],
 'guard':doc['guard'],'exact_refusal':doc['exact_collection_refusal'],'simulation_retry_armed':False})
loop.write_atomic(pathlib.Path(cfg['registration']),reg)
state['registration_sha256']=loop.sha256(cfg['registration']); loop.save_state(cfg,state)
doc['registration_sha256']=state['registration_sha256']
loop.write_atomic(r/'C1COLLECT3_RESULT.json',doc)
previous.event('S-C1COLLECT3_handoff',result=str(r/'C1COLLECT3_RESULT.json'),sha256=loop.sha256(r/'C1COLLECT3_RESULT.json'),status=doc['status'])
loop.append(a/'loop/cycle_ledger.jsonl',{'event':'S-C1COLLECT3_handoff','t':loop.now(),'result':str(r/'C1COLLECT3_RESULT.json'),'sha256':loop.sha256(r/'C1COLLECT3_RESULT.json')})
compact={k:v for k,v in doc.items() if k not in ('collection_running_evidence','zero_dispatch_repair')}
compact['remote_result']='gpuA:'+str(r/'C1COLLECT3_RESULT.json')
compact['remote_result_sha256']=loop.sha256(r/'C1COLLECT3_RESULT.json')
print(json.dumps(compact))
''', timeout=120)


def verify():
    receipt = json.loads((ROOT/'tools/loopresume/receipts_c1collect3/RESULT.json').read_text())
    for name, sha in receipt['source'].items():
        if name == 'scripts/c4/loop14_fast.py':
            continue  # Remote-only one-line overlay; existing local work stays unmodified.
        data = (ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, name
        ast.parse(data, filename=name)
    remote('gpuA', PRELUDE+f'''
receipt={receipt!r}
assert loop.sha256(r/'C1COLLECT3_RESULT.json')==receipt['remote_result_sha256']
doc=loop.read_json(r/'C1COLLECT3_RESULT.json')
assert doc['charged_total']==receipt['charged_total'] and doc['status']==receipt['status']
for name,sha in receipt['source'].items(): assert loop.sha256(root/name)==sha,name
for box,sha in receipt['preflight_sha256'].items():
 assert loop.sha256(r/('C1COLLECT3_PREFLIGHT_'+box+'.json'))==sha
 preflight=loop.read_json(r/('C1COLLECT3_PREFLIGHT_'+box+'.json'))
 assert preflight['status']=='PASS' and preflight['collector_import'] and preflight['onnx']['loaded'] and preflight['learned_policy_loaded']
assert loop.sha256(r/'C1COLLECT3_RECOVERY_BINDINGS.json')==receipt['bindings_sha256']
assert loop.sha256(r/'C1COLLECT3_RUNNING.json')==receipt['running_evidence_sha256']
assert loop.sha256(a/'CONFIG.json')==receipt['config_sha256']
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); reg=loop.check_registration(cfg)
assert receipt['charged_total']<=state['sim_s_charged']<=210000
assert reg['budget']['cycle_attempt_allowances']['1']==73866
assert reg['collection']['cycle_long_attempt_allowances']['1']==15000
assert loop.read_json(r/'C1COLLECT3_RECOVERY_TESTS.json')['exit']==0
assert loop.sha256(r/'C1COLLECT3_RECOVERY_TESTS.log')==receipt['tests']['log_sha256']
assert len(doc['canary'])==1 and next(iter(doc['canary'].values()))['verified']
print(json.dumps({{'verified':True,'snapshot_status':doc['status'],'current_charged':state['sim_s_charged'],
 'current_driver_live':previous.live(loop.read_json(a/'PID.json')),
 'phase_status':{{k:v['status'] for k,v in state['cycles'][1]['phases'].items()}}}}))
''', timeout=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['running', 'capture', 'verify'], default='verify', nargs='?')
    globals()[parser.parse_args().action]()
