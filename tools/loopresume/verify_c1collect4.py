"""No-dispatch handoff checks; only verification receipts/events are written."""
from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import remote


def main():
    remote('gpuA', PRELUDE+'''
binding=loop.read_json(r/'C1COLLECT4_BINDINGS.json')
for name,sha in binding['source'].items(): assert loop.sha256(root/name)==sha,name
tests=loop.read_json(r/'C1COLLECT4_TESTS.json'); assert tests['exit']==0
assert tests['source_sha256']==loop.sha256(r/'C1COLLECT4_SOURCE.json')
running=loop.read_json(r/'C1COLLECT4_RUNNING.json')
assert running['driver_live'] and running['main_priority_active']
for box in ('cpuP','cpuQ'):
 assert any(p['box']==box and p['verified'] and p['outputs'] and p['sumo'] for p in running['probes'].values())
assert len(running['new_attempts'])==9 and running['charged']==176473
assert sorted(x['seed'] for x in running['new_attempts'])==list(range(968161,968170))
assert all(x['attempt']==1 and x['sim_s_charged']==2841 for x in running['new_attempts'])
assert running['long_credit_charged_included']==15600 and running['guard']==binding['guard']
continuation=loop.read_json(r/'C1COLLECT4_CONTINUATION.json')
assert continuation['status']=='PASS' and continuation['wm_steps']==40 and continuation['retry_limit']==1
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); reg=loop.check_registration(cfg)
assert reg['budget']['sim_s_total']==210000 and state['sim_s_charged']<=210000
assert reg['collection']['cycle_long_attempt_allowances']['1']==15600
assert state['sim_s_charged']==176473
assert not (a/'OPERATIONAL_PAUSED.json').exists() and not (a/'WAIT_OWNER.json').exists()
pid=loop.read_json(a/'PID.json'); live=previous.live(pid)
assert live or (r/'C1COLLECT4_COMPLETE.json').exists()
j=loop.read_json(a/'loop/worlds_c01.json')
doc={'id':'S-C1COLLECT4','observed_utc':loop.now(),'verification':'PASS',
 'driver':pid,'driver_live':live,'phases':{n:p['status'] for n,p in state['cycles'][1]['phases'].items()},
 'charged_total':state['sim_s_charged'],'additional_charged':25569,'total_cap':210000,'remaining':33527,
 'cycle_charged':j['charged'],'cycle_attempt_allowance':73866,'long_credit_charged_included':15600,
 'long_credit_attempt_allowance':15600,'remaining_nine_dispatched':True,
 'collection_verified_running_both_cpus':True,'running_evidence_utc':running['t'],
 'main_priority_active_at_running_verification':True,
 'main_priority_active_now':any(c['phases'][n]['status']=='running' for c in state['cycles'] for n in ('collect','check')),
 'completed_worlds':sorted(map(int,j['worlds'])),'guard':j['native_smoke'],
 'new_world_attempts':running['new_attempts'],'collector_pids_at_running_verification':
 {box:[pid for p in running['probes'].values() if p['box']==box for pid in p['collectors']] for box in ('cpuP','cpuQ')},
 'numerical_proof':{box:next(p for p in running['probes'].values() if p['box']==box and p['verified']) for box in ('cpuP','cpuQ')},
 'continuation':continuation,'operational_retry_used':(r/'C1COLLECT4_RETRY.json').exists(),
 'source':binding['source'],'tests':tests,'source_manifest_sha256':loop.sha256(r/'C1COLLECT4_SOURCE.json'),
 'bindings_sha256':loop.sha256(r/'C1COLLECT4_BINDINGS.json'),
 'registration_sha256':state['registration_sha256'],'running_evidence_sha256':loop.sha256(r/'C1COLLECT4_RUNNING.json'),
 'continuation_receipt_sha256':loop.sha256(r/'C1COLLECT4_CONTINUATION.json'),
 'eta_collection_end_utc':['2026-10-26T22:32:00Z','2026-10-26T22:40:00Z'],
 'eta_wm_ft_end_utc':['2026-10-27T01:00:00Z','2026-10-27T01:30:00Z'],
 'eta_basis':'Concurrent nine worlds on two boxes; prior four-world dispatch ~4min; WM0 40 steps summed 8531.35s (~2h22m), plus conversion/setup allowance',
 'remote_result':str(r/'C1COLLECT4_RESULT.json')}
if not (r/'C1COLLECT4_RESULT.json').exists(): loop.write_atomic(r/'C1COLLECT4_RESULT.json',doc)
else: loop.write_atomic(r/'C1COLLECT4_VERIFY_LATEST.json',doc)
previous.event('S-C1COLLECT4_handoff_verified',result_sha256=loop.sha256(r/'C1COLLECT4_RESULT.json'),
 charged=state['sim_s_charged'],running_evidence_sha256=doc['running_evidence_sha256'])
print(json.dumps(doc)); print('RESULT_SHA256',loop.sha256(r/'C1COLLECT4_RESULT.json'))
''', timeout=120)


if __name__ == '__main__':
    main()
