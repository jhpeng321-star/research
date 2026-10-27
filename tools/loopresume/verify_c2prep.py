"""Read-only source/gate/budget verification; write a compact GPU handoff receipt."""
from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import remote


def main():
    remote('gpuA', PRELUDE+'''
from wms.control.nativeinit import verify_collection_gate
from tools.loopresume.c2prep_common import POLICY_SHA,budget_need
cfg=loop.load_config(a/'CONFIG.json'); s=loop.load_state(cfg); reg=loop.check_registration(cfg)
os.chdir(root); os.environ['C4_ALLOWED_ROOTS']=':'.join(cfg['policy']['allowed_roots'])
binding=loop.read_json(r/'C2PREP_BINDINGS.json')
for name,sha in binding['source'].items(): assert loop.sha256(root/name)==sha,name
tests=loop.read_json(r/'C2PREP_TESTS.json'); assert tests['exit']==0
assert tests['source_sha256']==loop.sha256(r/'C2PREP_SOURCE.json')
stop=a/'loop/policy/c02/STOP'
if stop.exists() and loop.read_json(stop).get('event')=='S-C2H60_cooperative_stop_requested':
 import torch
 from types import SimpleNamespace
 from wms.control.paircrit import PairedProvider,PairCritic,TailValue
 from wms.control.real_returns14 import NPZRealReturnProvider,ScaledTeamCritic
 torch.set_num_threads(1)
 launch=loop.read_json(r/'C2PREP_LAUNCH.json'); assert not previous.live(launch)
 stopped=loop.read_json(stop); assert not previous.live(stopped['child'])
 assert s['sim_s_charged']==176473 and reg['budget']['sim_s_total']==210000
 assert s['cycles'][2]['phases']['policy'].get('task')=='S-C2H60'
 assert all(s['cycles'][2]['phases'][n]['status']=='pending' for n in ('collect','convert','wm_ft','check'))
 ledger=[json.loads(row) for row in (a/'loop/seed_ledger.jsonl').read_text().splitlines()]
 assert not any(row.get('cycle')==2 for row in ledger)
 out=a/'loop/policy/c02'; summary=loop.read_json(out/'RUN_SUMMARY.json')
 assert summary['error']=='RuntimeError: own-run STOP requested; checkpoint saved'
 assert not (out/'native_gate_result.json').exists()
 checkpoint=torch.load(out/'policy_latest.pt',map_location='cpu',weights_only=True)
 assert checkpoint['update_count']==233
 critic=TailValue(ScaledTeamCritic(entry_backlog=True,value_scale=10000.),paired=True)
 critic.load_state_dict(checkpoint['critic']); critic.eval()
 converted=s['cycles'][1]['phases']['convert']['receipt']
 provider=PairedProvider(NPZRealReturnProvider(converted['returns']),
  NPZRealReturnProvider(converted['native_returns']),converted['pairing_receipt'],POLICY_SHA)
 monitor=object.__new__(PairCritic); monitor.provider=provider
 monitor.config=SimpleNamespace(policy_sha256=POLICY_SHA)
 accuracy=monitor.diagnostics(SimpleNamespace(device='cpu',critic=critic))
 curve=[json.loads(row) for row in (out/'native_curve.jsonl').read_text().splitlines()]
 smokes=[]
 for h in (60,30):
  smoke=loop.read_json(r/f'C2PREP_SMOKE_{h}.json'); updates=smoke.pop('updates')
  smoke['update_seconds']=[row['seconds'] for row in updates]; smokes.append(smoke)
 gate=loop.read_json(out/'gate_panel_result.json')
 doc={'verification':'PASS','status':'SUPERSEDED_BY_S-C2H60','t':loop.now(),
  'wm1_check':s['cycles'][1]['phases']['check']['receipt'],'accepted_wm':s['accepted_wm'],
  'own_driver':launch,'own_driver_live':False,'own_actor':stopped['child'],'own_actor_live':False,
  'cooperative_stop':stopped,'updates_completed':53,'checkpoint_update':233,
  'checkpoint_sha256':loop.sha256(out/'policy_latest.pt'),'summary_sha256':loop.sha256(out/'RUN_SUMMARY.json'),
  'horizon_used_s':30,'smokes':smokes,'final_native_gate':'NOT_PRODUCED: superseded before completion',
  'progress_gate':gate,'latest_native_panel':curve[-1],'paircrit_at_update233':accuracy,
  'critic_warmup':summary['critic_warmup'],'budget':budget_need(),'collection_dispatched':False,
  'successor_phase_observed':s['cycles'][2]['phases']['policy'],
  'tests':tests,'source_manifest_sha256':loop.sha256(r/'C2PREP_SOURCE.json'),
  'selection_sha256':loop.sha256(r/'C2PREP_SELECTION.json'),
  'registration_sha256_observed':s['registration_sha256']}
 loop.write_atomic(r/'C2PREP_SUPERSEDED.json',doc)
 print(json.dumps(doc)); print('SUPERSEDED_SHA256',loop.sha256(r/'C2PREP_SUPERSEDED.json'))
 raise SystemExit(0)
assert s['stopped']=='WAIT_OWNER_BUDGET' and s['sim_s_charged']==176473
assert s['registration_sha256']==loop.sha256(cfg['registration'])
assert reg['budget']['sim_s_total']==210000
c1=s['cycles'][1]['phases']; c2=s['cycles'][2]['phases']
assert c1['check']['receipt']['decision']=='ACCEPT' and c2['policy']['status']=='done'
assert all(c2[n]['status']=='pending' and c2[n]['attempt']==0 for n in ('collect','convert','wm_ft','check'))
assert not (a/'loop/worlds_c02.json').exists()
ledger=[json.loads(row) for row in (a/'loop/seed_ledger.jsonl').read_text().splitlines()]
assert not any(row.get('cycle')==2 for row in ledger)
native=verify_collection_gate(s['policy'])
assert native['passed'] and native['wm_sha256']==[s['accepted_wm']['sha256']]
assert native['wm_config_sha256']==[loop.sha256(s['accepted_wm']['config'])]
result=loop.read_json(r/'C2PREP_RESULT.json'); selection=loop.read_json(r/'C2PREP_SELECTION.json')
assert result['budget']==budget_need() and not result['collection_dispatched']
assert result['native_gate']==native
assert selection['horizon_s']<=60 and selection['wm1_check']['decision']=='ACCEPT'
assert c2['policy']['started']>=selection['t']
summary=loop.read_json(pathlib.Path(s['policy']['checkpoint']).with_name('RUN_SUMMARY.json'))
assert summary['resumed_from']['sha256']==POLICY_SHA and summary['resumed_from']['paircrit_boundary_reset']
assert summary['gpu_memory_fraction']==0.65
assert summary['paircrit_monitor']['source_policy_sha256']==POLICY_SHA
assert summary['paircrit_monitor']['fit']['n']==22 and summary['paircrit_monitor']['development']['n']==4
smokes=[{k:v for k,v in smoke.items() if k!='updates'} for smoke in selection['smokes']]
for compact,original in zip(smokes,selection['smokes']):
 compact['update_seconds']=[row['seconds'] for row in original.get('updates',[])]
launch=loop.read_json(r/'C2PREP_LAUNCH.json')
doc={'verification':'PASS','t':loop.now(),'wm1_check':c1['check']['receipt'],
 'accepted_wm':s['accepted_wm'],'horizon_used_s':selection['horizon_s'],
 'actual_training_horizons':result['actual_training_horizons'],'smokes':smokes,
 'native_gate':native,'policy':s['policy'],'paircrit_monitor':summary['paircrit_monitor'],
 'critic_warmup':summary['critic_warmup'],'budget':result['budget'],
 'collection_dispatched':False,'status':s['stopped'],'driver':launch,'driver_live':previous.live(launch),
 'tests':tests,'source_manifest_sha256':loop.sha256(r/'C2PREP_SOURCE.json'),
 'registration_sha256':s['registration_sha256'],'result_sha256':loop.sha256(r/'C2PREP_RESULT.json'),
 'selection_sha256':loop.sha256(r/'C2PREP_SELECTION.json'),
 'policy_phase':c2['policy']}
loop.write_atomic(r/'C2PREP_VERIFIED.json',doc)
print(json.dumps(doc)); print('VERIFIED_SHA256',loop.sha256(r/'C2PREP_VERIFIED.json'))
''')


if __name__ == '__main__':
    main()
