"""Validate and reduce S-C0DIAG-1 artifact evidence locally; never starts simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
load=lambda n:json.loads((HERE/n).read_text(encoding='utf-8'))
tapes=load('tapes.json'); replay=load('replay.json'); enriched=load('replay_enriched.json')
assert len(tapes)==52
assert {r['seed'] for r in tapes}==set(range(968129,968142))
assert len([r for r in replay if r['horizon']==10])==26
assert len([r for r in replay if r['horizon']==60 and r.get('requested_horizon',60)==60])==26
provs=[load('provenance_'+h+'.json') for h in ('gpuR','cpuP','cpuQ')]
assert provs[0]['source']==provs[1]['source']==provs[2]['source']
sha='fa2427772a49e9462c0be3172e402c515616a15e2930a263e1eafeacb23a0d5d'
assert all(p['collected_policy_sha']==sha for p in provs[1:])
assert load('audit.json')['policy_sha256']==sha
pooled={}
for arm in ('B','POLICY'):
    rs=[r for r in tapes if r['arm']==arm]
    weighted=lambda key,sub='mean':sum(r[key]['n']*r[key][sub] for r in rs if r[key]['n'])/sum(r[key]['n'] for r in rs)
    total=sum(r['speed']['n'] for r in rs)
    pooled[arm]={'cav_physics_rows':total,'speed':weighted('speed'),
        'crawl':sum(r['speed']['n']*r['crawl_fraction'] for r in rs)/total,
        'stop':sum(r['speed']['n']*r['stop_fraction'] for r in rs)/total,
        'lane_changes':sum(r['lane_changes'] for r in rs),'ramp_merges':sum(r['ramp_merges'] for r in rs)}
    if arm=='POLICY':
        pooled[arm].update(requested_mean=weighted('requested'),issued_mean=weighted('issued'),
            requested_negative_fraction=sum(r['requested']['n']*r['requested_negative_fraction'] for r in rs)/sum(r['requested']['n'] for r in rs),
            target_identity_maxerr=max(r['issued_target_identity_maxerr'] for r in rs),
            response_error_mean=weighted('response_minus_issued'))
        for b in range(4):
            n=sum(r['time_bins'][b]['speed']['n'] for r in rs)
            pooled[arm]['bin_'+str(b)]={
                'speed':sum(r['time_bins'][b]['speed']['n']*r['time_bins'][b]['speed']['mean'] for r in rs)/n,
                'crawl':sum(r['time_bins'][b]['speed']['n']*r['time_bins'][b]['crawl_fraction'] for r in rs)/n}
comparisons={}
for h in (10,60):
    rs=[r for r in replay if r['horizon']==h and r.get('requested_horizon',h)==h]
    wm=np.array([r['wm_policy_road']-r['wm_native_road'] for r in rs])
    su=np.array([r['sumo_policy_road']-r['sumo_native_road'] for r in rs])
    comparisons[h]={'n':len(rs),'wm_mean_delta':float(wm.mean()),'sumo_mean_delta':float(su.mean()),
                    'wm_worse':int((wm>0).sum()),'sumo_worse':int((su>0).sum()),'sign_agreement':float((np.sign(wm)==np.sign(su)).mean()),
                    'first_action_maxerr':max(r['first_executor_requested_maxerr'] for r in rs),
                    'first_lane_mismatches':sum(r['first_lane_mismatches'] for r in rs)}
    assert comparisons[h]['wm_worse']==comparisons[h]['sumo_worse']==26
    assert comparisons[h]['first_action_maxerr']<1e-5
    assert comparisons[h]['first_lane_mismatches']==0
own=load('ownership.json')[0]
base=next(r for r in enriched if r['seed']==968132 and r['horizon']==300)
for k in ('wm_policy_road','wm_native_road','wm_cav_speed','wm_cav_stop','wm_physical_ramp_merges','wm_lane_changes'):
    assert own[k]==base[k],(k,own[k],base[k])
cfg=provs[0]['configuration']
assert cfg['collect']['sumocredit']['gated_arm'] is False
assert cfg['collect']['sumocredit']['gate_candidates']==8
assert all(not w['collect']['gate_table_used'] for w in provs[0]['world_journal']['worlds'].values())
summary={'pooled_tapes':pooled,'cohort_comparisons':comparisons,'representative_300s':base,
         'formal_ramp_ownership_ablation':'identical on 968132@360; no automatic CAV ramp merges in that replay',
         'source_parity_files':len(provs[0]['source']),'policy_sha256':sha,
         'actor_fixed_states':load('actor.json'),
         'critic':load('critic.json'),
         'limits':['entries OFF in matched cohort replays; do not equate with full-network source-inclusive J',
                   'seeded initial actor is a reconstruction, not a saved u0 checkpoint',
                   'one low-share 300 s request correctly capped to 60 s; excluded from 300 s evidence',
                   'lane-churn causal share not separately identified',
                   'recorded invalid junction lanes retained in replay initial state; causal histories decoded from tape'],
         'artifacts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.json') if p.name!='SUMMARY.json'}}
(HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
words=len((HERE.parents[1]/'C0DIAG.md').read_text(encoding='utf-8').split())
assert words<=900,words
print('PASS: 52 tape rows, 26 paired starts at each horizon, source/checkpoint parity, ramp ablation, report word cap.')
print('REPORT_WORDS',words)
print(json.dumps({'pooled_tapes':pooled,'cohort_comparisons':comparisons},indent=2))
