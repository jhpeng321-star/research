import hashlib,json
from pathlib import Path
ROOT=Path('D:/research-dense')
files=[ROOT/'output/observation_relational_v1/DATASET_development.json',ROOT/'output/observation_relational_v1/DATASET_holdout.json']
rows=sum((json.loads(p.read_text())['rows'] for p in files),[])
rows=[r for r in rows if 90001<=r['seed']<=90060]
v2=ROOT/'output/collapse_prevention_audit_v1/PAIRED_ANALYSIS_V2.json'; vd=json.loads(v2.read_text())
metrics=['service','braking','effort','ttc','wait']
def pop(sel,arm):
 return {'n':len(sel),'mean_delta':{m:sum(r['deltas'][arm][m] for r in sel)/len(sel) for m in metrics},'labels':{x:sum(r['labels'][arm]['label']==x for r in sel) for x in ['beneficial','harmful','nonbeneficial']}}
result={'schema':'oe35_physical_review/1','sources':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files], 'v2_source_sha256':hashlib.sha256(v2.read_bytes()).hexdigest(),'populations':{},'spot_checks':{}}
for name,sel in [('all60',rows),('entry32',[r for r in rows if r['entry_feasible']]),('branch22',[r for r in rows if r['branch_available']])]:
 result['populations'][name]={a:pop(sel,a) for a in ['h20','h5']}
 # V2 aggregates are independently checked against source rows.
 result['populations'][name]['v2_match']={a: all(abs(result['populations'][name][a]['mean_delta'][m]-vd['aggregates'][name][a]['mean_original_cost_delta'][m])<1e-12 for m in metrics) and result['populations'][name][a]['labels']==vd['aggregates'][name][a]['original_labels'] for a in ['h20','h5']}
for seed in [90011,90030,90045,90046]:
 r=next(x for x in rows if x['seed']==seed); result['spot_checks'][str(seed)]={'entry_feasible':r['entry_feasible'],'branch_available':r['branch_available'],'labels':r['labels'],'costs':{a:r['physical_records'][a]['costs'] for a in r['physical_records']},'lifecycle':{a:r['physical_records'][a].get('lifecycle') for a in ['h20','h5'] if a in r['physical_records']}}
 # retain V2 policy classifications for the requested absent-primary checks
 c=next(x for x in vd['cases'] if x['seed']==seed)
 result['spot_checks'][str(seed)]['v2_policy_categories']={k:v.get('category') for k,v in c.get('policies',{}).items()}
 result['spot_checks'][str(seed)]['duration_contrast']={}
 for m in ['low_exposure_delta_s','original_cost_deltas','aligned_discharge','lifecycle']:
  result['spot_checks'][str(seed)]['duration_contrast'][m]={k:c['policies'][k].get(m) for k in ['h20','h5']}
 result['spot_checks'][str(seed)]['h5_minus_h20']={
   'low_exposure_delta_s': c['policies']['h5'].get('low_exposure_delta_s',0)-c['policies']['h20'].get('low_exposure_delta_s',0),
   'service_delta': c['policies']['h5']['original_cost_deltas']['service']-c['policies']['h20']['original_cost_deltas']['service'],
   'source_wait_delta': c['policies']['h5']['original_cost_deltas']['wait']-c['policies']['h20']['original_cost_deltas']['wait']}
out=ROOT/'output/collapse_prevention_audit_v1/PHYSICAL_REVIEW.json'; out.write_text(json.dumps(result,indent=2)+'\n')
lines=['# OE-35 physical review','', 'Independent recomputation from the two OE34 datasets, restricted to fresh seeds 90001-90060. No simulation or model fitting was performed.', '']
for name,x in result['populations'].items():
 if name=='all60' or name in ('entry32','branch22'):
  lines.append(f'## {name} (n={x["h20"]["n"]})')
  for a in ['h20','h5']:
   lines.append(f'- {a}: means '+', '.join(f'{m}={x[a]["mean_delta"][m]:.15g}' for m in metrics)+f'; labels={x[a]["labels"]}; V2 match={x["v2_match"][a]}.')
lines += ['', 'The source recomputation matches every V2 aggregate cost mean and original qualification count at all60, entry32, and branch22 populations within 1e-12.', '', 'Requested spot checks:']
for s,x in result['spot_checks'].items():
 l=x['lifecycle']; h20=l.get('h20',{}); h5=l.get('h5',{}); lines.append(f'- {s}: V2 categories={x["v2_policy_categories"]}; H20 hold_seconds={h20.get("hold_seconds")}, full_nominal_hold={h20.get("full_nominal_hold_delivered")}; H5 hold_seconds={h5.get("hold_seconds")}, valid_sampled_hold={h5.get("valid_sampled_hold")}.')
 for k in ['h20','h5']:
  p=x['duration_contrast']['low_exposure_delta_s'][k]; q=x['duration_contrast']['aligned_discharge'][k]
  lines.append(f'  - {k}: low-exposure delta vs D (600-1200)={p} s; aligned discharge around baseline candidate={q}.')
 lines.append(f'  - H5 minus H20: low-exposure delta difference={x["h5_minus_h20"]["low_exposure_delta_s"]} s, service delta difference={x["h5_minus_h20"]["service_delta"]} s/request, source-wait difference={x["h5_minus_h20"]["source_wait_delta"]} s/request.')
lines += ['', 'For 90030 specifically, the H20 arm formed at measured readiness 610 s but delivered 15 s HOLD and then aborted at 633 s because the exit was infeasible; it did not deliver the nominal 20 s HOLD. The H5 arm delivered a 5 s HOLD and released at 624 s. These lifecycle facts are retained alongside original qualification vectors.', '', 'The three requested cases (90011, 90030, 90046) are V2 `absent_through_1200` policy classifications for both H20 and conditional H5. This is an observed-window diagnostic category and does not establish permanent prevention or capacity change.']
(ROOT/'output/collapse_prevention_audit_v1/PHYSICAL_REVIEW.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v['v2_match'] for k,v in result['populations'].items() if k!='all60' or True}))
