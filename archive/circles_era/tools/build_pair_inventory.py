import hashlib, json
from pathlib import Path

ROOT = Path('D:/research-dense')
OUT = ROOT / 'output/collapse_prevention_audit_v1'
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = [ROOT/'output/observation_relational_v1/DATASET_development.json', ROOT/'output/observation_relational_v1/DATASET_holdout.json']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

rows=[]
for p in DATASETS:
    d=json.loads(p.read_text())
    rows.extend(r for r in d['rows'] if 90001 <= r['seed'] <= 90060)
rows.sort(key=lambda r:r['seed'])

def arm(a, rec):
    run=ROOT / rec['path'].replace('\\','/')
    files={n:{'exists':(run/n).is_file(), 'bytes':(run/n).stat().st_size if (run/n).is_file() else None,
              'sha256':sha(run/n) if (run/n).is_file() else None} for n in ['traffic.jsonl','intervals.jsonl','telemetry.jsonl','summary.json','manifest.json']}
    lifecycle=rec.get('lifecycle')
    timing=None
    if lifecycle:
        phases=lifecycle.get('phases',[])
        timing={'assembly_start_s': next((x['time_s'] for x in phases if x['phase']=='assembly'),None),
                'measured_ready_at_s': lifecycle.get('measured_ready_at_s'),
                'hold_start_s': next((x['time_s'] for x in phases if x['phase']=='hold'),None),
                'taper_start_s': next((x['time_s'] for x in phases if x['phase']=='taper'),None),
                'release_start_s': next((x['time_s'] for x in phases if x['phase']=='release'),None),
                'terminal_time_s': lifecycle.get('terminal_time_s'),
                'assembly_seconds': lifecycle.get('assembly_seconds'),
                'hold_seconds': lifecycle.get('hold_seconds'),
                'measured_exit_seconds': lifecycle.get('measured_exit_seconds'),
                'registered_maneuver_delivered': lifecycle.get('registered_maneuver_delivered'),
                'phase_entry_matches_measured_readiness': lifecycle.get('phase_entry_matches_measured_readiness'),
                'handback_verified': (lifecycle.get('handback') or {}).get('verified'),
                'wall_reengaged_after_terminal': lifecycle.get('wall_reengaged_after_terminal'),
                'command_contract_passed': (rec.get('command_contract') or {}).get('passed')}
    return {'arm':a,'run_dir':rec['path'],'manifest_sha256':rec.get('manifest_sha256'),'summary_sha256':rec.get('summary_sha256'),
            'costs':rec.get('costs'),'physical_vector':{'collisions':rec.get('collisions'),'teleports':rec.get('teleports'),'unfinished':rec.get('unfinished'),'population':rec.get('population'),'complete_requested':rec.get('complete_requested')},
            'timing':timing,'telemetry_files':files}

inventory={'schema':'collapse_prevention_pair_inventory/1','purpose':'OE-35 offline paired inventory; no new simulation or model fitting.',
 'source_datasets':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for p in DATASETS],
 'population_definition':{'seed_range':'90001..90060 inclusive','fresh_only':True,'excluded_reused_rows':'rows outside range / reused_groups; reused OE32 exposure excluded from fresh summaries'},
 'counts':{},'cases':[],
 'resolution_limitations':{
   'throughput':'No dedicated throughput time series or scalar is stored in the dataset rows. The original summaries expose inserted/arrived/requested_due_at_horizon and service seconds, so completion and service can be audited; throughput must be derived and is not silently asserted here.',
   'trajectory':'traffic.jsonl is regional aggregate traffic telemetry and intervals.jsonl is one-second physical interval aggregation. telemetry.jsonl is timestamped CAV/RSU and ownership event telemetry, not a complete per-vehicle trajectory stream for every actor; the OE-35 compact raw trajectory extractor is the authoritative source for collapse onset/prevention timing.',
   'timing':'Wall readiness, HOLD, taper, release and handback are copied only when lifecycle fields exist. Damping and refusal rows have no wall lifecycle, represented as null/no lifecycle rather than inferred timing.'}}
from collections import Counter
inventory['counts']={'fresh_rows':len(rows),'development_rows':sum(r['split']=='development' for r in rows),'holdout_rows':sum(r['split']=='holdout' for r in rows),
 'damping_reference_cases':len(rows),'entry_feasible_cases':sum(r['entry_feasible'] for r in rows),'entry_refusal_cases':sum(not r['entry_feasible'] for r in rows),
 'branch_available_cases':sum(r['branch_available'] for r in rows),'h20_wall_arm_cases':sum('h20' in r['physical_records'] for r in rows),
 'h5_wall_delivered_cases':sum(r['branch_available'] and 'h5' in r['physical_records'] for r in rows),
 'h5_fallback_to_h20_cases':sum(r.get('fallback_reason')=='hold not active at HOLD+5' for r in rows),
 'entry_refusal_fallback_cases':sum(r.get('fallback_reason')=='entry_refusal' for r in rows)}
for r in rows:
    inventory['cases'].append({'seed':r['seed'],'split':r['split'],'entry_feasible':r['entry_feasible'],'branch_available':r['branch_available'],'branch_time_s':r['branch_time_s'],
      'fallback_reason':r.get('fallback_reason'),'interpretation':r.get('interpretation'),'policy_source':r['policy_source'],
      'entry_decision_time_s':(r.get('entry_feature_evidence') or {}).get('decision_time_s'),
      'labels':r['labels'],'deltas':r['deltas'],'arm_sources':[arm(a,rec) for a,rec in r['physical_records'].items()]})
(OUT/'PAIR_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')

lines=['# OE-35 paired collapse prevention inventory','',f"Source datasets: `{inventory['source_datasets'][0]['path']}` and `{inventory['source_datasets'][1]['path']}`.",f"Dataset SHA-256: `{inventory['source_datasets'][0]['sha256']}`, `{inventory['source_datasets'][1]['sha256']}`.",'','Population is fresh seeds 90001-90060 inclusive; reused OE32 rows are excluded from all fresh summaries. No simulation, model fitting, or threshold selection was performed.','',f"Counts: {len(rows)} damping references; {inventory['counts']['entry_feasible_cases']} entry-feasible cases; {inventory['counts']['branch_available_cases']} H5-deliverable cases; {inventory['counts']['h20_wall_arm_cases']} H20 wall arms; {inventory['counts']['h5_fallback_to_h20_cases']} H5 fallback-to-H20 cases; {inventory['counts']['entry_refusal_cases']} entry refusals.",'','Entry refusals retain damping for both H20 and H5 (`fallback_reason=entry_refusal`), so they have no executed wall arm. Entry-feasible cases without a branch retain the H20 arm for H5 (`fallback_reason=hold not active at HOLD+5`); this is an explicit policy fallback, not an H5 release.','', 'Each case below lists the original run directories, manifest and summary hashes, original physical vector and labels, plus availability and hashes for traffic.jsonl, intervals.jsonl, telemetry.jsonl, summary.json and manifest.json. Wall timing is copied from the recorded lifecycle: assembly at 600 s, measured readiness, HOLD, taper, release/terminal and handback status. Missing timing is reported as null rather than inferred.','']
lines += ['Resolution limits: the dataset has no dedicated throughput time series or scalar; completion, arrivals and service seconds are available in original summaries, but throughput would need derivation. traffic.jsonl is regional aggregate telemetry and intervals.jsonl is one-second aggregation. telemetry.jsonl contains timestamped CAV/RSU and ownership events rather than complete per-vehicle trajectories for every actor. Collapse onset/prevention timing should use the OE-35 compact raw trajectory extractor. Damping/refusal rows have no wall lifecycle timing and are reported without inferred readiness or release.','']
for c in inventory['cases']:
    arms=', '.join(a['arm'] for a in c['arm_sources'])
    lines.append(f"- **{c['seed']} ({c['split']})**: D={c['entry_feasible']}; branch/H5 delivered={c['branch_available']}; branch time={c['branch_time_s']}; fallback={c['fallback_reason']}; arms={arms}.")
    for a in c['arm_sources']:
        v=a['physical_vector']; t=a['timing'];
        timing='no wall lifecycle timing' if t is None else f"ready={t['measured_ready_at_s']} s, hold={t['hold_start_s']}–{t['taper_start_s']} s, release={t['release_start_s']} s, terminal={t['terminal_time_s']} s, delivered={t['registered_maneuver_delivered']}, handback={t['handback_verified']}"
        avail=', '.join(f"{k}:{'yes' if x['exists'] else 'NO'}" for k,x in a['telemetry_files'].items())
        lines.append(f"  - {a['arm']}: run `{a['run_dir']}`; vector collisions={v['collisions']}, teleports={v['teleports']}, unfinished={v['unfinished']}, population={v['population']}, complete={v['complete_requested']}; {timing}; files {avail}.")
(OUT/'PAIR_INVENTORY.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(inventory['counts'],indent=2))
