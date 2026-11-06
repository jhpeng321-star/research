"""Print compact, reproducible tables from the copied empirical JSON evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'docs/r13e_offline'
report = json.loads((EVIDENCE/'report.json').read_text(encoding='utf-8'))
arms = list(report['arms'])
summary = report['rollouts']['state_cluster_run_mean']
metrics = [
    ('overlap_pair_steps','Overlap pair-steps/run',1),
    ('ttc_lt_1_5_rows','TTC<1.5 rows/run',1),
    ('ttc_lt_1_5_share','TTC<1.5 (%)',100),
    ('a_lt_minus3_rows','Hard-brake rows/run',1),
    ('a_lt_minus3_share','Hard-brake (%)',100),
    ('hdv_inserter_post_hard_brake_rows','Inserter post-HB rows/run',1),
    ('hdv_inserter_post_hard_brake_rows_share','Inserter post-HB rows (%)',100),
    ('hdv_inserter_post_hard_brake_events_share','Inserter post-HB events (%)',100),
    ('hdv_follower_post_hard_brake_rows','Follower post-HB rows/run',1),
    ('hdv_follower_post_hard_brake_rows_share','Follower post-HB rows (%)',100),
    ('hdv_follower_post_hard_brake_events_share','Follower post-HB events (%)',100),
    ('stuck_vehicle_steps','Stuck vehicle-steps/run',1),
    ('stuck_final_vehicles','Final stuck vehicles/run',1),
    ('stuck_max_wait_s','Max low-speed ramp wait (s)',1),
    ('merges_per_run','Ramp merges/run',1),
    ('delay_upstream_veh_s','Upstream delay proxy (veh-s)',1),
    ('delay_merge_veh_s','Merge delay proxy (veh-s)',1),
    ('delay_downstream_veh_s','Downstream delay proxy (veh-s)',1),
    ('delay_total_veh_s','Total delay proxy (veh-s)',1),
    ('merge_zone_mean_hdv_count','Merge-zone occupancy (HDVs)',1),
    ('merge_zone_mean_speed_mps','Merge-zone speed (m/s)',1),
    ('blowup','Dynamics-failure runs (%)',100),
]


def ci(value, scale=1):
    if value['mean'] is None:
        return 'n/a'
    point = f"{scale*value['mean']:.3f}"
    return point + (f" [{scale*value['lo']:.3f}, {scale*value['hi']:.3f}]" if value['ci_available'] else ' [CI n/a]')


print('ARMS')
print('| Metric | off | emergency | emergency+gs |\n|---|---|---|---|')
for key,label,scale in metrics:
    print('| '+label+' | '+' | '.join(ci(summary[arm][key],scale) for arm in arms)+' |')
contrasts = ['emergency_minus_off','emergency+gs_minus_off','emergency+gs_minus_emergency']
print('CONTRASTS')
print('| Metric | emergency - off | emergency+gs - off | emergency+gs - emergency |\n|---|---|---|---|')
for key,label,scale in metrics:
    print('| '+label+' | '+' | '.join(ci(summary[arm][key],scale) for arm in contrasts)+' |')
print('TOTALS',json.dumps(report['rollouts']['totals']))
print('POST_EVENT_COVERAGE')
for arm in arms:
    rows = report['rollouts']['rows']
    totals = {key:sum(row[arm][key] for row in rows) for key in (
        'hdv_insertion_events','post_insertion_truncated_events','hdv_inserter_post_events',
        'hdv_inserter_post_hard_brake_events','hdv_follower_post_events','hdv_follower_post_hard_brake_events')}
    totals['shares_n_and_clusters'] = {key:(summary[arm][key]['n'],summary[arm][key]['clusters'])
        for key in ('hdv_inserter_post_hard_brake_events_share','hdv_follower_post_hard_brake_events_share')}
    print(arm,json.dumps(totals))
gs = json.loads((EVIDENCE/'gs_equivalence.json').read_text(encoding='utf-8'))
print('GS',json.dumps(gs))
base_path = EVIDENCE/'base_checks.json'
if base_path.exists():
    base = json.loads(base_path.read_text(encoding='utf-8'))
    print('BASE_VERDICTS',json.dumps(base['verdicts']))
    print('| BASE class | round 9 | off | emergency | emergency+gs |\n|---|---|---|---|---|')
    for name in base['classes']['off']:
        values=[]
        for arm in ['r9',*arms]:
            c=base['classes'][arm][name]
            values.append(f"{c['bias']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]; MAE {c['mae']:.3f}; n={c['n']}; clusters={c.get('clusters')}")
        print('| '+name+' | '+' | '.join(values)+' |')
