"""Pair completed native FIT free pilots with independent lane-rate targets.

Restore only the known first-tape-frame target boundary from recorded FIT views.
This postprocessing never supplies observations to a forecast or refits settings.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from tools.rbint.remote import ssh
from .validate import sha

ROOT=Path(__file__).resolve().parents[2]


def exclude_internal_initialization(record,audit):
    """Apply the target's valid-lane definition, leaving raw forecasts intact."""
    assert (record['world'],record['sample'],record['variant']) == (audit['world'],audit['sample'],audit['variant'])
    assert record['receipt']['fitted_artifacts']==audit['receipt']['fitted_artifacts']
    assert record['receipt']['demand_rate_draws']==audit['receipt']['demand_rate_draws']
    removed=0
    for cell,count in audit['excluded_LC_counts'].items():
        assert record['cells'][cell]['count']>=count
        record['cells'][cell]['count']-=count;removed+=count
    return removed


def boundary_targets(receipts,out):
    """All-vehicle native300.0→300.5, independent of either forecast's choices."""
    rows=[]
    for path in sorted((receipts/'native300_b_v1_results').glob('w*.json')):
        row=json.loads(path.read_text())
        if 969420<=row['seed']<=969431:
            rows.append({key:row[key] for key in ('seed','episode','input_sha256')})
    assert len(rows)==12
    result=json.loads(ssh('cpuQ',f'''import json,pathlib,numpy as np,hashlib
inputs={rows!r};root=pathlib.Path('/home/ubuntu/wmdata20261031/views/episodes');result=[]
for spec in inputs:
 assert 969420<=spec['seed']<=969431 and spec['episode'].endswith('__native')
 p=root/spec['episode']
 for name in ('time_s.npy','veh_kin.npy','veh_mask.npy','vid.npy'):
  assert hashlib.sha256((p/name).read_bytes()).hexdigest()==spec['input_sha256'][name]
 t,k,m,ids=[np.load(p/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in ('time_s','veh_kin','veh_mask','vid')]
 i=int(np.searchsorted(t,300.));assert t[i]==300. and t[i+1]==300.5
 source=k[i,:,1:4].argmax(-1);dest=k[i+1,:,1:4].argmax(-1)
 normal=(k[i,:,1:4].sum(-1)==1)&(k[i+1,:,1:4].sum(-1)==1)
 good=m[i].astype(bool)&m[i+1].astype(bool)&(ids[i]==ids[i+1])&normal&(source!=dest)
 events=[]
 for j in np.flatnonzero(good):
  events.append(dict(vid=int(ids[i,j]),source=int(source[j]),destination=int(dest[j]),s_m=float(k[i+1,j,0])))
 result.append(dict(seed=spec['seed'],episode=spec['episode'],events=events,input_sha256=spec['input_sha256']))
print(json.dumps(dict(status='PASS',simulation_time_s=[300.,300.5],cohort='all vehicles, same ID, normal adjacent frames',rows=result)))
''',timeout=90))
    with out.open('x') as f:json.dump(result,f,indent=2)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kinds',nargs='+',default=['free_canary_v1','free_a_v1','free_b_v1','free_c_v1','free_d_v1'])
    ap.add_argument('--receipts',type=Path,default=ROOT/'docs/rb2lane')
    ap.add_argument('--targets',type=Path,default=Path('E:/agent-work/rb-shared/targets'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prefix-kind',default='prefix_v1')
    args=ap.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():raise ValueError('new publication required')
    records,identities=[],[]
    for kind in args.kinds:
        complete=json.loads((args.receipts/(kind+'_COMPLETE.json')).read_text())
        assert complete['complete'] and not complete['alive']
        root=args.receipts/(kind+'_results')
        identity=json.loads((root/'IDENTITY.json').read_text());identities.append(identity)
        summary=json.loads((root/'COMPLETE.json').read_text())
        assert summary['status']=='PASS' and summary['samples']==2 and summary['horizon']==300
        selected=[json.loads(p.read_text()) for p in root.glob('w*.json')]
        assert len(selected)==summary['records']
        records.extend(selected)
    assert len(records)==48 and len({(r['world'],r['sample'],r['variant']) for r in records})==48
    assert {r['world'] for r in records}==set(range(969420,969432))
    assert all(r['audit']['finite'] and r['audit']['conserved'] and not any(r['no_slot']) for r in records)
    assert len({i['source_commit'] for i in identities})==1
    initial_audit=json.loads((args.receipts/'INITIAL_LANE_AUDIT.json').read_text())
    affected={r['world'] for r in initial_audit['rows'] if r['invalid_lanes']}
    prefix_root=args.receipts/(args.prefix_kind+'_results')
    prefix_complete=json.loads((prefix_root/'COMPLETE.json').read_text())
    assert prefix_complete['status']=='PASS' and prefix_complete['n_steps']==1
    assert set(prefix_complete['worlds'])==affected
    prefix={(r['world'],r['sample'],r['variant']):r for r in
        (json.loads(p.read_text()) for p in prefix_root.glob('w*.json'))}
    removed=Counter()
    for record in records:
        if record['world'] in affected:
            key=record['world'],record['sample'],record['variant']
            removed[record['variant']]+=exclude_internal_initialization(record,prefix[key])/2
    paired={(r['world'],r['sample'],r['variant']):r for r in records}
    for world in range(969420,969432):
        for sample in range(2):
            a,b=(paired[world,sample,variant] for variant in ('source_default','fitted_lc'))
            assert a['audit']['due']==b['audit']['due']
            assert a['receipt']['demand_rate_draws']==b['receipt']['demand_rate_draws']
    args.out.mkdir()
    boundary=boundary_targets(args.receipts,args.out/'BOUNDARY_TARGETS.json')
    reference={}
    with (args.targets/'lc_rates_10s.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            world=int(row['seed']);assert 969400<=world<=969431
            if row['arm']=='native' and 969420<=world<=969431 and int(row['bin_index'])<30:
                key=(world,row['segment'],row['kind'],int(row['bin_index']))
                reference[key]=dict(count=int(row['lc_count']),veh_time_s=float(row['veh_time_s']))
    assert len(reference)==12*3*2*30
    restored=0
    for row in boundary['rows']:
        for event in row['events']:
            assert (event['source'],event['destination']) in ((0,1),(1,0),(2,1))
            pos=event['s_m'];region='upstream' if pos<0 else 'downstream' if pos>75.44 else 'merge'
            kind='ramp_merge' if event['source']==2 and event['destination']==1 else 'mainline'
            reference[row['seed'],region,kind,0]['count']+=1;restored+=1
    world_counts=defaultdict(dict)
    by_cell=defaultdict(lambda:defaultdict(Counter))
    with (args.out/'lane_rate_10s.csv').open('x',newline='') as f:
        fields=('world','variant','segment','kind','bin_index','recorded','predicted_mean',
            'recorded_veh_time_s','predicted_mean_veh_time_s','recorded_rate_per_veh_h','predicted_rate_per_veh_h')
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for world in range(969420,969432):
            real=sum(v['count'] for k,v in reference.items() if k[0]==world)
            world_counts[world]['recorded']=real
            for variant in ('source_default','fitted_lc'):
                models=[paired[world,sample,variant] for sample in range(2)]
                predicted=sum(c['count'] for r in models for c in r['cells'].values())/2
                world_counts[world][variant]=predicted
                for key,real_cell in reference.items():
                    if key[0]!=world:continue
                    _,region,kind,index=key
                    cells=[m['cells'][f'{region}/{kind}/{index}'] for m in models]
                    count=sum(c['count'] for c in cells)/2
                    exposure=sum(c['veh_time_s'] for c in cells)/2
                    writer.writerow(dict(world=world,variant=variant,segment=region,kind=kind,bin_index=index,
                        recorded=real_cell['count'],predicted_mean=count,recorded_veh_time_s=real_cell['veh_time_s'],
                        predicted_mean_veh_time_s=exposure,recorded_rate_per_veh_h=3600*real_cell['count']/real_cell['veh_time_s'] if real_cell['veh_time_s'] else None,
                        predicted_rate_per_veh_h=3600*count/exposure if exposure else None))
                    by_cell[variant][f'{region}/{kind}'].update(recorded=real_cell['count'],predicted=count,
                        recorded_veh_time_s=real_cell['veh_time_s'],predicted_veh_time_s=exposure)
    variants={}
    for variant in ('source_default','fitted_lc'):
        errors=np.array([row[variant]-row['recorded'] for row in world_counts.values()])
        crps,fair,coverage=[],[],[]
        for world,row in world_counts.items():
            draws=np.array([sum(c['count'] for c in paired[world,sample,variant]['cells'].values()) for sample in range(2)])
            absolute=float(np.abs(draws-row['recorded']).mean())
            pair=float(abs(draws[0]-draws[1]))
            crps.append(absolute-pair/4);fair.append(absolute-pair/2)
            coverage.append(bool(draws.min()<=row['recorded']<=draws.max()))
        variants[variant]=dict(mean_world_count_bias=float(errors.mean()),mean_world_count_mae=float(abs(errors).mean()),
            empirical_K2_count_crps=float(np.mean(crps)),fair_K2_count_crps=float(np.mean(fair)),
            K2_count_range_coverage=float(np.mean(coverage)),
            total_expected_changes=sum(row[variant] for row in world_counts.values()),
            lane_segment={name:dict(c,recorded_rate_per_veh_h=3600*c['recorded']/c['recorded_veh_time_s'] if c['recorded_veh_time_s'] else None,
                predicted_rate_per_veh_h=3600*c['predicted']/c['predicted_veh_time_s'] if c['predicted_veh_time_s'] else None)
                for name,c in by_cell[variant].items()})
    report=dict(status='COMPLETE',worlds=list(range(969420,969432)),samples=2,horizon=300,
        source_commit=identities[0]['source_commit'],variants=variants,world_counts=dict(world_counts),
        restored_first_frame_events=restored,target_sha256=sha(args.targets/'lc_rates_10s.csv'),
        excluded_initial_internal_normalizations=dict(removed),
        prefix_audit_complete_sha256=sha(prefix_root/'COMPLETE.json'),
        boundary_sha256=sha(args.out/'BOUNDARY_TARGETS.json'),all_finite_conserved=True,no_slot_events=0,
        common_idm_arrival_draws=True,learned_corrections=False,wmdata2_access=False,reserved_selection=False,
        confidence_intervals='not claimed: developmental K2 diagnostic, sparse cells',
        comparison='both variants own-causal-history warm; LC latent block only factor; native CAVs use same physics law',
        qualification='not the five-arm distributional gate; regional rates and timing must be inspected')
    with (args.out/'RESULTS.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
