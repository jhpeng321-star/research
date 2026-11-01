"""Verify/reassemble completed round-2 FIT receipts and recorded target matches.

No inference, population fitting, new tape reads, or evaluation-world selection.
"""
import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path

from .round2_validate import summarize
from .validate import sha

ROOT = Path(__file__).resolve().parents[2]
KINDS = ('native300_a_v1', 'native300_b_v1', 'arms60_a_v1', 'arms60_b_v1')


def load_rows(root):
    rows, manifests = [], {}
    for kind in KINDS:
        dispatch = json.loads((root/(kind+'_DISPATCH.json')).read_text())
        completed = json.loads((root/(kind+'_COMPLETE.json')).read_text())
        assert completed['complete'] and not completed['alive']
        assert completed['bundle_sha256'] == dispatch['bundle_sha256']
        summary_path = root/(kind+'_results')/'SUMMARY.json'
        summary = json.loads(summary_path.read_text())
        assert summary['status']=='COMPLETE' and summary['samples']==8
        selected = [json.loads(p.read_text()) for p in sorted((root/(kind+'_results')).glob('w*.json'))]
        selected.sort(key=lambda r:(dispatch['worlds'].index(r['seed']),dispatch['arms'].index(r['arm'])))
        assert {(r['seed'],r['arm']) for r in selected} == {(w,a) for w in dispatch['worlds'] for a in dispatch['arms']}
        assert summarize(selected,8) == summary['summary']
        for r in selected:
            assert 969400<=r['seed']<=969431
            assert r['conditioning']['saved_lc_driver_id_lookup'] is False
            assert r['conditioning']['conditioning_interval_s']==[240.,300.]
            assert r['conditioning']['lc_runtime_source_sha256'] == dispatch['source_sha256']['src/wms/physics/lc2013.py']
        manifests[kind] = dict(bundle_sha256=dispatch['bundle_sha256'], summary_sha256=sha(summary_path),
            process=completed['process'], source_sha256={name:dispatch['source_sha256'][name] for name in
            ('src/wms/physics/lc2013.py','src/wms/physics/lane_continuation.py','tools/rblane/round2_validate.py')})
        rows.extend(selected)
    assert len(rows)==80 and len({r['episode'] for r in rows})==80
    assert len({json.dumps(m['source_sha256'],sort_keys=True) for m in manifests.values()})==1
    return rows, manifests


def target_reference(path):
    result = {}
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            world = int(r['seed'])
            assert 969400<=world<=969431
            key=(world,r['arm'].lower(),int(r['vid']),float(r['t_s']),int(r['from_lane']),int(r['to_lane']))
            assert key not in result
            result[key]=float(r['s_m'])
    return result


def reconcile(rows, reference):
    absent, errors, n = [], [], 0
    for row in rows:
        window = row['horizons'][str(max(map(int,row['horizons'])))]
        for e in window['recorded_events']:
            vid, source, dest, at, pos = e
            key=(row['seed'],row['arm'],vid,at,source,dest)
            n+=1
            if key not in reference:
                absent.append(key)
            else:
                errors.append(abs(pos-reference[key]))
    unexpected=[key for key in absent if key[3]!=300.5]
    assert not unexpected, unexpected[:10]
    return dict(status='PASS_RECONCILED',scored_recorded_events=n,matched_nonboundary=len(errors),
        first_tape_frame_boundary_missing=len(absent),unexpected_missing=0,
        position_abs_max_m=max(errors),target_earliest_event_s=min(k[3] for k in reference),
        note='Target traj_base begins300.5 and cannot detect300.0→300.5; all other scored HDV events match. Target totals include CAV/connector-tolerant events excluded from conditional cohort.')


def motivation_csv(panels, out):
    fields=('panel','horizon_s','model','motivation','predicted','matched','precision','recall_contribution','positive_run_precision_2s','positive_run_recall_contribution_2s','matched_run_timing_mae_s')
    with (out/'motivation.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for panel,report in panels.items():
            for h,models in report.items():
                for model,scores in models.items():
                    for name,reason in scores['by_motivation'].items():
                        e,r=reason['exact'],reason['run_window_2s']
                        writer.writerow(dict(panel=panel,horizon_s=h,model=model,motivation=name,
                            **{k:e[k] for k in ('predicted','matched','precision')},recall_contribution=e['recall'],
                            positive_run_precision_2s=r['precision'],positive_run_recall_contribution_2s=r['recall'],
                            matched_run_timing_mae_s=r['timing_mae_s']))


def cells_csv(panels,out):
    fields=('panel','horizon_s','model','segment','source_lane','recorded','predicted_steps','predicted_runs',
        'matched_steps','exposure_veh_s','recorded_rate_per_veh_h','proposed_step_rate_per_veh_h','proposed_run_rate_per_veh_h')
    with (out/'lane_segment.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for panel,report in panels.items():
            for h,models in report.items():
                for model,scores in models.items():
                    for cell,values in scores['lane_segment'].items():
                        segment,lane=cell.split('/')
                        row={k:values.get(k,0) for k in fields if k in values}
                        for name in ('recorded','predicted_steps','predicted_runs','matched_steps'):
                            row.setdefault(name,0)
                        exposure=values.get('exposure_veh_s',0)
                        row.update(panel=panel,horizon_s=h,model=model,segment=segment,source_lane=lane,
                            proposed_run_rate_per_veh_h=3600*values.get('predicted_runs',0)/exposure if exposure else None)
                        writer.writerow(row)


def time_csv(rows,reference,out):
    # Bin by ARRIVAL time as target CSV does. Right endpoint events at360/600
    # are explicitly retained in an endpoint row, not smeared into the last bin.
    with (out/'timing_10s.csv').open('x',newline='') as f:
        fields=('seed','arm','horizon_s','bin_index','t_start_s','recorded_cohort','source_default_steps','source_default_runs','fitted_expected_steps','fitted_expected_runs','target_all_vehicles','endpoint')
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            for h,window in row['horizons'].items():
                end=300+int(h)
                def bins(events):
                    return Counter(int((e[3]-300)//10) for e in events)
                real=bins(window['recorded_events'])
                default=bins(window['source_default_events'])
                from .validate import compress_proposals
                default_runs=bins(compress_proposals(window['source_default_events'],.5))
                fitted=Counter();fitted_runs=Counter()
                for events in window['fitted_sample_events']:
                    fitted.update(bins(events));fitted_runs.update(bins(compress_proposals(events,.5)))
                targets=Counter(int((key[3]-300)//10) for key in reference
                    if key[0]==row['seed'] and key[1]==row['arm'] and 300<key[3]<=end)
                for index in range(int(h)//10+1):
                    writer.writerow(dict(seed=row['seed'],arm=row['arm'],horizon_s=h,bin_index=index,
                        t_start_s=300+10*index,recorded_cohort=real[index],source_default_steps=default[index],
                        source_default_runs=default_runs[index],fitted_expected_steps=fitted[index]/8,
                        fitted_expected_runs=fitted_runs[index]/8,target_all_vehicles=targets[index],
                        endpoint=index==int(h)//10))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipts',type=Path,default=ROOT/'docs/rb2lane')
    ap.add_argument('--targets',type=Path,default=Path('E:/agent-work/rb-shared/targets'))
    ap.add_argument('--sumo-source',type=Path,default=Path('E:/Temp/opencode/sumo-1.27.1-rb-lane'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():raise ValueError('new output required')
    rows,manifests=load_rows(args.receipts)
    native=[r for r in rows if r['arm']=='native']
    dev=copy.deepcopy([r for r in rows if r['seed']>=969420])
    assert len(native)==32 and len(dev)==60
    for r in dev:r['horizons']={'60':r['horizons']['60']}
    panels=dict(native_fit32=summarize(native,8),development12_fivearms=summarize(dev,8))
    # Every valid source lane has exactly one physical target in this corridor.
    # Thus mean hard-decision event Brier = E[count]+events-2E[TP]. Remove the
    # finite-K sample-mean variance bias: sum p(1-p)/(K-1). This is a proper
    # score estimator, not a calibrated probability transformation.
    for row in rows:
        for event in row['horizons'][str(max(map(int,row['horizons'])))]['recorded_events']:
            assert event[2] == {0:1,1:0,2:1}[event[1]]
    for panel in panels.values():
        for models in panel.values():
            for model,score in models.items():
                e,p=score['exact'],score['probability_scores']
                hard=e['predicted']+e['recorded']-2*e['matched']
                correction=(hard-p['event_brier_sum'])/7 if model=='fitted_samples' else 0.
                p.update(fair_K8_event_brier=(p['event_brier_sum']-correction)/p['n'],
                    never_change_event_brier=e['recorded']/p['n'],
                    finite_K_variance_correction=correction/p['n'])
    reference=target_reference(args.targets/'lc_events.csv')
    reconciliation=reconcile(rows,reference)
    smoke_path=args.receipts/'runtime_v1_SMOKE.json'
    smoke=json.loads(smoke_path.read_text())
    assert smoke['status']=='PASS' and len(smoke['results'])==4
    assert all(r['finite'] and r['conserved'] and r['final_lc_state_finite'] and r['future']['fitted_lc'] for r in smoke['results'])
    args.out.mkdir()
    report=dict(status='COMPLETE',panels=panels,manifests=manifests,target_reconciliation=reconciliation,
        sumo_reference_sha256={name:sha(args.sumo_source/name) for name in ('MSLCM_LC2013.cpp',
            'MSLCM_LC2013.h','MSLaneChanger.cpp','MSAbstractLaneChangeModel.cpp','MSCFModel.cpp','MSCFModel_IDM.cpp')},
        target_sha256=sha(args.targets/'lc_events.csv'),runtime_smoke_sha256=sha(smoke_path),
        exposure=dict(fit_population_worlds=list(range(969400,969432)),new_diagnosis_worlds=list(range(969400,969432)),
            development_panel_worlds=list(range(969420,969432)),reserved_selection=False,wmdata2_access=False,
            arms_exposed=['native','ctrl0','mappo','pi','circles'],population_refit=False,neural_training=False),
        cohort='HDV same identity/slot, normal-road adjacent snapshots; teacher-forced geometry, persistent per-driver profiles',
        confidence_intervals='not claimed: development diagnostic, not independent qualification',
        motivation_contract='predicted motivation; recall contribution to all recorded events, no observed motivation labels',
        source_caveat='frozen population fit on earlier kernel; own-start posterior reconditioned using reported current kernel',
        settings='same frozen pooled IDM for paired source-default/fitted LC; real-checkpoint smoke separately samples IDM+LC blocks')
    (args.out/'RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    motivation_csv(panels,args.out);cells_csv(panels,args.out);time_csv(rows,reference,args.out)
    print(json.dumps(dict(target_reconciliation=reconciliation,
        scores={panel:{h:{model:score['exact'] for model,score in models.items()} for h,models in results.items()}
            for panel,results in panels.items()}),indent=2))


if __name__=='__main__':main()
