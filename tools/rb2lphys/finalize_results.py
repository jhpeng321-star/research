"""Make the requested before/after table from cached, fully verified results."""
import argparse
import csv
import json
from pathlib import Path

from tools.rbeval.core import digest, load, save_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    result = load(args.comparison)
    assert len(result['audit']) == 48
    expected = {(mode,w,k) for mode in ('off','learned_v1') for w in range(969420,969432) for k in range(2)}
    assert {(r['params'],r['world'],r['future']) for r in result['audit']} == expected
    assert all(digest(r['path']) == r['sha256'] for r in result['audit'])
    assert result['final_qualification'] is False
    summary = {(r['params'],r['horizon_s']):r for r in result['summary']}
    assert all(r['worlds']==12 and r['K']==2 and r['confidence_interval'] is None for r in summary.values())
    rows = []
    metrics = dict(delay_crps=('delay_CRPS_veh_s',1),delay_fair_crps=('delay_fair_CRPS_veh_s',1),
        delay_bias=('delay_mean_bias_veh_s',1),upstream_rate=('upstream_rate_pct',100),
        upstream_cell_rate_distance_pp=('upstream_cell_rate_distance_pp',1),onset_ibs=('upstream_onset_IBS',1),
        speed_crps_upstream=('upstream_speed_CRPS_mps',1),speed_crps_merge=('merge_speed_CRPS_mps',1),
        speed_crps_ramp=('ramp_speed_CRPS_mps',1),lc_count_crps=('lane_count_CRPS',1),
        lc_10s_crps=('lane_10s_count_CRPS',1),lc_mean=('mean_lane_changes',1),
        range_coverage=('K2_delay_range_coverage_pct',100),
        lp_ramp_lane_road=('mean_ramp_lane_road_loss_veh_s',1),lp_ramp_source=('mean_ramp_source_wait_veh_s',1))
    for h in (120,300,600):
        a,b = summary['off',h],summary['learned_v1',h]
        for key,(label,scale) in metrics.items():
            real = 100*a['upstream_real'] if key=='upstream_rate' else a['lc_real'] if key=='lc_mean' else None
            rows.append(dict(horizon_s=h,metric=label,before=a[key]*scale,after=b[key]*scale,
                after_minus_before=(b[key]-a[key])*scale,real_reference=real,worlds=12,arms=5,K=2,CI95=None))
    assert args.out.is_dir()
    with (args.out/'before_after.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt = dict(complete=True,tasks=48,arm_futures=240,worlds=12,arms=5,K=2,
        comparison_sha256=digest(args.comparison),narrow_delay_rate_point_screen=result['development_screen_pass'],
        recommended_as_congestion_or_credibility_repair=False,final_qualification=False,
        rationale='Tiny delay gains, unchanged breakdown, mixed speed/onset/coverage; no valid singleton-cell CIs.',
        fit_worlds=list(range(969400,969432)),fit_interval_s=[240,300],
        diagnostic_worlds=list(range(969400,969432)),diagnostic_interval_s=[240,600],
        selection_worlds=list(range(969420,969432)),autonomous_interval_s=[300,900],
        final_reserved_worlds_used_for_fitting_or_selection=[],WMDATA2_used_for_fitting_or_selection=[],
        scope='FULL-v4 physical-source development ablation; integrated FULL-v5-LP remains INT-owned',
        GPU_jobs='all own jobs exited; gpuR13:14 audit idle1MiB/0%; no peer PID action')
    save_new(args.out/'RESULTS_RECEIPT.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
