"""Aggregate disjoint FIT receipts, retaining scope and version caveats."""
import argparse
import json
from pathlib import Path

from tools.rbeval.validate_merge import summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--version', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    receipts = [json.loads(Path(path).read_text()) for path in args.inputs]
    rows = [r for receipt in receipts for r in receipt['rows']]
    seeds = [r['seed'] for r in rows]
    if len(seeds) != len(set(seeds)) or not all(969400 <= seed <= 969431 for seed in seeds):
        raise ValueError('duplicate or prohibited seed; receipts must be disjoint')
    time_errors, pos_errors = [], []
    conditional = {k: 0 for k in ('tp', 'fp', 'fn', 'tn', 'observed_births', 'admitted_at_observed_birth')}
    for row in rows:
        c = row['cohort']
        real = {r['slot']: r for r in c['real_events']}
        pred = {r['slot']: r for r in c['predicted_events']}
        for identity in real.keys() & pred.keys():
            time_errors.append(pred[identity]['time_s'] - real[identity]['time_s'])
            pos_errors.append(pred[identity]['position_m'] - real[identity]['position_m'])
        for key in conditional:
            conditional[key] += row['conditional_recorded_state'][key]
    result = dict(version=args.version, seeds=sorted(seeds), n_worlds=len(rows),
        cohort=dict(real_merges=sum(r['cohort']['real_merges'] for r in rows),
            predicted_merges=sum(r['cohort']['predicted_merges'] for r in rows),
            matched_merges=len(time_errors), time_error_s=summary(time_errors),
            abs_time_error_s=summary(list(map(abs, time_errors))),
            position_error_m=summary(pos_errors), abs_position_error_m=summary(list(map(abs, pos_errors))),
            real_ramp_lane_speed_loss_veh_s=summary([r['cohort']['ramp_speed_loss_veh_s']['real'] for r in rows]),
            predicted_ramp_lane_speed_loss_veh_s=summary([r['cohort']['ramp_speed_loss_veh_s']['predicted'] for r in rows])),
        conditional_recorded_state=conditional,
        full_fleet_qualification=False,
        limitations=receipts[0]['limitations'], inputs=args.inputs)
    text = json.dumps(result, indent=2, allow_nan=False)
    Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
