"""Synthetic padded-slot arbitration benchmark; checks exact output equality."""
import argparse
import json
from pathlib import Path
import time

import torch

from wms.physics.lane_conflict import resolve_lane_conflicts
from wms.physics.lane_conflict_fast import resolve_lane_conflicts_fast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.out)
    if not output.parent.is_dir() or output.exists():
        raise ValueError('new output in existing parent required')
    s = torch.arange(640).float()[None].expand(5, -1).clone()*8-2500
    lane = (torch.arange(640)[None].expand(5, -1)%2).clone()
    proposed = lane.clone()
    proposed[:, [10, 60, 150, 500]] = 1-proposed[:, [10, 60, 150, 500]]
    present = torch.zeros_like(lane, dtype=torch.bool)
    present[:, :180] = True
    movable = torch.ones_like(present)
    movable[:, 60] = False
    values = [x.to(args.device) for x in (s, torch.full_like(s, 5.), lane, proposed, present, movable)]
    timings, outputs = {}, {}
    for name, function in (('dense', resolve_lane_conflicts), ('candidate_only', resolve_lane_conflicts_fast)):
        function(*values, 0.)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            outputs[name] = function(*values, 0.)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        timings[name] = (time.perf_counter()-start)/5
    if not torch.equal(outputs['dense'], outputs['candidate_only']):
        raise AssertionError('arbitration benchmark parity failed')
    result = dict(synthetic=True, device=args.device, batch=5, slots=640,
        present_per_row=180, movable_change_candidates_per_row=2,
        seconds_per_call=timings, speedup=timings['dense']/timings['candidate_only'],
        output_bit_exact=True, full_rollout_speedup_not_measured=True, training=False, sumo_runs=0)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
