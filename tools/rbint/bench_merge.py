"""Inference-only dense/query benchmark on synthetic padded traffic tensors."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from wms.physics.config import PhysicsConfig
from wms.physics.junction_merge import apply_merge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.out)
    if not output.parent.is_dir() or output.exists():
        raise ValueError('new output in an existing parent required')
    generator = torch.Generator().manual_seed(4015)
    slots = 640
    kin = torch.zeros(5, slots, 7)
    kin[..., 0] = torch.rand(5, slots, generator=generator)*3000-2500
    lanes = torch.randint(2, (5, slots), generator=generator)
    lanes[:, :32] = 2
    kin[:, :32, 0] = torch.arange(32)*20-500
    kin[..., 1:4] = torch.nn.functional.one_hot(lanes, 3)
    kin[..., 4] = torch.rand(5, slots, generator=generator)*30
    kin[..., 6] = 5.
    present = torch.zeros(5, slots, dtype=torch.bool)
    present[:, :180] = True
    kin, lanes, present = [x.to(args.device) for x in (kin, lanes, present)]
    dense_physics = PhysicsConfig(merge_model='sumo')
    fast_physics = replace(dense_physics, sumo_merge_query_fast=True)

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    times = {}
    outputs = {}
    with torch.no_grad():
        for name, physics in (('dense', dense_physics), ('ego_query', fast_physics)):
            apply_merge(kin, present, lanes, present, physics)
            sync()
            start = time.perf_counter()
            for _ in range(5):
                outputs[name] = apply_merge(kin, present, lanes, present, physics)
            sync()
            times[name] = (time.perf_counter()-start)/5
    if not torch.equal(outputs['dense'], outputs['ego_query']):
        raise AssertionError('benchmark parity failed')
    result = dict(synthetic=True, device=args.device, batch=5, slots=slots,
        present_per_row=180, ramp_per_row=32, seconds_per_call=times,
        speedup=times['dense']/times['ego_query'], complete_reservation_lanes_bit_exact=True,
        full_rollout_speedup_not_measured=True, training=False, sumo_runs=0)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
