"""Default-style source extension gate; executes only our own prior Git code."""
import argparse
import hashlib
import json
import subprocess
import sys
import types

import torch

from .operate import RECEIPTS, ROOT
from wms.physics.config import PhysicsConfig
from wms.physics.lc2013 import lane_change

BASELINE = 'aefc4bd9'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', default='OPTIONAL_STYLE_DEFAULT_EQUIVALENCE.json')
    args = parser.parse_args()
    assert args.receipt.endswith('.json') and '/' not in args.receipt and '\\' not in args.receipt
    out = RECEIPTS / args.receipt
    assert out.parent.is_dir() and not out.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw = subprocess.check_output(['git', 'show', BASELINE + ':src/wms/physics/lc2013.py'], cwd=ROOT)
    old = types.ModuleType('wms.physics._optional_style_baseline')
    old.__package__ = 'wms.physics'
    old.__file__ = str(ROOT / 'src/wms/physics/lc2013.py')
    sys.modules[old.__name__] = old
    exec(compile(raw, old.__file__, 'exec'), old.__dict__)
    torch.manual_seed(1702)
    physics = PhysicsConfig(sumo_fidelity=True)
    elements = decisions = changes = 0
    for heterogeneous in (False, True):
        before, after = {}, {}
        k = torch.zeros(8, 24, 7, dtype=torch.float32)
        k[..., 0] = torch.rand(8, 24) * 2300 - 2200
        k[..., 1:4] = torch.nn.functional.one_hot(torch.randint(0, 3, (8, 24)), 3)
        k[..., 4] = torch.rand(8, 24) * 35
        k[..., 6] = 5
        params = (dict(v0=20 + torch.rand(8, 24) * 15, accel=.5 + torch.rand(8, 24) * 2,
                       decel=.5 + torch.rand(8, 24) * 4, time_headway=.5 + torch.rand(8, 24) * 2,
                       min_gap=1 + torch.rand(8, 24) * 2.5) if heterogeneous else None)
        identity = torch.arange(192).reshape(8, 24)
        for step in range(120):
            k[..., 0] += k[..., 4] * physics.dt_s
            k[..., 5] = torch.rand(8, 24) * 6 - 3
            present = torch.rand(8, 24) > .05
            if step % 17 == 0:
                identity[:, 0] += 192  # recycled slots must reset equally
            before = old.lane_change(k, present, physics, memory=before, driver_params=params, identity=identity)
            after = lane_change(k, present, physics, memory=after, driver_params=params, identity=identity)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, equal_nan=True)
                elements += before[key].numel()
            changes += int(((before['next_lane'] != k[..., 1:4].argmax(-1)) & present).sum())
            decisions += present.numel()
    result = dict(status='PASS', baseline_commit=BASELINE, synthetic_vehicle_decisions=decisions,
                  compared_tensor_elements=elements, positive_proposal_steps=changes,
                  all_existing_outputs='BIT_EXACT',
                  scope='SUMO default LC style, population and heterogeneous physical settings; not a traffic score',
                  prior_lc_sha256=hashlib.sha256(raw).hexdigest(),
                  current_lc_sha256=hashlib.sha256((ROOT / 'src/wms/physics/lc2013.py').read_bytes()).hexdigest())
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
