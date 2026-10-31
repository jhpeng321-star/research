"""Default-population decision equality across the signed-cap source fix.

Old source is OUR committed WM implementation, not downloaded SUMO code.
Synthetic float32 observation snapshots, no traffic files or simulation.
"""
import hashlib
import json
import subprocess
import sys
import types

import torch

from .operate import RECEIPTS, ROOT
from wms.physics.config import PhysicsConfig
from wms.physics.lc2013 import lane_change

BASELINE = '0a1e7e9396bac5bc23ce1def030332ea6a79a8a3'


def main():
    out = RECEIPTS/'POPULATION_EQ_SIGNED_CAP.json'
    assert out.parent.is_dir() and not out.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw = subprocess.check_output(['git', 'show', BASELINE+':src/wms/physics/lc2013.py'], cwd=ROOT)
    previous = types.ModuleType('wms.physics._rblane_lc_original')
    previous.__package__ = 'wms.physics'
    previous.__file__ = str(ROOT/'src/wms/physics/lc2013.py')
    sys.modules[previous.__name__] = previous
    exec(compile(raw, previous.__file__, 'exec'), previous.__dict__)
    physics = PhysicsConfig(sumo_fidelity=True, open_road=True)
    torch.manual_seed(1001)
    before = after = None
    elements = 0
    for step in range(100):
        kin = torch.zeros(8, 12, 7, dtype=torch.float32)
        kin[..., 0] = torch.rand(8, 12) * 2500 - 2000
        lane = torch.randint(0, 3, (8, 12))
        kin[..., 1:4] = torch.nn.functional.one_hot(lane, 3)
        kin[..., 4] = torch.rand(8, 12) * 45
        kin[..., 5] = torch.rand(8, 12) * 8 - 4
        kin[..., 6] = 5
        mask = torch.rand(8, 12) > .1
        before = previous.lane_change(kin, mask, physics, memory=before)
        after = lane_change(kin, mask, physics, memory=after)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, equal_nan=True)
            elements += before[key].numel()
        for key in ('lc2013_left', 'lc2013_right', 'lc2013_keep_right', 'lc2013_lookahead'):
            assert torch.isfinite(after[key]).all()
    report = dict(status='PASS', baseline_commit=BASELINE, snapshot_dtype='float32',
        homogeneous_default_v0=physics.v0_mps, synthetic_vehicle_decisions=9600,
        compared_tensor_elements=elements, all_previous_outputs='BIT_EXACT',
        prior_lc_sha256=hashlib.sha256(raw).hexdigest(),
        current_lc_sha256=hashlib.sha256((ROOT/'src/wms/physics/lc2013.py').read_bytes()).hexdigest(),
        scope='homogeneous inherited defaults; no claim for heterogeneous inferred settings')
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
